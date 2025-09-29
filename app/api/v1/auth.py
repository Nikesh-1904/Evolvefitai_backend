# app/api/v1/auth.py
import logging
from fastapi import APIRouter, Depends, Request, HTTPException, status
from fastapi.responses import RedirectResponse
from fastapi_users import exceptions
from httpx_oauth.exceptions import GetIdEmailError

from app.core.config import settings
from app.core.auth import (
    auth_backend,
    fastapi_users,
    google_oauth_client,
    get_user_manager,
    get_jwt_strategy,
)
from app.schemas import UserRead, UserCreate, UserUpdate

# Set up logging
logger = logging.getLogger(__name__)

router = APIRouter()

# JWT auth router with prefix
router.include_router(
    fastapi_users.get_auth_router(auth_backend),
    prefix="/jwt",
    tags=["auth"]
)

router.include_router(
    fastapi_users.get_register_router(UserRead, UserCreate),
    prefix="/register",
    tags=["auth"]
)

router.include_router(
    fastapi_users.get_users_router(UserRead, UserUpdate),
    prefix="/users",
    tags=["users"]
)

# Google OAuth routes
if google_oauth_client:

    @router.get("/google/authorize", tags=["auth"])
    async def google_authorize(request: Request):
        """Initiate Google OAuth authorization"""
        if not google_oauth_client:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, 
                detail="Google OAuth is not configured"
            )
        
        redirect_url = str(request.url_for("google_callback"))
        authorization_url = await google_oauth_client.get_authorization_url(
            redirect_url,
            scope=["openid", "email", "profile"],
        )
        return {"authorization_url": authorization_url}

    @router.get("/google/callback", tags=["auth"])
    async def google_callback(
        request: Request,
        code: str = None,
        error: str = None,
        user_manager = Depends(get_user_manager),
    ):
        """Handle Google OAuth callback and redirect to frontend with token"""
        
        # Handle OAuth errors
        if error:
            logger.warning(f"OAuth error received: {error}")
            error_url = f"{settings.FRONTEND_URL}/login?error={error}"
            return RedirectResponse(url=error_url, status_code=status.HTTP_302_FOUND)

        if not code:
            logger.warning("OAuth callback missing authorization code")
            error_url = f"{settings.FRONTEND_URL}/login?error=missing_authorization_code"
            return RedirectResponse(url=error_url, status_code=status.HTTP_302_FOUND)

        try:
            # Exchange code for access token
            redirect_url = str(request.url_for("google_callback"))
            access_token = await google_oauth_client.get_access_token(code, redirect_url)
            user_id, user_email = await google_oauth_client.get_id_email(access_token["access_token"])

            logger.info(f"OAuth user info retrieved: {user_email}")

            try:
                user = await user_manager.oauth_callback(
                    oauth_name="google",
                    access_token=access_token["access_token"],
                    account_id=str(user_id),
                    account_email=user_email,
                    expires_at=access_token.get("expires_at"),
                    refresh_token=access_token.get("refresh_token"),
                    request=request,
                    associate_by_email=True,
                    is_verified_by_default=True,
                )
                logger.info(f"OAuth user login successful: {user.id}")

            except exceptions.UserAlreadyExists:
                logger.info(f"User exists, attempting to link OAuth account: {user_email}")
                try:
                    existing_user = await user_manager.get_by_email(user_email)
                    if existing_user:
                        user = await user_manager.oauth_callback(
                            oauth_name="google",
                            access_token=access_token["access_token"],
                            account_id=str(user_id),
                            account_email=user_email,
                            expires_at=access_token.get("expires_at"),
                            refresh_token=access_token.get("refresh_token"),
                            request=request,
                            associate_by_email=True,
                            is_verified_by_default=True,
                        )
                        logger.info(f"OAuth account linked successfully: {user.id}")
                    else:
                        raise
                except Exception as link_error:
                    logger.error(f"OAuth account linking failed: {link_error}")
                    error_url = f"{settings.FRONTEND_URL}/login?error=account_linking_failed"
                    return RedirectResponse(url=error_url, status_code=status.HTTP_302_FOUND)

            except Exception as oauth_error:
                logger.error(f"OAuth user creation failed: {oauth_error}", exc_info=True)
                error_url = f"{settings.FRONTEND_URL}/login?error=user_creation_failed"
                return RedirectResponse(url=error_url, status_code=status.HTTP_302_FOUND)

            # Generate JWT token
            jwt_strategy = get_jwt_strategy()
            token = await jwt_strategy.write_token(user)
            
            logger.info(f"JWT token generated for user: {user.id}")

            # Redirect to frontend with token in hash fragment
            frontend_callback_url = f"{settings.FRONTEND_URL}/auth/callback#access_token={token}&token_type=bearer"
            return RedirectResponse(url=frontend_callback_url, status_code=status.HTTP_302_FOUND)

        except GetIdEmailError:
            logger.error("Failed to get user profile from Google")
            error_url = f"{settings.FRONTEND_URL}/login?error=google_profile_access_failed"
            return RedirectResponse(url=error_url, status_code=status.HTTP_302_FOUND)
        
        except Exception as e:
            logger.error(f"Unexpected OAuth error: {e}", exc_info=True)
            error_url = f"{settings.FRONTEND_URL}/login?error=oauth_failed"
            return RedirectResponse(url=error_url, status_code=status.HTTP_302_FOUND)


# Custom user profile endpoints
@router.get("/me", response_model=UserRead, tags=["users"])
async def get_current_user(user=Depends(fastapi_users.current_user(active=True))):
    """Get current user profile"""
    logger.info(f"User profile requested: {user.id}")
    return user


@router.patch("/me", response_model=UserRead, tags=["users"])
async def update_current_user(
    user_update: UserUpdate,
    user=Depends(fastapi_users.current_user(active=True)),
    user_manager=Depends(get_user_manager),
):
    """Update current user profile with improved error handling and validation"""
    
    try:
        logger.info(f"User update requested: {user.id}")
        logger.debug(f"Update payload: {user_update.dict(exclude_unset=True)}")
        
        # Validate and sanitize dietary_restrictions
        if hasattr(user_update, 'dietary_restrictions') and user_update.dietary_restrictions is not None:
            # Ensure dietary_restrictions is a list
            if not isinstance(user_update.dietary_restrictions, list):
                logger.warning(f"dietary_restrictions is not a list: {type(user_update.dietary_restrictions)}")
                user_update.dietary_restrictions = []
            
            # Remove any None values from the list
            user_update.dietary_restrictions = [
                item for item in user_update.dietary_restrictions 
                if item is not None and item != ""
            ]
            
            logger.debug(f"Sanitized dietary_restrictions: {user_update.dietary_restrictions}")

        # Check for username uniqueness if username is being updated
        if user_update.username and user_update.username != user.username:
            # Note: This check might need adjustment based on your user_manager implementation
            logger.debug(f"Checking username uniqueness: {user_update.username}")
            
        # Perform the update
        updated_user = await user_manager.update(user_update, user)
        logger.info(f"User update successful: {user.id}")
        
        return updated_user
        
    except ValueError as ve:
        logger.error(f"Validation error during user update: {ve}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid data provided: {str(ve)}"
        )
        
    except Exception as e:
        logger.error(f"User update failed for user {user.id}: {e}", exc_info=True)
        
        # Check for common database constraint violations
        error_message = str(e).lower()
        if "unique constraint" in error_message or "duplicate" in error_message:
            if "username" in error_message:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Username already exists. Please choose a different username."
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="A field you're trying to update conflicts with existing data."
                )
        
        # Generic server error
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while updating your profile. Please try again."
        )