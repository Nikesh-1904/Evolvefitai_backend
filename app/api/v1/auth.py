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

# --- FIX: Removed prefix="/register" to create the correct endpoint path ---
router.include_router(
    fastapi_users.get_register_router(UserRead, UserCreate),
    tags=["auth"]
)

# Provides the standard /users/me endpoints
router.include_router(
    fastapi_users.get_users_router(UserRead, UserUpdate),
    prefix="/users",
    tags=["users"]
)

# Your custom Google OAuth routes are preserved below
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
        
        if error:
            logger.warning(f"OAuth error received: {error}")
            error_url = f"{settings.FRONTEND_URL}/login?error={error}"
            return RedirectResponse(url=error_url, status_code=status.HTTP_302_FOUND)

        if not code:
            logger.warning("OAuth callback missing authorization code")
            error_url = f"{settings.FRONTEND_URL}/login?error=missing_authorization_code"
            return RedirectResponse(url=error_url, status_code=status.HTTP_302_FOUND)

        try:
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

            jwt_strategy = get_jwt_strategy()
            token = await jwt_strategy.write_token(user)
            
            logger.info(f"JWT token generated for user: {user.id}")

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

# --- FIX: Removed custom /me and /me patch endpoints ---
# The router.include_router for get_users_router above already provides these
# at the more standard /users/me path, which we will now use.