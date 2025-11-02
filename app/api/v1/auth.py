# app/api/v1/auth.py - FIXED VERSION
import logging
from fastapi import APIRouter, Depends, Request, HTTPException, status
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi_users import exceptions
from httpx_oauth.exceptions import GetIdEmailError
from typing import Optional

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
logger.info(f"[Auth Router] get_user_manager imported from {get_user_manager.__module__}")

router = APIRouter()

# =============================================================================
# FASTAPI-USERS STANDARD ROUTES
# =============================================================================

# 1. JWT Authentication - provides /jwt/login and /jwt/logout
router.include_router(
    fastapi_users.get_auth_router(auth_backend),
    prefix="/jwt",
    tags=["auth"]
)

# 2. Registration - provides /register endpoint
# This creates the endpoint at /api/v1/auth/register
router.include_router(
    fastapi_users.get_register_router(UserRead, UserCreate),
    prefix="",  # Empty prefix so it's directly under /auth
    tags=["auth"]
)

# 3. User Management - provides /users/me endpoints
router.include_router(
    fastapi_users.get_users_router(UserRead, UserUpdate),
    prefix="/users",
    tags=["users"]
)

# =============================================================================
# CUSTOM GOOGLE OAUTH ROUTES
# =============================================================================

if google_oauth_client:

    @router.get("/google/authorize", tags=["auth"])
    async def google_authorize(request: Request):
        """
        Generate and return the authorization URL as JSON.
        Frontend should redirect user to this URL.
        """
        if not google_oauth_client:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, 
                detail="Google OAuth is not configured"
            )

        # Build the callback URL
        redirect_url = str(request.url_for("google_callback"))
        logger.info(f"OAuth redirect URL: {redirect_url}")

        # Get authorization URL from Google
        authorization_url = await google_oauth_client.get_authorization_url(
            redirect_url,
            scope=["openid", "email", "profile"],
        )

        logger.info(f"Generated authorization URL: {authorization_url}")

        # Return as JSON for frontend to handle
        return JSONResponse(
            content={"authorization_url": authorization_url}
        )

    @router.get("/google/callback", tags=["auth"])
    async def google_callback(
        request: Request,
        code: Optional[str] = None,
        error: Optional[str] = None,
        user_manager = Depends(get_user_manager),
    ):
        """Handle Google OAuth callback and redirect to frontend with token"""
        logger.info(f"[Auth Router] google_callback using get_user_manager from {get_user_manager.__module__}")
        
        # Handle OAuth errors
        if error:
            logger.warning(f"OAuth error received: {error}")
            error_url = f"{settings.CLIENT_FRONTEND_URL}/login?error={error}"
            return RedirectResponse(url=error_url, status_code=status.HTTP_302_FOUND)

        if not code:
            logger.warning("OAuth callback missing authorization code")
            error_url = f"{settings.CLIENT_FRONTEND_URL}/login?error=missing_authorization_code"
            return RedirectResponse(url=error_url, status_code=status.HTTP_302_FOUND)

        try:
            # Exchange code for access token
            redirect_url = str(request.url_for("google_callback"))
            logger.info(f"Exchanging code for token with redirect URL: {redirect_url}")
            
            access_token = await google_oauth_client.get_access_token(code, redirect_url)
            logger.info("Successfully obtained access token")
            
            # Get user info from Google
            user_id, user_email = await google_oauth_client.get_id_email(access_token["access_token"])
            logger.info(f"OAuth user info retrieved: {user_email}")

            # Create or login user via fastapi-users OAuth
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
                logger.info(f"OAuth user login/creation successful: {user.id}")

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
                    error_url = f"{settings.CLIENT_FRONTEND_URL}/login?error=account_linking_failed"
                    return RedirectResponse(url=error_url, status_code=status.HTTP_302_FOUND)

            except Exception as oauth_error:
                logger.error(f"OAuth user creation failed: {oauth_error}", exc_info=True)
                error_url = f"{settings.CLIENT_FRONTEND_URL}/login?error=user_creation_failed"
                return RedirectResponse(url=error_url, status_code=status.HTTP_302_FOUND)

            # Generate JWT token for the user
            jwt_strategy = get_jwt_strategy()
            token = await jwt_strategy.write_token(user)
            
            logger.info(f"JWT token generated for user: {user.id}")

            # Redirect to frontend with token
            frontend_callback_url = f"{settings.CLIENT_FRONTEND_URL}/oauth-callback?access_token={token}&token_type=bearer"
            return RedirectResponse(url=frontend_callback_url, status_code=status.HTTP_302_FOUND)

        except GetIdEmailError:
            logger.error("Failed to get user profile from Google")
            error_url = f"{settings.CLIENT_FRONTEND_URL}/login?error=google_profile_access_failed"
            return RedirectResponse(url=error_url, status_code=status.HTTP_302_FOUND)
        
        except Exception as e:
            logger.error(f"Unexpected OAuth error: {e}", exc_info=True)
            error_url = f"{settings.CLIENT_FRONTEND_URL}/login?error=oauth_failed"
            return RedirectResponse(url=error_url, status_code=status.HTTP_302_FOUND)

# =============================================================================
# HEALTH CHECK ENDPOINT
# =============================================================================

@router.get("/health", tags=["auth"])
async def auth_health_check():
    """
    Check if authentication system is working
    """
    return {
        "status": "healthy",
        "google_oauth_enabled": google_oauth_client is not None,
        "endpoints": {
            "register": "/api/v1/auth/register",
            "login": "/api/v1/auth/jwt/login",
            "logout": "/api/v1/auth/jwt/logout",
            "me": "/api/v1/auth/users/me",
            "google_authorize": "/api/v1/auth/google/authorize" if google_oauth_client else None,
        }
    }