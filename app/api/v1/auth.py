# app/api/v1/auth.py - UPDATED for fixed OAuth
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

router = APIRouter()

# =============================================================================
# FASTAPI-USERS STANDARD ROUTES
# =============================================================================

# 1. JWT Authentication - /jwt/login and /jwt/logout
router.include_router(
    fastapi_users.get_auth_router(auth_backend),
    prefix="/jwt",
    tags=["auth"]
)

# 2. Registration - /register
router.include_router(
    fastapi_users.get_register_router(UserRead, UserCreate),
    prefix="",
    tags=["auth"]
)

# 3. User Management - /users/me
router.include_router(
    fastapi_users.get_users_router(UserRead, UserUpdate),
    prefix="/users",
    tags=["users"]
)

# =============================================================================
# GOOGLE OAUTH ROUTES (Simplified)
# =============================================================================

if google_oauth_client:

    @router.get("/google/authorize", tags=["auth"])
    async def google_authorize(request: Request):
        """
        Start Google OAuth flow.
        Returns authorization URL for frontend to redirect to.
        """
        if not google_oauth_client:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Google OAuth is not configured. Please set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET."
            )

        try:
            redirect_url = str(request.url_for("google_callback"))
            logger.info(f"🔐 OAuth: Generating auth URL with callback: {redirect_url}")

            authorization_url = await google_oauth_client.get_authorization_url(
                redirect_url,
                scope=["openid", "email", "profile"],
            )

            logger.info(f"✅ OAuth: Authorization URL generated successfully")
            
            return JSONResponse(
                content={
                    "authorization_url": authorization_url,
                    "success": True
                }
            )
        except Exception as e:
            logger.error(f"❌ OAuth: Failed to generate authorization URL: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to initialize OAuth flow: {str(e)}"
            )

    @router.get("/google/callback", tags=["auth"])
    async def google_callback(
        request: Request,
        code: Optional[str] = None,
        error: Optional[str] = None,
        user_manager = Depends(get_user_manager),
    ):
        """
        Handle Google OAuth callback.
        Processes the auth code and creates/logs in user.
        """
        # Handle OAuth errors from Google
        if error:
            logger.error(f"❌ OAuth: Error from Google: {error}")
            error_url = f"{settings.CLIENT_FRONTEND_URL}/login?error=oauth_{error}"
            return RedirectResponse(url=error_url, status_code=status.HTTP_302_FOUND)

        if not code:
            logger.error("❌ OAuth: Missing authorization code")
            error_url = f"{settings.CLIENT_FRONTEND_URL}/login?error=missing_code"
            return RedirectResponse(url=error_url, status_code=status.HTTP_302_FOUND)

        try:
            # Step 1: Exchange code for access token
            redirect_url = str(request.url_for("google_callback"))
            logger.info(f"🔐 OAuth: Exchanging code for token...")
            
            access_token = await google_oauth_client.get_access_token(code, redirect_url)
            logger.info(f"✅ OAuth: Access token obtained")

            # Step 2: Get user info from Google
            logger.info(f"👤 OAuth: Fetching user info from Google...")
            user_id, user_email = await google_oauth_client.get_id_email(
                access_token["access_token"]
            )
            logger.info(f"✅ OAuth: User info retrieved - {user_email}")

            # Step 3: Create or login user via fastapi-users
            try:
                logger.info(f"💾 OAuth: Processing user account...")
                
                user = await user_manager.oauth_callback(
                    oauth_name="google",
                    access_token=access_token["access_token"],
                    account_id=str(user_id),
                    account_email=user_email,
                    expires_at=access_token.get("expires_at"),
                    refresh_token=access_token.get("refresh_token"),
                    request=request,
                    associate_by_email=True,  # Link to existing account if email matches
                    is_verified_by_default=True,  # Google accounts are verified
                )
                
                logger.info(f"✅ OAuth: User processed successfully - ID: {user.id}")

            except exceptions.UserAlreadyExists:
                # This shouldn't happen with associate_by_email=True, but handle it
                logger.info(f"🔗 OAuth: User exists, attempting account link...")
                
                existing_user = await user_manager.get_by_email(user_email)
                if not existing_user:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail="User exists but couldn't be found"
                    )
                
                user = existing_user
                logger.info(f"✅ OAuth: Linked to existing account - ID: {user.id}")

            # Step 4: Generate JWT token
            logger.info(f"🎫 OAuth: Generating JWT token...")
            jwt_strategy = get_jwt_strategy()
            token = await jwt_strategy.write_token(user)
            logger.info(f"✅ OAuth: JWT token generated")

            # Step 5: Redirect to frontend with token
            frontend_url = f"{settings.CLIENT_FRONTEND_URL}/oauth-callback"
            redirect_with_token = f"{frontend_url}?access_token={token}&token_type=bearer"
            
            logger.info(f"🎉 OAuth: Flow completed successfully, redirecting to frontend")
            return RedirectResponse(url=redirect_with_token, status_code=status.HTTP_302_FOUND)

        except GetIdEmailError as e:
            logger.error(f"❌ OAuth: Failed to get user info from Google: {e}")
            error_url = f"{settings.CLIENT_FRONTEND_URL}/login?error=google_api_failed"
            return RedirectResponse(url=error_url, status_code=status.HTTP_302_FOUND)
        
        except Exception as e:
            logger.error(f"❌ OAuth: Unexpected error: {e}", exc_info=True)
            error_url = f"{settings.CLIENT_FRONTEND_URL}/login?error=oauth_failed"
            return RedirectResponse(url=error_url, status_code=status.HTTP_302_FOUND)

else:
    logger.warning("⚠️  Google OAuth is disabled (credentials not configured)")

# =============================================================================
# HEALTH CHECK ENDPOINT
# =============================================================================

@router.get("/health", tags=["auth"])
async def auth_health_check():
    """
    Check authentication system health and configuration.
    """
    return {
        "status": "healthy",
        "authentication": {
            "jwt_enabled": True,
            "google_oauth_enabled": google_oauth_client is not None,
        },
        "endpoints": {
            "register": "/api/v1/auth/register",
            "login": "/api/v1/auth/jwt/login",
            "logout": "/api/v1/auth/jwt/logout",
            "profile": "/api/v1/auth/users/me",
            "update_profile": "/api/v1/auth/users/me (PATCH)",
            "google_authorize": "/api/v1/auth/google/authorize" if google_oauth_client else None,
            "google_callback": "/api/v1/auth/google/callback" if google_oauth_client else None,
        }
    }