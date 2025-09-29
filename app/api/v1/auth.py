# app/api/v1/auth.py
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
    jwt_authentication,
)
from app.schemas import UserRead, UserCreate, UserUpdate

router = APIRouter()

# Standard FastAPI-Users routes with JWT prefix for compatibility
router.include_router(
    fastapi_users.get_auth_router(auth_backend),
    prefix="/jwt",
    tags=["auth"],
)

router.include_router(
    fastapi_users.get_register_router(UserRead, UserCreate),
    prefix="/register",
    tags=["auth"],
)

router.include_router(
    fastapi_users.get_users_router(UserRead, UserUpdate),
    prefix="/users",
    tags=["users"],
)

# Custom Google OAuth implementation that redirects to frontend
if google_oauth_client:

    @router.get("/google/authorize", tags=["auth"])
    async def google_authorize(request: Request):
        """Initiate Google OAuth flow"""
        if not google_oauth_client:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Google OAuth is not configured"
            )
        
        # Build the callback URL for this endpoint
        redirect_url = str(request.url_for("google_callback"))
        
        # Get Google's authorization URL
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
        user_manager=Depends(get_user_manager)
    ):
        """Handle Google OAuth callback and redirect to frontend"""
        
        # Handle OAuth error responses
        if error:
            error_url = f"{settings.FRONTEND_URL}/login?error={error}"
            return RedirectResponse(url=error_url, status_code=status.HTTP_302_FOUND)
        
        if not code:
            error_url = f"{settings.FRONTEND_URL}/login?error=missing_authorization_code"
            return RedirectResponse(url=error_url, status_code=status.HTTP_302_FOUND)

        try:
            # Exchange authorization code for access token
            redirect_url = str(request.url_for("google_callback"))
            access_token = await google_oauth_client.get_access_token(code, redirect_url)
            
            # Get user info from Google
            user_id, user_email = await google_oauth_client.get_id_email(
                access_token["access_token"]
            )
            
            # Handle user creation/login through FastAPI-Users
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
            except exceptions.UserAlreadyExists:
                # User exists but not linked to OAuth, try to get existing user
                try:
                    existing_user = await user_manager.get_by_email(user_email)
                    if existing_user:
                        # Link OAuth account to existing user
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
                    else:
                        raise
                except Exception as link_error:
                    print(f"OAuth linking error: {link_error}")
                    error_url = f"{settings.FRONTEND_URL}/login?error=account_linking_failed"
                    return RedirectResponse(url=error_url, status_code=status.HTTP_302_FOUND)
            
            except Exception as oauth_error:
                print(f"OAuth user creation error: {oauth_error}")
                error_url = f"{settings.FRONTEND_URL}/login?error=user_creation_failed"
                return RedirectResponse(url=error_url, status_code=status.HTTP_302_FOUND)

            # Generate JWT token for the authenticated user
            token = await jwt_authentication.write_token(user)
            
            # Redirect to frontend with token in URL hash
            frontend_callback_url = f"{settings.FRONTEND_URL}/auth/callback#access_token={token}&token_type=bearer"
            return RedirectResponse(url=frontend_callback_url, status_code=status.HTTP_302_FOUND)
            
        except GetIdEmailError:
            error_url = f"{settings.FRONTEND_URL}/login?error=google_profile_access_failed"
            return RedirectResponse(url=error_url, status_code=status.HTTP_302_FOUND)
        
        except Exception as e:
            print(f"Unexpected OAuth error: {e}")
            error_url = f"{settings.FRONTEND_URL}/login?error=oauth_failed"
            return RedirectResponse(url=error_url, status_code=status.HTTP_302_FOUND)

# Additional custom endpoints for better integration
@router.get("/me", response_model=UserRead, tags=["users"])
async def get_current_user(user=Depends(fastapi_users.current_user(active=True))):
    """Get current user profile"""
    return user

@router.patch("/me", response_model=UserRead, tags=["users"]) 
async def update_current_user(
    user_update: UserUpdate,
    user=Depends(fastapi_users.current_user(active=True)),
    user_manager=Depends(get_user_manager),
):
    """Update current user profile"""
    return await user_manager.update(user_update, user)