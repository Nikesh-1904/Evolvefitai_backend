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

if google_oauth_client:

    @router.get("/google/authorize", tags=["auth"])
    async def google_authorize(request: Request):
        if not google_oauth_client:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Google OAuth is not configured")
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
        if error:
            error_url = f"{settings.FRONTEND_URL}/login?error={error}"
            return RedirectResponse(url=error_url, status_code=status.HTTP_302_FOUND)

        if not code:
            error_url = f"{settings.FRONTEND_URL}/login?error=missing_authorization_code"
            return RedirectResponse(url=error_url, status_code=status.HTTP_302_FOUND)

        try:
            redirect_url = str(request.url_for("google_callback"))
            access_token = await google_oauth_client.get_access_token(code, redirect_url)
            user_id, user_email = await google_oauth_client.get_id_email(access_token["access_token"])

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
                else:
                    error_url = f"{settings.FRONTEND_URL}/login?error=account_linking_failed"
                    return RedirectResponse(url=error_url, status_code=status.HTTP_302_FOUND)
            except Exception:
                error_url = f"{settings.FRONTEND_URL}/login?error=user_creation_failed"
                return RedirectResponse(url=error_url, status_code=status.HTTP_302_FOUND)

            jwt_strategy = get_jwt_strategy()
            token = await jwt_strategy.write_token(user)

            frontend_callback_url = f"{settings.FRONTEND_URL}/auth/callback#access_token={token}&token_type=bearer"
            return RedirectResponse(url=frontend_callback_url, status_code=status.HTTP_302_FOUND)

        except GetIdEmailError:
            error_url = f"{settings.FRONTEND_URL}/login?error=google_profile_access_failed"
            return RedirectResponse(url=error_url, status_code=status.HTTP_302_FOUND)
        except Exception:
            error_url = f"{settings.FRONTEND_URL}/login?error=oauth_failed"
            return RedirectResponse(url=error_url, status_code=status.HTTP_302_FOUND)


@router.get("/me", response_model=UserRead, tags=["users"])
async def get_current_user(user=Depends(fastapi_users.current_user(active=True))):
    return user


@router.patch("/me", response_model=UserRead, tags=["users"])
async def update_current_user(
    user_update: UserUpdate,
    user=Depends(fastapi_users.current_user(active=True)),
    user_manager=Depends(get_user_manager),
):
    return await user_manager.update(user_update, user)
