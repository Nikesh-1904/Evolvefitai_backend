# app/api/v1/auth.py
from fastapi import APIRouter, Depends
from fastapi_users import schemas
from app.core.config import settings
from app.core.auth import auth_backend, fastapi_users, google_oauth_client
from app.schemas import UserRead, UserCreate, UserUpdate
from app.core.auth import get_user_manager


router = APIRouter()

# JWT auth routes
# The prefix="/jwt" has been REMOVED from this section.
# This makes the login URL .../auth/login (Correct)
# instead of .../auth/jwt/login (Incorrectly nested)
router.include_router(
    fastapi_users.get_auth_router(auth_backend), tags=["auth"]
)

# Registration routes
# The prefix="/register" has been REMOVED from this section.
# This fixes the 404 error.
router.include_router(
    fastapi_users.get_register_router(UserRead, UserCreate),
    tags=["auth"],
)

# User management routes
# This prefix is correct because you likely want the route to be /users/me etc.
router.include_router(
    fastapi_users.get_users_router(UserRead, UserUpdate),
    prefix="/users",
    tags=["users"],
)

# Google OAuth routes (if configured)
if google_oauth_client:
    router.include_router(
        fastapi_users.get_oauth_router(
            google_oauth_client,
            auth_backend,
            settings.SECRET_KEY,
            associate_by_email=True,
            is_verified_by_default=True,
        ),
        prefix="/google",
        tags=["auth"],
    )

# Custom profile endpoint
@router.get("/me", response_model=UserRead, tags=["users"])
async def get_current_user_profile(user=Depends(fastapi_users.current_user(active=True))):
    """Get current user profile"""
    return user

@router.patch("/me", response_model=UserRead, tags=["users"])
async def update_current_user_profile(
    user_update: UserUpdate,
    user=Depends(fastapi_users.current_user(active=True)),
    user_manager=Depends(get_user_manager)
):
    """Update current user profile"""
    return await user_manager.update(user_update, user)