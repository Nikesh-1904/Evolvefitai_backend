# app/api/v1/auth.py

from fastapi import APIRouter
from app.core.auth import auth_backend, fastapi_users, google_oauth_client
from app.core.config import settings
from app.schemas import UserRead, UserCreate, UserUpdate

router = APIRouter()

# --- JWT Login and Logout ---
# This provides the POST /api/v1/auth/jwt/login endpoint that the frontend needs.
router.include_router(
    fastapi_users.get_auth_router(auth_backend),
    prefix="/jwt",
    tags=["authentication"],
)

# --- Registration ---
# This provides the POST /api/v1/auth/register endpoint that was returning a 404 error.
router.include_router(
    fastapi_users.get_register_router(UserRead, UserCreate),
    tags=["authentication"],
)

# --- Password Reset ---
# Provides POST /auth/forgot-password and POST /auth/reset-password
router.include_router(
    fastapi_users.get_reset_password_router(),
    tags=["authentication"],
)

# --- Email Verification ---
# Provides POST /auth/request-verify-token and POST /auth/verify
router.include_router(
    fastapi_users.get_verify_router(UserRead),
    tags=["authentication"],
)

# --- User Management ---
# Provides the GET /users/me and PATCH /users/me endpoints that the Profile page uses.
router.include_router(
    fastapi_users.get_users_router(UserRead, UserUpdate),
    prefix="/users",
    tags=["users"],
)

# --- Google OAuth ---
# Includes the authorize and callback endpoints for Google login.
# This uses the standard library implementation which should work seamlessly.
if google_oauth_client:
    router.include_router(
        fastapi_users.get_oauth_router(
            oauth_client=google_oauth_client,
            backend=auth_backend,
            state_secret=settings.SECRET_KEY,
            # The user is redirected here from Google, and the backend handles the rest.
            redirect_url=f"{settings.FRONTEND_URL}/auth/callback",
        ),
        prefix="/google",
        tags=["authentication"],
    )