import uuid
from typing import Optional
from fastapi import Depends, Request
from fastapi_users import BaseUserManager, FastAPIUsers, UUIDIDMixin
from fastapi_users.authentication import AuthenticationBackend, BearerTransport
from fastapi_users.authentication.strategy import JWTStrategy
from fastapi_users.db import SQLAlchemyUserDatabase
from httpx_oauth.clients.google import GoogleOAuth2

from app.core.config import settings
from app.core.database import get_async_session
from app.models import User

SECRET = settings.SECRET_KEY


class UserManager(UUIDIDMixin, BaseUserManager[User, uuid.UUID]):
    reset_password_token_secret = SECRET
    verification_token_secret = SECRET

    async def on_after_register(self, user: User, request: Optional[Request] = None):
        print(f"User {user.id} has registered.")

    async def on_after_forgot_password(
        self, user: User, token: str, request: Optional[Request] = None
    ):
        print(f"User {user.id} requested password reset. Token: {token}")

    async def on_after_request_verify(
        self, user: User, token: str, request: Optional[Request] = None
    ):
        print(f"Verification requested for user {user.id}. Token: {token}")


async def get_user_db(session=Depends(get_async_session)):
    yield SQLAlchemyUserDatabase(session, User)


async def get_user_manager(user_db=Depends(get_user_db)):
    yield UserManager(user_db)


def get_jwt_strategy() -> JWTStrategy:
    return JWTStrategy(secret=SECRET, lifetime_seconds=3600)


bearer_transport = BearerTransport(tokenUrl="auth/login") # Corrected from /auth/jwt/login

auth_backend = AuthenticationBackend(
    name="jwt",
    transport=bearer_transport,
    get_strategy=get_jwt_strategy,
)

# Google OAuth2 client setup if keys provided
if settings.GOOGLE_CLIENT_ID and settings.GOOGLE_CLIENT_SECRET:
    # --- THIS IS THE CORRECTED SECTION ---
    # We explicitly define the redirect_uri to prevent misinterpretation by proxies.
    redirect_uri = f"{settings.BACKEND_URL}/api/v1/auth/google/callback"
    
    google_oauth_client = GoogleOAuth2(
        client_id=settings.GOOGLE_CLIENT_ID,
        client_secret=settings.GOOGLE_CLIENT_SECRET,
        redirect_uri=redirect_uri,
        name="google" # Add name for multi-provider support
    )
else:
    google_oauth_client = None

fastapi_users = FastAPIUsers[User, uuid.UUID](get_user_manager, [auth_backend])

current_active_user = fastapi_users.current_user(active=True)
current_user = fastapi_users.current_user()
