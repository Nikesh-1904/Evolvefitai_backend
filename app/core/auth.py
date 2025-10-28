# app/core/auth.py
import uuid
from typing import Optional, Dict, Any

from fastapi import Depends, Request, Response
from fastapi_users import BaseUserManager, FastAPIUsers, UUIDIDMixin
from fastapi_users.authentication import AuthenticationBackend, BearerTransport, JWTStrategy
from fastapi_users.db import SQLAlchemyUserDatabase
from httpx_oauth.clients.google import GoogleOAuth2

from sqlalchemy import select
from sqlalchemy.orm import selectinload
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import get_async_session
from app.models import User, OAuthAccount

SECRET = settings.SECRET_KEY

class UserManager(UUIDIDMixin, BaseUserManager[User, uuid.UUID]):
    reset_password_token_secret = SECRET
    verification_token_secret = SECRET

    async def on_after_register(self, user: User, request: Optional[Request] = None):
        print(f"User {user.id} has registered.")

    async def on_after_forgot_password(self, user: User, token: str, request: Optional[Request] = None):
        print(f"User {user.id} has requested a password reset. Token: {token}")

    async def on_after_request_verify(self, user: User, token: str, request: Optional[Request] = None):
        print(f"Verification requested for user {user.id}. Token: {token}")

    async def on_after_login(
        self,
        user: User,
        request: Optional[Request] = None,
        response: Optional[Response] = None,
    ):
        print(f"User {user.id} logged in.")

class CustomUserDatabase(SQLAlchemyUserDatabase[User, uuid.UUID]):
    # Eager-load oauth_accounts everywhere a User is fetched

    async def get(self, id: uuid.UUID) -> Optional[User]:
        stmt = (
            select(User)
            .where(User.id == id)  # type: ignore[arg-type]
            .options(
                selectinload(User.oauth_accounts),
            )
        )
        return await self._get_user(stmt)

    async def get_by_email(self, email: str) -> Optional[User]:
        stmt = (
            select(User)
            .where(User.email == email)  # type: ignore[arg-type]
            .options(
                selectinload(User.oauth_accounts),
            )
        )
        return await self._get_user(stmt)

    async def get_by_oauth_account(self, oauth: str, account_id: str) -> Optional[User]:
        stmt = (
            select(User)
            .join(self.oauth_account_model)
            .where(self.oauth_account_model.oauth_name == oauth)      # type: ignore[arg-type]
            .where(self.oauth_account_model.account_id == account_id) # type: ignore[arg-type]
            .options(
                selectinload(User.oauth_accounts),
            )
        )
        return await self._get_user(stmt)

    async def create(self, create_dict: Dict[str, Any]) -> User:
        user = User(**create_dict)
        self.session.add(user)
        await self.session.commit()
        # Return with relationship loaded
        return await self.get(user.id)  # type: ignore

    async def add_oauth_account(self, user: User, oauth_account_dict: Dict[str, Any]) -> User:
        # Construct the OAuth account model and link
        oauth_account = self.oauth_account_model(**oauth_account_dict, user_id=user.id)
        self.session.add(oauth_account)

        # Append to the mapped relationship; safe because of eager loading above
        user.oauth_accounts.append(oauth_account)  # type: ignore[attr-defined]

        await self.session.commit()
        # Do not refresh() to avoid losing already loaded relationships
        return user

async def get_user_db(session: AsyncSession = Depends(get_async_session)):
    # IMPORTANT: include OAuthAccount model here
    yield CustomUserDatabase(session, User, OAuthAccount)

async def get_user_manager(user_db=Depends(get_user_db)):
    yield UserManager(user_db)

def get_jwt_strategy() -> JWTStrategy:
    return JWTStrategy(secret=SECRET, lifetime_seconds=3600 * 24)

bearer_transport = BearerTransport(tokenUrl="auth/jwt/login")

auth_backend = AuthenticationBackend(
    name="jwt",
    transport=bearer_transport,
    get_strategy=get_jwt_strategy,
)

google_oauth_client: Optional[GoogleOAuth2] = None
if settings.GOOGLE_CLIENT_ID and settings.GOOGLE_CLIENT_SECRET:
    google_oauth_client = GoogleOAuth2(
        client_id=settings.GOOGLE_CLIENT_ID,
        client_secret=settings.GOOGLE_CLIENT_SECRET,
    )

fastapi_users = FastAPIUsers[User, uuid.UUID](
    get_user_manager,
    [auth_backend],
)

current_active_user = fastapi_users.current_user(active=True)
