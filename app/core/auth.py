# app/core/auth.py
import uuid
from typing import Optional, Dict, Any
from fastapi import Depends, Request, Response  # <--- Import Response
from fastapi_users import BaseUserManager, FastAPIUsers, UUIDIDMixin
from fastapi_users.authentication import AuthenticationBackend, BearerTransport, JWTStrategy
from fastapi_users.db import SQLAlchemyUserDatabase
from httpx_oauth.clients.google import GoogleOAuth2
from sqlalchemy.orm import selectinload
from sqlalchemy import select
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

    async def on_after_forgot_password(
        self, user: User, token: str, request: Optional[Request] = None
    ):
        print(f"User {user.id} has requested a password reset. Token: {token}")

    async def on_after_request_verify(
        self, user: User, token: str, request: Optional[Request] = None
    ):
        print(f"Verification requested for user {user.id}. Token: {token}")

    # --- FIX: Added the 'response' parameter to match the library's expectation ---
    async def on_after_login(
        self,
        user: User,
        request: Optional[Request] = None,
        response: Optional[Response] = None
    ):
        print(f"User {user.id} logged in.")

class CustomUserDatabase(SQLAlchemyUserDatabase[User, uuid.UUID]):

    # --- FIX: Signature now matches the parent class ---
    async def get(self, id: uuid.UUID) -> Optional[User]:
        """
        Fetches a user by ID, eagerly loading oauth_accounts.
        """
        statement = (
            select(User)
            .where(User.id == id)  # type: ignore[arg-type]
            .options(
                selectinload(User.oauth_accounts),
                selectinload(User.achievements)  # <-- THE FIX
            )
        )
        return await self._get_user(statement)

    # --- FIX: Signature now matches the parent class ---
    async def get_by_email(self, email: str) -> Optional[User]:
        """
        Fetches a user by email, eagerly loading oauth_accounts
        to prevent async lazy-load errors during linking.
        """
        statement = (
            select(User)  # --- FIX: Use concrete User model
            .where(User.email == email)  # --- FIX: Use concrete User model
            .options(selectinload(User.oauth_accounts))  # --- FIX: THE RUNTIME FIX
        )
        return await self._get_user(statement)

    async def create(self, create_dict: Dict[str, Any]) -> User:
        """
        Creates a new user, then immediately fetches them
        with the oauth_accounts relationship loaded.
        """
        user = User(**create_dict)
        self.session.add(user)
        await self.session.commit()
        
        # REMOVED: await self.session.refresh(user)
        # This line was redundant and caused lazy-loading issues.
        
        # The .get() will fetch the newly committed user from the DB
        # with all the eager-loaded relationships.
        return await self.get(user.id)  # type: ignore 
# --- END NEW CLASS ---
    
async def get_user_db(session: AsyncSession = Depends(get_async_session)):
    yield CustomUserDatabase(session, User, OAuthAccount)


async def get_user_manager(user_db=Depends(get_user_db)):
    yield UserManager(user_db)


def get_jwt_strategy() -> JWTStrategy:
    return JWTStrategy(secret=SECRET, lifetime_seconds=3600 * 24)  # 24 hours


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