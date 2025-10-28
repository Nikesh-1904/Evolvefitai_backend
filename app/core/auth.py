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
    async def _get_user(self, stmt):
        result = await self.session.execute(stmt)
        return result.unique().scalars().first()

    async def get(self, id: uuid.UUID) -> Optional[User]:
        stmt = (
            select(User)
            .where(User.id == id)  # type: ignore[arg-type]
            .options(selectinload(User.oauth_accounts))
        )
        return await self._get_user(stmt)

    async def get_by_email(self, email: str) -> Optional[User]:
        stmt = (
            select(User)
            .where(User.email == email)  # type: ignore[arg-type]
            .options(selectinload(User.oauth_accounts))
        )
        return await self._get_user(stmt)

    async def get_by_oauth_account(self, oauth: str, account_id: str) -> Optional[User]:
        # Requires self.oauth_account_model; constructor must pass OAuthAccount
        stmt = (
            select(User)
            .join(self.oauth_account_model)  # type: ignore[attr-defined]
            .where(self.oauth_account_model.oauth_name == oauth)      # type: ignore[attr-defined]
            .where(self.oauth_account_model.account_id == account_id) # type: ignore[attr-defined]
            .options(selectinload(User.oauth_accounts))
        )
        return await self._get_user(stmt)

    async def create(self, create_dict: Dict[str, Any]) -> User:
        user = User(**create_dict)
        self.session.add(user)
        await self.session.commit()
        return await self.get(user.id)  # type: ignore

    async def add_oauth_account(self, user: User, oauth_account_dict: Dict[str, Any]) -> User:
        oauth_account = self.oauth_account_model(**oauth_account_dict, user_id=user.id)  # type: ignore[attr-defined]
        self.session.add(oauth_account)
        user.oauth_accounts.append(oauth_account)  # type: ignore[attr-defined]
        await self.session.commit()
        return user

async def get_user_db(session: AsyncSession = Depends(get_async_session)):
    db = CustomUserDatabase(session, User, OAuthAccount)
    print(f"[Auth DI] Constructed {type(db).__name__} from module {__name__}")
    if not hasattr(db, "oauth_account_model"):
        print(f"[Auth DI] ERROR: oauth_account_model missing on {type(db).__name__} from {__name__}")
        raise RuntimeError("OAuth adapter misconfigured: oauth_account_model missing; ensure CustomUserDatabase(session, User, OAuthAccount) is used and no duplicate core/auth module is imported.")
    yield db

async def get_user_manager(user_db=Depends(get_user_db)):
    has_oauth = hasattr(user_db, "oauth_account_model")
    print(f"[Auth DI] user_db={type(user_db).__name__}, oauth_account_model_present={has_oauth}, provider_module={__name__}")
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

fastapi_users = FastAPIUsers[User, uuid.UUID](get_user_manager, [auth_backend])
current_active_user = fastapi_users.current_user(active=True)
