# app/core/business_auth.py
"""Authentication system for gym owners/business users"""

import uuid
from typing import Optional
from datetime import datetime, timedelta
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.config import settings
from app.core.database import get_async_session
from app.models import GymOwner

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Bearer token scheme
bearer_scheme = HTTPBearer()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password"""
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create JWT access token for gym owner
    
    Args:
        data: Token payload (should include owner_id, gym_id, role)
        expires_delta: Token expiration time
    
    Returns:
        Encoded JWT token
    """
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(hours=24)  # 24-hour tokens
    
    to_encode.update({"exp": expire})
    to_encode.update({"role": "gym_owner"})  # Mark as business token
    
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt


async def get_gym_owner_by_email(
    email: str, 
    session: AsyncSession
) -> Optional[GymOwner]:
    """Get gym owner by email"""
    result = await session.execute(
        select(GymOwner).where(GymOwner.email == email)
    )
    return result.scalar_one_or_none()


async def authenticate_gym_owner(
    email: str,
    password: str,
    session: AsyncSession
) -> Optional[GymOwner]:
    """
    Authenticate gym owner with email and password
    
    Args:
        email: Owner's email
        password: Plain password
        session: Database session
    
    Returns:
        GymOwner if authenticated, None otherwise
    """
    owner = await get_gym_owner_by_email(email, session)
    
    if not owner:
        return None
    
    if not verify_password(password, owner.hashed_password):
        return None
    
    if not owner.is_active:
        return None
    
    return owner


async def get_current_gym_owner(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    session: AsyncSession = Depends(get_async_session)
) -> GymOwner:
    """
    Get current authenticated gym owner from JWT token
    
    This is the main dependency for protecting business endpoints
    
    Raises:
        HTTPException: If token is invalid or owner not found
    
    Returns:
        Authenticated GymOwner instance
    """
    token = credentials.credentials
    
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        owner_id: str = payload.get("sub")
        role: str = payload.get("role")
        
        if owner_id is None or role != "gym_owner":
            raise credentials_exception
            
    except JWTError:
        raise credentials_exception
    
    # Get owner from database
    owner = await session.get(GymOwner, uuid.UUID(owner_id))
    
    if owner is None:
        raise credentials_exception
    
    if not owner.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is inactive"
        )
    
    # Update last login
    owner.last_login = datetime.utcnow()
    await session.commit()
    
    return owner


async def verify_gym_access(
    gym_id: uuid.UUID,
    current_owner: GymOwner = Depends(get_current_gym_owner)
) -> GymOwner:
    """
    Verify that the current gym owner has access to the specified gym
    
    Use this as a dependency when endpoints need gym_id parameter
    
    Args:
        gym_id: Gym UUID to check access for
        current_owner: Currently authenticated owner
    
    Raises:
        HTTPException: If owner doesn't have access to this gym
    
    Returns:
        GymOwner if access granted
    """
    if current_owner.gym_id != gym_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have access to this gym"
        )
    
    return current_owner
