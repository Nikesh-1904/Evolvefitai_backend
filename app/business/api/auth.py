# app/business/api/auth.py
"""Authentication endpoints for gym owners"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import timedelta

from app.core.database import get_async_session
from app.core.business_auth import (
    authenticate_gym_owner,
    create_access_token,
    get_password_hash,
    get_current_gym_owner
)
from app import models, schemas

router = APIRouter()


@router.post("/register", response_model=schemas.GymOwnerRead, status_code=status.HTTP_201_CREATED)
async def register_gym_owner(
    owner_data: schemas.GymOwnerCreate,
    session: AsyncSession = Depends(get_async_session)
):
    """
    Register a new gym owner account
    
    - **email**: Owner's email (must be unique)
    - **password**: Account password (will be hashed)
    - **full_name**: Owner's full name
    - **phone_number**: Contact number (optional)
    - **gym_id**: Gym they will manage
    """
    # Check if email already exists
    existing = await session.execute(
        models.select(models.GymOwner).where(models.GymOwner.email == owner_data.email)
    )
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Verify gym exists
    gym = await session.get(models.Gym, owner_data.gym_id)
    if not gym:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Gym not found"
        )
    
    # Create gym owner
    hashed_password = get_password_hash(owner_data.password)
    
    new_owner = models.GymOwner(
        email=owner_data.email,
        hashed_password=hashed_password,
        full_name=owner_data.full_name,
        phone_number=owner_data.phone_number,
        gym_id=owner_data.gym_id,
        is_active=True
    )
    
    session.add(new_owner)
    await session.commit()
    await session.refresh(new_owner)
    
    return new_owner


@router.post("/login")
async def login_gym_owner(
    credentials: schemas.GymOwnerLogin,
    session: AsyncSession = Depends(get_async_session)
):
    """
    Login as gym owner and receive JWT access token
    
    - **email**: Owner's email
    - **password**: Account password
    
    Returns:
    - **access_token**: JWT token for authentication
    - **token_type**: Bearer
    - **owner**: Owner profile data
    """
    owner = await authenticate_gym_owner(credentials.email, credentials.password, session)
    
    if not owner:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create access token
    access_token_expires = timedelta(hours=24)
    access_token = create_access_token(
        data={"sub": str(owner.id), "gym_id": str(owner.gym_id)},
        expires_delta=access_token_expires
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "owner": {
            "id": str(owner.id),
            "email": owner.email,
            "full_name": owner.full_name,
            "gym_id": str(owner.gym_id),
            "is_active": owner.is_active
        }
    }


@router.get("/me", response_model=schemas.GymOwnerRead)
async def get_current_owner_profile(
    current_owner: models.GymOwner = Depends(get_current_gym_owner)
):
    """
    Get current gym owner's profile
    
    Requires authentication (Bearer token in header)
    """
    return current_owner


@router.patch("/me", response_model=schemas.GymOwnerRead)
async def update_owner_profile(
    updates: schemas.GymOwnerUpdate,
    current_owner: models.GymOwner = Depends(get_current_gym_owner),
    session: AsyncSession = Depends(get_async_session)
):
    """
    Update current gym owner's profile
    
    Can update:
    - full_name
    - phone_number
    """
    if updates.full_name is not None:
        current_owner.full_name = updates.full_name
    
    if updates.phone_number is not None:
        current_owner.phone_number = updates.phone_number
    
    await session.commit()
    await session.refresh(current_owner)
    
    return current_owner
