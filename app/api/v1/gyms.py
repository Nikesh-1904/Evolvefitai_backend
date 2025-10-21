# app/api/v1/gyms.py
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_
from typing import List, Optional
from datetime import datetime

from app.core.database import get_async_session
from app.core.auth import current_active_user
from app.models import User, Gym, GymBooking
from app.schemas import (
    Gym as GymSchema,
    GymCreate,
    GymOccupancyResponse,
    MessageResponse
)

router = APIRouter()


@router.get("/", response_model=List[GymSchema])
async def list_gyms(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    search: Optional[str] = Query(None, description="Search by gym name or city"),
    session: AsyncSession = Depends(get_async_session),
    current_user: User = Depends(current_active_user)
):
    """List all gyms with optional search and pagination"""
    query = select(Gym)
    
    if search:
        search_filter = or_(
            Gym.name.ilike(f"%{search}%"),
            Gym.city.ilike(f"%{search}%"),
            Gym.address.ilike(f"%{search}%")
        )
        query = query.where(search_filter)
    
    query = query.offset(skip).limit(limit).order_by(Gym.name)
    result = await session.execute(query)
    gyms = result.scalars().all()
    return gyms


@router.get("/{gym_id}", response_model=GymSchema)
async def get_gym(
    gym_id: int,
    session: AsyncSession = Depends(get_async_session),
    current_user: User = Depends(current_active_user)
):
    """Get a specific gym by ID"""
    query = select(Gym).where(Gym.id == gym_id)
    result = await session.execute(query)
    gym = result.scalar_one_or_none()
    
    if not gym:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Gym not found"
        )
    
    return gym


@router.post("/", response_model=GymSchema)
async def create_gym(
    gym_data: GymCreate,
    session: AsyncSession = Depends(get_async_session),
    current_user: User = Depends(current_active_user)
):
    """Create a new gym (admin only in production)"""
    # In production, you might want to add admin role check here
    
    # Check if gym with same google_place_id already exists
    if gym_data.google_place_id:
        existing_query = select(Gym).where(Gym.google_place_id == gym_data.google_place_id)
        existing_result = await session.execute(existing_query)
        if existing_result.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Gym with this Google Place ID already exists"
            )
    
    gym = Gym(**gym_data.model_dump())
    session.add(gym)
    await session.commit()
    await session.refresh(gym)
    return gym


@router.get("/{gym_id}/occupancy", response_model=GymOccupancyResponse)
async def get_gym_occupancy(
    gym_id: int,
    session: AsyncSession = Depends(get_async_session),
    current_user: User = Depends(current_active_user)
):
    """Get current occupancy status of a gym"""
    # Get gym details
    gym_query = select(Gym).where(Gym.id == gym_id)
    gym_result = await session.execute(gym_query)
    gym = gym_result.scalar_one_or_none()
    
    if not gym:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Gym not found"
        )
    
    # Get current active bookings (within current time window)
    current_time = datetime.utcnow()
    
    active_bookings_query = select(func.count(GymBooking.id)).where(
        and_(
            GymBooking.gym_id == gym_id,
            GymBooking.status == "active",
            GymBooking.start_time <= current_time,
            GymBooking.end_time >= current_time
        )
    )
    
    active_bookings_result = await session.execute(active_bookings_query)
    active_bookings_count = active_bookings_result.scalar() or 0
    
    # Calculate occupancy metrics
    current_occupancy = min(active_bookings_count, gym.max_capacity)
    overflow_count = max(0, active_bookings_count - gym.max_capacity)
    is_overcrowded = active_bookings_count > gym.max_capacity
    
    return GymOccupancyResponse(
        gym_id=gym.id,
        gym_name=gym.name,
        current_occupancy=current_occupancy,
        max_capacity=gym.max_capacity,
        overflow_count=overflow_count,
        is_overcrowded=is_overcrowded,
        active_bookings_count=active_bookings_count
    )


@router.post("/{gym_id}/join", response_model=MessageResponse)
async def join_gym(
    gym_id: int,
    session: AsyncSession = Depends(get_async_session),
    current_user: User = Depends(current_active_user)
):
    """Join a gym (set as user's primary gym)"""
    # Verify gym exists
    gym_query = select(Gym).where(Gym.id == gym_id)
    gym_result = await session.execute(gym_query)
    gym = gym_result.scalar_one_or_none()
    
    if not gym:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Gym not found"
        )
    
    # Update user's gym affiliation
    current_user.gym_id = gym_id
    current_user.last_gym_change = datetime.utcnow()
    
    await session.commit()
    
    return MessageResponse(
        message=f"Successfully joined {gym.name}",
        success=True
    )


@router.delete("/{gym_id}/leave", response_model=MessageResponse)
async def leave_gym(
    gym_id: int,
    session: AsyncSession = Depends(get_async_session),
    current_user: User = Depends(current_active_user)
):
    """Leave current gym"""
    if current_user.gym_id != gym_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You are not a member of this gym"
        )
    
    current_user.gym_id = None
    current_user.last_gym_change = datetime.utcnow()
    
    await session.commit()
    
    return MessageResponse(
        message="Successfully left the gym",
        success=True
    )
