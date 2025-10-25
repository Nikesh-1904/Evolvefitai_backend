# app/shared/api/gyms.py
"""Shared gym endpoints - used by both clients and business owners"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, or_
from typing import List, Optional
from datetime import datetime
import uuid

from app.core.database import get_async_session
from app.core.auth import current_active_user
from app import models, schemas

router = APIRouter()


@router.get("/", response_model=List[schemas.Gym])
async def list_gyms(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    search: Optional[str] = Query(None, description="Search by gym name or city"),
    session: AsyncSession = Depends(get_async_session),
    current_user: models.User = Depends(current_active_user)
):
    """
    List all gyms with optional search
    
    Available to all authenticated users
    """
    query = select(models.Gym)
    
    # Apply search filter
    if search:
        search_filter = or_(
            models.Gym.name.ilike(f"%{search}%"),
            models.Gym.city.ilike(f"%{search}%")
        )
        query = query.where(search_filter)
    
    # Apply pagination
    query = query.offset(skip).limit(limit).order_by(models.Gym.name)
    
    result = await session.execute(query)
    gyms = result.scalars().all()
    
    return gyms


@router.get("/{gym_id}", response_model=schemas.Gym)
async def get_gym_details(
    gym_id: uuid.UUID,
    session: AsyncSession = Depends(get_async_session),
    current_user: models.User = Depends(current_active_user)
):
    """
    Get detailed information about a specific gym
    """
    gym = await session.get(models.Gym, gym_id)
    
    if not gym:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Gym not found"
        )
    
    return gym


@router.get("/{gym_id}/occupancy", response_model=schemas.GymOccupancyResponse)
async def get_gym_occupancy(
    gym_id: uuid.UUID,
    session: AsyncSession = Depends(get_async_session),
    current_user: models.User = Depends(current_active_user)
):
    """
    Get current gym occupancy based on bookings
    
    **DEPRECATED**: Use /gyms/{gym_id}/occupancy/live for real-time data
    
    This endpoint uses booking system (old method)
    """
    gym = await session.get(models.Gym, gym_id)
    
    if not gym:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Gym not found"
        )
    
    # Count active bookings (within current hour)
    now = datetime.utcnow()
    current_hour_start = now.replace(minute=0, second=0, microsecond=0)
    current_hour_end = current_hour_start.replace(minute=59, second=59)
    
    current_occupancy = await session.scalar(
        select(func.count(models.GymBooking.id))
        .where(models.GymBooking.gym_id == gym_id)
        .where(models.GymBooking.status == "confirmed")
        .where(models.GymBooking.start_time <= now)
        .where(models.GymBooking.end_time >= now)
    )
    
    occupancy_percentage = round((current_occupancy / gym.capacity) * 100, 1) if gym.capacity > 0 else 0
    
    return {
        "gym_id": gym_id,
        "gym_name": gym.name,
        "current_occupancy": current_occupancy or 0,
        "capacity": gym.capacity,
        "occupancy_percentage": occupancy_percentage,
        "timestamp": datetime.utcnow()
    }


@router.get("/{gym_id}/occupancy/live", response_model=schemas.LiveOccupancyResponse)
async def get_live_gym_occupancy(
    gym_id: uuid.UUID,
    session: AsyncSession = Depends(get_async_session),
    current_user: models.User = Depends(current_active_user)
):
    """
    **NEW**: Get real-time gym occupancy based on QR check-ins
    
    This is the recommended endpoint for accurate occupancy data.
    
    Shows:
    - Current number of people checked in
    - Gym capacity
    - Occupancy percentage
    - List of currently present members (privacy: username only)
    """
    gym = await session.get(models.Gym, gym_id)
    
    if not gym:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Gym not found"
        )
    
    # Count users currently checked in (no check-out time)
    current_count = await session.scalar(
        select(func.count(models.GymAttendance.id))
        .where(models.GymAttendance.gym_id == gym_id)
        .where(models.GymAttendance.check_out_time.is_(None))
    )
    
    # Get list of currently checked-in users (privacy-conscious)
    result = await session.execute(
        select(models.GymAttendance, models.User)
        .join(models.User, models.GymAttendance.user_id == models.User.id)
        .where(models.GymAttendance.gym_id == gym_id)
        .where(models.GymAttendance.check_out_time.is_(None))
        .order_by(models.GymAttendance.check_in_time.desc())
    )
    
    # Only show username (no sensitive data)
    checked_in_users = [
        {
            "user_id": str(user.id) if current_user.gym_id == gym_id else None,  # Only show to same gym members
            "username": user.username or "Member",
            "check_in_time": attendance.check_in_time.isoformat()
        }
        for attendance, user in result.all()
    ]
    
    occupancy_percentage = round((current_count / gym.capacity) * 100, 1) if gym.capacity > 0 else 0
    
    return {
        "gym_id": gym.id,
        "gym_name": gym.name,
        "current_occupancy": current_count or 0,
        "capacity": gym.capacity,
        "occupancy_percentage": occupancy_percentage,
        "checked_in_users": checked_in_users,
        "timestamp": datetime.utcnow()
    }


@router.post("/{gym_id}/join")
async def join_gym_by_code(
    gym_id: uuid.UUID,
    gym_code: str = Query(..., description="Gym access code"),
    session: AsyncSession = Depends(get_async_session),
    current_user: models.User = Depends(current_active_user)
):
    """
    Join a gym using its access code
    
    - Verifies gym code
    - Associates user with gym
    - User can then book slots and access gym features
    """
    gym = await session.get(models.Gym, gym_id)
    
    if not gym:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Gym not found"
        )
    
    # Verify gym code
    if gym.gym_code != gym_code:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid gym code"
        )
    
    # Check if user already belongs to a gym
    if current_user.gym_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You are already a member of another gym. Please leave that gym first."
        )
    
    # Join the gym
    current_user.gym_id = gym_id
    current_user.membership_status = "ACTIVE"
    
    await session.commit()
    await session.refresh(current_user)
    
    return {
        "message": f"Successfully joined {gym.name}",
        "gym_id": str(gym_id),
        "gym_name": gym.name
    }


@router.post("/{gym_id}/leave")
async def leave_gym(
    gym_id: uuid.UUID,
    session: AsyncSession = Depends(get_async_session),
    current_user: models.User = Depends(current_active_user)
):
    """
    Leave current gym
    
    - Removes gym association
    - Sets membership status to EXPIRED
    - User can then join another gym
    """
    if current_user.gym_id != gym_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You are not a member of this gym"
        )
    
    gym_name = (await session.get(models.Gym, gym_id)).name if await session.get(models.Gym, gym_id) else "gym"
    
    # Leave gym
    current_user.gym_id = None
    current_user.membership_status = "EXPIRED"
    
    await session.commit()
    
    return {
        "message": f"Successfully left {gym_name}",
        "former_gym_id": str(gym_id)
    }
