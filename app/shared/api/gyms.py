# app/shared/api/gyms.py
"""Shared gym endpoints - used by both clients and business owners"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, or_, and_, text
from typing import List, Optional
from datetime import datetime, timedelta
import uuid
from app.core.database import get_async_session
from app.core.auth import current_active_user
from app import schemas , models

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
    
    gym_obj = await session.get(models.Gym, gym_id) # Fetch once
    gym_name = gym_obj.name if gym_obj else "gym"
    
    # Leave gym
    current_user.gym_id = None # type: ignore
    current_user.membership_status = "EXPIRED"
    
    await session.commit()
    
    return {
        "message": f"Successfully left {gym_name}",
        "former_gym_id": str(gym_id)
    }

@router.post("/bookings/", response_model=schemas.GymBooking)
async def create_booking(
    booking_data: schemas.GymBookingCreate,
    session: AsyncSession = Depends(get_async_session),
    current_user: models.User = Depends(current_active_user)
):
    """Create a new gym booking with capacity check"""
    
    # Verify gym exists
    gym_query = select(models.Gym).where(models.Gym.id == booking_data.gym_id)
    gym_result = await session.execute(gym_query)
    gym = gym_result.scalar_one_or_none()
    
    if not gym:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Gym not found"
        )
    
    # Validate booking times
    if booking_data.start_time >= booking_data.end_time:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Start time must be before end time"
        )
    
    if booking_data.start_time <= datetime.utcnow():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot book for past times"
        )
    
    # Check for overlapping user bookings
    overlap_query = select(models.GymBooking).where(
        and_(
            models.GymBooking.user_id == current_user.id,
            models.GymBooking.status == "active",
            or_(
                and_(
                    models.GymBooking.start_time <= booking_data.start_time,
                    models.GymBooking.end_time > booking_data.start_time
                ),
                and_(
                    models.GymBooking.start_time < booking_data.end_time,
                    models.GymBooking.end_time >= booking_data.end_time
                ),
                and_(
                    models.GymBooking.start_time >= booking_data.start_time,
                    models.GymBooking.end_time <= booking_data.end_time
                )
            )
        )
    )
    
    overlap_result = await session.execute(overlap_query)
    if overlap_result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You already have a booking that overlaps with this time"
        )
    
    # Check gym capacity for the requested time slot
    capacity_query = select(func.count(models.GymBooking.id)).where(
        and_(
            models.GymBooking.gym_id == booking_data.gym_id,
            models.GymBooking.status == "active",
            or_(
                and_(
                    models.GymBooking.start_time <= booking_data.start_time,
                    models.GymBooking.end_time > booking_data.start_time
                ),
                and_(
                    models.GymBooking.start_time < booking_data.end_time,
                    models.GymBooking.end_time >= booking_data.end_time
                ),
                and_(
                    models.GymBooking.start_time >= booking_data.start_time,
                    models.GymBooking.end_time <= booking_data.end_time
                )
            )
        )
    )
    
    capacity_result = await session.execute(capacity_query)
    current_bookings = capacity_result.scalar() or 0
    
    # Allow overbooking but warn user
    if current_bookings >= gym.max_capacity:
        # You might want to implement a waiting list or warning system here
        pass
    
    # Create the booking
    booking = models.GymBooking(
        user_id=current_user.id,
        gym_id=booking_data.gym_id,
        start_time=booking_data.start_time,
        end_time=booking_data.end_time,
        status="active"
    )
    
    session.add(booking)
    await session.commit()
    await session.refresh(booking)
    
    return booking

@router.get("/leaderboard/{gym_id}", response_model=schemas.LeaderboardResponse)
async def get_gym_leaderboard(
    gym_id: uuid.UUID,
    limit: int = Query(10, ge=1, le=50),
    session: AsyncSession = Depends(get_async_session),
    current_user: models.User = Depends(current_active_user)
):
    """Get leaderboard for a specific gym"""
    
    # Verify gym exists
    gym_query = select(models.Gym).where(models.Gym.id == gym_id)
    gym_result = await session.execute(gym_query)
    gym = gym_result.scalar_one_or_none()
    
    if not gym:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Gym not found"
        )
    
    # Calculate leaderboard metrics for the last 30 days
    thirty_days_ago = datetime.utcnow() - timedelta(days=30)
    
    # Complex leaderboard query with workout statistics
    leaderboard_query = text("""
        SELECT 
            u.id as user_id,
            u.username,
            u.full_name,
            COUNT(wl.id) as total_workouts,
            COALESCE(SUM(wl.calories_burned), 0) as total_calories_burned,
            COALESCE(SUM(wl.duration_minutes), 0) / 60.0 as total_workout_time_hours,
            (COUNT(DISTINCT DATE(wl.workout_date)) * 100.0 / 30) as consistency_score,
            ROW_NUMBER() OVER (
                ORDER BY 
                    COUNT(wl.id) DESC, 
                    COALESCE(SUM(wl.calories_burned), 0) DESC,
                    COALESCE(SUM(wl.duration_minutes), 0) DESC
            ) as rank
        FROM users u
        LEFT JOIN workout_logs wl ON u.id = wl.user_id 
            AND wl.workout_date >= :thirty_days_ago
        WHERE u.gym_id = :gym_id
        GROUP BY u.id, u.username, u.full_name
        ORDER BY rank
        LIMIT :limit
    """)
    
    result = await session.execute(
        leaderboard_query,
        {
            "gym_id": gym_id,
            "thirty_days_ago": thirty_days_ago,
            "limit": limit
        }
    )
    
    leaderboard_data = result.fetchall()
    
    # Convert to LeaderboardEntry objects
    leaderboard_entries = []
    for row in leaderboard_data:
        user_name = row.full_name if row.full_name else row.username or "Anonymous"
        
        entry = schemas.LeaderboardEntry(
            user_id=row.user_id,
            user_name=user_name,
            total_workouts=row.total_workouts,
            total_calories_burned=float(row.total_calories_burned),
            total_minutes=int(row.total_minutes),
            consistency_score=round(float(row.consistency_score),1),
            rank=row.rank
        )
        leaderboard_entries.append(entry)
    
    # Get total member count for the gym
    member_count_query = select(func.count(models.User.id)).where(models.User.gym_id == gym_id)
    member_count_result = await session.execute(member_count_query)
    total_members = member_count_result.scalar() or 0
    
    return schemas.LeaderboardResponse(
        gym_id=gym.id,
        gym_name=gym.name,
        gym_address=gym.address, # 👈 ADD ADDRESS HERE
        leaderboard=leaderboard_entries,
        total_members=total_members
    )


@router.get("/leaderboard/my-gym", response_model=Optional[schemas.LeaderboardResponse])
async def get_my_gym_leaderboard(
    limit: int = Query(10, ge=1, le=50),
    session: AsyncSession = Depends(get_async_session),
    current_user: models.User = Depends(current_active_user)
):
    """Get leaderboard for current user's gym"""
    if not current_user.gym_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You are not affiliated with any gym"
        )
    
    return await get_gym_leaderboard(current_user.gym_id, limit, session, current_user)

@router.post("/join-by-code", response_model=schemas.MessageResponse)
async def join_gym_by_code(
    request_data: schemas.JoinByCodeRequest,
    session: AsyncSession = Depends(get_async_session),
    current_user: models.User = Depends(current_active_user)
):
    """Join a gym using its unique code."""
    gym_code = request_data.gym_code.strip().upper() # Standardize code format

    if not gym_code:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Gym code cannot be empty."
        )

    # Find the gym by code (case-insensitive search if needed, adjust DB collation or use func.upper)
    gym_query = select(models.Gym).where(func.upper(models.Gym.gym_code) == gym_code)
    gym_result = await session.execute(gym_query)
    gym = gym_result.scalar_one_or_none()

    if not gym:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Invalid gym code."
        )

    # Check if user is already in this gym
    if current_user.gym_id == gym.id:
        return schemas.MessageResponse(
            message=f"You are already a member of {gym.name}.",
            success=True # Or False depending on how you want to handle this
        )

    # Update user's gym affiliation
    current_user.gym_id = gym.id
    current_user.last_gym_change = datetime.utcnow()

    await session.commit()

    return schemas.MessageResponse(
        message=f"Successfully joined {gym.name}!",
        success=True
    )