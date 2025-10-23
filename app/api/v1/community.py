# app/api/v1/community.py
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, desc, text, or_
from typing import List, Optional
from datetime import datetime, timedelta
import math

from app.core.database import get_async_session
from app.core.auth import current_active_user
from app.models import User, Gym, GymBooking, WorkoutLog
from app.schemas import (
    GymBooking as GymBookingSchema,
    GymBookingCreate,
    LeaderboardResponse,
    LeaderboardEntry,
    MessageResponse,
    JoinByCodeRequest
)

router = APIRouter()


@router.post("/bookings/", response_model=GymBookingSchema)
async def create_booking(
    booking_data: GymBookingCreate,
    session: AsyncSession = Depends(get_async_session),
    current_user: User = Depends(current_active_user)
):
    """Create a new gym booking with capacity check"""
    
    # Verify gym exists
    gym_query = select(Gym).where(Gym.id == booking_data.gym_id)
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
    overlap_query = select(GymBooking).where(
        and_(
            GymBooking.user_id == current_user.id,
            GymBooking.status == "active",
            or_(
                and_(
                    GymBooking.start_time <= booking_data.start_time,
                    GymBooking.end_time > booking_data.start_time
                ),
                and_(
                    GymBooking.start_time < booking_data.end_time,
                    GymBooking.end_time >= booking_data.end_time
                ),
                and_(
                    GymBooking.start_time >= booking_data.start_time,
                    GymBooking.end_time <= booking_data.end_time
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
    capacity_query = select(func.count(GymBooking.id)).where(
        and_(
            GymBooking.gym_id == booking_data.gym_id,
            GymBooking.status == "active",
            or_(
                and_(
                    GymBooking.start_time <= booking_data.start_time,
                    GymBooking.end_time > booking_data.start_time
                ),
                and_(
                    GymBooking.start_time < booking_data.end_time,
                    GymBooking.end_time >= booking_data.end_time
                ),
                and_(
                    GymBooking.start_time >= booking_data.start_time,
                    GymBooking.end_time <= booking_data.end_time
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
    booking = GymBooking(
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


@router.get("/bookings/", response_model=List[GymBookingSchema])
async def list_user_bookings(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    session: AsyncSession = Depends(get_async_session),
    current_user: User = Depends(current_active_user)
):
    """List current user's bookings"""
    query = select(GymBooking).where(
        GymBooking.user_id == current_user.id
    ).order_by(desc(GymBooking.start_time)).offset(skip).limit(limit)
    
    result = await session.execute(query)
    bookings = result.scalars().all()
    return bookings


@router.delete("/bookings/{booking_id}", response_model=MessageResponse)
async def cancel_booking(
    booking_id: int,
    session: AsyncSession = Depends(get_async_session),
    current_user: User = Depends(current_active_user)
):
    """Cancel a user's booking"""
    booking_query = select(GymBooking).where(
        and_(
            GymBooking.id == booking_id,
            GymBooking.user_id == current_user.id
        )
    )
    
    booking_result = await session.execute(booking_query)
    booking = booking_result.scalar_one_or_none()
    
    if not booking:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Booking not found"
        )
    
    if booking.status != "active":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Booking is already cancelled or completed"
        )
    
    booking.status = "cancelled"
    booking.cancelled_at = datetime.utcnow()
    
    await session.commit()
    
    return MessageResponse(
        message="Booking cancelled successfully",
        success=True
    )


@router.get("/leaderboard/{gym_id}", response_model=LeaderboardResponse)
async def get_gym_leaderboard(
    gym_id: int,
    limit: int = Query(10, ge=1, le=50),
    session: AsyncSession = Depends(get_async_session),
    current_user: User = Depends(current_active_user)
):
    """Get leaderboard for a specific gym"""
    
    # Verify gym exists
    gym_query = select(Gym).where(Gym.id == gym_id)
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
        
        entry = LeaderboardEntry(
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
    member_count_query = select(func.count(User.id)).where(User.gym_id == gym_id)
    member_count_result = await session.execute(member_count_query)
    total_members = member_count_result.scalar() or 0
    
    return LeaderboardResponse(
        gym_id=gym.id,
        gym_name=gym.name,
        gym_address=gym.address, # 👈 ADD ADDRESS HERE
        leaderboard=leaderboard_entries,
        total_members=total_members
    )


@router.get("/leaderboard/my-gym", response_model=Optional[LeaderboardResponse])
async def get_my_gym_leaderboard(
    limit: int = Query(10, ge=1, le=50),
    session: AsyncSession = Depends(get_async_session),
    current_user: User = Depends(current_active_user)
):
    """Get leaderboard for current user's gym"""
    if not current_user.gym_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You are not affiliated with any gym"
        )
    
    return await get_gym_leaderboard(current_user.gym_id, limit, session, current_user)

@router.post("/join-by-code", response_model=MessageResponse)
async def join_gym_by_code(
    request_data: JoinByCodeRequest,
    session: AsyncSession = Depends(get_async_session),
    current_user: User = Depends(current_active_user)
):
    """Join a gym using its unique code."""
    gym_code = request_data.gym_code.strip().upper() # Standardize code format

    if not gym_code:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Gym code cannot be empty."
        )

    # Find the gym by code (case-insensitive search if needed, adjust DB collation or use func.upper)
    gym_query = select(Gym).where(func.upper(Gym.gym_code) == gym_code)
    gym_result = await session.execute(gym_query)
    gym = gym_result.scalar_one_or_none()

    if not gym:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Invalid gym code."
        )

    # Check if user is already in this gym
    if current_user.gym_id == gym.id:
        return MessageResponse(
            message=f"You are already a member of {gym.name}.",
            success=True # Or False depending on how you want to handle this
        )

    # Update user's gym affiliation
    current_user.gym_id = gym.id
    current_user.last_gym_change = datetime.utcnow()

    await session.commit()

    return MessageResponse(
        message=f"Successfully joined {gym.name}!",
        success=True
    )