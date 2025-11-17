# app/business/api/members.py
"""Member management endpoints for gym owners"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_
from typing import List, Optional
from datetime import datetime, timedelta
import uuid

from app.core.database import get_async_session
from app.core.business_auth import get_current_gym_owner
from app import models, schemas

router = APIRouter()


@router.get("/", response_model=List[schemas.UserRead])
async def list_gym_members(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    search: Optional[str] = Query(None, description="Search by username, name, or email"),
    membership_status: Optional[str] = Query(None, description="Filter by membership status"),
    current_owner: models.GymOwner = Depends(get_current_gym_owner),
    session: AsyncSession = Depends(get_async_session)
):
    """
    Get list of all members in the gym

    - **skip**: Number of records to skip (pagination)
    - **limit**: Maximum records to return
    - **search**: Search query for username/name/email
    - **membership_status**: Filter by ACTIVE, EXPIRED, SUSPENDED
    """
    query = select(models.User).where(models.User.gym_id == current_owner.gym_id)

    # Apply search filter
    if search:
        search_filter = or_(
            models.User.username.ilike(f"%{search}%"),
            models.User.full_name.ilike(f"%{search}%"),
            models.User.email.ilike(f"%{search}%")
        )
        query = query.where(search_filter)

    # Apply membership status filter
    if membership_status:
        query = query.where(models.User.membership_status == membership_status)

    # Apply pagination
    query = query.offset(skip).limit(limit).order_by(models.User.username)

    result = await session.execute(query)
    members = result.scalars().all()

    return members


@router.get("/stats")
async def get_member_statistics(
    current_owner: models.GymOwner = Depends(get_current_gym_owner),
    session: AsyncSession = Depends(get_async_session)
):
    """
    Get statistical overview of gym members

    Returns:
    - total_members: Total number of members in the gym
    - active_members: Members with active membership
    - new_this_month: New members joined this month
    - growth_rate: Month-over-month growth percentage
    """
    gym_id = current_owner.gym_id

    # Get total members count
    total_query = select(func.count(models.User.id)).where(models.User.gym_id == gym_id)
    total_result = await session.execute(total_query)
    total_members = total_result.scalar() or 0

    # Get active members count
    active_query = select(func.count(models.User.id)).where(
        and_(
            models.User.gym_id == gym_id,
            models.User.membership_status == "ACTIVE"
        )
    )
    active_result = await session.execute(active_query)
    active_members = active_result.scalar() or 0

    # Get new members this month
    now = datetime.utcnow()
    first_day_of_month = datetime(now.year, now.month, 1)

    new_this_month_query = select(func.count(models.User.id)).where(
        and_(
            models.User.gym_id == gym_id,
            models.User.created_at >= first_day_of_month
        )
    )
    new_this_month_result = await session.execute(new_this_month_query)
    new_this_month = new_this_month_result.scalar() or 0

    # Calculate growth rate (simplified - comparing this month vs last month)
    # Get first day of last month
    if now.month == 1:
        first_day_last_month = datetime(now.year - 1, 12, 1)
    else:
        first_day_last_month = datetime(now.year, now.month - 1, 1)

    last_month_query = select(func.count(models.User.id)).where(
        and_(
            models.User.gym_id == gym_id,
            models.User.created_at >= first_day_last_month,
            models.User.created_at < first_day_of_month
        )
    )
    last_month_result = await session.execute(last_month_query)
    last_month_count = last_month_result.scalar() or 0

    # Calculate growth rate percentage
    if last_month_count > 0:
        growth_rate = ((new_this_month - last_month_count) / last_month_count) * 100
    else:
        growth_rate = 100.0 if new_this_month > 0 else 0.0

    return {
        "total_members": total_members,
        "active_members": active_members,
        "new_this_month": new_this_month,
        "growth_rate": round(growth_rate, 1)
    }


@router.get("/{user_id}", response_model=schemas.UserRead)
async def get_member_details(
    user_id: uuid.UUID,
    current_owner: models.GymOwner = Depends(get_current_gym_owner),
    session: AsyncSession = Depends(get_async_session)
):
    """
    Get detailed information about a specific member
    
    Returns full user profile including:
    - Personal info
    - Fitness goals
    - Gamification stats
    - Membership status
    """
    user = await session.get(models.User, user_id)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Member not found"
        )
    
    # Verify user belongs to this gym
    if user.gym_id != current_owner.gym_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This member belongs to a different gym"
        )
    
    return user


@router.patch("/{user_id}/membership-status")
async def update_member_status(
    user_id: uuid.UUID,
    new_status: str = Query(..., description="ACTIVE, EXPIRED, or SUSPENDED"),
    current_owner: models.GymOwner = Depends(get_current_gym_owner),
    session: AsyncSession = Depends(get_async_session)
):
    """
    Update a member's membership status
    
    - **ACTIVE**: Member can access gym
    - **EXPIRED**: Membership has expired
    - **SUSPENDED**: Temporarily suspended (e.g., non-payment)
    """
    if new_status not in ["ACTIVE", "EXPIRED", "SUSPENDED"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Status must be ACTIVE, EXPIRED, or SUSPENDED"
        )
    
    user = await session.get(models.User, user_id)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Member not found"
        )
    
    if user.gym_id != current_owner.gym_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This member belongs to a different gym"
        )
    
    user.membership_status = new_status
    await session.commit()
    
    return {
        "message": f"Membership status updated to {new_status}",
        "user_id": str(user_id),
        "new_status": new_status
    }


@router.delete("/{user_id}")
async def remove_member_from_gym(
    user_id: uuid.UUID,
    current_owner: models.GymOwner = Depends(get_current_gym_owner),
    session: AsyncSession = Depends(get_async_session)
):
    """
    Remove a member from the gym
    
    This sets their gym_id to NULL and deactivates their QR code.
    Does NOT delete the user account completely.
    """
    user = await session.get(models.User, user_id)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Member not found"
        )
    
    if user.gym_id != current_owner.gym_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This member belongs to a different gym"
        )
    
    # Remove from gym
    user.gym_id = None
    user.membership_status = "EXPIRED"
    
    # Deactivate QR code if exists
    if user.qr_codes:
        for qr_code in user.qr_codes:
            qr_code.is_active = False

    await session.commit()
    
    return {
        "message": "Member removed from gym successfully",
        "user_id": str(user_id)
    }


@router.get("/{user_id}/performance", response_model=schemas.PerformanceAnalysis)
async def get_member_performance(
    user_id: uuid.UUID,
    current_owner: models.GymOwner = Depends(get_current_gym_owner),
    session: AsyncSession = Depends(get_async_session)
):
    """
    Get AI-analyzed performance metrics for a member
    
    Shows:
    - Total workouts
    - Average workout duration
    - Consistency score
    - Performance trend (IMPROVING/STABLE/DECLINING)
    - Weak areas needing improvement
    - AI-generated suggestions
    """
    user = await session.get(models.User, user_id)
    
    if not user or user.gym_id != current_owner.gym_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Member not found or doesn't belong to your gym"
        )
    
    # Get most recent performance analysis
    result = await session.execute(
        select(models.MemberPerformance)
        .where(models.MemberPerformance.user_id == user_id)
        .order_by(models.MemberPerformance.analysis_date.desc())
        .limit(1)
    )
    performance = result.scalar_one_or_none()
    
    if not performance:
        # If no analysis exists, generate one on the fly
        # (In production, this should be done by a background task)
        from app.services.ai_services import analyze_member_performance
        
        performance_data = await analyze_member_performance(user_id, session)
        
        # Create performance record
        performance = models.MemberPerformance(
            user_id=user_id,
            gym_id=current_owner.gym_id,
            total_workouts=performance_data.get("total_workouts", 0),
            avg_workout_duration=performance_data.get("avg_duration", 0.0),
            consistency_score=performance_data.get("consistency_score", 0.0),
            performance_trend=performance_data.get("trend", "STABLE"),
            weak_areas=performance_data.get("weak_areas", []),
            suggestions=performance_data.get("suggestions", [])
        )
        
        session.add(performance)
        await session.commit()
        await session.refresh(performance)
    
    return performance


@router.get("/{user_id}/workout-history")
async def get_member_workout_history(
    user_id: uuid.UUID,
    days: int = Query(30, ge=1, le=365, description="Number of days to look back"),
    current_owner: models.GymOwner = Depends(get_current_gym_owner),
    session: AsyncSession = Depends(get_async_session)
):
    """
    Get member's workout history
    
    Returns workout logs for the specified time period
    """
    user = await session.get(models.User, user_id)
    
    if not user or user.gym_id != current_owner.gym_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Member not found"
        )
    
    since_date = datetime.utcnow() - timedelta(days=days)
    
    result = await session.execute(
        select(models.WorkoutLog)
        .where(models.WorkoutLog.user_id == user_id)
        .where(models.WorkoutLog.logged_at >= since_date)
        .order_by(models.WorkoutLog.logged_at.desc())
    )
    workouts = result.scalars().all()
    
    return {
        "user_id": str(user_id),
        "username": user.username,
        "period_days": days,
        "total_workouts": len(workouts),
        "workouts": [
            {
                "id": str(w.id),
                "name": w.name,
                "duration_minutes": w.duration_minutes,
                "calories_burned": w.calories_burned,
                "logged_at": w.logged_at.isoformat()
            }
            for w in workouts
        ]
    }
