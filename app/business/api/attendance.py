# app/business/api/attendance.py
"""Attendance tracking and QR code management endpoints"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_
from typing import List, Optional
from datetime import datetime, timedelta, date
import uuid

from app.core.database import get_async_session
from app.core.business_auth import get_current_gym_owner
from app.business.services.qr_service import qr_service
from app import models, schemas

router = APIRouter()


@router.post("/qr/generate/{user_id}", response_model=schemas.QRCodeResponse)
async def generate_qr_for_member(
    user_id: uuid.UUID,
    current_owner: models.GymOwner = Depends(get_current_gym_owner),
    session: AsyncSession = Depends(get_async_session)
):
    """
    Generate or regenerate QR code for a gym member
    
    - QR code is valid for 15 days
    - Contains encrypted user_id and gym_id
    - Can be scanned for check-in/check-out
    """
    # Verify user exists and belongs to this gym
    user = await session.get(models.User, user_id)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Member not found"
        )
    
    if user.gym_id != current_owner.gym_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This member doesn't belong to your gym"
        )
    
    # Generate or update QR code
    qr_code = await qr_service.get_or_create_qr_code(
        session=session,
        user_id=user_id,
        gym_id=current_owner.gym_id
    )
    
    # Update user's qr_code_id reference
    user.qr_code_id = qr_code.id
    await session.commit()
    
    return qr_code


@router.get("/qr/{user_id}", response_model=schemas.QRCodeResponse)
async def get_member_qr_code(
    user_id: uuid.UUID,
    current_owner: models.GymOwner = Depends(get_current_gym_owner),
    session: AsyncSession = Depends(get_async_session)
):
    """
    Get existing QR code for a member
    
    Returns the QR code image in base64 format
    """
    user = await session.get(models.User, user_id)
    
    if not user or user.gym_id != current_owner.gym_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Member not found"
        )
    
    # Get QR code
    result = await session.execute(
        select(models.UserQRCode)
        .where(models.UserQRCode.user_id == user_id)
        .where(models.UserQRCode.gym_id == current_owner.gym_id)
    )
    qr_code = result.scalar_one_or_none()
    
    if not qr_code:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="QR code not found for this member. Generate one first."
        )
    
    return qr_code


@router.post("/check-in", response_model=schemas.AttendanceRecord)
async def check_in_member(
    check_in_data: schemas.CheckInRequest,
    current_owner: models.GymOwner = Depends(get_current_gym_owner),
    session: AsyncSession = Depends(get_async_session)
):
    """
    Check in a member using QR code scan
    
    - Validates QR code
    - Creates attendance record
    - Checks for existing active check-in
    """
    # Validate QR code
    validation = qr_service.validate_qr_code(check_in_data.qr_code_data)
    
    if not validation["valid"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid QR code: {validation.get('error', 'Unknown error')}"
        )
    
    user_id = validation["user_id"]
    gym_id = validation["gym_id"]
    
    # Verify gym matches
    if gym_id != current_owner.gym_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="QR code belongs to a different gym"
        )
    
    # Check if user already has an active check-in (no check-out)
    existing_check_in = await session.execute(
        select(models.GymAttendance)
        .where(models.GymAttendance.user_id == user_id)
        .where(models.GymAttendance.gym_id == gym_id)
        .where(models.GymAttendance.check_out_time.is_(None))
        .order_by(models.GymAttendance.check_in_time.desc())
        .limit(1)
    )
    active_check_in = existing_check_in.scalar_one_or_none()
    
    if active_check_in:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Member is already checked in since {active_check_in.check_in_time.isoformat()}"
        )
    
    # Create new attendance record
    attendance = models.GymAttendance(
        user_id=user_id,
        gym_id=gym_id,
        check_in_time=datetime.utcnow(),
        qr_code_used=check_in_data.qr_code_data
    )
    
    session.add(attendance)
    await session.commit()
    await session.refresh(attendance)
    
    return attendance


@router.post("/check-out", response_model=schemas.AttendanceRecord)
async def check_out_member(
    check_out_data: schemas.CheckOutRequest,
    current_owner: models.GymOwner = Depends(get_current_gym_owner),
    session: AsyncSession = Depends(get_async_session)
):
    """
    Check out a member
    
    - Requires attendance_id from the check-in record
    - Calculates duration
    - Updates attendance record
    """
    attendance = await session.get(models.GymAttendance, check_out_data.attendance_id)
    
    if not attendance:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Attendance record not found"
        )
    
    if attendance.gym_id != current_owner.gym_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This attendance record belongs to a different gym"
        )
    
    if attendance.check_out_time:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Member is already checked out"
        )
    
    # Set check-out time and calculate duration
    check_out_time = datetime.utcnow()
    duration = (check_out_time - attendance.check_in_time).total_seconds() / 60  # minutes
    
    attendance.check_out_time = check_out_time
    attendance.duration_minutes = int(duration)
    
    await session.commit()
    await session.refresh(attendance)
    
    return attendance


@router.get("/today", response_model=List[schemas.AttendanceRecord])
async def get_todays_attendance(
    current_owner: models.GymOwner = Depends(get_current_gym_owner),
    session: AsyncSession = Depends(get_async_session)
):
    """
    Get all attendance records for today
    
    Shows all check-ins/check-outs from midnight to now
    """
    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    
    result = await session.execute(
        select(models.GymAttendance)
        .where(models.GymAttendance.gym_id == current_owner.gym_id)
        .where(models.GymAttendance.check_in_time >= today_start)
        .order_by(models.GymAttendance.check_in_time.desc())
    )
    
    attendance_records = result.scalars().all()
    
    return attendance_records


@router.get("/current-count", response_model=schemas.LiveOccupancyResponse)
async def get_current_occupancy(
    current_owner: models.GymOwner = Depends(get_current_gym_owner),
    session: AsyncSession = Depends(get_async_session)
):
    """
    Get real-time gym occupancy
    
    Shows:
    - Current number of people in gym (checked in, not checked out)
    - Gym capacity
    - Occupancy percentage
    - List of currently checked-in users
    """
    gym = await session.get(models.Gym, current_owner.gym_id)
    
    if not gym:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Gym not found"
        )
    
    # Count active check-ins (no check-out time)
    current_count = await session.scalar(
        select(func.count(models.GymAttendance.id))
        .where(models.GymAttendance.gym_id == current_owner.gym_id)
        .where(models.GymAttendance.check_out_time.is_(None))
    )
    
    # Get list of currently checked-in users
    result = await session.execute(
        select(models.GymAttendance, models.User)
        .join(models.User, models.GymAttendance.user_id == models.User.id)
        .where(models.GymAttendance.gym_id == current_owner.gym_id)
        .where(models.GymAttendance.check_out_time.is_(None))
        .order_by(models.GymAttendance.check_in_time.desc())
    )
    
    checked_in_users = [
        {
            "attendance_id": str(attendance.id),
            "user_id": str(user.id),
            "username": user.username,
            "full_name": user.full_name,
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


@router.get("/history")
async def get_attendance_history(
    start_date: Optional[date] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[date] = Query(None, description="End date (YYYY-MM-DD)"),
    user_id: Optional[uuid.UUID] = Query(None, description="Filter by specific user"),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    current_owner: models.GymOwner = Depends(get_current_gym_owner),
    session: AsyncSession = Depends(get_async_session)
):
    """
    Get attendance history with filters
    
    Filters:
    - Date range
    - Specific user
    - Pagination
    """
    # Default to last 30 days if no dates provided
    if not end_date:
        end_date = date.today()
    if not start_date:
        start_date = end_date - timedelta(days=30)
    
    query = select(models.GymAttendance).where(
        models.GymAttendance.gym_id == current_owner.gym_id
    )
    
    # Date filter
    start_datetime = datetime.combine(start_date, datetime.min.time())
    end_datetime = datetime.combine(end_date, datetime.max.time())
    
    query = query.where(
        and_(
            models.GymAttendance.check_in_time >= start_datetime,
            models.GymAttendance.check_in_time <= end_datetime
        )
    )
    
    # User filter
    if user_id:
        query = query.where(models.GymAttendance.user_id == user_id)
    
    # Pagination
    query = query.order_by(models.GymAttendance.check_in_time.desc()).offset(skip).limit(limit)
    
    result = await session.execute(query)
    records = result.scalars().all()
    
    return {
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "total_records": len(records),
        "records": [
            {
                "id": str(r.id),
                "user_id": str(r.user_id),
                "check_in_time": r.check_in_time.isoformat(),
                "check_out_time": r.check_out_time.isoformat() if r.check_out_time else None,
                "duration_minutes": r.duration_minutes
            }
            for r in records
        ]
    }


@router.get("/user/{user_id}")
async def get_user_attendance_history(
    user_id: uuid.UUID,
    days: int = Query(30, ge=1, le=365),
    current_owner: models.GymOwner = Depends(get_current_gym_owner),
    session: AsyncSession = Depends(get_async_session)
):
    """
    Get attendance history for a specific member
    
    Shows all check-ins for the last N days
    """
    user = await session.get(models.User, user_id)
    
    if not user or user.gym_id != current_owner.gym_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Member not found"
        )
    
    since_date = datetime.utcnow() - timedelta(days=days)
    
    result = await session.execute(
        select(models.GymAttendance)
        .where(models.GymAttendance.user_id == user_id)
        .where(models.GymAttendance.gym_id == current_owner.gym_id)
        .where(models.GymAttendance.check_in_time >= since_date)
        .order_by(models.GymAttendance.check_in_time.desc())
    )
    
    records = result.scalars().all()
    
    # Calculate stats
    total_visits = len(records)
    total_duration = sum(r.duration_minutes or 0 for r in records)
    avg_duration = total_duration / total_visits if total_visits > 0 else 0
    
    return {
        "user_id": str(user_id),
        "username": user.username,
        "period_days": days,
        "total_visits": total_visits,
        "total_duration_minutes": total_duration,
        "avg_duration_minutes": round(avg_duration, 1),
        "visits": [
            {
                "check_in": r.check_in_time.isoformat(),
                "check_out": r.check_out_time.isoformat() if r.check_out_time else None,
                "duration_minutes": r.duration_minutes
            }
            for r in records
        ]
    }
