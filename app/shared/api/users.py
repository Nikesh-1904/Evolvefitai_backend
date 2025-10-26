# app/shared/api/users.py
"""Shared user endpoints - profile, QR code, preferences"""
import uuid
import datetime
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc, and_
from typing import List

from app.core.database import get_async_session
from app.core.auth import current_active_user
from app import models, schemas

router = APIRouter()


@router.get("/me/qr-code", response_model=schemas.QRCodeResponse)
async def get_my_qr_code(
    current_user: models.User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session)
):
    """
    Get current user's QR code for gym check-in
    
    Returns:
    - QR code image (base64)
    - QR code data (encrypted)
    - Validity information
    
    If QR code doesn't exist or is expired, user should contact gym owner
    """
    if not current_user.gym_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You are not a member of any gym. Join a gym first to get a QR code."
        )
    
    # Get user's QR code
    result = await session.execute(
        select(models.UserQRCode)
        .where(models.UserQRCode.user_id == current_user.id)
        .where(models.UserQRCode.gym_id == current_user.gym_id)
        .where(models.UserQRCode.is_active)
    )
    qr_code = result.scalar_one_or_none()
    
    if not qr_code:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="QR code not found. Please contact your gym owner to generate one for you."
        )
    
    # Check if expired
    from datetime import datetime
    if qr_code.expires_at and qr_code.expires_at < datetime.utcnow():
        raise HTTPException(
            status_code=status.HTTP_410_GONE,
            detail="Your QR code has expired. Please contact your gym owner to regenerate it."
        )
    
    return qr_code


@router.get("/me/preferences", response_model=schemas.NotificationPreferences)
async def get_notification_preferences(
    current_user: models.User = Depends(current_active_user)
):
    """
    Get current user's notification preferences
    """
    preferences = current_user.notification_preferences or {
        "email_enabled": True,
        "app_enabled": True,
        "fee_reminder_days": [7, 3, 1]
    }
    
    return schemas.NotificationPreferences(**preferences)


@router.patch("/me/preferences", response_model=schemas.NotificationPreferences)
async def update_notification_preferences(
    preferences: schemas.NotificationPreferences,
    current_user: models.User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session)
):
    """
    Update notification preferences
    
    - email_enabled: Receive fee reminders via email
    - app_enabled: Receive in-app notifications
    - fee_reminder_days: Days before due date to send reminders (e.g., [7, 3, 1])
    """
    # Validate fee_reminder_days
    if preferences.fee_reminder_days:
        if not all(1 <= day <= 30 for day in preferences.fee_reminder_days):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Fee reminder days must be between 1 and 30"
            )
    
    # Update preferences
    current_user.notification_preferences = {
        "email_enabled": preferences.email_enabled,
        "app_enabled": preferences.app_enabled,
        "fee_reminder_days": preferences.fee_reminder_days
    }
    
    await session.commit()
    await session.refresh(current_user)
    
    return schemas.NotificationPreferences(**current_user.notification_preferences)


@router.get("/me/membership-info")
async def get_membership_info(
    current_user: models.User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session)
):
    """
    Get current user's membership information
    
    Shows:
    - Gym details
    - Membership status
    - Membership expiry
    - Latest fee information
    """
    if not current_user.gym_id:
        return {
            "is_member": False,
            "message": "You are not a member of any gym"
        }
    
    # Get gym details
    gym = await session.get(models.Gym, current_user.gym_id)
    
    # Get latest fee record
    latest_fee_result = await session.execute(
        select(models.MembershipFee)
        .where(models.MembershipFee.user_id == current_user.id)
        .where(models.MembershipFee.gym_id == current_user.gym_id)
        .order_by(models.MembershipFee.due_date.desc())
        .limit(1)
    )
    latest_fee = latest_fee_result.scalar_one_or_none()
    
    # Get total pending fees
    pending_fees_result = await session.execute(
        select(func.count(models.MembershipFee.id), func.sum(models.MembershipFee.amount))
        .where(models.MembershipFee.user_id == current_user.id)
        .where(models.MembershipFee.status == "PENDING")
    )
    pending_count, pending_amount = pending_fees_result.one()
    
    return {
        "is_member": True,
        "gym": {
            "id": str(gym.id),
            "name": gym.name,
            "city": gym.city,
            "monthly_fee": gym.monthly_fee
        } if gym else None,
        "membership_status": current_user.membership_status,
        "membership_expiry": current_user.membership_expiry.isoformat() if current_user.membership_expiry else None,
        "latest_fee": {
            "amount": latest_fee.amount,
            "due_date": latest_fee.due_date.isoformat(),
            "status": latest_fee.status,
            "paid_date": latest_fee.paid_date.isoformat() if latest_fee.paid_date else None
        } if latest_fee else None,
        "pending_fees": {
            "count": pending_count or 0,
            "total_amount": pending_amount or 0.0
        }
    }


@router.get("/me/notifications")
async def get_my_notifications(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    unread_only: bool = Query(False, description="Show only unread notifications"),
    current_user: models.User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session)
):
    """
    Get user's notifications
    
    Shows in-app notifications sent by gym owner
    """
    query = select(models.Notification).where(
        models.Notification.user_id == current_user.id
    )
    
    if unread_only:
        query = query.where(models.Notification.is_read)
    
    query = query.order_by(models.Notification.created_at.desc()).offset(skip).limit(limit)
    
    result = await session.execute(query)
    notifications = result.scalars().all()
    
    return {
        "total": len(notifications),
        "unread_count": sum(1 for n in notifications if not n.is_read),
        "notifications": [
            {
                "id": str(n.id),
                "type": n.notification_type,
                "title": n.title,
                "message": n.message,
                "is_read": n.is_read,
                "created_at": n.created_at.isoformat(),
                "read_at": n.read_at.isoformat() if n.read_at else None
            }
            for n in notifications
        ]
    }


@router.patch("/me/notifications/{notification_id}/read")
async def mark_notification_as_read(
    notification_id: uuid.UUID,
    current_user: models.User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session)
):
    """
    Mark a notification as read
    """
    notification = await session.get(models.Notification, notification_id)
    
    if not notification:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Notification not found"
        )
    
    if notification.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This notification doesn't belong to you"
        )
    
    notification.is_read = True
    notification.read_at = datetime.datetime.now(datetime.timezone.utc)
    
    await session.commit()
    
    return {"message": "Notification marked as read"}

@router.get("/bookings/", response_model=List[schemas.GymBooking])
async def list_user_bookings(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    session: AsyncSession = Depends(get_async_session),
    current_user: models.User = Depends(current_active_user)
):
    """List current user's bookings"""
    query = select(models.GymBooking).where(
        models.GymBooking.user_id == current_user.id
    ).order_by(desc(models.GymBooking.start_time)).offset(skip).limit(limit)
    
    result = await session.execute(query)
    bookings = result.scalars().all()
    return bookings


@router.delete("/bookings/{booking_id}", response_model=schemas.MessageResponse)
async def cancel_booking(
    booking_id: uuid.UUID,
    session: AsyncSession = Depends(get_async_session),
    current_user: models.User = Depends(current_active_user)
):
    """Cancel a user's booking"""
    booking_query = select(models.GymBooking).where(
        and_(
            models.GymBooking.id == booking_id,
            models.GymBooking.user_id == current_user.id
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
    booking.cancelled_at = datetime.datetime.now(datetime.timezone.utc)
    
    await session.commit()
    
    return schemas.MessageResponse(
        message="Booking cancelled successfully",
        success=True
    )
