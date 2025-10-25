# app/business/api/notifications.py
"""Notification management endpoints"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func ,and_
from typing import List, Optional
from datetime import datetime
import uuid

from app.core.database import get_async_session
from app.core.business_auth import get_current_gym_owner
from app.business.services.email_service import email_service
from app import models, schemas

router = APIRouter()


@router.post("/send", status_code=status.HTTP_201_CREATED)
async def send_notification(
    notification_data: schemas.NotificationCreate,
    current_owner: models.GymOwner = Depends(get_current_gym_owner),
    session: AsyncSession = Depends(get_async_session)
):
    """
    Send notification to specific members
    
    - **user_ids**: List of member UUIDs
    - **notification_type**: FEE_REMINDER, GENERAL, ANNOUNCEMENT
    - **title**: Notification title
    - **message**: Notification message
    - **send_email**: Send via email (default: false)
    - **send_app**: Create in-app notification (default: true)
    """
    gym = await session.get(models.Gym, current_owner.gym_id)
    
    sent_count = 0
    email_sent_count = 0
    errors = []
    
    for user_id in notification_data.user_ids:
        try:
            user = await session.get(models.User, user_id)
            
            if not user or user.gym_id != current_owner.gym_id:
                errors.append({
                    "user_id": str(user_id),
                    "error": "Member not found or doesn't belong to your gym"
                })
                continue
            
            # Create in-app notification if requested
            if notification_data.send_app:
                notification = models.Notification(
                    user_id=user_id,
                    gym_id=current_owner.gym_id,
                    notification_type=notification_data.notification_type,
                    title=notification_data.title,
                    message=notification_data.message,
                    sent_via_app=True,
                    sent_via_email=notification_data.send_email
                )
                session.add(notification)
                sent_count += 1
            
            # Send email if requested and user has email
            if notification_data.send_email and user.email:
                email_sent = await email_service.send_custom_notification(
                    to_emails=[user.email],
                    subject=notification_data.title,
                    message=notification_data.message,
                    gym_name=gym.name if gym else "Your Gym"
                )
                if email_sent > 0:
                    email_sent_count += 1
        
        except Exception as e:
            errors.append({
                "user_id": str(user_id),
                "error": str(e)
            })
    
    await session.commit()
    
    return {
        "message": "Notifications sent successfully",
        "app_notifications_sent": sent_count,
        "emails_sent": email_sent_count,
        "errors": errors
    }


@router.post("/broadcast")
async def broadcast_notification(
    title: str = Query(..., description="Notification title"),
    message: str = Query(..., description="Notification message"),
    send_email: bool = Query(False, description="Send via email"),
    current_owner: models.GymOwner = Depends(get_current_gym_owner),
    session: AsyncSession = Depends(get_async_session)
):
    """
    Broadcast notification to ALL active members of the gym
    
    Use this for gym-wide announcements
    """
    gym = await session.get(models.Gym, current_owner.gym_id)
    
    # Get all active members
    result = await session.execute(
        select(models.User)
        .where(models.User.gym_id == current_owner.gym_id)
        .where(models.User.membership_status == "ACTIVE")
    )
    members = result.scalars().all()
    
    if not members:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No active members found in this gym"
        )
    
    sent_count = 0
    email_addresses = []
    
    for member in members:
        # Create in-app notification
        notification = models.Notification(
            user_id=member.id,
            gym_id=current_owner.gym_id,
            notification_type="ANNOUNCEMENT",
            title=title,
            message=message,
            sent_via_app=True,
            sent_via_email=send_email
        )
        session.add(notification)
        sent_count += 1
        
        if send_email and member.email:
            email_addresses.append(member.email)
    
    await session.commit()
    
    # Send bulk emails if requested
    email_sent_count = 0
    if send_email and email_addresses:
        email_sent_count = await email_service.send_custom_notification(
            to_emails=email_addresses,
            subject=title,
            message=message,
            gym_name=gym.name if gym else "Your Gym"
        )
    
    return {
        "message": "Broadcast sent successfully",
        "total_members": len(members),
        "app_notifications_sent": sent_count,
        "emails_sent": email_sent_count
    }


@router.post("/fee-reminder/{fee_id}")
async def send_fee_reminder(
    fee_id: uuid.UUID,
    current_owner: models.GymOwner = Depends(get_current_gym_owner),
    session: AsyncSession = Depends(get_async_session)
):
    """
    Send manual fee reminder for a specific fee record
    
    Useful for immediate reminders outside of automated schedule
    """
    fee = await session.get(models.MembershipFee, fee_id)
    
    if not fee:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Fee record not found"
        )
    
    if fee.gym_id != current_owner.gym_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This fee belongs to a different gym"
        )
    
    if fee.status == "PAID":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot send reminder for already paid fee"
        )
    
    # Get user and gym details
    user = await session.get(models.User, fee.user_id)
    gym = await session.get(models.Gym, fee.gym_id)
    
    if not user or not user.email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Member not found or has no email address"
        )
    
    # Send email reminder
    email_sent = await email_service.send_fee_reminder(
        to_email=user.email,
        user_name=user.full_name or user.username or "Member",
        amount=fee.amount,
        due_date=fee.due_date,
        gym_name=gym.name if gym else "Your Gym"
    )
    
    if not email_sent:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to send email reminder"
        )
    
    # Create notification record
    notification = models.Notification(
        user_id=user.id,
        gym_id=current_owner.gym_id,
        notification_type="FEE_REMINDER",
        title="Fee Payment Reminder",
        message=f"Your gym fee of ₹{fee.amount} is due on {fee.due_date.strftime('%d %b %Y')}",
        sent_via_email=True,
        sent_via_app=True
    )
    session.add(notification)
    await session.commit()
    
    return {
        "message": "Fee reminder sent successfully",
        "fee_id": str(fee_id),
        "user_email": user.email
    }


@router.get("/history", response_model=List[schemas.NotificationRead])
async def get_notification_history(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    notification_type: Optional[str] = Query(None),
    current_owner: models.GymOwner = Depends(get_current_gym_owner),
    session: AsyncSession = Depends(get_async_session)
):
    """
    Get history of sent notifications
    
    Shows all notifications sent from this gym
    """
    query = select(models.Notification).where(
        models.Notification.gym_id == current_owner.gym_id
    )
    
    if notification_type:
        query = query.where(models.Notification.notification_type == notification_type)
    
    query = query.order_by(models.Notification.created_at.desc()).offset(skip).limit(limit)
    
    result = await session.execute(query)
    notifications = result.scalars().all()
    
    return notifications


@router.get("/stats")
async def get_notification_stats(
    current_owner: models.GymOwner = Depends(get_current_gym_owner),
    session: AsyncSession = Depends(get_async_session)
):
    """
    Get notification statistics
    
    Shows:
    - Total notifications sent
    - Read vs unread
    - Breakdown by type
    """
    # Total notifications
    total_count = await session.scalar(
        select(func.count(models.Notification.id))
        .where(models.Notification.gym_id == current_owner.gym_id)
    )
    
    # Read count
    read_count = await session.scalar(
        select(func.count(models.Notification.id))
        .where(models.Notification.gym_id == current_owner.gym_id)
        .where(models.Notification.is_read == True)
    )
    
    # Email sent count
    email_count = await session.scalar(
        select(func.count(models.Notification.id))
        .where(models.Notification.gym_id == current_owner.gym_id)
        .where(models.Notification.sent_via_email == True)
    )
    
    # By type
    type_breakdown = await session.execute(
        select(
            models.Notification.notification_type,
            func.count(models.Notification.id).label('count')
        )
        .where(models.Notification.gym_id == current_owner.gym_id)
        .group_by(models.Notification.notification_type)
    )
    
    return {
        "total_notifications": total_count or 0,
        "read_notifications": read_count or 0,
        "unread_notifications": (total_count or 0) - (read_count or 0),
        "email_sent_count": email_count or 0,
        "by_type": {row.notification_type: row.count for row in type_breakdown}
    }
