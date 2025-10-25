# app/business/services/fee_reminder_service.py
"""Background service for automatic fee reminders"""

from datetime import datetime, timedelta
from typing import List
import logging
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_

from app import models
from app.core.database import async_session_maker
from app.business.services.email_service import email_service

logger = logging.getLogger(__name__)


class FeeReminderService:
    """Service for sending automatic fee reminders"""

    async def send_daily_reminders(self) -> dict:
        """
        Send fee reminders to users based on their preferences
        
        This should be called daily (e.g., via cron job or scheduler)
        
        Returns:
            Dictionary with reminder statistics
        """
        async with async_session_maker() as session:
            stats = {
                "total_checked": 0,
                "reminders_sent": 0,
                "overdue_notices_sent": 0,
                "errors": 0
            }
            
            try:
                # Get all pending fees
                result = await session.execute(
                    select(models.MembershipFee)
                    .where(models.MembershipFee.status == "PENDING")
                )
                pending_fees = result.scalars().all()
                
                stats["total_checked"] = len(pending_fees)
                
                for fee in pending_fees:
                    try:
                        # Get user details
                        user = await session.get(models.User, fee.user_id)
                        if not user or not user.email:
                            continue
                        
                        # Get gym details
                        gym = await session.get(models.Gym, fee.gym_id)
                        if not gym:
                            continue
                        
                        # Check user notification preferences
                        preferences = user.notification_preferences or {}
                        email_enabled = preferences.get("email_enabled", True)
                        reminder_days = preferences.get("fee_reminder_days", [7, 3, 1])
                        
                        if not email_enabled:
                            continue
                        
                        # Calculate days until due
                        days_until_due = (fee.due_date.date() - datetime.utcnow().date()).days
                        
                        # Check if reminder should be sent
                        if days_until_due in reminder_days and days_until_due >= 0:
                            # Send reminder
                            success = await email_service.send_fee_reminder(
                                to_email=user.email,
                                user_name=user.full_name or user.username or "Member",
                                amount=fee.amount,
                                due_date=fee.due_date,
                                gym_name=gym.name
                            )
                            
                            if success:
                                stats["reminders_sent"] += 1
                                
                                # Create notification record
                                notification = models.Notification(
                                    user_id=user.id,
                                    gym_id=gym.id,
                                    notification_type="FEE_REMINDER",
                                    title="Fee Payment Reminder",
                                    message=f"Your fee of ₹{fee.amount} is due on {fee.due_date.strftime('%d %b %Y')}",
                                    sent_via_email=True,
                                    sent_via_app=True
                                )
                                session.add(notification)
                        
                        # Check for overdue fees
                        elif days_until_due < 0:
                            # Update fee status to OVERDUE
                            if fee.status != "OVERDUE":
                                fee.status = "OVERDUE"
                            
                            # Send overdue notice (once, when it becomes overdue)
                            if days_until_due == -1:  # Send only on first overdue day
                                success = await email_service.send_fee_overdue_notice(
                                    to_email=user.email,
                                    user_name=user.full_name or user.username or "Member",
                                    amount=fee.amount,
                                    due_date=fee.due_date,
                                    gym_name=gym.name
                                )
                                
                                if success:
                                    stats["overdue_notices_sent"] += 1
                                    
                                    # Create notification record
                                    notification = models.Notification(
                                        user_id=user.id,
                                        gym_id=gym.id,
                                        notification_type="FEE_OVERDUE",
                                        title="⚠️ Overdue Fee Payment",
                                        message=f"Your fee of ₹{fee.amount} is now overdue. Please pay immediately.",
                                        sent_via_email=True,
                                        sent_via_app=True
                                    )
                                    session.add(notification)
                    
                    except Exception as e:
                        logger.error(f"Error processing fee {fee.id}: {str(e)}")
                        stats["errors"] += 1
                        continue
                
                # Commit all notification records
                await session.commit()
                
            except Exception as e:
                logger.error(f"Error in daily reminders: {str(e)}")
                stats["errors"] += 1
            
            logger.info(f"Daily reminders complete: {stats}")
            return stats

    async def send_bulk_reminder(
        self,
        session: AsyncSession,
        gym_id: str,
        reminder_days: int = 7
    ) -> int:
        """
        Send bulk reminder for upcoming fees in a specific gym
        
        Args:
            session: Database session
            gym_id: Gym UUID
            reminder_days: Send reminder for fees due in N days
        
        Returns:
            Number of reminders sent
        """
        target_date = datetime.utcnow().date() + timedelta(days=reminder_days)
        
        result = await session.execute(
            select(models.MembershipFee)
            .where(
                and_(
                    models.MembershipFee.gym_id == gym_id,
                    models.MembershipFee.status == "PENDING",
                    models.MembershipFee.due_date.cast(models.DateTime.date) == target_date
                )
            )
        )
        fees = result.scalars().all()
        
        sent_count = 0
        for fee in fees:
            user = await session.get(models.User, fee.user_id)
            gym = await session.get(models.Gym, fee.gym_id)
            
            if user and user.email and gym:
                success = await email_service.send_fee_reminder(
                    to_email=user.email,
                    user_name=user.full_name or user.username or "Member",
                    amount=fee.amount,
                    due_date=fee.due_date,
                    gym_name=gym.name
                )
                if success:
                    sent_count += 1
        
        return sent_count


# Singleton instance
fee_reminder_service = FeeReminderService()
