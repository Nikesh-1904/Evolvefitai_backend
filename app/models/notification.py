import uuid
from datetime import datetime
from sqlalchemy import (
     String, Text, Boolean, DateTime, ForeignKey
)
from sqlalchemy.orm import relationship, Mapped, mapped_column
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import UUID as pgUUID

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .user import User
    from .gym import Gym
from .base import Base

class Notification(Base):
    """In-app and email notifications"""
    __tablename__ = "notifications"

    id: Mapped[uuid.UUID] = mapped_column(
        pgUUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        pgUUID(as_uuid=True), ForeignKey("user.id"), nullable=False
    )
    gym_id: Mapped[uuid.UUID] = mapped_column(
        pgUUID(as_uuid=True), ForeignKey("gyms.id"), nullable=False
    )
    
    notification_type: Mapped[str] = mapped_column(
        String, nullable=False  # FEE_REMINDER, FEE_OVERDUE, GENERAL, ANNOUNCEMENT
    )
    title: Mapped[str] = mapped_column(String, nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    
    is_read: Mapped[bool] = mapped_column(Boolean, default=False)
    sent_via_email: Mapped[bool] = mapped_column(Boolean, default=False)
    sent_via_app: Mapped[bool] = mapped_column(Boolean, default=True)
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    read_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="notifications")
    gym: Mapped["Gym"] = relationship("Gym", back_populates="notifications")
