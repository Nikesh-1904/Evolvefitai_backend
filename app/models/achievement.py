import uuid
from datetime import datetime
from sqlalchemy import (
    String, DateTime, ForeignKey
)
from sqlalchemy.orm import relationship, Mapped, mapped_column
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import UUID as pgUUID

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .user import User
from .base import Base

class UserAchievement(Base):
    """User's unlocked achievements"""
    __tablename__ = "user_achievements"

    id: Mapped[uuid.UUID] = mapped_column(
        pgUUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        pgUUID(as_uuid=True), ForeignKey("user.id")
    )
    achievement_id: Mapped[str] = mapped_column(String)
    unlocked_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    user: Mapped["User"] = relationship("User", back_populates="achievements")
