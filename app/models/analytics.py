import uuid
from datetime import datetime
from sqlalchemy import (
    Integer, String, DateTime, Float, ForeignKey, JSON
)
from sqlalchemy.orm import relationship, Mapped, mapped_column
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import UUID as pgUUID

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .user import User
    from .gym import Gym
from .base import Base

class MemberPerformance(Base):
    """AI-analyzed member performance metrics"""
    __tablename__ = "member_performance"

    id: Mapped[uuid.UUID] = mapped_column(
        pgUUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        pgUUID(as_uuid=True), ForeignKey("user.id"), nullable=False
    )
    gym_id: Mapped[uuid.UUID] = mapped_column(
        pgUUID(as_uuid=True), ForeignKey("gyms.id"), nullable=False
    )
    
    analysis_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    
    # Metrics
    total_workouts: Mapped[int] = mapped_column(Integer, default=0)
    avg_workout_duration: Mapped[float] = mapped_column(Float, default=0.0)
    consistency_score: Mapped[float] = mapped_column(Float, default=0.0)  # 0-100
    
    performance_trend: Mapped[str] = mapped_column(
        String, default="STABLE"  # IMPROVING, STABLE, DECLINING
    )
    
    weak_areas: Mapped[dict] = mapped_column(JSON, default=list)  # List of muscle groups
    suggestions: Mapped[dict] = mapped_column(JSON, default=list)  # AI recommendations
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="performance_records")
    gym: Mapped["Gym"] = relationship("Gym", back_populates="performance_records")