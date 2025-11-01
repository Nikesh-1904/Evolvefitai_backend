import uuid
from typing import List
from datetime import datetime
from sqlalchemy import (
    Integer, String, Text, Boolean, DateTime, Float, ForeignKey, JSON
)
from sqlalchemy.orm import relationship, Mapped, mapped_column
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import UUID as pgUUID
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .user import User
from .base import Base

class WorkoutPlan(Base):
    """AI-generated workout plan"""
    __tablename__ = "workout_plans"

    id: Mapped[uuid.UUID] = mapped_column(
        pgUUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        pgUUID(as_uuid=True), ForeignKey("user.id")
    )
    name: Mapped[str] = mapped_column(String)
    description: Mapped[str] = mapped_column(Text, nullable=True)
    difficulty: Mapped[str] = mapped_column(String, nullable=True)
    
    # ✅ FIX: Changed from duration_minutes to estimated_duration
    estimated_duration: Mapped[int] = mapped_column(Integer, nullable=True)
    
    exercises: Mapped[dict] = mapped_column(JSON)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    
    # ✅ FIX: Added missing AI fields
    ai_generated: Mapped[bool] = mapped_column(Boolean, default=False)
    ai_model: Mapped[str] = mapped_column(String, nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    user: Mapped["User"] = relationship("User", back_populates="workout_plans")


class WorkoutLog(Base):
    """Logged/completed workout with exercises"""
    __tablename__ = "workout_logs"

    id: Mapped[uuid.UUID] = mapped_column(
        pgUUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        pgUUID(as_uuid=True), ForeignKey("user.id")
    )
    workout_plan_id: Mapped[uuid.UUID] = mapped_column(
        pgUUID(as_uuid=True), ForeignKey("workout_plans.id"), nullable=True
    )
    name: Mapped[str] = mapped_column(String)
    duration_minutes: Mapped[int] = mapped_column(Integer)
    calories_burned: Mapped[float] = mapped_column(Float, nullable=True)
    exercises_completed: Mapped[dict] = mapped_column(JSON)
    notes: Mapped[str] = mapped_column(Text, nullable=True)
    
    # ✅ FIX: Added workout_date field (separate from logged_at)
    workout_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    
    logged_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    user: Mapped["User"] = relationship("User", back_populates="workout_logs")


class Exercise(Base):
    """Exercise library"""
    __tablename__ = "exercises"

    id: Mapped[uuid.UUID] = mapped_column(
        pgUUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String, unique=True, index=True)
    exercise_type: Mapped[str] = mapped_column(String)
    instructions: Mapped[str] = mapped_column(Text, nullable=True)
    muscle_groups: Mapped[dict] = mapped_column(JSON)
    equipment: Mapped[str] = mapped_column(String, nullable=True)
    difficulty: Mapped[str] = mapped_column(String, nullable=True)
    met_value: Mapped[float] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    videos: Mapped[List["ExerciseVideo"]] = relationship(
        "ExerciseVideo", back_populates="exercise", cascade="all, delete-orphan"
    )
    tips: Mapped[List["ExerciseTip"]] = relationship(
        "ExerciseTip", back_populates="exercise", cascade="all, delete-orphan"
    )


class ExerciseVideo(Base):
    """YouTube videos for exercises"""
    __tablename__ = "exercise_videos"

    id: Mapped[uuid.UUID] = mapped_column(
        pgUUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    exercise_id: Mapped[uuid.UUID] = mapped_column(
        pgUUID(as_uuid=True), ForeignKey("exercises.id")
    )
    
    # ✅ FIX: Added youtube_url field (keep video_id for compatibility)
    youtube_url: Mapped[str] = mapped_column(String, nullable=False)
    
    title: Mapped[str] = mapped_column(String)
    thumbnail_url: Mapped[str] = mapped_column(String, nullable=True)
    
    # ✅ FIX: Added duration field
    duration: Mapped[int] = mapped_column(Integer, nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    exercise: Mapped["Exercise"] = relationship("Exercise", back_populates="videos")


class ExerciseTip(Base):
    """Tips for exercises"""
    __tablename__ = "exercise_tips"

    id: Mapped[uuid.UUID] = mapped_column(
        pgUUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    exercise_id: Mapped[uuid.UUID] = mapped_column(
        pgUUID(as_uuid=True), ForeignKey("exercises.id")
    )
    
    # ✅ FIX: Changed from single 'tip' field to structured fields
    title: Mapped[str] = mapped_column(String, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    tip_type: Mapped[str] = mapped_column(String, nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    
    exercise: Mapped["Exercise"] = relationship("Exercise", back_populates="tips")