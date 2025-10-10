# app/models.py

import uuid
from typing import List
from datetime import datetime

from sqlalchemy import (
    Integer,
    String,
    Text,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    JSON,
)
from sqlalchemy.orm import relationship, Mapped, mapped_column
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import UUID as pgUUID

from app.core.database import Base
from fastapi_users.db import (
    SQLAlchemyBaseUserTableUUID,
    SQLAlchemyBaseOAuthAccountTableUUID,
)


class User(SQLAlchemyBaseUserTableUUID, Base):
    """User model with FastAPI Users integration"""
    __tablename__ = "users"
    
    username: Mapped[str] = mapped_column(String, unique=True, index=True, nullable=True)
    full_name: Mapped[str] = mapped_column(String, nullable=True)
    age: Mapped[int] = mapped_column(Integer, nullable=True)
    weight: Mapped[float] = mapped_column(Float, nullable=True)
    height: Mapped[float] = mapped_column(Float, nullable=True)
    gender: Mapped[str] = mapped_column(String, nullable=True)
    fitness_goal: Mapped[str] = mapped_column(String, nullable=True)
    experience_level: Mapped[str] = mapped_column(String, nullable=True)
    activity_level: Mapped[str] = mapped_column(String, nullable=True)
    dietary_restrictions: Mapped[list] = mapped_column(JSON, default=list)
    has_completed_onboarding: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), onupdate=func.now(), nullable=True)
    
    oauth_accounts: Mapped[List["OAuthAccount"]] = relationship(back_populates="user", lazy="joined")
    workout_logs: Mapped[List["WorkoutLog"]] = relationship(back_populates="user")
    workout_plans: Mapped[List["WorkoutPlan"]] = relationship(back_populates="user")
    meal_plans: Mapped[List["MealPlan"]] = relationship(back_populates="user")
    tip_interactions: Mapped[List["TipInteraction"]] = relationship(back_populates="user")
    video_preferences: Mapped[List["VideoPreference"]] = relationship(back_populates="user")


class OAuthAccount(SQLAlchemyBaseOAuthAccountTableUUID, Base):
    user_id: uuid.UUID = mapped_column(pgUUID(as_uuid=True), ForeignKey("users.id", ondelete="cascade"), nullable=False)
    user: Mapped["User"] = relationship(back_populates="oauth_accounts")


class Exercise(Base):
    __tablename__ = "exercises"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String, nullable=False, index=True)
    exercise_type: Mapped[str] = mapped_column(String, default='WEIGHT_BASED', nullable=False)
    muscle_groups: Mapped[List[str]] = mapped_column(JSON, default=list)
    equipment: Mapped[str] = mapped_column(String, nullable=True)
    difficulty: Mapped[str] = mapped_column(String, nullable=True)
    instructions: Mapped[str] = mapped_column(Text, nullable=True)
    met_value: Mapped[float] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    
    videos: Mapped[List["ExerciseVideo"]] = relationship(back_populates="exercise")
    tips: Mapped[List["ExerciseTip"]] = relationship(back_populates="exercise")


class ExerciseVideo(Base):
    __tablename__ = "exercise_videos"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    exercise_id: Mapped[int] = mapped_column(Integer, ForeignKey("exercises.id"))
    youtube_url: Mapped[str] = mapped_column(String, nullable=False)
    title: Mapped[str] = mapped_column(String, nullable=True)
    thumbnail_url: Mapped[str] = mapped_column(String, nullable=True)
    duration: Mapped[int] = mapped_column(Integer, nullable=True)
    popularity_score: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    
    exercise: Mapped["Exercise"] = relationship(back_populates="videos")
    preferences: Mapped[List["VideoPreference"]] = relationship(back_populates="video")


class WorkoutPlan(Base):
    __tablename__ = "workout_plans"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[uuid.UUID] = mapped_column(pgUUID(as_uuid=True), ForeignKey("users.id"))
    name: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=True)
    exercises: Mapped[list] = mapped_column(JSON, default=list)
    difficulty: Mapped[str] = mapped_column(String, nullable=True)
    estimated_duration: Mapped[int] = mapped_column(Integer, nullable=True)
    ai_generated: Mapped[bool] = mapped_column(Boolean, default=False)
    ai_model: Mapped[str] = mapped_column(String, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    
    user: Mapped["User"] = relationship(back_populates="workout_plans")


class WorkoutLog(Base):
    __tablename__ = "workout_logs"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[uuid.UUID] = mapped_column(pgUUID(as_uuid=True), ForeignKey("users.id"))
    workout_plan_id: Mapped[int] = mapped_column(Integer, ForeignKey("workout_plans.id"), nullable=True)
    exercises_completed: Mapped[list] = mapped_column(JSON, default=list)
    duration_minutes: Mapped[int] = mapped_column(Integer, nullable=True)
    calories_burned: Mapped[float] = mapped_column(Float, nullable=True)
    notes: Mapped[str] = mapped_column(Text, nullable=True)
    workout_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    
    user: Mapped["User"] = relationship(back_populates="workout_logs")


class MealPlan(Base):
    __tablename__ = "meal_plans"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[uuid.UUID] = mapped_column(pgUUID(as_uuid=True), ForeignKey("users.id"))
    name: Mapped[str] = mapped_column(String, nullable=False)
    target_calories: Mapped[int] = mapped_column(Integer, nullable=True)
    target_protein: Mapped[float] = mapped_column(Float, nullable=True)
    target_carbs: Mapped[float] = mapped_column(Float, nullable=True)
    target_fat: Mapped[float] = mapped_column(Float, nullable=True)
    meals: Mapped[dict] = mapped_column(JSON, default=dict)
    ai_generated: Mapped[bool] = mapped_column(Boolean, default=False)
    ai_model: Mapped[str] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    
    user: Mapped["User"] = relationship(back_populates="meal_plans")


class ExerciseTip(Base):
    __tablename__ = "exercise_tips"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    exercise_id: Mapped[int] = mapped_column(Integer, ForeignKey("exercises.id"))
    title: Mapped[str] = mapped_column(String, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    tip_type: Mapped[str] = mapped_column(String, nullable=True)
    popularity_score: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    
    exercise: Mapped["Exercise"] = relationship(back_populates="tips")
    interactions: Mapped[List["TipInteraction"]] = relationship(back_populates="tip")


class TipInteraction(Base):
    __tablename__ = "tip_interactions"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[uuid.UUID] = mapped_column(pgUUID(as_uuid=True), ForeignKey("users.id"))
    tip_id: Mapped[int] = mapped_column(Integer, ForeignKey("exercise_tips.id"))
    interaction_type: Mapped[str] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    
    user: Mapped["User"] = relationship(back_populates="tip_interactions")
    tip: Mapped["ExerciseTip"] = relationship(back_populates="interactions")


class VideoPreference(Base):
    __tablename__ = "video_preferences"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[uuid.UUID] = mapped_column(pgUUID(as_uuid=True), ForeignKey("users.id"))
    video_id: Mapped[int] = mapped_column(Integer, ForeignKey("exercise_videos.id"))
    preference: Mapped[str] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    
    user: Mapped["User"] = relationship(back_populates="video_preferences")
    video: Mapped["ExerciseVideo"] = relationship(back_populates="preferences")