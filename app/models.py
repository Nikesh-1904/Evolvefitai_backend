from fastapi_users.db import SQLAlchemyBaseUserTableUUID, SQLAlchemyBaseOAuthAccountTableUUID
from sqlalchemy import Column, Integer, String, Text, Boolean, DateTime, Float, ForeignKey, JSON
from sqlalchemy.orm import relationship, Mapped
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import UUID
import uuid
from typing import List

from app.core.database import Base

# --- The User model is now defined BEFORE the OAuthAccount model ---
class User(SQLAlchemyBaseUserTableUUID, Base):
    """User model with FastAPI Users integration"""
    __tablename__ = "users"
    
    # FastAPI Users provides: id (UUID), email, hashed_password, is_active, is_superuser, is_verified
    
    # Additional profile fields
    username = Column(String, unique=True, index=True)
    full_name = Column(String)
    
    # Fitness profile
    age = Column(Integer)
    weight = Column(Float)  # kg
    height = Column(Float)  # cm
    gender = Column(String)
    fitness_goal = Column(String)
    experience_level = Column(String)
    activity_level = Column(String)
    
    # Preferences
    dietary_restrictions = Column(JSON, default=list)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # This relationship links the User to the OAuthAccount model below
    oauth_accounts: Mapped[List["OAuthAccount"]] = relationship(lazy="joined")
    
    # Relationships to your other tables
    workout_logs = relationship("WorkoutLog", back_populates="user")
    workout_plans = relationship("WorkoutPlan", back_populates="user")
    meal_plans = relationship("MealPlan", back_populates="user")
    tip_interactions = relationship("TipInteraction", back_populates="user")
    video_preferences = relationship("VideoPreference", back_populates="user")

# --- This model now comes AFTER the User model ---
class OAuthAccount(SQLAlchemyBaseOAuthAccountTableUUID, Base):
    pass


# ... (The rest of your models file remains exactly the same) ...
class Exercise(Base):
    __tablename__ = "exercises"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False, index=True)
    category = Column(String)
    muscle_groups = Column(JSON, default=list)
    equipment = Column(String)
    difficulty = Column(String)
    instructions = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    videos = relationship("ExerciseVideo", back_populates="exercise")
    tips = relationship("ExerciseTip", back_populates="exercise")

class ExerciseVideo(Base):
    __tablename__ = "exercise_videos"
    
    id = Column(Integer, primary_key=True, index=True)
    exercise_id = Column(Integer, ForeignKey("exercises.id"))
    youtube_url = Column(String, nullable=False)
    title = Column(String)
    thumbnail_url = Column(String)
    duration = Column(Integer)
    popularity_score = Column(Float, default=0.0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    exercise = relationship("Exercise", back_populates="videos")
    preferences = relationship("VideoPreference", back_populates="video")

class WorkoutPlan(Base):
    __tablename__ = "workout_plans"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    name = Column(String, nullable=False)
    description = Column(Text)
    exercises = Column(JSON, default=list)
    difficulty = Column(String)
    estimated_duration = Column(Integer)
    ai_generated = Column(Boolean, default=False)
    ai_model = Column(String)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    user = relationship("User", back_populates="workout_plans")

class WorkoutLog(Base):
    __tablename__ = "workout_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    workout_plan_id = Column(Integer, ForeignKey("workout_plans.id"), nullable=True)
    exercises_completed = Column(JSON, default=list)
    duration_minutes = Column(Integer)
    notes = Column(Text)
    workout_date = Column(DateTime(timezone=True), server_default=func.now())
    
    user = relationship("User", back_populates="workout_logs")

class MealPlan(Base):
    __tablename__ = "meal_plans"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    name = Column(String, nullable=False)
    target_calories = Column(Integer)
    target_protein = Column(Float)
    target_carbs = Column(Float)
    target_fat = Column(Float)
    meals = Column(JSON, default=dict)
    ai_generated = Column(Boolean, default=False)
    ai_model = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    user = relationship("User", back_populates="meal_plans")

class ExerciseTip(Base):
    __tablename__ = "exercise_tips"
    
    id = Column(Integer, primary_key=True, index=True)
    exercise_id = Column(Integer, ForeignKey("exercises.id"))
    title = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    tip_type = Column(String)
    popularity_score = Column(Float, default=0.0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    exercise = relationship("Exercise", back_populates="tips")
    interactions = relationship("TipInteraction", back_populates="tip")

class TipInteraction(Base):
    __tablename__ = "tip_interactions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    tip_id = Column(Integer, ForeignKey("exercise_tips.id"))
    interaction_type = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    user = relationship("User", back_populates="tip_interactions")
    tip = relationship("ExerciseTip", back_populates="interactions")

class VideoPreference(Base):
    __tablename__ = "video_preferences"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    video_id = Column(Integer, ForeignKey("exercise_videos.id"))
    preference = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    user = relationship("User", back_populates="video_preferences")
    video = relationship("ExerciseVideo", back_populates="preferences")

