from pydantic import BaseModel, EmailStr
from typing import Optional, List, Dict, Any
from datetime import datetime
import uuid
from fastapi_users import schemas  # Add this line

# User schemas are now correctly inheriting from BaseModel
class UserRead(schemas.BaseUser[uuid.UUID]):
    id: uuid.UUID
    email: EmailStr
    username: Optional[str] = None
    full_name: Optional[str] = None
    age: Optional[int] = None
    weight: Optional[float] = None
    height: Optional[float] = None
    gender: Optional[str] = None
    fitness_goal: Optional[str] = None
    experience_level: Optional[str] = None
    activity_level: Optional[str] = None
    dietary_restrictions: Optional[List[str]] = []
    is_active: bool
    is_verified: bool
    created_at: datetime

    class Config:
        from_attributes = True

class UserCreate(schemas.BaseUserCreate):
    email: EmailStr
    password: str
    username: Optional[str] = None
    full_name: Optional[str] = None

class UserUpdate(schemas.BaseUserUpdate):
    username: Optional[str] = None
    full_name: Optional[str] = None
    age: Optional[int] = None
    weight: Optional[float] = None
    height: Optional[float] = None
    gender: Optional[str] = None
    fitness_goal: Optional[str] = None
    experience_level: Optional[str] = None
    activity_level: Optional[str] = None
    dietary_restrictions: Optional[List[str]] = []

# Exercise schemas
class ExerciseBase(BaseModel):
    name: str
    category: Optional[str] = None
    muscle_groups: Optional[List[str]] = []
    equipment: Optional[str] = None
    difficulty: Optional[str] = None
    instructions: Optional[str] = None

class ExerciseCreate(ExerciseBase):
    pass

class Exercise(ExerciseBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True

# Video schemas
class ExerciseVideoBase(BaseModel):
    youtube_url: str
    title: Optional[str] = None
    thumbnail_url: Optional[str] = None
    duration: Optional[int] = None

class ExerciseVideo(ExerciseVideoBase):
    id: int
    exercise_id: int
    popularity_score: float
    created_at: datetime

    class Config:
        from_attributes = True

# Workout schemas
class WorkoutPlanBase(BaseModel):
    name: str
    description: Optional[str] = None
    exercises: List[Dict[str, Any]] = []
    difficulty: Optional[str] = None
    estimated_duration: Optional[int] = None

class WorkoutPlanCreate(WorkoutPlanBase):
    pass

class WorkoutPlan(WorkoutPlanBase):
    id: int
    user_id: uuid.UUID
    ai_generated: bool
    ai_model: Optional[str] = None
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True

class WorkoutLogBase(BaseModel):
    exercises_completed: List[Dict[str, Any]] = []
    duration_minutes: Optional[int] = None
    notes: Optional[str] = None

class WorkoutLogCreate(WorkoutLogBase):
    workout_plan_id: Optional[int] = None

class WorkoutLog(WorkoutLogBase):
    id: int
    user_id: uuid.UUID
    workout_plan_id: Optional[int] = None
    workout_date: datetime

    class Config:
        from_attributes = True

# Meal Plan schemas
class MealPlanBase(BaseModel):
    name: str
    target_calories: int
    target_protein: float
    target_carbs: float
    target_fat: float
    meals: Dict[str, Any] = {}

class MealPlanCreate(MealPlanBase):
    pass

class MealPlan(MealPlanBase):
    id: int
    user_id: uuid.UUID
    ai_generated: bool
    ai_model: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True

# Tips schemas
class ExerciseTipBase(BaseModel):
    title: str
    content: str
    tip_type: Optional[str] = None

class ExerciseTip(ExerciseTipBase):
    id: int
    exercise_id: int
    popularity_score: float
    created_at: datetime

    class Config:
        from_attributes = True

# Interaction schemas
class TipInteractionCreate(BaseModel):
    tip_id: int
    interaction_type: str  # like, dislike

class VideoPreferenceCreate(BaseModel):
    video_id: int
    preference: str  # like, dislike

# AI Request/Response schemas
class WorkoutGenerationRequest(BaseModel):
    user_preferences: Optional[Dict[str, Any]] = {}
    duration_minutes: Optional[int] = 45

class PlateauAnalysis(BaseModel):
    is_plateau: bool
    confidence: float
    affected_exercises: List[str]
    recommendations: List[str]
    plateau_duration_weeks: int
    analysis_method: str
    ai_generated: bool = False

class MealPlanRequest(BaseModel):
    duration_days: int = 7
    preferences: Optional[Dict[str, Any]] = {}

# Response schemas
class MessageResponse(BaseModel):
    message: str
    success: bool = True