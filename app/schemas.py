# app/schemas.py
from pydantic import BaseModel, EmailStr, validator
from typing import Optional, List, Dict, Any, Union
from datetime import datetime, date
import uuid
from fastapi_users import schemas
from enum import Enum

# --- NEW Schemas for Workout Logging ---

# --- START OF NEW/UPDATED LOGGING SCHEMAS ---

# 1. Define our exercise types using an Enum for type safety
class ExerciseType(str, Enum):
    WEIGHT_BASED = "WEIGHT_BASED"
    REPS_ONLY = "REPS_ONLY"
    DURATION = "DURATION"
    DISTANCE_DURATION = "DISTANCE_DURATION"
    QUALITATIVE = "QUALITATIVE"


# 2. Define a schema for each type of logged data point ("set")
class LoggedSetWeightBased(BaseModel):
    reps: int
    weight: float


class LoggedSetRepsOnly(BaseModel):
    reps: int


class LoggedSetDuration(BaseModel):
    duration_seconds: int


class LoggedSetDistanceDuration(BaseModel):
    duration_seconds: int
    distance_km: float


class LoggedSetQualitative(BaseModel):
    duration_seconds: Optional[int] = None
    notes: Optional[str] = None


# 3. Create a Union type that can be any of the above schemas
AnyLoggedSet = Union[
    LoggedSetWeightBased,
    LoggedSetRepsOnly,
    LoggedSetDuration,
    LoggedSetDistanceDuration,
    LoggedSetQualitative,
]


# 4. Update LoggedExercise to use the new Union type and require the exercise_type
class LoggedExercise(BaseModel):
    name: str
    exercise_type: Optional[ExerciseType] = None  # We need this to know how to interpret the sets data
    sets: List[AnyLoggedSet]

# 1. Schema for the UserAchievement model
class UserAchievementBase(BaseModel):
    achievement_id: str

class UserAchievement(UserAchievementBase):
    id: int
    user_id: uuid.UUID
    unlocked_at: datetime

    class Config:
        from_attributes = True

# 2. Schema for the request to unlock an achievement
class AchievementUnlockRequest(BaseModel):
    achievement_id: str

# 3. Schema for the response from our new endpoints
class AchievementStatus(BaseModel):
    total_points: int
    level: int
    unlocked_achievements: List[UserAchievement]

# User schemas
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
    has_completed_onboarding: bool
    preferences: Optional[Dict[str, Any]] = {}
    gym_id: Optional[int] = None  # NEW
    last_gym_change: Optional[datetime] = None  # NEW
    is_active: bool
    is_verified: bool
    created_at: datetime
    total_points: int
    level: int
    # We will populate this list from the UserAchievement table
    achievements: List["UserAchievement"] = []

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
    has_completed_onboarding: Optional[bool] = None
    preferences: Optional[Dict[str, Any]] = None
    gym_id: Optional[int] = None  # NEW

    @validator("username")
    def username_must_not_be_empty(cls, v):
        if v is not None and v == "":
            raise ValueError("Username cannot be an empty string")
        return v


# Exercise schemas
class ExerciseBase(BaseModel):
    name: str
    exercise_type: Optional[ExerciseType] = ExerciseType.WEIGHT_BASED
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


# --- UPDATED Workout Log Schemas ---
class WorkoutLogBase(BaseModel):
    duration_minutes: Optional[int] = None
    notes: Optional[str] = None
    # We now expect a list of Pydantic models, not just dicts
    exercises_completed: List[LoggedExercise] = []


class WorkoutLogCreate(WorkoutLogBase):
    workout_plan_id: Optional[int] = None
    workout_date: Optional[datetime] = None


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


class LevelProgress(BaseModel):
    current_level: int
    current_points: int
    points_for_current_level: int
    points_for_next_level: int


class DashboardOverviewStats(BaseModel):
    workouts_completed: int
    total_workout_time_hours: float
    total_calories_burned: int
    level_progress: LevelProgress
    calories_change_percent: float
    time_change_percent: float


class TimeSeriesDataPoint(BaseModel):
    date: date
    value: float


class AnalyticsData(BaseModel):
    calorie_timeseries: List[TimeSeriesDataPoint]
    workout_heatmap: List[date]


class ExerciseSetData(BaseModel):
    reps: int
    weight: float


class ExerciseProgressionDataPoint(BaseModel):
    workout_date: date
    primary_metric_value: float
    metric_type: str  # e.g., 'volume', 'reps', 'duration_seconds'
    sets: List[Dict[str, Any]]  # Keep sets flexible


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
    interaction_type: str


class VideoPreferenceCreate(BaseModel):
    video_id: int
    preference: str


# AI Request/Response schemas
class WorkoutGenerationRequest(BaseModel):
    user_preferences: Optional[Dict[str, Any]] = {}
    duration_minutes: Optional[int] = 45
    target_muscle_groups: Optional[List[str]] = None
    num_exercises: Optional[int] = None
    workout_type: Optional[str] = None


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


class METValueResponse(BaseModel):
    exercise_name: str
    met_value: float


# Response schemas
class MessageResponse(BaseModel):
    message: str
    success: bool = True


# ========== NEW GYM AND BOOKING SCHEMAS ==========

class GymBase(BaseModel):
    name: str
    address: str
    city: str
    state: Optional[str] = None
    country: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    logo_url: Optional[str] = None
    operating_hours: Optional[Dict[str, Any]] = {}
    max_capacity: int = 100


class GymCreate(GymBase):
    gym_code: Optional[str] = None


class Gym(GymBase):
    id: int
    created_at: datetime
    updated_at: Optional[datetime] = None
    member_count: int = 0 # 👈 ADD THIS LINE with a default
    gym_code: Optional[str] = None

    class Config:
        from_attributes = True
class JoinByCodeRequest(BaseModel):
    gym_code: str
class GymBookingBase(BaseModel):
    gym_id: int
    start_time: datetime
    end_time: datetime


class GymBookingCreate(GymBookingBase):
    pass


class GymBooking(GymBookingBase):
    id: int
    user_id: uuid.UUID
    status: str
    created_at: datetime
    cancelled_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class GymOccupancyResponse(BaseModel):
    gym_id: int
    gym_name: str
    current_occupancy: int
    max_capacity: int
    overflow_count: int
    is_overcrowded: bool
    active_bookings_count: int


class LeaderboardEntry(BaseModel):
    user_id: uuid.UUID
    user_name: str # Combined name field
    total_workouts: int
    total_calories_burned: float # Keep this name, frontend uses total_calories
    total_minutes: int # Changed from hours to minutes
    consistency_score: float # Keep consistency_score, frontend will need to adapt
    rank: int


class LeaderboardResponse(BaseModel):
    gym_id: int
    gym_name: str
    gym_address: str # 👈 ADD THIS FIELD
    leaderboard: List[LeaderboardEntry]
    total_members: int


