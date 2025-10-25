# app/schemas.py
from pydantic import BaseModel, EmailStr, validator, Field
from typing import Optional, List, Dict, Any, Union
from datetime import datetime, date
import uuid
from fastapi_users import schemas
from enum import Enum

# ============================================================================
# SECTION 1: EXISTING CLIENT APP SCHEMAS (100% PRESERVED)
# ============================================================================

# --- Workout Logging Schemas ---

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
    exercise_type: Optional[ExerciseType] = None
    sets: List[AnyLoggedSet]


# --- Achievement Schemas ---

class UserAchievementBase(BaseModel):
    achievement_id: str


class UserAchievement(UserAchievementBase):
    id: int
    user_id: uuid.UUID
    unlocked_at: datetime

    class Config:
        from_attributes = True


class AchievementUnlockRequest(BaseModel):
    achievement_id: str


class AchievementStatus(BaseModel):
    total_points: int
    level: int
    unlocked_achievements: List[UserAchievement]


# --- User Schemas ---

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
    gym_id: Optional[int] = None
    last_gym_change: Optional[datetime] = None
    is_active: bool
    is_verified: bool
    created_at: datetime
    total_points: int
    level: int
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
    gym_id: Optional[int] = None

    @validator("username")
    def username_must_not_be_empty(cls, v):
        if v is not None and v == "":
            raise ValueError("Username cannot be an empty string")
        return v


# --- Exercise Schemas ---

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


# --- Video Schemas ---

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


# --- Workout Plan Schemas ---

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


# --- Workout Log Schemas ---

class WorkoutLogBase(BaseModel):
    duration_minutes: Optional[int] = None
    notes: Optional[str] = None
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


# --- Meal Plan Schemas ---

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


# --- Tips Schemas ---

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


# --- Analytics Schemas ---

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
    metric_type: str
    sets: List[Dict[str, Any]]


# --- Interaction Schemas ---

class TipInteractionCreate(BaseModel):
    tip_id: int
    interaction_type: str


class VideoPreferenceCreate(BaseModel):
    video_id: int
    preference: str


# --- AI Request/Response Schemas ---

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


# --- Response Schemas ---

class MessageResponse(BaseModel):
    message: str
    success: bool = True


# --- Gym and Booking Schemas ---

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
    member_count: int = 0
    gym_code: Optional[str] = None

    class Config:
        from_attributes = True


class JoinByCodeRequest(BaseModel):
    gym_code: str


class GymBookingBase(BaseModel):
    gym_id: uuid.UUID
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
    gym_id: uuid.UUID
    gym_name: str
    current_occupancy: int
    max_capacity: int
    overflow_count: int
    is_overcrowded: bool
    active_bookings_count: int


class LeaderboardEntry(BaseModel):
    user_id: uuid.UUID
    user_name: str
    total_workouts: int
    total_calories_burned: float
    total_minutes: int
    consistency_score: float
    rank: int


class LeaderboardResponse(BaseModel):
    gym_id: uuid.UUID
    gym_name: str
    gym_address: str
    leaderboard: List[LeaderboardEntry]
    total_members: int


# ============================================================================
# SECTION 2: NEW BUSINESS SCHEMAS (GYM OWNER FEATURES)
# ============================================================================

# --- Gym Owner Schemas ---

class GymOwnerBase(BaseModel):
    email: EmailStr
    full_name: str
    phone_number: Optional[str] = None


class GymOwnerCreate(GymOwnerBase):
    password: str
    gym_id: uuid.UUID


class GymOwnerRead(GymOwnerBase):
    id: uuid.UUID
    gym_id: uuid.UUID
    is_active: bool
    created_at: datetime
    last_login: Optional[datetime] = None

    class Config:
        from_attributes = True


class GymOwnerLogin(BaseModel):
    email: EmailStr
    password: str


# --- QR Code Schemas ---

class QRCodeResponse(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID
    qr_code_data: str
    qr_code_image_base64: str
    is_active: bool
    created_at: datetime
    expires_at: Optional[datetime] = None

    class Config:
        from_attributes = True


# --- Membership Fee Schemas ---

class MembershipFeeBase(BaseModel):
    user_id: uuid.UUID
    amount: float
    due_date: date
    notes: Optional[str] = None


class MembershipFeeCreate(MembershipFeeBase):
    pass


class MembershipFeeUpdate(BaseModel):
    status: Optional[str] = None
    paid_date: Optional[datetime] = None
    payment_method: Optional[str] = None
    notes: Optional[str] = None


class MembershipFeeRead(MembershipFeeBase):
    id: uuid.UUID
    gym_id: uuid.UUID
    status: str
    paid_date: Optional[datetime] = None
    payment_method: Optional[str] = None
    currency: str
    receipt_number: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


# --- Attendance Schemas ---

class CheckInRequest(BaseModel):
    qr_code_data: str


class CheckOutRequest(BaseModel):
    attendance_id: uuid.UUID


class AttendanceRecord(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID
    gym_id: uuid.UUID
    check_in_time: datetime
    check_out_time: Optional[datetime] = None
    duration_minutes: Optional[int] = None

    class Config:
        from_attributes = True


class LiveOccupancyResponse(BaseModel):
    gym_id: uuid.UUID
    gym_name: str
    current_occupancy: int
    capacity: int
    occupancy_percentage: float
    checked_in_users: List[Dict[str, Any]]
    timestamp: datetime


# --- Notification Schemas ---

class NotificationCreate(BaseModel):
    user_ids: List[uuid.UUID]
    notification_type: str
    title: str
    message: str
    send_email: bool = False
    send_app: bool = True


class NotificationRead(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID
    notification_type: str
    title: str
    message: str
    is_read: bool
    sent_via_email: bool
    sent_via_app: bool
    created_at: datetime
    read_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class NotificationPreferences(BaseModel):
    email_enabled: bool = True
    app_enabled: bool = True
    fee_reminder_days: List[int] = [7, 3, 1]


# --- Analytics Schemas ---

class MemberStats(BaseModel):
    user_id: uuid.UUID
    username: str
    full_name: str
    total_workouts: int
    avg_duration: float
    consistency_score: float
    membership_status: str
    last_payment_date: Optional[date] = None
    next_due_date: Optional[date] = None


class AnalyticsDashboard(BaseModel):
    total_members: int
    active_members: int
    revenue_this_month: float
    avg_attendance_per_day: float
    top_performers: List[MemberStats]
    recent_activity: List[AttendanceRecord]


class RevenueReport(BaseModel):
    total_revenue: float
    paid_count: int
    pending_count: int
    overdue_count: int
    breakdown_by_month: Dict[str, float]

class GymOwnerUpdate(BaseModel):
    full_name: Optional[str] = None
    phone_number: Optional[str] = None

class PerformanceAnalysis(BaseModel):
    total_workouts: int
    avg_duration: float
    consistency_score: float
    performance_trend: str
    weak_areas: Optional[List[str]] = []
    suggestions: Optional[List[str]] = []
    analysis_date: Optional[datetime] = None

MembershipFee = MembershipFeeRead

class BulkFeeCreate(BaseModel):
    user_ids: List[uuid.UUID]
    amount: float
    due_date: date
    notes: Optional[str] = None
