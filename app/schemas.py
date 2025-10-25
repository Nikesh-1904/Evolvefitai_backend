# app/schemas.py
from pydantic import BaseModel, EmailStr, validator, Field
from typing import Optional, List, Dict, Any, Union
from datetime import datetime, date
import uuid
from fastapi_users import schemas
from enum import Enum


# --- EXISTING Schemas for Workout Logging ---
class ExerciseType(str, Enum):
    WEIGHT_BASED = "WEIGHT_BASED"
    REPS_ONLY = "REPS_ONLY"
    DURATION = "DURATION"
    DISTANCE_DURATION = "DISTANCE_DURATION"
    QUALITATIVE = "QUALITATIVE"


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


class LoggedExercise(BaseModel):
    exercise_name: str
    exercise_type: ExerciseType
    sets: List[
        Union[
            LoggedSetWeightBased,
            LoggedSetRepsOnly,
            LoggedSetDuration,
            LoggedSetDistanceDuration,
            LoggedSetQualitative,
        ]
    ]
    notes: Optional[str] = None


class WorkoutLogCreate(BaseModel):
    workout_plan_id: Optional[uuid.UUID] = None
    name: str
    duration_minutes: int
    exercises_completed: List[LoggedExercise]
    notes: Optional[str] = None


class WorkoutLog(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID
    workout_plan_id: Optional[uuid.UUID]
    name: str
    duration_minutes: int
    calories_burned: Optional[float]
    exercises_completed: List[LoggedExercise]
    notes: Optional[str]
    logged_at: datetime

    class Config:
        from_attributes = True


# --- User Schemas ---
class UserRead(schemas.BaseUser[uuid.UUID]):
    username: Optional[str] = None
    full_name: Optional[str] = None
    age: Optional[int] = None
    weight: Optional[float] = None
    height: Optional[float] = None
    gender: Optional[str] = None
    fitness_level: Optional[str] = None
    fitness_goal: Optional[str] = None
    activity_level: Optional[str] = None
    dietary_preference: Optional[str] = None
    has_completed_onboarding: bool = False
    preferences: Optional[dict] = None
    total_points: int = 0
    current_level: int = 1
    gym_id: Optional[uuid.UUID] = None
    
    # 🆕 Business fields
    qr_code_id: Optional[uuid.UUID] = None
    membership_status: str = "ACTIVE"
    membership_expiry: Optional[datetime] = None
    notification_preferences: Optional[dict] = None


class UserCreate(schemas.BaseUserCreate):
    username: Optional[str] = None
    full_name: Optional[str] = None


class UserUpdate(schemas.BaseUserUpdate):
    username: Optional[str] = None
    full_name: Optional[str] = None
    age: Optional[int] = None
    weight: Optional[float] = None
    height: Optional[float] = None
    gender: Optional[str] = None
    fitness_level: Optional[str] = None
    fitness_goal: Optional[str] = None
    activity_level: Optional[str] = None
    dietary_preference: Optional[str] = None
    has_completed_onboarding: Optional[bool] = None
    preferences: Optional[dict] = None
    gym_id: Optional[uuid.UUID] = None
    
    # 🆕 Business fields
    notification_preferences: Optional[dict] = None


# --- Workout Plan Schemas ---
class WorkoutPlanCreate(BaseModel):
    name: str
    description: Optional[str] = None
    difficulty: Optional[str] = None
    duration_minutes: int
    exercises: Dict[str, Any]


class WorkoutPlan(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID
    name: str
    description: Optional[str]
    difficulty: Optional[str]
    duration_minutes: int
    exercises: Dict[str, Any]
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True


# --- Exercise Schemas ---
class ExerciseCreate(BaseModel):
    name: str
    exercise_type: str
    instructions: Optional[str] = None
    muscle_groups: List[str]
    equipment: Optional[str] = None
    difficulty: Optional[str] = None
    met_value: Optional[float] = None


class Exercise(BaseModel):
    id: uuid.UUID
    name: str
    exercise_type: str
    instructions: Optional[str]
    muscle_groups: List[str]
    equipment: Optional[str]
    difficulty: Optional[str]
    met_value: Optional[float]
    created_at: datetime

    class Config:
        from_attributes = True


# --- Meal Plan Schemas ---
class MealPlanCreate(BaseModel):
    name: str
    target_calories: float
    target_protein: Optional[float] = None
    target_carbs: Optional[float] = None
    target_fat: Optional[float] = None
    meals: Dict[str, Any]


class MealPlan(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID
    name: str
    target_calories: float
    target_protein: Optional[float]
    target_carbs: Optional[float]
    target_fat: Optional[float]
    meals: Dict[str, Any]
    created_at: datetime

    class Config:
        from_attributes = True


# --- Gym Schemas ---
class GymCreate(BaseModel):
    name: str
    description: Optional[str] = None
    address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    country: str = "India"
    pincode: Optional[str] = None
    capacity: int = 50
    gym_code: str
    
    # 🆕 Business fields
    monthly_fee: Optional[float] = None
    currency: str = "INR"
    fee_due_day: int = 1


class Gym(BaseModel):
    id: uuid.UUID
    name: str
    description: Optional[str]
    address: Optional[str]
    city: Optional[str]
    state: Optional[str]
    country: str
    pincode: Optional[str]
    capacity: int
    gym_code: str
    created_at: datetime
    
    # 🆕 Business fields
    monthly_fee: Optional[float]
    currency: str
    fee_due_day: int

    class Config:
        from_attributes = True


class GymOccupancyResponse(BaseModel):
    gym_id: uuid.UUID
    gym_name: str
    current_occupancy: int
    capacity: int
    occupancy_percentage: float
    timestamp: datetime


# --- Gym Booking Schemas ---
class GymBookingCreate(BaseModel):
    gym_id: uuid.UUID
    start_time: datetime
    end_time: datetime


class GymBooking(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID
    gym_id: uuid.UUID
    start_time: datetime
    end_time: datetime
    status: str
    created_at: datetime

    class Config:
        from_attributes = True


# --- Achievement Schemas ---
class UserAchievement(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID
    achievement_id: str
    unlocked_at: datetime

    class Config:
        from_attributes = True


class AchievementUnlock(BaseModel):
    achievement_id: str


# --- Leaderboard Schemas ---
class LeaderboardEntry(BaseModel):
    username: str
    total_points: int
    current_level: int
    rank: int


class LeaderboardResponse(BaseModel):
    gym_id: Optional[uuid.UUID]
    gym_name: Optional[str]
    leaderboard: List[LeaderboardEntry]
    total_members: int
    current_user_rank: Optional[int]


# --- Dashboard Schemas ---
class DashboardResponse(BaseModel):
    total_workouts: int
    total_workout_time: int
    total_calories_burned: float
    current_streak: int
    longest_streak: int
    achievements_unlocked: int
    current_points: int
    current_level: int


# --- Analytics Schemas ---
class AnalyticsResponse(BaseModel):
    period: str
    workout_count: int
    total_duration: int
    total_calories: float
    avg_calories_per_workout: float
    workout_dates: List[date]


# --- General Response Schemas ---
class MessageResponse(BaseModel):
    message: str


class JoinByCodeRequest(BaseModel):
    gym_code: str


# ========================================
# 🆕 NEW BUSINESS SCHEMAS
# ========================================

# --- Gym Owner Schemas ---
class GymOwnerCreate(BaseModel):
    email: EmailStr
    password: str
    full_name: str
    phone_number: Optional[str] = None
    gym_id: uuid.UUID


class GymOwnerLogin(BaseModel):
    email: EmailStr
    password: str


class GymOwnerRead(BaseModel):
    id: uuid.UUID
    email: str
    full_name: str
    phone_number: Optional[str]
    gym_id: uuid.UUID
    is_active: bool
    created_at: datetime
    last_login: Optional[datetime]

    class Config:
        from_attributes = True


class GymOwnerUpdate(BaseModel):
    full_name: Optional[str] = None
    phone_number: Optional[str] = None


# --- Membership Fee Schemas ---
class MembershipFeeCreate(BaseModel):
    user_id: uuid.UUID
    amount: float
    due_date: datetime
    notes: Optional[str] = None


class MembershipFeeUpdate(BaseModel):
    status: Optional[str] = None  # PAID, CANCELLED
    payment_method: Optional[str] = None
    paid_date: Optional[datetime] = None
    receipt_number: Optional[str] = None
    notes: Optional[str] = None


class MembershipFee(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID
    gym_id: uuid.UUID
    amount: float
    currency: str
    payment_date: datetime
    due_date: datetime
    paid_date: Optional[datetime]
    status: str
    payment_method: Optional[str]
    receipt_number: Optional[str]
    notes: Optional[str]
    created_by: uuid.UUID
    created_at: datetime
    updated_at: Optional[datetime]

    class Config:
        from_attributes = True


class BulkFeeCreate(BaseModel):
    user_ids: List[uuid.UUID]
    amount: float
    due_date: datetime


# --- QR Code Schemas ---
class QRCodeGenerate(BaseModel):
    user_id: uuid.UUID


class QRCodeResponse(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID
    gym_id: uuid.UUID
    qr_code_data: str
    qr_code_image_base64: str
    is_active: bool
    created_at: datetime
    expires_at: Optional[datetime]

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
    check_out_time: Optional[datetime]
    duration_minutes: Optional[int]
    qr_code_used: str
    created_at: datetime

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
    gym_id: uuid.UUID
    notification_type: str
    title: str
    message: str
    is_read: bool
    sent_via_email: bool
    sent_via_app: bool
    created_at: datetime
    read_at: Optional[datetime]

    class Config:
        from_attributes = True


class NotificationPreferences(BaseModel):
    email_enabled: bool
    app_enabled: bool
    fee_reminder_days: List[int]


# --- Member Performance Schemas ---
class PerformanceAnalysis(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID
    gym_id: uuid.UUID
    analysis_date: datetime
    total_workouts: int
    avg_workout_duration: float
    consistency_score: float
    performance_trend: str
    weak_areas: List[str]
    suggestions: List[str]
    created_at: datetime

    class Config:
        from_attributes = True


# --- Analytics Schemas ---
class RevenueReport(BaseModel):
    total_revenue: float
    pending_revenue: float
    collected_revenue: float
    overdue_count: int
    period_start: date
    period_end: date


class AttendanceTrends(BaseModel):
    date: date
    total_check_ins: int
    avg_duration_minutes: float
    peak_hour: int


class MemberStats(BaseModel):
    user_id: uuid.UUID
    username: str
    full_name: str
    total_workouts: int
    avg_duration: float
    consistency_score: float
    membership_status: str
    last_payment_date: Optional[datetime]
    next_due_date: Optional[datetime]


class AnalyticsDashboard(BaseModel):
    total_members: int
    active_members: int
    revenue_this_month: float
    avg_attendance_per_day: float
    top_performers: List[MemberStats]
    recent_activity: List[AttendanceRecord]
