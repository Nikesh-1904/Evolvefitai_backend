# app/schemas/__init__.py

# User Schemas
from .user import UserRead, UserCreate, UserUpdate

# Achievement Schemas
from .achievement import (
    UserAchievementBase, UserAchievement, AchievementUnlockRequest,
    AchievementStatus
)

# Workout Schemas
from .workout import (
    ExerciseType, LoggedSetWeightBased, LoggedSetRepsOnly, LoggedSetDuration,
    LoggedSetDistanceDuration, LoggedSetQualitative, AnyLoggedSet,
    LoggedExercise, ExerciseBase, ExerciseCreate, Exercise, WorkoutPlanBase,
    WorkoutPlanCreate, WorkoutPlan, WorkoutLogBase, WorkoutLogCreate, WorkoutLog
)

# Meal Plan Schemas
from .mealplan import MealPlanBase, MealPlanCreate, MealPlan

# Gym & Booking Schemas
from .gym import (
    GymBase, GymCreate, Gym, JoinByCodeRequest, GymBookingBase,
    GymBookingCreate, GymBooking, GymOccupancyResponse, LeaderboardEntry,
    LeaderboardResponse
)

# Interaction Schemas (Tips, Videos, Progression)
from .interaction import (
    ExerciseTipBase, ExerciseTip, ExerciseVideoBase, ExerciseVideo,
    TipInteractionCreate, VideoPreferenceCreate, ExerciseSetData,
    ExerciseProgressionDataPoint
)

# AI Schemas
from .ai import (
    WorkoutGenerationRequest, PlateauAnalysis, MealPlanRequest,
    GeneratedMealPlan
)

# Miscellaneous Schemas
from .misc import MessageResponse, METValueResponse, LevelProgress, TimeSeriesDataPoint

# --- Business Schemas ---
from .business_owner import (
    GymOwnerBase, GymOwnerCreate, GymOwnerRead, GymOwnerLogin, GymOwnerUpdate
)
from .business_qr import QRCodeResponse
from .business_fees import (
    MembershipFeeBase, MembershipFeeCreate, MembershipFeeUpdate,
    MembershipFeeRead, BulkFeeCreate
)
from .business_attendance import (
    CheckInRequest, CheckOutRequest, AttendanceRecord, LiveOccupancyResponse
)
from .business_notification import (
    NotificationCreate, NotificationRead, NotificationPreferences
)
from .business_analytics import (
    MemberStats, AnalyticsDashboard, RevenueReport, PerformanceAnalysis
)

from .stats import DashboardOverviewStats, AnalyticsData
# Handle alias if needed (optional, could also just use MembershipFeeRead everywhere)
MembershipFee = MembershipFeeRead

# Define __all__ to export everything and satisfy linters
__all__ = [
    # User
    "UserRead", "UserCreate", "UserUpdate",
    # Achievement
    "UserAchievementBase", "UserAchievement", "AchievementUnlockRequest",
    "AchievementStatus",
    # Workout
    "ExerciseType", "LoggedSetWeightBased", "LoggedSetRepsOnly", "LoggedSetDuration",
    "LoggedSetDistanceDuration", "LoggedSetQualitative", "AnyLoggedSet",
    "LoggedExercise", "ExerciseBase", "ExerciseCreate", "Exercise", "WorkoutPlanBase",
    "WorkoutPlanCreate", "WorkoutPlan", "WorkoutLogBase", "WorkoutLogCreate", "WorkoutLog",
    # Meal Plan
    "MealPlanBase", "MealPlanCreate", "MealPlan",
    # Gym & Booking
    "GymBase", "GymCreate", "Gym", "JoinByCodeRequest", "GymBookingBase",
    "GymBookingCreate", "GymBooking", "GymOccupancyResponse", "LeaderboardEntry",
    "LeaderboardResponse", "LiveOccupancyResponse", # Added LiveOccupancyResponse here from business_attendance for convenience
    # Interaction
    "ExerciseTipBase", "ExerciseTip", "ExerciseVideoBase", "ExerciseVideo",
    "TipInteractionCreate", "VideoPreferenceCreate", "ExerciseSetData",
    "ExerciseProgressionDataPoint",
    # AI
    "WorkoutGenerationRequest", "PlateauAnalysis", "MealPlanRequest",
    "GeneratedMealPlan",
    # Misc
    "MessageResponse", "METValueResponse", "LevelProgress", "TimeSeriesDataPoint",
    # Business Owner
    "GymOwnerBase", "GymOwnerCreate", "GymOwnerRead", "GymOwnerLogin", "GymOwnerUpdate",
    # Business QR
    "QRCodeResponse",
    # Business Fees
    "MembershipFeeBase", "MembershipFeeCreate", "MembershipFeeUpdate",
    "MembershipFeeRead", "MembershipFee", "BulkFeeCreate", # Added alias MembershipFee
    # Business Attendance
    "CheckInRequest", "CheckOutRequest", "AttendanceRecord",
    # Business Notification
    "NotificationCreate", "NotificationRead", "NotificationPreferences",
    # Business Analytics
    "MemberStats", "AnalyticsDashboard", "RevenueReport", "PerformanceAnalysis",
    "DashboardOverviewStats", "AnalyticsData",
]