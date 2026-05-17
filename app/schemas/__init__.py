# app/schemas/__init__.py

# User Schemas
from .user import UserRead, UserCreate, UserUpdate

# Achievement Schemas
from .achievement import (
    UserAchievementBase, UserAchievement, AchievementUnlockRequest,
    AchievementStatus,
)

# Workout Schemas
from .workout import (
    ExerciseType, LoggedSetWeightBased, LoggedSetRepsOnly, LoggedSetDuration,
    LoggedSetDistanceDuration, LoggedSetQualitative, AnyLoggedSet,
    LoggedExercise, ExerciseBase, ExerciseCreate, Exercise, WorkoutPlanBase,
    WorkoutPlanCreate, WorkoutPlan, WorkoutLogBase, WorkoutLogCreate, WorkoutLog,
)

# Gym Schemas (membership only)
from .gym import GymBase, GymCreate, Gym, JoinByCodeRequest

# Interaction Schemas (Tips, Videos, Progression)
from .interaction import (
    ExerciseTipBase, ExerciseTip, ExerciseVideoBase, ExerciseVideo,
    TipInteractionCreate, VideoPreferenceCreate, ExerciseSetData,
    ExerciseProgressionDataPoint,
)

# AI Schemas
from .ai import WorkoutGenerationRequest, PlateauAnalysis

# Miscellaneous Schemas
from .misc import MessageResponse, METValueResponse, LevelProgress, TimeSeriesDataPoint

# QR + notification preferences (still used by /users/me endpoints)
from .business_qr import QRCodeResponse
from .business_notification import NotificationPreferences

from .stats import DashboardOverviewStats, AnalyticsData

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
    # Gym
    "GymBase", "GymCreate", "Gym", "JoinByCodeRequest",
    # Interaction
    "ExerciseTipBase", "ExerciseTip", "ExerciseVideoBase", "ExerciseVideo",
    "TipInteractionCreate", "VideoPreferenceCreate", "ExerciseSetData",
    "ExerciseProgressionDataPoint",
    # AI
    "WorkoutGenerationRequest", "PlateauAnalysis",
    # Misc
    "MessageResponse", "METValueResponse", "LevelProgress", "TimeSeriesDataPoint",
    # QR + Notifications
    "QRCodeResponse", "NotificationPreferences",
    # Stats
    "DashboardOverviewStats", "AnalyticsData",
]
