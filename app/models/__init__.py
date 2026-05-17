# app/models/__init__.py
from .base import Base
from .user import User, OAuthAccount
from .gym import Gym, GymOwner, GymAttendance, UserQRCode, GymBooking
from .workout import WorkoutPlan, WorkoutLog, Exercise, ExerciseVideo, ExerciseTip
from .fees import MembershipFee
from .analytics import MemberPerformance
from .notification import Notification
from .achievement import UserAchievement

__all__ = [
    "Base",
    "User",
    "OAuthAccount",
    "Gym",
    "GymOwner",
    "GymAttendance",
    "UserQRCode",
    "GymBooking",
    "WorkoutPlan",
    "WorkoutLog",
    "Exercise",
    "ExerciseVideo",
    "ExerciseTip",
    "MembershipFee",
    "MemberPerformance",
    "Notification",
    "UserAchievement",
]