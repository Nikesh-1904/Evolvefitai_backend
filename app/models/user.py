import uuid
from typing import List, Optional
from datetime import datetime
from sqlalchemy import (
    Integer, String, Boolean, DateTime, Float, ForeignKey, JSON
)
from sqlalchemy.orm import relationship, Mapped, mapped_column
from sqlalchemy.dialects.postgresql import UUID as pgUUID
from fastapi_users.db import (
    SQLAlchemyBaseUserTableUUID,
    SQLAlchemyBaseOAuthAccountTableUUID,
)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .gym import Gym, GymBooking, UserQRCode, GymAttendance
    from .workout import WorkoutPlan, WorkoutLog
    from .mealplan import MealPlan
    from .achievement import UserAchievement
    from .fees import MembershipFee
    from .notification import Notification
    from .analytics import MemberPerformance
from .base import Base
    
class User(SQLAlchemyBaseUserTableUUID, Base):    
    """User model with FastAPI Users integration"""
    __tablename__ = "user"

    username: Mapped[str] = mapped_column(String, unique=True, index=True, nullable=True)
    full_name: Mapped[str] = mapped_column(String, nullable=True)
    age: Mapped[int] = mapped_column(Integer, nullable=True)
    weight: Mapped[float] = mapped_column(Float, nullable=True)
    height: Mapped[float] = mapped_column(Float, nullable=True)
    gender: Mapped[str] = mapped_column(String, nullable=True)
    fitness_level: Mapped[str] = mapped_column(String, nullable=True)
    fitness_goal: Mapped[str] = mapped_column(String, nullable=True)
    activity_level: Mapped[str] = mapped_column(String, nullable=True)
    dietary_preference: Mapped[str] = mapped_column(String, nullable=True)

    # Onboarding
    has_completed_onboarding: Mapped[bool] = mapped_column(Boolean, default=False)
    preferences: Mapped[dict] = mapped_column(JSON, nullable=True)

    # Gamification
    total_points: Mapped[int] = mapped_column(Integer, default=0)
    current_level: Mapped[int] = mapped_column(Integer, default=1)

    # Gym affiliation
    gym_id: Mapped[uuid.UUID] = mapped_column(
        pgUUID(as_uuid=True), ForeignKey("gyms.id"), nullable=True
    )

    # 🆕 BUSINESS FEATURES - QR & Membership
    qr_code_id: Mapped[uuid.UUID] = mapped_column(
        pgUUID(as_uuid=True), ForeignKey("user_qr_codes.id"), nullable=True
    )
    membership_status: Mapped[str] = mapped_column(
        String, default="ACTIVE"  # ACTIVE, EXPIRED, SUSPENDED
    )
    membership_expiry: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    
    # 🆕 Notification preferences
    notification_preferences: Mapped[dict] = mapped_column(
        JSON, 
        default=lambda: {
            "email_enabled": True,
            "app_enabled": True,
            "fee_reminder_days": [7, 3, 1]  # Days before due date
        }
    )

    # Relationships
    gym: Mapped["Gym"] = relationship("Gym", back_populates="members", foreign_keys=[gym_id])
    workout_plans: Mapped[List["WorkoutPlan"]] = relationship(
        "WorkoutPlan", back_populates="user", cascade="all, delete-orphan"
    )
    workout_logs: Mapped[List["WorkoutLog"]] = relationship(
        "WorkoutLog", back_populates="user", cascade="all, delete-orphan"
    )
    meal_plans: Mapped[List["MealPlan"]] = relationship(
        "MealPlan", back_populates="user", cascade="all, delete-orphan"
    )
    achievements: Mapped[List["UserAchievement"]] = relationship(
        "UserAchievement", back_populates="user", cascade="all, delete-orphan"
    )
    bookings: Mapped[List["GymBooking"]] = relationship(
        "GymBooking", back_populates="user", cascade="all, delete-orphan"
    )

    # 🆕 BUSINESS RELATIONSHIPS
    qr_code: Mapped[Optional["UserQRCode"]] = relationship(
        "UserQRCode", back_populates="user", foreign_keys="UserQRCode.user_id", uselist=False
    )
    fee_records: Mapped[List["MembershipFee"]] = relationship(
        "MembershipFee", back_populates="user", cascade="all, delete-orphan"
    )
    attendance_records: Mapped[List["GymAttendance"]] = relationship(
        "GymAttendance", back_populates="user", cascade="all, delete-orphan"
    )
    notifications: Mapped[List["Notification"]] = relationship(
        "Notification", back_populates="user", cascade="all, delete-orphan"
    )
    performance_records: Mapped[List["MemberPerformance"]] = relationship(
        "MemberPerformance", back_populates="user", cascade="all, delete-orphan"
    )
    oauth_accounts: Mapped[List["OAuthAccount"]] = relationship(
        "OAuthAccount", back_populates="user", cascade="all, delete-orphan"
    )


class OAuthAccount(SQLAlchemyBaseOAuthAccountTableUUID, Base):
    """OAuth account for social login"""
    __tablename__ = "oauth_account"

    user_id: Mapped[uuid.UUID] = mapped_column(
        pgUUID(as_uuid=True), ForeignKey("user.id"), nullable=False
    )

    user: Mapped["User"] = relationship("User", back_populates="oauth_accounts")
