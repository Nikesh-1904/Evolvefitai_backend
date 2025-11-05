import uuid
from typing import List, Optional
from datetime import datetime
from sqlalchemy import (
    Integer, String, Boolean, DateTime, Float, ForeignKey, JSON
)
from sqlalchemy.orm import relationship, Mapped, mapped_column
from sqlalchemy.dialects.postgresql import UUID as pgUUID
from sqlalchemy.sql import func
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
    
    # ✅ FIX: Add property for backward compatibility with 'level'
    @property
    def level(self) -> int:
        """Alias for current_level for backward compatibility"""
        return self.current_level
    
    @level.setter
    def level(self, value: int):
        """Setter for level property"""
        self.current_level = value

    # ✅ FIX: Add created_at timestamp
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Gym affiliation
    gym_id: Mapped[uuid.UUID] = mapped_column(
        pgUUID(as_uuid=True), ForeignKey("gyms.id"), nullable=True
    )
    
    membership_status: Mapped[str] = mapped_column(
        String, default="ACTIVE"  # ACTIVE, EXPIRED, SUSPENDED
    )
    membership_expiry: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    
    # ✅ FIX: Changed default to use server_default for mutable dict
    notification_preferences: Mapped[dict] = mapped_column(
        JSON,
        nullable=False,
        server_default='{"email_enabled": true, "app_enabled": true, "fee_reminder_days": [7, 3, 1]}'
    )

    # ✅ FIX: Relationships with lazy='selectin' for async compatibility
    gym: Mapped["Gym"] = relationship(
        "Gym", 
        back_populates="members", 
        foreign_keys=[gym_id],
        lazy="selectin"
    )
    workout_plans: Mapped[List["WorkoutPlan"]] = relationship(
        "WorkoutPlan", 
        back_populates="user", 
        cascade="all, delete-orphan",
        lazy="selectin"
    )
    workout_logs: Mapped[List["WorkoutLog"]] = relationship(
        "WorkoutLog", 
        back_populates="user", 
        cascade="all, delete-orphan",
        lazy="selectin"
    )
    meal_plans: Mapped[List["MealPlan"]] = relationship(
        "MealPlan", 
        back_populates="user", 
        cascade="all, delete-orphan",
        lazy="selectin"
    )
    achievements: Mapped[List["UserAchievement"]] = relationship(
        "UserAchievement", 
        back_populates="user", 
        cascade="all, delete-orphan",
        lazy="selectin"  # ✅ FIX: Changed from default to selectin
    )
    bookings: Mapped[List["GymBooking"]] = relationship(
        "GymBooking", 
        back_populates="user", 
        cascade="all, delete-orphan",
        lazy="selectin"
    )
    qr_codes: Mapped[Optional["UserQRCode"]] = relationship(
        "UserQRCode", 
        back_populates="user", 
        cascade="all, delete-orphan",
        lazy="selectin"
    )
    fee_records: Mapped[List["MembershipFee"]] = relationship(
        "MembershipFee", 
        back_populates="user", 
        cascade="all, delete-orphan",
        lazy="selectin"
    )
    attendance_records: Mapped[List["GymAttendance"]] = relationship(
        "GymAttendance", 
        back_populates="user", 
        cascade="all, delete-orphan",
        lazy="selectin"
    )
    notifications: Mapped[List["Notification"]] = relationship(
        "Notification", 
        back_populates="user", 
        cascade="all, delete-orphan",
        lazy="selectin"
    )
    performance_records: Mapped[List["MemberPerformance"]] = relationship(
        "MemberPerformance", 
        back_populates="user", 
        cascade="all, delete-orphan",
        lazy="selectin"
    )
    oauth_accounts: Mapped[List["OAuthAccount"]] = relationship(
        "OAuthAccount", 
        back_populates="user", 
        cascade="all, delete-orphan",
        lazy="selectin"
    )


class OAuthAccount(SQLAlchemyBaseOAuthAccountTableUUID, Base):
    """OAuth account for social login"""
    __tablename__ = "oauth_account"

    user_id: Mapped[uuid.UUID] = mapped_column(
        pgUUID(as_uuid=True), ForeignKey("user.id"), nullable=False
    )

    user: Mapped["User"] = relationship("User", back_populates="oauth_accounts")