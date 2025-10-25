# app/models.py
import uuid
from typing import List, Optional
from datetime import datetime
from sqlalchemy import (
    Integer,
    String,
    Text,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    JSON,
)
from sqlalchemy.orm import relationship, Mapped, mapped_column
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import UUID as pgUUID
from app.core.database import Base
from fastapi_users.db import (
    SQLAlchemyBaseUserTableUUID,
    SQLAlchemyBaseOAuthAccountTableUUID,
)


class User(SQLAlchemyBaseUserTableUUID, Base):
    """User model with FastAPI Users integration"""
    __tablename__ = "users"

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


class OAuthAccount(SQLAlchemyBaseOAuthAccountTableUUID, Base):
    """OAuth account for social login"""
    pass


class Gym(Base):
    """Gym/fitness center model"""
    __tablename__ = "gyms"

    id: Mapped[uuid.UUID] = mapped_column(
        pgUUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=True)
    address: Mapped[str] = mapped_column(String, nullable=True)
    city: Mapped[str] = mapped_column(String, nullable=True)
    state: Mapped[str] = mapped_column(String, nullable=True)
    country: Mapped[str] = mapped_column(String, default="India")
    pincode: Mapped[str] = mapped_column(String, nullable=True)
    capacity: Mapped[int] = mapped_column(Integer, default=50)
    
    # Gym code for joining
    gym_code: Mapped[str] = mapped_column(String, unique=True, index=True)
    
    # 🆕 BUSINESS FEATURES - Fee management
    monthly_fee: Mapped[float] = mapped_column(Float, nullable=True)
    currency: Mapped[str] = mapped_column(String, default="INR")
    fee_due_day: Mapped[int] = mapped_column(Integer, default=1)  # Day of month (1-31)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Relationships
    members: Mapped[List["User"]] = relationship(
        "User", back_populates="gym", foreign_keys="User.gym_id"
    )
    bookings: Mapped[List["GymBooking"]] = relationship(
        "GymBooking", back_populates="gym", cascade="all, delete-orphan"
    )

    # 🆕 BUSINESS RELATIONSHIPS
    owners: Mapped[List["GymOwner"]] = relationship(
        "GymOwner", back_populates="gym", cascade="all, delete-orphan"
    )
    fee_records: Mapped[List["MembershipFee"]] = relationship(
        "MembershipFee", back_populates="gym", cascade="all, delete-orphan"
    )
    attendance_records: Mapped[List["GymAttendance"]] = relationship(
        "GymAttendance", back_populates="gym", cascade="all, delete-orphan"
    )
    qr_codes: Mapped[List["UserQRCode"]] = relationship(
        "UserQRCode", back_populates="gym", cascade="all, delete-orphan"
    )
    notifications: Mapped[List["Notification"]] = relationship(
        "Notification", back_populates="gym", cascade="all, delete-orphan"
    )
    performance_records: Mapped[List["MemberPerformance"]] = relationship(
        "MemberPerformance", back_populates="gym", cascade="all, delete-orphan"
    )


# 🆕 NEW MODEL: Gym Owner
class GymOwner(Base):
    """Gym owner/administrator accounts"""
    __tablename__ = "gym_owners"

    id: Mapped[uuid.UUID] = mapped_column(
        pgUUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    email: Mapped[str] = mapped_column(String, unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String, nullable=False)
    full_name: Mapped[str] = mapped_column(String, nullable=False)
    phone_number: Mapped[str] = mapped_column(String, nullable=True)
    
    gym_id: Mapped[uuid.UUID] = mapped_column(
        pgUUID(as_uuid=True), ForeignKey("gyms.id"), nullable=False
    )
    
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    last_login: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Relationships
    gym: Mapped["Gym"] = relationship("Gym", back_populates="owners")
    created_fees: Mapped[List["MembershipFee"]] = relationship(
        "MembershipFee", back_populates="created_by_owner", foreign_keys="MembershipFee.created_by"
    )


# 🆕 NEW MODEL: Membership Fee
class MembershipFee(Base):
    """Fee records for gym members"""
    __tablename__ = "membership_fees"

    id: Mapped[uuid.UUID] = mapped_column(
        pgUUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        pgUUID(as_uuid=True), ForeignKey("users.id"), nullable=False
    )
    gym_id: Mapped[uuid.UUID] = mapped_column(
        pgUUID(as_uuid=True), ForeignKey("gyms.id"), nullable=False
    )
    
    amount: Mapped[float] = mapped_column(Float, nullable=False)
    currency: Mapped[str] = mapped_column(String, default="INR")
    
    # Payment tracking
    payment_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    due_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    paid_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)
    
    status: Mapped[str] = mapped_column(
        String, default="PENDING"  # PENDING, PAID, OVERDUE, CANCELLED
    )
    payment_method: Mapped[str] = mapped_column(
        String, nullable=True  # CASH, UPI, CARD, NET_BANKING, etc.
    )
    receipt_number: Mapped[str] = mapped_column(String, unique=True, nullable=True)
    notes: Mapped[str] = mapped_column(Text, nullable=True)
    
    created_by: Mapped[uuid.UUID] = mapped_column(
        pgUUID(as_uuid=True), ForeignKey("gym_owners.id"), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), onupdate=func.now(), nullable=True
    )

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="fee_records")
    gym: Mapped["Gym"] = relationship("Gym", back_populates="fee_records")
    created_by_owner: Mapped["GymOwner"] = relationship(
        "GymOwner", back_populates="created_fees", foreign_keys=[created_by]
    )


# 🆕 NEW MODEL: Gym Attendance (QR Check-in/out)
class GymAttendance(Base):
    """QR-based attendance tracking"""
    __tablename__ = "gym_attendance"

    id: Mapped[uuid.UUID] = mapped_column(
        pgUUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        pgUUID(as_uuid=True), ForeignKey("users.id"), nullable=False
    )
    gym_id: Mapped[uuid.UUID] = mapped_column(
        pgUUID(as_uuid=True), ForeignKey("gyms.id"), nullable=False
    )
    
    check_in_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    check_out_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    duration_minutes: Mapped[int] = mapped_column(Integer, nullable=True)
    
    qr_code_used: Mapped[str] = mapped_column(String, nullable=False)
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="attendance_records")
    gym: Mapped["Gym"] = relationship("Gym", back_populates="attendance_records")


# 🆕 NEW MODEL: User QR Code
class UserQRCode(Base):
    """Unique QR codes for gym members"""
    __tablename__ = "user_qr_codes"

    id: Mapped[uuid.UUID] = mapped_column(
        pgUUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        pgUUID(as_uuid=True), ForeignKey("users.id"), nullable=False, unique=True
    )
    gym_id: Mapped[uuid.UUID] = mapped_column(
        pgUUID(as_uuid=True), ForeignKey("gyms.id"), nullable=False
    )
    
    qr_code_data: Mapped[str] = mapped_column(Text, nullable=False)  # Encrypted payload
    qr_code_image_base64: Mapped[str] = mapped_column(Text, nullable=False)  # Base64 image
    
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)

    # Relationships
    user: Mapped["User"] = relationship(
        "User", back_populates="qr_code", foreign_keys=[user_id]
    )
    gym: Mapped["Gym"] = relationship("Gym", back_populates="qr_codes")


# 🆕 NEW MODEL: Notification
class Notification(Base):
    """In-app and email notifications"""
    __tablename__ = "notifications"

    id: Mapped[uuid.UUID] = mapped_column(
        pgUUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        pgUUID(as_uuid=True), ForeignKey("users.id"), nullable=False
    )
    gym_id: Mapped[uuid.UUID] = mapped_column(
        pgUUID(as_uuid=True), ForeignKey("gyms.id"), nullable=False
    )
    
    notification_type: Mapped[str] = mapped_column(
        String, nullable=False  # FEE_REMINDER, FEE_OVERDUE, GENERAL, ANNOUNCEMENT
    )
    title: Mapped[str] = mapped_column(String, nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    
    is_read: Mapped[bool] = mapped_column(Boolean, default=False)
    sent_via_email: Mapped[bool] = mapped_column(Boolean, default=False)
    sent_via_app: Mapped[bool] = mapped_column(Boolean, default=True)
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    read_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="notifications")
    gym: Mapped["Gym"] = relationship("Gym", back_populates="notifications")


# 🆕 NEW MODEL: Member Performance
class MemberPerformance(Base):
    """AI-analyzed member performance metrics"""
    __tablename__ = "member_performance"

    id: Mapped[uuid.UUID] = mapped_column(
        pgUUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        pgUUID(as_uuid=True), ForeignKey("users.id"), nullable=False
    )
    gym_id: Mapped[uuid.UUID] = mapped_column(
        pgUUID(as_uuid=True), ForeignKey("gyms.id"), nullable=False
    )
    
    analysis_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    
    # Metrics
    total_workouts: Mapped[int] = mapped_column(Integer, default=0)
    avg_workout_duration: Mapped[float] = mapped_column(Float, default=0.0)
    consistency_score: Mapped[float] = mapped_column(Float, default=0.0)  # 0-100
    
    performance_trend: Mapped[str] = mapped_column(
        String, default="STABLE"  # IMPROVING, STABLE, DECLINING
    )
    
    weak_areas: Mapped[dict] = mapped_column(JSON, default=list)  # List of muscle groups
    suggestions: Mapped[dict] = mapped_column(JSON, default=list)  # AI recommendations
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="performance_records")
    gym: Mapped["Gym"] = relationship("Gym", back_populates="performance_records")


# ⚪ EXISTING MODELS (Unchanged)
class WorkoutPlan(Base):
    """AI-generated workout plan"""
    __tablename__ = "workout_plans"

    id: Mapped[uuid.UUID] = mapped_column(
        pgUUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        pgUUID(as_uuid=True), ForeignKey("users.id")
    )
    name: Mapped[str] = mapped_column(String)
    description: Mapped[str] = mapped_column(Text, nullable=True)
    difficulty: Mapped[str] = mapped_column(String, nullable=True)
    duration_minutes: Mapped[int] = mapped_column(Integer)
    exercises: Mapped[dict] = mapped_column(JSON)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    user: Mapped["User"] = relationship("User", back_populates="workout_plans")


class WorkoutLog(Base):
    """Logged/completed workout with exercises"""
    __tablename__ = "workout_logs"

    id: Mapped[uuid.UUID] = mapped_column(
        pgUUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        pgUUID(as_uuid=True), ForeignKey("users.id")
    )
    workout_plan_id: Mapped[uuid.UUID] = mapped_column(
        pgUUID(as_uuid=True), ForeignKey("workout_plans.id"), nullable=True
    )
    name: Mapped[str] = mapped_column(String)
    duration_minutes: Mapped[int] = mapped_column(Integer)
    calories_burned: Mapped[float] = mapped_column(Float, nullable=True)
    exercises_completed: Mapped[dict] = mapped_column(JSON)
    notes: Mapped[str] = mapped_column(Text, nullable=True)
    logged_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    user: Mapped["User"] = relationship("User", back_populates="workout_logs")


class Exercise(Base):
    """Exercise library"""
    __tablename__ = "exercises"

    id: Mapped[uuid.UUID] = mapped_column(
        pgUUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String, unique=True, index=True)
    exercise_type: Mapped[str] = mapped_column(String)
    instructions: Mapped[str] = mapped_column(Text, nullable=True)
    muscle_groups: Mapped[dict] = mapped_column(JSON)
    equipment: Mapped[str] = mapped_column(String, nullable=True)
    difficulty: Mapped[str] = mapped_column(String, nullable=True)
    met_value: Mapped[float] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    videos: Mapped[List["ExerciseVideo"]] = relationship(
        "ExerciseVideo", back_populates="exercise", cascade="all, delete-orphan"
    )
    tips: Mapped[List["ExerciseTip"]] = relationship(
        "ExerciseTip", back_populates="exercise", cascade="all, delete-orphan"
    )


class ExerciseVideo(Base):
    """YouTube videos for exercises"""
    __tablename__ = "exercise_videos"

    id: Mapped[uuid.UUID] = mapped_column(
        pgUUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    exercise_id: Mapped[uuid.UUID] = mapped_column(
        pgUUID(as_uuid=True), ForeignKey("exercises.id")
    )
    video_id: Mapped[str] = mapped_column(String)
    title: Mapped[str] = mapped_column(String)
    thumbnail_url: Mapped[str] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    exercise: Mapped["Exercise"] = relationship("Exercise", back_populates="videos")


class ExerciseTip(Base):
    """Tips for exercises"""
    __tablename__ = "exercise_tips"

    id: Mapped[uuid.UUID] = mapped_column(
        pgUUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    exercise_id: Mapped[uuid.UUID] = mapped_column(
        pgUUID(as_uuid=True), ForeignKey("exercises.id")
    )
    tip: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    exercise: Mapped["Exercise"] = relationship("Exercise", back_populates="tips")


class MealPlan(Base):
    """Meal plan for user"""
    __tablename__ = "meal_plans"

    id: Mapped[uuid.UUID] = mapped_column(
        pgUUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        pgUUID(as_uuid=True), ForeignKey("users.id")
    )
    name: Mapped[str] = mapped_column(String)
    target_calories: Mapped[float] = mapped_column(Float)
    target_protein: Mapped[float] = mapped_column(Float, nullable=True)
    target_carbs: Mapped[float] = mapped_column(Float, nullable=True)
    target_fat: Mapped[float] = mapped_column(Float, nullable=True)
    meals: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    user: Mapped["User"] = relationship("User", back_populates="meal_plans")


class GymBooking(Base):
    """Gym slot booking"""
    __tablename__ = "gym_bookings"

    id: Mapped[uuid.UUID] = mapped_column(
        pgUUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        pgUUID(as_uuid=True), ForeignKey("users.id")
    )
    gym_id: Mapped[uuid.UUID] = mapped_column(
        pgUUID(as_uuid=True), ForeignKey("gyms.id")
    )
    start_time: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    end_time: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String, default="confirmed")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    user: Mapped["User"] = relationship("User", back_populates="bookings")
    gym: Mapped["Gym"] = relationship("Gym", back_populates="bookings")


class UserAchievement(Base):
    """User's unlocked achievements"""
    __tablename__ = "user_achievements"

    id: Mapped[uuid.UUID] = mapped_column(
        pgUUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        pgUUID(as_uuid=True), ForeignKey("users.id")
    )
    achievement_id: Mapped[str] = mapped_column(String)
    unlocked_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    user: Mapped["User"] = relationship("User", back_populates="achievements")
