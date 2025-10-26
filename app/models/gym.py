import uuid
from typing import List
from datetime import datetime
from sqlalchemy import (
    Integer, String, Text, Boolean, DateTime, Float, ForeignKey
)
from sqlalchemy.orm import relationship, Mapped, mapped_column
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import UUID as pgUUID

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .user import User
    from .fees import MembershipFee
    from .notification import Notification
    from .analytics import MemberPerformance
from .base import Base

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

