import uuid
from datetime import datetime
from sqlalchemy import (
    String, Text, DateTime, Float, ForeignKey
)
from sqlalchemy.orm import relationship, Mapped, mapped_column
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import UUID as pgUUID

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .user import User
    from .gym import Gym, GymOwner
from .base import Base

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
    user: Mapped["User"] = relationship("User", back_populates="fee_records")
    gym: Mapped["Gym"] = relationship("Gym", back_populates="fee_records")
    created_by_owner: Mapped["GymOwner"] = relationship(
        "GymOwner", back_populates="created_fees", foreign_keys=[created_by]
    )
