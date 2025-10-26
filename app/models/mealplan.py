import uuid
from datetime import datetime
from sqlalchemy import (
    String, DateTime, Float, ForeignKey, JSON
)
from sqlalchemy.orm import relationship, Mapped, mapped_column
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import UUID as pgUUID

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .user import User
from .base import Base

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

