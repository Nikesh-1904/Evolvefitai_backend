# app/schemas/mealplan.py
from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import datetime
import uuid

class MealPlanBase(BaseModel):
    name: str
    target_calories: int
    target_protein: float
    target_carbs: float
    target_fat: float
    meals: Dict[str, Any] = {}


class MealPlanCreate(MealPlanBase):
    pass


class MealPlan(MealPlanBase):
    id: uuid.UUID
    user_id: uuid.UUID
    ai_generated: bool
    ai_model: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True
