# app/schemas/ai.py
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from .mealplan import MealPlanBase

class WorkoutGenerationRequest(BaseModel):
    user_preferences: Optional[Dict[str, Any]] = {}
    duration_minutes: Optional[int] = 45
    target_muscle_groups: Optional[List[str]] = None
    num_exercises: Optional[int] = None
    workout_type: Optional[str] = None


class PlateauAnalysis(BaseModel):
    is_plateau: bool
    confidence: float
    affected_exercises: List[str]
    recommendations: List[str]
    plateau_duration_weeks: int
    analysis_method: str
    ai_generated: bool = False


class MealPlanRequest(BaseModel):
    duration_days: int = 7
    preferences: Optional[Dict[str, Any]] = {}

class GeneratedMealPlan(MealPlanBase):
    """Response schema for AI-generated meal plans that aren't stored in DB yet"""
    ai_generated: bool = True
    ai_model: Optional[str] = None