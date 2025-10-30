# app/schemas/workout.py
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
import uuid
from enum import Enum

class ExerciseType(str, Enum):
    WEIGHT_BASED = "WEIGHT_BASED"
    REPS_ONLY = "REPS_ONLY"
    DURATION = "DURATION"
    DISTANCE_DURATION = "DISTANCE_DURATION"
    QUALITATIVE = "QUALITATIVE"


# 2. Define a schema for each type of logged data point ("set")
class LoggedSetWeightBased(BaseModel):
    reps: int
    weight: float


class LoggedSetRepsOnly(BaseModel):
    reps: int


class LoggedSetDuration(BaseModel):
    duration_seconds: int


class LoggedSetDistanceDuration(BaseModel):
    duration_seconds: int
    distance_km: float


class LoggedSetQualitative(BaseModel):
    duration_seconds: Optional[int] = None
    notes: Optional[str] = None


# 3. Create a Union type that can be any of the above schemas
AnyLoggedSet = Union[
    LoggedSetWeightBased,
    LoggedSetRepsOnly,
    LoggedSetDuration,
    LoggedSetDistanceDuration,
    LoggedSetQualitative,
]


# 4. Update LoggedExercise to use the new Union type and require the exercise_type
class LoggedExercise(BaseModel):
    name: str
    exercise_type: Optional[ExerciseType] = None
    sets: List[AnyLoggedSet]

class ExerciseBase(BaseModel):
    name: str
    exercise_type: Optional[ExerciseType] = ExerciseType.WEIGHT_BASED
    muscle_groups: Optional[List[str]] = []
    equipment: Optional[str] = None
    difficulty: Optional[str] = None
    instructions: Optional[str] = None


class ExerciseCreate(ExerciseBase):
    pass


class Exercise(ExerciseBase):
    model_config = ConfigDict(from_attributes=True)
    id: uuid.UUID
    created_at: datetime


class WorkoutPlanBase(BaseModel):
    name: str
    description: Optional[str] = None
    exercises: List[Dict[str, Any]] = []
    difficulty: Optional[str] = None
    estimated_duration: Optional[int] = None


class WorkoutPlanCreate(WorkoutPlanBase):
    pass


class WorkoutPlan(WorkoutPlanBase):
    model_config = ConfigDict(from_attributes=True)
    id: uuid.UUID
    user_id: uuid.UUID
    ai_generated: bool
    ai_model: Optional[str] = None
    is_active: bool
    created_at: datetime



# --- Workout Log Schemas ---

class WorkoutLogBase(BaseModel):
    duration_minutes: Optional[int] = None
    notes: Optional[str] = None
    exercises_completed: List[LoggedExercise] = []


class WorkoutLogCreate(WorkoutLogBase):
    workout_plan_id: Optional[int] = None
    workout_date: Optional[datetime] = None


class WorkoutLog(WorkoutLogBase):
    model_config = ConfigDict(from_attributes=True)
    id: uuid.UUID
    user_id: uuid.UUID
    workout_plan_id: Optional[int] = None
    workout_date: datetime

