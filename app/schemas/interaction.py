# app/schemas/interaction.py
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime, date
import uuid

class ExerciseTipBase(BaseModel):
    title: str
    content: str
    tip_type: Optional[str] = None


class ExerciseTip(ExerciseTipBase):
    id: uuid.UUID
    exercise_id: int
    popularity_score: float
    created_at: datetime

    class Config:
        from_attributes = True

class ExerciseVideoBase(BaseModel):
    youtube_url: str
    title: Optional[str] = None
    thumbnail_url: Optional[str] = None
    duration: Optional[int] = None


class ExerciseVideo(ExerciseVideoBase):
    id: uuid.UUID
    exercise_id: int
    popularity_score: float
    created_at: datetime

    class Config:
        from_attributes = True

class TipInteractionCreate(BaseModel):
    tip_id: uuid.UUID
    interaction_type: str


class VideoPreferenceCreate(BaseModel):
    video_id: uuid.UUID
    preference: str

class ExerciseSetData(BaseModel):
    reps: int
    weight: float


class ExerciseProgressionDataPoint(BaseModel):
    workout_date: date
    primary_metric_value: float
    metric_type: str
    sets: List[Dict[str, Any]]
