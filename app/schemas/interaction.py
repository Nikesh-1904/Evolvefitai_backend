# app/schemas/interaction.py
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime, date

class ExerciseTipBase(BaseModel):
    title: str
    content: str
    tip_type: Optional[str] = None


class ExerciseTip(ExerciseTipBase):
    id: int
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
    id: int
    exercise_id: int
    popularity_score: float
    created_at: datetime

    class Config:
        from_attributes = True

class TipInteractionCreate(BaseModel):
    tip_id: int
    interaction_type: str


class VideoPreferenceCreate(BaseModel):
    video_id: int
    preference: str

class ExerciseSetData(BaseModel):
    reps: int
    weight: float


class ExerciseProgressionDataPoint(BaseModel):
    workout_date: date
    primary_metric_value: float
    metric_type: str
    sets: List[Dict[str, Any]]
