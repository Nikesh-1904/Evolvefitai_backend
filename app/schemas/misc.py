# app/schemas/misc.py
from pydantic import BaseModel
from datetime import date

class METValueResponse(BaseModel):
    exercise_name: str
    met_value: float


# --- Response Schemas ---

class MessageResponse(BaseModel):
    message: str
    success: bool = True
    
class LevelProgress(BaseModel):
    current_level: int
    current_points: int
    points_for_current_level: int
    points_for_next_level: int

class TimeSeriesDataPoint(BaseModel):
    date: date
    value: float