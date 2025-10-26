# app/schemas/stats.py
from pydantic import BaseModel
from typing import List
from datetime import date

# Import dependent schemas from their new locations
from .misc import LevelProgress, TimeSeriesDataPoint

class DashboardOverviewStats(BaseModel):
    workouts_completed: int
    total_workout_time_hours: float
    total_calories_burned: int
    level_progress: LevelProgress
    calories_change_percent: float
    time_change_percent: float
class AnalyticsData(BaseModel):
    calorie_timeseries: List[TimeSeriesDataPoint]
    workout_heatmap: List[date]

