# app/api/v1/stats.py

import math
from datetime import date, timedelta
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List

from app.core.database import get_async_session
from app.core.auth import current_active_user
from app import models, schemas

router = APIRouter()

@router.get("/overview", response_model=schemas.DashboardOverviewStats)
async def get_dashboard_overview(
    current_user: models.User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session)
):
    """Get calculated overview statistics for the main dashboard cards and level progress."""
    
    query = (
        select(
            func.count(models.WorkoutLog.id).label("workouts_completed"),
            func.sum(models.WorkoutLog.duration_minutes).label("total_duration_minutes"),
            func.sum(models.WorkoutLog.calories_burned).label("total_calories_burned")
        )
        .where(models.WorkoutLog.user_id == current_user.id)
    )
    
    result = await session.execute(query)
    stats = result.first()

    total_calories_burned = stats.total_calories_burned or 0
    
    # --- LEVEL PROGRESS CALCULATION ---
    points = total_calories_burned / 2
    level = 1
    points_for_current_level = 0
    points_for_next_level = 100

    temp_points = points
    temp_threshold = 100
    while temp_points >= temp_threshold:
        level += 1
        points_for_current_level = temp_threshold
        temp_points -= temp_threshold
        temp_threshold *= 5
        points_for_next_level = temp_threshold

    level_progress_data = schemas.LevelProgress(
        current_level=level,
        current_points=int(points),
        points_for_current_level=points_for_current_level,
        points_for_next_level=points_for_next_level
    )

    return schemas.DashboardOverviewStats(
        workouts_completed=stats.workouts_completed or 0,
        total_workout_time_hours=round((stats.total_duration_minutes or 0) / 60, 1),
        total_calories_burned=int(total_calories_burned),
        level_progress=level_progress_data,
    )

# In stats.py
@router.get("/analytics", response_model=schemas.AnalyticsData)
async def get_analytics_data(
    aggregate_by: str = Query("day", enum=["day", "week", "month"]),
    current_user: models.User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session)
):
    """Get time-series data for analytics charts (heatmap and calorie graph)."""

    # --- HEATMAP DATA (Corrected) ---
    heatmap_query = select(func.distinct(func.date(models.WorkoutLog.workout_date))).where(
        models.WorkoutLog.user_id == current_user.id
    )
    heatmap_result = await session.execute(heatmap_query)
    workout_heatmap = heatmap_result.scalars().all()

    # --- CALORIE TIME-SERIES DATA ---
    calories_query = (
        select(
            func.date_trunc(aggregate_by, models.WorkoutLog.workout_date).label("date"),
            func.sum(models.WorkoutLog.calories_burned).label("value")
        )
        .where(models.WorkoutLog.user_id == current_user.id)
        .group_by(func.date_trunc(aggregate_by, models.WorkoutLog.workout_date))
        .order_by(func.date_trunc(aggregate_by, models.WorkoutLog.workout_date).asc())
    )
    calories_result = await session.execute(calories_query)
    calorie_timeseries = calories_result.all()

    return schemas.AnalyticsData(
        calorie_timeseries=[schemas.TimeSeriesDataPoint(date=row.date, value=row.value or 0) for row in calorie_timeseries],
        workout_heatmap=workout_heatmap
    )