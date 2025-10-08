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
# In stats.py
@router.get("/analytics", response_model=schemas.AnalyticsData)
async def get_analytics_data(
    aggregate_by: str = Query("day", enum=["day", "week", "month"]),
    current_user: models.User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session)
):
    """Get time-series data for analytics charts (heatmap and calorie graph)."""

    # --- HEATMAP DATA ---
    heatmap_query = select(func.distinct(func.date(models.WorkoutLog.workout_date))).where(
        models.WorkoutLog.user_id == current_user.id
    )
    heatmap_result = await session.execute(heatmap_query)
    workout_heatmap = heatmap_result.scalars().all()

    # --- CALORIE TIME-SERIES DATA (Corrected) ---
    # The 'date_trunc' function needs to be used in both select and group_by
    date_agg = func.date_trunc(aggregate_by, models.WorkoutLog.workout_date)
    
    calories_query = (
        select(
            date_agg.label("date"),
            func.sum(models.WorkoutLog.calories_burned).label("value")
        )
        .where(models.WorkoutLog.user_id == current_user.id)
        .group_by(date_agg)
        .order_by(date_agg.asc())
    )
    calories_result = await session.execute(calories_query)
    calorie_timeseries = calories_result.all()

    return schemas.AnalyticsData(
        calorie_timeseries=[schemas.TimeSeriesDataPoint(date=row.date, value=row.value or 0) for row in calorie_timeseries],
        workout_heatmap=workout_heatmap
    )
    
@router.get("/exercise-progression", response_model=List[schemas.ExerciseProgressionDataPoint])
async def get_exercise_progression(
    exercise_name: str,
    current_user: models.User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session)
):
    """Get the historical progression for a single exercise for the current user."""
    
    # 1. Fetch all workout logs for the user, ordered by date.
    query = (
        select(models.WorkoutLog)
        .where(models.WorkoutLog.user_id == current_user.id)
        .order_by(models.WorkoutLog.workout_date.asc())
    )
    result = await session.execute(query)
    all_logs = result.scalars().all()
    
    progression_data = []

    # 2. Process the logs in Python to find the specific exercise.
    for log in all_logs:
        # The exercises are stored in a JSON column.
        for exercise in log.exercises_completed:
            if exercise.get("name", "").lower() == exercise_name.lower():
                
                # 3. Calculate total volume (sets * reps * weight) for this workout day.
                total_volume = sum(
                    s.get('reps', 0) * s.get('weight', 0)
                    for s in exercise.get('sets', [])
                )

                # 4. Create the data point for the response.
                data_point = schemas.ExerciseProgressionDataPoint(
                    workout_date=log.workout_date.date(),
                    total_volume=total_volume,
                    sets=[schemas.ExerciseSetData(**s) for s in exercise.get('sets', [])]
                )
                progression_data.append(data_point)

                # Found the exercise for this log, move to the next log.
                break 
                
    return progression_data

@router.get("/logged-exercises", response_model=List[str])
async def get_logged_exercises(
    current_user: models.User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session)
):
    """Get a list of all unique exercise names from a user's logs."""
    query = (
        select(models.WorkoutLog.exercises_completed)
        .where(models.WorkoutLog.user_id == current_user.id)
    )
    result = await session.execute(query)
    all_exercise_lists = result.scalars().all()

    unique_exercise_names = set()
    for exercise_list in all_exercise_lists:
        for exercise in exercise_list:
            if "name" in exercise:
                unique_exercise_names.add(exercise["name"])

    return sorted(list(unique_exercise_names))