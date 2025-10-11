# app/api/v1/stats.py

from datetime import date, timedelta
from fastapi import APIRouter, Depends, Query
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List

from app.core.database import get_async_session
from app.core.auth import current_active_user
from app import models, schemas

router = APIRouter()

# In Evolvefitai_backend/app/api/v1/stats.py

@router.get("/overview", response_model=schemas.DashboardOverviewStats)
async def get_dashboard_overview(
    current_user: models.User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session)
):
    """Get calculated overview statistics for the main dashboard cards and level progress."""
    
    today = date.today()
    yesterday = today - timedelta(days=1)

    # --- Query for LIFETIME stats ---
    lifetime_query = (
        select(
            func.count(models.WorkoutLog.id).label("workouts_completed"),
            func.sum(models.WorkoutLog.calories_burned).label("total_calories_burned")
        )
        .where(models.WorkoutLog.user_id == current_user.id)
    )
    lifetime_result = await session.execute(lifetime_query)
    lifetime_stats = lifetime_result.first()
    total_lifetime_calories = lifetime_stats.total_calories_burned if lifetime_stats else 0
    
    # --- Query for TODAY'S stats ---
    today_query = (
        select(
            func.sum(models.WorkoutLog.duration_minutes).label("duration"),
            func.sum(models.WorkoutLog.calories_burned).label("calories")
        )
        .where(models.WorkoutLog.user_id == current_user.id)
        .where(func.date(models.WorkoutLog.workout_date) == today)
    )
    today_result = await session.execute(today_query)
    today_stats = today_result.first()
    today_calories = today_stats.calories or 0
    today_duration = today_stats.duration or 0


    # --- Query for YESTERDAY'S stats (for comparison) ---
    yesterday_query = (
        select(
            func.sum(models.WorkoutLog.duration_minutes).label("duration"),
            func.sum(models.WorkoutLog.calories_burned).label("calories")
        )
        .where(models.WorkoutLog.user_id == current_user.id)
        .where(func.date(models.WorkoutLog.workout_date) == yesterday)
    )
    yesterday_result = await session.execute(yesterday_query)
    yesterday_stats = yesterday_result.first()
    yesterday_calories = yesterday_stats.calories or 0
    yesterday_duration = yesterday_stats.duration or 0


    # Helper function to calculate percentage change
    def calculate_change(today_val, yesterday_val):
        today_val = today_val or 0
        yesterday_val = yesterday_val or 0
        if yesterday_val == 0:
            return 100.0 if today_val > 0 else 0.0
        return ((today_val - yesterday_val) / yesterday_val) * 100

    # Calculate changes using the correct '.calories' and '.duration' attributes
    calories_change = calculate_change(today_calories, yesterday_calories)
    time_change = calculate_change(today_duration, yesterday_duration)

    # ... (Level progress calculation remains the same) ...
    total_lifetime_calories = lifetime_stats.total_calories_burned or 0
    points = total_lifetime_calories / 2
    level = 1
    points_for_current_level = 0
    points_for_next_level = 100
    temp_threshold = 100
    while points >= temp_threshold:
        level += 1
        points_for_current_level = temp_threshold
        temp_threshold *= 5
        points_for_next_level = temp_threshold
    
    level_progress_data = schemas.LevelProgress(
        current_level=level,
        current_points=int(points),
        points_for_current_level=points_for_current_level,
        points_for_next_level=points_for_next_level
    )

    # --- Construct the final response using the correct '.calories' attribute ---
    return schemas.DashboardOverviewStats(
        total_calories_burned=int(today_calories),
        total_workout_time_hours=round(today_duration / 60, 1),
        workouts_completed=lifetime_stats.workouts_completed or 0,
        level_progress=level_progress_data,
        calories_change_percent=calories_change,
        time_change_percent=time_change
    )

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
    workout_heatmap: List[date] = list(heatmap_result.scalars().all())

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
    
    query = (
        select(models.WorkoutLog)
        .where(models.WorkoutLog.user_id == current_user.id)
        .order_by(models.WorkoutLog.workout_date.asc())
    )
    result = await session.execute(query)
    all_logs = result.scalars().all()
    
    progression_data = []

    for log in all_logs:
        for exercise in log.exercises_completed:
            if exercise.get("name", "").lower() == exercise_name.lower():
                
                ex_type = exercise.get("exercise_type")
                sets = exercise.get("sets", [])
                
                primary_metric_value = 0
                metric_type = "unknown"

                # Calculate the primary metric based on exercise type
                if ex_type == "WEIGHT_BASED":
                    metric_type = "Volume (kg)"
                    primary_metric_value = sum(s.get("reps", 0) * s.get("weight", 0) for s in sets)
                elif ex_type == "REPS_ONLY":
                    metric_type = "Total Reps"
                    primary_metric_value = sum(s.get("reps", 0) for s in sets)
                elif ex_type == "DURATION":
                    metric_type = "Total Duration (sec)"
                    primary_metric_value = sum(s.get("duration_seconds", 0) for s in sets)
                elif ex_type == "DISTANCE_DURATION":
                    metric_type = "Total Distance (km)"
                    primary_metric_value = sum(s.get("distance_km", 0) for s in sets)
                
                if metric_type != "unknown":
                    data_point = schemas.ExerciseProgressionDataPoint(
                        workout_date=log.workout_date.date(),
                        primary_metric_value=primary_metric_value,
                        metric_type=metric_type,
                        sets=sets
                    )
                    progression_data.append(data_point)
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