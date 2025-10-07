# app/api/v1/stats.py

import math
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_async_session
from app.core.auth import current_active_user
from app import models, schemas

router = APIRouter()

@router.get("/dashboard", response_model=schemas.DashboardStats)
async def get_dashboard_stats(
    current_user: models.User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session)
):
    """Get calculated dashboard statistics for the current user."""
    
    # Create a query to get all stats in one database call
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

    # Handle the case where a user has no logs yet
    workouts_completed = stats.workouts_completed or 0
    total_duration_minutes = stats.total_duration_minutes or 0
    total_calories_burned = stats.total_calories_burned or 0

    # Calculate Fitness Level based on our 5x multiplier rule
    points = total_calories_burned / 2
    level = 1
    threshold = 100
    if points >= threshold:
        # A logarithmic approach is efficient for calculating levels with multipliers
        # Level = 1 + floor(log(points / 100) / log(5))
        level = 1 + math.floor(math.log(points / 100, 5))

    return schemas.DashboardStats(
        workouts_completed=workouts_completed,
        total_workout_time_hours=round(total_duration_minutes / 60, 1),
        total_calories_burned=int(total_calories_burned),
        fitness_level=f"Level {level}",
    )