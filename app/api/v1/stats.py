# app/api/v1/stats.py

import math
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_async_session
from app.core.auth import current_active_user
from app import models, schemas

router = APIRouter()

# In Evolvefitai_backend/app/api/v1/stats.py

@router.get("/dashboard", response_model=schemas.DashboardStats)
async def get_dashboard_stats(
    current_user: models.User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session)
):
    """Get calculated dashboard statistics for the current user."""
    
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

    workouts_completed = stats.workouts_completed or 0
    total_duration_minutes = stats.total_duration_minutes or 0
    total_calories_burned = stats.total_calories_burned or 0

    # --- START: CORRECTED LEVEL CALCULATION ---
    
    points = total_calories_burned / 2
    level = 1
    threshold = 100  # Points needed to reach Level 2

    # This loop correctly finds the user's level based on the 5x multiplier
    while points >= threshold:
        level += 1
        # The next threshold is 5 times the current one
        threshold *= 5

    # --- END: CORRECTED LEVEL CALCULATION ---

    return schemas.DashboardStats(
        workouts_completed=workouts_completed,
        total_workout_time_hours=round(total_duration_minutes / 60, 1),
        total_calories_burned=int(total_calories_burned),
        fitness_level=f"Level {level}",
    )