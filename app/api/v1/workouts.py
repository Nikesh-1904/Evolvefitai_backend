# app/api/v1/workouts.py

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional
from datetime import datetime, timezone  # ✅ ADDED timezone
import math  # ✅ FIX #1: Added missing math import
import logging  # ✅ FIX #1: Added logging import
from app.core.database import get_async_session
from app.core.auth import current_active_user
from app import models, schemas
from app.schemas import ExerciseType
import uuid

logger = logging.getLogger(__name__)  # ✅ FIX #1: Initialize logger

router = APIRouter()

@router.get("/plans", response_model=List[schemas.WorkoutPlan])
async def get_user_workout_plans(
    current_user: models.User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session)
):
    """Get all workout plans for current user"""
    result = await session.execute(
        select(models.WorkoutPlan)
        .where(models.WorkoutPlan.user_id == current_user.id)
        .where(models.WorkoutPlan.is_active )
        .order_by(models.WorkoutPlan.created_at.desc())
    )
    return result.scalars().all()

@router.get("/plans/{plan_id}", response_model=schemas.WorkoutPlan)
async def get_workout_plan_by_id(
    plan_id: uuid.UUID,
    current_user: models.User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session)
):
    """Get a single workout plan by its ID for the current user."""
    result = await session.execute(
        select(models.WorkoutPlan)
        .where(models.WorkoutPlan.id == plan_id)
        .where(models.WorkoutPlan.user_id == current_user.id)
    )
    db_plan = result.scalars().first()

    if db_plan is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workout plan not found",
        )
    
    return db_plan

@router.post("/plans", response_model=schemas.WorkoutPlan)
async def create_workout_plan(
    workout_plan: schemas.WorkoutPlanCreate,
    current_user: models.User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session)
):
    """Create a new workout plan (manual user creation)"""
    exercises_as_dicts = [ex for ex in workout_plan.exercises]

    db_plan = models.WorkoutPlan(
        user_id=current_user.id,
        name=workout_plan.name,
        description=workout_plan.description,
        exercises=exercises_as_dicts,
        difficulty=workout_plan.difficulty,
        estimated_duration=workout_plan.estimated_duration,
        ai_generated=False
    )
    session.add(db_plan)
    await session.commit()
    await session.refresh(db_plan)
    return db_plan

levelSystem = {
  1: { 'minPoints': 0, 'maxPoints': 499 },
  2: { 'minPoints': 500, 'maxPoints': 999 },
  3: { 'minPoints': 1000, 'maxPoints': 1999 },
  4: { 'minPoints': 2000, 'maxPoints': 3999 },
  5: { 'minPoints': 4000, 'maxPoints': 7999 },
  6: { 'minPoints': 8000, 'maxPoints': 15999 },
  7: { 'minPoints': 16000, 'maxPoints': 31999 },
  8: { 'minPoints': 32000, 'maxPoints': 63999 },
  9: { 'minPoints': 64000, 'maxPoints': 127999 },
  10: { 'minPoints': 128000, 'maxPoints': float('inf') },
}

def calculate_level(points: int) -> int:
    for level, data in levelSystem.items():
        if points >= data['minPoints'] and points <= data['maxPoints']:
            return int(level)
    return 1

@router.get("/logs", response_model=List[schemas.WorkoutLog])
async def get_workout_logs(
    limit: int = 20,
    current_user: models.User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session)
):
    """Get workout logs for the current user."""
    result = await session.execute(
        select(models.WorkoutLog)
        .where(models.WorkoutLog.user_id == current_user.id)
        .order_by(models.WorkoutLog.workout_date.desc())
        .limit(limit)
    )
    logs = result.scalars().all()

    for log in logs:
        if log.exercises_completed:
            for exercise in log.exercises_completed:
                if "exercise_type" not in exercise:
                    exercise["exercise_type"] = "WEIGHT_BASED"
    
    return logs

@router.post("/logs", response_model=schemas.WorkoutLog)
async def log_workout(
    workout_log: schemas.WorkoutLogCreate,
    current_user: models.User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session)
):
    """Log a completed workout and calculate calories burned using per-exercise MET values."""
    
    total_calories_burned: float = 0.0
    user_weight_kg = getattr(current_user, 'weight', 70.0) or 70.0
    DEFAULT_MET_VALUE = 3.5
    
    for exercise_log in workout_log.exercises_completed:
        try:
            result = await session.execute(
                select(models.Exercise).where(
                    func.lower(models.Exercise.name) == func.lower(exercise_log.name)
                )
            )
            exercise_db = result.scalars().first()
            
            if exercise_db and exercise_db.met_value is not None:
                met_value = float(exercise_db.met_value)
            else:
                met_value = DEFAULT_MET_VALUE
                logger.warning(
                    f"No MET value found for exercise '{exercise_log.name}', "
                    f"using default {DEFAULT_MET_VALUE}"
                )
        except Exception as e:
            logger.error(f"Error fetching exercise '{exercise_log.name}': {e}")
            met_value = DEFAULT_MET_VALUE

        exercise_duration_seconds = 0
        
        if exercise_log.exercise_type in [
            schemas.ExerciseType.DURATION, 
            schemas.ExerciseType.DISTANCE_DURATION, 
            schemas.ExerciseType.QUALITATIVE
        ]:
            exercise_duration_seconds = sum(
                getattr(s, 'duration_seconds', 0) 
                for s in exercise_log.sets 
                if hasattr(s, 'duration_seconds')
            )
                  
        elif exercise_log.exercise_type in [
            schemas.ExerciseType.WEIGHT_BASED, 
            schemas.ExerciseType.REPS_ONLY
        ]:
            num_sets = len(exercise_log.sets)
            exercise_duration_seconds = num_sets * 60

        if exercise_duration_seconds > 0:
            duration_hours = exercise_duration_seconds / 3600.0
            calories_for_exercise = duration_hours * met_value * user_weight_kg
            total_calories_burned += calories_for_exercise

    exercises_completed_as_dicts = [ex.model_dump() for ex in workout_log.exercises_completed]
    
    # ✅ FIX #2: Use timezone-aware datetime
    workout_date = workout_log.workout_date if workout_log.workout_date else datetime.now(timezone.utc)
    
    db_log = models.WorkoutLog(
        user_id=current_user.id,
        workout_plan_id=workout_log.workout_plan_id,
        exercises_completed=exercises_completed_as_dicts,
        duration_minutes=workout_log.duration_minutes,
        calories_burned=round(total_calories_burned),
        notes=workout_log.notes,
        workout_date=workout_date
    )
    session.add(db_log)
    
    # ✅ FIX #1: math.floor now works because math is imported
    points_earned = math.floor(total_calories_burned / 2)
    
    if points_earned > 0:
        current_user.total_points += points_earned
        current_user.level = calculate_level(current_user.total_points)
    
    await session.commit()
    await session.refresh(db_log)
    
    logger.info(
        f"Workout logged: {db_log.id} for user {current_user.id}, "
        f"burned {total_calories_burned:.1f} calories, earned {points_earned} points"
    )
    
    return db_log

@router.get("/exercises", response_model=List[schemas.Exercise])
async def get_exercises(
    exercise_type: Optional[str] = None,
    skip: int = 0,
    limit: int = 100,
    session: AsyncSession = Depends(get_async_session)
):
    """Get exercises, optionally filtered by category"""
    query = select(models.Exercise)
    
    if exercise_type:
        query = query.where(models.Exercise.exercise_type == exercise_type)
    
    query = query.offset(skip).limit(limit)
    result = await session.execute(query)
    return result.scalars().all()

@router.post("/exercises", response_model=schemas.Exercise)
async def create_exercise(
    exercise: schemas.ExerciseCreate,
    session: AsyncSession = Depends(get_async_session)
):
    """Create a new exercise"""
    db_exercise = models.Exercise(**exercise.model_dump())
    session.add(db_exercise)
    await session.commit()
    await session.refresh(db_exercise)
    return db_exercise