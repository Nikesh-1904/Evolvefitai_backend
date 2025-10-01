# app/api/v1/workouts.py

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List
from datetime import datetime

from app.core.database import get_async_session
from app.core.auth import current_active_user
from app import models, schemas

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
        .where(models.WorkoutPlan.is_active == True)
        .order_by(models.WorkoutPlan.created_at.desc())
    )
    return result.scalars().all()

@router.post("/plans", response_model=schemas.WorkoutPlan)
async def create_workout_plan(
    workout_plan: schemas.WorkoutPlanCreate,
    current_user: models.User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session)
):
    """Create a new workout plan (manual user creation)"""
    # Convert exercises to JSON-compatible dicts before saving
    exercises_as_dicts = [ex for ex in workout_plan.exercises]

    db_plan = models.WorkoutPlan(
        user_id=current_user.id,
        name=workout_plan.name,
        description=workout_plan.description,
        exercises=exercises_as_dicts, # Ensure exercises are stored as dicts
        difficulty=workout_plan.difficulty,
        estimated_duration=workout_plan.estimated_duration,
        ai_generated=False # Manually created plans are not AI generated
    )
    session.add(db_plan)
    await session.commit()
    await session.refresh(db_plan)
    return db_plan

# --- UPDATED ENDPOINT ---
@router.post("/logs", response_model=schemas.WorkoutLog)
async def log_workout(
    workout_log: schemas.WorkoutLogCreate,
    current_user: models.User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session)
):
    """Log a completed workout with detailed set information."""
    
    # Pydantic automatically converts the incoming JSON to our schema objects.
    # To save to a JSON column, we need to convert them back to dicts.
    exercises_completed_as_dicts = [ex.dict() for ex in workout_log.exercises_completed]
    
    db_log = models.WorkoutLog(
        user_id=current_user.id,
        workout_plan_id=workout_log.workout_plan_id,
        exercises_completed=exercises_completed_as_dicts,
        duration_minutes=workout_log.duration_minutes,
        notes=workout_log.notes,
        workout_date=workout_log.workout_date or datetime.utcnow()
    )
    session.add(db_log)
    await session.commit()
    await session.refresh(db_log)
    return db_log

@router.get("/logs", response_model=List[schemas.WorkoutLog])
async def get_workout_logs(
    limit: int = 20,
    current_user: models.User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session)
):
    """Get workout logs for current user"""
    result = await session.execute(
        select(models.WorkoutLog)
        .where(models.WorkoutLog.user_id == current_user.id)
        .order_by(models.WorkoutLog.workout_date.desc())
        .limit(limit)
    )
    return result.scalars().all()

@router.get("/exercises", response_model=List[schemas.Exercise])
async def get_exercises(
    category: str = None,
    skip: int = 0,
    limit: int = 100,
    session: AsyncSession = Depends(get_async_session)
):
    """Get exercises, optionally filtered by category"""
    query = select(models.Exercise)
    
    if category:
        query = query.where(models.Exercise.category == category)
    
    query = query.offset(skip).limit(limit)
    result = await session.execute(query)
    return result.scalars().all()

@router.post("/exercises", response_model=schemas.Exercise)
async def create_exercise(
    exercise: schemas.ExerciseCreate,
    session: AsyncSession = Depends(get_async_session)
):
    """Create a new exercise"""
    db_exercise = models.Exercise(**exercise.dict())
    session.add(db_exercise)
    await session.commit()
    await session.refresh(db_exercise)
    return db_exercise