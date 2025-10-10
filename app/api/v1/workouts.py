# app/api/v1/workouts.py

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List
from datetime import datetime

from app.core.database import get_async_session
from app.core.auth import current_active_user
from app import models, schemas
from app.services.ai_services import ai_workout_generator
from app.schemas import ExerciseType # 👈 Make sure ExerciseType is imported


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

@router.get("/plans/{plan_id}", response_model=schemas.WorkoutPlan)
async def get_workout_plan_by_id(
    plan_id: int,
    current_user: models.User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session)
):
    """Get a single workout plan by its ID for the current user."""
    result = await session.execute(
        select(models.WorkoutPlan)
        .where(models.WorkoutPlan.id == plan_id)
        # This is a critical security check to ensure users can only see their own plans.
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
    """Log a completed workout and calculate calories burned using per-exercise MET values."""
    
    total_calories_burned = 0
    user_weight_kg = getattr(current_user, 'weight', 70.0) or 70.0 # Default to 70kg if no weight
    DEFAULT_MET_VALUE = 3.5
    
    # 1. Loop through each logged exercise to calculate its specific calorie burn.
    for exercise_log in workout_log.exercises_completed:
        # Find the exercise in our database to get its MET value.
        result = await session.execute(
            select(models.Exercise).where(models.Exercise.name.ilike(exercise_log.name))
        )
        exercise_db = result.scalars().first()
        met_value = getattr(exercise_db, 'met_value', DEFAULT_MET_VALUE) or DEFAULT_MET_VALUE

        exercise_duration_seconds = 0
        
        # 2. Estimate the duration of the exercise based on its type.
        if exercise_log.exercise_type in [ExerciseType.DURATION, ExerciseType.DISTANCE_DURATION, ExerciseType.QUALITATIVE]:
            # For these types, duration is logged directly.
            exercise_duration_seconds = sum(s.duration_seconds for s in exercise_log.sets if s.duration_seconds)
        
        elif exercise_log.exercise_type in [ExerciseType.WEIGHT_BASED, ExerciseType.REPS_ONLY]:
            # For strength/reps, we estimate duration. A reasonable estimate is ~60 seconds per set (work + rest).
            num_sets = len(exercise_log.sets)
            exercise_duration_seconds = num_sets * 60

        # 3. Calculate calories for this single exercise and add to the total.
        duration_hours = exercise_duration_seconds / 3600.0
        calories_for_exercise = duration_hours * met_value * user_weight_kg
        total_calories_burned += calories_for_exercise

    # 4. Save the workout log with the accurate total calorie count.
    exercises_completed_as_dicts = [ex.model_dump() for ex in workout_log.exercises_completed]
    db_log = models.WorkoutLog(
        user_id=current_user.id,
        workout_plan_id=workout_log.workout_plan_id,
        exercises_completed=exercises_completed_as_dicts,
        duration_minutes=workout_log.duration_minutes,
        calories_burned=round(total_calories_burned), # Use our new accurate calculation
        notes=workout_log.notes,
        workout_date=workout_log.workout_date or datetime.utcnow()
    )
    session.add(db_log)
    await session.commit()
    await session.refresh(db_log)
    return db_log

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