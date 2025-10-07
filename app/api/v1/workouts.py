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
    
    exercises_completed_as_dicts = [ex.model_dump() for ex in workout_log.exercises_completed]
    
    # --- START: ADVANCED CALORIE CALCULATION ---

    met_values = []
    DEFAULT_MET_VALUE = 3.5

    # 1. Loop through each exercise the user logged.
    for exercise_log in workout_log.exercises_completed:
        exercise_name = exercise_log.name
        
        # 2. Try to find the exercise and its MET value in our database first.
        result = await session.execute(
            select(models.Exercise).where(models.Exercise.name.ilike(exercise_name))
        )
        exercise_db = result.scalars().first()
        
        if exercise_db and exercise_db.met_value:
            met_values.append(exercise_db.met_value)
        else:
            # 3. If not in DB, call the AI to get the MET value.
            ai_met_value = await ai_workout_generator.get_met_value_for_exercise(exercise_name)
            met_values.append(ai_met_value)
            
            # 4. (Self-populating) If the exercise exists, update it with the new MET value.
            if exercise_db:
                exercise_db.met_value = ai_met_value
                session.add(exercise_db)

    # 5. Calculate the average MET value for the session.
    average_met = sum(met_values) / len(met_values) if met_values else DEFAULT_MET_VALUE

    # 6. Use the average MET in our calorie formula.
    user_weight_kg = current_user.weight if current_user.weight else 70.0
    duration_hours = (workout_log.duration_minutes or 0) / 60.0
    calories_burned = round(duration_hours * average_met * user_weight_kg)

    # --- END: ADVANCED CALORIE CALCULATION ---
    
    db_log = models.WorkoutLog(
        user_id=current_user.id,
        workout_plan_id=workout_log.workout_plan_id,
        exercises_completed=exercises_completed_as_dicts,
        duration_minutes=workout_log.duration_minutes,
        calories_burned=calories_burned,
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