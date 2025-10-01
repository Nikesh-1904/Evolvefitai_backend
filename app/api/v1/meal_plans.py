# app/api/v1/meal_plans.py

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List

from app.core.database import get_async_session
from app.core.auth import current_active_user
from app import models, schemas

router = APIRouter()

@router.post("", response_model=schemas.MealPlan, status_code=status.HTTP_201_CREATED)
async def create_meal_plan(
    meal_plan: schemas.MealPlanCreate,
    current_user: models.User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session)
):
    """
    Save a new meal plan to the database.
    """
    # Convert meals dict to ensure JSON compatibility if it contains Pydantic models
    meals_as_dict = meal_plan.meals

    db_plan = models.MealPlan(
        user_id=current_user.id,
        name=meal_plan.name,
        target_calories=meal_plan.target_calories,
        target_protein=meal_plan.target_protein,
        target_carbs=meal_plan.target_carbs,
        target_fat=meal_plan.target_fat,
        meals=meals_as_dict,
        ai_generated=True, # For now, assume all saved plans are from AI
        ai_model= "AI Model" # Placeholder, this will be populated by the generation endpoint
    )
    session.add(db_plan)
    await session.commit()
    await session.refresh(db_plan)
    return db_plan

@router.get("", response_model=List[schemas.MealPlan])
async def get_user_meal_plans(
    current_user: models.User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session)
):
    """
    Get all meal plans for the current user.
    """
    result = await session.execute(
        select(models.MealPlan)
        .where(models.MealPlan.user_id == current_user.id)
        .order_by(models.MealPlan.created_at.desc())
    )
    return result.scalars().all()