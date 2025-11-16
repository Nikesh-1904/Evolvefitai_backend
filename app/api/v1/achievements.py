# app/api/v1/achievements.py

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import Dict, Any

from app.core.database import get_async_session
from app.core.auth import current_active_user
from app import models, schemas

router = APIRouter()

# --- BACKEND SOURCE OF TRUTH ---
# Copied from your AchievementsContext.js
# This ensures the backend validates points and levels
achievementDefinitions: Dict[str, Dict[str, Any]] = {
  'first_workout': { 'id': 'first_workout', 'points': 50 },
  'workout_streak_7': { 'id': 'workout_streak_7', 'points': 200 },
  'workout_streak_30': { 'id': 'workout_streak_30', 'points': 1000 },
  'workouts_25': { 'id': 'workouts_25', 'points': 300 },
  'workouts_100': { 'id': 'workouts_100', 'points': 1500 },
  'workout_time_10h': { 'id': 'workout_time_10h', 'points': 250 },
  'workout_time_100h': { 'id': 'workout_time_100h', 'points': 2000 },
  'weight_lifted_1000kg': { 'id': 'weight_lifted_1000kg', 'points': 400 },
  'ai_workout_10': { 'id': 'ai_workout_10', 'points': 300 },
  'profile_complete': { 'id': 'profile_complete', 'points': 100 },
  'early_bird': { 'id': 'early_bird', 'points': 150 },
  'night_owl': { 'id': 'night_owl', 'points': 150 },
  'weight_loss_5kg': { 'id': 'weight_loss_5kg', 'points': 500 },
  'strength_gain': { 'id': 'strength_gain', 'points': 400 },
}

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
# --- END OF SOURCE OF TRUTH ---


@router.get("/status", response_model=schemas.AchievementStatus)
async def get_achievement_status(
    user: models.User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session)
):
    """Get the user's current points, level, and unlocked achievements."""
    
    # We must explicitly load the achievements relationship
    result = await session.execute(
        select(models.UserAchievement).where(models.UserAchievement.user_id == user.id)
    )
    unlocked_list_models = result.scalars().all()
    unlocked_list_schemas = [schemas.UserAchievement.model_validate(ach) for ach in unlocked_list_models]
    
    return schemas.AchievementStatus(
        total_points=user.total_points,
        level=user.level,
        unlocked_achievements=unlocked_list_schemas
    )


@router.post("/unlock", response_model=schemas.AchievementStatus)
async def unlock_achievement(
    request: schemas.AchievementUnlockRequest,
    user: models.User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session)
):
    """
    Called by the frontend when it detects an achievement has been unlocked.
    The backend validates it, awards points, and updates the level.
    """
    achievement_id = request.achievement_id
    
    # 1. Validate the achievement ID
    achievement_def = achievementDefinitions.get(achievement_id)
    if not achievement_def:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Achievement '{achievement_id}' not found."
        )
        
    # 2. Check if user already unlocked it (idempotency)
    existing_query = select(models.UserAchievement).where(
        models.UserAchievement.user_id == user.id,
        models.UserAchievement.achievement_id == achievement_id
    )
    existing_result = await session.execute(existing_query)
    existing_achievement = existing_result.scalars().first()
    if existing_achievement:
        # Not an error, just return the current status
        return await get_achievement_status(user, session)
        
    # 3. It's new! Unlock it and update user stats.
    
    # Create the achievement record
    new_achievement = models.UserAchievement(
        user_id=user.id,
        achievement_id=achievement_id
    )
    session.add(new_achievement)
    
    # Update user's points and level
    points_to_add = achievement_def.get('points', 0)
    user.total_points += points_to_add
    user.level = calculate_level(user.total_points)
    
    # Commit changes
    await session.commit()
    await session.refresh(user)
    
    # 4. Return the new, updated status
    return await get_achievement_status(user, session)