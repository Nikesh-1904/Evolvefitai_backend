from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List

from app.core.database import get_async_session
from app.core.auth import current_active_user
from app import models, schemas
from app.services.ai_services import ai_workout_generator

router = APIRouter()

@router.post("/workouts/generate", response_model=schemas.WorkoutPlan)
async def generate_workout_plan(
    request: schemas.WorkoutGenerationRequest,
    current_user: models.User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session)
):
    """Generate AI-powered workout plan"""
    try:
        # Generate workout using AI
        workout_data = await ai_workout_generator.generate_workout(
            current_user, 
            request.duration_minutes
        )
        
        # Create workout plan in database
        workout_plan = models.WorkoutPlan(
            user_id=current_user.id,
            name=workout_data["name"],
            description=workout_data["description"],
            exercises=workout_data["exercises"],
            difficulty=workout_data.get("difficulty", "intermediate"),
            estimated_duration=workout_data["estimated_duration"],
            ai_generated=workout_data.get("ai_generated", False),
            ai_model=workout_data.get("ai_model", "Unknown")
        )
        
        session.add(workout_plan)
        await session.commit()
        await session.refresh(workout_plan)
        
        return workout_plan
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Workout generation failed: {str(e)}")

@router.get("/exercises/{exercise_id}/videos", response_model=List[schemas.ExerciseVideo])
async def get_exercise_videos(
    exercise_id: int,
    current_user: models.User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session)
):
    """Get video suggestions for an exercise"""
    try:
        # Get exercise
        result = await session.execute(select(models.Exercise).where(models.Exercise.id == exercise_id))
        exercise = result.scalar_one_or_none()
        
        if not exercise:
            raise HTTPException(status_code=404, detail="Exercise not found")
        
        # Get existing videos
        result = await session.execute(
            select(models.ExerciseVideo).where(models.ExerciseVideo.exercise_id == exercise_id)
        )
        videos = result.scalars().all()
        
        # If no videos exist, create dummy ones
        if not videos:
            dummy_videos = [
                {
                    "exercise_id": exercise_id,
                    "youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                    "title": f"{exercise.name} - Proper Form Tutorial",
                    "thumbnail_url": "https://img.youtube.com/vi/dQw4w9WgXcQ/maxresdefault.jpg",
                    "duration": 300,
                    "popularity_score": 0.8
                },
                {
                    "exercise_id": exercise_id,
                    "youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                    "title": f"{exercise.name} - Common Mistakes",
                    "thumbnail_url": "https://img.youtube.com/vi/dQw4w9WgXcQ/maxresdefault.jpg",
                    "duration": 240,
                    "popularity_score": 0.7
                }
            ]
            
            for video_data in dummy_videos:
                video = models.ExerciseVideo(**video_data)
                session.add(video)
                videos.append(video)
            
            await session.commit()
            
            for video in videos:
                await session.refresh(video)
        
        return videos
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get videos: {str(e)}")

@router.get("/exercises/{exercise_id}/tips", response_model=List[schemas.ExerciseTip])
async def get_exercise_tips(
    exercise_id: int,
    current_user: models.User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session)
):
    """Get tips for an exercise"""
    try:
        # Get tips
        result = await session.execute(
            select(models.ExerciseTip)
            .where(models.ExerciseTip.exercise_id == exercise_id)
            .order_by(models.ExerciseTip.popularity_score.desc())
        )
        tips = result.scalars().all()
        
        # If no tips exist, create default ones
        if not tips:
            default_tips = [
                {
                    "exercise_id": exercise_id,
                    "title": "Focus on Form",
                    "content": "Proper form is more important than heavy weight. Focus on controlled movements.",
                    "tip_type": "form",
                    "popularity_score": 0.8
                },
                {
                    "exercise_id": exercise_id,
                    "title": "Breathing Technique",
                    "content": "Exhale during exertion, inhale during relaxation.",
                    "tip_type": "breathing",
                    "popularity_score": 0.7
                }
            ]
            
            for tip_data in default_tips:
                tip = models.ExerciseTip(**tip_data)
                session.add(tip)
                tips.append(tip)
            
            await session.commit()
            
            for tip in tips:
                await session.refresh(tip)
        
        return tips
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get tips: {str(e)}")

@router.post("/tips/{tip_id}/interact", response_model=schemas.MessageResponse)
async def interact_with_tip(
    tip_id: int,
    interaction: schemas.TipInteractionCreate,
    current_user: models.User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session)
):
    """Like or dislike a tip"""
    try:
        # Check existing interaction
        result = await session.execute(
            select(models.TipInteraction).where(
                models.TipInteraction.user_id == current_user.id,
                models.TipInteraction.tip_id == tip_id
            )
        )
        existing = result.scalar_one_or_none()
        
        if existing:
            existing.interaction_type = interaction.interaction_type
        else:
            new_interaction = models.TipInteraction(
                user_id=current_user.id,
                tip_id=tip_id,
                interaction_type=interaction.interaction_type
            )
            session.add(new_interaction)
        
        await session.commit()
        return schemas.MessageResponse(message="Interaction recorded successfully")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to record interaction: {str(e)}")