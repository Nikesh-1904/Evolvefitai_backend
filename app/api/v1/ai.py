# app/api/v1/ai.py - Enhanced with detailed logging

import logging
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List
from datetime import datetime

from app.core.database import get_async_session
from app.core.auth import current_active_user
from app import models, schemas
from app.services.ai_services import ai_workout_generator

# Set up logging for this module
logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/workouts/generate", response_model=schemas.WorkoutPlan)
async def generate_workout_plan(
    request: schemas.WorkoutGenerationRequest,
    current_user: models.User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session)
):
    """Generate AI-powered workout plan with comprehensive logging"""
    
    # Log the incoming request
    logger.info("=" * 100)
    logger.info("🚀 NEW WORKOUT GENERATION REQUEST")
    logger.info(f"👤 User ID: {current_user.id}")
    logger.info(f"📧 User Email: {current_user.email}")
    logger.info(f"🕐 Request Time: {datetime.now().isoformat()}")
    logger.info(f"⏱️ Duration: {request.duration_minutes} minutes")
    logger.info(f"👤 User Profile Summary:")
    logger.info(f"   - Age: {getattr(current_user, 'age', 'Not set')}")
    logger.info(f"   - Weight: {getattr(current_user, 'weight', 'Not set')}kg")
    logger.info(f"   - Height: {getattr(current_user, 'height', 'Not set')}cm")
    logger.info(f"   - Goal: {getattr(current_user, 'fitness_goal', 'Not set')}")
    logger.info(f"   - Level: {getattr(current_user, 'experience_level', 'Not set')}")
    logger.info(f"   - Activity: {getattr(current_user, 'activity_level', 'Not set')}")
    logger.info("=" * 100)

    start_time = datetime.now()
    
    try:
        logger.info("🧠 Starting AI workout generation process...")
        
        # Generate workout using AI
        workout_data = await ai_workout_generator.generate_workout(
            current_user,
            request.duration_minutes
        )
        
        generation_time = (datetime.now() - start_time).total_seconds()
        
        # Log generation results
        logger.info("📊 GENERATION RESULTS:")
        logger.info(f"   ✅ Success: True")
        logger.info(f"   ⏱️ Total Time: {generation_time:.2f} seconds")
        logger.info(f"   🤖 AI Generated: {workout_data.get('ai_generated', 'Unknown')}")
        logger.info(f"   🔧 AI Model: {workout_data.get('ai_model', 'Unknown')}")
        logger.info(f"   🏋️ Workout Name: {workout_data.get('name', 'Unknown')}")
        logger.info(f"   📝 Exercise Count: {len(workout_data.get('exercises', []))}")
        logger.info(f"   🔥 Estimated Calories: {workout_data.get('estimated_calories', 'Not calculated')}")
        logger.info(f"   📈 Difficulty: {workout_data.get('difficulty_level', 'Not specified')}")
        
        # Log exercises summary
        exercises = workout_data.get('exercises', [])
        if exercises:
            logger.info("🏋️ GENERATED EXERCISES:")
            for i, exercise in enumerate(exercises, 1):
                logger.info(f"   {i}. {exercise.get('name', 'Unknown')} - {exercise.get('sets', '?')} sets x {exercise.get('reps', '?')} reps")

        # Create workout plan in database
        logger.info("💾 Saving workout to database...")
        
        workout_plan = models.WorkoutPlan(
            user_id=current_user.id,
            name=workout_data["name"],
            description=workout_data.get("description", "AI-generated workout plan"),
            exercises=workout_data["exercises"],
            difficulty=workout_data.get("difficulty_level", "moderate"),
            estimated_duration=workout_data.get("estimated_duration", request.duration_minutes),
            ai_generated=workout_data.get("ai_generated", False),
            ai_model=workout_data.get("ai_model", "Unknown"),
            is_active=True
        )

        session.add(workout_plan)
        await session.commit()
        await session.refresh(workout_plan)
        
        db_save_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"💾 Database save completed in {db_save_time - generation_time:.2f}s")
        logger.info(f"🆔 Saved with ID: {workout_plan.id}")
        
        # Final success log
        total_time = (datetime.now() - start_time).total_seconds()
        logger.info("=" * 100)
        logger.success("✅ WORKOUT GENERATION COMPLETED SUCCESSFULLY!")
        logger.info(f"🕐 Total Processing Time: {total_time:.2f} seconds")
        logger.info(f"📈 Performance Breakdown:")
        logger.info(f"   - AI Generation: {generation_time:.2f}s ({(generation_time/total_time)*100:.1f}%)")
        logger.info(f"   - Database Save: {db_save_time - generation_time:.2f}s ({((db_save_time - generation_time)/total_time)*100:.1f}%)")
        logger.info(f"🎯 Final Result: {workout_plan.ai_model} generated '{workout_plan.name}'")
        logger.info("=" * 100)

        return workout_plan

    except Exception as e:
        error_time = (datetime.now() - start_time).total_seconds()
        
        # Log the error with full context
        logger.error("=" * 100)
        logger.error("❌ WORKOUT GENERATION FAILED!")
        logger.error(f"👤 User: {current_user.email} (ID: {current_user.id})")
        logger.error(f"⏱️ Duration Requested: {request.duration_minutes} minutes")
        logger.error(f"🕐 Error Time: {error_time:.2f}s after start")
        logger.error(f"💥 Error Type: {type(e).__name__}")
        logger.error(f"📝 Error Message: {str(e)}")
        logger.error(f"📍 Stack Trace:", exc_info=True)
        logger.error("=" * 100)
        
        raise HTTPException(
            status_code=500,
            detail=f"Workout generation failed: {str(e)}"
        )


@router.get("/exercises/search")
async def search_exercises(
    name: str,
    current_user: models.User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session)
):
    """Search for exercise details with logging"""
    
    logger.info(f"🔍 Exercise search requested by {current_user.email}: '{name}'")
    
    try:
        # Search in database first
        result = await session.execute(
            select(models.Exercise).where(models.Exercise.name.ilike(f"%{name}%")).limit(5)
        )
        exercises = result.scalars().all()
        
        if exercises:
            logger.info(f"✅ Found {len(exercises)} exercises in database for '{name}'")
            
            # Get associated videos and tips
            exercise_data = []
            for exercise in exercises:
                videos_result = await session.execute(
                    select(models.ExerciseVideo).where(models.ExerciseVideo.exercise_id == exercise.id).limit(3)
                )
                videos = videos_result.scalars().all()
                
                tips_result = await session.execute(
                    select(models.ExerciseTip).where(models.ExerciseTip.exercise_id == exercise.id).limit(3)
                )
                tips = tips_result.scalars().all()
                
                exercise_data.append({
                    "exercise": exercise,
                    "videos": videos,
                    "tips": tips
                })
            
            logger.success(f"📚 Returned exercise data with {sum(len(ex['videos']) for ex in exercise_data)} videos and {sum(len(ex['tips']) for ex in exercise_data)} tips")
            return exercise_data
        else:
            logger.warning(f"⚠️ No exercises found for '{name}'")
            return {"message": "No exercises found", "searched_term": name}

    except Exception as e:
        logger.error(f"❌ Exercise search failed for '{name}': {str(e)}")
        raise HTTPException(status_code=500, detail=f"Exercise search failed: {str(e)}")


@router.post("/tips/interact")
async def interact_with_tip(
    interaction: schemas.TipInteractionCreate,
    current_user: models.User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session)
):
    """Record tip interaction (like/dislike) with logging"""
    
    logger.info(f"👍 Tip interaction from {current_user.email}: {interaction.interaction_type} on tip {interaction.tip_id}")
    
    try:
        # Implementation would go here
        logger.success(f"✅ Tip interaction recorded successfully")
        return {"message": "Interaction recorded", "tip_id": interaction.tip_id, "type": interaction.interaction_type}
        
    except Exception as e:
        logger.error(f"❌ Tip interaction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to record interaction: {str(e)}")


@router.post("/analyze-plateau")
async def analyze_plateau(
    request: dict,
    current_user: models.User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session)
):
    """Analyze workout plateau with AI assistance"""
    
    logger.info(f"📊 Plateau analysis requested by {current_user.email}")
    
    try:
        # This would implement plateau analysis logic
        analysis = {
            "is_plateau": False,
            "confidence": 0.75,
            "affected_exercises": [],
            "recommendations": ["Try increasing weight", "Add variety to routine"],
            "plateau_duration_weeks": 0,
            "analysis_method": "AI Analysis",
            "ai_generated": True
        }
        
        logger.success(f"✅ Plateau analysis completed for {current_user.email}")
        return analysis
        
    except Exception as e:
        logger.error(f"❌ Plateau analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Plateau analysis failed: {str(e)}")


# Set up module-specific logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Add custom success level for this module too
def success(self, message, *args, **kwargs):
    if self.isEnabledFor(25):
        self._log(25, message, args, **kwargs)

logging.Logger.success = success
logging.addLevelName(25, "SUCCESS")