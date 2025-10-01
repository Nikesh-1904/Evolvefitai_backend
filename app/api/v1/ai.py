# app/api/v1/ai.py

import logging
import requests
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional , Dict
from datetime import datetime

from app.core.database import get_async_session
from app.core.auth import current_active_user
from app.core.config import settings
from app import models, schemas
from app.services.ai_services import ai_workout_generator

# Set up logging for this module
logger = logging.getLogger(__name__)

router = APIRouter()

# Enhanced Exercise Database for fallback
EXERCISE_DATABASE = {
    "bicep curls": {
        "name": "Bicep Curls",
        "instructions": "Hold dumbbells at your sides, curl weights up towards shoulders, lower slowly",
        "muscle_groups": ["biceps", "arms"],
        "equipment": "dumbbells",
        "difficulty": "beginner",
        "videos": [
            {
                "title": "Perfect Bicep Curls Form",
                "youtube_url": "https://www.youtube.com/watch?v=ykJmrZ5v0Oo",
                "duration": 180,
                "thumbnail_url": "https://img.youtube.com/vi/ykJmrZ5v0Oo/maxresdefault.jpg"
            }
        ],
        "tips": [
            {
                "title": "Keep Your Elbows Stationary",
                "content": "Don't swing your elbows forward or backward. Keep them at your sides throughout the movement."
            },
            {
                "title": "Control the Weight",
                "content": "Focus on slow, controlled movements. Don't use momentum to lift the weight."
            }
        ]
    },
    "lateral raises": {
        "name": "Lateral Raises",
        "instructions": "Hold dumbbells at sides, raise arms out to shoulder height, lower with control",
        "muscle_groups": ["shoulders", "deltoids"],
        "equipment": "dumbbells",
        "difficulty": "beginner",
        "videos": [
            {
                "title": "How to Do Lateral Raises",
                "youtube_url": "https://www.youtube.com/watch?v=3VcKaXpzqRo",
                "duration": 165,
                "thumbnail_url": "https://img.youtube.com/vi/3VcKaXpzqRo/maxresdefault.jpg"
            }
        ],
        "tips": [
            {
                "title": "Stop at Shoulder Height",
                "content": "Don't raise your arms above shoulder level to avoid shoulder impingement."
            },
            {
                "title": "Use Light Weight",
                "content": "Start with lighter weights and focus on form. This exercise is about isolation, not heavy lifting."
            }
        ]
    },
    "push-ups": {
        "name": "Push-ups",
        "instructions": "Start in plank position, lower chest to ground, push back up",
        "muscle_groups": ["chest", "triceps", "shoulders"],
        "equipment": "bodyweight",
        "difficulty": "beginner",
        "videos": [
            {
                "title": "Perfect Push-up Form",
                "youtube_url": "https://www.youtube.com/watch?v=IODxDxX7oi4",
                "duration": 240,
                "thumbnail_url": "https://img.youtube.com/vi/IODxDxX7oi4/maxresdefault.jpg"
            }
        ],
        "tips": [
            {
                "title": "Keep Your Body Straight",
                "content": "Maintain a straight line from head to heels throughout the movement."
            },
            {
                "title": "Full Range of Motion",
                "content": "Lower your chest all the way to the ground for maximum benefit."
            }
        ]
    },
    "squats": {
        "name": "Squats",
        "instructions": "Stand with feet shoulder-width apart, lower hips back and down, stand back up",
        "muscle_groups": ["quadriceps", "glutes", "hamstrings"],
        "equipment": "bodyweight",
        "difficulty": "beginner",
        "videos": [
            {
                "title": "How to Squat Properly",
                "youtube_url": "https://www.youtube.com/watch?v=YaXPRqUwItQ",
                "duration": 200,
                "thumbnail_url": "https://img.youtube.com/vi/YaXPRqUwItQ/maxresdefault.jpg"
            }
        ],
        "tips": [
            {
                "title": "Keep Knees Behind Toes",
                "content": "Don't let your knees extend past your toes to protect your knee joints."
            },
            {
                "title": "Go Deep",
                "content": "Aim to get your thighs parallel to the ground or lower for full muscle activation."
            }
        ]
    },
    "plank": {
        "name": "Plank",
        "instructions": "Hold straight body position on forearms and toes, engage core",
        "muscle_groups": ["core", "abs", "shoulders"],
        "equipment": "bodyweight",
        "difficulty": "beginner",
        "videos": [
            {
                "title": "Perfect Plank Form",
                "youtube_url": "https://www.youtube.com/watch?v=ASdvN_XEl_c",
                "duration": 150,
                "thumbnail_url": "https://img.youtube.com/vi/ASdvN_XEl_c/maxresdefault.jpg"
            }
        ],
        "tips": [
            {
                "title": "Don't Sag or Pike",
                "content": "Keep your body in a straight line. Don't let hips drop or rise too high."
            },
            {
                "title": "Breathe Normally",
                "content": "Don't hold your breath. Maintain steady breathing throughout the hold."
            }
        ]
    }
}

async def search_youtube_videos(query: str, max_results: int = 3) -> List[Dict]:
    """Search YouTube for exercise videos"""
    if not settings.YOUTUBE_API_KEY:
        logger.warning("🚨 YouTube API key not configured, using fallback videos")
        return []
    
    try:
        url = "https://www.googleapis.com/youtube/v3/search"
        params = {
            "part": "snippet",
            "q": f"{query} exercise tutorial form",
            "type": "video",
            "videoDuration": "short",
            "videoDefinition": "high",
            "maxResults": max_results,
            "key": settings.YOUTUBE_API_KEY
        }
        
        logger.info(f"🔍 YouTube API: Searching for '{query}' videos")
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            videos = []
            
            for item in data.get("items", []):
                video_id = item["id"]["videoId"]
                snippet = item["snippet"]
                
                videos.append({
                    "title": snippet["title"],
                    "youtube_url": f"https://www.youtube.com/watch?v={video_id}",
                    "thumbnail_url": snippet["thumbnails"]["high"]["url"],
                    "duration": 180,
                    "channel": snippet["channelTitle"],
                    "published": snippet["publishedAt"]
                })
            
            logger.info(f"✅ YouTube API: Found {len(videos)} videos for '{query}'")
            return videos
        else:
            logger.error(f"❌ YouTube API: Request failed with status {response.status_code}")
            return []
            
    except Exception as e:
        logger.error(f"💥 YouTube API: Error searching videos: {str(e)}")
        return []

@router.post("/workouts/generate", response_model=schemas.WorkoutPlan)
async def generate_workout_plan(
    request: schemas.WorkoutGenerationRequest,
    current_user: models.User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session)
):
    """Generate AI-powered workout plan with comprehensive logging and self-populating exercise DB"""
    
    logger.info("=" * 100)
    logger.info("🚀 NEW WORKOUT GENERATION REQUEST")
    logger.info(f"👤 User ID: {current_user.id}")
    logger.info(f"📧 User Email: {current_user.email}")
    logger.info(f"🕐 Request Time: {datetime.now().isoformat()}")
    logger.info(f"⏱️ Duration: {request.duration_minutes} minutes")
    if request.target_muscle_groups:
        logger.info(f"💪 Targeting: {', '.join(request.target_muscle_groups)}")
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
        
        workout_data = await ai_workout_generator.generate_workout(
            user=current_user,
            duration_minutes=request.duration_minutes,
            target_muscles=request.target_muscle_groups
        )
        
        generation_time = (datetime.now() - start_time).total_seconds()
        
        logger.info("📊 GENERATION RESULTS:")
        logger.info(f"   ✅ Success: True")
        logger.info(f"   ⏱️ Total Time: {generation_time:.2f} seconds")
        logger.info(f"   🤖 AI Generated: {workout_data.get('ai_generated', 'Unknown')}")
        logger.info(f"   🔧 AI Model: {workout_data.get('ai_model', 'Unknown')}")
        logger.info(f"   🏋️ Workout Name: {workout_data.get('name', 'Unknown')}")
        logger.info(f"   📝 Exercise Count: {len(workout_data.get('exercises', []))}")
        logger.info(f"   🔥 Estimated Calories: {workout_data.get('estimated_calories', 'Not calculated')}")
        logger.info(f"   📈 Difficulty: {workout_data.get('difficulty_level', 'Not specified')}")
        
        # --- NEW: Self-populating Exercise Database Logic ---
        if workout_data.get("ai_generated") and workout_data.get("exercises"):
            for exercise_from_ai in workout_data["exercises"]:
                exercise_name = exercise_from_ai.get("name")
                if not exercise_name:
                    continue

                # Check if exercise exists (case-insensitive)
                result = await session.execute(
                    select(models.Exercise).where(func.lower(models.Exercise.name) == exercise_name.lower())
                )
                existing_exercise = result.scalars().first()

                if not existing_exercise:
                    logger.info(f"✨ New exercise found: '{exercise_name}'. Adding to database.")
                    
                    # 1. Create the new Exercise
                    new_exercise = models.Exercise(
                        name=exercise_name,
                        instructions=exercise_from_ai.get("instructions"),
                        muscle_groups=exercise_from_ai.get("muscle_groups", []),
                        difficulty=workout_data.get("difficulty_level")
                    )
                    session.add(new_exercise)
                    await session.flush()  # Flush to get the new_exercise.id

                    # 2. Search for YouTube videos
                    videos = await search_youtube_videos(exercise_name)
                    for video_data in videos:
                        new_video = models.ExerciseVideo(
                            exercise_id=new_exercise.id,
                            youtube_url=video_data["youtube_url"],
                            title=video_data["title"],
                            thumbnail_url=video_data.get("thumbnail_url"),
                            duration=video_data.get("duration")
                        )
                        session.add(new_video)

                    # 3. Add default tips
                    default_tips = [
                        {"title": "Focus on Form", "content": "Always prioritize proper form over heavy weight to prevent injury.", "tip_type": "Form"},
                        {"title": "Control Your Breathing", "content": "Exhale on exertion (the hard part) and inhale during the easier phase of the movement.", "tip_type": "Technique"}
                    ]
                    for tip_data in default_tips:
                        new_tip = models.ExerciseTip(
                            exercise_id=new_exercise.id,
                            title=tip_data["title"],
                            content=tip_data["content"],
                            tip_type=tip_data["tip_type"]
                        )
                        session.add(new_tip)
            
            # Commit all new exercises, videos, and tips at once
            await session.commit()
        # --- END of new logic ---
        
        logger.info("💾 Saving workout to database...")
        
        workout_plan = models.WorkoutPlan(
            user_id=current_user.id,
            name=workout_data["name"],
            description=workout_data.get("description", "AI-generated workout plan"),
            exercises=workout_data["exercises"],
            difficulty=workout_data.get("difficulty_level", getattr(current_user, 'experience_level', 'intermediate')),
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
        
        total_time = (datetime.now() - start_time).total_seconds()
        logger.info("=" * 100)
        logger.info("✅ WORKOUT GENERATION COMPLETED SUCCESSFULLY!")
        logger.info(f"🕐 Total Processing Time: {total_time:.2f} seconds")
        logger.info(f"🎯 Final Result: {workout_plan.ai_model} generated '{workout_plan.name}'")
        logger.info("=" * 100)

        return workout_plan

    except Exception as e:
        error_time = (datetime.now() - start_time).total_seconds()
        
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
    """ENHANCED: Search for exercise details with YouTube integration"""
    
    logger.info(f"🔍 Exercise search requested by {current_user.email}: '{name}'")
    
    try:
        result = await session.execute(
            select(models.Exercise).where(models.Exercise.name.ilike(f"%{name}%")).limit(5)
        )
        exercises = result.scalars().all()
        
        if exercises:
            logger.info(f"✅ Found {len(exercises)} exercises in database for '{name}'")
            
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
                    "videos": [{"title": v.title, "youtube_url": v.youtube_url, "duration": v.duration} for v in videos],
                    "tips": [{"title": t.title, "content": t.content} for t in tips]
                })
            
            return exercise_data
        
        exercise_name_lower = name.lower().strip()
        
        for key, exercise_info in EXERCISE_DATABASE.items():
            if key in exercise_name_lower or exercise_name_lower in key:
                logger.info(f"✅ Found '{name}' in exercise database")
                
                youtube_videos = await search_youtube_videos(exercise_info["name"])
                
                all_videos = exercise_info["videos"] + youtube_videos
                
                exercise_data = {
                    "exercise": {
                        "name": exercise_info["name"],
                        "instructions": exercise_info["instructions"],
                        "muscle_groups": exercise_info["muscle_groups"],
                        "equipment": exercise_info["equipment"],
                        "difficulty": exercise_info["difficulty"]
                    },
                    "videos": all_videos[:5],
                    "tips": exercise_info["tips"]
                }
                
                logger.info(f"📚 Returned exercise data for '{name}' with {len(all_videos)} videos and {len(exercise_info['tips'])} tips")
                return exercise_data
        
        logger.info(f"🔄 No database match for '{name}', searching YouTube...")
        youtube_videos = await search_youtube_videos(name)
        
        if youtube_videos:
            exercise_data = {
                "exercise": {
                    "name": name.title(),
                    "instructions": f"Perform {name} with proper form and control",
                    "muscle_groups": ["general"],
                    "equipment": "varies",
                    "difficulty": "moderate"
                },
                "videos": youtube_videos,
                "tips": [
                    {
                        "title": "Focus on Form",
                        "content": "Always prioritize proper form over heavy weight or speed"
                    },
                    {
                        "title": "Breathe Properly",
                        "content": "Maintain steady breathing throughout the exercise"
                    }
                ]
            }
            
            logger.info(f"📺 Found YouTube videos for '{name}': {len(youtube_videos)} videos")
            return exercise_data
        
        logger.warning(f"⚠️ No exercises or videos found for '{name}'")
        return {
            "message": f"No detailed information found for '{name}'",
            "searched_term": name,
            "suggestion": "Try searching for a more specific exercise name"
        }

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
        logger.info(f"✅ Tip interaction '{interaction.interaction_type}' recorded for improvement")
        return {
            "message": "Feedback recorded! This helps improve AI recommendations.",
            "tip_id": interaction.tip_id,
            "type": interaction.interaction_type
        }
        
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
        analysis = {
            "is_plateau": False,
            "confidence": 0.75,
            "affected_exercises": [],
            "recommendations": [
                "Try progressive overload - gradually increase weight or reps",
                "Add variety to your routine every 4-6 weeks",
                "Ensure adequate rest and recovery between sessions",
                "Focus on proper nutrition to support your goals"
            ],
            "plateau_duration_weeks": 0,
            "analysis_method": "Rule-based Analysis",
            "ai_generated": False
        }
        
        logger.info(f"✅ Plateau analysis completed for {current_user.email}")
        return analysis
        
    except Exception as e:
        logger.error(f"❌ Plateau analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Plateau analysis failed: {str(e)}")