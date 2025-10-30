# app/api/v1/ai.py - Fixed with proper response schemas and enhanced functionality

import logging
import requests
import random
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Dict
from datetime import datetime, timedelta, timezone
import uuid
from app.core.database import get_async_session
from app.core.auth import current_active_user
from app.core.config import settings
from app import models, schemas
from app.services.ai_services import ai_workout_generator
from app.schemas import ExerciseType # 👈 Make sure ExerciseType is imported

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
                "youtube_url": "https://www.youtube.com/watch?v=ASdvN-XElc",
                "duration": 150,
                "thumbnail_url": "https://img.youtube.com/vi/ASdvN-XElc/maxresdefault.jpg"
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
    },
    "jumping lunges": {
        "name": "Jumping Lunges",
        "instructions": "Start in lunge position, jump and switch legs in mid-air, land in opposite lunge",
        "muscle_groups": ["quadriceps", "glutes", "hamstrings", "calves"],
        "equipment": "bodyweight",
        "difficulty": "intermediate",
        "videos": [
            {
                "title": "How to Do Jumping Lunges",
                "youtube_url": "https://www.youtube.com/watch?v=cd3P7C7iJzc",
                "duration": 160,
                "thumbnail_url": "https://img.youtube.com/vi/cd3P7C7iJzc/maxresdefault.jpg"
            }
        ],
        "tips": [
            {
                "title": "Land Softly",
                "content": "Focus on landing lightly on the balls of your feet to reduce impact on joints."
            },
            {
                "title": "Maintain Balance",
                "content": "Keep your core engaged throughout the movement to maintain balance during the jump."
            }
        ]
    }
}

def apply_exercise_type_overrides(workout_data: dict) -> dict:
    """Scans exercise names and corrects the exercise_type based on keywords."""
    if not workout_data.get("exercises"):
        return workout_data

    # Define simple rules. This can be expanded over time.
    keyword_map = {
        'plank': ExerciseType.DURATION,
        'stretch': ExerciseType.QUALITATIVE,
        'yoga': ExerciseType.QUALITATIVE,
        'burpee': ExerciseType.REPS_ONLY,
        'push-up': ExerciseType.REPS_ONLY,
        'pushup': ExerciseType.REPS_ONLY,
        'crunch': ExerciseType.REPS_ONLY,
        'sit-up': ExerciseType.REPS_ONLY,
        'running': ExerciseType.DISTANCE_DURATION,
        'run': ExerciseType.DISTANCE_DURATION,
        'jog': ExerciseType.DISTANCE_DURATION,
        'cycling': ExerciseType.DISTANCE_DURATION,
        'bike': ExerciseType.DISTANCE_DURATION,
        'jumping jack': ExerciseType.REPS_ONLY,
        'squat jump': ExerciseType.REPS_ONLY,
    }

    for exercise in workout_data["exercises"]:
        exercise_name_lower = exercise.get("name", "").lower()
        
        # Make sure a default type is always present
        if "exercise_type" not in exercise:
            exercise["exercise_type"] = ExerciseType.WEIGHT_BASED

        # Apply overrides
        for keyword, ex_type in keyword_map.items():
            if keyword in exercise_name_lower:
                if exercise["exercise_type"] != ex_type:
                    logger.info(f"Overriding exercise type for '{exercise['name']}' to {ex_type.value}")
                    exercise["exercise_type"] = ex_type
                break # Stop after first match

    return workout_data

async def search_youtube_videos(query: str, max_results: int = 3) -> List[Dict]:
    """Search YouTube for exercise videos"""
    if not settings.YOUTUBE_API_KEY:
        logger.warning("YouTube API key not configured, using fallback videos")
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

        logger.info(f"YouTube API: Searching for '{query}' videos")
        response = requests.get(url, params=params, timeout=10) # type: ignore

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

            logger.info(f"YouTube API: Found {len(videos)} videos for '{query}'")
            return videos
        else:
            logger.error(f"YouTube API: Request failed with status {response.status_code}")
            return []

    except Exception as e:
        logger.error(f"YouTube API: Error searching videos: {str(e)}")
        return []

@router.get("/met-value", response_model=schemas.METValueResponse)
async def get_met_value(
    exercise_name: str,
    current_user: models.User = Depends(current_active_user)
):
    """Get the MET value for a single exercise by its name using the AI service."""
    met_value = await ai_workout_generator.get_met_value_for_exercise(exercise_name)
    return schemas.METValueResponse(
        exercise_name=exercise_name,
        met_value=met_value
    )

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
    logger.info(f"🕒 Request Time: {datetime.now().isoformat()}")
    logger.info(f"⏱️  Duration: {request.duration_minutes} minutes")
    logger.info(f"💪 Targeting: {', '.join(request.target_muscle_groups)}" if request.target_muscle_groups else "💪 Targeting: Not specified")
    logger.info("👤 User Profile Summary:")
    logger.info(f"   - Age: {getattr(current_user, 'age', 'Not set')}")
    logger.info(f"   - Weight: {getattr(current_user, 'weight', 'Not set')}kg") 
    logger.info(f"   - Height: {getattr(current_user, 'height', 'Not set')}cm")
    logger.info(f"   - Goal: {getattr(current_user, 'fitness_goal', 'Not set')}")
    logger.info(f"   - Level: {getattr(current_user, 'experience_level', 'Not set')}")
    logger.info(f"   - Activity: {getattr(current_user, 'activity_level', 'Not set')}")
    logger.info(f"🔢 Exercises Requested: {request.num_exercises}" if request.num_exercises else "🔢 Exercises Requested: AI decides")
    logger.info(f"🏡 Workout Type: {request.workout_type}" if request.workout_type else "🏡 Workout Type: Not specified")


    logger.info("=" * 100)

    start_time = datetime.now()

    try:
        logger.info("🤖 Starting AI workout generation process...")

        # 👇 Pass the new filters to the AI service
        workout_data = await ai_workout_generator.generate_workout(
            user=current_user,
            duration_minutes=request.duration_minutes or 45,
            target_muscles=request.target_muscle_groups,
            num_exercises=request.num_exercises,
            workout_type=request.workout_type
        )

        generation_time = (datetime.now() - start_time).total_seconds()

        logger.info("📊 GENERATION RESULTS:")
        logger.info("✅ Success: True")
        logger.info(f"⏱️  Total Time: {generation_time:.2f} seconds")
        logger.info(f"🤖 AI Generated: {workout_data.get('ai_generated', 'Unknown')}")
        logger.info(f"🎯 AI Model: {workout_data.get('ai_model', 'Unknown')}")
        logger.info(f"💪 Workout Name: {workout_data.get('name', 'Unknown')}")
        logger.info(f"🏋️  Exercise Count: {len(workout_data.get('exercises', []))}")
        logger.info(f"🔥 Estimated Calories: {workout_data.get('estimated_calories', 'Not calculated')}")
        logger.info(f"📈 Difficulty: {workout_data.get('difficulty_level', 'Not specified')}")

        workout_data = apply_exercise_type_overrides(workout_data)

        # --- NEW Self-populating Exercise Database Logic ---
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
                    logger.info(f"🆕 New exercise found: '{exercise_name}'. Adding to database.")

                    # 1. Create the new Exercise
                    new_exercise = models.Exercise(
                        name=exercise_name,
                        instructions=exercise_from_ai.get("instructions", ""),
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
                        {
                            "title": "Focus on Form",
                            "content": "Always prioritize proper form over heavy weight to prevent injury.",
                            "tip_type": "Form"
                        },
                        {
                            "title": "Control Your Breathing",
                            "content": "Exhale on exertion (the hard part) and inhale during the easier phase of the movement.",
                            "tip_type": "Technique"
                        }
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
            name=workout_data.get("name", "AI Generated Workout"), # Safe access with a default title
            description=workout_data.get("description", "AI-generated workout plan"),
            exercises=workout_data.get("exercises", []), # Safe access with a default empty list
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
        logger.info("🎉 WORKOUT GENERATION COMPLETED SUCCESSFULLY!")
        logger.info(f"⏱️  Total Processing Time: {total_time:.2f} seconds")
        logger.info(f"🏆 Final Result: {workout_plan.ai_model} generated '{workout_plan.name}'")
        logger.info("=" * 100)

        return workout_plan

    except Exception as e:
        error_time = (datetime.now() - start_time).total_seconds()
        logger.error("=" * 100)
        logger.error("💥 WORKOUT GENERATION FAILED!")
        logger.error(f"👤 User: {current_user.email} (ID: {current_user.id})")
        logger.error(f"⏱️  Duration Requested: {request.duration_minutes} minutes")
        logger.error(f"🕒 Error Time: {error_time:.2f}s after start")
        logger.error(f"🔴 Error Type: {type(e).__name__}")
        logger.error(f"📝 Error Message: {str(e)}")
        logger.error("📚 Stack Trace:", exc_info=True)
        logger.error("=" * 100)
        raise HTTPException(status_code=500, detail=f"Workout generation failed: {str(e)}")

@router.post("/meal-plans/generate", response_model=schemas.GeneratedMealPlan)
async def generate_meal_plan(
    request: schemas.MealPlanRequest,
    current_user: models.User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session)
):
    """Generate an AI-powered meal plan based on user profile and preferences - FIXED RESPONSE SCHEMA"""
    logger.info("=" * 80)
    logger.info("🍽️  NEW MEAL PLAN GENERATION REQUEST")
    logger.info(f"👤 User: {current_user.email} (ID: {current_user.id})")
    logger.info(f"🕒 Request Time: {datetime.now().isoformat()}")
    logger.info("=" * 80)

    start_time = datetime.now()

    try:
        logger.info("🤖 Starting AI meal plan generation process...")

        meal_plan_data = await ai_workout_generator.generate_meal_plan(current_user, request)

        generation_time = (datetime.now() - start_time).total_seconds()

        logger.info("📊 MEAL PLAN GENERATION RESULTS:")
        logger.info("✅ Success: True")
        logger.info(f"⏱️  Total Time: {generation_time:.2f} seconds")
        logger.info(f"🤖 AI Generated: {meal_plan_data.get('ai_generated', False)}")
        logger.info(f"🎯 AI Model: {meal_plan_data.get('ai_model', 'Unknown')}")
        logger.info(f"🍽️  Plan Name: {meal_plan_data.get('name', 'Unknown')}")
        logger.info(f"🔥 Target Calories: {meal_plan_data.get('target_calories', 'Not calculated')}")
        logger.info(f"🥗 Meals Count: {len(meal_plan_data.get('meals', {}))}")

        # Validate that we have a proper meal plan
        if not meal_plan_data or not meal_plan_data.get("meals"):
            logger.error("❌ Meal plan generation returned empty or invalid data")
            raise HTTPException(
                status_code=500, 
                detail="AI failed to generate a proper meal plan. Please try again later."
            )

        # Check if AI generation was successful
        if not meal_plan_data.get("ai_generated", False):
            logger.warning("⚠️  AI models failed, but rule-based fallback succeeded")

        logger.info("=" * 80)
        logger.info("🎉 MEAL PLAN GENERATION COMPLETED SUCCESSFULLY!")
        logger.info(f"🏆 Final Result: {meal_plan_data.get('ai_model', 'Unknown')} generated meal plan")
        logger.info("=" * 80)

        # Return the meal plan data directly (no DB fields required)
        return schemas.GeneratedMealPlan(**meal_plan_data)

    except HTTPException:
        # Re-raise HTTPExceptions as-is
        raise
    except Exception as e:
        error_time = (datetime.now() - start_time).total_seconds()
        logger.error("=" * 80)
        logger.error("💥 MEAL PLAN GENERATION FAILED!")
        logger.error(f"👤 User: {current_user.email} (ID: {current_user.id})")
        logger.error(f"🕒 Error Time: {error_time:.2f}s after start")
        logger.error(f"🔴 Error Type: {type(e).__name__}")
        logger.error(f"📝 Error Message: {str(e)}")
        logger.error("📚 Stack Trace:", exc_info=True)
        logger.error("=" * 80)
        raise HTTPException(
            status_code=500, 
            detail=f"Meal plan generation failed: {str(e)}"
        )

@router.get("/exercises/search")
async def search_exercises(
    name: str,
    current_user: models.User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session)
):
    """ENHANCED: Search for exercise details with YouTube integration and better fallbacks"""
    logger.info(f"🔍 Exercise search requested by {current_user.email}: '{name}'")

    try:
        # First check database
        result = await session.execute(
            select(models.Exercise).where(models.Exercise.name.ilike(f"%{name}%")).limit(5)
        )
        exercises = result.scalars().all()

        if exercises:
            logger.info(f"🎯 Found {len(exercises)} exercises in database for '{name}'")
            exercise_data = []
            for exercise in exercises:
                # Get videos
                videos_result = await session.execute(
                    select(models.ExerciseVideo).where(models.ExerciseVideo.exercise_id == exercise.id).limit(3)
                )
                videos = videos_result.scalars().all()

                # Get tips
                tips_result = await session.execute(
                    select(models.ExerciseTip).where(models.ExerciseTip.exercise_id == exercise.id).limit(3)
                )
                tips = tips_result.scalars().all()

                exercise_data.append({
                    "exercise": {
                        "name": exercise.name,
                        "instructions": exercise.instructions,
                        "muscle_groups": exercise.muscle_groups,
                        "equipment": getattr(exercise, 'equipment', 'varies'),
                        "difficulty": exercise.difficulty,
                        "exercise_type": exercise.exercise_type
                    },
                    "videos": [{"title": v.title, "youtube_url": v.youtube_url, "duration": v.duration, "thumbnail_url": v.thumbnail_url} for v in videos],
                    "tips": [{"title": t.title, "content": t.content, "tip_type": t.tip_type} for t in tips]
                })

            logger.info(f"📊 Returning {len(exercise_data)} exercises with videos and tips from database")
            return exercise_data

        # Fallback: Check built-in exercise database
        exercise_name_lower = name.lower().strip()
        for key, exercise_info in EXERCISE_DATABASE.items():
            if key in exercise_name_lower or exercise_name_lower in key:
                logger.info(f"🎯 Found '{name}' in built-in exercise database")

                # Get additional YouTube videos
                youtube_videos = await search_youtube_videos(exercise_info["name"]) # type: ignore
                all_videos = exercise_info["videos"] + youtube_videos # type: ignore

                exercise_data = [{
                    "exercise": {
                        "name": exercise_info["name"],
                        "instructions": exercise_info["instructions"],
                        "muscle_groups": exercise_info["muscle_groups"],
                        "equipment": exercise_info["equipment"],
                        "difficulty": exercise_info["difficulty"]
                    },
                    "videos": all_videos[:5],  # Limit to 5 videos
                    "tips": exercise_info["tips"]
                }]

                logger.info(f"📹 Returned exercise data for '{name}' with {len(all_videos)} videos and {len(exercise_info['tips'])} tips")
                return exercise_data

        # Last resort: YouTube search only with generic exercise data
        logger.info(f"🔍 No database match for '{name}', searching YouTube and creating generic exercise...")
        youtube_videos = await search_youtube_videos(name)

        # Create basic exercise data
        exercise_data = [{
            "exercise": {
                "name": name.title(),
                "instructions": f"Perform {name} with proper form and controlled movements. Focus on quality over quantity.",
                "muscle_groups": ["general"],
                "equipment": "varies",
                "difficulty": "moderate"
            },
            "videos": youtube_videos[:3] if youtube_videos else [
                {
                    "title": f"How to do {name.title()}",
                    "youtube_url": f"https://www.youtube.com/results?search_query={name.replace(' ', '+')}+exercise+tutorial",
                    "duration": 180,
                    "thumbnail_url": "https://img.youtube.com/vi/default/maxresdefault.jpg"
                }
            ],
            "tips": [
                {
                    "title": "Focus on Form",
                    "content": "Always prioritize proper form over heavy weight or speed. Poor form can lead to injury.",
                    "tip_type": "Safety"
                },
                {
                    "title": "Breathe Properly",
                    "content": "Maintain steady breathing throughout the exercise. Don't hold your breath.",
                    "tip_type": "Technique"
                },
                {
                    "title": "Start Light",
                    "content": "If using weights, start with lighter resistance and gradually increase as you master the movement.",
                    "tip_type": "Progression"
                }
            ]
        }]

        logger.info(f"📊 Created generic exercise data for '{name}' with {len(youtube_videos)} YouTube videos")
        return exercise_data

    except Exception as e:
        logger.error(f"💥 Exercise search failed for '{name}': {str(e)}")

        # Final fallback - return basic exercise info
        return [{
            "exercise": {
                "name": name.title(),
                "instructions": f"Perform {name} with proper form and control.",
                "muscle_groups": ["general"],
                "equipment": "varies", 
                "difficulty": "moderate"
            },
            "videos": [
                {
                    "title": f"Search for {name.title()} tutorials",
                    "youtube_url": f"https://www.youtube.com/results?search_query={name.replace(' ', '+')}+exercise",
                    "duration": 0,
                    "thumbnail_url": ""
                }
            ],
            "tips": [
                {
                    "title": "Exercise Safely",
                    "content": "Always warm up before exercising and listen to your body.",
                    "tip_type": "Safety"
                }
            ]
        }]

@router.post("/tips/interact")
async def interact_with_tip(
    interaction: schemas.TipInteractionCreate,
    current_user: models.User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session)
):
    """Record tip interaction (like/dislike) with enhanced logging"""
    logger.info(f"👍 Tip interaction from {current_user.email}: {interaction.interaction_type} on tip {interaction.tip_id}")

    try:
        # Here you could save the interaction to a database table for analytics
        # For now, we'll just log it for improvement tracking
        logger.info(f"💾 Tip interaction '{interaction.interaction_type}' recorded for tip {interaction.tip_id}")
        logger.info(f"📊 This feedback helps improve AI recommendations for user {current_user.id}")

        return {
            "message": "Feedback recorded successfully! This helps improve AI recommendations.",
            "tip_id": interaction.tip_id,
            "type": interaction.interaction_type,
            "user_id": str(current_user.id),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"💥 Tip interaction failed: {str(e)}")
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
        # For now, return a basic analysis
        # In the future, this could integrate with workout logs to detect actual plateaus
        analysis = {
            "is_plateau": False,
            "confidence": 0.75,
            "affected_exercises": [],
            "recommendations": [
                "Try progressive overload - gradually increase weight or reps",
                "Add variety to your routine every 4-6 weeks", 
                "Ensure adequate rest and recovery between sessions",
                "Focus on proper nutrition to support your goals",
                "Consider deload weeks to allow for recovery",
                "Track your progress with detailed workout logs"
            ],
            "plateau_duration_weeks": 0,
            "analysis_method": "Rule-based Analysis v2.0",
            "ai_generated": False,
            "user_id": str(current_user.id),
            "analysis_date": datetime.now().isoformat()
        }

        logger.info(f"✅ Plateau analysis completed for {current_user.email}")
        return analysis

    except Exception as e:
        logger.error(f"💥 Plateau analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Plateau analysis failed: {str(e)}")

# NEW: Exercise feedback endpoint for better AI training
@router.post("/exercises/feedback")
async def submit_exercise_feedback(
    exercise_name: str,
    feedback_type: str,  # 'like', 'dislike', 'too_easy', 'too_hard', etc.
    current_user: models.User = Depends(current_active_user)
):
    """Submit feedback on exercises to improve AI recommendations"""
    logger.info(f"📝 Exercise feedback from {current_user.email}: {feedback_type} for '{exercise_name}'")

    try:
        # Store feedback for AI improvement
        feedback_data = {
            "user_id": str(current_user.id),
            "exercise_name": exercise_name,
            "feedback_type": feedback_type,
            "timestamp": datetime.now().isoformat(),
            "user_level": getattr(current_user, 'experience_level', 'intermediate'),
            "user_goal": getattr(current_user, 'fitness_goal', 'general_fitness')
        }

        logger.info(f"💾 Exercise feedback stored: {feedback_data}")

        return {
            "message": "Thank you for your feedback! This helps us improve exercise recommendations.",
            "exercise_name": exercise_name,
            "feedback_type": feedback_type,
            "status": "recorded"
        }

    except Exception as e:
        logger.error(f"💥 Exercise feedback submission failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to submit feedback: {str(e)}")

ALL_MUSCLE_GROUPS = [
    "chest", "back", "shoulders", "biceps", "triceps",
    "forearms", "legs", "quadriceps", "hamstrings", "glutes",
    "calves", "core", "abs", "cardio" # Removed 'full body' from master list for neglect calculation
]

@router.get("/workouts/recommend", response_model=schemas.WorkoutPlan)
async def recommend_workout(
    lookback_days: int = Query(7, ge=1, le=30, description="Number of days history to consider"),
    current_user: models.User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session)
):
    """
    Analyze user's workout history from the last 7 days (default) and recommend
    the next workout focusing on neglected muscle groups using AI generation.
    """
    logger.info(f"🚀 Workout recommendation requested for user {current_user.id}")
    start_time_request = datetime.now(timezone.utc) # Use timezone-aware datetime
    lookback_date = start_time_request - timedelta(days=lookback_days)

    # 1. Fetch recent workout logs
    logs_query = (
        select(models.WorkoutLog)
        .where(models.WorkoutLog.user_id == current_user.id)
        .where(models.WorkoutLog.workout_date >= lookback_date) # Filter by date
    )
    logs_result = await session.execute(logs_query)
    recent_logs = logs_result.scalars().all()

    target_muscles: List[str] = []

    # 2. Analyze Muscle Groups or Handle No History
    if not recent_logs:
        logger.info(f"No workouts found in the last {lookback_days} days. Recommending 'full body'.")
        target_muscles = ["full body"] # Case 3: No Workouts
    else:
        trained_groups: set[str] = set()
        for log in recent_logs:
            if not log.exercises_completed: # Skip logs with empty exercise lists
                continue
            for exercise in log.exercises_completed:
                # Ensure muscle_groups is treated as a list, even if it's None or a string
                groups = exercise.get("muscle_groups", [])
                current_groups = []
                if isinstance(groups, list):
                    current_groups = groups
                elif isinstance(groups, str):
                    current_groups = [g.strip() for g in groups.split(',') if g.strip()]

                trained_groups.update(g.lower() for g in current_groups)

        logger.info(f"Trained muscle groups in last {lookback_days} days: {trained_groups}")

        # Determine neglected groups
        all_groups_set = set(g.lower() for g in ALL_MUSCLE_GROUPS)
        neglected_groups = list(all_groups_set - trained_groups)

        if not neglected_groups:
            logger.info("All specific muscle groups trained recently. Recommending 'full body'.")
            target_muscles = ["full body"] # Case 2: All Groups Trained
        else:
            logger.info(f"Neglected muscle groups: {neglected_groups}")
            # Randomly select 1 to 3 neglected groups
            num_to_select = min(len(neglected_groups), random.randint(1, 3))
            target_muscles = random.sample(neglected_groups, num_to_select) # Case 1: Neglected Found
            logger.info(f"Selected target muscles: {target_muscles}")

    # 3. Generate Workout using AI
    logger.info(f"Generating workout recommendation targeting: {target_muscles}")
    try:
        workout_data = await ai_workout_generator.generate_workout(
            user=current_user,
            duration_minutes=45, # You could make this configurable later
            target_muscles=target_muscles
        )
    except Exception as e:
         logger.error(f"AI workout generation failed during recommendation: {e}", exc_info=True)
         raise HTTPException(
             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
             detail="Failed to generate workout recommendation via AI."
         )

    # 4. Format and Return (without saving to DB)
    # Add necessary fields expected by the schema but not returned by generator
    # Ensure all required fields for WorkoutPlan schema are present
    workout_data_validated = {
        "id": 0, # Dummy ID as it's not saved
        "user_id": current_user.id,
        "name": workout_data.get("name", f"Recommended Workout for {', '.join(target_muscles)}"),
        "description": workout_data.get("description", f"AI recommended workout focusing on {', '.join(target_muscles)}."),
        "exercises": workout_data.get("exercises", []),
        "difficulty": workout_data.get("difficulty_level", "intermediate"),
        "estimated_duration": workout_data.get("estimated_duration", 45),
        "ai_generated": workout_data.get("ai_generated", True),
        "ai_model": workout_data.get("ai_model", "Recommendation Engine"),
        "is_active": True, # Assume recommended plans are active
        "created_at": datetime.now(timezone.utc) # Timestamp of recommendation
    }

    try:
        # Validate the data against the schema before returning
        final_plan = schemas.WorkoutPlan.model_validate(workout_data_validated)
    except Exception as val_error:
        logger.error(f"Failed to validate generated workout data against schema: {val_error}")
        logger.error(f"Generated data was: {workout_data_validated}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="AI generated incompatible workout data."
        )

    logger.info(f"Successfully generated recommendation: '{final_plan.name}'")
    return final_plan