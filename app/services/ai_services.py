# app/services/ai_services.py - Enhanced with detailed logging

import json
import requests
import random
import numpy as np
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from app.core.config import settings

# Set up logging
logger = logging.getLogger(__name__)

class GroqAI:
    """Free AI using Groq's fast inference API"""
    def __init__(self):
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {settings.GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        self.model = "llama3-8b-8192"

    def generate_text(self, prompt: str, max_tokens: int = 500) -> Optional[str]:
        """Generate text using Groq's free API"""
        logger.info(f"🤖 GROQ AI: Starting text generation with model {self.model}")
        
        if not settings.GROQ_API_KEY:
            logger.error("🚨 GROQ AI: API key not configured")
            return None
        
        try:
            payload = {
                "messages": [{"role": "user", "content": prompt}],
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": 0.7
            }
            
            logger.info(f"🚀 GROQ AI: Sending request to {self.api_url}")
            logger.debug(f"🔍 GROQ AI: Payload - Model: {self.model}, Max tokens: {max_tokens}")
            
            start_time = datetime.now()
            response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=30)
            end_time = datetime.now()
            response_time = (end_time - start_time).total_seconds()
            
            logger.info(f"⏱️ GROQ AI: Response received in {response_time:.2f} seconds")
            
            if response.status_code == 200:
                data = response.json()
                generated_text = data["choices"][0]["message"]["content"]
                logger.success(f"✅ GROQ AI: Successfully generated {len(generated_text)} characters")
                logger.debug(f"🔤 GROQ AI: Generated text preview: {generated_text[:100]}...")
                return generated_text
            else:
                logger.error(f"❌ GROQ AI: Request failed with status {response.status_code}")
                logger.error(f"🔍 GROQ AI: Error response: {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            logger.error("⏰ GROQ AI: Request timed out after 30 seconds")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"🌐 GROQ AI: Network error: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"💥 GROQ AI: Unexpected error: {str(e)}")
            return None


class HuggingFaceAI:
    """Alternative free AI using HuggingFace Inference API"""
    def __init__(self):
        self.api_url = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-large"
        self.headers = {"Authorization": f"Bearer {settings.HUGGINGFACE_API_KEY}"}

    def generate_text(self, prompt: str, max_length: int = 500) -> Optional[str]:
        """Generate text using HuggingFace API"""
        logger.info("🤗 HUGGINGFACE AI: Starting text generation")
        
        if not settings.HUGGINGFACE_API_KEY:
            logger.error("🚨 HUGGINGFACE AI: API key not configured")
            return None
            
        try:
            payload = {
                "inputs": prompt,
                "parameters": {"max_length": max_length, "temperature": 0.7}
            }
            
            logger.info(f"🚀 HUGGINGFACE AI: Sending request to {self.api_url}")
            
            start_time = datetime.now()
            response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=30)
            end_time = datetime.now()
            response_time = (end_time - start_time).total_seconds()
            
            logger.info(f"⏱️ HUGGINGFACE AI: Response received in {response_time:.2f} seconds")
            
            if response.status_code == 200:
                result = response.json()
                if result and len(result) > 0:
                    generated_text = result[0].get("generated_text", "")
                    logger.success(f"✅ HUGGINGFACE AI: Successfully generated {len(generated_text)} characters")
                    return generated_text
                    
            logger.error(f"❌ HUGGINGFACE AI: Request failed with status {response.status_code}")
            return None
            
        except Exception as e:
            logger.error(f"💥 HUGGINGFACE AI: Error: {str(e)}")
            return None


class OllamaAI:
    """Local AI using Ollama"""
    def __init__(self):
        self.api_url = f"{settings.OLLAMA_URL}/api/generate"
        self.model = "llama2"

    def generate_text(self, prompt: str) -> Optional[str]:
        """Generate text using local Ollama"""
        logger.info(f"🏠 OLLAMA AI: Starting text generation with model {self.model}")
        
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False
            }
            
            logger.info(f"🚀 OLLAMA AI: Sending request to {self.api_url}")
            
            start_time = datetime.now()
            response = requests.post(self.api_url, json=payload, timeout=60)
            end_time = datetime.now()
            response_time = (end_time - start_time).total_seconds()
            
            logger.info(f"⏱️ OLLAMA AI: Response received in {response_time:.2f} seconds")
            
            if response.status_code == 200:
                data = response.json()
                generated_text = data.get("response", "")
                logger.success(f"✅ OLLAMA AI: Successfully generated {len(generated_text)} characters")
                return generated_text
            else:
                logger.error(f"❌ OLLAMA AI: Request failed with status {response.status_code}")
                return None
                
        except requests.exceptions.ConnectionError:
            logger.error("🔌 OLLAMA AI: Connection failed - Ollama server not running")
            return None
        except requests.exceptions.Timeout:
            logger.error("⏰ OLLAMA AI: Request timed out after 60 seconds")
            return None
        except Exception as e:
            logger.error(f"💥 OLLAMA AI: Error: {str(e)}")
            return None


class AIWorkoutGenerator:
    def __init__(self):
        self.groq = GroqAI()
        self.hf = HuggingFaceAI()
        self.ollama = OllamaAI()
        
        # Exercise database for fallback
        self.exercise_database = {
            "chest": ["Push-ups", "Bench Press", "Chest Dips", "Incline Push-ups"],
            "back": ["Pull-ups", "Bent-over Rows", "Lat Pulldowns", "Superman"],
            "legs": ["Squats", "Lunges", "Leg Press", "Calf Raises"],
            "shoulders": ["Shoulder Press", "Lateral Raises", "Front Raises", "Pike Push-ups"],
            "arms": ["Bicep Curls", "Tricep Dips", "Hammer Curls", "Close-grip Push-ups"],
            "core": ["Plank", "Crunches", "Russian Twists", "Mountain Climbers"],
            "cardio": ["Jumping Jacks", "Burpees", "High Knees", "Jump Rope"]
        }

    def _create_user_profile(self, user) -> str:
        """Create detailed user profile string for AI"""
        profile_parts = []
        
        if user.age:
            profile_parts.append(f"Age: {user.age}")
        if user.weight:
            profile_parts.append(f"Weight: {user.weight}kg")
        if user.height:
            profile_parts.append(f"Height: {user.height}cm")
        if user.gender:
            profile_parts.append(f"Gender: {user.gender}")
        if user.fitness_goal:
            profile_parts.append(f"Goal: {user.fitness_goal.replace('_', ' ')}")
        if user.experience_level:
            profile_parts.append(f"Experience: {user.experience_level}")
        if user.activity_level:
            profile_parts.append(f"Activity Level: {user.activity_level.replace('_', ' ')}")
        
        return ", ".join(profile_parts) if profile_parts else "General fitness enthusiast"

    async def generate_workout(self, user, duration_minutes: int = 45) -> Dict[str, Any]:
        """Generate workout using multiple AI models with detailed logging"""
        
        # Log the workout generation attempt
        logger.info("=" * 80)
        logger.info(f"🏋️ WORKOUT GENERATION STARTED for user {user.id}")
        logger.info(f"📊 Request details: Duration={duration_minutes}min, Goal={user.fitness_goal}, Level={user.experience_level}")
        logger.info("=" * 80)
        
        user_profile = self._create_user_profile(user)
        logger.info(f"👤 User Profile: {user_profile}")
        
        workout_plan = None
        generation_method = None
        generation_time = None
        
        start_total_time = datetime.now()
        
        # Try AI services in priority order
        logger.info("🎯 Starting AI generation attempts...")
        
        # Method 1: Groq AI (Primary)
        logger.info("🔄 Attempting Method 1: Groq AI (Primary)")
        start_time = datetime.now()
        workout_plan = await self._generate_with_groq(user_profile, duration_minutes)
        if workout_plan:
            generation_method = "Groq AI"
            generation_time = (datetime.now() - start_time).total_seconds()
            logger.success(f"🎉 SUCCESS: Groq AI generated workout in {generation_time:.2f}s")
        else:
            logger.warning("⚠️ Groq AI failed, trying next method...")

        # Method 2: Ollama AI (Backup)
        if not workout_plan:
            logger.info("🔄 Attempting Method 2: Ollama AI (Local Backup)")
            start_time = datetime.now()
            workout_plan = await self._generate_with_ollama(user_profile, duration_minutes)
            if workout_plan:
                generation_method = "Ollama AI"
                generation_time = (datetime.now() - start_time).total_seconds()
                logger.success(f"🎉 SUCCESS: Ollama AI generated workout in {generation_time:.2f}s")
            else:
                logger.warning("⚠️ Ollama AI failed, trying next method...")

        # Method 3: HuggingFace AI (Secondary Backup)
        if not workout_plan:
            logger.info("🔄 Attempting Method 3: HuggingFace AI (Secondary Backup)")
            start_time = datetime.now()
            workout_plan = await self._generate_with_huggingface(user_profile, duration_minutes)
            if workout_plan:
                generation_method = "HuggingFace AI"
                generation_time = (datetime.now() - start_time).total_seconds()
                logger.success(f"🎉 SUCCESS: HuggingFace AI generated workout in {generation_time:.2f}s")
            else:
                logger.warning("⚠️ HuggingFace AI failed, using fallback...")

        # Method 4: Rule-based fallback (Guaranteed)
        if not workout_plan:
            logger.info("🔄 Attempting Method 4: Rule-based Fallback (Guaranteed)")
            start_time = datetime.now()
            workout_plan = self._generate_fallback_workout(user, duration_minutes)
            generation_method = "Rule-based System"
            generation_time = (datetime.now() - start_time).total_seconds()
            logger.success(f"🎉 SUCCESS: Rule-based system generated workout in {generation_time:.2f}s")

        total_time = (datetime.now() - start_total_time).total_seconds()
        
        # Final success logging
        logger.info("=" * 80)
        logger.success(f"✅ WORKOUT GENERATION COMPLETED")
        logger.info(f"🤖 Generation Method: {generation_method}")
        logger.info(f"⏱️ Generation Time: {generation_time:.2f}s")
        logger.info(f"🕐 Total Process Time: {total_time:.2f}s")
        logger.info(f"🏋️ Generated Workout: '{workout_plan.get('name', 'Unknown')}'")
        logger.info(f"📝 Exercise Count: {len(workout_plan.get('exercises', []))}")
        logger.info(f"🧠 AI Generated: {workout_plan.get('ai_generated', False)}")
        logger.info(f"🔧 AI Model: {workout_plan.get('ai_model', 'Unknown')}")
        logger.info("=" * 80)
        
        return workout_plan

    async def _generate_with_groq(self, user_profile: str, duration: int) -> Optional[Dict]:
        """Generate workout using Groq AI"""
        logger.info("🤖 Groq Generation: Creating AI prompt...")
        
        prompt = f"""Create a personalized {duration}-minute workout plan for a person with the following profile: {user_profile}

Generate a complete workout with 5-7 exercises. Each exercise should be practical and effective.

Respond ONLY with valid JSON in this exact format:
{{
  "name": "Custom Workout Name",
  "description": "Brief motivational description of the workout",
  "exercises": [
    {{
      "name": "Exercise Name",
      "sets": 3,
      "reps": "8-12",
      "duration_seconds": 30,
      "rest_seconds": 60,
      "instructions": "Clear step-by-step instructions",
      "muscle_groups": ["primary_muscle", "secondary_muscle"],
      "equipment": "bodyweight",
      "difficulty": "moderate"
    }}
  ],
  "estimated_duration": {duration},
  "estimated_calories": 300,
  "difficulty_level": "moderate"
}}

Make sure the JSON is valid and complete. No additional text."""

        logger.debug(f"📝 Groq Prompt length: {len(prompt)} characters")
        
        response = self.groq.generate_text(prompt, max_tokens=800)
        
        if response:
            logger.info("🔍 Groq: Parsing AI response...")
            try:
                # Extract JSON from response
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                
                if json_start != -1 and json_end > json_start:
                    json_str = response[json_start:json_end]
                    logger.debug(f"📄 Groq: Extracted JSON ({len(json_str)} chars)")
                    
                    workout_data = json.loads(json_str)
                    
                    # Add AI metadata
                    workout_data["ai_generated"] = True
                    workout_data["ai_model"] = "Groq Llama3-8B"
                    workout_data["generation_timestamp"] = datetime.now().isoformat()
                    
                    logger.success("✅ Groq: Successfully parsed workout JSON")
                    return workout_data
                else:
                    logger.error("❌ Groq: No valid JSON found in response")
                    
            except json.JSONDecodeError as e:
                logger.error(f"❌ Groq: JSON parsing failed: {str(e)}")
            except Exception as e:
                logger.error(f"❌ Groq: Unexpected parsing error: {str(e)}")
        
        return None

    async def _generate_with_ollama(self, user_profile: str, duration: int) -> Optional[Dict]:
        """Generate workout using Ollama AI"""
        logger.info("🏠 Ollama Generation: Creating AI prompt...")
        
        prompt = f"Create a {duration}-minute workout for: {user_profile}. List 5 exercises with sets and reps."
        
        response = self.ollama.generate_text(prompt)
        
        if response:
            logger.info("🔍 Ollama: Processing response...")
            # Parse the text response and create structured workout
            workout_plan = self._parse_workout_text(response, duration)
            workout_plan["ai_generated"] = True
            workout_plan["ai_model"] = "Ollama Llama2"
            workout_plan["generation_timestamp"] = datetime.now().isoformat()
            
            logger.success("✅ Ollama: Successfully created workout from text")
            return workout_plan
        
        return None

    async def _generate_with_huggingface(self, user_profile: str, duration: int) -> Optional[Dict]:
        """Generate workout using HuggingFace AI"""
        logger.info("🤗 HuggingFace Generation: Creating AI prompt...")
        
        prompt = f"Generate a {duration}-minute workout for: {user_profile}"
        
        response = self.hf.generate_text(prompt)
        
        if response:
            logger.info("🔍 HuggingFace: Processing response...")
            workout_plan = self._parse_workout_text(response, duration)
            workout_plan["ai_generated"] = True
            workout_plan["ai_model"] = "HuggingFace DialoGPT"
            workout_plan["generation_timestamp"] = datetime.now().isoformat()
            
            logger.success("✅ HuggingFace: Successfully created workout from text")
            return workout_plan
        
        return None

    def _parse_workout_text(self, text: str, duration: int) -> Dict:
        """Parse AI-generated text into structured workout"""
        logger.info("📝 Parsing AI text response into structured workout...")
        
        # Simple text parsing logic
        exercises = []
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if any(keyword in line.lower() for keyword in ['exercise', 'workout', 'rep', 'set']):
                # Extract exercise name (simplified)
                exercise_name = line.split(':')[0] if ':' in line else line
                exercises.append({
                    "name": exercise_name.strip(),
                    "sets": 3,
                    "reps": "10-12",
                    "duration_seconds": 30,
                    "rest_seconds": 60,
                    "instructions": f"Perform {exercise_name} with proper form",
                    "muscle_groups": ["general"],
                    "equipment": "bodyweight"
                })
        
        # Ensure we have at least some exercises
        if len(exercises) < 3:
            exercises = [
                {"name": "Push-ups", "sets": 3, "reps": "10-15", "muscle_groups": ["chest"]},
                {"name": "Squats", "sets": 3, "reps": "12-15", "muscle_groups": ["legs"]},
                {"name": "Plank", "sets": 3, "reps": "30s hold", "muscle_groups": ["core"]}
            ]
        
        return {
            "name": f"AI Generated {duration}-min Workout",
            "description": "Personalized workout created by AI",
            "exercises": exercises[:7],  # Max 7 exercises
            "estimated_duration": duration,
            "estimated_calories": duration * 5  # Rough estimate
        }

    def _generate_fallback_workout(self, user, duration: int) -> Dict[str, Any]:
        """Generate rule-based workout when AI fails"""
        logger.info("🎯 Rule-based Generation: Creating structured workout...")
        
        goal = user.fitness_goal or "general_fitness"
        level = user.experience_level or "beginner"
        
        logger.info(f"🎯 Rule-based: Goal={goal}, Level={level}")
        
        # Select muscle groups based on goal
        if goal == "weight_loss":
            focus_groups = ["cardio", "legs", "core", "chest", "back"]
        elif goal == "muscle_gain":
            focus_groups = ["chest", "back", "legs", "shoulders", "arms"]
        elif goal == "strength":
            focus_groups = ["legs", "chest", "back", "shoulders", "core"]
        else:
            focus_groups = ["chest", "back", "legs", "core", "arms"]
        
        exercises = []
        for i, muscle_group in enumerate(focus_groups[:6]):
            exercise_name = random.choice(self.exercise_database[muscle_group])
            exercises.append({
                "name": exercise_name,
                "sets": 2 if level == "beginner" else 3 if level == "intermediate" else 4,
                "reps": "12-15" if goal == "weight_loss" else "8-12" if goal == "muscle_gain" else "10-12",
                "duration_seconds": 45 if muscle_group == "cardio" else 30,
                "rest_seconds": 45 if level == "beginner" else 60 if level == "intermediate" else 75,
                "instructions": f"Perform {exercise_name} with controlled movements and proper form",
                "muscle_groups": [muscle_group],
                "equipment": "bodyweight",
                "difficulty": level
            })
        
        workout_plan = {
            "name": f"Smart {goal.replace('_', ' ').title()} Workout",
            "description": f"Intelligently designed {level}-level workout for {goal.replace('_', ' ')} goals",
            "exercises": exercises,
            "estimated_duration": duration,
            "estimated_calories": duration * (4 if level == "beginner" else 5 if level == "intermediate" else 6),
            "difficulty_level": level,
            "ai_generated": False,
            "ai_model": "Rule-Based Algorithm",
            "generation_timestamp": datetime.now().isoformat()
        }
        
        logger.success(f"✅ Rule-based: Generated {len(exercises)} exercises")
        return workout_plan


# Create global instance
ai_workout_generator = AIWorkoutGenerator()

# Custom logging formatter for colorful logs
class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels"""
    
    grey = "\x1b[38;21m"
    blue = "\x1b[34m"
    yellow = "\x1b[33m"
    red = "\x1b[31m"
    green = "\x1b[32m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    
    FORMATS = {
        logging.DEBUG: grey + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset,
        logging.INFO: blue + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset,
        logging.WARNING: yellow + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset,
        logging.ERROR: red + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset,
        logging.CRITICAL: bold_red + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt='%Y-%m-%d %H:%M:%S')
        return formatter.format(record)

# Set up enhanced logging
def setup_ai_logging():
    """Set up enhanced logging for AI services"""
    logger.setLevel(logging.DEBUG)
    
    # Console handler with colors
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(ColoredFormatter())
    logger.addHandler(console_handler)
    
    # File handler for persistent logs
    file_handler = logging.FileHandler('ai_generation.log')
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

# Initialize logging
setup_ai_logging()

# Add custom success level
def success(self, message, *args, **kwargs):
    if self.isEnabledFor(25):
        self._log(25, message, args, **kwargs)

logging.Logger.success = success
logging.addLevelName(25, "SUCCESS")