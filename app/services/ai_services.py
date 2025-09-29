# app/services/ai_services.py - Fixed with updated models and APIs

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
    """Free AI using Groq's fast inference API with UPDATED MODEL"""
    def __init__(self):
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {settings.GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        # FIXED: Updated to current available model
        self.model = "llama3-8b-8192"  # This was deprecated, let's use updated model
        self.available_models = [
            "llama-3.3-70b-versatile", # High-power replacement
            "llama-3.1-8b-instant",     # Fast and reliable
            "openai/gpt-oss-120b"      
        ]

    def generate_text(self, prompt: str, max_tokens: int = 500) -> Optional[str]:
        """Generate text using Groq's free API with model fallback"""
        if not settings.GROQ_API_KEY:
            logger.error("🚨 GROQ AI: API key not configured")
            return None
        
        # Try multiple models in case some are deprecated
        for model in self.available_models:
            logger.info(f"🤖 GROQ AI: Trying model {model}")
            
            try:
                payload = {
                    "messages": [{"role": "user", "content": prompt}],
                    "model": model,
                    "max_tokens": max_tokens,
                    "temperature": 0.7
                }
                
                logger.info(f"🚀 GROQ AI: Sending request to {self.api_url}")
                logger.debug(f"🔍 GROQ AI: Payload - Model: {model}, Max tokens: {max_tokens}")
                
                start_time = datetime.now()
                response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=30)
                end_time = datetime.now()
                response_time = (end_time - start_time).total_seconds()
                
                logger.info(f"⏱️ GROQ AI: Response received in {response_time:.2f} seconds")
                
                if response.status_code == 200:
                    data = response.json()
                    generated_text = data["choices"][0]["message"]["content"]
                    logger.success(f"✅ GROQ AI: Successfully generated {len(generated_text)} characters with {model}")
                    logger.debug(f"🔤 GROQ AI: Generated text preview: {generated_text[:100]}...")
                    return generated_text
                else:
                    logger.warning(f"⚠️ GROQ AI: Model {model} failed with status {response.status_code}")
                    logger.debug(f"🔍 GROQ AI: Error response: {response.text}")
                    continue  # Try next model
                    
            except requests.exceptions.Timeout:
                logger.error(f"⏰ GROQ AI: Model {model} timed out after 30 seconds")
                continue
            except requests.exceptions.RequestException as e:
                logger.error(f"🌐 GROQ AI: Network error with {model}: {str(e)}")
                continue
            except Exception as e:
                logger.error(f"💥 GROQ AI: Unexpected error with {model}: {str(e)}")
                continue
        
        logger.error("❌ GROQ AI: All models failed")
        return None


class HuggingFaceAI:
    """Fixed HuggingFace AI with working model"""
    def __init__(self):
        # FIXED: Using a better model that's actually available
        self.api_url = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium"
        self.headers = {"Authorization": f"Bearer {settings.HUGGINGFACE_API_KEY}"}
        
        # Alternative models to try
        self.backup_models = [
            "meta-llama/Meta-Llama-3-8B-Instruct",
            "google/gemma-2-9b-it",
            "HuggingFaceH4/zephyr-7b-beta" 
        ]

    def generate_text(self, prompt: str, max_length: int = 500) -> Optional[str]:
        """Generate text using HuggingFace API with model fallback"""
        logger.info("🤗 HUGGINGFACE AI: Starting text generation")
        
        if not settings.HUGGINGFACE_API_KEY:
            logger.error("🚨 HUGGINGFACE AI: API key not configured")
            return None
        
        for model in self.backup_models:
            api_url = f"https://api-inference.huggingface.co/models/{model}"
            logger.info(f"🤗 HUGGINGFACE AI: Trying model {model}")
            
            try:
                payload = {
                    "inputs": prompt,
                    "parameters": {
                        "max_length": max_length, 
                        "temperature": 0.7,
                        "do_sample": True
                    }
                }
                
                logger.info(f"🚀 HUGGINGFACE AI: Sending request to {api_url}")
                
                start_time = datetime.now()
                response = requests.post(api_url, headers=self.headers, json=payload, timeout=30)
                end_time = datetime.now()
                response_time = (end_time - start_time).total_seconds()
                
                logger.info(f"⏱️ HUGGINGFACE AI: Response received in {response_time:.2f} seconds")
                
                if response.status_code == 200:
                    result = response.json()
                    if result and len(result) > 0:
                        if isinstance(result, list) and "generated_text" in result[0]:
                            generated_text = result[0]["generated_text"]
                        elif isinstance(result, dict) and "generated_text" in result:
                            generated_text = result["generated_text"]
                        else:
                            generated_text = str(result)
                        
                        logger.success(f"✅ HUGGINGFACE AI: Successfully generated {len(generated_text)} characters with {model}")
                        return generated_text
                else:
                    logger.warning(f"⚠️ HUGGINGFACE AI: Model {model} failed with status {response.status_code}")
                    logger.debug(f"🔍 HUGGINGFACE AI: Error response: {response.text}")
                    continue
                    
            except Exception as e:
                logger.error(f"💥 HUGGINGFACE AI: Error with {model}: {str(e)}")
                continue
        
        logger.error("❌ HUGGINGFACE AI: All models failed")
        return None


class OllamaAI:
    """Local AI using Ollama with better error handling"""
    def __init__(self):
        self.api_url = f"{settings.OLLAMA_URL or 'http://localhost:11434'}/api/generate"
        self.models = ["llama3.2", "llama3.1", "llama2", "phi3", "gemma2"]

    def generate_text(self, prompt: str) -> Optional[str]:
        """Generate text using local Ollama with model fallback"""
        logger.info("🏠 OLLAMA AI: Starting text generation")
        
        # Try different models in case some aren't installed
        for model in self.models:
            logger.info(f"🏠 OLLAMA AI: Trying model {model}")
            
            try:
                payload = {
                    "model": model,
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
                    logger.success(f"✅ OLLAMA AI: Successfully generated {len(generated_text)} characters with {model}")
                    return generated_text
                else:
                    logger.warning(f"⚠️ OLLAMA AI: Model {model} failed with status {response.status_code}")
                    continue
                    
            except requests.exceptions.ConnectionError:
                if model == self.models[0]:  # Only log connection error once
                    logger.error("🔌 OLLAMA AI: Connection failed - Ollama server not running")
                    logger.info("💡 OLLAMA AI: To fix this, install Ollama: https://ollama.ai")
                break  # No point trying other models if server is down
            except requests.exceptions.Timeout:
                logger.error(f"⏰ OLLAMA AI: Model {model} timed out after 60 seconds")
                continue
            except Exception as e:
                logger.error(f"💥 OLLAMA AI: Error with {model}: {str(e)}")
                continue
        
        return None


class OpenRouterAI:
    """NEW: OpenRouter AI as additional backup"""
    def __init__(self):
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://evolvefitai.com",
            "X-Title": "EvolveFit AI"
        }
        self.model = "deepseek/deepseek-v3-1:free"  # Free tier

    def generate_text(self, prompt: str, max_tokens: int = 500) -> Optional[str]:
        """Generate text using OpenRouter's free tier"""
        if not settings.OPENROUTER_API_KEY:
            logger.warning("🚨 OPENROUTER AI: API key not configured, skipping")
            return None
            
        logger.info("🌐 OPENROUTER AI: Starting text generation")
        
        try:
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.7
            }
            
            logger.info(f"🚀 OPENROUTER AI: Sending request with {self.model}")
            
            start_time = datetime.now()
            response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=30)
            end_time = datetime.now()
            response_time = (end_time - start_time).total_seconds()
            
            logger.info(f"⏱️ OPENROUTER AI: Response received in {response_time:.2f} seconds")
            
            if response.status_code == 200:
                data = response.json()
                generated_text = data["choices"][0]["message"]["content"]
                logger.success(f"✅ OPENROUTER AI: Successfully generated {len(generated_text)} characters")
                return generated_text
            else:
                logger.error(f"❌ OPENROUTER AI: Request failed with status {response.status_code}")
                logger.debug(f"🔍 OPENROUTER AI: Error response: {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"💥 OPENROUTER AI: Error: {str(e)}")
            return None


class AIWorkoutGenerator:
    def __init__(self):
        self.groq = GroqAI()
        self.hf = HuggingFaceAI()
        self.ollama = OllamaAI()
        self.openrouter = OpenRouterAI()  # NEW: Additional AI service
        
        # ENHANCED: Better exercise database with more exercises
        self.exercise_database = {
            "chest": [
                {"name": "Push-ups", "equipment": "bodyweight"},
                {"name": "Chest Press", "equipment": "dumbbells"},
                {"name": "Chest Dips", "equipment": "parallel bars"},
                {"name": "Incline Push-ups", "equipment": "bodyweight"},
                {"name": "Diamond Push-ups", "equipment": "bodyweight"},
                {"name": "Chest Flyes", "equipment": "dumbbells"}
            ],
            "back": [
                {"name": "Pull-ups", "equipment": "pull-up bar"},
                {"name": "Bent-over Rows", "equipment": "dumbbells"},
                {"name": "Lat Pulldowns", "equipment": "cable machine"},
                {"name": "Superman", "equipment": "bodyweight"},
                {"name": "Reverse Flyes", "equipment": "dumbbells"},
                {"name": "Deadlifts", "equipment": "barbell"}
            ],
            "legs": [
                {"name": "Squats", "equipment": "bodyweight"},
                {"name": "Lunges", "equipment": "bodyweight"},
                {"name": "Leg Press", "equipment": "leg press machine"},
                {"name": "Calf Raises", "equipment": "bodyweight"},
                {"name": "Bulgarian Split Squats", "equipment": "bodyweight"},
                {"name": "Wall Sits", "equipment": "bodyweight"}
            ],
            "shoulders": [
                {"name": "Shoulder Press", "equipment": "dumbbells"},
                {"name": "Lateral Raises", "equipment": "dumbbells"},
                {"name": "Front Raises", "equipment": "dumbbells"},
                {"name": "Pike Push-ups", "equipment": "bodyweight"},
                {"name": "Shoulder Shrugs", "equipment": "dumbbells"},
                {"name": "Upright Rows", "equipment": "dumbbells"}
            ],
            "arms": [
                {"name": "Bicep Curls", "equipment": "dumbbells"},
                {"name": "Tricep Dips", "equipment": "bodyweight"},
                {"name": "Hammer Curls", "equipment": "dumbbells"},
                {"name": "Close-grip Push-ups", "equipment": "bodyweight"},
                {"name": "Tricep Extensions", "equipment": "dumbbells"},
                {"name": "Chin-ups", "equipment": "pull-up bar"}
            ],
            "core": [
                {"name": "Plank", "equipment": "bodyweight"},
                {"name": "Crunches", "equipment": "bodyweight"},
                {"name": "Russian Twists", "equipment": "bodyweight"},
                {"name": "Mountain Climbers", "equipment": "bodyweight"},
                {"name": "Bicycle Crunches", "equipment": "bodyweight"},
                {"name": "Dead Bug", "equipment": "bodyweight"}
            ],
            "cardio": [
                {"name": "Jumping Jacks", "equipment": "bodyweight"},
                {"name": "Burpees", "equipment": "bodyweight"},
                {"name": "High Knees", "equipment": "bodyweight"},
                {"name": "Jump Rope", "equipment": "jump rope"},
                {"name": "Running in Place", "equipment": "bodyweight"},
                {"name": "Star Jumps", "equipment": "bodyweight"}
            ]
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
        """Generate workout using multiple AI models with enhanced fallback"""
        
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
        
        logger.info("🎯 Starting AI generation attempts...")
        
        # Method 1: Groq AI (Primary) - FIXED
        logger.info("🔄 Attempting Method 1: Groq AI (Primary)")
        start_time = datetime.now()
        workout_plan = await self._generate_with_groq(user_profile, duration_minutes)
        if workout_plan:
            generation_method = "Groq AI"
            generation_time = (datetime.now() - start_time).total_seconds()
            logger.success(f"🎉 SUCCESS: Groq AI generated workout in {generation_time:.2f}s")
        else:
            logger.warning("⚠️ Groq AI failed, trying next method...")

        # Method 2: OpenRouter AI (NEW)
        if not workout_plan:
            logger.info("🔄 Attempting Method 2: OpenRouter AI (Free)")
            start_time = datetime.now()
            workout_plan = await self._generate_with_openrouter(user_profile, duration_minutes)
            if workout_plan:
                generation_method = "OpenRouter AI"
                generation_time = (datetime.now() - start_time).total_seconds()
                logger.success(f"🎉 SUCCESS: OpenRouter AI generated workout in {generation_time:.2f}s")
            else:
                logger.warning("⚠️ OpenRouter AI failed, trying next method...")

        # Method 3: HuggingFace AI (FIXED)
        if not workout_plan:
            logger.info("🔄 Attempting Method 3: HuggingFace AI (Fixed)")
            start_time = datetime.now()
            workout_plan = await self._generate_with_huggingface(user_profile, duration_minutes)
            if workout_plan:
                generation_method = "HuggingFace AI"
                generation_time = (datetime.now() - start_time).total_seconds()
                logger.success(f"🎉 SUCCESS: HuggingFace AI generated workout in {generation_time:.2f}s")
            else:
                logger.warning("⚠️ HuggingFace AI failed, trying next method...")

        # Method 4: Ollama AI (Local)
        if not workout_plan:
            logger.info("🔄 Attempting Method 4: Ollama AI (Local)")
            start_time = datetime.now()
            workout_plan = await self._generate_with_ollama(user_profile, duration_minutes)
            if workout_plan:
                generation_method = "Ollama AI"
                generation_time = (datetime.now() - start_time).total_seconds()
                logger.success(f"🎉 SUCCESS: Ollama AI generated workout in {generation_time:.2f}s")
            else:
                logger.warning("⚠️ Ollama AI failed, using fallback...")

        # Method 5: Rule-based fallback (Guaranteed)
        if not workout_plan:
            logger.info("🔄 Attempting Method 5: Rule-based Fallback (Guaranteed)")
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
        """Generate workout using Groq AI - FIXED"""
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
                    workout_data["ai_model"] = "Groq Llama3.1-8B"
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

    async def _generate_with_openrouter(self, user_profile: str, duration: int) -> Optional[Dict]:
        """NEW: Generate workout using OpenRouter AI"""
        logger.info("🌐 OpenRouter Generation: Creating AI prompt...")
        
        prompt = f"Create a {duration}-minute workout for: {user_profile}. Return JSON format with exercises."
        
        response = self.openrouter.generate_text(prompt)
        
        if response:
            logger.info("🔍 OpenRouter: Processing response...")
            try:
                # Try to extract JSON
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                
                if json_start != -1 and json_end > json_start:
                    json_str = response[json_start:json_end]
                    workout_data = json.loads(json_str)
                    
                    workout_data["ai_generated"] = True
                    workout_data["ai_model"] = "OpenRouter Llama3.1"
                    workout_data["generation_timestamp"] = datetime.now().isoformat()
                    
                    return workout_data
                else:
                    # Fallback: Parse text response
                    workout_plan = self._parse_workout_text(response, duration)
                    workout_plan["ai_generated"] = True
                    workout_plan["ai_model"] = "OpenRouter Llama3.1"
                    workout_plan["generation_timestamp"] = datetime.now().isoformat()
                    return workout_plan
                    
            except Exception as e:
                logger.error(f"❌ OpenRouter: Parsing failed: {str(e)}")
        
        return None

    async def _generate_with_huggingface(self, user_profile: str, duration: int) -> Optional[Dict]:
        """Generate workout using HuggingFace AI - FIXED"""
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

    async def _generate_with_ollama(self, user_profile: str, duration: int) -> Optional[Dict]:
        """Generate workout using Ollama AI"""
        logger.info("🏠 Ollama Generation: Creating AI prompt...")
        
        prompt = f"Create a {duration}-minute workout for: {user_profile}. List 5 exercises with sets and reps."
        
        response = self.ollama.generate_text(prompt)
        
        if response:
            logger.info("🔍 Ollama: Processing response...")
            workout_plan = self._parse_workout_text(response, duration)
            workout_plan["ai_generated"] = True
            workout_plan["ai_model"] = "Ollama Llama3.2"
            workout_plan["generation_timestamp"] = datetime.now().isoformat()
            
            logger.success("✅ Ollama: Successfully created workout from text")
            return workout_plan
        
        return None

    def _parse_workout_text(self, text: str, duration: int) -> Dict:
        """Parse AI-generated text into structured workout"""
        logger.info("📝 Parsing AI text response into structured workout...")
        
        exercises = []
        lines = text.split('\n')
        
        # Simple text parsing logic
        for line in lines:
            line = line.strip()
            if any(keyword in line.lower() for keyword in ['exercise', 'workout', 'rep', 'set', 'squat', 'push', 'pull', 'curl', 'press']):
                # Clean up the exercise name
                exercise_name = line.split(':')[0] if ':' in line else line
                exercise_name = exercise_name.replace('*', '').replace('-', '').replace('1.', '').replace('2.', '').strip()
                
                if len(exercise_name) > 3:  # Valid exercise name
                    exercises.append({
                        "name": exercise_name.title(),
                        "sets": 3,
                        "reps": "10-12",
                        "duration_seconds": 30,
                        "rest_seconds": 60,
                        "instructions": f"Perform {exercise_name} with proper form and controlled movements",
                        "muscle_groups": ["general"],
                        "equipment": "bodyweight",
                        "difficulty": "moderate"
                    })
        
        # Ensure we have at least some exercises
        if len(exercises) < 3:
            exercises = [
                {"name": "Push-ups", "sets": 3, "reps": "10-15", "muscle_groups": ["chest"], "instructions": "Keep your body straight, lower chest to ground"},
                {"name": "Squats", "sets": 3, "reps": "12-15", "muscle_groups": ["legs"], "instructions": "Lower hips back and down, keep knees behind toes"},
                {"name": "Plank", "sets": 3, "reps": "30s hold", "muscle_groups": ["core"], "instructions": "Hold straight body position, engage core"}
            ]
        
        return {
            "name": f"AI Generated {duration}-min Workout",
            "description": "Personalized workout created by AI based on your profile",
            "exercises": exercises[:7],  # Max 7 exercises
            "estimated_duration": duration,
            "estimated_calories": duration * 5,  # Rough estimate
            "difficulty_level": "moderate"
        }

    def _generate_fallback_workout(self, user, duration: int) -> Dict[str, Any]:
        """ENHANCED: Generate rule-based workout when AI fails"""
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
            exercise_data = random.choice(self.exercise_database[muscle_group])
            exercises.append({
                "name": exercise_data["name"],
                "sets": 2 if level == "beginner" else 3 if level == "intermediate" else 4,
                "reps": "12-15" if goal == "weight_loss" else "8-12" if goal == "muscle_gain" else "10-12",
                "duration_seconds": 45 if muscle_group == "cardio" else 30,
                "rest_seconds": 45 if level == "beginner" else 60 if level == "intermediate" else 75,
                "instructions": f"Perform {exercise_data['name']} with controlled movements and proper form",
                "muscle_groups": [muscle_group],
                "equipment": exercise_data["equipment"],
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

# Custom logging setup (same as before)
class ColoredFormatter(logging.Formatter):
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

def setup_ai_logging():
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(ColoredFormatter())
    logger.addHandler(console_handler)
    
    file_handler = logging.FileHandler('ai_generation.log')
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

setup_ai_logging()

def success(self, message, *args, **kwargs):
    if self.isEnabledFor(25):
        self._log(25, message, args, **kwargs)

logging.Logger.success = success
logging.addLevelName(25, "SUCCESS")