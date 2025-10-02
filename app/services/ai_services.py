# app/services/ai_services.py

import json
import requests
import random
import numpy as np
import logging
import re
import time
import asyncio
from typing import Dict, List, Any, Optional

from app.core.config import settings
from app import models, schemas # Make sure schemas is imported

# Set up logging
logger = logging.getLogger(__name__)

class GroqAI:
    """Free AI using Groq's fast inference API - with JSON Mode enabled"""

    def __init__(self):
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {settings.GROQ_API_KEY}",
            "Content-Type": "application/json",
        }
        self.available_models = [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "openai/gpt-oss-120b"
        ]
        self.current_model_index = 0

    def safe_json_extract(self, text: str) -> Optional[Dict]:
        if not text or not text.strip():
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            try:
                json_start = text.find('{')
                json_end = text.rfind('}') + 1
                if json_start != -1 and json_end > json_start:
                    json_str = text[json_start:json_end]
                    return json.loads(json_str)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse JSON even after slicing: {text[:500]}")
                return None
        return None

    def generate_text(self, prompt: str, max_tokens: int = 2048) -> Optional[str]:
        if not settings.GROQ_API_KEY:
            logger.warning("🚨 GROQ AI: API key not configured, skipping")
            return None

        for attempt in range(len(self.available_models)):
            model = self.available_models[self.current_model_index]
            logger.info(f"🚀 GROQ AI: Trying model {model}")

            try:
                payload = {
                    "messages": [{"role": "user", "content": prompt}],
                    "model": model,
                    "max_tokens": max_tokens,
                    "temperature": 0.2,
                    "response_format": {"type": "json_object"},
                }

                response = requests.post(
                    self.api_url, headers=self.headers, json=payload, timeout=30
                )

                if response.status_code == 200:
                    data = response.json()
                    content = data["choices"][0]["message"]["content"]
                    logger.debug(f"🔍 GROQ AI: Raw response: {content[:200]}...")
                    return content
                else:
                    logger.warning(
                        f"⚠️ GROQ AI: Model {model} failed with status {response.status_code} - {response.text}"
                    )
                    self.current_model_index = (
                        self.current_model_index + 1
                    ) % len(self.available_models)
                    continue

            except Exception as e:
                logger.error(f"❌ GROQ AI: Model {model} error: {str(e)}")
                self.current_model_index = (
                    self.current_model_index + 1
                ) % len(self.available_models)
                continue

        logger.error("❌ GROQ AI: All models failed")
        return None


class OpenRouterAI:
    """Free AI using OpenRouter's free models"""
    
    def __init__(self):
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": settings.FRONTEND_URL,
            "X-Title": "EvolveFit AI"
        }
        
        self.available_models = [
            "microsoft/phi-4-reasoning-plus",
            "z-ai/glm-4.5-air:free",
            "deepseek-ai/deepseek-v3-1:free",
            "mistralai/devstral-small-2505:free",
            "microsoft/mai-ds-r1:free",
            "google/gemma-2-9b-it:free",
            "meta-llama/llama-3.1-8b-instruct:free"
        ]
        self.current_model_index = 0
    
    def generate_text(self, prompt: str, max_tokens: int = 1024) -> Optional[str]:
        if not settings.OPENROUTER_API_KEY:
            logger.warning("🚨 OPENROUTER AI: API key not configured, skipping")
            return None
        
        for attempt in range(len(self.available_models)):
            model = self.available_models[self.current_model_index]
            logger.info(f"🌐 OPENROUTER AI: Trying model {model}")
            
            try:
                payload = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": 0.3
                }
                
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    content = data["choices"][0]["message"]["content"]
                    logger.info(f"✅ OPENROUTER AI: Successfully generated with {model}")
                    return content
                else:
                    logger.warning(f"⚠️ OPENROUTER AI: Model {model} failed with status {response.status_code}")
                    self.current_model_index = (self.current_model_index + 1) % len(self.available_models)
                    continue
                    
            except Exception as e:
                logger.error(f"❌ OPENROUTER AI: Model {model} error: {str(e)}")
                self.current_model_index = (self.current_model_index + 1) % len(self.available_models)
                continue
        
        logger.error("❌ OPENROUTER AI: All models failed")
        return None


class HuggingFaceAI:
    """Free AI using HuggingFace Inference API - UPDATED MODELS"""
    
    def __init__(self):
        self.headers = {
            "Authorization": f"Bearer {settings.HUGGINGFACE_API_KEY}",
            "Content-Type": "application/json"
        }
        
        self.available_models = [
            "mistralai/Mistral-7B-Instruct-v0.3",  # Primary: Best all-rounder
            "HuggingFaceH4/zephyr-7b-beta",        # Secondary: More creative
            "google/gemma-2b-it"
        ]
        self.current_model_index = 0
    
    def generate_text(self, prompt: str, max_tokens: int = 1024) -> Optional[str]:
        if not settings.HUGGINGFACE_API_KEY:
            logger.warning("🚨 HUGGINGFACE AI: API key not configured, skipping")
            return None
        
        logger.info("🤗 HUGGINGFACE AI: Starting text generation")
        
        for attempt in range(len(self.available_models)):
            model = self.available_models[self.current_model_index]
            logger.info(f"🤗 HUGGINGFACE AI: Trying model {model}")
            
            try:
                api_url = f"https://api-inference.huggingface.co/models/{model}"
                logger.info(f"🚀 HUGGINGFACE AI: Sending request to {api_url}")
                
                payload = {
                    "inputs": prompt,
                    "parameters": {
                        "max_new_tokens": max_tokens,
                        "temperature": 0.7,
                        "do_sample": True,
                        "return_full_text": False
                    }
                }
                
                start_time = time.time()
                response = requests.post(api_url, headers=self.headers, json=payload, timeout=30)
                end_time = time.time()
                
                logger.info(f"⏱️ HUGGINGFACE AI: Response received in {end_time - start_time:.2f} seconds")
                
                if response.status_code == 200:
                    try:
                        data = response.json()
                        if isinstance(data, list) and len(data) > 0:
                            generated_text = data[0].get('generated_text', '')
                            if generated_text:
                                logger.info(f"✅ HUGGINGFACE AI: Successfully generated with {model}")
                                return generated_text
                        
                        logger.warning(f"⚠️ HUGGINGFACE AI: Model {model} returned empty response")
                    except json.JSONDecodeError:
                        logger.warning(f"⚠️ HUGGINGFACE AI: Model {model} returned invalid JSON")
                        
                elif response.status_code == 503:
                    logger.warning(f"⚠️ HUGGINGFACE AI: Model {model} is loading, trying next...")
                else:
                    logger.warning(f"⚠️ HUGGINGFACE AI: Model {model} failed with status {response.status_code}")
                    logger.debug(f"🔍 HUGGINGFACE AI: Error response: {response.text}")
                
                self.current_model_index = (self.current_model_index + 1) % len(self.available_models)
                
            except Exception as e:
                logger.error(f"❌ HUGGINGFACE AI: Model {model} error: {str(e)}")
                self.current_model_index = (self.current_model_index + 1) % len(self.available_models)
                continue
        
        logger.error("❌ HUGGINGFACE AI: All models failed")
        return None


class OllamaAI:
    """Local AI using Ollama"""
    
    def __init__(self):
        self.api_url = f"{settings.OLLAMA_URL}/api/generate"
        self.available_models = [
            "llama3.2",
            "llama3.1", 
            "mistral",
            "codellama",
            "gemma2"
        ]
        self.current_model_index = 0
    
    def generate_text(self, prompt: str, max_tokens: int = 500) -> Optional[str]:
        logger.info("🏠 OLLAMA AI: Starting text generation")
        
        for attempt in range(len(self.available_models)):
            model = self.available_models[self.current_model_index]
            logger.info(f"🏠 OLLAMA AI: Trying model {model}")
            
            try:
                logger.info(f"🚀 OLLAMA AI: Sending request to {self.api_url}")
                
                payload = {
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": 0.7
                    }
                }
                
                response = requests.post(self.api_url, json=payload, timeout=60)
                
                if response.status_code == 200:
                    data = response.json()
                    generated_text = data.get('response', '')
                    if generated_text:
                        logger.info(f"✅ OLLAMA AI: Successfully generated with {model}")
                        return generated_text
                    
                logger.warning(f"⚠️ OLLAMA AI: Model {model} returned empty response")
                self.current_model_index = (self.current_model_index + 1) % len(self.available_models)
                
            except requests.exceptions.ConnectionError:
                logger.error("🔌 OLLAMA AI: Connection failed - Ollama server not running")
                logger.info("💡 OLLAMA AI: To fix this, install Ollama: https://ollama.ai")
                return None
            except Exception as e:
                logger.error(f"❌ OLLAMA AI: Model {model} error: {str(e)}")
                self.current_model_index = (self.current_model_index + 1) % len(self.available_models)
                continue
        
        logger.error("❌ OLLAMA AI: All models failed")
        return None


class RuleBasedWorkoutGenerator:
    """Guaranteed workout generation using rule-based system"""
    
    def __init__(self):
        self.exercise_database = {
            "strength": {
                "beginner": [
                    {"name": "Wall Push-ups", "sets": 2, "reps": "8-12", "calories_per_set": 15},
                    {"name": "Chair Squats", "sets": 2, "reps": "8-12", "calories_per_set": 20},
                    {"name": "Standing Calf Raises", "sets": 2, "reps": "10-15", "calories_per_set": 10}
                ],
                "intermediate": [
                    {"name": "Incline Push-ups", "sets": 3, "reps": "8-12", "calories_per_set": 25},
                    {"name": "Bent-over Rows", "sets": 3, "reps": "8-12", "calories_per_set": 30},
                    {"name": "Lunges", "sets": 3, "reps": "8-12", "calories_per_set": 35},
                    {"name": "Pike Push-ups", "sets": 3, "reps": "8-12", "calories_per_set": 30},
                    {"name": "Tricep Dips", "sets": 3, "reps": "8-12", "calories_per_set": 25}
                ],
                "advanced": [
                    {"name": "Standard Push-ups", "sets": 4, "reps": "12-15", "calories_per_set": 35},
                    {"name": "Jump Squats", "sets": 4, "reps": "10-15", "calories_per_set": 45},
                    {"name": "Pull-ups", "sets": 4, "reps": "5-10", "calories_per_set": 40},
                    {"name": "Burpees", "sets": 3, "reps": "8-12", "calories_per_set": 50}
                ]
            },
            "cardio": { "beginner": [], "intermediate": [], "advanced": [] },
            "muscle_gain": { "beginner": [], "intermediate": [], "advanced": [] }
        }
    
    def generate_workout(self, goal: str, level: str, duration: int = 30) -> Dict:
        logger.info("🎯 Rule-based Generation: Creating structured workout...")
        logger.info(f"🎯 Rule-based: Goal={goal}, Level={level}")
        
        exercises = self.exercise_database.get(goal, {}).get(level, [])
        if not exercises:
            exercises = self.exercise_database["strength"]["intermediate"]
        
        num_exercises = {"beginner": 3, "intermediate": 5, "advanced": 6}.get(level, 4)
        selected_exercises = random.sample(exercises, min(num_exercises, len(exercises)))
        
        total_calories = sum(ex["calories_per_set"] * ex["sets"] for ex in selected_exercises)
        
        workout_names = {
            "strength": f"Power {level.capitalize()} Strength",
            "cardio": f"High-Energy {level.capitalize()} Cardio",
            "muscle_gain": f"Smart Muscle Gain Workout"
        }
        workout_name = workout_names.get(goal, f"{level.capitalize()} Workout")
        
        logger.info(f"✅ Rule-based: Generated {len(selected_exercises)} exercises")
        
        return {
            "name": workout_name,
            "goal": goal,
            "difficulty_level": level,
            "estimated_duration": duration,
            "exercises": [
                {
                    "name": ex["name"],
                    "sets": ex["sets"],
                    "reps": ex["reps"],
                    "instructions": f"Perform {ex['sets']} sets of {ex['reps']} reps"
                }
                for ex in selected_exercises
            ],
            "estimated_calories": total_calories,
            "ai_generated": False,
            "ai_model": "Rule-Based Algorithm"
        }


class AIWorkoutService:
    """Main service coordinating all AI providers with fallback chain"""
    
    def __init__(self):
        self.groq_ai = GroqAI()
        self.openrouter_ai = OpenRouterAI()
        self.huggingface_ai = HuggingFaceAI()
        self.ollama_ai = OllamaAI()
        self.rule_based = RuleBasedWorkoutGenerator()
    
    def create_ai_prompt(
        self,
        goal: str,
        level: str,
        duration: int,
        preferences: Dict = None,
        target_muscles: Optional[List[str]] = None
    ) -> str:
        
        prompt = f"Generate a {duration}-minute workout. The user's primary goal is '{goal}' and their fitness level is '{level}'."
        
        if target_muscles:
            muscle_list = ", ".join(target_muscles)
            prompt += f"\nCRITICAL: The workout MUST primarily focus on the following muscle groups: {muscle_list}."
        
        prompt += """

Requirements:
- Provide a creative and motivating name for the workout.
- The number of exercises should be appropriate for the duration and fitness level.
- Provide a brief description of the overall workout.
- For EACH exercise, provide the following details: name, sets, reps, instructions, and a list of primary muscle_groups targeted.

Output ONLY the following JSON format. Do NOT include any text before or after the JSON object.
{{
    "name": "Workout Name",
    "description": "A brief overview of the workout.",
    "difficulty_level": "{level}",
    "estimated_duration": {duration},
    "estimated_calories": 250,
    "exercises": [
        {{
            "name": "Exercise Name",
            "sets": 3,
            "reps": "8-12",
            "instructions": "Detailed instructions on how to perform this exercise.",
            "muscle_groups": ["chest", "triceps"]
        }}
    ]
}}"""
        
        if preferences:
            prompt += f"\nUser Preferences: {preferences}"
        
        return prompt

    def generate_workout_sync(self, goal: str, level: str, duration: int = 30, preferences: Dict = None, target_muscles: Optional[List[str]] = None) -> Dict:
        start_time = time.time()
        
        logger.info("=" * 80)
        logger.info("🏋️ STARTING AI WORKOUT GENERATION")
        logger.info(f"🎯 Goal: {goal} | Level: {level} | Duration: {duration}min")
        if target_muscles:
            logger.info(f"💪 Targeting: {', '.join(target_muscles)}")
        logger.info("=" * 80)
        
        prompt = self.create_ai_prompt(goal, level, duration, preferences, target_muscles)
        
        # Method 1: Try Groq AI first
        logger.info("🔄 Attempting Method 1: Groq AI (Fastest)")
        try:
            raw_response = self.groq_ai.generate_text(prompt)
            if raw_response:
                workout_data = self.groq_ai.safe_json_extract(raw_response)
                if workout_data and "exercises" in workout_data:
                    logger.info(f"🎉 SUCCESS: Groq AI generated workout")
                    workout_data.update({"ai_generated": True, "ai_model": f"Groq AI ({self.groq_ai.available_models[self.groq_ai.current_model_index]})"})
                    return workout_data
                else:
                    logger.error("❌ Groq: Invalid JSON structure in response")
        except Exception as e:
            logger.error(f"❌ Groq: Generation failed: {str(e)}")
        
        logger.warning("⚠️ Groq AI failed, trying next method...")
        
        # Method 2: Try OpenRouter AI
        logger.info("🔄 Attempting Method 2: OpenRouter AI (Free)")
        try:
            raw_response = self.openrouter_ai.generate_text(prompt)
            if raw_response:
                workout_data = self.groq_ai.safe_json_extract(raw_response)
                if workout_data and "exercises" in workout_data:
                    logger.info("🎉 SUCCESS: OpenRouter AI generated workout")
                    workout_data.update({"ai_generated": True, "ai_model": "OpenRouter AI"})
                    return workout_data
        except Exception as e:
            logger.error(f"❌ OpenRouter: Generation failed: {str(e)}")

        logger.warning("⚠️ OpenRouter AI failed, trying next method...")
        
        # Method 3: Try HuggingFace AI
        logger.info("🔄 Attempting Method 3: HuggingFace AI")
        try:
            raw_response = self.huggingface_ai.generate_text(prompt)
            if raw_response:
                workout_data = self.groq_ai.safe_json_extract(raw_response)
                if workout_data and "exercises" in workout_data:
                    logger.info("🎉 SUCCESS: HuggingFace AI generated workout")
                    workout_data.update({"ai_generated": True, "ai_model": "HuggingFace AI"})
                    return workout_data
        except Exception as e:
            logger.error(f"❌ HuggingFace: Generation failed: {str(e)}")

        logger.warning("⚠️ HuggingFace AI failed, trying next method...")
        
        # Method 4: Try Ollama AI
        logger.info("🔄 Attempting Method 4: Ollama AI (Local)")
        try:
            raw_response = self.ollama_ai.generate_text(prompt)
            if raw_response:
                workout_data = self.groq_ai.safe_json_extract(raw_response)
                if workout_data and "exercises" in workout_data:
                    logger.info("🎉 SUCCESS: Ollama AI generated workout")
                    workout_data.update({"ai_generated": True, "ai_model": "Ollama AI"})
                    return workout_data
        except Exception as e:
            logger.error(f"❌ Ollama: Generation failed: {str(e)}")

        logger.warning("⚠️ All AI models failed, using fallback...")
        
        workout_data = self.rule_based.generate_workout(goal, level, duration)
        logger.info(f"🎉 SUCCESS: Rule-based system generated workout")
        return workout_data

    async def generate_workout(self, user: models.User, duration_minutes: int, target_muscles: Optional[List[str]] = None) -> Dict:
        logger.info("🧠 Starting AI workout generation process with data validation...")
        
        goal = getattr(user, 'fitness_goal', None) or 'general_fitness'
        level = getattr(user, 'experience_level', None) or 'intermediate'
        
        logger.info(f"Using Goal: '{goal}', Level: '{level}' for AI prompt.")

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, 
            self.generate_workout_sync,
            goal,
            level, 
            duration_minutes,
            None,
            target_muscles
        )
        
        result.update({
            "description": f"AI-generated {goal} workout for {level} level",
            "estimated_duration": duration_minutes
        })
        
        return result

    # --- NEW: MEAL PLAN GENERATION ---
    def _calculate_tdee(self, user: models.User) -> int:
        if not all([user.age, user.weight, user.height, user.gender, user.activity_level]):
            logger.warning(f"User {user.id} missing profile data for TDEE. Using default 2200 kcal.")
            return 2200

        if user.gender.lower() == 'male':
            bmr = 88.362 + (13.397 * user.weight) + (4.799 * user.height) - (5.677 * user.age)
        elif user.gender.lower() == 'female':
            bmr = 447.593 + (9.247 * user.weight) + (3.098 * user.height) - (4.330 * user.age)
        else:
            bmr = (88.362 + 447.593) / 2 + (11.322 * user.weight) + (3.948 * user.height) - (5.003 * user.age)

        activity_multipliers = {'sedentary': 1.2, 'lightly_active': 1.375, 'moderate': 1.55, 'very_active': 1.725, 'extremely_active': 1.9}
        multiplier = activity_multipliers.get(user.activity_level, 1.55)
        tdee = bmr * multiplier

        goal_adjustments = {'weight_loss': -500, 'muscle_gain': 300, 'strength': 200, 'endurance': 100, 'general_fitness': 0}
        adjustment = goal_adjustments.get(user.fitness_goal, 0)
        
        final_calories = round(tdee + adjustment)
        logger.info(f"Calculated TDEE for user {user.id}: {final_calories} kcal")
        return final_calories

    def create_meal_plan_prompt(self, user: models.User, target_calories: int) -> str:
        diet_restrictions = ", ".join(user.dietary_restrictions) if user.dietary_restrictions else "none"

        prompt = f"""
        Generate a one-day meal plan for a user with the following profile:
        - Fitness Goal: {user.fitness_goal or 'general fitness'}
        - Dietary Restrictions: {diet_restrictions}
        - Target Daily Calories: Approximately {target_calories} kcal

        Requirements:
        - The meal plan must be structured with four meals: Breakfast, Lunch, Dinner, and Snacks.
        - For EACH meal, provide: a 'name', a list of 'ingredients' with quantities, simple 'instructions', and estimated 'calories', 'protein', 'carbs', and 'fat'.
        - The total calories for the day must be close to the {target_calories} kcal target.
        - Calculate and provide total 'target_calories', 'target_protein', 'target_carbs', and 'target_fat' for the day.

        Output ONLY the following JSON format:
        {{
            "name": "AI Generated Meal Plan", "target_calories": {target_calories}, "target_protein": 150, "target_carbs": 200, "target_fat": 60,
            "meals": {{
                "breakfast": {{"name": "Hearty Oatmeal", "ingredients": ["1 cup oats", "1/2 cup berries"], "instructions": "Mix and cook.", "calories": 400, "protein": 30, "carbs": 50, "fat": 10}},
                "lunch": {{"name": "Grilled Chicken Salad", "ingredients": ["150g chicken breast", "2 cups greens"], "instructions": "Combine.", "calories": 500, "protein": 40, "carbs": 20, "fat": 25}},
                "dinner": {{"name": "Salmon with Quinoa", "ingredients": ["150g salmon", "1 cup quinoa"], "instructions": "Bake and serve.", "calories": 600, "protein": 45, "carbs": 50, "fat": 20}},
                "snacks": {{"name": "Greek Yogurt", "ingredients": ["1 cup yogurt", "1/4 cup nuts"], "instructions": "Combine.", "calories": 300, "protein": 25, "carbs": 15, "fat": 15}}
            }}
        }}
        """
        return prompt

    async def generate_meal_plan(self, user: models.User, request: schemas.MealPlanRequest) -> Dict:
        logger.info(f"Starting meal plan generation for user {user.id}")
        
        target_calories = self._calculate_tdee(user)
        prompt = self.create_meal_plan_prompt(user, target_calories)
        
        logger.info("Attempting meal plan generation with GroqAI...")
        try:
            raw_response = self.groq_ai.generate_text(prompt)
            if raw_response:
                meal_plan_data = self.groq_ai.safe_json_extract(raw_response)
                if meal_plan_data and "meals" in meal_plan_data:
                    logger.info("🎉 SUCCESS: Groq AI generated meal plan")
                    meal_plan_data["ai_generated"] = True
                    meal_plan_data["ai_model"] = f"Groq AI ({self.groq_ai.available_models[self.groq_ai.current_model_index]})"
                    return meal_plan_data
                else:
                    logger.error("❌ Groq (Meal Plan): Invalid JSON structure in response")
        except Exception as e:
            logger.error(f"❌ Groq (Meal Plan): Generation failed: {str(e)}")

        logger.warning("⚠️ AI meal plan generation failed. Returning empty plan.")
        return {"name": "Failed to Generate Plan", "meals": {}, "ai_generated": False, "target_calories": 0, "target_protein": 0, "target_carbs": 0, "target_fat": 0}


ai_workout_service = AIWorkoutService()
ai_workout_generator = ai_workout_service
__all__ = ['ai_workout_generator', 'ai_workout_service', 'AIWorkoutService']