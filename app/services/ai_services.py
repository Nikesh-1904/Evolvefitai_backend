# app/services/ai_services.py

import json
import requests
import random
import logging
import time
import asyncio
from typing import Dict, List, Optional
from app.core.config import settings
from app import models, schemas  # Make sure schemas is imported and deployed

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
            logger.warning("⚠️ GROQ AI: API key not configured, skipping")
            return None

        for attempt in range(len(self.available_models)):
            model = self.available_models[self.current_model_index]
            logger.info(f"⚠️ GROQ AI: Trying model {model}")

            try:
                payload = {
                    "messages": [{"role": "user", "content": prompt}],
                    "model": model,
                    "max_tokens": max_tokens,
                    "temperature": 0.2,
                    "response_format": {"type": "json_object"}  # Enable JSON mode
                }

                response = requests.post(
                    self.api_url, 
                    headers=self.headers, 
                    json=payload, 
                    timeout=30
                )

                if response.status_code == 200:
                    data = response.json()
                    content = data['choices'][0]['message']['content']
                    logger.debug(f"⚠️ GROQ AI: Raw response: {content[:200]}...")
                    return content
                else:
                    logger.warning(f"⚠️ GROQ AI: Model {model} failed with status {response.status_code} - {response.text}")
                    self.current_model_index = (self.current_model_index + 1) % len(self.available_models)
                    continue

            except Exception as e:
                logger.error(f"⚠️ GROQ AI: Model {model} error: {str(e)}")
                self.current_model_index = (self.current_model_index + 1) % len(self.available_models)
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
            "HTTP-Referer": settings.CLIENT_FRONTEND_URL,
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
            logger.info(f"⚠️ OPENROUTER AI: Trying model {model}")

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
                    content = data['choices'][0]['message']['content']
                    logger.info(f"✅ OPENROUTER AI: Successfully generated with {model}")
                    return content
                else:
                    logger.warning(f"⚠️ OPENROUTER AI: Model {model} failed with status {response.status_code}")
                    self.current_model_index = (self.current_model_index + 1) % len(self.available_models)
                    continue

            except Exception as e:
                logger.error(f"⚠️ OPENROUTER AI: Model {model} error: {str(e)}")
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
            "meta-llama/Meta-Llama-3-8B-Instruct",
            "google/gemma-2-9b-it", 
            "HuggingFaceH4/zephyr-7b-beta",
        ]
        self.current_model_index = 0

    def generate_text(self, prompt: str, max_tokens: int = 1024) -> Optional[str]:
        if not settings.HUGGINGFACE_API_KEY:
            logger.warning("⚠️ HUGGINGFACE AI: API key not configured, skipping")
            return None

        logger.info("⚠️ HUGGINGFACE AI: Starting text generation")

        for attempt in range(len(self.available_models)):
            model = self.available_models[self.current_model_index]
            logger.info(f"⚠️ HUGGINGFACE AI: Trying model {model}")

            try:
                api_url = f"https://api-inference.huggingface.co/models/{model}"
                logger.info(f"⚠️ HUGGINGFACE AI: Sending request to {api_url}")

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
                logger.info(f"⚠️ HUGGINGFACE AI: Response received in {end_time - start_time:.2f} seconds")

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
                    logger.debug(f"⚠️ HUGGINGFACE AI: Error response: {response.text}")

                self.current_model_index = (self.current_model_index + 1) % len(self.available_models)

            except Exception as e:
                logger.error(f"⚠️ HUGGINGFACE AI: Model {model} error: {str(e)}")
                self.current_model_index = (self.current_model_index + 1) % len(self.available_models)
                continue

        logger.error("❌ HUGGINGFACE AI: All models failed")
        return None

class OllamaAI:
    """Local AI using Ollama"""

    def __init__(self):
        self.api_url = f"{settings.OLLAMA_BASE_URL}/api/generate"
        self.available_models = ["llama3.2", "llama3.1", "mistral", "codellama", "gemma2"]
        self.current_model_index = 0

    def generate_text(self, prompt: str, max_tokens: int = 500) -> Optional[str]:
        logger.info("🔌 OLLAMA AI: Starting text generation")

        for attempt in range(len(self.available_models)):
            model = self.available_models[self.current_model_index]
            logger.info(f"🔌 OLLAMA AI: Trying model {model}")

            try:
                logger.info(f"🔌 OLLAMA AI: Sending request to {self.api_url}")

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
                    logger.warning(f"🔌 OLLAMA AI: Model {model} returned empty response")

                self.current_model_index = (self.current_model_index + 1) % len(self.available_models)

            except requests.exceptions.ConnectionError:
                logger.error("🔌 OLLAMA AI: Connection failed - Ollama server not running")
                logger.info("🔌 OLLAMA AI: To fix this, install Ollama: https://ollama.ai")
                return None
            except Exception as e:
                logger.error(f"🔌 OLLAMA AI: Model {model} error: {str(e)}")
                self.current_model_index = (self.current_model_index + 1) % len(self.available_models)
                continue

        logger.error("🔌 OLLAMA AI: All models failed")
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
            "cardio": {
                "beginner": [],
                "intermediate": [], 
                "advanced": []
            },
            "muscle_gain": {
                "beginner": [],
                "intermediate": [],
                "advanced": []
            }
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
            "muscle_gain": "Smart Muscle Gain Workout"
        }
        workout_name = workout_names.get(goal, f"{level.capitalize()} Workout")

        logger.info(f"🎯 Rule-based: Generated {len(selected_exercises)} exercises")

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
                } for ex in selected_exercises
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
        num_exercises: Optional[int] = None,
        workout_type: Optional[str] = None,
        target_muscles: Optional[List[str]] = None
    ) -> str:
        
        # --- NEW HIGH-DETAIL PROMPT ---

        prompt = f"""
You are an expert fitness coach and workout designer named 'Evolve AI'. Your task is to generate a personalized workout plan based on the user's profile and requests. You must adhere strictly to the JSON output format specified.

**User Profile:**
- **Primary Goal:** {goal}
- **Fitness Level:** {level}

**Workout Request:**
- **Total Duration:** {duration} minutes
- **Specific Workout Type:** {workout_type or 'Not specified, use best judgment'}
- **Target Muscle Groups:** {', '.join(target_muscles) if target_muscles else 'Not specified'}
- **Number of Exercises:** {num_exercises or 'You decide the optimal number'}

**Core Instructions:**
1.  **Create a motivating and descriptive name** for the workout.
2.  **For EACH exercise, you MUST provide the following keys:**
    - `name`: The common name of the exercise.
    - `sets`: The number of sets.
    - `reps`: The number of repetitions or duration for a set.
    - `instructions`: Clear, concise instructions on how to perform the exercise.
    - `muscle_groups`: A list of the primary muscles targeted.
    - `met_value`: Your best estimate of the MET (Metabolic Equivalent of Task) value for this exercise.
    - `exercise_type`: This is CRITICAL. You must classify the exercise into one of the following exact categories based on how it is logged:
        - `WEIGHT_BASED`: For exercises where the user logs both weight and reps (e.g., Bench Press, Dumbbell Curls).
        - `REPS_ONLY`: For bodyweight exercises where only reps are logged (e.g., Push-ups, Burpees, Crunches).
        - `DURATION`: For exercises held for time (e.g., Plank, Wall Sit).
        - `DISTANCE_DURATION`: For cardio where distance and time are logged (e.g., Running, Cycling).
        - `QUALITATIVE`: For activities like yoga or stretching where the user logs completion and notes.

**Output Format:**
- You must respond ONLY with a single, valid JSON object.
- Do not include any introductory text, explanations, or markdown formatting like ```json.

**Example Scenarios:**

* **Example 1: A strength exercise.**
    {{
      "name": "Dumbbell Bench Press",
      "sets": 3,
      "reps": "8-12",
      "instructions": "Lie on a flat bench with a dumbbell in each hand. Push the dumbbells up until your arms are fully extended.",
      "muscle_groups": ["chest", "triceps", "shoulders"],
      "met_value": 5.0,
      "exercise_type": "WEIGHT_BASED"
    }}

* **Example 2: A bodyweight exercise.**
    {{
      "name": "Burpees",
      "sets": 4,
      "reps": "10",
      "instructions": "Start in a standing position. Drop into a squat, kick your feet back into a plank, perform a push-up, return to the squat, and jump up explosively.",
      "muscle_groups": ["full body", "core", "legs"],
      "met_value": 8.0,
      "exercise_type": "REPS_ONLY"
    }}

* **Example 3: A duration-based exercise.**
    {{
      "name": "Forearm Plank",
      "sets": 3,
      "reps": "60 seconds",
      "instructions": "Hold a straight line from your head to your heels, resting on your forearms and toes. Keep your core engaged.",
      "muscle_groups": ["core", "abs"],
      "met_value": 3.0,
      "exercise_type": "DURATION"
    }}

* **Example 4: A yoga/qualitative exercise.**
    {{
      "name": "Cool-down Stretching",
      "sets": 1,
      "reps": "5 minutes",
      "instructions": "Perform a series of static stretches, holding each for 30 seconds. Focus on the muscles worked during the session.",
      "muscle_groups": ["full body"],
      "met_value": 2.5,
      "exercise_type": "QUALITATIVE"
    }}
    
**Complete Example of Expected Output:**
{{
  "name": "Full Body Ignition",
  "description": "A 45-minute workout to build strength and endurance across all major muscle groups.",
  "difficulty_level": "intermediate",
  "estimated_duration": 45,
  "exercises": [
    {{
      "name": "Dumbbell Bench Press",
      "sets": 3,
      "reps": "8-12",
      "instructions": "Lie on a flat bench...",
      "muscle_groups": ["chest", "triceps", "shoulders"],
      "met_value": 5.0,
      "exercise_type": "WEIGHT_BASED"
    }},
    {{
      "name": "Burpees",
      "sets": 4,
      "reps": "10",
      "instructions": "Start in a standing position...",
      "muscle_groups": ["full body", "core", "legs"],
      "met_value": 8.0,
      "exercise_type": "REPS_ONLY"
    }}
  ]
}}

Now, generate the complete JSON object for the user's request.
"""
        return prompt

    def generate_workout_sync(
        self, 
        goal: str, 
        level: str, 
        duration: int = 30, 
        num_exercises: Optional[int] = None,
        workout_type: Optional[str] = None,
        target_muscles: Optional[List[str]] = None
    ) -> Dict:
        # ... (logging code remains the same)
        logger.info(f"🎯 Goal: {goal} | Level: {level} | Duration: {duration}min")
        if num_exercises: 
            logger.info(f"🔢 Exercises: {num_exercises}")
        if workout_type: 
            logger.info(f"🏡 Type: {workout_type}")
        if target_muscles: 
            logger.info(f"💪 Targeting: {', '.join(target_muscles)}")
        logger.info("=" * 80)

        prompt = self.create_ai_prompt(goal, level, duration, num_exercises, workout_type, target_muscles)

        # Method 1: Try Groq AI first
        logger.info("🔥 Attempting Method 1: Groq AI (Fastest)")
        try:
            raw_response = self.groq_ai.generate_text(prompt)
            if raw_response:
                workout_data = self.groq_ai.safe_json_extract(raw_response)
                if workout_data and "exercises" in workout_data:
                    logger.info("✅ SUCCESS: Groq AI generated workout")
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
                    logger.info("✅ SUCCESS: OpenRouter AI generated workout")
                    workout_data.update({"ai_generated": True, "ai_model": "OpenRouter AI"})
                    return workout_data
        except Exception as e:
            logger.error(f"❌ OpenRouter: Generation failed: {str(e)}")

        logger.warning("⚠️ OpenRouter AI failed, trying next method...")

        # Method 3: Try HuggingFace AI
        logger.info("🤗 Attempting Method 3: HuggingFace AI")
        try:
            raw_response = self.huggingface_ai.generate_text(prompt)
            if raw_response:
                workout_data = self.groq_ai.safe_json_extract(raw_response)
                if workout_data and "exercises" in workout_data:
                    logger.info("✅ SUCCESS: HuggingFace AI generated workout")
                    workout_data.update({"ai_generated": True, "ai_model": "HuggingFace AI"})
                    return workout_data
        except Exception as e:
            logger.error(f"❌ HuggingFace: Generation failed: {str(e)}")

        logger.warning("⚠️ HuggingFace AI failed, trying next method...")

        # Method 4: Try Ollama AI
        logger.info("🔌 Attempting Method 4: Ollama AI (Local)")
        try:
            raw_response = self.ollama_ai.generate_text(prompt)
            if raw_response:
                workout_data = self.groq_ai.safe_json_extract(raw_response)
                if workout_data and "exercises" in workout_data:
                    logger.info("✅ SUCCESS: Ollama AI generated workout")
                    workout_data.update({"ai_generated": True, "ai_model": "Ollama AI"})
                    return workout_data
        except Exception as e:
            logger.error(f"❌ Ollama: Generation failed: {str(e)}")

        # Final fallback: Rule-based system
        logger.warning("⚠️ All AI models failed, using fallback...")
        workout_data = self.rule_based.generate_workout(goal, level, duration)
        logger.info("✅ SUCCESS: Rule-based system generated workout")
        return workout_data

    async def generate_workout(
        self, 
        user: models.User, 
        duration_minutes: int, 
        num_exercises: Optional[int] = None,
        workout_type: Optional[str] = None,
        target_muscles: Optional[List[str]] = None
    ) -> Dict:
        logger.info("🚀 Starting AI workout generation process...")

        goal = getattr(user, 'fitness_goal', 'general_fitness')
        level = getattr(user, 'experience_level', 'intermediate')

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, 
            self.generate_workout_sync, 
            goal, level, duration_minutes, num_exercises, workout_type, target_muscles
        )
        result.update({
            "description": f"AI-generated {goal} workout for {level} level",
            "estimated_duration": duration_minutes
        })

        return result



    async def get_met_value_for_exercise(self, exercise_name: str) -> float:
        """Gets a MET value for a single exercise name using AI, with a fallback."""
        logger.info(f"🔎 Getting MET value for exercise: '{exercise_name}'")
        
        prompt = f'''What is the MET (Metabolic Equivalent of Task) value for the exercise "{exercise_name}"?
        
        Respond ONLY with a JSON object in the format {{"met_value": 5.5}}. 
        If you don't know the MET value, respond with {{"met_value": null}}.
        '''
        
        # Use a fast AI provider for this quick lookup
        raw_response = self.groq_ai.generate_text(prompt, max_tokens=50)
        
        if raw_response:
            try:
                data = self.groq_ai.safe_json_extract(raw_response)
                if data and "met_value" in data and data["met_value"] is not None:
                    met_value = float(data["met_value"])
                    logger.info(f"✅ AI found MET value for '{exercise_name}': {met_value}")
                    return met_value
            except (ValueError, TypeError) as e:
                logger.warning(f"⚠️ Could not parse MET value from AI response: {e}")

        # If AI fails or returns null, use our default value
        logger.warning(f"⚠️ Using default MET value for '{exercise_name}'")
        return 3.5  # Our agreed-upon default


# Initialize services
ai_workout_service = AIWorkoutService()

# Export for other modules
ai_workout_generator = ai_workout_service

__all__ = ['ai_workout_generator', 'ai_workout_service', 'AIWorkoutService']
