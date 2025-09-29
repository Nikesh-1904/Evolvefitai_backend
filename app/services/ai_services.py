import json
import requests
import random
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from app.core.config import settings

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
        if not settings.GROQ_API_KEY:
            return None
            
        try:
            payload = {
                "messages": [{"role": "user", "content": prompt}],
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": 0.7
            }
            
            response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"]
            else:
                print(f"Groq API Error: {response.status_code}")
                return None
        except Exception as e:
            print(f"Groq API failed: {e}")
            return None

class HuggingFaceAI:
    """Free AI using Hugging Face Inference API"""
    
    def __init__(self):
        self.api_url = "https://api-inference.huggingface.co/models/"
        self.headers = {"Authorization": f"Bearer {settings.HUGGINGFACE_API_KEY}"}
    
    def query_model(self, model_name: str, payload: dict, max_retries: int = 3) -> Optional[Any]:
        """Query Hugging Face model with retries"""
        if not settings.HUGGINGFACE_API_KEY:
            return None
            
        url = f"{self.api_url}{model_name}"
        
        for attempt in range(max_retries):
            try:
                response = requests.post(url, headers=self.headers, json=payload, timeout=30)
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 503:
                    import time
                    time.sleep(5)
                    continue
                else:
                    print(f"HF API Error: {response.status_code}")
                    break
            except Exception as e:
                print(f"HF Request failed: {e}")
                if attempt < max_retries - 1:
                    import time
                    time.sleep(2)
                    continue
                break
        
        return None

class OllamaAI:
    """Free local AI using Ollama"""
    
    def __init__(self):
        self.api_url = f"{settings.OLLAMA_URL}/api/generate"
        self.model = "llama2"
    
    def generate_text(self, prompt: str) -> Optional[str]:
        """Generate text using local Ollama"""
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False
            }
            
            response = requests.post(self.api_url, json=payload, timeout=60)
            
            if response.status_code == 200:
                return response.json()["response"]
            else:
                return None
        except Exception as e:
            print(f"Ollama not available: {e}")
            return None

class AIWorkoutGenerator:
    """AI-powered workout generator using FREE models"""
    
    def __init__(self):
        self.groq = GroqAI()
        self.hf = HuggingFaceAI()
        self.ollama = OllamaAI()
        
        # Fallback exercise database
        self.exercise_database = {
            "chest": ["Push-ups", "Chest Press", "Dips", "Incline Push-ups"],
            "back": ["Pull-ups", "Rows", "Lat Pulldowns", "Reverse Flyes"],
            "legs": ["Squats", "Lunges", "Leg Press", "Calf Raises"],
            "shoulders": ["Shoulder Press", "Lateral Raises", "Front Raises"],
            "arms": ["Bicep Curls", "Tricep Dips", "Hammer Curls"],
            "core": ["Plank", "Crunches", "Russian Twists", "Mountain Climbers"],
            "cardio": ["Burpees", "Jumping Jacks", "High Knees", "Sprint Intervals"]
        }
    
    async def generate_workout(self, user, duration_minutes: int = 45) -> Dict[str, Any]:
        """Generate AI-powered workout plan"""
        
        user_profile = self._create_user_profile(user)
        
        # Try AI services in order
        workout_plan = None
        
        # 1. Try Groq first
        workout_plan = await self._generate_with_groq(user_profile, duration_minutes)
        
        # 2. Try Ollama
        if not workout_plan:
            workout_plan = await self._generate_with_ollama(user_profile, duration_minutes)
        
        # 3. Fallback
        if not workout_plan:
            workout_plan = self._generate_fallback_workout(user, duration_minutes)
        
        return workout_plan
    
    def _create_user_profile(self, user) -> str:
        """Create user profile string for AI"""
        profile_parts = []
        
        if user.age:
            profile_parts.append(f"Age: {user.age}")
        if user.experience_level:
            profile_parts.append(f"Experience: {user.experience_level}")
        if user.fitness_goal:
            profile_parts.append(f"Goal: {user.fitness_goal}")
        if user.weight:
            profile_parts.append(f"Weight: {user.weight}kg")
        if user.height:
            profile_parts.append(f"Height: {user.height}cm")
        
        return ", ".join(profile_parts) if profile_parts else "General fitness"
    
    async def _generate_with_groq(self, user_profile: str, duration: int) -> Optional[Dict[str, Any]]:
        """Generate workout using Groq AI"""
        prompt = f"""
Create a personalized {duration}-minute workout for: {user_profile}

Generate 5-7 exercises with:
- Exercise name
- Sets and reps
- Rest periods
- Brief instructions

Format as JSON:
{{
    "name": "Workout name",
    "description": "Brief description",
    "exercises": [
        {{
            "name": "Exercise name",
            "sets": 3,
            "reps": "10-12",
            "rest_seconds": 60,
            "instructions": "How to perform",
            "muscle_groups": ["muscle1", "muscle2"]
        }}
    ],
    "estimated_duration": {duration}
}}

Respond ONLY with valid JSON.
"""
        
        response = self.groq.generate_text(prompt, max_tokens=800)
        if response:
            try:
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = response[json_start:json_end]
                    workout_data = json.loads(json_str)
                    workout_data["ai_generated"] = True
                    workout_data["ai_model"] = "Groq Llama3"
                    return workout_data
            except Exception as e:
                print(f"Groq JSON parsing failed: {e}")
        
        return None
    
    async def _generate_with_ollama(self, user_profile: str, duration: int) -> Optional[Dict[str, Any]]:
        """Generate workout using Ollama"""
        prompt = f"Create a {duration}-minute workout for: {user_profile}. List 5 exercises with sets and reps."
        
        response = self.ollama.generate_text(prompt)
        if response:
            workout_plan = self._parse_workout_text(response, duration)
            if workout_plan:
                workout_plan["ai_generated"] = True
                workout_plan["ai_model"] = "Ollama Llama2"
                return workout_plan
        
        return None
    
    def _parse_workout_text(self, text: str, duration: int) -> Optional[Dict[str, Any]]:
        """Parse AI-generated workout text"""
        exercises = []
        lines = text.split('\\n')
        
        for line in lines:
            line = line.strip().lower()
            for muscle_group, exercise_list in self.exercise_database.items():
                for exercise in exercise_list:
                    if exercise.lower() in line:
                        exercises.append({
                            "name": exercise,
                            "sets": 3,
                            "reps": "10-12",
                            "rest_seconds": 60,
                            "instructions": f"Perform {exercise} with proper form",
                            "muscle_groups": [muscle_group]
                        })
                        break
        
        if len(exercises) < 4:
            return None
        
        return {
            "name": "AI Generated Workout",
            "description": "Personalized workout based on your profile",
            "exercises": exercises[:6],
            "estimated_duration": duration
        }
    
    def _generate_fallback_workout(self, user, duration: int) -> Dict[str, Any]:
        """Generate fallback workout using rules"""
        goal = user.fitness_goal or "general_fitness"
        level = user.experience_level or "beginner"
        
        if goal == "weight_loss":
            focus_groups = ["cardio", "legs", "core", "chest"]
        elif goal == "muscle_gain":
            focus_groups = ["chest", "back", "legs", "shoulders"]
        else:
            focus_groups = ["chest", "back", "legs", "core"]
        
        exercises = []
        for muscle_group in focus_groups[:5]:
            exercise_name = random.choice(self.exercise_database[muscle_group])
            exercises.append({
                "name": exercise_name,
                "sets": 3 if level == "beginner" else 4,
                "reps": "12-15" if goal == "weight_loss" else "8-12",
                "rest_seconds": 60,
                "instructions": f"Perform {exercise_name} with controlled movements",
                "muscle_groups": [muscle_group]
            })
        
        return {
            "name": f"Smart {goal.replace('_', ' ').title()} Workout",
            "description": f"Designed for {level} level {goal} goals",
            "exercises": exercises,
            "estimated_duration": duration,
            "ai_generated": False,
            "ai_model": "Rule-Based System"
        }

# Initialize AI services
ai_workout_generator = AIWorkoutGenerator()