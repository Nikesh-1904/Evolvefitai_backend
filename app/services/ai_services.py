# app/services/ai_services.py - FINAL CORRECTED VERSION - Proper Async Compatibility

import json
import requests
import random
import numpy as np
import logging
import re
import time
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from app.core.config import settings

# Set up logging
logger = logging.getLogger(__name__)

class GroqAI:
    """Free AI using Groq's fast inference API - FIXED JSON PARSING"""
    
    def __init__(self):
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {settings.GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Updated working models for Groq 2025
        self.available_models = [
            "llama-3.3-70b-versatile",  # Latest Llama 3.3
            "llama-3.1-8b-instant",     # Fast and reliable
            "llama-3.1-70b-versatile",  # High performance
            "mixtral-8x7b-32768",       # Good alternative
            "gemma2-9b-it"              # Google's model
        ]
        self.current_model_index = 0
    
    def safe_json_extract(self, text: str) -> Optional[Dict]:
        """Safely extract JSON from AI response with multiple fallback methods"""
        if not text or not text.strip():
            return None
            
        # Method 1: Try direct JSON parsing
        try:
            return json.loads(text.strip())
        except:
            pass
        
        # Method 2: Extract JSON between first { and last }
        try:
            json_start = text.find('{')
            json_end = text.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                json_str = text[json_start:json_end]
                return json.loads(json_str)
        except:
            pass
        
        # Method 3: Use regex to find JSON blocks
        try:
            json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
            matches = re.findall(json_pattern, text, re.DOTALL)
            for match in matches:
                try:
                    return json.loads(match)
                except:
                    continue
        except:
            pass
        
        # Method 4: Try to fix common JSON issues
        try:
            # Remove trailing commas
            cleaned = re.sub(r',(\s*[}\]])', r'\1', text)
            # Extract between first { and last }
            json_start = cleaned.find('{')
            json_end = cleaned.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                json_str = cleaned[json_start:json_end]
                return json.loads(json_str)
        except:
            pass
        
        return None
    
    def generate_text(self, prompt: str, max_tokens: int = 500) -> Optional[str]:
        """Generate text using Groq's API with improved JSON parsing"""
        if not settings.GROQ_API_KEY:
            logger.warning("🚨 GROQ AI: API key not configured, skipping")
            return None
        
        for attempt in range(len(self.available_models)):
            model = self.available_models[self.current_model_index]
            logger.info(f"🚀 GROQ AI: Trying model {model}")
            
            try:
                # Enhanced prompt for better JSON output
                enhanced_prompt = f"""Please respond with ONLY valid JSON format. No additional text or explanations.

{prompt}

Remember: Respond with ONLY the JSON object, nothing else."""
                
                payload = {
                    "messages": [{"role": "user", "content": enhanced_prompt}],
                    "model": model,
                    "max_tokens": max_tokens,
                    "temperature": 0.3  # Lower temperature for more consistent output
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
                    
                    # Log raw response for debugging
                    logger.debug(f"🔍 GROQ AI: Raw response: {content[:200]}...")
                    
                    return content
                else:
                    logger.warning(f"⚠️ GROQ AI: Model {model} failed with status {response.status_code}")
                    self.current_model_index = (self.current_model_index + 1) % len(self.available_models)
                    continue
                    
            except Exception as e:
                logger.error(f"❌ GROQ AI: Model {model} error: {str(e)}")
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
            "HTTP-Referer": settings.FRONTEND_URL,
            "X-Title": "EvolveFit AI"
        }
        
        # Updated free models for OpenRouter 2025
        self.available_models = [
            "microsoft/phi-4-reasoning-plus",  # New powerful free model
            "z-ai/glm-4.5-air:free",          # GLM free model
            "deepseek-ai/deepseek-v3-1:free", # DeepSeek free
            "mistralai/devstral-small-2505:free",  # Mistral free
            "microsoft/mai-ds-r1:free",        # Microsoft free
            "google/gemma-2-9b-it:free",       # Gemma free
            "meta-llama/llama-3.1-8b-instruct:free"  # Llama free
        ]
        self.current_model_index = 0
    
    def generate_text(self, prompt: str, max_tokens: int = 500) -> Optional[str]:
        """Generate text using OpenRouter's free models"""
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
        
        # Updated working models for HuggingFace 2025
        self.available_models = [
            "microsoft/DialoGPT-large",        # Conversational AI
            "gpt2",                            # Always available
            "facebook/blenderbot-400M-distill", # Chatbot
            "microsoft/DialoGPT-medium",       # Medium conversational model
            "EleutherAI/gpt-neo-2.7B",        # GPT-like model
            "google/flan-t5-large",           # Instruction-following model
            "bigscience/bloomz-560m",         # Multilingual model
            "HuggingFaceH4/zephyr-7b-beta"    # Instruction model
        ]
        self.current_model_index = 0
    
    def generate_text(self, prompt: str, max_tokens: int = 200) -> Optional[str]:
        """Generate text using HuggingFace's free models"""
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
                
                # Try next model
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
        """Generate text using local Ollama"""
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


# Rule-based workout generation as fallback
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
                "beginner": [
                    {"name": "Walking in Place", "sets": 3, "reps": "30 seconds", "calories_per_set": 20},
                    {"name": "Arm Circles", "sets": 2, "reps": "15", "calories_per_set": 10},
                    {"name": "Step-ups", "sets": 2, "reps": "10", "calories_per_set": 25}
                ],
                "intermediate": [
                    {"name": "Jumping Jacks", "sets": 3, "reps": "30", "calories_per_set": 35},
                    {"name": "High Knees", "sets": 3, "reps": "30 seconds", "calories_per_set": 40},
                    {"name": "Mountain Climbers", "sets": 3, "reps": "20", "calories_per_set": 45}
                ],
                "advanced": [
                    {"name": "Burpees", "sets": 4, "reps": "15", "calories_per_set": 60},
                    {"name": "Sprint in Place", "sets": 4, "reps": "30 seconds", "calories_per_set": 50},
                    {"name": "Jump Squats", "sets": 4, "reps": "20", "calories_per_set": 55}
                ]
            },
            "muscle_gain": {
                "beginner": [
                    {"name": "Wall Push-ups", "sets": 3, "reps": "8-12", "calories_per_set": 15},
                    {"name": "Chair Squats", "sets": 3, "reps": "10-15", "calories_per_set": 25},
                    {"name": "Standing Calf Raises", "sets": 3, "reps": "12-15", "calories_per_set": 12}
                ],
                "intermediate": [
                    {"name": "Incline Push-ups", "sets": 3, "reps": "8-12", "calories_per_set": 25},
                    {"name": "Bent-over Rows", "sets": 3, "reps": "8-12", "calories_per_set": 30},
                    {"name": "Wall Sits", "sets": 3, "reps": "8-12", "calories_per_set": 20},
                    {"name": "Front Raises", "sets": 3, "reps": "8-12", "calories_per_set": 20},
                    {"name": "Tricep Extensions", "sets": 3, "reps": "8-12", "calories_per_set": 25}
                ],
                "advanced": [
                    {"name": "Diamond Push-ups", "sets": 4, "reps": "8-12", "calories_per_set": 40},
                    {"name": "Single-leg Squats", "sets": 4, "reps": "6-10", "calories_per_set": 50},
                    {"name": "Handstand Push-ups", "sets": 3, "reps": "5-8", "calories_per_set": 60},
                    {"name": "Archer Push-ups", "sets": 3, "reps": "6-10", "calories_per_set": 45}
                ]
            }
        }
    
    def generate_workout(self, goal: str, level: str, duration: int = 30) -> Dict:
        """Generate a structured workout using rule-based logic"""
        logger.info("🎯 Rule-based Generation: Creating structured workout...")
        logger.info(f"🎯 Rule-based: Goal={goal}, Level={level}")
        
        # Get exercises for the goal and level
        exercises = self.exercise_database.get(goal, {}).get(level, [])
        if not exercises:
            # Fallback to intermediate strength
            exercises = self.exercise_database["strength"]["intermediate"]
        
        # Select random exercises (3-6 exercises based on level)
        num_exercises = {"beginner": 3, "intermediate": 5, "advanced": 6}.get(level, 4)
        selected_exercises = random.sample(exercises, min(num_exercises, len(exercises)))
        
        # Calculate total calories
        total_calories = sum(ex["calories_per_set"] * ex["sets"] for ex in selected_exercises)
        
        # Generate workout name
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
            "difficulty": level,
            "duration": duration,
            "exercises": [
                {
                    "name": ex["name"],
                    "sets": ex["sets"],
                    "reps": ex["reps"],
                    "description": f"Perform {ex['sets']} sets of {ex['reps']} reps"
                }
                for ex in selected_exercises
            ],
            "estimated_calories": total_calories,
            "ai_generated": False,
            "ai_model": "Rule-Based Algorithm"
        }


# Main AI Workout Service
class AIWorkoutService:
    """Main service coordinating all AI providers with fallback chain"""
    
    def __init__(self):
        self.groq_ai = GroqAI()
        self.openrouter_ai = OpenRouterAI()
        self.huggingface_ai = HuggingFaceAI()
        self.ollama_ai = OllamaAI()
        self.rule_based = RuleBasedWorkoutGenerator()
    
    def create_ai_prompt(self, goal: str, level: str, duration: int, preferences: Dict = None) -> str:
        """Create optimized prompt for AI workout generation"""
        prompt = f"""Generate a {duration}-minute {goal} workout for {level} level.

Requirements:
- Goal: {goal}
- Fitness Level: {level}
- Duration: {duration} minutes
- Include 4-6 exercises
- Provide sets and reps for each exercise

Output ONLY this JSON format:
{{
    "name": "Workout Name",
    "goal": "{goal}",
    "difficulty": "{level}",
    "duration": {duration},
    "exercises": [
        {{
            "name": "Exercise Name",
            "sets": 3,
            "reps": "8-12",
            "description": "How to perform this exercise"
        }}
    ],
    "estimated_calories": 250
}}"""
        
        if preferences:
            prompt += f"\nPreferences: {preferences}"
        
        return prompt
    
    def generate_workout_sync(self, goal: str, level: str, duration: int = 30, preferences: Dict = None) -> Dict:
        """Generate workout using AI providers with fallback chain - SYNCHRONOUS"""
        start_time = time.time()
        
        logger.info("=" * 80)
        logger.info("🏋️ STARTING AI WORKOUT GENERATION")
        logger.info(f"🎯 Goal: {goal} | Level: {level} | Duration: {duration}min")
        logger.info("=" * 80)
        
        prompt = self.create_ai_prompt(goal, level, duration, preferences)
        
        # Method 1: Try Groq AI first
        logger.info("🔄 Attempting Method 1: Groq AI (Fastest)")
        logger.info("⚡ Groq Generation: Creating AI prompt...")
        try:
            groq_start = time.time()
            raw_response = self.groq_ai.generate_text(prompt)
            if raw_response:
                # Use the safe JSON extraction method
                workout_data = self.groq_ai.safe_json_extract(raw_response)
                if workout_data and "exercises" in workout_data:
                    groq_end = time.time()
                    logger.info(f"🎉 SUCCESS: Groq AI generated workout in {groq_end - groq_start:.2f}s")
                    workout_data.update({
                        "ai_generated": True,
                        "ai_model": "Groq AI",
                        "generation_time": round(groq_end - groq_start, 2)
                    })
                    return workout_data
                else:
                    logger.error("❌ Groq: Invalid JSON structure in response")
            else:
                logger.error("❌ Groq: No response generated")
        except Exception as e:
            logger.error(f"❌ Groq: Generation failed: {str(e)}")
        
        logger.warning("⚠️ Groq AI failed, trying next method...")
        
        # Method 2: Try OpenRouter AI
        logger.info("🔄 Attempting Method 2: OpenRouter AI (Free)")
        logger.info("🌐 OpenRouter Generation: Creating AI prompt...")
        try:
            openrouter_response = self.openrouter_ai.generate_text(prompt)
            if openrouter_response:
                workout_data = self.groq_ai.safe_json_extract(openrouter_response)
                if workout_data and "exercises" in workout_data:
                    logger.info("🎉 SUCCESS: OpenRouter AI generated workout")
                    workout_data.update({
                        "ai_generated": True,
                        "ai_model": "OpenRouter AI"
                    })
                    return workout_data
        except Exception as e:
            logger.error(f"❌ OpenRouter: Generation failed: {str(e)}")
        
        logger.warning("⚠️ OpenRouter AI failed, trying next method...")
        
        # Method 3: Try HuggingFace AI
        logger.info("🔄 Attempting Method 3: HuggingFace AI (Fixed)")
        logger.info("🤗 HuggingFace Generation: Creating AI prompt...")
        try:
            hf_response = self.huggingface_ai.generate_text(prompt)
            if hf_response:
                # Try to extract JSON from the response
                workout_data = self.groq_ai.safe_json_extract(hf_response)
                if workout_data and "exercises" in workout_data:
                    logger.info("🎉 SUCCESS: HuggingFace AI generated workout")
                    workout_data.update({
                        "ai_generated": True,
                        "ai_model": "HuggingFace AI"
                    })
                    return workout_data
        except Exception as e:
            logger.error(f"❌ HuggingFace: Generation failed: {str(e)}")
        
        logger.warning("⚠️ HuggingFace AI failed, trying next method...")
        
        # Method 4: Try Ollama AI
        logger.info("🔄 Attempting Method 4: Ollama AI (Local)")
        logger.info("🏠 Ollama Generation: Creating AI prompt...")
        try:
            ollama_response = self.ollama_ai.generate_text(prompt)
            if ollama_response:
                workout_data = self.groq_ai.safe_json_extract(ollama_response)
                if workout_data and "exercises" in workout_data:
                    logger.info("🎉 SUCCESS: Ollama AI generated workout")
                    workout_data.update({
                        "ai_generated": True,
                        "ai_model": "Ollama AI"
                    })
                    return workout_data
        except Exception as e:
            logger.error(f"❌ Ollama: Generation failed: {str(e)}")
        
        logger.warning("⚠️ Ollama AI failed, using fallback...")
        
        # Method 5: Rule-based fallback (guaranteed to work)
        logger.info("🔄 Attempting Method 5: Rule-based Fallback (Guaranteed)")
        try:
            rule_start = time.time()
            workout_data = self.rule_based.generate_workout(goal, level, duration)
            rule_end = time.time()
            
            logger.info(f"🎉 SUCCESS: Rule-based system generated workout in {rule_end - rule_start:.2f}s")
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Add timing information
            workout_data.update({
                "generation_time": round(rule_end - rule_start, 2),
                "total_processing_time": round(total_time, 2)
            })
            
            logger.info("=" * 80)
            logger.info("✅ WORKOUT GENERATION COMPLETED")
            logger.info(f"🤖 Generation Method: {workout_data.get('ai_model', 'Rule-based System')}")
            logger.info(f"⏱️ Generation Time: {workout_data.get('generation_time', 0):.2f}s")
            logger.info(f"🕐 Total Process Time: {total_time:.2f}s")
            logger.info(f"🏋️ Generated Workout: '{workout_data.get('name', 'Unknown')}'")
            logger.info(f"📝 Exercise Count: {len(workout_data.get('exercises', []))}")
            logger.info(f"🧠 AI Generated: {workout_data.get('ai_generated', False)}")
            logger.info(f"🔧 AI Model: {workout_data.get('ai_model', 'Unknown')}")
            logger.info("=" * 80)
            
            return workout_data
        
        except Exception as e:
            logger.error(f"❌ CRITICAL: Even rule-based fallback failed: {str(e)}")
            # Return absolute minimal fallback
            return {
                "name": "Basic Workout",
                "goal": goal,
                "difficulty": level,
                "duration": duration,
                "exercises": [
                    {"name": "Push-ups", "sets": 3, "reps": "10", "description": "Standard push-ups"},
                    {"name": "Squats", "sets": 3, "reps": "15", "description": "Bodyweight squats"}
                ],
                "estimated_calories": 100,
                "ai_generated": False,
                "ai_model": "Emergency Fallback"
            }

    # ASYNC METHOD - This is what your ai.py expects
    async def generate_workout(self, user, duration_minutes: int) -> Dict:
        """Async wrapper that matches your ai.py expectations - CORRECTED SIGNATURE"""
        logger.info("🧠 Starting AI workout generation process...")
        
        # Extract user details safely
        goal = getattr(user, 'fitness_goal', 'muscle_gain')
        level = getattr(user, 'fitness_level', 'intermediate')
        
        # Run the synchronous generation in a thread to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, 
            self.generate_workout_sync,
            goal,
            level, 
            duration_minutes,
            None
        )
        
        # Add required fields for compatibility with your ai.py
        result.update({
            "description": f"AI-generated {goal} workout for {level} level",
            "estimated_duration": duration_minutes
        })
        
        return result


# Create the service instances - BOTH for compatibility
ai_workout_service = AIWorkoutService()

# *** CRITICAL FIX: This is the exact object your ai.py imports ***
ai_workout_generator = ai_workout_service

# Debug: Log what we're exporting
logger.info(f"🔧 ai_workout_generator type: {type(ai_workout_generator)}")
logger.info(f"🔧 ai_workout_generator.generate_workout callable: {callable(getattr(ai_workout_generator, 'generate_workout', None))}")

# Also make sure it's definitely available at module level
__all__ = ['ai_workout_generator', 'ai_workout_service', 'AIWorkoutService']