from fastapi import APIRouter

from app.api.v1 import auth, workouts, ai

api_router = APIRouter()

# Include all route modules
api_router.include_router(auth.router, prefix="/auth", tags=["authentication"])
api_router.include_router(workouts.router, prefix="/workouts", tags=["workouts"])
api_router.include_router(ai.router, prefix="/ai", tags=["ai"])