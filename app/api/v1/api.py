from fastapi import APIRouter

from app.api.v1 import auth, workouts, ai, stats, achievements, gyms, users

api_router = APIRouter()

# Include all route modules
api_router.include_router(auth.router, prefix="/auth", tags=["authentication"])
api_router.include_router(workouts.router, prefix="/workouts", tags=["workouts"])
api_router.include_router(ai.router, prefix="/ai", tags=["ai"])
api_router.include_router(stats.router, prefix="/stats", tags=["stats"])
api_router.include_router(achievements.router, prefix="/achievements", tags=["achievements"])
api_router.include_router(gyms.router, prefix="/gyms", tags=["gyms"])
api_router.include_router(users.router, prefix="/users", tags=["users"])
