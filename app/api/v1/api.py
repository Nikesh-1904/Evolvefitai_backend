from fastapi import APIRouter

from app.api.v1 import auth, workouts, meal_plans, ai, stats, gyms, community, achievements

api_router = APIRouter()

# Include all route modules
api_router.include_router(auth.router, prefix="/auth", tags=["authentication"])
api_router.include_router(workouts.router, prefix="/workouts", tags=["workouts"])
api_router.include_router(ai.router, prefix="/ai", tags=["ai"])
api_router.include_router(meal_plans.router, prefix="/meal-plans", tags=["meal_plans"])
api_router.include_router(stats.router, prefix="/stats", tags=["stats"]) # 👈 Add this line
api_router.include_router(gyms.router, prefix="/gyms", tags=["gyms"])
api_router.include_router(community.router, prefix="/community", tags=["community"])
api_router.include_router(achievements.router, prefix="/achievements", tags=["achievements"])