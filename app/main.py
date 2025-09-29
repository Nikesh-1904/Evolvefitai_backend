from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.api.v1.api import api_router
from app.core.database import create_db_and_tables

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="EvolveFit AI - Your Intelligent Fitness Companion with AI-powered workouts, meal planning, and plateau detection"
)

# CORS configuration for Vercel frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Local development
        "https://*.vercel.app",   # Vercel deployments
        settings.FRONTEND_URL,    # Production frontend
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router, prefix="/api/v1")

@app.on_event("startup")
async def on_startup():
    """Create database tables on startup"""
    await create_db_and_tables()

@app.get("/")
async def root():
    return {"message": "EvolveFit AI Backend is running!", "version": settings.VERSION}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": settings.VERSION}