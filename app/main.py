# app/main.py
"""FastAPI application entry point"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from app.core.config import settings
from app.core.database import create_db_and_tables

# Import routers
from app.api.v1.api import api_router  # Existing client routes
from app.business.api import business_router  # Business routes
from app.shared.api import shared_router  # Shared routes
from app import models  # Ensure all models are imported for Alembic

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events
    
    Startup:
    - Create database tables if they don't exist
    - Start background services (future: fee reminders scheduler)
    
    Shutdown:
    - Clean up resources
    """
    # Startup
    logger.info("Starting EvolveFit AI Backend...")
    logger.info(f"Environment: {'Development' if settings.DEBUG else 'Production'}")
    
    # Create database tables
    await create_db_and_tables()
    logger.info("Database tables created/verified")
    
    # TODO: Start background scheduler for fee reminders
    # from app.business.services.fee_reminder_service import start_scheduler
    # start_scheduler()
    
    logger.info("Application startup complete")
    
    yield
    
    # Shutdown
    logger.info("Shutting down EvolveFit AI Backend...")
    # TODO: Stop background scheduler
    logger.info("Shutdown complete")


# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    version="2.0.0",
    description="Fitness tracking platform with AI-powered workouts, meal plans, and gym management",
    lifespan=lifespan
)

origins = [
    "https://evolvefitai-frontend.vercel.app",  # Your frontend
    "http://localhost:3000",                  # For local development
]

if isinstance(settings.ALLOWED_ORIGINS, list):
    origins.extend(settings.ALLOWED_ORIGINS)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers

# Client routes (existing)
app.include_router(
    api_router,
    prefix="/api/v1"
)

# Business routes (new)
app.include_router(
    business_router,
    prefix="/api/v1/business",
    tags=["business"]
)

# Shared routes (new)
app.include_router(
    shared_router,
    prefix="/api/v1",
    tags=["shared"]
)

# Root endpoint
@app.get("/", tags=["root"])
async def root():
    """API root endpoint"""
    return {
        "message": "Welcome to EvolveFit AI API",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health"
    }

# Health check endpoint
@app.get("/health", tags=["health"])
async def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "app": settings.APP_NAME,
        "version": "2.0.0"
    }


# Request logging middleware (optional)
@app.middleware("http")
async def log_requests(request, call_next):
    """Log all HTTP requests"""
    logger.info(f"{request.method} {request.url.path}")
    response = await call_next(request)
    logger.info(f"Status: {response.status_code}")
    return response


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG
    )
