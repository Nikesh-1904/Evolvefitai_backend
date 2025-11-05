# app/main.py
"""FastAPI application entry point"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
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

import os
import sys

def validate_environment():
    """Validate critical environment variables on startup"""
    print("\n" + "="*60)
    print("🔍 ENVIRONMENT VALIDATION")
    print("="*60)
    
    # Critical variables that must be set
    critical_vars = {
        'DATABASE_URL': os.getenv('DATABASE_URL'),
        'SECRET_KEY': os.getenv('SECRET_KEY'),
    }
    
    # Optional but recommended
    optional_vars = {
        'QR_ENCRYPTION_KEY': os.getenv('QR_ENCRYPTION_KEY'),
        'GROQ_API_KEY': os.getenv('GROQ_API_KEY'),
        'GOOGLE_CLIENT_ID': os.getenv('GOOGLE_CLIENT_ID'),
    }
    
    # Check critical variables
    missing_critical = [var for var, value in critical_vars.items() if not value]
    
    if missing_critical:
        print("❌ CRITICAL ERROR: Missing required environment variables:")
        for var in missing_critical:
            print(f"   ✗ {var}")
        print("\n💡 Fix: Set these variables in Railway dashboard")
        print("   Go to: Railway Project → Your Service → Variables")
        sys.exit(1)
    
    # Report on all variables
    print("\n✅ Critical variables:")
    for var in critical_vars:
        print(f"   ✓ {var}")
    
    print("\n⚙️  Optional variables:")
    for var, value in optional_vars.items():
        status = "✓" if value else "✗"
        print(f"   {status} {var}")
    
    # Special warning for QR key
    if not optional_vars['QR_ENCRYPTION_KEY']:
        print("\n⚠️  WARNING: QR_ENCRYPTION_KEY not set")
        print("   QR code check-in features will be disabled")
        print("   Generate with: python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\"")
    
    print("\n" + "="*60 + "\n")

# Call validation
validate_environment()

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

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Enhanced validation error handler that logs the actual error details
    """
    # Get the request body for debugging
    body = None
    try:
        body = await request.body()
        body_str = body.decode('utf-8')
    except:
        body_str = "Could not decode body"
    
    # Log detailed error information
    logger.error("=" * 80)
    logger.error("🚨 VALIDATION ERROR")
    logger.error(f"📍 Endpoint: {request.method} {request.url.path}")
    logger.error(f"📦 Request Body: {body_str}")
    logger.error(f"❌ Validation Errors:")
    for error in exc.errors():
        logger.error(f"   - Field: {error.get('loc')}")
        logger.error(f"     Type: {error.get('type')}")
        logger.error(f"     Message: {error.get('msg')}")
        logger.error(f"     Input: {error.get('input')}")
    logger.error("=" * 80)
    
    # Return user-friendly error response
    return JSONResponse(
        status_code=422,
        content={
            "detail": exc.errors(),
            "message": "Validation failed. Please check your request data.",
            "errors": [
                {
                    "field": ".".join(str(loc) for loc in error.get("loc", [])),
                    "message": error.get("msg", ""),
                    "type": error.get("type", "")
                }
                for error in exc.errors()
            ]
        }
    )
