# app/core/config.py
"""Application configuration using Pydantic Settings"""

import os
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Pydantic will automatically load variables from .env and the environment
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,  # Reads DATABASE_URL or database_url
        extra="ignore"
    )
    
    # Application
    APP_NAME: str = "EvolveFit AI"
    # Pydantic automatically converts "true", "1", "false", "0" from env vars
    DEBUG: bool = False
    
    # Database
    # Will be overridden by DATABASE_URL env var on Railway
    DATABASE_URL: str = "postgresql+asyncpg://user:password@localhost/evolvefitai"
    
    # Security
    # Will be overridden by SECRET_KEY env var on Railway
    SECRET_KEY: str = "your-secret-key-here-please-change-me-for-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440  # 24 hours
    
    # CORS
    # Pydantic can parse this from an env var like:
    # ALLOWED_ORIGINS="http://url1.com,http://url2.com"
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "https://evolvefit-ai.vercel.app",
        "https://evolvefit-business.vercel.app",
        "https://evolvefitai-frontend.vercel.app",  # Current production frontend
    ]
    
    # AI Services (will be read from env vars)
    GROQ_API_KEY: str = ""
    OPENROUTER_API_KEY: str = ""
    HUGGINGFACE_API_KEY: str = ""
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    
    # YouTube API (will be read from env var)
    YOUTUBE_API_KEY: str = ""
    
    # Google OAuth (will be read from env vars)
    GOOGLE_CLIENT_ID: str = ""
    GOOGLE_CLIENT_SECRET: str = ""
    GOOGLE_REDIRECT_URI: str = "http://localhost:8000/api/v1/auth/google/callback"
    
    # Email Configuration (will be read from env vars)
    SMTP_HOST: str = "smtp.gmail.com"
    SMTP_PORT: int = 587
    SMTP_USER: str = ""
    SMTP_PASSWORD: str = ""  # Gmail App Password
    SENDER_EMAIL: str = ""
    SENDER_NAME: str = "EvolveFit AI"
    
    # QR Code Security (will be read from env var)
    # MUST be set in production: 
    # python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
    QR_ENCRYPTION_KEY: str = ""
    QR_VALIDITY_DAYS: int = 15
    
    # Fee Management
    # Can be overridden by env var: DEFAULT_FEE_REMINDER_DAYS="7,3,1"
    DEFAULT_FEE_REMINDER_DAYS: List[int] = [7, 3, 1]  # Days before due date
    
    # Frontend URLs (for email links)
    CLIENT_FRONTEND_URL: str = "http://localhost:3000"
    BUSINESS_FRONTEND_URL: str = "http://localhost:3001"


# Create settings instance
settings = Settings()