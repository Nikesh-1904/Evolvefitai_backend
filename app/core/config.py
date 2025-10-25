# app/core/config.py
"""Application configuration using Pydantic Settings"""

import os
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Application
    APP_NAME: str = "EvolveFit AI"
    DEBUG: bool = False
    
    # Database
    DATABASE_URL: str
    
    # Security
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440  # 24 hours
    
    # CORS
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "https://evolvefit-ai.vercel.app",
        "https://evolvefit-business.vercel.app"
    ]
    
    # AI Services
    GROQ_API_KEY: str = ""
    OPENROUTER_API_KEY: str = ""
    HUGGINGFACE_API_KEY: str = ""
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    
    # YouTube API
    YOUTUBE_API_KEY: str = ""
    
    # Google OAuth
    GOOGLE_CLIENT_ID: str = ""
    GOOGLE_CLIENT_SECRET: str = ""
    GOOGLE_REDIRECT_URI: str = "http://localhost:8000/api/v1/auth/google/callback"
    
    # Email Configuration (Gmail SMTP)
    SMTP_HOST: str = "smtp.gmail.com"
    SMTP_PORT: int = 587
    SMTP_USER: str = ""
    SMTP_PASSWORD: str = ""  # Gmail App Password
    SENDER_EMAIL: str = ""
    SENDER_NAME: str = "EvolveFit AI"
    
    # QR Code Security
    QR_ENCRYPTION_KEY: str = ""  # Generate with: python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
    QR_VALIDITY_DAYS: int = 15
    
    # Fee Management
    DEFAULT_FEE_REMINDER_DAYS: List[int] = [7, 3, 1]  # Days before due date
    
    # Frontend URLs (for email links)
    CLIENT_FRONTEND_URL: str = "http://localhost:3000"
    BUSINESS_FRONTEND_URL: str = "http://localhost:3001"


# Create settings instance
settings = Settings()
