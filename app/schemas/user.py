# app/schemas/user.py
from pydantic import EmailStr, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
import uuid
from fastapi_users import schemas

# Import UserAchievement for the relationship hint
from .achievement import UserAchievement

class UserRead(schemas.BaseUser[uuid.UUID]):
    id: uuid.UUID
    email: EmailStr
    username: Optional[str] = None
    full_name: Optional[str] = None
    age: Optional[int] = None
    weight: Optional[float] = None
    height: Optional[float] = None
    gender: Optional[str] = None
    fitness_goal: Optional[str] = None
    experience_level: Optional[str] = None
    activity_level: Optional[str] = None
    dietary_restrictions: Optional[List[str]] = []
    has_completed_onboarding: bool
    preferences: Optional[Dict[str, Any]] = {}
    gym_id: Optional[int] = None
    last_gym_change: Optional[datetime] = None
    is_active: bool
    is_verified: bool
    created_at: datetime
    total_points: int
    level: int
    achievements: List[UserAchievement] = []
    class Config:
        from_attributes = True

class UserCreate(schemas.BaseUserCreate):
    email: EmailStr
    password: str
    username: Optional[str] = None
    full_name: Optional[str] = None


class UserUpdate(schemas.BaseUserUpdate):
    username: Optional[str] = None
    full_name: Optional[str] = None
    age: Optional[int] = None
    weight: Optional[float] = None
    height: Optional[float] = None
    gender: Optional[str] = None
    fitness_goal: Optional[str] = None
    experience_level: Optional[str] = None
    activity_level: Optional[str] = None
    dietary_restrictions: Optional[List[str]] = []
    has_completed_onboarding: Optional[bool] = None
    preferences: Optional[Dict[str, Any]] = None
    gym_id: Optional[int] = None

    @validator("username")
    def username_must_not_be_empty(cls, v):
        if v is not None and v == "":
            raise ValueError("Username cannot be an empty string")
        return v

    class Config:
        from_attributes = True