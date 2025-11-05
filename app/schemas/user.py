# app/schemas/user.py
from pydantic import EmailStr, validator, ConfigDict
from typing import Optional, List, Dict, Any
from datetime import datetime
import uuid
from fastapi_users import schemas

# Import UserAchievement for the relationship hint
from .achievement import UserAchievement

class UserRead(schemas.BaseUser[uuid.UUID]):
    model_config = ConfigDict(
        from_attributes=True,
        # ✅ FIX: Don't validate on assignment to avoid eager loading issues
        validate_assignment=False
    )
    
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
    gym_id: Optional[uuid.UUID] = None
    last_gym_change: Optional[datetime] = None
    is_active: bool
    is_verified: bool
    created_at: datetime  # ✅ FIX: Added this required field
    total_points: int
    level: int
    
    # ✅ FIX: Make achievements optional with default
    achievements: List[UserAchievement] = []
    

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