# app/schemas/achievement.py
from pydantic import BaseModel, ConfigDict
from typing import List
from datetime import datetime
import uuid

class UserAchievementBase(BaseModel):
    achievement_id: str


class UserAchievement(UserAchievementBase):
    model_config = ConfigDict(from_attributes=True)  
    id: uuid.UUID
    user_id: uuid.UUID
    unlocked_at: datetime



class AchievementUnlockRequest(BaseModel):
    achievement_id: str


class AchievementStatus(BaseModel):
    total_points: int
    level: int
    unlocked_achievements: List[UserAchievement]
