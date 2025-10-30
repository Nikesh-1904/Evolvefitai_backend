# app/schemas/achievement.py
from pydantic import BaseModel
from typing import List
from datetime import datetime
import uuid

class UserAchievementBase(BaseModel):
    achievement_id: str


class UserAchievement(UserAchievementBase):
    id: uuid.UUID
    user_id: uuid.UUID
    unlocked_at: datetime

    class Config:
        from_attributes = True


class AchievementUnlockRequest(BaseModel):
    achievement_id: str


class AchievementStatus(BaseModel):
    total_points: int
    level: int
    unlocked_achievements: List[UserAchievement]
