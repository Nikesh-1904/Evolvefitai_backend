# app/schemas/gym.py
from pydantic import BaseModel, ConfigDict
from typing import Optional, List, Dict, Any
from datetime import datetime
import uuid

class GymBase(BaseModel):
    name: str
    address: str
    city: str
    state: Optional[str] = None
    country: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    logo_url: Optional[str] = None
    operating_hours: Optional[Dict[str, Any]] = {}
    max_capacity: int = 100


class GymCreate(GymBase):
    gym_code: Optional[str] = None


class Gym(GymBase):
    model_config = ConfigDict(from_attributes=True)
    id: uuid.UUID
    created_at: datetime
    updated_at: Optional[datetime] = None
    member_count: int = 0
    gym_code: Optional[str] = None


class JoinByCodeRequest(BaseModel):
    gym_code: str


class GymBookingBase(BaseModel):
    gym_id: uuid.UUID
    start_time: datetime
    end_time: datetime


class GymBookingCreate(GymBookingBase):
    pass


class GymBooking(GymBookingBase):
    model_config = ConfigDict(from_attributes=True)
    id: uuid.UUID
    user_id: uuid.UUID
    status: str
    created_at: datetime
    cancelled_at: Optional[datetime] = None



class GymOccupancyResponse(BaseModel):
    gym_id: uuid.UUID
    gym_name: str
    current_occupancy: int
    max_capacity: int
    overflow_count: int
    is_overcrowded: bool
    active_bookings_count: int


class LeaderboardEntry(BaseModel):
    user_id: uuid.UUID
    user_name: str
    total_workouts: int
    total_calories_burned: float
    total_minutes: int
    consistency_score: float
    rank: int


class LeaderboardResponse(BaseModel):
    gym_id: uuid.UUID
    gym_name: str
    gym_address: str
    leaderboard: List[LeaderboardEntry]
    total_members: int
