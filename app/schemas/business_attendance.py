# app/schemas/business_attendance.py
from pydantic import BaseModel, ConfigDict
from typing import Optional, List, Dict, Any
from datetime import datetime
import uuid

class CheckInRequest(BaseModel):
    qr_code_data: str


class CheckOutRequest(BaseModel):
    attendance_id: uuid.UUID


class AttendanceRecord(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: uuid.UUID
    user_id: uuid.UUID
    gym_id: uuid.UUID
    check_in_time: datetime
    check_out_time: Optional[datetime] = None
    duration_minutes: Optional[int] = None



class LiveOccupancyResponse(BaseModel):
    gym_id: uuid.UUID
    gym_name: str
    current_occupancy: int
    capacity: int
    occupancy_percentage: float
    checked_in_users: List[Dict[str, Any]]
    timestamp: datetime
