# app/schemas/business_owner.py
from pydantic import BaseModel, EmailStr ,  ConfigDict
from typing import Optional
from datetime import datetime
import uuid

class GymOwnerBase(BaseModel):
    email: EmailStr
    full_name: str
    phone_number: Optional[str] = None


class GymOwnerCreate(GymOwnerBase):
    password: str
    gym_id: uuid.UUID


class GymOwnerRead(GymOwnerBase):
    model_config = ConfigDict(from_attributes=True)
    id: uuid.UUID
    gym_id: uuid.UUID
    is_active: bool
    created_at: datetime
    last_login: Optional[datetime] = None



class GymOwnerLogin(BaseModel):
    email: EmailStr
    password: str

class GymOwnerUpdate(BaseModel):
    full_name: Optional[str] = None
    phone_number: Optional[str] = None

