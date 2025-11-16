# app/schemas/gym_registration.py
"""Schema for combined gym + owner registration"""

from pydantic import BaseModel, EmailStr
from typing import Optional


class GymWithOwnerCreate(BaseModel):
    """Schema for registering a new gym with its owner"""
    # Gym details
    gym_name: str
    address: Optional[str] = None
    phone: Optional[str] = None

    # Owner details
    owner_name: str  # Maps to GymOwner.full_name
    email: EmailStr
    password: str


class GymRegistrationResponse(BaseModel):
    """Response after successful gym + owner registration"""
    access_token: str
    token_type: str
    user: dict  # Contains owner details and gym info
