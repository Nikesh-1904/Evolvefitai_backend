# app/schemas/business_fees.py
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime, date
import uuid

class MembershipFeeBase(BaseModel):
    user_id: uuid.UUID
    amount: float
    due_date: date
    notes: Optional[str] = None


class MembershipFeeCreate(MembershipFeeBase):
    pass


class MembershipFeeUpdate(BaseModel):
    status: Optional[str] = None
    paid_date: Optional[datetime] = None
    payment_method: Optional[str] = None
    notes: Optional[str] = None


class MembershipFeeRead(MembershipFeeBase):
    id: uuid.UUID
    gym_id: uuid.UUID
    status: str
    paid_date: Optional[datetime] = None
    payment_method: Optional[str] = None
    currency: str
    receipt_number: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True

class BulkFeeCreate(BaseModel):
    user_ids: List[uuid.UUID]
    amount: float
    due_date: date
    notes: Optional[str] = None

