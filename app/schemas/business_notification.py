# app/schemas/business_notification.py
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
import uuid

class NotificationCreate(BaseModel):
    user_ids: List[uuid.UUID]
    notification_type: str
    title: str
    message: str
    send_email: bool = False
    send_app: bool = True


class NotificationRead(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID
    notification_type: str
    title: str
    message: str
    is_read: bool
    sent_via_email: bool
    sent_via_app: bool
    created_at: datetime
    read_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class NotificationPreferences(BaseModel):
    email_enabled: bool = True
    app_enabled: bool = True
    fee_reminder_days: List[int] = [7, 3, 1]

