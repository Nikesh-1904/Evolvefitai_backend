# app/schemas/business_notification.py
from pydantic import BaseModel
from typing import List


class NotificationPreferences(BaseModel):
    email_enabled: bool = True
    app_enabled: bool = True
    fee_reminder_days: List[int] = [7, 3, 1]
