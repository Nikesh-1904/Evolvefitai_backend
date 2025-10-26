# app/schemas/business_qr.py
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
import uuid

class QRCodeResponse(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID
    qr_code_data: str
    qr_code_image_base64: str
    is_active: bool
    created_at: datetime
    expires_at: Optional[datetime] = None

    class Config:
        from_attributes = True
