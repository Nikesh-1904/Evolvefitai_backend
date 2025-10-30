# app/schemas/business_qr.py
from pydantic import BaseModel, ConfigDict
from typing import Optional
from datetime import datetime
import uuid

class QRCodeResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: uuid.UUID
    user_id: uuid.UUID
    qr_code_data: str
    qr_code_image_base64: str
    is_active: bool
    created_at: datetime
    expires_at: Optional[datetime] = None

