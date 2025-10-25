# app/business/services/qr_service.py
"""QR Code generation and validation service"""

import uuid
import qrcode
import io
import base64
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from cryptography.fernet import Fernet
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import logging

from app.core.config import settings
from app import models

logger = logging.getLogger(__name__)


class QRCodeService:
    """Service for generating and validating QR codes for gym members"""

    def __init__(self):
        """Initialize QR service with encryption key"""
        # Generate encryption key if not in settings
        # In production, this should be in environment variables
        try:
            self.cipher = Fernet(settings.QR_ENCRYPTION_KEY.encode())
        except Exception:
            # Fallback: generate a key (for development only!)
            logger.warning("QR_ENCRYPTION_KEY not found in settings, generating temporary key")
            self.cipher = Fernet(Fernet.generate_key())

    def generate_qr_code(
        self, 
        user_id: uuid.UUID, 
        gym_id: uuid.UUID,
        validity_days: int = 15
    ) -> Dict[str, Any]:
        """
        Generate encrypted QR code for a user
        
        Args:
            user_id: User's UUID
            gym_id: Gym's UUID
            validity_days: Number of days QR code is valid (default: 15)
        
        Returns:
            Dictionary with QR code data and image
        """
        try:
            # Create payload with timestamp
            issued_at = datetime.utcnow()
            expires_at = issued_at + timedelta(days=validity_days)
            
            payload = f"{user_id}:{gym_id}:{issued_at.timestamp()}"
            
            # Encrypt payload
            encrypted_data = self.cipher.encrypt(payload.encode()).decode()
            
            # Generate QR code image
            qr = qrcode.QRCode(
                version=1,
                error_correction=qrcode.constants.ERROR_CORRECT_H,
                box_size=10,
                border=4,
            )
            qr.add_data(encrypted_data)
            qr.make(fit=True)
            
            # Create image
            img = qr.make_image(fill_color="black", back_color="white")
            
            # Convert to base64
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            return {
                "qr_code_data": encrypted_data,
                "qr_code_image_base64": img_base64,
                "issued_at": issued_at,
                "expires_at": expires_at,
                "validity_days": validity_days
            }
            
        except Exception as e:
            logger.error(f"Error generating QR code: {str(e)}")
            raise ValueError(f"Failed to generate QR code: {str(e)}")

    def validate_qr_code(self, qr_code_data: str) -> Dict[str, Any]:
        """
        Decrypt and validate QR code
        
        Args:
            qr_code_data: Encrypted QR code string
        
        Returns:
            Dictionary with user_id, gym_id, and validation status
        
        Raises:
            ValueError: If QR code is invalid or expired
        """
        try:
            # Decrypt data
            decrypted = self.cipher.decrypt(qr_code_data.encode()).decode()
            
            # Parse payload
            parts = decrypted.split(":")
            if len(parts) != 3:
                raise ValueError("Invalid QR code format")
            
            user_id_str, gym_id_str, timestamp_str = parts
            
            # Convert to proper types
            user_id = uuid.UUID(user_id_str)
            gym_id = uuid.UUID(gym_id_str)
            issued_timestamp = float(timestamp_str)
            
            # Check expiry (15 days validity)
            issued_at = datetime.fromtimestamp(issued_timestamp)
            expires_at = issued_at + timedelta(days=15)
            
            if datetime.utcnow() > expires_at:
                raise ValueError("QR code has expired")
            
            return {
                "valid": True,
                "user_id": user_id,
                "gym_id": gym_id,
                "issued_at": issued_at,
                "expires_at": expires_at
            }
            
        except Exception as e:
            logger.error(f"QR validation failed: {str(e)}")
            return {
                "valid": False,
                "error": str(e)
            }

    async def get_or_create_qr_code(
        self,
        session: AsyncSession,
        user_id: uuid.UUID,
        gym_id: uuid.UUID
    ) -> models.UserQRCode:
        """
        Get existing QR code or create new one if expired/doesn't exist
        
        Args:
            session: Database session
            user_id: User's UUID
            gym_id: Gym's UUID
        
        Returns:
            UserQRCode model instance
        """
        # Check if user already has a QR code
        result = await session.execute(
            select(models.UserQRCode)
            .where(models.UserQRCode.user_id == user_id)
            .where(models.UserQRCode.gym_id == gym_id)
        )
        existing_qr = result.scalar_one_or_none()
        
        # Check if QR code exists and is still valid
        if existing_qr and existing_qr.is_active:
            if existing_qr.expires_at and existing_qr.expires_at > datetime.utcnow():
                logger.info(f"Using existing QR code for user {user_id}")
                return existing_qr
        
        # Generate new QR code
        qr_data = self.generate_qr_code(user_id, gym_id)
        
        if existing_qr:
            # Update existing record
            existing_qr.qr_code_data = qr_data["qr_code_data"]
            existing_qr.qr_code_image_base64 = qr_data["qr_code_image_base64"]
            existing_qr.is_active = True
            existing_qr.expires_at = qr_data["expires_at"]
            await session.commit()
            await session.refresh(existing_qr)
            logger.info(f"Regenerated QR code for user {user_id}")
            return existing_qr
        else:
            # Create new record
            new_qr = models.UserQRCode(
                user_id=user_id,
                gym_id=gym_id,
                qr_code_data=qr_data["qr_code_data"],
                qr_code_image_base64=qr_data["qr_code_image_base64"],
                is_active=True,
                expires_at=qr_data["expires_at"]
            )
            session.add(new_qr)
            await session.commit()
            await session.refresh(new_qr)
            logger.info(f"Created new QR code for user {user_id}")
            return new_qr

    async def deactivate_qr_code(
        self,
        session: AsyncSession,
        user_id: uuid.UUID,
        gym_id: uuid.UUID
    ) -> bool:
        """
        Deactivate a user's QR code (e.g., when membership expires)
        
        Args:
            session: Database session
            user_id: User's UUID
            gym_id: Gym's UUID
        
        Returns:
            True if deactivated, False if not found
        """
        result = await session.execute(
            select(models.UserQRCode)
            .where(models.UserQRCode.user_id == user_id)
            .where(models.UserQRCode.gym_id == gym_id)
        )
        qr_code = result.scalar_one_or_none()
        
        if qr_code:
            qr_code.is_active = False
            await session.commit()
            logger.info(f"Deactivated QR code for user {user_id}")
            return True
        
        return False


# Singleton instance
qr_service = QRCodeService()
