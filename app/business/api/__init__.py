# app/business/api/__init__.py
"""Business API endpoints for gym owners"""

from fastapi import APIRouter
from . import auth, members, fees

# Create main business router
business_router = APIRouter()

# Include sub-routers
business_router.include_router(auth.router, prefix="/auth", tags=["business-auth"])
business_router.include_router(members.router, prefix="/members", tags=["business-members"])
business_router.include_router(fees.router, prefix="/fees", tags=["business-fees"])

__all__ = ["business_router", "auth", "members", "fees"]
