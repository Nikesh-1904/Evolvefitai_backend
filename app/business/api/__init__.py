# app/business/api/__init__.py
"""Business API endpoints for gym owners"""

from fastapi import APIRouter
from . import auth, members, fees, attendance, notifications, analytics

# Create main business router
business_router = APIRouter()

# Include all sub-routers
business_router.include_router(auth.router, prefix="/auth", tags=["business-auth"])
business_router.include_router(members.router, prefix="/members", tags=["business-members"])
business_router.include_router(fees.router, prefix="/fees", tags=["business-fees"])
business_router.include_router(attendance.router, prefix="/attendance", tags=["business-attendance"])
business_router.include_router(notifications.router, prefix="/notifications", tags=["business-notifications"])
business_router.include_router(analytics.router, prefix="/analytics", tags=["business-analytics"])

__all__ = ["business_router", "auth", "members", "fees", "attendance", "notifications", "analytics"]
