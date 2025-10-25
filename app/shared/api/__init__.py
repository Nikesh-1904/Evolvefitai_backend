# app/shared/api/__init__.py
"""Shared API endpoints accessible to both clients and business owners"""

from fastapi import APIRouter
from . import gyms, users

# Create shared router
shared_router = APIRouter()

# Include sub-routers
shared_router.include_router(gyms.router, prefix="/gyms", tags=["gyms"])
shared_router.include_router(users.router, prefix="/users", tags=["users"])

__all__ = ["shared_router", "gyms", "users"]
