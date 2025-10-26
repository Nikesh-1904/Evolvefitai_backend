# app/schemas/business_analytics.py
from pydantic import BaseModel
from typing import Optional, List, Dict
from datetime import datetime, date
import uuid
# Import AttendanceRecord if it's used here (it is in AnalyticsDashboard)
from .business_attendance import AttendanceRecord

class MemberStats(BaseModel):
    user_id: uuid.UUID
    username: str
    full_name: str
    total_workouts: int
    avg_duration: float
    consistency_score: float
    membership_status: str
    last_payment_date: Optional[date] = None
    next_due_date: Optional[date] = None


class AnalyticsDashboard(BaseModel):
    total_members: int
    active_members: int
    revenue_this_month: float
    avg_attendance_per_day: float
    top_performers: List[MemberStats]
    recent_activity: List[AttendanceRecord]


class RevenueReport(BaseModel):
    total_revenue: float
    paid_count: int
    pending_count: int
    overdue_count: int
    breakdown_by_month: Dict[str, float]

class PerformanceAnalysis(BaseModel):
    total_workouts: int
    avg_duration: float
    consistency_score: float
    performance_trend: str
    weak_areas: Optional[List[str]] = []
    suggestions: Optional[List[str]] = []
    analysis_date: Optional[datetime] = None
