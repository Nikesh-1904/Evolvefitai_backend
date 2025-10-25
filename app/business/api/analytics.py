# app/business/api/analytics.py
"""Analytics and reporting endpoints for gym owners"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_, case
from typing import List, Optional
from datetime import datetime, timedelta, date
import uuid

from app.core.database import get_async_session
from app.core.business_auth import get_current_gym_owner
from app import models, schemas

router = APIRouter()


@router.get("/overview", response_model=schemas.AnalyticsDashboard)
async def get_dashboard_overview(
    current_owner: models.GymOwner = Depends(get_current_gym_owner),
    session: AsyncSession = Depends(get_async_session)
):
    """
    Get high-level analytics dashboard
    
    Shows:
    - Total members & active members
    - Revenue this month
    - Average daily attendance
    - Top performing members
    - Recent activity
    """
    # Total members count
    total_members = await session.scalar(
        select(func.count(models.User.id))
        .where(models.User.gym_id == current_owner.gym_id)
    )
    
    # Active members (status = ACTIVE)
    active_members = await session.scalar(
        select(func.count(models.User.id))
        .where(models.User.gym_id == current_owner.gym_id)
        .where(models.User.membership_status == "ACTIVE")
    )
    
    # Revenue this month
    month_start = datetime.utcnow().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    
    revenue_result = await session.execute(
        select(func.sum(models.MembershipFee.amount))
        .where(models.MembershipFee.gym_id == current_owner.gym_id)
        .where(models.MembershipFee.status == "PAID")
        .where(models.MembershipFee.paid_date >= month_start)
    )
    revenue_this_month = revenue_result.scalar() or 0.0
    
    # Average daily attendance (last 30 days)
    thirty_days_ago = datetime.utcnow() - timedelta(days=30)
    
    attendance_count = await session.scalar(
        select(func.count(models.GymAttendance.id))
        .where(models.GymAttendance.gym_id == current_owner.gym_id)
        .where(models.GymAttendance.check_in_time >= thirty_days_ago)
    )
    avg_attendance_per_day = round((attendance_count or 0) / 30, 1)
    
    # Top performers (by total points)
    top_performers_result = await session.execute(
        select(models.User)
        .where(models.User.gym_id == current_owner.gym_id)
        .where(models.User.membership_status == "ACTIVE")
        .order_by(models.User.total_points.desc())
        .limit(5)
    )
    top_performers = top_performers_result.scalars().all()
    
    # Recent attendance activity
    recent_activity_result = await session.execute(
        select(models.GymAttendance)
        .where(models.GymAttendance.gym_id == current_owner.gym_id)
        .order_by(models.GymAttendance.check_in_time.desc())
        .limit(10)
    )
    recent_activity = recent_activity_result.scalars().all()
    
    # Build top performers list
    top_performers_list = []
    for user in top_performers:
        # Get workout stats
        workout_count = await session.scalar(
            select(func.count(models.WorkoutLog.id))
            .where(models.WorkoutLog.user_id == user.id)
        )
        
        avg_duration_result = await session.scalar(
            select(func.avg(models.WorkoutLog.duration_minutes))
            .where(models.WorkoutLog.user_id == user.id)
        )
        
        top_performers_list.append(schemas.MemberStats(
            user_id=user.id,
            username=user.username or "Unknown",
            full_name=user.full_name or user.username or "Unknown",
            total_workouts=workout_count or 0,
            avg_duration=round(avg_duration_result or 0, 1),
            consistency_score=70.0,  # Placeholder - can calculate properly
            membership_status=user.membership_status,
            last_payment_date=None,  # Can fetch from fees
            next_due_date=None
        ))
    
    return schemas.AnalyticsDashboard(
        total_members=total_members or 0,
        active_members=active_members or 0,
        revenue_this_month=revenue_this_month,
        avg_attendance_per_day=avg_attendance_per_day,
        top_performers=top_performers_list,
        recent_activity=recent_activity
    )


@router.get("/revenue", response_model=schemas.RevenueReport)
async def get_revenue_report(
    start_date: Optional[date] = Query(None),
    end_date: Optional[date] = Query(None),
    current_owner: models.GymOwner = Depends(get_current_gym_owner),
    session: AsyncSession = Depends(get_async_session)
):
    """
    Get revenue report for a date range
    
    Shows:
    - Total revenue (paid)
    - Pending revenue
    - Collected revenue
    - Overdue count
    """
    # Default to current month if no dates provided
    if not end_date:
        end_date = date.today()
    if not start_date:
        start_date = end_date.replace(day=1)
    
    start_datetime = datetime.combine(start_date, datetime.min.time())
    end_datetime = datetime.combine(end_date, datetime.max.time())
    
    # Total revenue (all fees in period)
    total_revenue_result = await session.execute(
        select(func.sum(models.MembershipFee.amount))
        .where(models.MembershipFee.gym_id == current_owner.gym_id)
        .where(and_(
            models.MembershipFee.payment_date >= start_datetime,
            models.MembershipFee.payment_date <= end_datetime
        ))
    )
    total_revenue = total_revenue_result.scalar() or 0.0
    
    # Pending revenue
    pending_revenue_result = await session.execute(
        select(func.sum(models.MembershipFee.amount))
        .where(models.MembershipFee.gym_id == current_owner.gym_id)
        .where(models.MembershipFee.status == "PENDING")
        .where(and_(
            models.MembershipFee.due_date >= start_datetime,
            models.MembershipFee.due_date <= end_datetime
        ))
    )
    pending_revenue = pending_revenue_result.scalar() or 0.0
    
    # Collected revenue (paid)
    collected_revenue_result = await session.execute(
        select(func.sum(models.MembershipFee.amount))
        .where(models.MembershipFee.gym_id == current_owner.gym_id)
        .where(models.MembershipFee.status == "PAID")
        .where(and_(
            models.MembershipFee.paid_date >= start_datetime,
            models.MembershipFee.paid_date <= end_datetime
        ))
    )
    collected_revenue = collected_revenue_result.scalar() or 0.0
    
    # Overdue count
    overdue_count = await session.scalar(
        select(func.count(models.MembershipFee.id))
        .where(models.MembershipFee.gym_id == current_owner.gym_id)
        .where(or_(
            models.MembershipFee.status == "PENDING",
            models.MembershipFee.status == "OVERDUE"
        ))
        .where(models.MembershipFee.due_date < datetime.utcnow())
    )
    
    return schemas.RevenueReport(
        total_revenue=total_revenue,
        pending_revenue=pending_revenue,
        collected_revenue=collected_revenue,
        overdue_count=overdue_count or 0,
        period_start=start_date,
        period_end=end_date
    )


@router.get("/attendance-trends")
async def get_attendance_trends(
    days: int = Query(30, ge=1, le=365),
    current_owner: models.GymOwner = Depends(get_current_gym_owner),
    session: AsyncSession = Depends(get_async_session)
):
    """
    Get attendance trends over time
    
    Returns daily attendance counts and patterns
    """
    start_date = datetime.utcnow() - timedelta(days=days)
    
    # Get attendance by date
    result = await session.execute(
        select(
            func.date(models.GymAttendance.check_in_time).label('date'),
            func.count(models.GymAttendance.id).label('check_ins'),
            func.avg(models.GymAttendance.duration_minutes).label('avg_duration')
        )
        .where(models.GymAttendance.gym_id == current_owner.gym_id)
        .where(models.GymAttendance.check_in_time >= start_date)
        .group_by(func.date(models.GymAttendance.check_in_time))
        .order_by(func.date(models.GymAttendance.check_in_time))
    )
    
    trends = []
    for row in result:
        trends.append({
            "date": row.date.isoformat(),
            "total_check_ins": row.check_ins,
            "avg_duration_minutes": round(row.avg_duration or 0, 1)
        })
    
    return {
        "period_days": days,
        "data": trends
    }


@router.get("/member-performance")
async def get_all_member_performance(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    current_owner: models.GymOwner = Depends(get_current_gym_owner),
    session: AsyncSession = Depends(get_async_session)
):
    """
    Get performance analytics for all members
    
    Useful for identifying members who need attention
    """
    result = await session.execute(
        select(models.MemberPerformance)
        .where(models.MemberPerformance.gym_id == current_owner.gym_id)
        .order_by(models.MemberPerformance.analysis_date.desc())
        .offset(skip)
        .limit(limit)
    )
    
    performance_records = result.scalars().all()
    
    return {
        "total_analyzed": len(performance_records),
        "members": [
            {
                "user_id": str(p.user_id),
                "total_workouts": p.total_workouts,
                "avg_duration": p.avg_workout_duration,
                "consistency_score": p.consistency_score,
                "trend": p.performance_trend,
                "weak_areas": p.weak_areas,
                "suggestions": p.suggestions,
                "analyzed_on": p.analysis_date.isoformat()
            }
            for p in performance_records
        ]
    }


@router.get("/retention")
async def get_retention_metrics(
    current_owner: models.GymOwner = Depends(get_current_gym_owner),
    session: AsyncSession = Depends(get_async_session)
):
    """
    Get member retention metrics
    
    Shows:
    - Active vs expired members
    - Churn rate
    - Average membership duration
    """
    # Total members ever
    total_ever = await session.scalar(
        select(func.count(models.User.id))
        .where(models.User.gym_id == current_owner.gym_id)
    )
    
    # Currently active
    active_count = await session.scalar(
        select(func.count(models.User.id))
        .where(models.User.gym_id == current_owner.gym_id)
        .where(models.User.membership_status == "ACTIVE")
    )
    
    # Expired
    expired_count = await session.scalar(
        select(func.count(models.User.id))
        .where(models.User.gym_id == current_owner.gym_id)
        .where(models.User.membership_status == "EXPIRED")
    )
    
    # Suspended
    suspended_count = await session.scalar(
        select(func.count(models.User.id))
        .where(models.User.gym_id == current_owner.gym_id)
        .where(models.User.membership_status == "SUSPENDED")
    )
    
    # Retention rate
    retention_rate = round((active_count / total_ever * 100), 1) if total_ever > 0 else 0
    churn_rate = round(100 - retention_rate, 1)
    
    return {
        "total_members_ever": total_ever or 0,
        "active_members": active_count or 0,
        "expired_members": expired_count or 0,
        "suspended_members": suspended_count or 0,
        "retention_rate_percent": retention_rate,
        "churn_rate_percent": churn_rate
    }


@router.get("/export")
async def export_analytics_report(
    report_type: str = Query(..., description="revenue, attendance, or members"),
    format: str = Query("json", description="json or csv"),
    current_owner: models.GymOwner = Depends(get_current_gym_owner),
    session: AsyncSession = Depends(get_async_session)
):
    """
    Export analytics data
    
    Supports:
    - Revenue report
    - Attendance report
    - Member list
    
    Formats:
    - JSON
    - CSV (future enhancement)
    """
    if report_type not in ["revenue", "attendance", "members"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="report_type must be 'revenue', 'attendance', or 'members'"
        )
    
    # For now, return JSON format
    # CSV export can be added later
    
    if report_type == "revenue":
        # Get all fee records
        result = await session.execute(
            select(models.MembershipFee)
            .where(models.MembershipFee.gym_id == current_owner.gym_id)
            .order_by(models.MembershipFee.due_date.desc())
        )
        fees = result.scalars().all()
        
        return {
            "report_type": "revenue",
            "generated_at": datetime.utcnow().isoformat(),
            "data": [
                {
                    "fee_id": str(f.id),
                    "user_id": str(f.user_id),
                    "amount": f.amount,
                    "status": f.status,
                    "due_date": f.due_date.isoformat(),
                    "paid_date": f.paid_date.isoformat() if f.paid_date else None
                }
                for f in fees
            ]
        }
    
    elif report_type == "attendance":
        # Get attendance records
        result = await session.execute(
            select(models.GymAttendance)
            .where(models.GymAttendance.gym_id == current_owner.gym_id)
            .order_by(models.GymAttendance.check_in_time.desc())
            .limit(1000)  # Limit for performance
        )
        attendance = result.scalars().all()
        
        return {
            "report_type": "attendance",
            "generated_at": datetime.utcnow().isoformat(),
            "data": [
                {
                    "attendance_id": str(a.id),
                    "user_id": str(a.user_id),
                    "check_in": a.check_in_time.isoformat(),
                    "check_out": a.check_out_time.isoformat() if a.check_out_time else None,
                    "duration_minutes": a.duration_minutes
                }
                for a in attendance
            ]
        }
    
    else:  # members
        result = await session.execute(
            select(models.User)
            .where(models.User.gym_id == current_owner.gym_id)
        )
        members = result.scalars().all()
        
        return {
            "report_type": "members",
            "generated_at": datetime.utcnow().isoformat(),
            "data": [
                {
                    "user_id": str(m.id),
                    "username": m.username,
                    "full_name": m.full_name,
                    "email": m.email,
                    "membership_status": m.membership_status,
                    "total_points": m.total_points,
                    "current_level": m.current_level
                }
                for m in members
            ]
        }
