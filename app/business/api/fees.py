# app/business/api/fees.py
"""Fee management endpoints for gym owners"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_
from typing import List, Optional
from datetime import datetime, timedelta, date
import uuid

from app.core.database import get_async_session
from app.core.business_auth import get_current_gym_owner
from app.business.services.email_service import email_service
from app import models, schemas

router = APIRouter()


@router.post("/", response_model=schemas.MembershipFee, status_code=status.HTTP_201_CREATED)
async def create_fee_record(
    fee_data: schemas.MembershipFeeCreate,
    current_owner: models.GymOwner = Depends(get_current_gym_owner),
    session: AsyncSession = Depends(get_async_session)
):
    """
    Create a new fee record for a member
    
    - **user_id**: Member's UUID
    - **amount**: Fee amount
    - **due_date**: Payment due date
    - **notes**: Optional notes
    """
    # Verify user exists and belongs to this gym
    user = await session.get(models.User, fee_data.user_id)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Member not found"
        )
    
    if user.gym_id != current_owner.gym_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This member doesn't belong to your gym"
        )
    
    # Get gym details for currency
    gym = await session.get(models.Gym, current_owner.gym_id)
    
    # Generate receipt number
    receipt_number = f"FEE-{datetime.utcnow().strftime('%Y%m%d')}-{uuid.uuid4().hex[:8].upper()}"
    
    # Create fee record
    new_fee = models.MembershipFee(
        user_id=fee_data.user_id,
        gym_id=current_owner.gym_id,
        amount=fee_data.amount,
        currency=gym.currency if gym else "INR",
        due_date=fee_data.due_date,
        status="PENDING",
        receipt_number=receipt_number,
        notes=fee_data.notes,
        created_by=current_owner.id
    )
    
    session.add(new_fee)
    await session.commit()
    await session.refresh(new_fee)
    
    return new_fee


@router.get("/", response_model=List[schemas.MembershipFee])
async def list_fees(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    status_filter: Optional[str] = Query(None, description="PENDING, PAID, OVERDUE, CANCELLED"),
    user_id: Optional[uuid.UUID] = Query(None, description="Filter by specific member"),
    current_owner: models.GymOwner = Depends(get_current_gym_owner),
    session: AsyncSession = Depends(get_async_session)
):
    """
    Get list of all fee records for the gym
    
    Can filter by:
    - Status (PENDING, PAID, OVERDUE, CANCELLED)
    - Specific member (user_id)
    """
    query = select(models.MembershipFee).where(
        models.MembershipFee.gym_id == current_owner.gym_id
    )
    
    if status_filter:
        query = query.where(models.MembershipFee.status == status_filter)
    
    if user_id:
        query = query.where(models.MembershipFee.user_id == user_id)
    
    query = query.order_by(models.MembershipFee.due_date.desc()).offset(skip).limit(limit)
    
    result = await session.execute(query)
    fees = result.scalars().all()
    
    return fees


@router.get("/overdue", response_model=List[schemas.MembershipFee])
async def get_overdue_fees(
    current_owner: models.GymOwner = Depends(get_current_gym_owner),
    session: AsyncSession = Depends(get_async_session)
):
    """
    Get all overdue fee records
    
    Returns fees where:
    - Status is PENDING or OVERDUE
    - Due date has passed
    """
    result = await session.execute(
        select(models.MembershipFee)
        .where(models.MembershipFee.gym_id == current_owner.gym_id)
        .where(or_(
            models.MembershipFee.status == "PENDING",
            models.MembershipFee.status == "OVERDUE"
        ))
        .where(models.MembershipFee.due_date < datetime.utcnow())
        .order_by(models.MembershipFee.due_date)
    )
    
    overdue_fees = result.scalars().all()
    
    # Update status to OVERDUE if still PENDING
    for fee in overdue_fees:
        if fee.status == "PENDING":
            fee.status = "OVERDUE"
    
    await session.commit()
    
    return overdue_fees


@router.get("/upcoming")
async def get_upcoming_fees(
    days: int = Query(7, ge=1, le=90, description="Look ahead this many days"),
    current_owner: models.GymOwner = Depends(get_current_gym_owner),
    session: AsyncSession = Depends(get_async_session)
):
    """
    Get fees due in the next N days
    
    Useful for planning and sending reminders
    """
    start_date = datetime.utcnow()
    end_date = start_date + timedelta(days=days)
    
    result = await session.execute(
        select(models.MembershipFee)
        .where(models.MembershipFee.gym_id == current_owner.gym_id)
        .where(models.MembershipFee.status == "PENDING")
        .where(and_(
            models.MembershipFee.due_date >= start_date,
            models.MembershipFee.due_date <= end_date
        ))
        .order_by(models.MembershipFee.due_date)
    )
    
    upcoming_fees = result.scalars().all()
    
    return {
        "period_days": days,
        "count": len(upcoming_fees),
        "fees": [
            {
                "id": str(f.id),
                "user_id": str(f.user_id),
                "amount": f.amount,
                "due_date": f.due_date.isoformat(),
                "days_remaining": (f.due_date.date() - datetime.utcnow().date()).days
            }
            for f in upcoming_fees
        ]
    }


@router.get("/{fee_id}", response_model=schemas.MembershipFee)
async def get_fee_details(
    fee_id: uuid.UUID,
    current_owner: models.GymOwner = Depends(get_current_gym_owner),
    session: AsyncSession = Depends(get_async_session)
):
    """Get detailed information about a specific fee record"""
    fee = await session.get(models.MembershipFee, fee_id)
    
    if not fee:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Fee record not found"
        )
    
    if fee.gym_id != current_owner.gym_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This fee belongs to a different gym"
        )
    
    return fee


@router.patch("/{fee_id}", response_model=schemas.MembershipFee)
async def update_fee_record(
    fee_id: uuid.UUID,
    updates: schemas.MembershipFeeUpdate,
    current_owner: models.GymOwner = Depends(get_current_gym_owner),
    session: AsyncSession = Depends(get_async_session)
):
    """
    Update a fee record
    
    Can update:
    - status (mark as PAID, CANCELLED)
    - payment_method
    - paid_date
    - receipt_number
    - notes
    """
    fee = await session.get(models.MembershipFee, fee_id)
    
    if not fee:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Fee record not found"
        )
    
    if fee.gym_id != current_owner.gym_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This fee belongs to a different gym"
        )
    
    # Update fields if provided
    if updates.status is not None:
        fee.status = updates.status
        if updates.status == "PAID" and not fee.paid_date:
            fee.paid_date = datetime.utcnow()
    
    if updates.payment_method is not None:
        fee.payment_method = updates.payment_method
    
    if updates.paid_date is not None:
        fee.paid_date = updates.paid_date
    
    if updates.receipt_number is not None:
        fee.receipt_number = updates.receipt_number
    
    if updates.notes is not None:
        fee.notes = updates.notes
    
    await session.commit()
    await session.refresh(fee)
    
    return fee


@router.delete("/{fee_id}")
async def delete_fee_record(
    fee_id: uuid.UUID,
    current_owner: models.GymOwner = Depends(get_current_gym_owner),
    session: AsyncSession = Depends(get_async_session)
):
    """
    Delete (cancel) a fee record
    
    Sets status to CANCELLED instead of actually deleting
    """
    fee = await session.get(models.MembershipFee, fee_id)
    
    if not fee:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Fee record not found"
        )
    
    if fee.gym_id != current_owner.gym_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This fee belongs to a different gym"
        )
    
    fee.status = "CANCELLED"
    await session.commit()
    
    return {"message": "Fee record cancelled", "fee_id": str(fee_id)}


@router.post("/bulk-create")
async def bulk_create_fees(
    bulk_data: schemas.BulkFeeCreate,
    current_owner: models.GymOwner = Depends(get_current_gym_owner),
    session: AsyncSession = Depends(get_async_session)
):
    """
    Create fee records for multiple members at once
    
    Useful for monthly fee generation
    
    - **user_ids**: List of member UUIDs
    - **amount**: Fee amount (same for all)
    - **due_date**: Due date (same for all)
    """
    created_fees = []
    errors = []
    
    gym = await session.get(models.Gym, current_owner.gym_id)
    
    for user_id in bulk_data.user_ids:
        try:
            user = await session.get(models.User, user_id)
            
            if not user or user.gym_id != current_owner.gym_id:
                errors.append({
                    "user_id": str(user_id),
                    "error": "Member not found or doesn't belong to your gym"
                })
                continue
            
            receipt_number = f"FEE-{datetime.utcnow().strftime('%Y%m%d')}-{uuid.uuid4().hex[:8].upper()}"
            
            new_fee = models.MembershipFee(
                user_id=user_id,
                gym_id=current_owner.gym_id,
                amount=bulk_data.amount,
                currency=gym.currency if gym else "INR",
                due_date=bulk_data.due_date,
                status="PENDING",
                receipt_number=receipt_number,
                created_by=current_owner.id
            )
            
            session.add(new_fee)
            created_fees.append(str(user_id))
            
        except Exception as e:
            errors.append({
                "user_id": str(user_id),
                "error": str(e)
            })
    
    await session.commit()
    
    return {
        "success_count": len(created_fees),
        "error_count": len(errors),
        "created_for_users": created_fees,
        "errors": errors
    }
