# app/shared/api/gyms.py
"""Gym membership endpoints used by the client app (join by code, leave gym)."""

from datetime import datetime
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_async_session
from app.core.auth import current_active_user
from app import schemas, models

router = APIRouter()


@router.post("/{gym_id}/leave")
async def leave_gym(
    gym_id: uuid.UUID,
    session: AsyncSession = Depends(get_async_session),
    current_user: models.User = Depends(current_active_user),
):
    """Leave the current gym.

    Removes the user's gym association and marks membership as EXPIRED so they
    can join a different gym.
    """
    if current_user.gym_id != gym_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You are not a member of this gym",
        )

    gym_obj = await session.get(models.Gym, gym_id)
    gym_name = gym_obj.name if gym_obj else "gym"

    current_user.gym_id = None  # type: ignore
    current_user.membership_status = "EXPIRED"

    await session.commit()

    return {
        "message": f"Successfully left {gym_name}",
        "former_gym_id": str(gym_id),
    }


@router.post("/join-by-code", response_model=schemas.MessageResponse)
async def join_gym_by_code(
    request_data: schemas.JoinByCodeRequest,
    session: AsyncSession = Depends(get_async_session),
    current_user: models.User = Depends(current_active_user),
):
    """Join a gym using its unique code."""
    gym_code = request_data.gym_code.strip().upper()

    if not gym_code:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Gym code cannot be empty.",
        )

    gym_query = select(models.Gym).where(func.upper(models.Gym.gym_code) == gym_code)
    gym_result = await session.execute(gym_query)
    gym = gym_result.scalar_one_or_none()

    if not gym:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Invalid gym code.",
        )

    if current_user.gym_id == gym.id:
        return schemas.MessageResponse(
            message=f"You are already a member of {gym.name}.",
            success=True,
        )

    current_user.gym_id = gym.id
    current_user.last_gym_change = datetime.utcnow()

    await session.commit()
    await session.refresh(current_user)

    return schemas.MessageResponse(
        message=f"Successfully joined {gym.name}!",
        success=True,
    )
