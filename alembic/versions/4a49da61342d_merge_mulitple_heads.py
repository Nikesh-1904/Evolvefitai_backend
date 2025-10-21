"""Merge mulitple heads

Revision ID: 4a49da61342d
Revises: 8821bd55a06e, gym_booking_additions
Create Date: 2025-10-21 18:17:25.665457

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '4a49da61342d'
down_revision = ('8821bd55a06e', 'gym_booking_additions')
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass