"""add dietary_restrictions to user

Revision ID: add_dietary_restrictions
Revises: fix_workout_log_name
Create Date: 2025-11-05 11:30:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = 'add_dietary_restrictions'
down_revision: Union[str, Sequence[str], None] = 'fix_workout_log_name'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add dietary_restrictions column to user table"""
    op.add_column('user', 
        sa.Column('dietary_restrictions', 
                  postgresql.JSON(), 
                  nullable=True,
                  server_default='[]')
    )


def downgrade() -> None:
    """Remove dietary_restrictions column"""
    op.drop_column('user', 'dietary_restrictions')