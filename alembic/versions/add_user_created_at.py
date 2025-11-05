"""add created_at to user table

Revision ID: add_user_created_at
Revises: 3ccd83744279
Create Date: 2025-11-05 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'add_user_created_at'
down_revision: Union[str, Sequence[str], None] = '3ccd83744279'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Add created_at column to user table if it doesn't exist
    # Using server_default for existing rows
    op.add_column('user', 
        sa.Column('created_at', 
                  sa.DateTime(timezone=True), 
                  server_default=sa.text('(CURRENT_TIMESTAMP)'), 
                  nullable=False)
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('user', 'created_at')