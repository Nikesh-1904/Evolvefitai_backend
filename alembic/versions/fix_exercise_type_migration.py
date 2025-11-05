"""add default to exercise_type

Revision ID: fix_exercise_type_null
Revises: add_user_created_at
Create Date: 2025-11-05 10:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'fix_exercise_type_null'
down_revision: Union[str, Sequence[str], None] = 'add_user_created_at'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema - Make exercise_type have a default value"""
    
    # Option 1: Make the column nullable (allows None)
    # op.alter_column('exercises', 'exercise_type', nullable=True)
    
    # Option 2: Add a default value (better approach)
    # First, update any existing NULL values
    op.execute("""
        UPDATE exercises 
        SET exercise_type = 'WEIGHT_BASED' 
        WHERE exercise_type IS NULL
    """)
    
    # Then add a server default for future inserts
    op.alter_column('exercises', 'exercise_type',
                    server_default='WEIGHT_BASED',
                    nullable=False)


def downgrade() -> None:
    """Downgrade schema"""
    # Remove the default
    op.alter_column('exercises', 'exercise_type',
                    server_default=None,
                    nullable=False)