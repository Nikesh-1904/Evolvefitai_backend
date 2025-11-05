"""make workout_logs name nullable

Revision ID: fix_workout_log_name
Revises: fix_exercise_type_null
Create Date: 2025-11-05 11:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'fix_workout_log_name'
down_revision: Union[str, Sequence[str], None] = 'fix_exercise_type_null'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Make workout_logs.name nullable and set default for existing NULL values"""
    
    # First, update any existing NULL values with a default name
    op.execute("""
        UPDATE workout_logs 
        SET name = COALESCE(
            (SELECT wp.name FROM workout_plans wp WHERE wp.id = workout_logs.workout_plan_id),
            'Workout - ' || TO_CHAR(workout_date, 'Month DD, YYYY')
        )
        WHERE name IS NULL
    """)
    
    # Then make the column nullable
    op.alter_column('workout_logs', 'name',
                    existing_type=sa.String(),
                    nullable=True)


def downgrade() -> None:
    """Revert workout_logs.name to not nullable"""
    
    # First set any NULL values to a default before making it NOT NULL again
    op.execute("""
        UPDATE workout_logs 
        SET name = 'Unnamed Workout' 
        WHERE name IS NULL
    """)
    
    # Then make the column NOT nullable
    op.alter_column('workout_logs', 'name',
                    existing_type=sa.String(),
                    nullable=False)