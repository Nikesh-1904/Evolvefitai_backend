"""Fix all schema mismatches and add missing fields

Revision ID: fix_schema_v2
Revises: a6311b88e4ab
Create Date: 2025-11-02 12:00:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'fix_schema_v2'
down_revision: Union[str, Sequence[str], None] = 'a6311b88e4ab'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Apply schema fixes"""
    
    # ===== FIX 1: WorkoutPlan - Add missing fields =====
    print("Adding ai_generated and ai_model to workout_plans...")
    op.add_column('workout_plans', 
        sa.Column('ai_generated', sa.Boolean(), nullable=False, server_default='false')
    )
    op.add_column('workout_plans', 
        sa.Column('ai_model', sa.String(), nullable=True)
    )
    
    # Rename duration_minutes to estimated_duration
    print("Renaming duration_minutes to estimated_duration in workout_plans...")
    op.alter_column('workout_plans', 'duration_minutes',
        new_column_name='estimated_duration',
        nullable=True
    )
    
    # ===== FIX 2: WorkoutLog - Add workout_date field =====
    print("Adding workout_date to workout_logs...")
    # Set default to logged_at for existing records
    op.add_column('workout_logs',
        sa.Column('workout_date', sa.DateTime(timezone=True), nullable=True)
    )
    # Backfill existing data
    op.execute("UPDATE workout_logs SET workout_date = logged_at WHERE workout_date IS NULL")
    # Now make it non-nullable
    op.alter_column('workout_logs', 'workout_date', nullable=False)
    
    # ===== FIX 3: MealPlan - Add missing fields =====
    print("Adding ai_generated and ai_model to meal_plans...")
    op.add_column('meal_plans',
        sa.Column('ai_generated', sa.Boolean(), nullable=False, server_default='false')
    )
    op.add_column('meal_plans',
        sa.Column('ai_model', sa.String(), nullable=True)
    )
    
    # ===== FIX 4: ExerciseVideo - Add missing fields =====
    print("Adding youtube_url and duration to exercise_videos...")
    
    # Add video_id column if it doesn't exist (for backward compatibility)
    try:
        op.add_column('exercise_videos',
            sa.Column('video_id', sa.String(), nullable=True)
        )
    except Exception:
        print("video_id column already exists or error adding it")
    
    # First, add youtube_url as nullable
    op.add_column('exercise_videos',
        sa.Column('youtube_url', sa.String(), nullable=True)
    )
    # Backfill from video_id (construct URL) if video_id exists
    try:
        op.execute("""
            UPDATE exercise_videos 
            SET youtube_url = 'https://www.youtube.com/watch?v=' || video_id 
            WHERE youtube_url IS NULL AND video_id IS NOT NULL
        """)
    except Exception:
        # If video_id doesn't exist, just set a default
        op.execute("""
            UPDATE exercise_videos 
            SET youtube_url = 'https://www.youtube.com/watch?v=default'
            WHERE youtube_url IS NULL
        """)
    
    # Make it non-nullable
    op.alter_column('exercise_videos', 'youtube_url', nullable=False)
    
    # Add duration field
    op.add_column('exercise_videos',
        sa.Column('duration', sa.Integer(), nullable=True)
    )
    
    # ===== FIX 5: ExerciseTip - Restructure fields =====
    print("Restructuring exercise_tips table...")
    # Add new fields
    op.add_column('exercise_tips',
        sa.Column('title', sa.String(), nullable=True)
    )
    op.add_column('exercise_tips',
        sa.Column('content', sa.Text(), nullable=True)
    )
    op.add_column('exercise_tips',
        sa.Column('tip_type', sa.String(), nullable=True)
    )
    
    # Migrate data from 'tip' to 'content' and set generic title
    op.execute("""
        UPDATE exercise_tips 
        SET title = 'Exercise Tip',
            content = tip,
            tip_type = 'General'
        WHERE content IS NULL
    """)
    
    # Make title and content non-nullable
    op.alter_column('exercise_tips', 'title', nullable=False)
    op.alter_column('exercise_tips', 'content', nullable=False)
    
    # Drop old 'tip' column
    op.drop_column('exercise_tips', 'tip')
    
    # ===== FIX 6: User - Remove qr_code_id circular dependency =====
    print("Removing circular foreign key from user table...")
    # Drop the foreign key constraint first
    try:
        op.drop_constraint('fk_user_qr_code_id_user_qr_codes', 'user', type_='foreignkey')
    except Exception as e:
        print(f"Constraint might not exist: {e}")
    
    # Drop the column
    try:
        op.drop_column('user', 'qr_code_id')
    except Exception as e:
        print(f"Column might not exist: {e}")
    
    # ===== FIX 7: User - Fix notification_preferences default =====
    print("Fixing notification_preferences default...")
    op.alter_column('user', 'notification_preferences',
        server_default='{"email_enabled": true, "app_enabled": true, "fee_reminder_days": [7, 3, 1]}'
    )
    
    print("✅ Schema migration completed successfully!")


def downgrade() -> None:
    """Revert schema fixes"""
    
    # Revert in reverse order
    print("Reverting schema changes...")
    
    # Revert notification_preferences
    op.alter_column('user', 'notification_preferences', server_default=None)
    
    # Re-add qr_code_id (if needed for rollback)
    op.add_column('user',
        sa.Column('qr_code_id', postgresql.UUID(as_uuid=True), nullable=True)
    )
    op.create_foreign_key(
        'fk_user_qr_code_id_user_qr_codes',
        'user', 'user_qr_codes',
        ['qr_code_id'], ['id']
    )
    
    # Revert ExerciseTip
    op.add_column('exercise_tips', sa.Column('tip', sa.Text(), nullable=True))
    op.execute("UPDATE exercise_tips SET tip = content WHERE tip IS NULL")
    op.alter_column('exercise_tips', 'tip', nullable=False)
    op.drop_column('exercise_tips', 'tip_type')
    op.drop_column('exercise_tips', 'content')
    op.drop_column('exercise_tips', 'title')
    
    # Revert ExerciseVideo
    op.drop_column('exercise_videos', 'duration')
    op.drop_column('exercise_videos', 'youtube_url')
    
    # Revert MealPlan
    op.drop_column('meal_plans', 'ai_model')
    op.drop_column('meal_plans', 'ai_generated')
    
    # Revert WorkoutLog
    op.drop_column('workout_logs', 'workout_date')
    
    # Revert WorkoutPlan
    op.alter_column('workout_plans', 'estimated_duration',
        new_column_name='duration_minutes'
    )
    op.drop_column('workout_plans', 'ai_model')
    op.drop_column('workout_plans', 'ai_generated')
    
    print("✅ Schema downgrade completed!")