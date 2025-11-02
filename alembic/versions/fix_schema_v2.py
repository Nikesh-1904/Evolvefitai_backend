"""Fix all schema mismatches and add missing fields

Revision ID: fix_schema_v2
Revises: a6311b88e4ab
Create Date: 2024-01-01 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.exc import ProgrammingError, OperationalError


# revision identifiers, used by Alembic.
revision = 'fix_schema_v2'
down_revision = 'a6311b88e4ab'
branch_labels = None
depends_on = None


def column_exists(table_name, column_name):
    """Check if a column exists in a table"""
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = [col['name'] for col in inspector.get_columns(table_name)]
    return column_name in columns


def upgrade():
    # Fix workout_plans table
    print("Adding ai_generated and ai_model to workout_plans...")
    if not column_exists('workout_plans', 'ai_generated'):
        op.add_column('workout_plans', sa.Column('ai_generated', sa.Boolean(), nullable=True))
    
    if not column_exists('workout_plans', 'ai_model'):
        op.add_column('workout_plans', sa.Column('ai_model', sa.String(), nullable=True))
    
    # Rename duration_minutes to estimated_duration if needed
    if column_exists('workout_plans', 'duration_minutes') and not column_exists('workout_plans', 'estimated_duration'):
        print("Renaming duration_minutes to estimated_duration in workout_plans...")
        op.alter_column('workout_plans', 'duration_minutes', new_column_name='estimated_duration')
    elif not column_exists('workout_plans', 'estimated_duration'):
        op.add_column('workout_plans', sa.Column('estimated_duration', sa.Integer(), nullable=True))
    
    # Fix workout_logs table
    print("Adding workout_date to workout_logs...")
    if not column_exists('workout_logs', 'workout_date'):
        op.add_column('workout_logs', sa.Column('workout_date', sa.Date(), nullable=True))
    
    # Fix meal_plans table
    print("Adding ai_generated and ai_model to meal_plans...")
    if not column_exists('meal_plans', 'ai_generated'):
        op.add_column('meal_plans', sa.Column('ai_generated', sa.Boolean(), nullable=True))
    
    if not column_exists('meal_plans', 'ai_model'):
        op.add_column('meal_plans', sa.Column('ai_model', sa.String(), nullable=True))
    
    # Fix exercise_videos table
    print("Adding youtube_url and duration to exercise_videos...")
    if not column_exists('exercise_videos', 'youtube_url'):
        op.add_column('exercise_videos', sa.Column('youtube_url', sa.String(), nullable=True))
    
    if not column_exists('exercise_videos', 'duration'):
        op.add_column('exercise_videos', sa.Column('duration', sa.Integer(), nullable=True))
    
    # Check for video_id column (this was causing the error)
    if not column_exists('exercise_videos', 'video_id'):
        print("Adding video_id to exercise_videos...")
        op.add_column('exercise_videos', sa.Column('video_id', sa.String(), nullable=True))
    else:
        print("video_id column already exists, skipping...")


def downgrade():
    # Remove added columns
    if column_exists('exercise_videos', 'video_id'):
        op.drop_column('exercise_videos', 'video_id')
    
    if column_exists('exercise_videos', 'duration'):
        op.drop_column('exercise_videos', 'duration')
    
    if column_exists('exercise_videos', 'youtube_url'):
        op.drop_column('exercise_videos', 'youtube_url')
    
    if column_exists('meal_plans', 'ai_model'):
        op.drop_column('meal_plans', 'ai_model')
    
    if column_exists('meal_plans', 'ai_generated'):
        op.drop_column('meal_plans', 'ai_generated')
    
    if column_exists('workout_logs', 'workout_date'):
        op.drop_column('workout_logs', 'workout_date')
    
    # Rename back if it was renamed
    if column_exists('workout_plans', 'estimated_duration') and not column_exists('workout_plans', 'duration_minutes'):
        op.alter_column('workout_plans', 'estimated_duration', new_column_name='duration_minutes')
    
    if column_exists('workout_plans', 'ai_model'):
        op.drop_column('workout_plans', 'ai_model')
    
    if column_exists('workout_plans', 'ai_generated'):
        op.drop_column('workout_plans', 'ai_generated')