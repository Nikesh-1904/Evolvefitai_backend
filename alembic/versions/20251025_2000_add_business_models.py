"""add business models

Revision ID: add_business_models
Revises: (your previous migration ID - find it in alembic/versions/)
Create Date: 2025-10-25 20:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID
import uuid

# revision identifiers, used by Alembic.
revision = 'add_business_models'
down_revision = 'a4d3c81bbd20'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add business-related tables"""
    
    # 1. Gym Owners table
    op.create_table(
        'gym_owners',
        sa.Column('id', UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('email', sa.String(), nullable=False, unique=True, index=True),
        sa.Column('hashed_password', sa.String(), nullable=False),
        sa.Column('full_name', sa.String(), nullable=False),
        sa.Column('phone_number', sa.String(), nullable=True),
        sa.Column('gym_id', UUID(as_uuid=True), sa.ForeignKey('gyms.id'), nullable=False),
        sa.Column('is_active', sa.Boolean(), default=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('last_login', sa.DateTime(timezone=True), nullable=True),
    )
    
    # 2. User QR Codes table
    op.create_table(
        'user_qr_codes',
        sa.Column('id', UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('user_id', UUID(as_uuid=True), sa.ForeignKey('users.id'), nullable=False, unique=True),
        sa.Column('gym_id', UUID(as_uuid=True), sa.ForeignKey('gyms.id'), nullable=False),
        sa.Column('qr_code_data', sa.Text(), nullable=False),
        sa.Column('qr_code_image_base64', sa.Text(), nullable=False),
        sa.Column('is_active', sa.Boolean(), default=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=True),
    )
    
    # 3. Membership Fees table
    op.create_table(
        'membership_fees',
        sa.Column('id', UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('user_id', UUID(as_uuid=True), sa.ForeignKey('users.id'), nullable=False),
        sa.Column('gym_id', UUID(as_uuid=True), sa.ForeignKey('gyms.id'), nullable=False),
        sa.Column('amount', sa.Float(), nullable=False),
        sa.Column('currency', sa.String(), default='INR'),
        sa.Column('payment_date', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('due_date', sa.DateTime(timezone=True), nullable=False),
        sa.Column('paid_date', sa.DateTime(timezone=True), nullable=True),
        sa.Column('status', sa.String(), default='PENDING'),
        sa.Column('payment_method', sa.String(), nullable=True),
        sa.Column('receipt_number', sa.String(), unique=True, nullable=True),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.Column('created_by', UUID(as_uuid=True), sa.ForeignKey('gym_owners.id'), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), onupdate=sa.func.now(), nullable=True),
    )
    
    # 4. Gym Attendance table
    op.create_table(
        'gym_attendance',
        sa.Column('id', UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('user_id', UUID(as_uuid=True), sa.ForeignKey('users.id'), nullable=False),
        sa.Column('gym_id', UUID(as_uuid=True), sa.ForeignKey('gyms.id'), nullable=False),
        sa.Column('check_in_time', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('check_out_time', sa.DateTime(timezone=True), nullable=True),
        sa.Column('duration_minutes', sa.Integer(), nullable=True),
        sa.Column('qr_code_used', sa.String(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    
    # 5. Notifications table
    op.create_table(
        'notifications',
        sa.Column('id', UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('user_id', UUID(as_uuid=True), sa.ForeignKey('users.id'), nullable=False),
        sa.Column('gym_id', UUID(as_uuid=True), sa.ForeignKey('gyms.id'), nullable=False),
        sa.Column('notification_type', sa.String(), nullable=False),
        sa.Column('title', sa.String(), nullable=False),
        sa.Column('message', sa.Text(), nullable=False),
        sa.Column('is_read', sa.Boolean(), default=False),
        sa.Column('sent_via_email', sa.Boolean(), default=False),
        sa.Column('sent_via_app', sa.Boolean(), default=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('read_at', sa.DateTime(timezone=True), nullable=True),
    )
    
    # 6. Member Performance table
    op.create_table(
        'member_performance',
        sa.Column('id', UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('user_id', UUID(as_uuid=True), sa.ForeignKey('users.id'), nullable=False),
        sa.Column('gym_id', UUID(as_uuid=True), sa.ForeignKey('gyms.id'), nullable=False),
        sa.Column('analysis_date', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('total_workouts', sa.Integer(), default=0),
        sa.Column('avg_workout_duration', sa.Float(), default=0.0),
        sa.Column('consistency_score', sa.Float(), default=0.0),
        sa.Column('performance_trend', sa.String(), default='STABLE'),
        sa.Column('weak_areas', sa.JSON(), default=list),
        sa.Column('suggestions', sa.JSON(), default=list),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    
    # 7. Add new columns to existing users table
    op.add_column('users', sa.Column('qr_code_id', UUID(as_uuid=True), sa.ForeignKey('user_qr_codes.id'), nullable=True))
    op.add_column('users', sa.Column('membership_status', sa.String(), default='ACTIVE'))
    op.add_column('users', sa.Column('membership_expiry', sa.DateTime(timezone=True), nullable=True))
    op.add_column('users', sa.Column('notification_preferences', sa.JSON(), nullable=True))
    
    # 8. Add new columns to existing gyms table
    op.add_column('gyms', sa.Column('monthly_fee', sa.Float(), nullable=True))
    op.add_column('gyms', sa.Column('currency', sa.String(), default='INR'))
    op.add_column('gyms', sa.Column('fee_due_day', sa.Integer(), default=1))
    
    # 9. Create indexes for better query performance
    op.create_index('idx_membership_fees_status', 'membership_fees', ['status'])
    op.create_index('idx_membership_fees_due_date', 'membership_fees', ['due_date'])
    op.create_index('idx_gym_attendance_check_in', 'gym_attendance', ['gym_id', 'check_in_time'])
    op.create_index('idx_notifications_user_unread', 'notifications', ['user_id', 'is_read'])


def downgrade() -> None:
    """Remove business-related tables"""
    
    # Drop indexes
    op.drop_index('idx_notifications_user_unread')
    op.drop_index('idx_gym_attendance_check_in')
    op.drop_index('idx_membership_fees_due_date')
    op.drop_index('idx_membership_fees_status')
    
    # Drop columns from existing tables
    op.drop_column('gyms', 'fee_due_day')
    op.drop_column('gyms', 'currency')
    op.drop_column('gyms', 'monthly_fee')
    op.drop_column('users', 'notification_preferences')
    op.drop_column('users', 'membership_expiry')
    op.drop_column('users', 'membership_status')
    op.drop_column('users', 'qr_code_id')
    
    # Drop new tables
    op.drop_table('member_performance')
    op.drop_table('notifications')
    op.drop_table('gym_attendance')
    op.drop_table('membership_fees')
    op.drop_table('user_qr_codes')
    op.drop_table('gym_owners')
