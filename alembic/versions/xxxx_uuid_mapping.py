"""Migrate gyms.id from Integer to UUID

Revision ID: migrate_gyms_id_to_uuid
Revises: add_business_models
Create Date: 2025-10-26 00:00:00.000000

"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

revision = 'migrate_gyms_id_to_uuid'
down_revision = 'add_business_models'
branch_labels = None
depends_on = None

def upgrade():
    # 1. Add new uuid_id column to gyms
    op.add_column('gyms', sa.Column('uuid_id', UUID(as_uuid=True), nullable=True))

    # 2. Populate uuid_id with generated UUIDs
    op.execute('UPDATE gyms SET uuid_id = gen_random_uuid() WHERE uuid_id IS NULL')

    # 3. Add new uuid_gym_id columns to dependent tables

    dependent_tables = [
        'gym_owners',
        'user_qr_codes',
        'membership_fees',
        'gym_attendance',
        'notifications',
        'member_performance'
    ]

    for table in dependent_tables:
        op.add_column(table, sa.Column('uuid_gym_id', UUID(as_uuid=True), nullable=True))

    # 4. Copy UUID values to dependent tables
    op.execute("""
        UPDATE gym_owners g
        SET uuid_gym_id = gy.uuid_id
        FROM gyms gy
        WHERE g.gym_id = gy.id
    """)
    op.execute("""
        UPDATE user_qr_codes u
        SET uuid_gym_id = gy.uuid_id
        FROM gyms gy
        WHERE u.gym_id = gy.id
    """)
    op.execute("""
        UPDATE membership_fees m
        SET uuid_gym_id = gy.uuid_id
        FROM gyms gy
        WHERE m.gym_id = gy.id
    """)
    op.execute("""
        UPDATE gym_attendance a
        SET uuid_gym_id = gy.uuid_id
        FROM gyms gy
        WHERE a.gym_id = gy.id
    """)
    op.execute("""
        UPDATE notifications n
        SET uuid_gym_id = gy.uuid_id
        FROM gyms gy
        WHERE n.gym_id = gy.id
    """)
    op.execute("""
        UPDATE member_performance p
        SET uuid_gym_id = gy.uuid_id
        FROM gyms gy
        WHERE p.gym_id = gy.id
    """)

    # 5. Drop foreign key constraints referencing integer gym_id columns
    op.drop_constraint('gym_owners_gym_id_fkey', 'gym_owners', type_='foreignkey')
    op.drop_constraint('user_qr_codes_gym_id_fkey', 'user_qr_codes', type_='foreignkey')
    op.drop_constraint('membership_fees_gym_id_fkey', 'membership_fees', type_='foreignkey')
    op.drop_constraint('gym_attendance_gym_id_fkey', 'gym_attendance', type_='foreignkey')
    op.drop_constraint('notifications_gym_id_fkey', 'notifications', type_='foreignkey')
    op.drop_constraint('member_performance_gym_id_fkey', 'member_performance', type_='foreignkey')

    # 6. Drop old integer gym_id columns
    for table in dependent_tables:
        op.drop_column(table, 'gym_id')

    # 7. Rename new uuid_gym_id columns to gym_id and set non-nullable
    for table in dependent_tables:
        op.alter_column(table, 'uuid_gym_id', new_column_name='gym_id', nullable=False)

    # 8. Change primary key on gyms table
    op.drop_constraint('gyms_pkey', 'gyms', type_='primary')
    op.drop_column('gyms', 'id')
    op.alter_column('gyms', 'uuid_id', new_column_name='id', nullable=False)
    op.create_primary_key('gyms_pkey', 'gyms', ['id'])

    # 9. Recreate foreign keys referencing gyms.id (UUID)
    op.create_foreign_key('gym_owners_gym_id_fkey', 'gym_owners', 'gyms', ['gym_id'], ['id'], ondelete='CASCADE')
    op.create_foreign_key('user_qr_codes_gym_id_fkey', 'user_qr_codes', 'gyms', ['gym_id'], ['id'], ondelete='CASCADE')
    op.create_foreign_key('membership_fees_gym_id_fkey', 'membership_fees', 'gyms', ['gym_id'], ['id'], ondelete='CASCADE')
    op.create_foreign_key('gym_attendance_gym_id_fkey', 'gym_attendance', 'gyms', ['gym_id'], ['id'], ondelete='CASCADE')
    op.create_foreign_key('notifications_gym_id_fkey', 'notifications', 'gyms', ['gym_id'], ['id'], ondelete='CASCADE')
    op.create_foreign_key('member_performance_gym_id_fkey', 'member_performance', 'gyms', ['gym_id'], ['id'], ondelete='CASCADE')

def downgrade():
    # Reverse update left for safety and clarity; implement if rollback needed
    pass
