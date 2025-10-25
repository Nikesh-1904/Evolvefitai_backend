"""merge heads

Revision ID: cc17bc21059c
Revises: add_business_models, migrate_gyms_id_to_uuid
Create Date: 2025-10-26 00:44:43.296740

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'cc17bc21059c'
down_revision = ('add_business_models', 'migrate_gyms_id_to_uuid')
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass