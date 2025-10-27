"""Merge divergent branches after fixing dependencies

Revision ID: d7b36e5ec791
Revises: add_business_models, cc17bc21059c
Create Date: 2025-10-27 14:27:51.804523

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'd7b36e5ec791'
down_revision = ('add_business_models', 'cc17bc21059c')
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass