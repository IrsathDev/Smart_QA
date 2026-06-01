"""Initial Smart Q&A V3 schema.

Revision ID: 20260601_01
Revises:
"""
from alembic import op

from backend.app.database import Base
from backend.app import models  # noqa: F401

revision = "20260601_01"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    Base.metadata.create_all(bind=op.get_bind())


def downgrade() -> None:
    Base.metadata.drop_all(bind=op.get_bind())
