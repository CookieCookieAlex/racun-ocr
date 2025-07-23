"""initial schema

Revision ID: bb33e7ca7f49
Revises: 3759b1366241
Create Date: 2025-07-20 14:16:39.087972

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from app.db.models import Base

# revision identifiers, used by Alembic.
revision: str = 'bb33e7ca7f49'
down_revision: Union[str, Sequence[str], None] = '3759b1366241'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None



def upgrade():
    bind = op.get_bind()
    Base.metadata.create_all(bind=bind)

def downgrade():
    bind = op.get_bind()
    Base.metadata.drop_all(bind=bind)
