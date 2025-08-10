"""exec tables

Revision ID: 96cdf90bd77b
Revises: 5d67705bdcb2
Create Date: 2025-08-07 21:56:08.445195

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '96cdf90bd77b'
down_revision: Union[str, Sequence[str], None] = '5d67705bdcb2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
