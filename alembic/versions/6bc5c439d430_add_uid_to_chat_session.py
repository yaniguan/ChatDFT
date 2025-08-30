"""add uid to chat_session

Revision ID: 6bc5c439d430
Revises: 33c035e33090
Create Date: 2025-08-14 13:33:58.502245

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '6bc5c439d430'
down_revision: Union[str, Sequence[str], None] = '33c035e33090'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    op.add_column('chat_session', sa.Column('uid', sa.String(), nullable=True))
    # 任选一个扩展：
    op.execute('CREATE EXTENSION IF NOT EXISTS pgcrypto;')
    op.execute("UPDATE chat_session SET uid = gen_random_uuid()::text WHERE uid IS NULL;")
    # 如果你用 uuid-ossp：
    # op.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp";')
    # op.execute("UPDATE chat_session SET uid = uuid_generate_v4()::text WHERE uid IS NULL;")
    op.create_unique_constraint('uq_chat_session_uid', 'chat_session', ['uid'])
    op.alter_column('chat_session', 'uid', nullable=False)


def downgrade() -> None:
    """Downgrade schema."""
    pass
