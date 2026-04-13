"""add orchestration tables (closed-loop engine)

Adds three tables for the closed-loop orchestrator:
  - orchestration_run     one closed-loop run per row
  - orchestration_step    one iteration per row (FK → orchestration_run)
  - reward_signal         one reward observation per row

Until this migration runs the tables are created opportunistically by
``Base.metadata.create_all`` in ``server.main.lifespan``; this gives prod
a real, downgradable schema change instead.

Revision ID: a7f2c8e91d04
Revises: 6bc5c439d430
Create Date: 2026-04-13 12:00:00
"""
from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a7f2c8e91d04"
down_revision: Union[str, Sequence[str], None] = "6bc5c439d430"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "orchestration_run",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("session_id", sa.Integer(),
                  sa.ForeignKey("chat_session.id", ondelete="CASCADE"),
                  nullable=True),
        sa.Column("status", sa.String(), server_default="running"),
        sa.Column("stop_reason", sa.String(), nullable=True),
        sa.Column("iteration", sa.Integer(), server_default="0"),
        sa.Column("confidence", sa.Float(), server_default="0.5"),
        sa.Column("reward_ema", sa.Float(), server_default="0.0"),
        sa.Column("config", sa.JSON(), nullable=True),
        sa.Column("intent", sa.JSON(), nullable=True),
        sa.Column("final_state", sa.JSON(), nullable=True),
        sa.Column("started_at", sa.DateTime(), server_default=sa.func.now()),
        sa.Column("ended_at", sa.DateTime(), nullable=True),
    )
    op.create_index("idx_orchrun_session", "orchestration_run", ["session_id"])
    op.create_index("idx_orchrun_status",  "orchestration_run", ["status"])

    op.create_table(
        "orchestration_step",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("run_id", sa.Integer(),
                  sa.ForeignKey("orchestration_run.id", ondelete="CASCADE"),
                  nullable=False),
        sa.Column("iteration", sa.Integer(), nullable=False),
        sa.Column("executed_task_id", sa.Integer(), nullable=True),
        sa.Column("executed_agent",   sa.String(),  nullable=True),
        sa.Column("success",          sa.Boolean(), nullable=True),
        sa.Column("reward",           sa.Float(),   nullable=True),
        sa.Column("confidence_after", sa.Float(),   nullable=True),
        sa.Column("proposed_actions", sa.JSON(), nullable=True),
        sa.Column("rejected_actions", sa.JSON(), nullable=True),
        sa.Column("notes",            sa.Text(), nullable=True),
        sa.Column("started_at",       sa.DateTime(), server_default=sa.func.now()),
        sa.Column("ended_at",         sa.DateTime(), nullable=True),
        sa.UniqueConstraint("run_id", "iteration", name="uq_orchstep_run_iter"),
    )
    op.create_index("idx_orchstep_run", "orchestration_step", ["run_id"])

    op.create_table(
        "reward_signal",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("run_id", sa.Integer(),
                  sa.ForeignKey("orchestration_run.id", ondelete="CASCADE"),
                  nullable=True),
        sa.Column("session_id", sa.Integer(),
                  sa.ForeignKey("chat_session.id", ondelete="CASCADE"),
                  nullable=True),
        sa.Column("iteration",        sa.Integer(), nullable=True),
        sa.Column("species",          sa.String(),  nullable=True),
        sa.Column("surface",          sa.String(),  nullable=True),
        sa.Column("reaction_type",    sa.String(),  nullable=True),
        sa.Column("predicted_trend",  sa.String(),  nullable=True),
        sa.Column("dft_value",        sa.Float(),   nullable=True),
        sa.Column("reward",           sa.Float(),   nullable=False),
        sa.Column("converged",        sa.Boolean(), server_default=sa.true()),
        sa.Column("details",          sa.Text(),    nullable=True),
        sa.Column("created_at",       sa.DateTime(), server_default=sa.func.now()),
    )
    op.create_index("idx_reward_run",     "reward_signal", ["run_id"])
    op.create_index("idx_reward_session", "reward_signal", ["session_id"])


def downgrade() -> None:
    op.drop_index("idx_reward_session", table_name="reward_signal")
    op.drop_index("idx_reward_run",     table_name="reward_signal")
    op.drop_table("reward_signal")

    op.drop_index("idx_orchstep_run", table_name="orchestration_step")
    op.drop_table("orchestration_step")

    op.drop_index("idx_orchrun_status",  table_name="orchestration_run")
    op.drop_index("idx_orchrun_session", table_name="orchestration_run")
    op.drop_table("orchestration_run")
