# server/db.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from datetime import datetime
import uuid


def uid():
    return str(uuid.uuid4())

from typing import AsyncGenerator
from sqlalchemy import (
    Column, Integer, String, Text, DateTime, ForeignKey, JSON, Float,
    UniqueConstraint, Boolean, Index
)
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base, relationship

# -----------------------------------------------------------------------------
# Engine / Session
# -----------------------------------------------------------------------------
# 例：postgresql+asyncpg://user:pass@host:5432/chatdft
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+asyncpg://postgres:password@localhost:5432/chatdft"
)

# echo 可通过环境变量打开：SQLALCHEMY_ECHO=1
ECHO = bool(int(os.environ.get("SQLALCHEMY_ECHO", "0")))
engine = create_async_engine(
    DATABASE_URL,
    future=True,
    echo=False,           # 需要可开 True
    pool_pre_ping=True,
)
AsyncSessionLocal = sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

# 用异步会话工厂；名称仍叫 SessionLocal，避免全项目替换
SessionLocal = async_sessionmaker(
    bind=engine,
    expire_on_commit=False,
    class_=AsyncSession,
)
Base = declarative_base()

# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------
class ChatSession(Base):
    __tablename__ = "chat_session"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    uid         = Column(String, unique=True, default=uid)
    name        = Column(String, nullable=False)
    user_id     = Column(Integer)
    project     = Column(String)
    tags        = Column(JSON)
    description = Column(Text)
    status      = Column(String, default="active")   # active / archived / ...
    pinned      = Column(Boolean, default=False)

    created_at  = Column(DateTime, default=datetime.utcnow)
    updated_at  = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    messages = relationship(
        "ChatMessage",
        back_populates="session",
        cascade="all, delete-orphan",     # 如只做软删除，可注释掉
        passive_deletes=True
    )
class ChatMessage(Base):
    __tablename__ = "chat_message"

    id               = Column(Integer, primary_key=True, autoincrement=True)
    session_id       = Column(Integer, ForeignKey("chat_session.id", ondelete="CASCADE"), index=True)
    role             = Column(String, nullable=False, index=True)  # user / assistant / system
    content          = Column(Text, nullable=False)
    msg_type         = Column(String)
    intent_stage     = Column(String)
    intent_area      = Column(String)
    specific_intent  = Column(String)
    confidence       = Column(Float)
    llm_model        = Column(String)
    parent_id        = Column(Integer, ForeignKey("chat_message.id"), nullable=True)
    attachments      = Column(JSON)
    references       = Column(JSON)
    feedback         = Column(Text)
    token_usage      = Column(Integer)
    duration         = Column(Float)

    created_at       = Column(DateTime, default=datetime.utcnow)
    updated_at       = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    session = relationship("ChatSession", back_populates="messages")

    __table_args__ = (
        Index('idx_chatmessage_session_id', 'session_id'),
        Index('idx_chatmessage_role', 'role'),
    )

class Knowledge(Base):
    __tablename__ = "knowledge"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    title       = Column(String)
    content     = Column(Text)
    source_type = Column(String)   # "arxiv", "wiki", "manual", "web", etc.
    source_id   = Column(String)   # e.g. arxiv_id / wiki_id / crossref key
    url         = Column(String)
    doi         = Column(String, index=True)  # 稳定回源/去重
    embedding   = Column(JSON)       # 可用于向量检索（或迁移到 pgvector）
    tags        = Column(String)
    created_at  = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint('source_type', 'source_id', name='_knowledge_src_uc'),
        UniqueConstraint('doi', name='_knowledge_doi_uc'),
    )

class Hypothesis(Base):
    __tablename__ = "hypothesis"

    id             = Column(Integer, primary_key=True, autoincrement=True)
    session_id     = Column(Integer, ForeignKey("chat_session.id", ondelete="SET NULL"))
    message_id     = Column(Integer, ForeignKey("chat_message.id", ondelete="SET NULL"))
    intent_stage   = Column(String)
    intent_area    = Column(String)
    hypothesis     = Column(Text)
    confidence     = Column(Float)
    agent          = Column(String)
    feedback       = Column(Text)
    tags           = Column(JSON)
    created_at     = Column(DateTime, default=datetime.utcnow)

class IntentPhrase(Base):
    __tablename__ = "intent_phrase"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    intent_stage  = Column(String)
    intent_area   = Column(String)
    specific_task = Column(String)
    phrase        = Column(Text, nullable=False)
    confidence    = Column(Float, default=1.0)
    source        = Column(String, default="user")
    lang          = Column(String, default="en")
    remark        = Column(Text)
    created_at    = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint('intent_stage', 'intent_area', 'specific_task', 'phrase', name='_intent_phrase_uc'),
    )

class WorkflowTask(Base):
    __tablename__ = "workflow_task"

    id             = Column(Integer, primary_key=True, autoincrement=True)
    session_id     = Column(Integer, ForeignKey("chat_session.id", ondelete="CASCADE"), index=True)
    message_id     = Column(Integer, ForeignKey("chat_message.id", ondelete="SET NULL"))
    parent_task_id = Column(Integer, ForeignKey("workflow_task.id", ondelete="SET NULL"), nullable=True)

    step_order     = Column(Integer)
    name           = Column(String)
    description    = Column(Text)
    agent          = Column(String)   # e.g. "run_dft" / "post_analysis"
    engine         = Column(String)   # e.g. "VASP"
    status         = Column(String, default="idle")

    input_data     = Column(JSON)
    output_data    = Column(JSON)
    error_msg      = Column(Text)
    run_time       = Column(Float)

    created_at     = Column(DateTime, default=datetime.utcnow)
    updated_at     = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index('idx_workflowtask_session_id', 'session_id'),
        Index('idx_workflowtask_name', 'name'),
        Index('idx_workflowtask_agent', 'agent'),
        Index('idx_workflowtask_status', 'status'),
    )

# ---------------- Execution Layer Structures ----------------
class BulkStructure(Base):
    __tablename__ = "bulk_structure"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    formula       = Column(String)
    structure_data= Column(JSON)

class SlabStructure(Base):
    __tablename__ = "slab_structure"

    id                  = Column(Integer, primary_key=True, autoincrement=True)
    bulk_id             = Column(Integer, ForeignKey("bulk_structure.id", ondelete="CASCADE"))

    miller_index        = Column(String, nullable=False)  # "1 1 1"
    supercell_size      = Column(String)                  # "4x4x1"
    layers              = Column(Integer)
    fixed_layers        = Column(Integer)
    vacuum_thickness    = Column(Float)                   # Å
    termination         = Column(String)                  # "O-terminated"

    shift               = Column(Float)
    symmetry_reduction  = Column(Boolean, default=False)
    is_symmetric_slab   = Column(Boolean, default=True)
    min_slab_size       = Column(Float)
    min_vacuum_size     = Column(Float)

    slab_data           = Column(JSON)
    cif_path            = Column(String)
    poscar_path         = Column(String)

    tags                = Column(JSON)
    created_at          = Column(DateTime, default=datetime.utcnow)

class AdsorptionStructure(Base):
    __tablename__ = "adsorption_structure"

    id                    = Column(Integer, primary_key=True, autoincrement=True)
    slab_id               = Column(Integer, ForeignKey("slab_structure.id", ondelete="CASCADE"))
    adsorbate_name        = Column(String)        # e.g. "H2O"
    adsorbate_formula     = Column(String)
    adsorption_site       = Column(String)        # "top", "bridge", "hollow"
    site_coordinates      = Column(JSON)          # 分数坐标
    coverage              = Column(Float)         # 覆盖率
    orientation           = Column(JSON)          # 方向信息
    height_above_surface  = Column(Float)         # Å

    is_relaxed            = Column(Boolean, default=False)
    adsorption_energy     = Column(Float)
    cif_path              = Column(String)
    poscar_path           = Column(String)

    tags                  = Column(JSON)
    created_at            = Column(DateTime, default=datetime.utcnow)

class ModificationStructure(Base):
    __tablename__ = "modification_structure"

    id                 = Column(Integer, primary_key=True, autoincrement=True)
    parent_type        = Column(String)     # "bulk" | "slab" | "adsorption"
    parent_id          = Column(Integer)    # 对应 parent 的 ID

    modification_type  = Column(String)     # "doping" | "vacancy" | "strain" | ...
    parameters         = Column(JSON)
    modified_data      = Column(JSON)
    cif_path           = Column(String)
    poscar_path        = Column(String)

    created_at         = Column(DateTime, default=datetime.utcnow)

# ---------------- Calculation / Scheduling / Post-analysis ----------------
class CalculationParameter(Base):
    __tablename__ = "calculation_parameter"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    structure_type  = Column(String)            # "bulk" | "slab" | "adsorption" | "modification"
    structure_id    = Column(Integer)           # 对应结构 ID
    engine          = Column(String)            # "VASP" | "CP2K" | ...
    task_type       = Column(String)            # "relax" | "dos" | "neb" | "bader" | ...
    incar_settings  = Column(JSON)              # 对 VASP
    input_files     = Column(JSON)              # 其他引擎输入
    explanation     = Column(Text)              # LLM 解释
    suggestions     = Column(JSON)              # 参数优化建议
    created_at      = Column(DateTime, default=datetime.utcnow)

class JobSchedule(Base):
    __tablename__ = "job_schedule"

    id               = Column(Integer, primary_key=True, autoincrement=True)
    calculation_id   = Column(Integer, ForeignKey("calculation_parameter.id", ondelete="CASCADE"))
    scheduler_type   = Column(String)          # "SLURM" | "PBS" | "K8s" | "Cloud"
    cluster_name     = Column(String)
    queue            = Column(String)
    nodes            = Column(Integer)
    ntasks_per_node  = Column(Integer)
    walltime         = Column(String)
    submission_script= Column(Text)
    status           = Column(String)          # "submitted" | "running" | "completed" | "failed"
    submitted_at     = Column(DateTime)
    completed_at     = Column(DateTime)
    created_at       = Column(DateTime, default=datetime.utcnow)

class PostAnalysis(Base):
    __tablename__ = "post_analysis"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    calculation_id  = Column(Integer, ForeignKey("calculation_parameter.id", ondelete="CASCADE"))
    analysis_type   = Column(String)           # "optimization" | "dos" | "band" | "elf" | "bader" | "neb" | ...
    input_files     = Column(JSON)             # e.g. {"DOSCAR": "...", "ELFCAR": "..."}
    extracted_data  = Column(JSON)             # e.g. {"band_gap": 1.2, "fermi_level": 5.1}
    plots           = Column(JSON)             # 路径或元数据
    llm_summary     = Column(Text)
    created_at      = Column(DateTime, default=datetime.utcnow)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI Depends 用法:
        async with get_session() as s: ...  (框架内部处理)
    """
    async with AsyncSessionLocal() as s:
        # 如果你在事务里做 commit/rollback，这里也可以 try/finally
        yield s
        # FastAPI 会在退出依赖时结束上下文；不需要手动 close

# 说明：
# - 开发环境可在 FastAPI 启动时自动建表：
#   in server/main.py
#     from server.db import Base, engine
#     @app.on_event("startup")
#     async def _create_all():
#         async with engine.begin() as conn:
#             await conn.run_sync(Base.metadata.create_all)
# - 生产环境建议使用 Alembic 做迁移，避免直接 create_all。

# ==== Executions (for records & finetuning) ====
class ExecutionRun(Base):
    __tablename__ = "execution_run"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey("chat_session.id"), nullable=True)
    workdir = Column(String, nullable=False)
    tasks_json = Column(JSON)        # 全量 tasks（plan 阶段的）
    selected_ids = Column(JSON)      # 本次执行选择了哪些 id
    results_json = Column(JSON)      # 每步 {step, status, ...}
    summary_json = Column(JSON)      # 可选：post_agent 的汇总
    meta = Column(JSON)              # 预留字段（cluster、dry_run 等）
    created_at = Column(DateTime, default=datetime.utcnow)

class ExecutionStep(Base):
    __tablename__ = "execution_step"
    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey("execution_run.id"), index=True)
    step_order = Column(Integer)
    name = Column(String)
    agent = Column(String)
    input_data = Column(JSON)        # 提交前的输入（参数、表单）
    output_data = Column(JSON)       # 该步的产出（job_id、文件、数值）
    status = Column(String)          # done / error / skipped
    created_at = Column(DateTime, default=datetime.utcnow)


# === New Tables: Job / FileAsset / ResultRow ===
class Job(Base):
    __tablename__ = "job"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    job_uid      = Column(String, unique=True, default=uid)
    session_uid  = Column(String, ForeignKey("chat_session.uid"), index=True)
    title        = Column(String)
    server_name  = Column(String)        # 目标服务器
    scheduler    = Column(String)        # pbs/slurm
    batch_uid    = Column(String, index=True)  # 批次号
    params       = Column(JSON)          # 任意参数（ppn/partition等）
    remote_dir   = Column(String)
    local_dir    = Column(String)
    slurm_or_pbs_id = Column(String)
    status       = Column(String, default="created")
    meta         = Column(JSON)
    created_at   = Column(DateTime, default=datetime.utcnow)
    updated_at   = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
class FileAsset(Base):
    __tablename__ = "file_asset"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    job_uid     = Column(String, ForeignKey("job.job_uid"), index=True)
    kind        = Column(String)   # POSCAR/OUT/CSV/etc
    path_remote = Column(String)
    path_local  = Column(String)
    size        = Column(Integer)
    created_at  = Column(DateTime, default=datetime.utcnow)
class ResultRow(Base):
    __tablename__ = "result_row"

    id        = Column(Integer, primary_key=True, autoincrement=True)
    job_uid   = Column(String, ForeignKey("job.job_uid"), index=True)
    step      = Column(String)
    energy    = Column(String)
    info      = Column(JSON)
    created_at= Column(DateTime, default=datetime.utcnow)