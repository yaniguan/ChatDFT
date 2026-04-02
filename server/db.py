# server/db.py
# -*- coding: utf-8 -*-
"""
ChatDFT — unified database schema.

Design goals
------------
1. RAG-ready: KnowledgeDoc + KnowledgeChunk with optional pgvector embeddings.
2. Full audit trail: AgentLog captures every LLM call (cost, latency, model).
3. Dynamic mechanisms: ReactionSystem + MechanismGraph replace the static REGISTRY.
4. Structured results: DFTResult stores parsed numbers linked to workflow tasks.
5. Soft deletes: deleted_at on ChatSession; all hard cascades removed from user data.

pgvector
--------
Install the PostgreSQL extension: CREATE EXTENSION IF NOT EXISTS vector;
Install Python package:          pip install pgvector
If pgvector is not available the embedding column falls back to JSON (list[float]).
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime
from typing import AsyncGenerator

from sqlalchemy import (
    Boolean, Column, DateTime, Float, ForeignKey,
    Index, Integer, JSON, String, Text, UniqueConstraint,
)
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

# ---------------------------------------------------------------------------
# pgvector — optional
# ---------------------------------------------------------------------------
try:
    from pgvector.sqlalchemy import Vector  # type: ignore
    _VECTOR_DIM = 1536          # text-embedding-3-small
    _HAS_PGVECTOR = True
except ImportError:
    Vector = None
    _VECTOR_DIM = 1536
    _HAS_PGVECTOR = False


def _vec_column():
    """Return a pgvector Vector column, or a JSON fallback."""
    if _HAS_PGVECTOR:
        return Column(Vector(_VECTOR_DIM), nullable=True)
    return Column(JSON, nullable=True)   # store as list[float]


def uid() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Engine / Session
# ---------------------------------------------------------------------------
DATABASE_URL = os.environ.get("DATABASE_URL", "")
if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL environment variable is required. "
        "Example: postgresql+asyncpg://user@localhost:5432/chatdft_ase"
    )
ECHO = bool(int(os.environ.get("SQLALCHEMY_ECHO", "0")))

engine = create_async_engine(
    DATABASE_URL, future=True, echo=ECHO,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
    pool_recycle=3600,
)

AsyncSessionLocal = async_sessionmaker(bind=engine, expire_on_commit=False, class_=AsyncSession)
SessionLocal = AsyncSessionLocal   # alias kept for backward compat

Base = declarative_base()


# ===========================================================================
# CHAT / SESSION LAYER
# ===========================================================================

class ChatSession(Base):
    __tablename__ = "chat_session"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    uid         = Column(String, unique=True, default=uid)
    name        = Column(String, nullable=False)
    user_id     = Column(Integer)
    project     = Column(String)
    tags        = Column(JSON)
    description = Column(Text)
    status      = Column(String, default="active")   # active / archived
    pinned      = Column(Boolean, default=False)

    # soft delete — filter WHERE deleted_at IS NULL
    deleted_at  = Column(DateTime, nullable=True)

    created_at  = Column(DateTime, default=datetime.utcnow)
    updated_at  = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    messages = relationship(
        "ChatMessage", back_populates="session",
        cascade="all, delete-orphan", passive_deletes=True,
    )


class ChatMessage(Base):
    __tablename__ = "chat_message"

    id               = Column(Integer, primary_key=True, autoincrement=True)
    session_id       = Column(Integer, ForeignKey("chat_session.id", ondelete="CASCADE"), index=True)
    role             = Column(String, nullable=False, index=True)  # user/assistant/system
    content          = Column(Text, nullable=False)
    msg_type         = Column(String)   # intent/hypothesis/plan/rxn_network/analysis/clarification
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

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    session = relationship("ChatSession", back_populates="messages")

    __table_args__ = (
        Index("idx_chatmessage_session_id", "session_id"),
        Index("idx_chatmessage_role", "role"),
        Index("idx_chatmessage_msg_type", "msg_type"),
    )


class Hypothesis(Base):
    __tablename__ = "hypothesis"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    session_id   = Column(Integer, ForeignKey("chat_session.id", ondelete="SET NULL"), nullable=True)
    message_id   = Column(Integer, ForeignKey("chat_message.id", ondelete="SET NULL"), nullable=True)
    intent_stage = Column(String)
    intent_area  = Column(String)
    hypothesis   = Column(Text)
    confidence   = Column(Float)
    agent        = Column(String)
    feedback     = Column(Text)
    tags         = Column(JSON)
    created_at   = Column(DateTime, default=datetime.utcnow)


class IntentPhrase(Base):
    """Few-shot exemplars for the intent agent."""
    __tablename__ = "intent_phrase"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    intent_stage  = Column(String)
    intent_area   = Column(String)
    specific_task = Column(String)
    phrase        = Column(Text, nullable=False)
    confidence    = Column(Float, default=1.0)
    source        = Column(String, default="user")   # user / agent / literature
    lang          = Column(String, default="en")
    remark        = Column(Text)
    created_at    = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("intent_stage", "intent_area", "specific_task", "phrase",
                         name="_intent_phrase_uc"),
    )


# ===========================================================================
# DYNAMIC MECHANISM LAYER  (replaces static registry.py)
# ===========================================================================

class ReactionSystem(Base):
    """
    Canonical description of a reaction system.
    Acts as the cache key for MechanismGraph.
    Two systems are "the same" if (domain, surface, reactant, product) match.
    """
    __tablename__ = "reaction_system"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    domain     = Column(String, nullable=False)   # electrochemical / thermal / photo / homogeneous
    surface    = Column(String, nullable=True)    # "Pt(111)", "Cu(100)", None for homogeneous
    reactant   = Column(String, nullable=False)   # "C4H10", "CO2", "N2"
    product    = Column(String, nullable=False)   # "C4H8", "CH3OH", "NH3"

    # Serialised conditions: {"pH": 7, "potential": -0.8, "T": 300, "P": 1}
    conditions = Column(JSON, default=dict)

    # Hash for fast dedup (sha256 of domain+surface+reactant+product)
    system_hash = Column(String, index=True, nullable=False)

    source      = Column(String, default="user")  # user / literature / inferred
    created_at  = Column(DateTime, default=datetime.utcnow)

    mechanisms  = relationship("MechanismGraph", back_populates="system",
                               cascade="all, delete-orphan")

    __table_args__ = (
        UniqueConstraint("system_hash", name="_rxn_system_hash_uc"),
        Index("idx_rxnsystem_domain_surface", "domain", "surface"),
    )


class MechanismGraph(Base):
    """
    LLM-generated (and human-validated) mechanism for a ReactionSystem.
    Replaces the hardcoded REGISTRY dict.  Multiple graphs per system allowed
    (e.g. CO-path vs HCOO-path both map to CO2RR on Cu).
    """
    __tablename__ = "mechanism_graph"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    system_id     = Column(Integer, ForeignKey("reaction_system.id", ondelete="CASCADE"), index=True)
    name          = Column(String, nullable=False)        # "CO2RR_CO_path"
    family        = Column(String)                        # "CO2RR", "dehydrogenation", ...
    intermediates = Column(JSON, default=list)            # ["CO*", "COOH*", ...]
    steps         = Column(JSON, default=list)            # [{"r":[...], "p":[...], "kind":"PCET"}, ...]
    coads         = Column(JSON, default=list)            # [["CO*","H*"], ...]
    ts_candidates = Column(JSON, default=list)            # ["CO2*→COOH*", ...]
    provenance    = Column(JSON, default=dict)            # {"source":"llm", "model":"gpt-4o", ...}
    confidence    = Column(Float, default=0.0)

    # Human validation
    validated     = Column(Boolean, default=False)
    validated_by  = Column(String, nullable=True)        # "user:john" / "agent:analyze"
    validated_at  = Column(DateTime, nullable=True)

    created_at    = Column(DateTime, default=datetime.utcnow)
    updated_at    = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    system = relationship("ReactionSystem", back_populates="mechanisms")

    __table_args__ = (
        Index("idx_mechgraph_system_id", "system_id"),
        Index("idx_mechgraph_family", "family"),
    )


# ===========================================================================
# KNOWLEDGE / RAG LAYER
# ===========================================================================

class KnowledgeDoc(Base):
    """
    A full literature source (paper, review, textbook chapter, web page).
    One doc → many KnowledgeChunks for vector search.
    """
    __tablename__ = "knowledge_doc"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    title       = Column(String, nullable=False)
    abstract    = Column(Text)
    full_text   = Column(Text)       # if available (PDF parsed)
    source_type = Column(String)     # arxiv / doi / web / manual / zotero
    source_id   = Column(String)     # arxiv id, DOI string, URL, ...
    url         = Column(String)
    doi         = Column(String)
    authors     = Column(JSON)       # ["Smith J", "Li X", ...]
    year        = Column(Integer)
    journal     = Column(String)
    tags        = Column(JSON)       # ["CO2RR", "Pt", "electrocatalysis"]

    # Citation count for relevance boosting (populated asynchronously)
    citation_count = Column(Integer, default=0)

    ingested_at = Column(DateTime, default=datetime.utcnow)

    chunks = relationship("KnowledgeChunk", back_populates="doc",
                          cascade="all, delete-orphan")

    __table_args__ = (
        UniqueConstraint("source_type", "source_id", name="_kdoc_src_uc"),
        Index("idx_kdoc_doi", "doi"),
        Index("idx_kdoc_year", "year"),
        # tags is JSON — no btree index; use GIN at DB level if needed
    )


class KnowledgeChunk(Base):
    """
    Fixed-size text chunk from a KnowledgeDoc, with a vector embedding.
    This is the unit retrieved during semantic RAG.
    """
    __tablename__ = "knowledge_chunk"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    doc_id     = Column(Integer, ForeignKey("knowledge_doc.id", ondelete="CASCADE"), index=True)
    chunk_idx  = Column(Integer, nullable=False)   # 0-based order within doc
    text       = Column(Text, nullable=False)
    token_count = Column(Integer)

    # pgvector column (or JSON fallback)
    embedding  = _vec_column()

    # Section hint for smarter retrieval
    section    = Column(String)     # "abstract" / "introduction" / "results" / "methods"

    created_at = Column(DateTime, default=datetime.utcnow)

    doc = relationship("KnowledgeDoc", back_populates="chunks")

    __table_args__ = (
        UniqueConstraint("doc_id", "chunk_idx", name="_kchunk_doc_idx_uc"),
        Index("idx_kchunk_doc_id", "doc_id"),
    )


# ===========================================================================
# AGENT AUDIT LOG
# ===========================================================================

class AgentLog(Base):
    """
    Logs every LLM call made by any agent.
    Enables cost tracking, debugging, and future fine-tuning dataset creation.
    """
    __tablename__ = "agent_log"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    session_id   = Column(Integer, ForeignKey("chat_session.id", ondelete="SET NULL"), nullable=True, index=True)
    agent_name   = Column(String, nullable=False)   # "intent_agent", "plan_agent", ...
    call_type    = Column(String)                   # "llm", "embed", "search", "rag"
    model        = Column(String)                   # "gpt-4o", "text-embedding-3-small"
    input_tokens = Column(Integer, default=0)
    output_tokens = Column(Integer, default=0)
    latency_ms   = Column(Integer, default=0)
    success      = Column(Boolean, default=True)
    error_msg    = Column(Text, nullable=True)

    # Abbreviated preview for debugging (not full prompt to save space)
    input_preview  = Column(Text)   # first 500 chars of prompt
    output_preview = Column(Text)   # first 500 chars of response

    # Full payload stored only when needed (e.g. failed calls for debugging)
    full_input  = Column(JSON, nullable=True)
    full_output = Column(JSON, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_agentlog_session_id", "session_id"),
        Index("idx_agentlog_agent_name", "agent_name"),
        Index("idx_agentlog_created_at", "created_at"),
    )


# ===========================================================================
# WORKFLOW / TASK LAYER
# ===========================================================================

class WorkflowTask(Base):
    __tablename__ = "workflow_task"

    id             = Column(Integer, primary_key=True, autoincrement=True)
    session_id     = Column(Integer, ForeignKey("chat_session.id", ondelete="CASCADE"), index=True)
    message_id     = Column(Integer, ForeignKey("chat_message.id", ondelete="SET NULL"), nullable=True)
    parent_task_id = Column(Integer, ForeignKey("workflow_task.id", ondelete="SET NULL"), nullable=True)

    step_order  = Column(Integer)
    name        = Column(String)
    description = Column(Text)
    agent       = Column(String)    # "structure.relax_slab", "neb.ci_neb", ...
    engine      = Column(String)    # "VASP", "QE", "CP2K"
    task_type   = Column(String)    # "slab_build", "adsorption", "neb", "gcdft", "post_analysis"
    status      = Column(String, default="idle")   # idle/queued/running/done/failed

    # Dependency graph: list of task IDs that must complete before this one
    depends_on  = Column(JSON, default=list)

    input_data  = Column(JSON)
    output_data = Column(JSON)
    error_msg   = Column(Text)
    run_time    = Column(Float)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index("idx_workflowtask_session_id", "session_id"),
        Index("idx_workflowtask_status", "status"),
        Index("idx_workflowtask_task_type", "task_type"),
    )


# ===========================================================================
# DFT RESULTS LAYER
# ===========================================================================

class DFTResult(Base):
    """
    Structured DFT output linked to a WorkflowTask.
    Parsed from OUTCAR/vasprun.xml by post_analysis_agent.
    Used by analyze_agent to build ΔG diagrams and draw conclusions.
    """
    __tablename__ = "dft_result"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    task_id      = Column(Integer, ForeignKey("workflow_task.id", ondelete="CASCADE"), index=True)
    session_id   = Column(Integer, ForeignKey("chat_session.id", ondelete="CASCADE"), index=True)

    result_type  = Column(String, nullable=False)
    # "adsorption_energy" | "reaction_energy" | "activation_barrier" |
    # "gibbs_free_energy" | "dos" | "bader" | "converged_structure"

    species      = Column(String)    # "C4H9*", "H*", ...
    surface      = Column(String)    # "Pt(111)"
    site         = Column(String)    # "top", "bridge", "hollow"
    value        = Column(Float)     # primary numeric result (eV)
    unit         = Column(String, default="eV")

    # Extra structured data (barriers, PDOS peaks, Bader charges, etc.)
    extra        = Column(JSON, default=dict)

    # Reference to raw files
    job_uid      = Column(String, ForeignKey("job.job_uid"), nullable=True, index=True)
    outcar_path  = Column(String, nullable=True)

    # Quality flags
    converged    = Column(Boolean, default=True)
    warnings     = Column(JSON, default=list)

    created_at   = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_dftresult_task_id", "task_id"),
        Index("idx_dftresult_session_id", "session_id"),
        Index("idx_dftresult_result_type", "result_type"),
    )


# ===========================================================================
# EXECUTION / HPC LAYER  (unchanged from previous version)
# ===========================================================================

class Job(Base):
    __tablename__ = "job"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    job_uid         = Column(String, unique=True, default=uid)
    session_uid     = Column(String, ForeignKey("chat_session.uid"), index=True)
    title           = Column(String)
    server_name     = Column(String)
    scheduler       = Column(String)         # pbs / slurm
    batch_uid       = Column(String, index=True)
    params          = Column(JSON)
    remote_dir      = Column(String)
    local_dir       = Column(String)
    slurm_or_pbs_id = Column(String)
    status          = Column(String, default="created")
    meta            = Column(JSON)
    created_at      = Column(DateTime, default=datetime.utcnow)
    updated_at      = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class FileAsset(Base):
    __tablename__ = "file_asset"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    job_uid     = Column(String, ForeignKey("job.job_uid"), index=True)
    kind        = Column(String)    # POSCAR / OUTCAR / vasprun.xml / plot.png
    path_remote = Column(String)
    path_local  = Column(String)
    size        = Column(Integer)
    created_at  = Column(DateTime, default=datetime.utcnow)


class ResultRow(Base):
    __tablename__ = "result_row"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    job_uid    = Column(String, ForeignKey("job.job_uid"), index=True)
    step       = Column(String)
    energy     = Column(String)
    info       = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)


class ExecutionRun(Base):
    __tablename__ = "execution_run"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    session_id   = Column(Integer, ForeignKey("chat_session.id"), nullable=True)
    workdir      = Column(String, nullable=False)
    tasks_json   = Column(JSON)
    selected_ids = Column(JSON)
    results_json = Column(JSON)
    summary_json = Column(JSON)
    meta         = Column(JSON)
    created_at   = Column(DateTime, default=datetime.utcnow)


class ExecutionTask(Base):
    __tablename__ = "execution_task"

    id         = Column(String, primary_key=True, default=uid)
    session_id = Column(Integer, ForeignKey("chat_session.id", ondelete="CASCADE"), index=True)
    order_idx  = Column(Integer, nullable=False)
    title      = Column(String, nullable=False)
    task_type  = Column(String, nullable=False)
    payload    = Column(JSON, nullable=False)
    status     = Column(String, default="PLANNED", index=True)
    hpc_job_id = Column(String, nullable=True)
    local_dir  = Column(String, nullable=True)
    remote_dir = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index("idx_execution_task_session_order", "session_id", "order_idx"),
        Index("idx_execution_task_status", "status"),
    )


class ExecutionStep(Base):
    __tablename__ = "execution_step"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    run_id      = Column(Integer, ForeignKey("execution_run.id"), index=True)
    step_order  = Column(Integer)
    name        = Column(String)
    agent       = Column(String)
    input_data  = Column(JSON)
    output_data = Column(JSON)
    status      = Column(String)   # done / error / skipped
    created_at  = Column(DateTime, default=datetime.utcnow)


# ===========================================================================
# STRUCTURE LAYER  (kept for backward compat)
# ===========================================================================

class BulkStructure(Base):
    __tablename__ = "bulk_structure"

    id             = Column(Integer, primary_key=True, autoincrement=True)
    formula        = Column(String)
    structure_data = Column(JSON)
    created_at     = Column(DateTime, default=datetime.utcnow)


class SlabStructure(Base):
    __tablename__ = "slab_structure"

    id                 = Column(Integer, primary_key=True, autoincrement=True)
    bulk_id            = Column(Integer, ForeignKey("bulk_structure.id", ondelete="CASCADE"))
    miller_index       = Column(String, nullable=False)
    supercell_size     = Column(String)
    layers             = Column(Integer)
    fixed_layers       = Column(Integer)
    vacuum_thickness   = Column(Float)
    termination        = Column(String)
    shift              = Column(Float)
    symmetry_reduction = Column(Boolean, default=False)
    is_symmetric_slab  = Column(Boolean, default=True)
    min_slab_size      = Column(Float)
    min_vacuum_size    = Column(Float)
    slab_data          = Column(JSON)
    cif_path           = Column(String)
    poscar_path        = Column(String)
    tags               = Column(JSON)
    created_at         = Column(DateTime, default=datetime.utcnow)


class AdsorptionStructure(Base):
    __tablename__ = "adsorption_structure"

    id                   = Column(Integer, primary_key=True, autoincrement=True)
    slab_id              = Column(Integer, ForeignKey("slab_structure.id", ondelete="CASCADE"))
    adsorbate_name       = Column(String)
    adsorbate_formula    = Column(String)
    adsorption_site      = Column(String)
    site_coordinates     = Column(JSON)
    coverage             = Column(Float)
    orientation          = Column(JSON)
    height_above_surface = Column(Float)
    is_relaxed           = Column(Boolean, default=False)
    adsorption_energy    = Column(Float)
    cif_path             = Column(String)
    poscar_path          = Column(String)
    tags                 = Column(JSON)
    created_at           = Column(DateTime, default=datetime.utcnow)


class ModificationStructure(Base):
    __tablename__ = "modification_structure"

    id                = Column(Integer, primary_key=True, autoincrement=True)
    parent_type       = Column(String)   # bulk / slab / adsorption
    parent_id         = Column(Integer)
    modification_type = Column(String)   # doping / vacancy / strain
    parameters        = Column(JSON)
    modified_data     = Column(JSON)
    cif_path          = Column(String)
    poscar_path       = Column(String)
    created_at        = Column(DateTime, default=datetime.utcnow)


# ===========================================================================
# HIGH-THROUGHPUT (HTP) LAYER  — PostgreSQL replica of htp_dataset.db
# Consolidates SQLite data so the LLM context can see historical HTP runs.
# ===========================================================================

class HTPRun(Base):
    """
    One HTP dataset generation campaign.
    Created by htp_agent.generate_htp_dataset(); replaces htp_dataset.db as
    the canonical record so hypothesis/analyze agents can query it via SQL.
    """
    __tablename__ = "htp_run"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    session_id  = Column(Integer, ForeignKey("chat_session.id", ondelete="SET NULL"), nullable=True, index=True)
    db_path     = Column(String)           # local SQLite path (still kept for VASP jobs)
    strategy    = Column(String)           # rattle | strain | alloy | vacancy | …
    n_total     = Column(Integer)
    n_done      = Column(Integer, default=0)
    n_failed    = Column(Integer, default=0)
    encut       = Column(Integer, default=450)
    kpoints     = Column(String, default="4 4 1")
    # JSON list of {label, formula, n_atoms} summaries (not full POSCARs)
    base_labels = Column(JSON, default=list)
    extra_kwargs = Column(JSON, default=dict)
    status      = Column(String, default="generating")   # generating/pending/running/done/failed
    created_at  = Column(DateTime, default=datetime.utcnow)
    updated_at  = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    structures = relationship("HTPStructure", back_populates="run", cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_htprun_session_id", "session_id"),
        Index("idx_htprun_status", "status"),
    )


class HTPStructure(Base):
    """
    One structure in an HTP run.  Mirrors a row in the ase.db SQLite file but
    lives in PostgreSQL so the LLM can aggregate across runs.
    """
    __tablename__ = "htp_structure"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    run_id      = Column(Integer, ForeignKey("htp_run.id", ondelete="CASCADE"), index=True)
    ase_db_id   = Column(Integer, nullable=True)   # row id in the SQLite ase.db
    label       = Column(String)                   # e.g. "Cu111_rattle_042"
    formula     = Column(String)
    n_atoms     = Column(Integer)
    strategy    = Column(String)
    status      = Column(String, default="pending")   # pending/running/done/failed
    energy_eV   = Column(Float, nullable=True)
    forces_max  = Column(Float, nullable=True)         # max |F| (eV/Å)
    stress_max  = Column(Float, nullable=True)         # max stress component (eV/Å³)
    converged   = Column(Boolean, nullable=True)
    job_uid     = Column(String, ForeignKey("job.job_uid"), nullable=True, index=True)
    created_at  = Column(DateTime, default=datetime.utcnow)
    updated_at  = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    run = relationship("HTPRun", back_populates="structures")

    __table_args__ = (
        Index("idx_htpstruct_run_id", "run_id"),
        Index("idx_htpstruct_status", "status"),
    )


class StructureT2S(Base):
    """
    Text-to-Structure training dataset.
    Each row pairs a natural-language description with the corresponding POSCAR + atoms JSON.
    Used to fine-tune or evaluate structure generation models.
    """
    __tablename__ = "structure_t2s"

    id               = Column(Integer, primary_key=True, autoincrement=True)
    session_id       = Column(Integer, ForeignKey("chat_session.id", ondelete="SET NULL"), nullable=True, index=True)
    formula          = Column(String, nullable=False)
    material         = Column(String)          # e.g. "Pt"
    facet            = Column(String)          # e.g. "111"
    crystal_system   = Column(String)          # "fcc" / "bcc" / "hcp"
    adsorbates       = Column(JSON, default=list)   # ["CO", "H"]
    natural_language = Column(Text)            # human-readable description
    poscar_content   = Column(Text)            # raw POSCAR text
    atoms_json       = Column(JSON)            # serialized atoms for viz
    n_atoms          = Column(Integer)
    is_optimized     = Column(Boolean, default=False)
    energy_eV        = Column(Float, nullable=True)  # DFT total energy if available
    provenance       = Column(JSON, default=dict)    # {"source": "plan_agent", "session": ...}
    tags             = Column(JSON, default=list)
    created_at       = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_t2s_material_facet", "material", "facet"),
        Index("idx_t2s_session_id", "session_id"),
    )


class CalculationParameter(Base):
    __tablename__ = "calculation_parameter"

    id             = Column(Integer, primary_key=True, autoincrement=True)
    structure_type = Column(String)
    structure_id   = Column(Integer)
    engine         = Column(String)
    task_type      = Column(String)
    incar_settings = Column(JSON)
    input_files    = Column(JSON)
    explanation    = Column(Text)
    suggestions    = Column(JSON)
    created_at     = Column(DateTime, default=datetime.utcnow)


class JobSchedule(Base):
    __tablename__ = "job_schedule"

    id               = Column(Integer, primary_key=True, autoincrement=True)
    calculation_id   = Column(Integer, ForeignKey("calculation_parameter.id", ondelete="CASCADE"))
    scheduler_type   = Column(String)
    cluster_name     = Column(String)
    queue            = Column(String)
    nodes            = Column(Integer)
    ntasks_per_node  = Column(Integer)
    walltime         = Column(String)
    submission_script = Column(Text)
    status           = Column(String)
    submitted_at     = Column(DateTime)
    completed_at     = Column(DateTime)
    created_at       = Column(DateTime, default=datetime.utcnow)


class PostAnalysis(Base):
    __tablename__ = "post_analysis"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    calculation_id = Column(Integer, ForeignKey("calculation_parameter.id", ondelete="CASCADE"))
    analysis_type  = Column(String)
    input_files    = Column(JSON)
    extracted_data = Column(JSON)
    plots          = Column(JSON)
    llm_summary    = Column(Text)
    created_at     = Column(DateTime, default=datetime.utcnow)


# ===========================================================================
# MULTI-MODAL KNOWLEDGE — Figures
# ===========================================================================

class KnowledgeFigure(Base):
    """
    A figure extracted from a KnowledgeDoc (PDF page).
    The 'description' field is a GPT-4o-vision caption, which is then embedded
    for semantic retrieval — allowing "find papers with volcano plots" queries.
    Future: store 3-D structure descriptors here too.
    """
    __tablename__ = "knowledge_figure"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    doc_id      = Column(Integer, ForeignKey("knowledge_doc.id", ondelete="CASCADE"), index=True)
    page_num    = Column(Integer, default=0)
    figure_idx  = Column(Integer, default=0)   # 0-based index within page
    caption     = Column(Text)                 # text caption extracted from PDF
    description = Column(Text)                 # GPT-4o-vision generated description
    embedding   = _vec_column()                # embedding of description text
    image_path  = Column(String)               # local path to saved PNG/JPG
    width       = Column(Integer)
    height      = Column(Integer)
    created_at  = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("doc_id", "page_num", "figure_idx", name="_kfig_doc_page_idx_uc"),
        Index("idx_kfig_doc_id", "doc_id"),
    )


class LiteratureUpdateLog(Base):
    """
    Audit log for scheduled / manual literature ingestion runs.
    Used to avoid redundant fetches and to track knowledge-base growth.
    """
    __tablename__ = "literature_update_log"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    run_at       = Column(DateTime, default=datetime.utcnow)
    trigger      = Column(String, default="scheduler")  # scheduler / manual / upload
    queries_used = Column(JSON)                          # list of arXiv search strings
    n_new_docs   = Column(Integer, default=0)
    n_new_chunks = Column(Integer, default=0)
    n_new_figs   = Column(Integer, default=0)
    duration_s   = Column(Float, default=0.0)
    error        = Column(Text, nullable=True)


class StructureLibrary(Base):
    """
    Text-to-Structure library.
    Each built structure (surface / molecule / adsorption config) is stored here
    with a rich natural-language description and the ASE code that created it.
    Enables future T2S retrieval: give a text query → get the matching POSCAR.
    """
    __tablename__ = "structure_library"

    id             = Column(Integer, primary_key=True, autoincrement=True)
    session_id     = Column(Integer, nullable=True, index=True)
    structure_type = Column(String,  nullable=False)   # "surface" | "molecule" | "adsorption"
    label          = Column(String,  nullable=False)   # e.g. "Pt(111)-4x4x3"
    formula        = Column(String,  nullable=True)    # e.g. "Pt48"
    smiles         = Column(String,  nullable=True)    # molecules only
    description    = Column(Text,    nullable=True)    # rich text description for T2S
    ase_code       = Column(Text,    nullable=True)    # reproducible Python code
    poscar         = Column(Text,    nullable=True)    # full VASP POSCAR
    plot_png_b64   = Column(Text,    nullable=True)    # base64 structure image
    meta           = Column(JSON,    nullable=True)    # facet, layers, site_type, rotation, …
    embedding      = _vec_column()                     # pgvector embedding of description
    created_at     = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_strulib_type",    "structure_type"),
        Index("idx_strulib_session", "session_id"),
        Index("idx_strulib_label",   "label"),
    )


class PlanTaskState(Base):
    """
    Persistent per-task step state for plan execution.
    Survives browser refresh and server restarts.
    One row per (session_id, task_plan_id).
    """
    __tablename__ = "plan_task_state"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    session_id      = Column(Integer, index=True, nullable=False)
    task_plan_id    = Column(Integer, nullable=False)
    task_name       = Column(String,  nullable=True)
    poscar          = Column(Text,    nullable=True)   # final selected POSCAR
    plot_png_b64    = Column(Text,    nullable=True)   # structure image
    all_configs     = Column(JSON,    nullable=True)   # list of {poscar, plot_b64, site_type, ...}
    selected_config = Column(Integer, default=0)       # index into all_configs
    scripts         = Column(JSON,    nullable=True)   # {INCAR, KPOINTS, job_sh, ...}
    job_id          = Column(String,  nullable=True)   # SGE/PBS job id
    remote_path     = Column(String,  nullable=True)
    results         = Column(JSON,    nullable=True)   # parsed results dict
    energy_eV       = Column(Float,   nullable=True)
    updated_at      = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index("idx_plantaskstate_session_task", "session_id", "task_plan_id", unique=True),
    )


# ===========================================================================
# Helpers
# ===========================================================================

async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as s:
        yield s


HAS_PGVECTOR = _HAS_PGVECTOR
VECTOR_DIM = _VECTOR_DIM
