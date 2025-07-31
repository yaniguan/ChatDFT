from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, JSON, Float, UniqueConstraint, Boolean
from datetime import datetime

import os

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql+asyncpg://postgres:password@localhost:5432/chatdft")

engine = create_async_engine(DATABASE_URL, echo=True)
AsyncSessionLocal = sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()

# 用户、会话、任务模型举例
class ChatSession(Base):
    __tablename__ = "chat_session"
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False)
    user_id = Column(Integer)              # 新增
    project = Column(String)               # 新增
    tags = Column(String)                  # 新增，或用 JSON
    description = Column(Text)
    status = Column(String, default="active")
    pinned = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    meta = Column(JSON)                    # 预留其它元数据

class ChatMessage(Base):
    __tablename__ = "chat_message"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey("chat_session.id"))
    role = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    msg_type = Column(String)              # 新增
    intent = Column(String)                # 这条对话消息的语义意图。例如 “material_search”。用户每发一句话，经由 intent agent 解析后，存入该字段。这样你后续能直接统计/检索“最近提过哪些意图”，或“intent 和回复内容的关系”。
    confidence = Column(Float)             # 新增
    llm_model = Column(String)             # 新增
    source = Column(String)                # 新增
    parent_id = Column(Integer)            # 新增，树状对话
    attachments = Column(JSON)             # 新增
    references = Column(JSON)              # 新增
    feedback = Column(Text)                # 用户评价
    token_usage = Column(Integer)
    duration = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Knowledge(Base):
    __tablename__ = "knowledge"
    id = Column(Integer, primary_key=True)
    title = Column(String)
    content = Column(Text)
    source_type = Column(String)   # "arxiv", "wiki", "manual", "web", etc.
    source_id = Column(String)     # 例如 arxiv_id/wiki_id
    url = Column(String)
    embedding = Column(JSON)       # 支持向量检索
    tags = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

class Paper(Base):
    __tablename__ = "paper"
    id = Column(Integer, primary_key=True)
    arxiv_id = Column(String)
    title = Column(String)
    abstract = Column(Text)
    authors = Column(String)
    year = Column(Integer)
    venue = Column(String)
    url = Column(String)
    pdf_path = Column(String)
    tags = Column(String)
    embedding = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

class Wiki(Base):
    __tablename__ = "wiki"
    id = Column(Integer, primary_key=True)
    title = Column(String)
    content = Column(Text)
    url = Column(String)
    tags = Column(String)
    embedding = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

class Hypothesis(Base):
    __tablename__ = "hypothesis"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey("chat_session.id"))
    message_id = Column(Integer, ForeignKey("chat_message.id"))
    intent = Column(String) #  该假说/目标所属的意图类型（比如这条假说是关于 DOS 计算的，intent=postprocess_dos）。通常和 ChatMessage 是 1:1 或 1:n 的关系（一个对话可能推导出多个假说）。便于溯源和跨 session 归纳：如“所有关于 error_analysis 的假说长啥样”。
    hypothesis = Column(Text, nullable=False)
    confidence = Column(Float)
    agent = Column(String)    # "gpt-4o", "user", etc.
    feedback = Column(Text)
    tags = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

class IntentPhrase(Base):
    __tablename__ = "intent_phrase"
    id = Column(Integer, primary_key=True, autoincrement=True)
    intent = Column(String, nullable=False)         # 意图分类， 这个短语属于哪种意图（作为意图的 few-shot 表达示例）。存储意图训练数据：比如 “how do I relax the structure” → intent: “structure_building”。
    phrase = Column(Text, nullable=False)           # 用户表达
    confidence = Column(Float, default=1.0)         # 可选，置信度
    source = Column(String, default="user")         # 可选，来源
    author = Column(String)                         # 可选，创建者
    example_type = Column(String, default="positive") # Few-shot prompt 辅助
    lang = Column(String, default="en")             # 语言
    remark = Column(Text)                           # 管理员备注
    created_at = Column(DateTime, default=datetime.utcnow)
    __table_args__ = (UniqueConstraint('intent', 'phrase', name='_intent_phrase_uc'),)

class TaskObjectPhrase(Base):
    __tablename__ = "task_object_phrase"
    id = Column(Integer, primary_key=True)
    task_type = Column(String, nullable=False)
    object = Column(String)         # 如材料名/模型
    phrase = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    tags = Column(String)
    
class WorkflowTask(Base):
    __tablename__ = "workflow_task"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey("chat_session.id"))
    step_id = Column(Integer)
    name = Column(String)
    description = Column(Text)
    agent = Column(String)
    status = Column(String, default="idle")
    result = Column(JSON)
    intent = Column(String) #  这个工作流 step 属于哪个意图（比如“生成 POSCAR”属于 structure_building）。
    input_data = Column(JSON)
    output_data = Column(JSON)
    parent_task_id = Column(Integer, ForeignKey("workflow_task.id"), nullable=True)
    priority = Column(Integer, default=0)
    tags = Column(String)
    run_time = Column(Float)
    error_msg = Column(Text)
    owner = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)

class Material(Base):
    __tablename__ = "material"
    id = Column(Integer, primary_key=True, autoincrement=True)
    formula = Column(String, nullable=False)
    source = Column(String)
    cif = Column(Text)
    properties = Column(JSON)
    mp_id = Column(String)
    structure_type = Column(String)
    dimensionality = Column(Integer)
    spacegroup = Column(String)
    lattice_params = Column(JSON)
    elements = Column(JSON)
    bandgap = Column(Float)
    magnetic = Column(String)
    status = Column(String)
    tags = Column(String)
    owner = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)