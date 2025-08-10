# server/chat/contracts.py
from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field

class Intent(BaseModel):
    domain: str = "catalysis"
    problem_type: str = "hydrogen evolution reaction"
    system: Dict[str, Any] = Field(default_factory=lambda: {"material":"Pt","catalyst":"Pt","facet":"111"})
    conditions: Dict[str, Any] = Field(default_factory=dict)

class HypothesisBundle(BaseModel):
    # LLM产物（markdown）+ 结构化内容（便于Plan）
    md: str
    steps: List[str] = []               # elementary steps (A -> B)
    intermediates: List[str] = []       # ['CO*', 'H*', 'CO2(g)']
    coads: List[Tuple[str,str]] = []    # [("CO*","H*"), ...]
    ts: List[str] = []                  # 额外TS候选（可与steps重叠）
    confidence: float = 0.0

class Task(BaseModel):
    id: int
    section: str
    name: str
    agent: str
    description: str = ""
    params: Dict[str, Any] = Field(default_factory=dict)
    meta: Dict[str, Any] = Field(default_factory=dict)

class Plan(BaseModel):
    intent: Intent
    hypothesis: HypothesisBundle
    tasks: List[Task]
    parallel_groups: Dict[int,List[int]] = Field(default_factory=dict)  # group_id -> [task_id]

class RunEvent(BaseModel):
    run_id: str
    step_id: Optional[int] = None
    phase: str  # 'queued'|'running'|'done'|'error'
    payload: Dict[str, Any] = Field(default_factory=dict) # 小而美：能耗/能量/路径/错误等