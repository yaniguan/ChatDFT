from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class JobCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    formula: str = Field(default="", max_length=128)
    poscar: str = Field(default="")


class ConvergenceStep(BaseModel):
    step: int
    energy: float
    force: float


class JobRead(BaseModel):
    id: str
    name: str
    status: str
    formula: str
    poscar: str
    created_at: Optional[datetime]
    updated_at: Optional[datetime]
    started_at: Optional[datetime]
    finished_at: Optional[datetime]
    energy: Optional[float]
    error: Optional[str]
    convergence: list[dict[str, Any]]
    structure_xyz: Optional[str]


class JobList(BaseModel):
    items: list[JobRead]
    total: int
    page: int
    page_size: int
