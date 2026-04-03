# server/chat/structure_library_routes.py
"""
Structure Library — save, list, and search structures for text-to-structure (T2S).

Endpoints:
  POST /structure/save          — insert or update one entry
  POST /structure/search        — keyword search over label + description
  POST /structure/list          — list all entries (optionally filtered by type/session)
  GET  /structure/{id}          — fetch single entry by primary key
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select, or_
from sqlalchemy.ext.asyncio import AsyncSession

from server.db import StructureLibrary, get_session

router = APIRouter(prefix="/structure", tags=["structure_library"])


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class StructureSaveReq(BaseModel):
    session_id:     Optional[int]  = None
    structure_type: str                        # "surface" | "molecule" | "adsorption"
    label:          str
    formula:        Optional[str]  = None
    smiles:         Optional[str]  = None
    description:    Optional[str]  = None
    ase_code:       Optional[str]  = None
    poscar:         Optional[str]  = None
    plot_png_b64:   Optional[str]  = None
    meta:           Optional[Dict] = None


class StructureSearchReq(BaseModel):
    query:          str
    structure_type: Optional[str] = None       # filter by type if given
    session_id:     Optional[int] = None
    limit:          int = 20


class StructureListReq(BaseModel):
    structure_type: Optional[str] = None
    session_id:     Optional[int] = None
    limit:          int = 100


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _row_to_dict(r: StructureLibrary) -> Dict[str, Any]:
    return {
        "id":             r.id,
        "session_id":     r.session_id,
        "structure_type": r.structure_type,
        "label":          r.label,
        "formula":        r.formula,
        "smiles":         r.smiles,
        "description":    r.description,
        "ase_code":       r.ase_code,
        "poscar":         r.poscar,
        "plot_png_b64":   r.plot_png_b64,
        "meta":           r.meta,
        "created_at":     r.created_at.isoformat() if r.created_at else None,
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/save")
async def save_structure(
    body: StructureSaveReq,
    db: AsyncSession = Depends(get_session),
) -> Any:
    """Insert a new StructureLibrary entry. Always creates a new row (T2S training data)."""
    row = StructureLibrary(
        session_id     = body.session_id,
        structure_type = body.structure_type,
        label          = body.label,
        formula        = body.formula,
        smiles         = body.smiles,
        description    = body.description,
        ase_code       = body.ase_code,
        poscar         = body.poscar,
        plot_png_b64   = body.plot_png_b64,
        meta           = body.meta,
    )
    db.add(row)
    await db.commit()
    await db.refresh(row)
    return {"ok": True, "id": row.id}


@router.post("/search")
async def search_structures(
    body: StructureSearchReq,
    db: AsyncSession = Depends(get_session),
) -> Any:
    """
    Keyword search over label + description columns (case-insensitive ILIKE).
    For pgvector semantic search add embedding lookup here in the future.
    """
    q = body.query.strip()
    if not q:
        return {"ok": False, "detail": "query must not be empty"}

    stmt = select(StructureLibrary).where(
        or_(
            StructureLibrary.label.ilike(f"%{q}%"),
            StructureLibrary.description.ilike(f"%{q}%"),
            StructureLibrary.formula.ilike(f"%{q}%"),
        )
    )
    if body.structure_type:
        stmt = stmt.where(StructureLibrary.structure_type == body.structure_type)
    if body.session_id:
        stmt = stmt.where(StructureLibrary.session_id == body.session_id)
    stmt = stmt.order_by(StructureLibrary.created_at.desc()).limit(body.limit)

    rows = (await db.execute(stmt)).scalars().all()
    return {"ok": True, "results": [_row_to_dict(r) for r in rows], "n": len(rows)}


@router.post("/list")
async def list_structures(
    body: StructureListReq,
    db: AsyncSession = Depends(get_session),
) -> Any:
    """List all structure entries, optionally filtered."""
    stmt = select(StructureLibrary).order_by(StructureLibrary.created_at.desc())
    if body.structure_type:
        stmt = stmt.where(StructureLibrary.structure_type == body.structure_type)
    if body.session_id:
        stmt = stmt.where(StructureLibrary.session_id == body.session_id)
    stmt = stmt.limit(body.limit)

    rows = (await db.execute(stmt)).scalars().all()
    return {"ok": True, "structures": [_row_to_dict(r) for r in rows], "n": len(rows)}


@router.get("/{structure_id}")
async def get_structure(
    structure_id: int,
    db: AsyncSession = Depends(get_session),
) -> Any:
    row = (await db.execute(
        select(StructureLibrary).where(StructureLibrary.id == structure_id)
    )).scalars().first()
    if not row:
        raise HTTPException(status_code=404, detail="Structure not found")
    return {"ok": True, "structure": _row_to_dict(row)}
