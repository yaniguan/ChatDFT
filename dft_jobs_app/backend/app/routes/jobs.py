from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..db import Job, get_session
from ..queue import job_queue, publish_update
from ..schemas import JobCreate, JobList, JobRead

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


@router.post("", response_model=JobRead, status_code=201)
async def create_job(
    payload: JobCreate,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> JobRead:
    job = Job(
        id=str(uuid.uuid4()),
        name=payload.name,
        formula=payload.formula,
        poscar=payload.poscar,
        status="pending",
        convergence=[],
    )
    session.add(job)
    await session.commit()
    await session.refresh(job)

    job_queue.enqueue("app.worker.run_dft_job", job.id, job_id=job.id, job_timeout=3600)
    publish_update(job.id, {"type": "created", "job": job.to_dict()})
    return JobRead(**job.to_dict())


@router.get("", response_model=JobList)
async def list_jobs(
    session: Annotated[AsyncSession, Depends(get_session)],
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
) -> JobList:
    total_stmt = select(func.count()).select_from(Job)
    total = (await session.execute(total_stmt)).scalar_one()

    stmt = (
        select(Job)
        .order_by(Job.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    rows = (await session.execute(stmt)).scalars().all()
    items = [JobRead(**r.to_dict()) for r in rows]
    return JobList(items=items, total=int(total), page=page, page_size=page_size)


@router.get("/{job_id}", response_model=JobRead)
async def get_job(
    job_id: str,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> JobRead:
    job = await session.get(Job, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return JobRead(**job.to_dict())


@router.delete("/{job_id}", status_code=204, response_class=Response)
async def delete_job(
    job_id: str,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> Response:
    job = await session.get(Job, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")

    try:
        rq_job = job_queue.fetch_job(job_id)
        if rq_job is not None:
            rq_job.cancel()
    except Exception:
        pass

    await session.delete(job)
    await session.commit()
    publish_update(job_id, {"type": "deleted", "job_id": job_id})
    return Response(status_code=204)
