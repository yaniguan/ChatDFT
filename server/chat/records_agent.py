# server/chat/records_agent.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from fastapi import APIRouter, Request, HTTPException
from typing import Any, Dict, List, Optional
from datetime import datetime

from sqlalchemy import select, and_, or_
from sqlalchemy.exc import SQLAlchemyError

from server.db import AsyncSessionLocal, ExecutionRun, ExecutionStep

router = APIRouter()


# ----------------------------- helpers -----------------------------

def _to_int(x, default: int) -> int:
    try:
        v = int(x)
        return v if v >= 0 else default
    except Exception:
        return default

def _cap_limit(n: int, cap: int = 200) -> int:
    n = int(n or 0)
    return min(max(n, 1), cap)

def _json_or_none(x: Any) -> Any:
    # 统一把非 dict/list 的奇怪东西转 None，避免 DB JSON 列报错
    if isinstance(x, (dict, list)):
        return x
    return None

def _run_dict(r: ExecutionRun) -> Dict[str, Any]:
    return {
        "id": r.id,
        "session_id": r.session_id,
        "workdir": r.workdir,
        "summary": getattr(r, "summary_json", None),
        "created_at": r.created_at.isoformat() if getattr(r, "created_at", None) else None,
        "updated_at": r.updated_at.isoformat() if getattr(r, "updated_at", None) else None,
    }

def _step_dict(s: ExecutionStep) -> Dict[str, Any]:
    return {
        "id": s.id,
        "run_id": s.run_id,
        "step_order": s.step_order,
        "name": s.name,
        "agent": s.agent,
        "input_data": s.input_data,
        "output_data": s.output_data,
        "status": s.status,
        "created_at": s.created_at.isoformat() if getattr(s, "created_at", None) else None,
        "updated_at": s.updated_at.isoformat() if getattr(s, "updated_at", None) else None,
    }


# ----------------------------- query APIs -----------------------------

@router.post("/chat/records/list")
async def list_runs(request: Request):
    """
    列出 ExecutionRun（分页 + 筛选）
    body:
      session_id?: int
      status?: str               # 在 summary_json["status"] 里匹配（宽松）
      q?: str                    # 在 workdir / summary_json 里粗搜
      start?: ISO time
      end?: ISO time
      limit?: int (<=200)
      offset?: int
    """
    body = await request.json() if request.headers.get("content-type","").startswith("application/json") else {}
    session_id = body.get("session_id")
    status = (body.get("status") or "").strip()
    q = (body.get("q") or "").strip()
    limit  = _cap_limit(_to_int(body.get("limit"), 20))
    offset = _to_int(body.get("offset"), 0)

    # 时间过滤（可选）
    start = body.get("start")
    end   = body.get("end")
    dt_start: Optional[datetime] = None
    dt_end: Optional[datetime] = None
    try:
        if start: dt_start = datetime.fromisoformat(start)  # type: ignore
        if end:   dt_end   = datetime.fromisoformat(end)    # type: ignore
    except Exception:
        pass

    async with AsyncSessionLocal() as s:
        conds = []
        if session_id is not None:
            conds.append(ExecutionRun.session_id == session_id)
        if dt_start:
            conds.append(ExecutionRun.created_at >= dt_start)  # type: ignore
        if dt_end:
            conds.append(ExecutionRun.created_at <= dt_end)    # type: ignore

        # 关键字/状态粗匹配（尽量不依赖 JSONB 函数，以兼容多 DB）
        if q:
            conds.append(or_(
                ExecutionRun.workdir.ilike(f"%{q}%"),
                # 某些后端支持：cast(summary_json as text) ilike
                # 为简化，这里用属性访问；若不行可在模型层提供虚拟列
                # type: ignore
                # getattr(ExecutionRun, "summary_json_text").ilike(f"%{q}%")
            ))
        if status:
            # 粗暴：要求 summary_json 里存在一个 status 字段且包含 status 子串（应用端生成时尽量写入）
            # 如需精确匹配，请在模型层为 status 建立单独列
            pass

        qy = select(ExecutionRun).order_by(ExecutionRun.created_at.desc())
        if conds:
            qy = qy.where(and_(*conds))
        rows = (await s.execute(qy.offset(offset).limit(limit))).scalars().all()

    return {"ok": True, "runs": [_run_dict(r) for r in rows], "limit": limit, "offset": offset}


@router.post("/chat/records/get")
async def get_run(request: Request):
    """
    获取某次 run + 全量 steps（按 step_order 升序）
    body:
      run_id: int
    """
    body = await request.json()
    run_id = body.get("run_id")
    if not run_id:
        raise HTTPException(400, "run_id required")

    async with AsyncSessionLocal() as s:
        run = (await s.execute(select(ExecutionRun).where(ExecutionRun.id == run_id))).scalars().first()
        if not run:
            raise HTTPException(404, "run not found")
        steps = (await s.execute(
            select(ExecutionStep).where(ExecutionStep.run_id == run_id).order_by(ExecutionStep.step_order.asc())
        )).scalars().all()

    return {"ok": True, "run": _run_dict(run), "steps": [_step_dict(x) for x in steps]}


@router.post("/chat/records/list_steps")
async def list_steps(request: Request):
    """
    分页列出某个 run 的 steps
    body:
      run_id: int
      limit?: int (<=200)
      offset?: int
    """
    body = await request.json()
    run_id = body.get("run_id")
    if not run_id:
        raise HTTPException(400, "run_id required")

    limit  = _cap_limit(_to_int(body.get("limit"), 50))
    offset = _to_int(body.get("offset"), 0)

    async with AsyncSessionLocal() as s:
        steps = (await s.execute(
            select(ExecutionStep)
            .where(ExecutionStep.run_id == run_id)
            .order_by(ExecutionStep.step_order.asc())
            .offset(offset).limit(limit)
        )).scalars().all()

    return {"ok": True, "steps": [_step_dict(x) for x in steps], "limit": limit, "offset": offset}


# ----------------------------- mutate APIs -----------------------------

@router.post("/chat/records/create_run")
async def create_run(request: Request):
    """
    新建一次执行（外部在 /chat/execute 前先创建，拿 run_id）
    body:
      session_id?: int
      workdir?: str
      summary?: dict | list
    """
    body = await request.json()
    session_id = body.get("session_id")
    workdir = (body.get("workdir") or "").strip() or None
    summary = _json_or_none(body.get("summary"))

    async with AsyncSessionLocal() as s:
        try:
            run = ExecutionRun(session_id=session_id, workdir=workdir, summary_json=summary)  # type: ignore
            s.add(run)
            await s.flush()
            await s.commit()
            return {"ok": True, "run_id": run.id}
        except SQLAlchemyError as e:
            await s.rollback()
            raise HTTPException(500, f"create_run failed: {e}")


@router.post("/chat/records/append_step")
async def append_step(request: Request):
    """
    为某个 run 追加步骤
    body:
      run_id: int
      step_order?: int   # 不给就用当前最大+1
      name: str
      agent: str
      input_data?: json
      status?: str       # default "queued"
    """
    body = await request.json()
    run_id = body.get("run_id")
    if not run_id:
        raise HTTPException(400, "run_id required")

    name  = (body.get("name") or "").strip()
    agent = (body.get("agent") or "").strip()
    if not name or not agent:
        raise HTTPException(400, "name and agent required")

    step_order = body.get("step_order")
    input_data = _json_or_none(body.get("input_data"))
    status = (body.get("status") or "queued").strip()

    async with AsyncSessionLocal() as s:
        try:
            # 计算默认 step_order
            if step_order is None:
                last = (await s.execute(
                    select(ExecutionStep.step_order)
                    .where(ExecutionStep.run_id == run_id)
                    .order_by(ExecutionStep.step_order.desc())
                    .limit(1)
                )).scalar()
                step_order = int(last or 0) + 1

            st = ExecutionStep(  # type: ignore
                run_id=run_id, step_order=step_order,
                name=name, agent=agent, input_data=input_data,
                status=status
            )
            s.add(st)
            await s.flush()
            await s.commit()
            return {"ok": True, "step_id": st.id, "step_order": step_order}
        except SQLAlchemyError as e:
            await s.rollback()
            raise HTTPException(500, f"append_step failed: {e}")


@router.post("/chat/records/update_step")
async def update_step(request: Request):
    """
    部分更新步骤（不存在则 404）
    body:
      step_id: int
      name?: str
      agent?: str
      status?: str
      step_order?: int
      input_data?: json
      output_data?: json
    """
    body = await request.json()
    step_id = body.get("step_id")
    if not step_id:
        raise HTTPException(400, "step_id required")

    async with AsyncSessionLocal() as s:
        st = (await s.execute(select(ExecutionStep).where(ExecutionStep.id == step_id))).scalars().first()
        if not st:
            raise HTTPException(404, "step not found")

        changed = False
        try:
            if "name" in body and (body["name"] or "").strip():
                st.name = body["name"].strip(); changed = True
            if "agent" in body and (body["agent"] or "").strip():
                st.agent = body["agent"].strip(); changed = True
            if "status" in body and (body["status"] or "").strip():
                st.status = body["status"].strip(); changed = True
            if "step_order" in body and body["step_order"] is not None:
                st.step_order = int(body["step_order"]); changed = True
            if "input_data" in body:
                st.input_data = _json_or_none(body["input_data"]); changed = True
            if "output_data" in body:
                st.output_data = _json_or_none(body["output_data"]); changed = True

            if changed:
                await s.flush()
                await s.commit()
            return {"ok": True, "step": _step_dict(st)}
        except SQLAlchemyError as e:
            await s.rollback()
            raise HTTPException(500, f"update_step failed: {e}")


@router.post("/chat/records/finalize_run")
async def finalize_run(request: Request):
    """
    更新一次 run 的最终摘要/工作目录等。
    body:
      run_id: int
      workdir?: str
      summary?: json
    """
    body = await request.json()
    run_id = body.get("run_id")
    if not run_id:
        raise HTTPException(400, "run_id required")

    workdir = (body.get("workdir") or "").strip() or None
    summary = _json_or_none(body.get("summary"))

    async with AsyncSessionLocal() as s:
        run = (await s.execute(select(ExecutionRun).where(ExecutionRun.id == run_id))).scalars().first()
        if not run:
            raise HTTPException(404, "run not found")
        try:
            if workdir is not None:
                run.workdir = workdir
            if summary is not None:
                run.summary_json = summary  # type: ignore
            await s.flush()
            await s.commit()
            return {"ok": True, "run": _run_dict(run)}
        except SQLAlchemyError as e:
            await s.rollback()
            raise HTTPException(500, f"finalize_run failed: {e}") 