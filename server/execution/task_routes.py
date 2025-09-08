# server/execution/task_routes.py
# -*- coding: utf-8 -*-


from __future__ import annotations

from typing import List, Optional, Dict, Any, Tuple
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession


# Here the logger are import!!!
import logging
log = logging.getLogger(__name__)

# ---- 兼容不同项目里 db 依赖注入函数的命名 ----
try:
    from server.db import get_async_session  # 标准命名
except Exception:  # pragma: no cover
    try:
        from server.db import get_session as get_async_session  # type: ignore
    except Exception:
        from server.db import get_db as get_async_session  # type: ignore

# 你的 ORM 模型
from server.db import ExecutionTask, ChatSession

# 直接复用我们在 agent_routes.py 中实现的统一流水线
from server.execution.agent_routes import _pipeline as agent_pipeline  # type: ignore

router = APIRouter(prefix="/exec", tags=["Execution"])

# =========================
# Pydantic 请求体
# =========================
class ExecTaskIn(BaseModel):
    order_idx: int = 1
    title: str
    task_type: str = Field(default="model")  # structure/adsorption/coadsorption/neb/dos/post/model
    payload: Dict[str, Any] = Field(default_factory=dict)

class ExecCommitReq(BaseModel):
    session_id: int
    tasks: List[ExecTaskIn]

class ExecDispatchReq(BaseModel):
    session_id: int
    task_ids: Optional[List[int]] = None       # 为空则提交该 session 的全部 ExecutionTask
    submit: bool = True                        # True=提交 HPC；False=prepare-only
    fetch: bool = True                         # 提交后是否拉回
    wait: bool = False                         # 是否阻塞等待（通常不等）
    poll: int = 60                             # wait 轮询秒
    do_post: bool = True                       # 拉回后是否本地 post-analysis

# =========================
# Helpers
# =========================
def _agent_from_task_type(tt: str) -> str:
    """
    将 UI 里的 task_type 映射到 agent_routes 支持的 agent 名称。
    必须与 server/execution/agent_routes.py 中的规范一致。
    """
    t = (tt or "").lower()
    if t == "structure":
        return "structure.relax_slab"
    if t == "adsorption":
        return "adsorption.scan"
    if t == "coadsorption":
        return "adsorption.co"
    if t == "neb":
        return "neb.run"
    if t in ("dos", "electronic"):
        return "electronic.dos"
    if t == "post":
        return "post.analysis"
    # 兜底：纯计算作业
    return "run_dft"

def _engine_from_payload(p: Dict[str, Any]) -> str:
    # 支持 payload.calc.engine / payload.engine；否则默认 vasp
    try:
        return ((p.get("calc") or {}).get("engine") or p.get("engine") or "vasp").lower()
    except Exception:
        return "vasp"

def _build_agent_opts(row: ExecutionTask, *, submit: bool, fetch: bool, wait: bool, poll: int, do_post: bool, job_name: str) -> Dict[str, Any]:
    payload = row.payload or {}
    hpc_cfg = (payload.get("hpc") or {})
    return {
        "engine": _engine_from_payload(payload),
        "cluster": hpc_cfg.get("cluster", "hoffman2"),
        "submit": bool(submit),
        "wait": bool(wait),
        "fetch": bool(fetch),
        "poll": int(poll),
        "do_post": bool(do_post),
        "job_name": job_name,
        # 提供常用的 fetch 过滤；agent_routes 会用 body 中的值覆盖默认值
        "fetch_filters": [
            "OUTCAR","vasprun.xml","OSZICAR","stdout*","stderr*",
            "CONTCAR","DOSCAR","EIGENVAL","PROCAR","CHGCAR","AECCAR*","ELFCAR","ACF.dat","PARCHG*"
        ],
    }

# =========================
# 路由
# =========================
@router.post("/tasks/commit")
async def commit_tasks(req: ExecCommitReq, db: AsyncSession = Depends(get_async_session)):
    """
    覆盖式提交 ExecutionTask：先清旧，再插入新任务。
    """

    # Here is like a test thing OK???
    log.info("Testing only:")
    log.info("Committed %d tasks for session %s", len(req.tasks), req.session_id)

    await db.execute(delete(ExecutionTask).where(ExecutionTask.session_id == req.session_id))
    await db.flush()

    for i, t in enumerate(req.tasks, start=1):
        row = ExecutionTask(
            session_id=req.session_id,
            order_idx=t.order_idx or i,
            title=t.title,
            task_type=t.task_type,
            payload=t.payload,
            status="queued",
        )
        db.add(row)

    await db.commit()
    return {"ok": True, "count": len(req.tasks)}

# @router.post("/tasks/list")
# async def list_tasks(session_id: int, db: AsyncSession = Depends(get_async_session)):

from fastapi import Body
@router.post("/tasks/list")
async def list_tasks(session_id: int = Body(..., embed=True), db: AsyncSession = Depends(get_async_session)):
    q = await db.execute(
        select(ExecutionTask)
        .where(ExecutionTask.session_id == session_id)
        .order_by(ExecutionTask.order_idx)
    )
    rows = q.scalars().all()
    return {
        "ok": True,
        "tasks": [
            {
                "id": r.id,
                "order_idx": r.order_idx,
                "title": r.title,
                "task_type": r.task_type,
                "payload": r.payload,
                "status": r.status,
            }
            for r in rows
        ],
    }

@router.post("/tasks/dispatch")
async def dispatch_tasks(req: ExecDispatchReq, db: AsyncSession = Depends(get_async_session)):
    """
    将 ExecutionTask 下发到统一流水线（与 /agent/run 相同逻辑）：
    - submit=False 时仅 prepare（生成输入+POSCAR 预览），不提交 HPC
    - submit=True 则提交（可选 fetch/wait/post）
    """
    q = select(ExecutionTask).where(ExecutionTask.session_id == req.session_id)
    if req.task_ids:
        q = q.where(ExecutionTask.id.in_(req.task_ids))
    q = q.order_by(ExecutionTask.order_idx)

    rows = (await db.execute(q)).scalars().all()
    if not rows:
        raise HTTPException(status_code=404, detail="No ExecutionTask to dispatch.")

    sess = (
        await db.execute(select(ChatSession).where(ChatSession.id == req.session_id))
    ).scalar_one_or_none()
    sess_name = (sess.name if sess else f"sess{req.session_id}").replace(" ", "_")

    out: Dict[str, Any] = {"ok": True, "submitted": [], "errors": []}

    for r in rows:
        agent = _agent_from_task_type(r.task_type)
        job_name = f"{sess_name}-{r.order_idx:02d}-{r.title}".replace(" ", "_")[:100]

        # 组装给流水线的“任务卡”
        task_card = {
            "id": r.id,
            "name": r.title,
            "agent": agent,
            "params": {"payload": r.payload or {}},
            # 也可以在 payload 内额外透传结构/模板参数
        }
        opts = _build_agent_opts(
            r,
            submit=req.submit,
            fetch=req.fetch,
            wait=req.wait,
            poll=req.poll,
            do_post=req.do_post,
            job_name=job_name,
        )

        try:
            ok, res = await agent_pipeline(agent, task_card, opts)  # <— 直接调用统一流水线
            r.status = "submitted" if ok else "failed"
            if ok:
                out["submitted"].append({"id": r.id, "title": r.title, "resp": res})
            else:
                out["ok"] = False
                out["errors"].append({"id": r.id, "title": r.title, "error": res.get("status") or "pipeline failed", "resp": res})
        except Exception as e:  # pragma: no cover
            r.status = "failed"
            out["ok"] = False
            out["errors"].append({"id": r.id, "title": r.title, "error": str(e)})

    await db.commit()
    return out