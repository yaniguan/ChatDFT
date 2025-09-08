# server/execution/agent_routes.py
# -*- coding: utf-8 -*-
"""
微路由：供前端“单步 Run / Prepare-only”使用。
路径前缀：/agent/...

功能概览
- StructureAgent：bulk/slab/adsorb/modify
- ParametersAgent：VASP/QE 输入生成（INCAR/KPOINTS 或 *.in）
- HPCAgent：渲染脚本 → rsync → 提交（可选等待 & 拉回）
- PostAnalysisAgent：本地结果聚合（可选）
- Prepare-only：submit=False 时仅生成本地输入与结构，返回预览，不提交 HPC

请求体可选字段：
- task: Dict（等价 plan 里单个 task，至少带 params.payload）
- agent: str（如 structure.relax_slab / adsorption.scan / electronic.dos / neb.run / run_dft / meta.clarify / knowledge.search）
- engine: "vasp"|"qe"（若没给，从 task.params.payload.engine 取，默认 vasp）
- cluster: "hoffman2"|...（在你的 cluster 配置中定义）
- submit: bool = True      # 提交到 HPC；False=仅本地生成
- wait: bool = False       # 是否阻塞等待队列/运行完成
- fetch: bool = True       # 是否从 HPC 拉回输出（submit=True 时有效）
- poll: int = 60           # wait 的轮询秒
- fetch_filters: [ ... ]   # fetch 时的文件过滤
- do_post: bool = True     # 拉回后是否本地 PostAnalysis
- job_name: str            # HPC 脚本 job 名称
- run_id: int              # 事件记录用
"""

from __future__ import annotations
from typing import Any, Dict, Tuple, List, Optional
from pathlib import Path
import tempfile
import json
import os
import re

from fastapi import APIRouter, Request

# 执行层（与你的现有文件匹配）
from .structure_agent import StructureAgent
from .parameters_agent import ParametersAgent
from .hpc_agent import HPCAgent
from .post_analysis_agent import PostAnalysisAgent

router = APIRouter(prefix="/agent", tags=["agent"])

# ============================ 事件（可选） ============================
try:
    from server.execution.utils.events import post_event  # 如果没有就用兜底
except Exception:
    async def post_event(_payload):  # 不阻塞主流程
        return

# Here this function are for posting info only
async def _emit(run_id: Optional[int], step_id: Optional[int], phase: str, payload: Optional[dict] = None):
    """phase: queued | pre_submit | running | done | error | skipped"""
    try:
        await post_event({
            "run_id": run_id,
            "step_id": step_id,
            "phase": phase,
            "payload": payload or {}
        })
    except Exception:
        pass

# ============================ 工具函数 ============================

async def _json(request: Request) -> Dict[str, Any]:
    try:
        data = await request.json()
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}

def _ok(msg: str, extra: Dict[str, Any] | None = None):
    payload = {"ok": True, "message": msg}
    if extra: payload.update(extra)
    return payload

def _fail(detail: str, status: int = 400):
    return {"ok": False, "detail": detail, "status_code": status}

def _make_job_dir(task: Dict[str, Any]) -> Path:
    # 更安全/可读的 job 目录名： 00_Name（去除特殊字符）
    tid = int(task.get("id", 0))
    name = (task.get("name") or "Task")
    name = re.sub(r"[^\w\-\.]+", "_", name).strip("_") or "Task"
    base = tempfile.mkdtemp(prefix="chatdft_")
    job_dir = Path(base) / f"{tid:02d}_{name}"
    job_dir.mkdir(parents=True, exist_ok=True)
    return job_dir

def _norm_agent(name: str) -> str:
    n = (name or "").strip().lower()
    aliases = {
        "clarify": "meta.clarify",
        "scope_proposal": "meta.scope",
        "lit_search": "knowledge.search",
        "builder": "structure.relax_slab",
        "intermediate_builder": "structure.intermediates",
        "adsorption_builder": "adsorption.scan",
        "postprocess_energy": "post.energy",
        "postprocess_coads": "adsorption.co",
        "electronic_props": "electronic.dos",
    }
    return aliases.get(n, n or "unknown")

def _extract_engine(task: Dict[str, Any], fallback: str = "vasp") -> str:
    p = ((task.get("params") or {}).get("payload") or {})
    return (p.get("engine") or fallback).lower()

def _ensure_json(path: Path, obj: Any):
    try:
        path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    except Exception:
        pass

# POSCAR 轻量预览（不依赖 pymatgen/ase）
def _poscar_preview(p: Path) -> dict:
    try:
        t = p.read_text().splitlines()
        if len(t) < 8:
            return {"ok": False, "error": "too short POSCAR"}
        title = t[0].strip()
        scale = float(t[1].strip())
        # 3×3 晶格
        lat = []
        for i in range(2, 5):
            parts = [float(x) for x in t[i].split()]
            lat.append(parts[:3])
        elems = [x for x in t[5].split() if x]
        nums  = [int(x) for x in t[6].split() if x]
        natom = sum(nums) if nums else None
        return {"ok": True, "title": title, "scale": scale, "lattice": lat, "elements": elems, "counts": nums, "n_atoms": natom}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def _first_file(d: Path, names: List[str]) -> Optional[Path]:
    for n in names:
        p = d / n
        if p.exists(): return p
    return None

# ============================ 执行适配 ============================

def _run_structure(task: Dict[str, Any], job_dir: Path) -> Dict[str, Any]:
    sagent = StructureAgent()
    return sagent.build(task, job_dir)

def _run_parameters(task: Dict[str, Any], job_dir: Path) -> Dict[str, Any]:
    pagent = ParametersAgent()
    return pagent.generate(task, job_dir)

def _run_hpc(task: Dict[str, Any], job_dir: Path, *, cluster: str, engine: str,
             submit: bool, wait: bool, fetch: bool, poll: int, fetch_filters: list | None,
             job_name: str | None) -> Dict[str, Any]:
    step_ctx = {
        "name": job_name or (task.get("name") or f"{engine}_job"),
        "engine": engine,
        "template_vars": ((task.get("params") or {}).get("payload") or {}).get("template_vars") or {},
        "ntasks": ((task.get("params") or {}).get("payload") or {}).get("ntasks"),
        "walltime": ((task.get("params") or {}).get("payload") or {}).get("walltime"),
    }
    hpc = HPCAgent(cluster=cluster, dry_run=not submit and False, sync_back=fetch)
    script = hpc.prepare_script(step_ctx, job_dir)

    job_id = "NOT_SUBMITTED"
    if submit:
        job_id = hpc.submit(job_dir)
        if wait:
            hpc.wait(job_id, poll=poll)
        if fetch:
            hpc.fetch_outputs(job_dir, filters=fetch_filters)

    return {"script": str(script), "job_id": job_id}


def _run_post(job_dir: Path) -> Dict[str, Any]:
    post = PostAnalysisAgent()
    try:
        meta = post.analyze(job_dir)
        return {"ok": True, "post": meta}
    except Exception as e:
        return {"ok": False, "error": f"post-analysis failed: {e}"}


async def _run_knowledge_local(task: Dict[str, Any], job_dir: Path) -> Dict[str, Any]:
    import httpx
    base = os.getenv("CHATDFT_BACKEND", "http://127.0.0.1:8000").rstrip("/")
    payload = {
        "query":  (((task.get("params") or {}).get("payload") or {}).get("query")) or "",
        "intent": (((task.get("params") or {}).get("payload") or {}).get("intent")) or {},
        "limit":  int((((task.get("params") or {}).get("payload") or {}).get("limit")) or 10),
        "fast":   True,
    }
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(f"{base}/chat/knowledge", json=payload)
            r.raise_for_status()
            kb = r.json()
    except Exception as e:
        _ensure_json(job_dir / "knowledge_error.json", {"error": str(e), "payload": payload})
        return {"ok": False, "error": f"knowledge.search failed: {e}"}

    _ensure_json(job_dir / "papers.json", kb.get("records") or [])
    (job_dir / "background.txt").write_text(kb.get("background") or "")
    (job_dir / "summary.txt").write_text(kb.get("result") or "ok")
    return {"ok": True, "n": len(kb.get("records") or []), "files": ["papers.json","background.txt","summary.txt"]}

# ============================ 统一流水线 ============================

async def _pipeline(agent_name: str, task: Dict[str, Any], opts: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """
    - knowledge.search: 仅拉文献落盘
    - meta.*: 仅落 meta.json
    - structure.* / adsorption.* / electronic.* / neb.* / run_dft:
        prepare-only（submit=False）→ 早返回预览
        submit=True → Structure → Parameters → HPC（submit/wait/fetch）
    - post.analysis: 本地聚合
    """
    agent = _norm_agent(agent_name)
    job_dir = _make_job_dir(task)

    # 事件元数据
    run_id  = opts.get("run_id") or ((task.get("meta") or {}).get("run_id"))
    step_id = task.get("id")

    # 入队事件
    await _emit(run_id, step_id, "queued", {"agent": agent, "task": task, "job_dir": str(job_dir)})

    try:
        # 选项
        engine  = (opts.get("engine") or _extract_engine(task, "vasp")).lower()
        cluster = (opts.get("cluster") or "hoffman2")
        submit  = bool(opts.get("submit", True))
        wait    = bool(opts.get("wait", False))
        fetch   = bool(opts.get("fetch", True))
        poll    = int(opts.get("poll", 60))
        fetch_filters = opts.get("fetch_filters") or None
        do_post = bool(opts.get("do_post", True))
        job_name = opts.get("job_name")

        out: Dict[str, Any] = {"agent": agent, "job_dir": str(job_dir)}

        # 默认 fetch 过滤（前端传了就覆盖）
        DEFAULT_FETCH_FILTERS = [
            "OUTCAR", "vasprun.xml", "OSZICAR", "stdout*", "stderr*",
            "CONTCAR", "DOSCAR", "EIGENVAL", "PROCAR",
            "CHGCAR", "AECCAR*", "ELFCAR", "ACF.dat", "PARCHG*"
        ]
        fetch_filters = fetch_filters or DEFAULT_FETCH_FILTERS

        # 0) meta / knowledge
        if agent in {"meta.clarify", "meta.scope"}:
            (job_dir / "meta.json").write_text(json.dumps(task.get("params") or {}, indent=2))
            out.update({"status": "done(meta)"})
            await _emit(run_id, step_id, "done", {"status": out["status"], "job_dir": str(job_dir)})
            return True, out

        if agent == "knowledge.search":
            k = await _run_knowledge_local(task, job_dir)
            status = "done(knowledge)" if k.get("ok") else "error(knowledge)"
            out.update({"status": status, **k})
            phase = "done" if k.get("ok") else "error"
            await _emit(run_id, step_id, phase, {"status": status, "job_dir": str(job_dir), **k})
            return k.get("ok", False), out

        # 1) structure / adsorption / electronic / neb / run_dft
        if agent.startswith("structure.") or agent.startswith("adsorption.") or agent in {
            "electronic.dos", "neb.run", "run_dft", "post.energy"
        }:
            sres = _run_structure(task, job_dir)
            out["structure"] = sres

            pres = _run_parameters(task, job_dir)
            out["parameters"] = pres

            # ====== 早返回（Prepare-only）======
            if not submit:
                poscar = _first_file(job_dir, ["slab.POSCAR", "ads.POSCAR", "bulk.POSCAR", "structure.POSCAR", "POSCAR"])
                preview = _poscar_preview(poscar) if poscar else {"ok": False, "error": "no POSCAR found"}
                files = sorted([p.name for p in job_dir.iterdir() if p.is_file()])
                out.update({
                    "status": "done(prepared)",
                    "prepared": True,
                    "files": files,
                    "poscar_file": poscar.name if poscar else None,
                    "preview": preview,
                })
                await _emit(run_id, step_id, "done", {"status": "done(prepared)", "job_dir": str(job_dir), "preview": preview})
                return True, out
            # ====== 早返回到此为止 ======

            # 提交 HPC
            await _emit(run_id, step_id, "pre_submit", {
                "engine": engine, "cluster": cluster, "job_name": job_name, "job_dir": str(job_dir)
            })

            hres = _run_hpc(task, job_dir, cluster=cluster, engine=engine,
                            submit=submit, wait=wait, fetch=fetch, poll=poll,
                            fetch_filters=fetch_filters, job_name=job_name)
            out["hpc"] = hres

            job_id = hres.get("job_id", "NOT_SUBMITTED")
            if submit:
                await _emit(run_id, step_id, "running", {"job_id": job_id, "job_dir": str(job_dir)})

            if do_post and (wait or fetch):
                post_res = _run_post(job_dir)
                out["post"] = post_res

            status = "done(hpc)" if submit else "done(prepared)"
            out["status"] = status
            await _emit(run_id, step_id, "done", {
                "status": status, "job_id": job_id, "job_dir": str(job_dir), "has_post": bool(out.get("post"))
            })
            return True, out

        # 2) post-only
        if agent == "post.analysis":
            post_res = _run_post(job_dir)
            ok = bool(post_res.get("ok"))
            status = "done(post)" if ok else "error(post)"
            out.update({"status": status, "post": post_res})
            await _emit(run_id, step_id, "done" if ok else "error", {
                "status": status, "job_dir": str(job_dir), **post_res
            })
            return ok, out

        # 未知 agent
        out["status"] = f"skipped (unknown agent: {agent_name})"
        await _emit(run_id, step_id, "skipped", {"status": out["status"], "job_dir": str(job_dir)})
        return True, out

    except Exception as e:
        await _emit(run_id, step_id, "error", {"error": str(e), "agent": agent, "job_dir": str(job_dir)})
        return False, {"agent": agent, "job_dir": str(job_dir), "status": f"error: {e}"}

# ============================ 路由 ============================

@router.post("/run")
async def agent_run(request: Request):
    body = await _json(request)
    # 合理默认：打开 post + fetch，并给出常用过滤
    body.setdefault("do_post", True)
    body.setdefault("fetch", True)
    body.setdefault("fetch_filters", [
        "OUTCAR","vasprun.xml","OSZICAR","stdout*","stderr*",
        "CONTCAR","DOSCAR","EIGENVAL","PROCAR","CHGCAR","AECCAR*","ELFCAR","ACF.dat","PARCHG*"
    ])
    agent = body.get("agent") or ""
    task  = body.get("task")  or {}
    ok, res = await _pipeline(agent, task, body)
    # 语义上执行失败也返回 200 给前端（便于界面展示错误）
    return _ok("Pipeline finished.", res) if ok else _fail(res.get("status","unknown error"), 200)

@router.post("/clarify")
async def agent_clarify(request: Request):
    body = await _json(request)
    return _ok("Scope clarified.", {"echo": body})

@router.post("/scope_proposal")
async def agent_scope_proposal(request: Request):
    body = await _json(request)
    n = int(body.get("n_options", 3) or 3)
    options = [f"Option {i+1}: focus on {body.get('focus','kinetics')}" for i in range(n)]
    return _ok("Scope proposed.", {"options": options, "echo": body})

@router.post("/knowledge.search")
async def agent_knowledge_search(request: Request):
    body = await _json(request)
    task = {"id": body.get("id", 0), "name": "Literature scan", "params": {"payload": {
        "query": body.get("query") or (body.get("payload") or {}).get("query") or "",
        "intent": body.get("intent") or (body.get("payload") or {}).get("intent") or {},
        "limit": int(body.get("limit") or 10),
    }}}
    ok, res = await _pipeline("knowledge.search", task, {"submit": False})
    return _ok("Literature fetched.", res) if ok else _fail(res.get("status","knowledge failed"), 200)

@router.post("/structure.relax_slab")
async def agent_structure_relax_slab(request: Request):
    body = await _json(request)
    task = body.get("task") or {}
    ok, res = await _pipeline("structure.relax_slab", task, body)
    return _ok("Slab/bulk/chain/crystal prepared/submitted.", res) if ok else _fail(res.get("status","error"), 200)

@router.post("/structure.relax_adsorbate")
async def agent_structure_relax_adsorbate(request: Request):
    body = await _json(request)
    task = body.get("task") or {}
    ok, res = await _pipeline("structure.relax_adsorbate", task, body)
    return _ok("Adsorbate prepared/submitted.", res) if ok else _fail(res.get("status","error"), 200)

@router.post("/adsorption.scan")
async def agent_adsorption_scan(request: Request):
    body = await _json(request)
    task = body.get("task") or {}
    ok, res = await _pipeline("adsorption.scan", task, body)
    return _ok("Adsorption site scan prepared/submitted.", res) if ok else _fail(res.get("status","error"), 200)

@router.post("/adsorption.co")
async def agent_adsorption_co(request: Request):
    body = await _json(request)
    task = body.get("task") or {}
    ok, res = await _pipeline("adsorption.co", task, body)
    return _ok("Co-adsorption prepared/submitted.", res) if ok else _fail(res.get("status","error"), 200)

@router.post("/neb.run")
async def agent_neb_run(request: Request):
    body = await _json(request)
    task = body.get("task") or {}
    ok, res = await _pipeline("neb.run", task, body)
    return _ok("NEB prepared/submitted.", res) if ok else _fail(res.get("status","error"), 200)

@router.post("/electronic.dos")
async def agent_electronic_dos(request: Request):
    body = await _json(request)
    task = body.get("task") or {}
    ok, res = await _pipeline("electronic.dos", task, body)
    return _ok("DOS/PDOS/Bader prepared/submitted.", res) if ok else _fail(res.get("status","error"), 200)

@router.post("/post.analysis")
async def agent_post_analysis(request: Request):
    body = await _json(request)
    if body.get("job_dir"):
        try:
            post = PostAnalysisAgent()
            meta = post.analyze(Path(body["job_dir"]))
            return _ok("Post analysis finished.", {"post": meta})
        except Exception as e:
            return _fail(f"post-analysis failed: {e}", 200)
    task = body.get("task") or {}
    ok, res = await _pipeline("post.analysis", task, body)
    return _ok("Post analysis finished.", res) if ok else _fail(res.get("status","error"), 200)

@router.post("/report_plot")
async def agent_report_plot(request: Request):
    body = await _json(request)
    return _ok("Figures generated.", {"figs": ["plot1.svg", "plot2.svg"], "echo": body})

# -------------------------- HPC job helpers --------------------------
@router.post("/job/status")
async def agent_job_status(request: Request):
    """
    Check a single HPC job status.
    Input: {"cluster": "hoffman2", "job_id": "12345"}
    Output: {"ok": true, "status": "RUNNING"|"COMPLETED"|...}
    """
    body = await _json(request)
    cluster = (body.get("cluster") or "hoffman2")
    job_id = (body.get("job_id") or "").strip()
    if not job_id:
        return _fail("missing job_id", 200)
    try:
        hpc = HPCAgent(cluster=cluster, dry_run=False, sync_back=False)
        st = hpc.check_job_status(job_id)
        # print("Here we check the st output:")
        # print(st)
        # return _ok("status", {"status": 'Testing'})
        return _ok("status", {"status": st})
    except Exception as e:
        return _fail(f"status check failed: {e}", 200)
