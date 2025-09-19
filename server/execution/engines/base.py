# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional, Sequence
from string import Template
import json, time, subprocess, os, re, datetime

# 事件通道（可选）
try:
    from server.execution.utils.events import post_event
except Exception:
    async def post_event(_payload):  # 兜底：不阻塞
        return

# 可选：统一的 RunEvent（若没有就用 dict）
try:
    from server.chat.contracts import RunEvent  # noqa
    def _pack_event(run_id, step_id, phase, payload=None):
        return RunEvent(run_id=run_id, step_id=step_id, phase=phase, payload=payload or {}).model_dump()
except Exception:
    def _pack_event(run_id, step_id, phase, payload=None):
        return {"run_id": run_id, "step_id": step_id, "phase": phase, "payload": payload or {}}

async def _emit(run_id: Optional[int], step_id: Optional[int], phase: str, payload: Dict[str, Any] | None = None):
    try:
        await post_event(_pack_event(run_id, step_id, phase, payload))
    except Exception:
        pass

def _now_iso():
    return datetime.datetime.utcnow().isoformat() + "Z"

class BaseEngine:
    """
    统一的 Engine 基类：带输入输出记录与事件上报。
    子类通常只需要覆盖默认参数或附加文件过滤列表。
    """
    name: str = "base"
    default_submit: str = "bash run.slurm"  # 子类可以改，如 "sbatch run.slurm" / "qsub job.sh"
    slurm_fname: str = "run.slurm"
    pbs_fname: str = "job.sh"
    default_filters: Sequence[str] = ()

    def __init__(self):
        self.job_dir: Optional[Path] = None
        self.payload: Dict[str, Any] = {}
        self.run_id: Optional[int] = None
        self.step_id: Optional[int] = None
        self.scheduler: str = "slurm"  # or "pbs"
        self.template_used: Optional[str] = None

    # ---------- context ----------
    def set_context(self, job_dir: Path, payload: Dict[str, Any] | None = None,
                    run_id: Optional[int] = None, step_id: Optional[int] = None,
                    scheduler: str = "slurm"):
        self.job_dir = Path(job_dir)
        self.job_dir.mkdir(parents=True, exist_ok=True)
        self.payload = dict(payload or {})
        self.run_id = run_id
        self.step_id = step_id
        self.scheduler = (scheduler or "slurm").lower()
        return self

    # ---------- logging ----------
    def _log(self, event: str, data: Dict[str, Any]):
        try:
            logp = self.job_dir / "_engine_log.jsonl" if self.job_dir else None
            if logp:
                line = {"ts": _now_iso(), "engine": self.name, "event": event, **(data or {})}
                with open(logp, "a", encoding="utf-8") as f:
                    f.write(json.dumps(line, ensure_ascii=False) + "\n")
        except Exception:
            pass

    # ---------- template helpers ----------
    def _find_default_template(self) -> Path:
        """
        server/execution/templates/<engine>/<slurm.sh|pbs.sh>
        """
        if self.scheduler == "pbs":
            fname = self.pbs_fname
        else:
            fname = self.slurm_fname
        base = Path(__file__).resolve().parent.parent / "templates" / self.name
        return base / fname

    def _render_template(self, template_text: str, subs: Dict[str, Any]) -> str:
        return Template(template_text).safe_substitute(subs)

    def prepare_script(self,
                       subs: Dict[str, Any],
                       *,
                       template_path: Optional[Path] = None,
                       template_text: Optional[str] = None) -> Path:
        """
        渲染提交脚本到 job_dir；记录渲染变量与最终文本。
        subs: {job_name, ntasks, walltime, account/partition/qos/bin...}
        """
        assert self.job_dir is not None, "call set_context() first"

        # 选模板
        tpl_path = template_path or self._find_default_template()
        text = template_text or (tpl_path.read_text(encoding="utf-8") if tpl_path.exists() else "")
        if not text.strip():
            raise FileNotFoundError(f"[{self.name}] Template missing or empty: {tpl_path}")

        script_name = self.pbs_fname if self.scheduler == "pbs" else self.slurm_fname
        script = self.job_dir / script_name
        out = self._render_template(text, subs or {})
        script.write_text(out, encoding="utf-8")
        script.chmod(0o755)
        self.template_used = str(tpl_path)

        # 记录
        self._log("engine.prepare", {"subs": subs, "template": str(tpl_path), "script": str(script)})
        # 事件
        from asyncio import create_task
        create_task(_emit(self.run_id, self.step_id, "engine.prepare", {
            "engine": self.name, "script": str(script), "template": str(tpl_path), "subs": subs
        }))
        return script

    # ---------- submit/wait/fetch ----------
    def _sh(self, cmd: str) -> str:
        out = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return (out.stdout or out.stderr or "").strip()

    def submit(self, submit_cmd: Optional[str] = None) -> str:
        """
        提交作业到队列；返回 job_id。submit_cmd 默认由子类给出。
        """
        assert self.job_dir is not None, "call set_context() first"
        cmd = submit_cmd or self.default_submit
        out = self._sh(f"cd {self.job_dir} && {cmd}")
        job_id = self._parse_job_id(out) or "UNKNOWN"

        self._log("engine.submit", {"cmd": cmd, "stdout": out, "job_id": job_id})
        from asyncio import create_task
        create_task(_emit(self.run_id, self.step_id, "engine.submit", {
            "engine": self.name, "cmd": cmd, "stdout": out, "job_id": job_id
        }))
        return job_id

    def wait(self, job_id: str, poll: int = 60, timeout: Optional[int] = None,
             status_cmd: Optional[str] = None):
        """
        轮询等待。默认通过 squeue/qstat；子类可改 status_cmd 或覆写。
        """
        start = time.time()
        status_cmd = status_cmd or (f"squeue -j {job_id} -h -o %T" if self.scheduler == "slurm"
                                    else f"qstat {job_id} 2>&1 || true")
        try:
            while True:
                out = self._sh(status_cmd)
                # 简单判断
                finished = (self.scheduler == "slurm" and (out.strip() == "" or "COMPLETED" in out)) or \
                           (self.scheduler == "pbs" and ("Unknown Job Id" in out or out.strip() == ""))

                from asyncio import create_task
                create_task(_emit(self.run_id, self.step_id, "engine.wait_tick", {
                    "job_id": job_id, "status_raw": out[-200:]
                }))
                self._log("engine.wait_tick", {"job_id": job_id, "status_raw": out})

                if finished:
                    self._log("engine.done", {"job_id": job_id})
                    create_task(_emit(self.run_id, self.step_id, "engine.done", {"job_id": job_id}))
                    break

                if timeout and (time.time() - start) > timeout:
                    raise TimeoutError(f"Engine wait timeout: {job_id}")

                time.sleep(max(5, int(poll)))
        except Exception as e:
            self._log("engine.error", {"job_id": job_id, "error": str(e)})
            from asyncio import create_task
            create_task(_emit(self.run_id, self.step_id, "engine.error", {"job_id": job_id, "error": str(e)}))
            raise

    def fetch_outputs(self, filters: Sequence[str] | None = None):
        """
        这里只落本地记录。真正 rsync 的动作一般由 HPCAgent 完成。
        但为了统一，这里留接口；子类可实现远端拷回。
        """
        flt = list(filters or self.default_filters or [])
        self._log("engine.fetch", {"filters": flt})
        from asyncio import create_task
        create_task(_emit(self.run_id, self.step_id, "engine.fetch", {"filters": flt}))

    # ---------- utils ----------
    def _parse_job_id(self, text: str) -> str:
        """
        默认解析，适配 slurm/pbs 的典型输出。子类可覆写。
        """
        m = re.search(r"Submitted batch job (\d+)", text)
        if not m:
            m = re.search(r"(\d+(?:\.\S+)?)", text)  # PBS 形式
        return m.group(1) if m else ""