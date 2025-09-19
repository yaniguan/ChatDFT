# server/execution/hpc_agent.py
# -*- coding: utf-8 -*-
"""
HPCAgent — unified wrapper for Slurm / PBS queues over SSH/rsync.

Now supports per-session/project namespaces:
Remote path becomes <remote_base>/<namespace>/<job_dir.name>/ if namespace is provided.
"""

from __future__ import annotations

import re
import shlex
import subprocess
import time
from pathlib import Path
from string import Template
from typing import Dict, Optional, Sequence, Any

import json
import yaml
import json, unicodedata  # 新增
# server/execution/hpc_agent.py
from pathlib import Path

# class HPCAgent:
#     # 你的类里大概已有 __init__(cluster, dry_run=False, sync_back=True) 等
#     # 确保暴露以下方法名称（或在 task_routes 中按你的真实方法名调用）：
#     def prepare_script(self, step_ctx: dict, job_dir: Path) -> Path: ...
#     def submit(self, job_dir: Path) -> str: ...

    # def wait(self, job_id: str, poll: int = 60): 
    #     print(f'job_id: {job_id}')
    #     sleep(60)
    
    

    # def fetch_outputs(self, job_dir: Path, filters=None): ...
    
def _slug(s) -> str:
    s = str(s or "").strip()
    # 统一到 ASCII，去掉奇异 unicode（比如全角破折号）
    try:
        s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    except Exception:
        pass
    # 仅保留字母数字._-，其他都变成连字符
    import re
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or "job"
# ---------------------------------------------------------------------
# Load cluster config
# ---------------------------------------------------------------------
_CFG_PATH = Path(__file__).parent / "cluster_config.yaml"
CONF: Dict[str, dict] = yaml.safe_load(_CFG_PATH.read_text()) if _CFG_PATH.exists() else {}


def _run(cmd: str, *, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, shell=True, check=check, capture_output=True, text=True)


def _join_url(*parts: str) -> str:
    return "/".join([str(p).strip("/").rstrip() for p in parts if p is not None and str(p).strip() != ""])


def _slugify(x: Any) -> str:
    s = str(x or "").strip()
    if not s:
        return ""
    s = s.replace("\\", "/").replace(" ", "_")
    # 只保留常见安全字符
    out = []
    for ch in s:
        if ch.isalnum() or ch in "-_.@/":
            out.append(ch)
    return "".join(out).strip("/")


class HPCAgent:
    def __init__(
        self,
        *,
        cluster: str = "hoffman2",
        account: Optional[str] = None,
        partition: Optional[str] = None,
        qos: Optional[str] = None,
        nodes: int = 1,
        ntasks: Optional[int] = None,
        time_limit: str = "24:00:00",
        dry_run: bool = False,
        sync_back: bool = True,
        work_root_remote: Optional[str] = None,  # override remote_base if wanted
        namespace: Optional[str] = None,         # <—— 新增，会话/项目命名空间
        **_ignore,
    ):
        if cluster not in CONF:
            raise KeyError(f"{cluster} not found in cluster_config.yaml")
        self.cluster = cluster
        self.conf = dict(CONF[cluster])
        self.runtime_project: str | None = None
        self.runtime_session_id: str | int | None = None
        self.scheduler = (self.conf.get("scheduler") or "slurm").lower()
        if self.scheduler not in {"slurm", "pbs", "sge"}:
            raise ValueError("scheduler must be 'slurm', 'pbs' or 'sge'")

        self.account = account or self.conf.get("account")
        self.partition = partition or self.conf.get("partition")
        self.qos = qos or self.conf.get("qos")
        self.nodes = int(nodes or self.conf.get("nodes") or 1)
        self.ntasks = int(ntasks or self.conf.get("ntasks") or 24)
        self.time_limit = time_limit or self.conf.get("time_limit") or "24:00:00"

        self.dry = bool(dry_run)
        self.sync_back = bool(sync_back)

        self.host = self.conf["host"]  # e.g., "user@hoffman2.idre.ucla.edu"
        self.ssh_options = self.conf.get(
            "ssh_options",
            "-o BatchMode=yes -o StrictHostKeyChecking=accept-new"
        )

        # remote_base 规整
        rb = (work_root_remote or self.conf.get("remote_base") or "~/chatdft_jobs").strip()
        if rb.startswith("~"):
            self.remote_base = rb
        else:
            if not rb.startswith("/"):
                rb = "/" + rb.lstrip("/")
            self.remote_base = "/" + _join_url(rb)

        # 命名空间（可选）
        self.namespace = _slugify(namespace)

        self.vasp_bin = self.conf.get("vasp_bin", "vasp_std")
        self.qe_pw_bin = self.conf.get("qe_pw_bin", "pw.x")
        self.template_override = self.conf.get("template")

        # optional custom cmds
        self.submit_cmd = self.conf.get("submit_cmd")
        self.status_cmd = self.conf.get("status_cmd")
        self.cancel_cmd = self.conf.get("cancel_cmd")
        self.job_id_regex = self.conf.get(
            "job_id_regex",
            r"Submitted batch job (\d+)" if self.scheduler == "slurm" else r"(\d+(?:\.\S+)?)",
        )
    def set_runtime_context(self, *, project: str | None = None, session_id: str | int | None = None):
        self.runtime_project = project
        self.runtime_session_id = session_id
    # ------------------------- public helpers -------------------------
    def set_namespace(self, namespace: Optional[str]):
        """Allow caller to update namespace after init."""
        self.namespace = _slugify(namespace)

    # -----------------------------------------------------------------
    # Scripts
    # -----------------------------------------------------------------
    def prepare_script(self, step: dict, job_dir: Path) -> Path:
        engine = (step.get("engine") or "vasp").lower()
        # 按调度器决定模板名
        tpl_name = {
            "slurm": "slurm.sh",
            "pbs":   "pbs.sh",
            "sge":   "sge.sh",         # ← 新增
        }[self.scheduler]

        cluster_tpl_dir = Path(__file__).parent / "templates" / self.cluster
        engine_tpl_dir  = Path(__file__).parent / "templates" / engine
        tpl_path = cluster_tpl_dir / tpl_name if (cluster_tpl_dir / tpl_name).exists() else engine_tpl_dir / tpl_name
        if not tpl_path.exists():
            raise FileNotFoundError(f"Template not found: {tpl_path}")

        subs = dict(
            job_name=step.get("name", f"chatdft-{engine}"),
            account=self.account or "",
            partition=self.partition or "",
            qos=self.qos or "",
            nodes=self.nodes,
            ntasks=int(step.get("ntasks") or self.ntasks),
            walltime=step.get("walltime", self.time_limit),
            vasp_bin=self.vasp_bin,
            pw_bin=self.qe_pw_bin,
            engine=engine,
        )
        subs.update(step.get("template_vars") or {})

        text = Template(tpl_path.read_text()).safe_substitute(subs)
        fname = {"slurm": "run.slurm", "pbs": "job.sh", "sge": "job.sh"}[self.scheduler]  # ← sge 也用 job.sh
        script = job_dir / fname
        script.write_text(text); script.chmod(0o755)

        # 可选：落盘远端信息（便于订阅）
        remote = self._remote_path(job_dir)
        (job_dir / "_remote.json").write_text(json.dumps({
            "cluster": self.cluster, "scheduler": self.scheduler, "remote_dir": remote
        }, indent=2))
        return script

    # -----------------------------------------------------------------
    # Remote paths & rsync
    # -----------------------------------------------------------------
    def _remote_root(self) -> str:
        """remote_base（含 ~ 或绝对路径）"""
        if self.remote_base.startswith("~"):
            return self.remote_base
        return "/" + _join_url(self.remote_base)

    def _remote_path(self, job_dir: Path) -> str:
        # 组织 remote 相对路径：<project>/<s<sid>>/<safe job name>
        parts = []
        if self.runtime_project:
            parts.append(_slug(self.runtime_project))
        if self.runtime_session_id:
            parts.append(f"s{self.runtime_session_id}")
        parts.append(_slug(job_dir.name))
        rel = "/".join(parts)

        if self.remote_base.startswith("~"):
            remote = _join_url(self.remote_base, rel)
        else:
            remote = "/" + _join_url(self.remote_base, rel)

        # 落盘远端元数据，便于 records/hypothesis 订阅
        try:
            (job_dir / "_remote.json").write_text(json.dumps({
                "cluster": self.cluster,
                "scheduler": self.scheduler,
                "remote_base": self.remote_base,
                "remote_dir": remote,
                "project": self.runtime_project,
                "session_id": self.runtime_session_id,
            }, indent=2))
        except Exception:
            pass

        return remote

    def _write_remote_meta(self, job_dir: Path):
        meta = {
            "cluster": self.cluster,
            "host": self.host,
            "remote_base": self._remote_root(),
            "namespace": self.namespace or "",
            "remote_job_dir": self._remote_path(job_dir),
        }
        try:
            (job_dir / "_remote.json").write_text(json.dumps(meta, indent=2))
        except Exception:
            pass

    def _ssh_run(self, cmd: str, *, check: bool = False) -> subprocess.CompletedProcess:
        full = f"ssh {self.ssh_options} {self.host} {shlex.quote(cmd)}"
        if self.dry:
            print("[DRY]", full)
            return subprocess.CompletedProcess(args=full, returncode=0, stdout="", stderr="")
        return _run(full, check=check)

    def _rsync_up(self, local: Path, remote: str):
        remote_q = shlex.quote(remote)   # ← 安全引用
        cmd = [
            "rsync", "-az",
            "-e", f"ssh {self.ssh_options}",
            "--delete-delay",
            "--rsync-path", f"mkdir -p {remote_q} && rsync",    # ← 远端先创建
            f"{str(local)}/",
            f"{self.host}:{remote_q}/",                          # ← 目的端也引用
        ]
        if self.dry:
            print("[DRY]", " ".join(cmd)); return
        subprocess.run(cmd, check=True)

    def _rsync_down(self, remote: str, local: Path, filters: Sequence[str] | None = None):
        local.mkdir(parents=True, exist_ok=True)
        if filters:
            for pat in filters:
                cmd = [
                    "rsync", "-az",
                    "-e", f"ssh {self.ssh_options}",
                    f"{self.host}:{remote}/{pat}",
                    f"{str(local)}/",
                ]
                subprocess.run(cmd, check=False)
        else:
            cmd = [
                "rsync", "-az",
                "-e", f"ssh {self.ssh_options}",
                f"{self.host}:{remote}/",
                f"{str(local)}/",
            ]
            subprocess.run(cmd, check=True)

    # -----------------------------------------------------------------
    # Submit / Status / Wait / Cancel / Fetch
    # -----------------------------------------------------------------
    def submit(self, job_dir: Path) -> str:
        remote = self._remote_path(job_dir)
        self._rsync_up(job_dir, remote)

        # 缺省命令：按调度器选择
        if self.submit_cmd:
            sub = self.submit_cmd
        else:
            sub = {
                "slurm": "sbatch run.slurm",
                "pbs":   "qsub job.sh",
                "sge":   "qsub job.sh",      # ← SGE
            }[self.scheduler]

        shell = f"set -e; cd {shlex.quote(remote)}; {sub}"
        proc = self._ssh_run(shell, check=False)
        (job_dir / "_submit_stdout.txt").write_text(proc.stdout or "")
        (job_dir / "_submit_stderr.txt").write_text(proc.stderr or "")
        if proc.returncode != 0:
            raise RuntimeError("submit failed (rc=%s). See _submit_stderr.txt" % proc.returncode)

        jid = self._parse_job_id((proc.stdout or "") + "\n" + (proc.stderr or ""))
        (job_dir / "JOBID").write_text(jid or "UNKNOWN")
        return jid

    def status(self, job_id: str) -> str:

        if self.scheduler == "slurm":
            cmd = self.status_cmd or f"squeue -j {job_id} -h -o %T"
            out = self._ssh_run(cmd, check=False).stdout.strip()
            return out or "COMPLETED"
        else:  # pbs or sge
            cmd = self.status_cmd or f"qstat {job_id} 2>&1 || true"
            out = self._ssh_run(cmd, check=False).stdout.strip()
            if "Unknown" in out or out == "":
                return "COMPLETED"
            return "RUNNING"
    def check_job_status(self, job_id: str) -> str:

        default_cmd = f"qacct -j {job_id} | grep -E 'end_time' 2>&1 || true"
        cmd = default_cmd
        try:
            cmd = cmd.format(job_id=job_id, jid=job_id)
        except Exception:
            pass

        res = self._ssh_run(cmd, check=False)

        text = ((res.stdout or "") + "\n" + (res.stderr or "")).strip().lower()

        # print("The checked status are right hereafasdfasf")
        # print(text)

        if not text:
            return "COMPLETED"

        not_exist_markers = [
            "error: ",
            "not found",
        ]
        if any(marker in text for marker in not_exist_markers):
            return "RUNNING"

        return "COMPLETED"

    def wait(self, job_id: str, poll: int = 60, timeout: Optional[int] = None):
        if self.dry:
            return
        start = time.time()
        while True:
            st = self.status(job_id)
            if st in {"", "COMPLETED"}:
                print(f"[done] {job_id}")
                break
            if timeout and (time.time() - start) > timeout:
                raise TimeoutError(f"Job {job_id} timeout after {timeout}s")
            time.sleep(poll)

    # def check_job_status(self, job_id: str) -> dict:
    #     try:
    #         # Execute qstat command to check job status
    #         cmd = f"qstat -j {job_id} 2>&1 || true"
    #         result = self._ssh_run(cmd, check=False)
    #         out = result.stdout.strip()
    #         err = result.stderr.strip()
            
    #         # Check if job doesn't exist or has finished
    #         if ("do not exist" in out or "do not exist" in err or 
    #             "Unknown Job Id" in out or "Unknown Job Id" in err or
    #             (out == "" and err == "")):
    #             return {
    #                 "status": "COMPLETED",
    #                 "message": f"Job {job_id} has finished or does not exist"
    #             }
    #         else:
    #             # Job is still running or queued
    #             return {
    #                 "status": "RUNNING",
    #                 "message": f"Job {job_id} is still running or queued",
    #                 "details": out if out else err
    #             }
    #     except Exception as e:
    #         return {
    #             "status": "ERROR",
    #             "message": f"Failed to check job status: {str(e)}"
    #         }

    def cancel(self, job_id: str):
        cmd = self.cancel_cmd or ("scancel {jid}" if self.scheduler == "slurm" else "qdel {jid}")
        cmd = cmd.format(jid=job_id)
        if self.dry:
            print("[DRY] cancel:", cmd)
            return
        self._ssh_run(cmd, check=False)

    def fetch_outputs(self, job_dir: Path, filters: Sequence[str] | None = None):
        if self.dry or not self.sync_back:
            return
        remote = self._remote_path(job_dir)
        self._rsync_down(remote, job_dir, filters=filters)

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------
    def _remote_meta_dict(self, job_dir: Path) -> dict:
        """构造当前远端元信息（不含 job_id/status）。"""
        return {
            "cluster": self.cluster,
            "host": self.host,
            "remote_base": self.remote_base if self.remote_base.startswith("~")
                        else ("/" + "/".join(str(self.remote_base).strip("/").split("/"))),
            "namespace": self.namespace or "",
            "remote_job_dir": self._remote_path(job_dir),
        }

    def _write_remote_meta(self, job_dir: Path, **extra):
        """
        将远端信息落盘到 job_dir/_remote.json；extra 可增量写入 job_id/status/timestamps 等。
        采用 merge 策略，便于多次更新（submit、wait、fetch）。
        """
        meta_path = job_dir / "_remote.json"
        base = self._remote_meta_dict(job_dir)
        try:
            if meta_path.exists():
                import json as _json
                old = _json.loads(meta_path.read_text() or "{}")
                base.update(old or {})
        except Exception:
            pass
        base.update({k: v for k, v in (extra or {}).items() if v is not None})
        try:
            import json as _json
            meta_path.write_text(_json.dumps(base, indent=2))
        except Exception:
            pass
    def _parse_job_id(self, text: str) -> str:
        m = re.search(self.job_id_regex, text or "")
        return (m.group(1) if m else "").strip() or "UNKNOWN"