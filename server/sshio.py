# server/sshio.py
# -*- coding: utf-8 -*-
import os, shlex, re, subprocess

# 允许两种用法：
# 1) 新接口（推荐）：传入 svr dict（来自 settings.get_server(...)）
# 2) 旧接口：使用环境里的 HPC_HOST/HPC_USER/REMOTE_BASE
try:
    from .settings import settings, get_server  # get_server 可选
    HAS_SETTINGS = True
except Exception:
    HAS_SETTINGS = False


# ----------------- low-level helpers -----------------

def _target(svr):
    """return 'user@host'"""
    return f'{svr["user"]}@{svr["host"]}'

def _remote_base(svr):
    return svr["remote_base"]

def _ensure_svr(svr):
    """fallback to legacy .env values"""
    if svr:
        return svr
    if not HAS_SETTINGS:
        raise RuntimeError("No server config provided and settings not available.")
    return {
        "host": settings.HPC_HOST,
        "user": settings.HPC_USER,
        "remote_base": settings.REMOTE_BASE,
        # 没有 scheduler 时默认 PBS；只会影响 submit/poll
        "scheduler": getattr(settings, "SCHEDULER", "pbs"),
    }

def _ssh_run(svr, cmd_str, check=True, text=True):
    """
    在远端执行单条命令（可带管道）。
    cmd_str 是字符串；我们不额外包 bash -lc，ssh 默认会用登录 shell 执行。
    """
    cmd = ["ssh", _target(svr), cmd_str]
    return subprocess.run(cmd, check=check, text=text, capture_output=True)

def _ssh_out(svr, cmd_str):
    """return stdout.strip() 或空字符串；不抛异常"""
    try:
        res = _ssh_run(svr, cmd_str, check=True, text=True)
        return (res.stdout or "").strip()
    except subprocess.CalledProcessError as e:
        return (e.stdout or "").strip()


# ----------------- public API -----------------

def ensure_remote_dir(job_uid, svr=None):
    """
    在远端创建目录：<remote_base>/<job_uid>
    """
    svr = _ensure_svr(svr)
    remote = f"{_remote_base(svr).rstrip('/')}/{job_uid}"
    _ssh_run(svr, f"mkdir -p {shlex.quote(remote)}")


def push_files(job_uid, local_dir, svr=None):
    """
    rsync 本地目录到远端：<remote_base>/<job_uid>/
    """
    svr = _ensure_svr(svr)
    remote = f"{_target(svr)}:{_remote_base(svr).rstrip('/')}/{job_uid}/"
    subprocess.check_call(["rsync", "-av", local_dir.rstrip("/") + "/", remote])


def submit(job_uid, svr=None):
    """
    根据 svr['scheduler'] 选择 qsub 或 sbatch 提交 ase.sh。
    返回 job id（尽量返回纯数字/ID 部分）。
    """
    svr = _ensure_svr(svr)
    remote = f"{_remote_base(svr).rstrip('/')}/{job_uid}"
    if str(svr.get("scheduler", "pbs")).lower().startswith("pbs"):
        out = _ssh_out(svr, f"cd {shlex.quote(remote)} && qsub ase.sh")
        # 典型：10124433.hoffman2.idre.ucla.edu
        m = re.search(r"(\d+)", out)
        return m.group(1) if m else (out or "")
    else:
        out = _ssh_out(svr, f"cd {shlex.quote(remote)} && sbatch ase.sh")
        # 典型：Submitted batch job 1234567
        m = re.search(r"(\d+)", out)
        return m.group(1) if m else (out or "")

# 兼容旧路径（你 main.py 里还在用 submit_qsub）
def submit_qsub(job_uid, svr=None):
    return submit(job_uid, svr)


def poll_status(job_id: str, svr=None):
    """
    轮询作业状态：
      - PBS：qstat -f <id>，解析 'job_state =' 行
      - Slurm：squeue -h -j <id> -o %T
    映射：queued/running/done/exiting/failed
    若不在队列（或命令失败）→ 认为 done
    """
    svr = _ensure_svr(svr)
    scheduler = str(svr.get("scheduler", "pbs")).lower()

    try:
        if scheduler.startswith("pbs"):
            # 如果作业已经完成，qstat 可能返回非零并且没有输出
            out = _ssh_out(svr, f"qstat -f {shlex.quote(job_id)} 2>/dev/null")
            if not out:
                return "done"
            # 解析 'job_state = R' 这样的行
            state = None
            for line in out.splitlines():
                if "job_state" in line:
                    # 允许有空格：job_state = R
                    parts = line.strip().split("=")
                    if len(parts) >= 2:
                        state = parts[1].strip().split()[0]
                        break
            if not state:
                return "done"
            mapping = {"Q": "queued", "R": "running", "C": "done", "E": "exiting", "H": "queued"}
            return mapping.get(state, state)

        else:
            # Slurm：%T 输出简短状态，如 RUNNING / PENDING / COMPLETED / FAILED
            st = _ssh_out(svr, f"squeue -h -j {shlex.quote(job_id)} -o %T 2>/dev/null")
            if not st:
                return "done"
            st = st.strip().upper()
            if st == "PENDING":
                return "queued"
            if st == "RUNNING":
                return "running"
            if st in ("COMPLETED",):
                return "done"
            if st in ("COMPLETING", "CONFIGURING"):
                return "running"
            if st in ("CANCELLED", "FAILED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL"):
                return "failed"
            return st.lower()
    except Exception:
        # 保守：出错按 done 处理，避免阻塞同步
        return "done"


def pull_results(job_uid, local_dir, svr=None):
    """
    从远端 <remote_base>/<job_uid>/ rsync 结果到本地 local_dir
    默认只拉典型结果文件，避免大文件占满本地。
    """
    svr = _ensure_svr(svr)
    os.makedirs(local_dir, exist_ok=True)
    remote = f"{_target(svr)}:{_remote_base(svr).rstrip('/')}/{job_uid}/"

    # 仅包含常见结果；可按需扩展