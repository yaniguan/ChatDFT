from __future__ import annotations

import math
import random
import time
from datetime import datetime, timezone

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from .config import settings
from .db import Job
from .queue import publish_update

sync_engine = create_engine(settings.sync_database_url, future=True, pool_pre_ping=True)
SyncSessionLocal = sessionmaker(bind=sync_engine, expire_on_commit=False)


H2O_XYZ = """3
H2O optimized geometry
O    0.000000    0.000000    0.117790
H    0.000000    0.755453   -0.471161
H    0.000000   -0.755453   -0.471161
"""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _broadcast(session: Session, job: Job, event: str) -> None:
    publish_update(job.id, {"type": event, "job": job.to_dict()})


def run_dft_job(job_id: str) -> None:
    """Simulate a DFT SCF + relax loop and stream updates via Redis pubsub."""
    session: Session = SyncSessionLocal()
    try:
        job = session.get(Job, job_id)
        if job is None:
            return

        job.status = "running"
        job.started_at = _now()
        job.convergence = []
        session.commit()
        session.refresh(job)
        _broadcast(session, job, "started")

        total_steps = 20
        baseline = -10.0
        fail_step = random.randint(1, total_steps) if random.random() < 0.05 else None

        convergence: list[dict[str, float | int]] = []
        for i in range(1, total_steps + 1):
            time.sleep(1.0)

            if fail_step is not None and i == fail_step:
                job.status = "failed"
                job.error = f"SCF divergence at step {i}"
                job.finished_at = _now()
                session.commit()
                session.refresh(job)
                _broadcast(session, job, "failed")
                return

            decay = 1.0 - math.exp(-i / 5.0)
            noise = random.uniform(-0.02, 0.02)
            energy = baseline - decay * 5.0 + noise
            force = 2.5 * math.exp(-i / 4.0) + random.uniform(0, 0.05)

            convergence.append({"step": i, "energy": energy, "force": force})
            job.convergence = list(convergence)
            job.energy = energy
            session.commit()
            session.refresh(job)
            _broadcast(session, job, "progress")

        job.status = "completed"
        job.structure_xyz = H2O_XYZ
        job.finished_at = _now()
        session.commit()
        session.refresh(job)
        _broadcast(session, job, "completed")
    except Exception as exc:
        try:
            job = session.get(Job, job_id)
            if job is not None:
                job.status = "failed"
                job.error = str(exc)
                job.finished_at = _now()
                session.commit()
                session.refresh(job)
                _broadcast(session, job, "failed")
        except Exception:
            pass
        raise
    finally:
        session.close()
