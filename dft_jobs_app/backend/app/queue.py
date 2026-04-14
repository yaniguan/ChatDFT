from __future__ import annotations

import json
from typing import Any

import redis
from rq import Queue

from .config import settings

redis_conn = redis.Redis.from_url(settings.redis_url)
job_queue = Queue(settings.rq_queue, connection=redis_conn)


def publish_update(job_id: str, payload: dict[str, Any]) -> None:
    """Publish a JSON payload to the per-job channel and the global channel."""
    data = json.dumps(payload, default=str)
    redis_conn.publish(f"job:{job_id}", data)
    redis_conn.publish("jobs:all", data)
