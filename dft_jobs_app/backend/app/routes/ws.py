from __future__ import annotations

import asyncio
import logging

import redis.asyncio as aioredis
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..config import settings

router = APIRouter()
logger = logging.getLogger(__name__)


async def _pump(ws: WebSocket, channel: str) -> None:
    redis_client = aioredis.from_url(settings.redis_url, decode_responses=True)
    pubsub = redis_client.pubsub()
    await pubsub.subscribe(channel)

    listener_task: asyncio.Task[None] | None = None
    client_task: asyncio.Task[None] | None = None

    async def forward() -> None:
        async for msg in pubsub.listen():
            if msg is None:
                continue
            if msg.get("type") != "message":
                continue
            data = msg.get("data")
            if isinstance(data, bytes):
                data = data.decode("utf-8")
            await ws.send_text(data)

    async def drain_client() -> None:
        # Drain incoming frames so the websocket keeps its state in sync and
        # we detect client disconnects promptly.
        while True:
            await ws.receive_text()

    try:
        listener_task = asyncio.create_task(forward())
        client_task = asyncio.create_task(drain_client())
        done, pending = await asyncio.wait(
            {listener_task, client_task}, return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
    except WebSocketDisconnect:
        pass
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("ws pump error on %s: %s", channel, exc)
    finally:
        try:
            await pubsub.unsubscribe(channel)
            await pubsub.close()
        except Exception:
            pass
        try:
            await redis_client.close()
        except Exception:
            pass


@router.websocket("/ws/jobs")
async def ws_all_jobs(ws: WebSocket) -> None:
    await ws.accept()
    await _pump(ws, "jobs:all")


@router.websocket("/ws/jobs/{job_id}")
async def ws_single_job(ws: WebSocket, job_id: str) -> None:
    await ws.accept()
    await _pump(ws, f"job:{job_id}")
