"""WebSocket endpoint for real-time scan progress updates."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from option_alpha.pipeline.progress import ScanProgress

logger = logging.getLogger(__name__)

router = APIRouter()

# Connected WebSocket clients.
_clients: set[WebSocket] = set()

# Lock for thread-safe client management.
_lock = asyncio.Lock()


async def broadcast_progress(progress: ScanProgress) -> None:
    """Broadcast scan progress to all connected WebSocket clients.

    This is the callback passed to ScanOrchestrator.run_scan(on_progress=...).
    """
    data = progress.model_dump(mode="json")
    message = json.dumps(data)

    async with _lock:
        disconnected: list[WebSocket] = []
        for ws in _clients:
            try:
                await ws.send_text(message)
            except Exception:
                disconnected.append(ws)
        for ws in disconnected:
            _clients.discard(ws)


@router.websocket("/ws/scan-progress")
async def scan_progress_ws(websocket: WebSocket) -> None:
    """WebSocket endpoint for scan progress updates.

    Clients connect here to receive real-time progress JSON messages
    during a scan. Auto-cleanup on disconnect.
    """
    await websocket.accept()
    async with _lock:
        _clients.add(websocket)
    logger.info("WebSocket client connected (%d total)", len(_clients))

    try:
        # Keep connection alive; client sends no data.
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        async with _lock:
            _clients.discard(websocket)
        logger.info("WebSocket client disconnected (%d remaining)", len(_clients))


def get_connected_count() -> int:
    """Return number of connected WebSocket clients."""
    return len(_clients)
