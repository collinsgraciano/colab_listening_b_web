"""Pipeline Run API — 启动 / 继续 / 停止 / 状态 / SSE 日志."""
import asyncio
import json

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from ..config_manager import load_config
from ..pipeline_service import get_service

router = APIRouter()


@router.post("/api/run/start")
async def api_run_start(request: Request):
    service = get_service()
    data = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
    config = data.get("config") or load_config()
    resume = data.get("resume", False)
    step_mode = data.get("step_mode", False)
    ok = service.start(config, resume=resume, step_mode=step_mode)
    return {"ok": ok, "status": service.status}


@router.post("/api/run/continue")
async def api_run_continue():
    """Continue to next step in step mode."""
    service = get_service()
    service.continue_step()
    return {"ok": True, "status": service.status}


@router.post("/api/run/stop")
async def api_run_stop():
    service = get_service()
    service.stop()
    return {"ok": True, "status": service.status}


@router.get("/api/run/status")
async def api_run_status():
    service = get_service()
    return service.get_progress()


@router.get("/api/run/logs")
async def api_run_logs():
    """SSE endpoint for streaming logs."""
    service = get_service()

    async def event_stream():
        last_idx = 0
        # Send existing logs first
        existing = service.get_logs_since(last_idx)
        last_idx = len(service.log_lines)
        for line in existing:
            yield f"data: {json.dumps({'type': 'log', 'line': line})}\n\n"

        # Stream new logs
        while True:
            status = service.get_progress()
            # Send new logs
            new_logs = service.get_logs_since(last_idx)
            last_idx = len(service.log_lines)
            for line in new_logs:
                yield f"data: {json.dumps({'type': 'log', 'line': line})}\n\n"

            # Send status update
            yield f"data: {json.dumps({'type': 'status', **status})}\n\n"

            if status["status"] not in ("running", "paused"):
                break
            await asyncio.sleep(0.5)

        # Final status
        status = service.get_progress()
        yield f"data: {json.dumps({'type': 'done', **status})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/api/run/logs/since/{since}")
async def api_run_logs_since(since: int):
    """Get logs since a given index (for non-SSE polling)."""
    service = get_service()
    logs = service.get_logs_since(since)
    total = len(service.log_lines)
    return {"logs": logs, "total": total, "status": service.get_progress()}
