"""模式效果测试 API（页面路由在 pages.py）— 一次性生成迷你素材 + 零消耗本地合成."""
import asyncio
import json

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from ..mode_test_service import get_mode_test_service

router = APIRouter()


@router.get("/api/mode_test/status")
async def api_mode_test_status():
    return get_mode_test_service().full_status()


@router.post("/api/mode_test/generate")
async def api_mode_test_generate(request: Request):
    """生成指定模式的迷你测试素材（消耗积分，与主 pipeline 互斥）。"""
    data = await request.json()
    mode = data.get("mode", "")
    topic = str(data.get("topic", "") or "")
    force = bool(data.get("force", False))
    ok, msg = get_mode_test_service().start_generate(mode, topic=topic, force=force)
    return {"ok": ok, "message": msg}


@router.post("/api/mode_test/compose")
async def api_mode_test_compose(request: Request):
    """对已生成素材批量本地合成测试视频（零积分）。"""
    data = await request.json()
    modes = data.get("modes") or []
    ok, msg = get_mode_test_service().start_compose(modes)
    return {"ok": ok, "message": msg}


@router.post("/api/mode_test/stop")
async def api_mode_test_stop():
    service = get_mode_test_service()
    service.stop()
    return {"ok": True}


@router.get("/api/mode_test/logs")
async def api_mode_test_logs():
    """SSE endpoint for mode-test logs."""
    service = get_mode_test_service()

    async def event_stream():
        last_idx = 0
        existing = service.get_logs_since(last_idx)
        last_idx = len(service.log_lines)
        for line in existing:
            yield f"data: {json.dumps({'type': 'log', 'line': line})}\n\n"

        while True:
            status = service.get_progress()
            new_logs = service.get_logs_since(last_idx)
            last_idx = len(service.log_lines)
            for line in new_logs:
                yield f"data: {json.dumps({'type': 'log', 'line': line})}\n\n"
            yield f"data: {json.dumps({'type': 'status', **status})}\n\n"
            if status["status"] not in ("running", "paused"):
                break
            await asyncio.sleep(0.5)

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


@router.get("/api/mode_test/video/{mode}/{video_name}")
async def api_mode_test_video(mode: str, video_name: str):
    """播放测试视频（仅限 active 素材集目录内，防路径穿越）。"""
    service = get_mode_test_service()
    if mode not in ("original", "original_static", "original_cutout", "quest"):
        return JSONResponse({"error": "未知模式"}, status_code=400)
    # 路径穿越防护：只允许纯文件名
    if "/" in video_name or "\\" in video_name or ".." in video_name:
        return JSONResponse({"error": "非法路径"}, status_code=400)
    set_dir = service._active_set(mode)
    if set_dir is None:
        return JSONResponse({"error": "素材集不存在"}, status_code=404)
    video_path = set_dir / video_name
    if not video_path.exists():
        video_path = set_dir / "videos" / video_name
    if not video_path.exists():
        return JSONResponse({"error": "Not found"}, status_code=404)
    return FileResponse(str(video_path), media_type="video/mp4")
