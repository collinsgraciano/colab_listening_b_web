"""批量生成队列 API — 入队 / 启动 / 暂停 / 恢复 / 移除 / 清理 / 停止当前 / 状态."""
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from ..batch_queue_service import get_batch_queue

router = APIRouter()


@router.get("/api/batch_queue/status")
async def api_batch_queue_status():
    return get_batch_queue().status()


@router.post("/api/batch_queue/add")
async def api_batch_queue_add(request: Request):
    data = await request.json()
    added, rejected = get_batch_queue().add_items(data.get("items"))
    return {"ok": bool(added), "added": added, "rejected": rejected}


@router.post("/api/batch_queue/start")
async def api_batch_queue_start():
    ok, msg = get_batch_queue().start()
    if not ok:
        return JSONResponse({"ok": False, "error": msg}, status_code=400)
    return {"ok": True, "message": msg}


@router.post("/api/batch_queue/pause")
async def api_batch_queue_pause():
    ok, msg = get_batch_queue().pause()
    if not ok:
        return JSONResponse({"ok": False, "error": msg}, status_code=400)
    return {"ok": True, "message": msg}


@router.post("/api/batch_queue/resume")
async def api_batch_queue_resume():
    ok, msg = get_batch_queue().resume()
    if not ok:
        return JSONResponse({"ok": False, "error": msg}, status_code=400)
    return {"ok": True, "message": msg}


@router.post("/api/batch_queue/remove")
async def api_batch_queue_remove(request: Request):
    data = await request.json()
    item_id = str(data.get("item_id", "")).strip()
    if not item_id:
        return JSONResponse({"ok": False, "error": "缺少 item_id"}, status_code=400)
    ok, msg = get_batch_queue().remove(item_id)
    if not ok:
        return JSONResponse({"ok": False, "error": msg}, status_code=400)
    return {"ok": True, "message": msg}


@router.post("/api/batch_queue/clear")
async def api_batch_queue_clear(request: Request):
    data = await request.json()
    scope = str(data.get("scope", "finished")).strip() or "finished"
    if scope not in ("finished", "all"):
        return JSONResponse({"ok": False, "error": f"无效 scope: {scope}"}, status_code=400)
    ok, msg, _removed = get_batch_queue().clear(scope)
    if not ok:
        return JSONResponse({"ok": False, "error": msg}, status_code=400)
    return {"ok": True, "message": msg}


@router.post("/api/batch_queue/stop_current")
async def api_batch_queue_stop_current():
    ok, msg = get_batch_queue().stop_current()
    if not ok:
        return JSONResponse({"ok": False, "error": msg}, status_code=400)
    return {"ok": True, "message": msg}
