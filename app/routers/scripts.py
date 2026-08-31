"""Scripts Library — 批量生成 / AI 审查 / 修复（页面路由在 pages.py）."""
import asyncio
import json

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..config_manager import (
    PARAM_SPEC, get_provider_options, load_config, load_llm_providers,
)
from ..paths import SCRIPTS_FORM_PATH
from ..sse import sse_line as _sse, SSE_HEADERS as _SSE_HEADERS
from ..topics_io import (
    load_topics_data as _load_topics_data,
    load_used_topic_names as _load_used_topic_names,
)
from .. import script_library

router = APIRouter()


def _load_scripts_form() -> dict:
    """Remember the last-used provider/model on the scripts page."""
    if SCRIPTS_FORM_PATH.exists():
        try:
            data = json.loads(SCRIPTS_FORM_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _remember_scripts_form(provider: str, model: str) -> None:
    if not provider and not model:
        return
    cfg = _load_scripts_form()
    if provider:
        cfg["provider"] = provider
    if model:
        cfg["model"] = model
    SCRIPTS_FORM_PATH.parent.mkdir(parents=True, exist_ok=True)
    SCRIPTS_FORM_PATH.write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")


@router.get("/api/scripts/form_options")
async def api_scripts_form_options():
    """Everything the generate form needs in one call."""
    config = load_config()
    return {
        "providers": get_provider_options(),
        "sensenova_models": PARAM_SPEC["sensenova_model"]["options"],
        "openai_models": PARAM_SPEC["openai_model"]["options"],
        "custom_providers": load_llm_providers(),
        "topics_data": _load_topics_data(config),
        "used_topics": _load_used_topic_names(config),
        "used_by_mode": script_library.used_by_mode_all(),
        "library_topics": script_library.library_topics_by_mode(),
        "default_lines": script_library.DEFAULT_LINES,
        "last": _load_scripts_form(),  # 上次使用的 provider/model（优先于 current 回显）
        "current": {
            "provider": config.get("llm_provider", "sensenova"),
            "sensenova_model": config.get("sensenova_model", ""),
            "openai_model": config.get("openai_model", ""),
            "cefr": config.get("cefr", "A2"),
        },
    }


@router.post("/api/scripts/form_memory")
async def api_scripts_form_memory(request: Request):
    """Persist the user's last-selected provider/model on the scripts page."""
    data = await request.json()
    _remember_scripts_form(str(data.get("provider", "")), str(data.get("model", "")))
    return {"ok": True}


@router.get("/api/scripts")
async def api_scripts_list(structure: str = "", status: str = "", q: str = ""):
    return {"scripts": script_library.list_scripts(structure, status, q)}


# 注意：必须注册在 /api/scripts/{sid} 之前，否则 "batch_status" 会被当作 sid 匹配
@router.get("/api/scripts/batch_status")
async def api_scripts_batch_status():
    """后台批量任务状态（生成/审查/修复共用）— 页面刷新后恢复进度显示。"""
    return script_library.batch_status()


@router.get("/api/scripts/{sid}")
async def api_script_get(sid: str):
    doc = script_library.get_script_doc(sid)
    if not doc:
        return JSONResponse({"error": "脚本不存在"}, status_code=404)
    return doc


@router.put("/api/scripts/{sid}")
async def api_script_update(sid: str, request: Request):
    data = await request.json()
    doc = script_library.update_script(sid, data)
    if not doc:
        return JSONResponse({"error": "脚本不存在"}, status_code=404)
    return {"ok": True, "meta": script_library.doc_meta(doc)}


@router.delete("/api/scripts/{sid}")
async def api_script_delete(sid: str):
    return {"ok": script_library.delete_script(sid)}


@router.post("/api/scripts/{sid}/reset_used")
async def api_script_reset_used(sid: str):
    doc = script_library.reset_used(sid)
    if not doc:
        return JSONResponse({"error": "脚本不存在"}, status_code=404)
    return {"ok": True, "meta": script_library.doc_meta(doc)}


@router.post("/api/scripts/generate")
async def api_scripts_generate(request: Request):
    """SSE: batch-generate scripts for the given topics."""
    data = await request.json()
    structure = data.get("structure", "original")
    if structure not in ("original", "original_static", "original_cutout", "quest"):
        return JSONResponse({"error": f"未知模式: {structure}"}, status_code=400)

    topics = [str(t).strip() for t in data.get("topics", []) if str(t).strip()]
    try:
        random_count = max(0, min(int(data.get("random_count", 0) or 0), 30))
    except (TypeError, ValueError):
        random_count = 0
    # 服务端随机抽取仅 API 直调可用；Web 页面在前端抽取（恒传 random_count:0）
    if random_count > 0:
        import random as _random
        config = load_config()
        topics_data = _load_topics_data(config)
        # 各模式独立排除：只排除当前模式已用的主题（同主题可在其他模式再用）
        used_mode = set(script_library.used_topics_for_mode(structure))
        pool = [t for ts in topics_data.values() for t in ts
                if t not in used_mode and t not in topics]
        _random.shuffle(pool)
        topics += pool[:random_count]

    if not topics:
        return JSONResponse(
            {"error": "未选择主题（主题库也没有可随机抽取的未用主题）"}, status_code=400)

    params = {
        "provider": data.get("provider", ""),
        "model": data.get("model", ""),
        "structure": structure,
        "cefr": data.get("cefr", "A2"),
        "num_lines": data.get("num_lines", ""),
        "topics": topics,
    }
    _remember_scripts_form(data.get("provider", ""), data.get("model", ""))

    import queue as _queue
    q: _queue.Queue = _queue.Queue()
    try:
        script_library.start_generate_thread(params, q)
    except RuntimeError as e:
        return JSONResponse({"error": str(e)}, status_code=400)

    async def event_stream():
        yield _sse({"type": "progress",
                    "message": f"开始批量生成 {len(topics)} 个脚本（串行执行）..."})
        while True:
            item = await asyncio.to_thread(q.get, True, None)
            if item is None:
                break
            kind, payload = item
            if kind == "progress":
                yield _sse({"type": "progress", "message": payload})
            elif kind == "script":
                yield _sse({"type": "script", "data": payload})
            elif kind == "error_item":
                yield _sse({"type": "error_item", "data": payload})
            elif kind == "done":
                yield _sse({"type": "done", "data": payload})
                break
            else:  # fatal
                yield _sse({"type": "error", "error": payload})
                break

    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers=_SSE_HEADERS)


@router.post("/api/scripts/generate/stop")
async def api_scripts_generate_stop():
    script_library.request_stop_batch()
    return {"ok": True}


@router.post("/api/scripts/review")
async def api_scripts_review(request: Request):
    """SSE: batch AI review of library scripts."""
    data = await request.json()
    ids = data.get("ids", [])
    if not isinstance(ids, list) or not ids:
        return JSONResponse({"error": "ids 不能为空"}, status_code=400)
    provider = data.get("provider", "")
    model = data.get("model", "")

    import queue as _queue
    q: _queue.Queue = _queue.Queue()
    try:
        script_library.start_review_thread(ids, provider, model, q)
    except RuntimeError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    _remember_scripts_form(provider, model)

    async def event_stream():
        yield _sse({"type": "progress",
                    "message": f"开始 AI 审查 {len(ids)} 个脚本..."})
        while True:
            item = await asyncio.to_thread(q.get, True, None)
            if item is None:
                break
            kind, payload = item
            if kind == "progress":
                yield _sse({"type": "progress", "message": payload})
            elif kind == "reviewed":
                yield _sse({"type": "reviewed", "data": payload})
            elif kind == "error_item":
                yield _sse({"type": "error_item", "data": payload})
            elif kind == "done":
                yield _sse({"type": "done", "data": payload})
                break
            else:  # fatal
                yield _sse({"type": "error", "error": payload})
                break

    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers=_SSE_HEADERS)


@router.post("/api/scripts/fix")
async def api_script_fix(request: Request):
    """SSE: fix SELECTED issues on one script via LLM patch, then re-review."""
    data = await request.json()
    sid = str(data.get("id", "")).strip()
    issues = data.get("issues", [])
    if not sid or not isinstance(issues, list) or not issues:
        return JSONResponse({"error": "缺少脚本 id 或未勾选任何问题"}, status_code=400)
    provider = data.get("provider", "")
    model = data.get("model", "")
    re_review = bool(data.get("re_review", True))

    import queue as _queue
    q: _queue.Queue = _queue.Queue()
    try:
        script_library.start_fix_thread(sid, issues, provider, model, re_review, q)
    except RuntimeError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    _remember_scripts_form(provider, model)

    async def event_stream():
        yield _sse({"type": "progress", "message": "开始修复选中的问题..."})
        while True:
            item = await asyncio.to_thread(q.get, True, None)
            if item is None:
                break
            kind, payload = item
            if kind == "progress":
                yield _sse({"type": "progress", "message": payload})
            elif kind == "fixed":
                yield _sse({"type": "fixed", "data": payload})
            elif kind == "reviewed":
                yield _sse({"type": "reviewed", "data": payload})
            elif kind == "done":
                yield _sse({"type": "done", "data": payload})
                break
            else:  # fatal
                yield _sse({"type": "error", "error": payload})
                break

    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers=_SSE_HEADERS)
