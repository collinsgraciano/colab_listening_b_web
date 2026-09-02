"""Topics API + Topics AI — 主题库 CRUD / 随机抽取 / AI 生成与审查."""
import asyncio
import json
import threading
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..config_manager import load_config
from ..sse import sse_line as _sse, SSE_HEADERS as _SSE_HEADERS
from ..topics_io import (
    load_topics_data as _load_topics_data,
    load_used_topic_names as _load_used_topic_names,
    save_topics_data as _save_topics_data,
    save_used_topics as _save_used_topics,
)
from .. import topics_ai

router = APIRouter()


@router.get("/api/topics")
async def api_get_topics():
    config = load_config()
    topics_file = config.get("topics_file", "")
    if not topics_file or not Path(topics_file).exists():
        return {"topics": {}}
    try:
        data = json.loads(Path(topics_file).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"topics": {}}
    return {"topics": data}


@router.post("/api/topics/add")
async def api_add_topic(request: Request):
    data = await request.json()
    config = load_config()
    topics_file = config.get("topics_file", "")
    if not topics_file:
        return JSONResponse({"ok": False, "error": "未配置主题库文件"}, status_code=400)

    try:
        topics_data = _load_topics_data(config)
    except (json.JSONDecodeError, OSError):
        topics_data = {}

    category = data.get("category", "").strip()
    topic = data.get("topic", "").strip()
    if not category or not topic:
        return JSONResponse({"ok": False, "error": "分类和主题不能为空"}, status_code=400)

    # 跨分类去重检查（归一化比较）
    norm_topic = topics_ai._norm(topic)
    for cat, topics in topics_data.items():
        for existing in topics:
            if topics_ai._norm(existing) == norm_topic:
                return JSONResponse({"ok": False, "error": f"主题已存在于分类 [{cat}]"}, status_code=400)

    if category not in topics_data:
        topics_data[category] = []
    if topic not in topics_data[category]:
        topics_data[category].append(topic)

    _save_topics_data(topics_file, topics_data)
    return {"ok": True, "topics": topics_data}


@router.delete("/api/topics/{category}/{index}")
async def api_delete_topic(category: str, index: int):
    config = load_config()
    topics_file = config.get("topics_file", "")
    if not topics_file or not Path(topics_file).exists():
        return JSONResponse({"ok": False, "error": "主题库不存在"}, status_code=404)

    try:
        topics_data = json.loads(Path(topics_file).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return JSONResponse({"ok": False, "error": "主题库文件损坏"}, status_code=500)

    if category in topics_data and 0 <= index < len(topics_data[category]):
        topics_data[category].pop(index)
        _save_topics_data(topics_file, topics_data)
        return {"ok": True, "topics": topics_data}
    return JSONResponse({"ok": False, "error": "未找到"}, status_code=404)


@router.post("/api/topics/add_category")
async def api_add_category(request: Request):
    data = await request.json()
    config = load_config()
    topics_file = config.get("topics_file", "")
    if not topics_file:
        return JSONResponse({"ok": False, "error": "未配置主题库文件"}, status_code=400)

    try:
        topics_data = _load_topics_data(config)
    except (json.JSONDecodeError, OSError):
        topics_data = {}

    category = data.get("category", "").strip()
    if not category:
        return JSONResponse({"ok": False, "error": "分类名不能为空"}, status_code=400)
    if category not in topics_data:
        topics_data[category] = []
        _save_topics_data(topics_file, topics_data)
    return {"ok": True, "topics": topics_data}


@router.get("/api/topics/random")
async def api_random_topic(request: Request):
    """Pick a random topic from topics.json (excluding used).

    ?mark=true will mark the topic as used (atomic write).
    """
    config = load_config()
    topics_file = config.get("topics_file", "")
    if not topics_file or not Path(topics_file).exists():
        return JSONResponse({"ok": False, "error": "主题库不存在"}, status_code=404)

    try:
        topics_data = json.loads(Path(topics_file).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return JSONResponse({"ok": False, "error": "主题库文件损坏"}, status_code=500)

    used_file = config.get("used_topics_file", "")
    if not used_file:
        output_dir = config.get("output_dir", "./output")
        used_file = str(Path(output_dir) / "used_topics.json")
    try:
        used = json.loads(Path(used_file).read_text(encoding="utf-8")) if Path(used_file).exists() else {}
    except (json.JSONDecodeError, OSError):
        used = {}

    all_topics = []
    for cat, topics in topics_data.items():
        for t in topics:
            if t not in used:
                all_topics.append(t)
    if not all_topics:
        return {"ok": False, "error": "没有可用主题（所有主题均已使用，请重置已用列表或添加新主题）"}

    import random
    from datetime import datetime as _dt
    chosen = random.choice(all_topics)

    mark = request.query_params.get("mark", "").lower() in ("1", "true", "yes")
    if mark:
        used[chosen] = {"used_at": _dt.now().strftime("%Y-%m-%d %H:%M:%S")}
        _save_used_topics(used_file, used)

    return {"ok": True, "topic": chosen}


@router.get("/api/topics/used")
async def api_used_topics():
    config = load_config()
    used_file = config.get("used_topics_file", "")
    if not used_file:
        output_dir = config.get("output_dir", "./output")
        used_file = str(Path(output_dir) / "used_topics.json")
    if not Path(used_file).exists():
        return {"used": []}
    try:
        data = json.loads(Path(used_file).read_text(encoding="utf-8"))
        return {"used": data}
    except (json.JSONDecodeError, OSError):
        return {"used": []}


@router.post("/api/topics/used/reset")
async def api_reset_used():
    config = load_config()
    used_file = config.get("used_topics_file", "")
    if not used_file:
        output_dir = config.get("output_dir", "./output")
        used_file = str(Path(output_dir) / "used_topics.json")
    if Path(used_file).exists():
        Path(used_file).unlink()
    return {"ok": True}


@router.post("/api/topics/ai/generate")
async def api_topics_ai_generate(request: Request):
    """SSE: AI-generate new topics for a category (or suggest a new category)."""
    data = await request.json()
    mode = data.get("mode", "category")
    category = str(data.get("category", "")).strip()
    try:
        count = max(1, min(int(data.get("count", 10) or 10), 30))
    except (TypeError, ValueError):
        count = 10
    hint = str(data.get("hint", "")).strip()[:500]

    config = load_config()
    topics_data = _load_topics_data(config)
    used = _load_used_topic_names(config)

    async def event_stream():
        try:
            yield _sse({"type": "progress", "message": "正在调用 AI 生成话题，请稍候..."})
            if mode == "new_category":
                result = await asyncio.to_thread(
                    topics_ai.suggest_category, count, topics_data, used, hint)
            else:
                if not category or category not in topics_data:
                    yield _sse({"type": "error", "error": f"分类不存在: {category}"})
                    return
                result = await asyncio.to_thread(
                    topics_ai.generate_topics, category, count, topics_data, used, hint)
            if not result.get("topics"):
                yield _sse({"type": "error",
                            "error": "AI 未返回有效话题（可能与现有话题重复），请重试或在补充要求中调整方向"})
                return
            yield _sse({"type": "result", "data": result})
        except Exception as e:
            yield _sse({"type": "error", "error": str(e)[:300]})

    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=_SSE_HEADERS)


@router.post("/api/topics/ai/suggest_categories")
async def api_topics_ai_suggest_categories(request: Request):
    """AI 推荐若干候选新分类（仅名称+理由），供「+ 新分类」表单选择。"""
    data = await request.json()
    try:
        count = max(1, min(int(data.get("count", 5) or 5), 10))
    except (TypeError, ValueError):
        count = 5
    config = load_config()
    topics_data = _load_topics_data(config)
    if not topics_data:
        return JSONResponse({"error": "topics.json 不存在或为空"}, status_code=400)
    try:
        result = await asyncio.to_thread(topics_ai.suggest_categories, count, topics_data)
    except Exception as e:
        return JSONResponse({"error": str(e)[:300]}, status_code=502)
    if not result.get("categories"):
        return JSONResponse({"error": "AI 未返回有效分类建议，请重试"}, status_code=502)
    return {"ok": True, **result}


@router.post("/api/topics/ai/review")
async def api_topics_ai_review():
    """SSE: full-library audit — local duplicate check + AI batched review."""
    config = load_config()
    topics_data = _load_topics_data(config)
    if not topics_data:
        return JSONResponse({"error": "topics.json 不存在或为空"}, status_code=400)

    import queue as _queue
    q: _queue.Queue = _queue.Queue()

    def run():
        try:
            def cb(i, n):
                q.put(("progress", f"正在审查批次 {i}/{n}（AI 语义分析）..."))
            result = topics_ai.review_topics(topics_data, progress_cb=cb)
            q.put(("result", result))
        except Exception as e:
            q.put(("error", str(e)[:300]))
        finally:
            q.put(None)

    threading.Thread(target=run, daemon=True).start()

    async def event_stream():
        yield _sse({"type": "progress", "message": "开始审查主题库（先做本地去重检查）..."})
        while True:
            item = await asyncio.to_thread(q.get, True, None)
            if item is None:
                break
            kind, payload = item
            if kind == "progress":
                yield _sse({"type": "progress", "message": payload})
            elif kind == "result":
                yield _sse({"type": "result", "data": payload})
                break
            else:
                yield _sse({"type": "error", "error": payload})
                break

    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=_SSE_HEADERS)


@router.post("/api/topics/ai/apply")
async def api_topics_ai_apply(request: Request):
    """Apply review suggestions: [{"action": remove|rename, "category", "topic", "new_topic"?}]."""
    data = await request.json()
    actions = data.get("actions", [])
    if not isinstance(actions, list) or not actions:
        return JSONResponse({"error": "actions 不能为空"}, status_code=400)
    config = load_config()
    topics_file = config.get("topics_file", "")
    if not topics_file or not Path(topics_file).exists():
        return JSONResponse({"error": "topics.json 不存在"}, status_code=400)
    result = await asyncio.to_thread(topics_ai.apply_suggestions, topics_file, actions)
    return {"ok": True, **result}


@router.post("/api/topics/update")
async def api_topics_update(request: Request):
    """Rename a single topic in place (category + exact topic string match)."""
    data = await request.json()
    category = data.get("category", "")
    topic = data.get("topic", "")
    new_topic = str(data.get("new_topic", "")).strip()
    if not category or not topic or not new_topic:
        return JSONResponse({"error": "缺少参数"}, status_code=400)
    config = load_config()
    topics_file = config.get("topics_file", "")
    if not topics_file or not Path(topics_file).exists():
        return JSONResponse({"error": "topics.json 不存在"}, status_code=400)
    result = await asyncio.to_thread(
        topics_ai.apply_suggestions, topics_file,
        [{"action": "rename", "category": category, "topic": topic, "new_topic": new_topic}])
    if result["applied"] > 0:
        return {"ok": True, "topics": result["topics"]}
    reason = result["skipped"][0] if result["skipped"] else "未知错误"
    return JSONResponse({"error": f"重命名失败: {reason}"}, status_code=400)
