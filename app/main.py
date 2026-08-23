"""FastAPI main application — all routes and API endpoints."""
import asyncio
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path

from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import (
    HTMLResponse, JSONResponse, StreamingResponse,
    FileResponse, RedirectResponse, PlainTextResponse,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Suppress noisy Windows ConnectionResetError on video stream disconnect
logging.getLogger("asyncio").setLevel(logging.CRITICAL)

from .config_manager import (
    PARAM_SPEC, GROUP_META, get_default_config,
    load_config, save_config,
    list_presets, save_preset, load_preset, delete_preset,
    build_cli_args, detect_local_mcp_token,
    load_llm_providers, save_llm_providers,
    get_provider_options, resolve_provider,
)
from .pipeline_service import get_service, STEP_ORDER
from .config_manager import detect_local_mcp_token
from . import topics_ai

# Paths
WEB_ROOT = Path(__file__).parent.parent.resolve()
TEMPLATES_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"
PIPELINE_DIR = WEB_ROOT / "pipeline"

# FastAPI app
app = FastAPI(title="Listening Video Generator")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


# Custom Jinja2 filter for date formatting
def _datestr(ts):
    if not ts:
        return ""
    try:
        from datetime import datetime
        d = datetime.fromtimestamp(float(ts))
        now = datetime.now()
        diff = (now - d).total_seconds()
        if diff < 3600:
            return f"{int(diff/60)}分钟前"
        elif diff < 86400:
            return f"{int(diff/3600)}小时前"
        elif diff < 604800:
            return f"{int(diff/86400)}天前"
        else:
            return d.strftime("%m-%d")
    except (ValueError, TypeError):
        return ""


templates.env.filters["datestr"] = _datestr


# ===========================================================================
# Page routes
# ===========================================================================

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    service = get_service()
    config = load_config()
    return templates.TemplateResponse(request, "dashboard.html", {
        "config": config,
        "runner": service,
        "active_page": "dashboard",
    })


@app.get("/config", response_class=HTMLResponse)
async def config_page(request: Request):
    config = load_config()
    presets = list_presets()
    # Inject dynamic LLM provider options into PARAM_SPEC
    PARAM_SPEC["llm_provider"]["options"] = get_provider_options()
    # Group params by group
    grouped = {}
    for key, spec in PARAM_SPEC.items():
        g = spec["group"]
        if g not in grouped:
            grouped[g] = []
        grouped[g].append((key, spec, config.get(key, spec["default"])))
    # Sort groups by order
    sorted_groups = sorted(grouped.items(), key=lambda x: GROUP_META.get(x[0], {}).get("order", 99))
    return templates.TemplateResponse(request, "config.html", {
        "config": config,
        "params": PARAM_SPEC,
        "grouped": sorted_groups,
        "group_meta": GROUP_META,
        "presets": presets,
        "active_page": "config",
    })


@app.get("/topics", response_class=HTMLResponse)
async def topics_page(request: Request):
    config = load_config()
    topics_file = config.get("topics_file", "")
    topics_data = {}
    if topics_file and Path(topics_file).exists():
        try:
            topics_data = json.loads(Path(topics_file).read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass

    used_file = config.get("used_topics_file", "")
    if not used_file:
        output_dir = config.get("output_dir", "./output")
        used_file = str(Path(output_dir) / "used_topics.json")
    used_topics = []
    if Path(used_file).exists():
        try:
            used_topics = json.loads(Path(used_file).read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass

    return templates.TemplateResponse(request, "topics.html", {
        "topics_data": topics_data,
        "used_topics": used_topics,
        "topics_file": topics_file,
        "active_page": "topics",
    })


@app.get("/runs/{name}/gallery", response_class=HTMLResponse)
async def gallery_page(request: Request, name: str):
    config = load_config()
    output_dir = Path(config.get("output_dir", "./output"))
    run_dir = output_dir / name
    script_path = run_dir / "script.json"
    script = {}
    if script_path.exists():
        try:
            script = json.loads(script_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    # YouTube metadata: prefer final youtube_metadata.json (real chapter timestamps
    # + hashtags injected in step 4.5), fallback to raw script fields
    yt_meta = {}
    yt_meta_path = run_dir / "youtube_metadata.json"
    if yt_meta_path.exists():
        try:
            yt_meta = json.loads(yt_meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    images_dir = run_dir / "images"
    images = sorted([f.name for f in images_dir.glob("*.png")]) if images_dir.exists() else []
    clips_dir = run_dir / "clips"
    clips = sorted([f.name for f in clips_dir.glob("*.mp4")]) if clips_dir.exists() else []
    audio_dir = run_dir / "audio"
    audio = sorted([f.name for f in audio_dir.glob("*.mp3")]) if audio_dir.exists() else []

    # Final videos are in work_dir root (not videos/ subdir which has intermediates)
    videos = []
    for v in sorted(run_dir.glob("*.mp4")):
        # Skip intermediate files
        if v.name.startswith("final_no_sub") or v.name.startswith("final_video_norm"):
            continue
        videos.append(v.name)
    # Also check videos/ dir for any extras
    videos_subdir = run_dir / "videos"
    if videos_subdir.exists():
        for v in sorted(videos_subdir.glob("*.mp4")):
            if v.name not in videos and not v.name.startswith("final_no_sub") and not v.name.startswith("final_video_norm"):
                videos.append(v.name)

    return templates.TemplateResponse(request, "gallery.html", {
        "run_name": name,
        "script": script,
        "images": images,
        "clips": clips,
        "audio": audio,
        "videos": videos,
        "has_yt_meta": bool(yt_meta),
        "yt_title": yt_meta.get("title") or script.get("youtube_title", ""),
        "yt_title_en": yt_meta.get("title_en") or script.get("youtube_title_en", ""),
        "yt_desc": yt_meta.get("description") or script.get("youtube_description", ""),
        "yt_desc_en": yt_meta.get("description_en") or script.get("youtube_description_en", ""),
        "yt_tags": yt_meta.get("tags") or script.get("youtube_tags", []),
        "active_page": "runs",
    })


@app.get("/runs", response_class=HTMLResponse)
async def runs_page(request: Request):
    config = load_config()
    output_dir = Path(config.get("output_dir", "./output"))
    runs = []
    if output_dir.exists():
        for d in sorted(output_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
            if not d.is_dir():
                continue
            script_path = d / "script.json"
            videos_dir = d / "videos"
            thumbnail = d / "thumbnail.jpg"
            run_info = {
                "name": d.name,
                "path": str(d),
                "created": d.stat().st_mtime,
                "has_script": script_path.exists(),
                "has_thumbnail": thumbnail.exists(),
                "thumbnail_url": f"/api/runs/{d.name}/thumbnail" if thumbnail.exists() else "",
            }
            # Find video files — final videos are in work_dir root, not videos/
            video_files = []
            for v in d.glob("*.mp4"):
                if v.name.startswith("final_no_sub") or v.name.startswith("final_video_norm"):
                    continue
                video_files.append({
                    "name": v.name,
                    "size_mb": round(v.stat().st_size / (1024*1024), 1),
                    "url": f"/api/runs/{d.name}/video/{v.name}",
                })
            # Also check videos/ subdir for intermediates (but don't show them as main)
            run_info["videos"] = video_files
            # Load script metadata
            if script_path.exists():
                try:
                    script = json.loads(script_path.read_text(encoding="utf-8"))
                    run_info["title"] = script.get("youtube_title", script.get("title", d.name))
                    run_info["title_en"] = script.get("youtube_title_en", "")
                    run_info["cefr"] = script.get("cefr", "")
                    run_info["structure"] = script.get("structure", "")
                except (json.JSONDecodeError, OSError):
                    run_info["title"] = d.name
            else:
                run_info["title"] = d.name
            runs.append(run_info)

    return templates.TemplateResponse(request, "runs.html", {
        "runs": runs,
        "active_page": "runs",
    })


# ===========================================================================
# Config API
# ===========================================================================

@app.post("/api/config/save")
async def api_save_config(request: Request):
    data = await request.json()
    config = load_config()
    config.update(data)
    save_config(config)
    return {"ok": True}


@app.post("/api/config/save_all")
async def api_save_all_config(request: Request):
    data = await request.json()
    save_config(data)
    return {"ok": True}


@app.get("/api/config")
async def api_get_config():
    return load_config()


@app.get("/api/config/defaults")
async def api_get_defaults():
    return get_default_config()


@app.post("/api/config/preset/save")
async def api_save_preset(request: Request):
    data = await request.json()
    name = data.get("name", "")
    config = data.get("config", {})
    if not name:
        return JSONResponse({"ok": False, "error": "名称不能为空"}, status_code=400)
    save_preset(name, config)
    return {"ok": True, "name": name}


@app.get("/api/config/preset/load/{name}")
async def api_load_preset(name: str):
    try:
        return load_preset(name)
    except FileNotFoundError:
        return JSONResponse({"error": "Preset not found"}, status_code=404)


@app.delete("/api/config/preset/{name}")
async def api_delete_preset(name: str):
    delete_preset(name)
    return {"ok": True}


@app.get("/api/config/presets")
async def api_list_presets():
    return {"presets": list_presets()}


# ===========================================================================
# Pipeline Run API
# ===========================================================================

@app.post("/api/run/start")
async def api_run_start(request: Request):
    service = get_service()
    data = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
    config = data.get("config") or load_config()
    resume = data.get("resume", False)
    step_mode = data.get("step_mode", False)
    ok = service.start(config, resume=resume, step_mode=step_mode)
    return {"ok": ok, "status": service.status}


@app.post("/api/run/continue")
async def api_run_continue():
    """Continue to next step in step mode."""
    service = get_service()
    service.continue_step()
    return {"ok": True, "status": service.status}


@app.post("/api/run/stop")
async def api_run_stop():
    service = get_service()
    service.stop()
    return {"ok": True, "status": service.status}


@app.get("/api/run/status")
async def api_run_status():
    service = get_service()
    return service.get_progress()


@app.get("/api/run/logs")
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


@app.get("/api/run/logs/since/{since}")
async def api_run_logs_since(since: int):
    """Get logs since a given index (for non-SSE polling)."""
    service = get_service()
    logs = service.get_logs_since(since)
    total = len(service.log_lines)
    return {"logs": logs, "total": total, "status": service.get_progress()}


# ===========================================================================
# Topics API
# ===========================================================================

@app.get("/api/topics")
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


@app.post("/api/topics/add")
async def api_add_topic(request: Request):
    data = await request.json()
    config = load_config()
    topics_file = config.get("topics_file", "")
    if not topics_file:
        return JSONResponse({"ok": False, "error": "未配置主题库文件"}, status_code=400)

    topics_data = {}
    if Path(topics_file).exists():
        topics_data = json.loads(Path(topics_file).read_text(encoding="utf-8"))

    category = data.get("category", "").strip()
    topic = data.get("topic", "").strip()
    if not category or not topic:
        return JSONResponse({"ok": False, "error": "分类和主题不能为空"}, status_code=400)

    if category not in topics_data:
        topics_data[category] = []
    if topic not in topics_data[category]:
        topics_data[category].append(topic)

    Path(topics_file).write_text(
        json.dumps(topics_data, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"ok": True, "topics": topics_data}


@app.delete("/api/topics/{category}/{index}")
async def api_delete_topic(category: str, index: int):
    config = load_config()
    topics_file = config.get("topics_file", "")
    if not topics_file or not Path(topics_file).exists():
        return JSONResponse({"ok": False, "error": "主题库不存在"}, status_code=404)

    topics_data = json.loads(Path(topics_file).read_text(encoding="utf-8"))
    if category in topics_data and 0 <= index < len(topics_data[category]):
        topics_data[category].pop(index)
        Path(topics_file).write_text(
            json.dumps(topics_data, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"ok": True, "topics": topics_data}
    return JSONResponse({"ok": False, "error": "未找到"}, status_code=404)


@app.post("/api/topics/add_category")
async def api_add_category(request: Request):
    data = await request.json()
    config = load_config()
    topics_file = config.get("topics_file", "")
    if not topics_file:
        return JSONResponse({"ok": False, "error": "未配置主题库文件"}, status_code=400)

    topics_data = {}
    if Path(topics_file).exists():
        topics_data = json.loads(Path(topics_file).read_text(encoding="utf-8"))

    category = data.get("category", "").strip()
    if not category:
        return JSONResponse({"ok": False, "error": "分类名不能为空"}, status_code=400)
    if category not in topics_data:
        topics_data[category] = []
        Path(topics_file).write_text(
            json.dumps(topics_data, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"ok": True, "topics": topics_data}


@app.get("/api/topics/random")
async def api_random_topic():
    """Pick a random topic from topics.json (excluding used)."""
    config = load_config()
    topics_file = config.get("topics_file", "")
    if not topics_file or not Path(topics_file).exists():
        return JSONResponse({"ok": False, "error": "主题库不存在"}, status_code=404)

    import random
    topics_data = json.loads(Path(topics_file).read_text(encoding="utf-8"))
    used_file = config.get("used_topics_file", "")
    if not used_file:
        output_dir = config.get("output_dir", "./output")
        used_file = str(Path(output_dir) / "used_topics.json")
    used = []
    if Path(used_file).exists():
        used = json.loads(Path(used_file).read_text(encoding="utf-8"))

    all_topics = []
    for cat, topics in topics_data.items():
        for t in topics:
            if t not in used:
                all_topics.append(t)
    if not all_topics:
        return {"ok": False, "error": "没有可用主题"}
    return {"ok": True, "topic": random.choice(all_topics)}


@app.get("/api/topics/used")
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


@app.post("/api/topics/used/reset")
async def api_reset_used():
    config = load_config()
    used_file = config.get("used_topics_file", "")
    if not used_file:
        output_dir = config.get("output_dir", "./output")
        used_file = str(Path(output_dir) / "used_topics.json")
    if Path(used_file).exists():
        Path(used_file).unlink()
    return {"ok": True}


# ===========================================================================
# Topics AI — generation & review
# ===========================================================================

def _load_topics_data(config: dict) -> dict:
    """Load topics.json → {category: [topics]} (empty dict on missing/broken)."""
    topics_file = config.get("topics_file", "")
    if topics_file and Path(topics_file).exists():
        try:
            return json.loads(Path(topics_file).read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _load_used_topic_names(config: dict) -> list[str]:
    """Load used_topics.json → list of topic names."""
    used_file = config.get("used_topics_file", "")
    if not used_file:
        output_dir = config.get("output_dir", "./output")
        used_file = str(Path(output_dir) / "used_topics.json")
    if Path(used_file).exists():
        try:
            return list(json.loads(Path(used_file).read_text(encoding="utf-8")).keys())
        except (json.JSONDecodeError, OSError):
            return []
    return []


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


@app.post("/api/topics/ai/generate")
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
                    topics_ai.suggest_category, count, topics_data, hint)
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


@app.post("/api/topics/ai/review")
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


@app.post("/api/topics/ai/apply")
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


@app.post("/api/topics/update")
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


# ===========================================================================
# Runs API
# ===========================================================================

@app.get("/api/runs")
async def api_list_runs():
    config = load_config()
    output_dir = Path(config.get("output_dir", "./output"))
    runs = []
    if output_dir.exists():
        for d in sorted(output_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
            if not d.is_dir():
                continue
            script_path = d / "script.json"
            videos_dir = d / "videos"
            run_info = {"name": d.name, "created": d.stat().st_mtime}
            if script_path.exists():
                try:
                    script = json.loads(script_path.read_text(encoding="utf-8"))
                    run_info["title"] = script.get("youtube_title", d.name)
                except (json.JSONDecodeError, OSError):
                    run_info["title"] = d.name
            else:
                run_info["title"] = d.name
            run_info["videos"] = []
            for v in d.glob("*.mp4"):
                if v.name.startswith("final_no_sub") or v.name.startswith("final_video_norm"):
                    continue
                run_info["videos"].append({
                    "name": v.name,
                    "size_mb": round(v.stat().st_size / (1024*1024), 1),
                })
            runs.append(run_info)
    return {"runs": runs}


@app.get("/api/runs/{name}/script")
async def api_get_script(name: str):
    config = load_config()
    output_dir = Path(config.get("output_dir", "./output"))
    script_path = output_dir / name / "script.json"
    if not script_path.exists():
        return JSONResponse({"error": "Not found"}, status_code=404)
    return json.loads(script_path.read_text(encoding="utf-8"))


@app.get("/api/runs/{name}/thumbnail")
async def api_get_thumbnail(name: str):
    config = load_config()
    output_dir = Path(config.get("output_dir", "./output"))
    thumb = output_dir / name / "thumbnail.jpg"
    if not thumb.exists():
        return JSONResponse({"error": "Not found"}, status_code=404)
    return FileResponse(str(thumb), media_type="image/jpeg")


@app.get("/api/runs/{name}/video/{video_name}")
async def api_get_video(name: str, video_name: str):
    config = load_config()
    output_dir = Path(config.get("output_dir", "./output"))
    # Final videos are in work_dir root; clips are in clips/ subdir
    video_path = output_dir / name / video_name
    if not video_path.exists():
        video_path = output_dir / name / "clips" / video_name
    if not video_path.exists():
        video_path = output_dir / name / "videos" / video_name
    if not video_path.exists():
        return JSONResponse({"error": "Not found"}, status_code=404)
    return FileResponse(str(video_path), media_type="video/mp4")


@app.get("/api/runs/{name}/images/{image_name}")
async def api_get_image(name: str, image_name: str):
    config = load_config()
    output_dir = Path(config.get("output_dir", "./output"))
    img_path = output_dir / name / "images" / image_name
    if not img_path.exists():
        return JSONResponse({"error": "Not found"}, status_code=404)
    return FileResponse(str(img_path), media_type="image/png")


@app.get("/api/runs/{name}/images_list")
async def api_list_images(name: str):
    config = load_config()
    output_dir = Path(config.get("output_dir", "./output"))
    images_dir = output_dir / name / "images"
    if not images_dir.exists():
        return {"images": []}
    images = sorted([f.name for f in images_dir.glob("*.png")])
    return {"images": images}


@app.delete("/api/runs/{name}")
async def api_delete_run(name: str):
    import shutil
    config = load_config()
    output_dir = Path(config.get("output_dir", "./output"))
    run_dir = output_dir / name
    if run_dir.exists() and run_dir.is_dir():
        # Safety: ensure it's within output_dir
        if str(run_dir.resolve()).startswith(str(output_dir.resolve())):
            shutil.rmtree(str(run_dir))
            return {"ok": True}
    return JSONResponse({"ok": False, "error": "Cannot delete"}, status_code=400)


# ===========================================================================
# MCP Token detection
# ===========================================================================

@app.get("/api/mcp/detect_token")
async def api_detect_token():
    token = detect_local_mcp_token()
    if token:
        # Mask for display but return full for use
        return {"ok": True, "token": token}
    return {"ok": False, "error": "未检测到本地 MCP Token"}


# ===========================================================================
# Character reuse API
# ===========================================================================

@app.get("/api/character_sources")
async def api_character_sources():
    """List available previous runs for character reuse, with images and all characters."""
    config = load_config()
    output_dir = Path(config.get("output_dir", "./output"))
    sources = []
    if output_dir.exists():
        for d in sorted(output_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
            if not d.is_dir():
                continue
            script_path = d / "script.json"
            if not script_path.exists():
                continue
            try:
                script = json.loads(script_path.read_text(encoding="utf-8"))
                img_dir = d / "images"

                # Auto-detect structure by checking which image files exist
                # (script.json does not store structure — it's a CLI arg)
                if (img_dir / "pose_char_a_0.png").exists():
                    structure = "quest"
                elif (img_dir / "char_scene.png").exists():
                    structure = "original"
                else:
                    continue  # no character images

                # Build character list based on detected structure
                if structure == "quest":
                    char_keys = ["char_a", "char_b", "char_c", "host"]
                    char_labels = {"char_a": "角色A", "char_b": "角色B", "char_c": "角色C", "host": "主持人"}
                    img_name_for = lambda key: f"pose_{key}_0.png"
                else:
                    char_keys = ["char_a", "char_b"]
                    char_labels = {"char_a": "角色A", "char_b": "角色B"}
                    img_name_for = lambda key: "char_scene.png"

                characters = []
                for key in char_keys:
                    desc = script.get(f"{key}_description", "")
                    gender = script.get(f"{key}_gender", "")
                    role = script.get(f"{key}_role", "")
                    qwen_speaker = script.get(f"{key}_qwen_speaker", "")
                    img_name = img_name_for(key)
                    img_exists = (img_dir / img_name).exists()
                    characters.append({
                        "key": key,
                        "label": char_labels.get(key, key),
                        "description": desc,
                        "gender": gender,
                        "role": role,
                        "qwen_speaker": qwen_speaker,
                        "image_url": f"/api/runs/{d.name}/images/{img_name}" if img_exists else "",
                    })

                sources.append({
                    "name": d.name,
                    "title": script.get("youtube_title", script.get("title", d.name)),
                    "structure": structure,
                    "characters": characters,
                })
            except (json.JSONDecodeError, OSError):
                continue
    return {"sources": sources}


# ===========================================================================
# Health check
# ===========================================================================

@app.get("/api/health")
async def health():
    return {"ok": True, "pipeline_dir": str(PIPELINE_DIR),
            "pipeline_exists": PIPELINE_DIR.exists()}


# ===========================================================================
# Script editor API
# ===========================================================================

@app.get("/api/runs/{name}/script/edit")
async def api_get_script_edit(name: str):
    """Get script for editing (returns full JSON)."""
    config = load_config()
    output_dir = Path(config.get("output_dir", "./output"))
    script_path = output_dir / name / "script.json"
    if not script_path.exists():
        return JSONResponse({"error": "Script not found"}, status_code=404)
    return json.loads(script_path.read_text(encoding="utf-8"))


@app.post("/api/runs/{name}/script/save")
async def api_save_script(name: str, request: Request):
    """Save edited script JSON."""
    config = load_config()
    output_dir = Path(config.get("output_dir", "./output"))
    script_path = output_dir / name / "script.json"
    if not script_path.exists():
        return JSONResponse({"error": "Script not found"}, status_code=404)
    data = await request.json()
    script_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"ok": True}


# ===========================================================================
# Image gallery API
# ===========================================================================

@app.get("/api/runs/{name}/gallery")
async def api_gallery(name: str):
    """List all images for a run, grouped by type."""
    config = load_config()
    output_dir = Path(config.get("output_dir", "./output"))
    images_dir = output_dir / name / "images"
    if not images_dir.exists():
        return {"images": [], "clips": []}
    
    images = sorted([f.name for f in images_dir.glob("*.png")])
    clips_dir = output_dir / name / "clips"
    clips = sorted([f.name for f in clips_dir.glob("*.mp4")]) if clips_dir.exists() else []
    audio_dir = output_dir / name / "audio"
    audio = sorted([f.name for f in audio_dir.glob("*.mp3")]) if audio_dir.exists() else []
    
    # Final videos in work_dir root (excluding intermediate files)
    run_dir = output_dir / name
    final_videos = []
    if run_dir.exists():
        for v in sorted(run_dir.glob("*.mp4")):
            if v.name.startswith("final_no_sub") or v.name.startswith("final_video_norm"):
                continue
            final_videos.append(v.name)

    return {
        "images": images,
        "clips": clips,
        "audio": audio,
        "final_videos": final_videos,
        "image_urls": {f: f"/api/runs/{name}/images/{f}" for f in images},
        "clip_urls": {f: f"/api/runs/{name}/video/{f}" for f in clips},
        "audio_urls": {f: f"/api/runs/{name}/audio/{f}" for f in audio},
        "final_video_urls": {f: f"/api/runs/{name}/video/{f}" for f in final_videos},
    }


@app.get("/api/runs/{name}/audio/{audio_name}")
async def api_get_audio(name: str, audio_name: str):
    config = load_config()
    output_dir = Path(config.get("output_dir", "./output"))
    audio_path = output_dir / name / "audio" / audio_name
    if not audio_path.exists():
        return JSONResponse({"error": "Not found"}, status_code=404)
    return FileResponse(str(audio_path), media_type="audio/mpeg")


@app.get("/api/runs/{name}/srt")
async def api_get_srt(name: str):
    """Serve the SRT subtitle file for step-mode review."""
    config = load_config()
    output_dir = Path(config.get("output_dir", "./output"))
    srt_path = output_dir / name / "subtitles" / "output.srt"
    if not srt_path.exists():
        return JSONResponse({"error": "SRT not found"}, status_code=404)
    return PlainTextResponse(srt_path.read_text(encoding="utf-8"), media_type="text/plain")


# ===========================================================================
# Character Library API
# ===========================================================================

LIBRARY_DIR = WEB_ROOT / "configs" / "character_library"


@app.get("/api/character_library")
async def api_library_list():
    """List all saved characters in the library."""
    if not LIBRARY_DIR.exists():
        return {"characters": []}
    chars = []
    for d in sorted(LIBRARY_DIR.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        meta_path = d / "meta.json"
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            thumb = d / "thumb.png"
            meta["image_url"] = f"/api/character_library/{d.name}/image" if thumb.exists() else ""
            chars.append(meta)
        except (json.JSONDecodeError, OSError):
            continue
    return {"characters": chars}


@app.post("/api/character_library/save")
async def api_library_save(request: Request):
    """Save a character from a run into the library."""
    import shutil
    data = await request.json()
    run_name = data.get("run_name", "")
    char_key = data.get("char_key", "")
    custom_name = data.get("name", "").strip()
    structure = data.get("structure", "quest")

    if not run_name or not char_key:
        return JSONResponse({"ok": False, "error": "缺少参数"}, status_code=400)

    config = load_config()
    output_dir = Path(config.get("output_dir", "./output"))
    run_dir = output_dir / run_name
    script_path = run_dir / "script.json"
    if not script_path.exists():
        return JSONResponse({"ok": False, "error": "运行不存在"}, status_code=404)

    script = json.loads(script_path.read_text(encoding="utf-8"))
    desc = script.get(f"{char_key}_description", "")
    gender = script.get(f"{char_key}_gender", "")
    role = script.get(f"{char_key}_role", "")
    if not desc:
        return JSONResponse({"ok": False, "error": "角色描述为空"}, status_code=400)

    # Generate ID
    lib_id = f"char_{int(time.time())}_{char_key}"
    lib_dir = LIBRARY_DIR / lib_id
    lib_dir.mkdir(parents=True, exist_ok=True)

    # Copy images
    src_img_dir = run_dir / "images"
    copied_files = []
    if structure == "quest":
        for j in range(8):
            src = src_img_dir / f"pose_{char_key}_{j}.png"
            if src.exists():
                shutil.copy2(str(src), str(lib_dir / src.name))
                copied_files.append(src.name)
        atlas = src_img_dir / f"pose_atlas_{char_key}.png"
        if atlas.exists():
            shutil.copy2(str(atlas), str(lib_dir / atlas.name))
        if char_key == "host":
            hb = src_img_dir / "host_bg.png"
            if hb.exists():
                shutil.copy2(str(hb), str(lib_dir / "host_bg.png"))
    else:
        cs = src_img_dir / "char_scene.png"
        if cs.exists():
            shutil.copy2(str(cs), str(lib_dir / "char_scene.png"))

    # Copy thumbnail (pose_0 or char_scene)
    thumb_src = src_img_dir / f"pose_{char_key}_0.png" if structure == "quest" else src_img_dir / "char_scene.png"
    if thumb_src.exists():
        shutil.copy2(str(thumb_src), str(lib_dir / "thumb.png"))

    # Save metadata
    meta = {
        "id": lib_id,
        "name": custom_name or f"{char_key} ({role})",
        "description": desc,
        "gender": gender,
        "structure": structure,
        "source_run": run_name,
        "source_key": char_key,
        "created": time.time(),
    }
    (lib_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    return {"ok": True, "id": lib_id, "meta": meta}


@app.delete("/api/character_library/{lib_id}")
async def api_library_delete(lib_id: str):
    import shutil
    lib_dir = LIBRARY_DIR / lib_id
    if lib_dir.exists() and lib_dir.is_dir():
        if str(lib_dir.resolve()).startswith(str(LIBRARY_DIR.resolve())):
            shutil.rmtree(str(lib_dir))
            return {"ok": True}
    return JSONResponse({"ok": False, "error": "未找到"}, status_code=404)


@app.get("/api/character_library/{lib_id}/image")
async def api_library_image(lib_id: str):
    thumb = LIBRARY_DIR / lib_id / "thumb.png"
    if not thumb.exists():
        return JSONResponse({"error": "Not found"}, status_code=404)
    return FileResponse(str(thumb), media_type="image/png")


# ===========================================================================
# QwenTTS Voice Management
# ===========================================================================

QWEN_VOICE_CONFIG_PATH = WEB_ROOT / "configs" / "qwen_voice_config.json"
CUSTOM_VOICES_DIR = WEB_ROOT / "configs" / "custom_voices"


def _load_qwen_voice_config() -> dict:
    """Load qwen_voice_config.json with defaults."""
    defaults = {
        "default_male": "Ryan",
        "default_female": "Vivian",
        "default_host_female": "Serena",
        "custom_voices": [],
    }
    if QWEN_VOICE_CONFIG_PATH.exists():
        try:
            saved = json.loads(QWEN_VOICE_CONFIG_PATH.read_text(encoding="utf-8"))
            for k in defaults:
                if k in saved:
                    defaults[k] = saved[k]
        except (json.JSONDecodeError, OSError):
            pass
    return defaults


def _save_qwen_voice_config(config: dict) -> None:
    QWEN_VOICE_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    QWEN_VOICE_CONFIG_PATH.write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


@app.get("/voices", response_class=HTMLResponse)
async def voices_page(request: Request):
    """QwenTTS voice management page."""
    config = load_config()
    # Fetch library characters
    library_chars = []
    if LIBRARY_DIR.exists():
        for d in sorted(LIBRARY_DIR.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
            meta_path = d / "meta.json"
            if not meta_path.exists():
                continue
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                thumb = d / "thumb.png"
                meta["image_url"] = f"/api/character_library/{d.name}/image" if thumb.exists() else ""
                library_chars.append(meta)
            except (json.JSONDecodeError, OSError):
                continue
    return templates.TemplateResponse(request, "voices.html", {
        "config": config,
        "library_chars": library_chars,
        "active_page": "voices",
    })


@app.get("/api/qwen_voices/speakers")
async def api_qwen_speakers():
    """List all available voices (preset + custom)."""
    import sys as _sys
    _pipeline = str(PIPELINE_DIR)
    if _pipeline not in _sys.path:
        _sys.path.insert(0, _pipeline)
    from qwen_tts_engine import QWEN_SPEAKERS, get_all_voices
    return {"speakers": get_all_voices(), "presets": QWEN_SPEAKERS}


@app.put("/api/character_library/{lib_id}/voice")
async def api_library_set_voice(lib_id: str, request: Request):
    """Set Qwen TTS speaker for a library character."""
    lib_dir = LIBRARY_DIR / lib_id
    if not lib_dir.exists():
        return JSONResponse({"ok": False, "error": "未找到"}, status_code=404)
    meta_path = lib_dir / "meta.json"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return JSONResponse({"ok": False, "error": "meta.json 读取失败"}, status_code=500)
    data = await request.json()
    speaker = data.get("qwen_speaker", "")
    meta["qwen_speaker"] = speaker
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"ok": True, "meta": meta}


@app.post("/api/qwen_voices/preview")
async def api_qwen_preview(request: Request):
    """Preview a voice by generating a short sample audio."""
    import sys as _sys
    import tempfile
    _pipeline = str(PIPELINE_DIR)
    if _pipeline not in _sys.path:
        _sys.path.insert(0, _pipeline)

    data = await request.json()
    speaker = data.get("speaker", "Vivian")
    language = data.get("language", "english")
    text = data.get("text", "Hello, this is a voice test.")

    config = load_config()
    model_path = config.get("qwen_model_path", r"H:\models\Qwen3-TTS-12Hz-0.6B-CustomVoice")
    base_model_path = config.get("qwen_base_model_path", r"H:\models\Qwen3-TTS-12Hz-1.7B-Base")
    device = config.get("qwen_device", "cuda:0")

    try:
        from qwen_tts_engine import QwenTTSEngine
        engine = QwenTTSEngine(model_path, device, base_model_path)
        # Use a temp file for the preview
        tmp_dir = Path(tempfile.mkdtemp())
        out_path = str(tmp_dir / "preview.mp3")
        if language == "chinese":
            engine.synth_chinese(text, speaker, out_path, rate="+0%")
        else:
            engine.synth_english(text, speaker, out_path, rate="+0%")
        return FileResponse(out_path, media_type="audio/mpeg",
                            filename="preview.mp3",
                            headers={"Content-Disposition": "attachment; filename=preview.mp3"})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.get("/api/qwen_voices/defaults")
async def api_qwen_defaults_get():
    """Get default voice configuration."""
    return _load_qwen_voice_config()


@app.put("/api/qwen_voices/defaults")
async def api_qwen_defaults_put(request: Request):
    """Update default voice configuration."""
    data = await request.json()
    config = _load_qwen_voice_config()
    for key in ["default_male", "default_female", "default_host_female"]:
        if key in data:
            config[key] = data[key]
    _save_qwen_voice_config(config)
    return {"ok": True, "config": config}


@app.post("/api/qwen_voices/custom")
async def api_qwen_custom_create(
    name: str = Form(""),
    gender: str = Form(""),
    language: str = Form("english"),
    ref_text: str = Form(""),
    ref_audio: UploadFile = File(...),
):
    """Create a custom cloned voice from reference audio."""
    if not name or not ref_text:
        return JSONResponse({"ok": False, "error": "缺少名称或参考文字"}, status_code=400)

    # Safe filename
    safe_name = "".join(c for c in name if c.isalnum() or c in "-_") or "custom"
    CUSTOM_VOICES_DIR.mkdir(parents=True, exist_ok=True)
    audio_path = CUSTOM_VOICES_DIR / f"{safe_name}.wav"
    content = await ref_audio.read()
    audio_path.write_bytes(content)

    config = _load_qwen_voice_config()
    # Remove existing entry with same name
    config["custom_voices"] = [v for v in config["custom_voices"] if v["name"] != name]
    config["custom_voices"].append({
        "name": name,
        "description": f"自定义克隆音色 ({gender})",
        "gender": gender,
        "language": language,
        "ref_audio": str(audio_path),
        "ref_text": ref_text,
        "created": time.time(),
    })
    _save_qwen_voice_config(config)
    return {"ok": True, "name": name}


@app.delete("/api/qwen_voices/custom/{name}")
async def api_qwen_custom_delete(name: str):
    """Delete a custom voice."""
    config = _load_qwen_voice_config()
    target = None
    for v in config["custom_voices"]:
        if v["name"] == name:
            target = v
            break
    if not target:
        return JSONResponse({"ok": False, "error": "未找到"}, status_code=404)
    # Delete audio file
    audio_path = Path(target.get("ref_audio", ""))
    if audio_path.exists():
        audio_path.unlink()
    # Remove from config
    config["custom_voices"] = [v for v in config["custom_voices"] if v["name"] != name]
    _save_qwen_voice_config(config)
    return {"ok": True}


# ===========================================================================
# AI Test — LLM Playground
# ===========================================================================

AI_TEST_CONFIG_PATH = WEB_ROOT / "configs" / "ai_test_config.json"


def _load_ai_test_config() -> dict:
    defaults = {"system_prompt": ""}
    if AI_TEST_CONFIG_PATH.exists():
        try:
            saved = json.loads(AI_TEST_CONFIG_PATH.read_text(encoding="utf-8"))
            for k in defaults:
                if k in saved:
                    defaults[k] = saved[k]
        except (json.JSONDecodeError, OSError):
            pass
    return defaults


def _save_ai_test_config(cfg: dict) -> None:
    AI_TEST_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    AI_TEST_CONFIG_PATH.write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")


@app.get("/ai_test", response_class=HTMLResponse)
async def ai_test_page(request: Request):
    config = load_config()
    ai_cfg = _load_ai_test_config()
    return templates.TemplateResponse(request, "ai_test.html", {
        "config": config,
        "active_page": "ai_test",
        "system_prompt": ai_cfg.get("system_prompt", ""),
    })


@app.get("/api/ai_test/config")
async def api_ai_test_config_get():
    config = load_config()
    ai_cfg = _load_ai_test_config()
    return {
        "llm_provider": config.get("llm_provider", "sensenova"),
        "sensenova_model": config.get("sensenova_model", "deepseek-v4-flash"),
        "openai_model": config.get("openai_model", "grok-4.6"),
        "openai_base_url": config.get("openai_base_url", ""),
        "system_prompt": ai_cfg.get("system_prompt", ""),
        "custom_providers": load_llm_providers(),
    }


@app.put("/api/ai_test/config")
async def api_ai_test_config_put(request: Request):
    data = await request.json()
    ai_cfg = _load_ai_test_config()
    if "system_prompt" in data:
        ai_cfg["system_prompt"] = data["system_prompt"]
    _save_ai_test_config(ai_cfg)
    return {"ok": True}


# --- Custom LLM Provider CRUD ---

@app.get("/api/ai_test/providers")
async def api_providers_list():
    return {"providers": load_llm_providers()}


@app.post("/api/ai_test/providers")
async def api_providers_create(request: Request):
    data = await request.json()
    name = (data.get("name") or "").strip()
    base_url = (data.get("base_url") or "").strip()
    if not name or not base_url:
        return JSONResponse({"ok": False, "error": "名称和 Base URL 不能为空"}, status_code=400)
    providers = load_llm_providers()
    # Check duplicate name
    if any(p["name"] == name for p in providers):
        return JSONResponse({"ok": False, "error": "名称已存在"}, status_code=400)
    provider = {
        "id": f"custom_{int(time.time()*1000)}",
        "name": name,
        "base_url": base_url,
        "api_key": (data.get("api_key") or "").strip(),
        "models": [m.strip() for m in (data.get("models") or "").split(",") if m.strip()],
    }
    providers.append(provider)
    save_llm_providers(providers)
    return {"ok": True, "provider": provider}


@app.put("/api/ai_test/providers/{provider_id}")
async def api_providers_update(provider_id: str, request: Request):
    data = await request.json()
    providers = load_llm_providers()
    target = None
    for p in providers:
        if p["id"] == provider_id:
            target = p
            break
    if not target:
        return JSONResponse({"ok": False, "error": "未找到"}, status_code=404)
    if "name" in data:
        new_name = data["name"].strip()
        if new_name and new_name != target["name"]:
            if any(p["name"] == new_name for p in providers if p["id"] != provider_id):
                return JSONResponse({"ok": False, "error": "名称已存在"}, status_code=400)
            target["name"] = new_name
    if "base_url" in data:
        target["base_url"] = data["base_url"].strip()
    if "api_key" in data:
        target["api_key"] = data["api_key"].strip()
    if "models" in data:
        if isinstance(data["models"], str):
            target["models"] = [m.strip() for m in data["models"].split(",") if m.strip()]
        else:
            target["models"] = data["models"]
    save_llm_providers(providers)
    return {"ok": True, "provider": target}


@app.delete("/api/ai_test/providers/{provider_id}")
async def api_providers_delete(provider_id: str):
    providers = load_llm_providers()
    providers = [p for p in providers if p["id"] != provider_id]
    save_llm_providers(providers)
    return {"ok": True}


@app.post("/api/ai_test/chat")
async def api_ai_test_chat(request: Request):
    """SSE streaming chat completion endpoint.

    Reads config for API keys, supports SenseNova and OpenAI-compatible providers.
    Returns text/event-stream with token-level streaming.
    """
    import urllib.request
    import urllib.error

    data = await request.json()
    messages = data.get("messages", [])
    system_prompt = data.get("system_prompt", "")
    provider = data.get("provider", "sensenova")
    model = data.get("model", "")
    temperature = float(data.get("temperature", 0.8))
    max_tokens = int(data.get("max_tokens", 8192))
    reasoning_effort = data.get("reasoning_effort", "low")
    api_key_override = data.get("api_key", "").strip()

    config = load_config()

    # Override config provider with the one selected in AI test page
    config["llm_provider"] = provider

    p_type, base_url, api_key, resolved_model = resolve_provider(config)
    if api_key_override:
        api_key = api_key_override
    if model:
        resolved_model = model
    model = resolved_model
    # p_type is "sensenova" or "openai" (custom → openai)
    provider = p_type

    if not api_key:
        return JSONResponse(
            {"error": f"未配置 {provider} 的 API Key，请在参数配置页面或左侧设置中填写"},
            status_code=400)

    # Build messages with system prompt
    full_messages = []
    if system_prompt:
        full_messages.append({"role": "system", "content": system_prompt})
    full_messages.extend(messages)

    body = {
        "model": model,
        "messages": full_messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if provider != "openai":
        body["reasoning_effort"] = reasoning_effort

    body_bytes = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=body_bytes,
        method="POST",
    )
    req.add_header("Authorization", f"Bearer {api_key}")
    req.add_header("Content-Type", "application/json")
    req.add_header("User-Agent", "CodelyLLM/1.0")

    async def event_stream():
        try:
            import time as _time
            t0 = _time.time()
            usage_data = None
            resp = urllib.request.urlopen(req, timeout=180)
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                if not line.startswith("data: "):
                    continue
                payload = line[6:].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                # Extract token
                choices = chunk.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        yield f"data: {json.dumps({'type': 'token', 'content': content})}\n\n"
                # Extract usage (some APIs send it in the last chunk)
                if chunk.get("usage"):
                    usage_data = chunk["usage"]

            elapsed = _time.time() - t0
            yield f"data: {json.dumps({'type': 'done', 'elapsed': round(elapsed, 2), 'usage': usage_data})}\n\n"

        except urllib.error.HTTPError as e:
            err = e.read().decode("utf-8", errors="replace")[:500]
            yield f"data: {json.dumps({'type': 'error', 'error': f'HTTP {e.code}: {err}'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)[:300]})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ===========================================================================
# Character Library Management Page
# ===========================================================================

# Background image generation status tracking
_gen_status: dict[str, dict] = {}  # lib_id → {status, poses, error, started_at}


def _generate_char_images(lib_id: str, description: str, structure: str, mcp_tokens: str):
    """Background thread: generate pose atlas via MCP, download, split into poses."""
    import shutil
    lib_dir = LIBRARY_DIR / lib_id
    if not lib_dir.exists():
        _gen_status[lib_id] = {"status": "error", "error": "角色目录不存在", "poses": []}
        return

    _gen_status[lib_id] = {"status": "generating", "error": "", "poses": [], "started_at": time.time()}

    try:
        # Add pipeline to sys.path for MCP imports
        _pipeline = str(PIPELINE_DIR)
        if _pipeline not in sys.path:
            sys.path.insert(0, _pipeline)
        from mcp_client import reinitialize as mcp_reinit, call_tool, parse_task_id, poll_task, download_file
        from atlas_split import split_atlas

        # Initialize MCP with configured tokens
        if mcp_tokens:
            tokens = [t.strip() for t in mcp_tokens.split("\n") if t.strip()]
            tokens_str = ",".join(tokens)
            mcp_reinit(tokens=tokens)
        else:
            mcp_reinit()

        _STYLE = ("3D cartoon style, Pixar-like, warm soft lighting, "
                  "cel-shaded with thin clean black outline, "
                  "vibrant saturated colors, smooth surfaces")

        # Use "char_a" as source_key for all manually created characters
        src_key = "char_a"

        # Delete old pose images (for regeneration)
        for old in lib_dir.glob(f"pose_{src_key}_*.png"):
            old.unlink()
        old_atlas = lib_dir / f"pose_atlas_{src_key}.png"
        if old_atlas.exists():
            old_atlas.unlink()

        if structure == "quest":
            # 4×2 grid atlas (8 poses)
            atlas_prompt = (
                f"4x2 grid character pose sheet, eight poses of the same character, "
                f"{description}, "
                f"top row left to right: speaking with mouth open, listening with slight smile, "
                f"thinking with hand on chin, surprised with raised eyebrows, "
                f"bottom row left to right: nodding in agreement, waving right hand, "
                f"pointing forward, laughing with eyes closed, "
                f"half-body close-up, waist up, all eight poses same character same outfit, "
                f"plain white background, {_STYLE}, "
                f"no props, no objects, no scene, no text"
            )
            gen_params = {
                "prompt": atlas_prompt,
                "provider": "seedream",
                "image_size": "4992x3328",
                "output_format": "png",
                "is_segmentation": True,
            }
        else:
            # Original structure: single character scene image
            atlas_prompt = (
                f"{description}, standing pose, half-body close-up, waist up, "
                f"plain white background, {_STYLE}, "
                f"no props, no objects, no scene, no text"
            )
            gen_params = {
                "prompt": atlas_prompt,
                "provider": "seedream",
                "image_size": "landscape_16_9",
                "output_format": "png",
            }

        print(f"  [CharGen] Generating atlas for {lib_id}...")
        result = call_tool("generate_image", gen_params)
        task_id = parse_task_id(result)
        if not task_id:
            _gen_status[lib_id] = {"status": "error", "error": "MCP 返回无 task_id", "poses": []}
            return

        data = poll_task(task_id, interval=10, max_wait=600)
        url = data.get("url", "")
        if not url:
            _gen_status[lib_id] = {"status": "error", "error": "MCP 生成失败：无图片 URL", "poses": []}
            return

        if structure == "quest":
            # Download atlas, split into 8 poses（分隔线检测 + 残边修剪，无缝时回退等分）
            atlas_path = str(lib_dir / f"pose_atlas_{src_key}.png")
            download_file(url, atlas_path)
            print(f"  [CharGen] Downloaded atlas for {lib_id}")

            pose_files = [f"pose_{src_key}_{idx}.png" for idx in range(8)]
            split_atlas(atlas_path, 4, 2,
                        [str(lib_dir / f) for f in pose_files],
                        log_prefix="[CharGen]")
            print(f"  [CharGen] Split into {len(pose_files)} poses for {lib_id}")
        else:
            # Original: save as char_scene.png
            atlas_path = str(lib_dir / "char_scene.png")
            download_file(url, atlas_path)
            pose_files = ["char_scene.png"]
            print(f"  [CharGen] Saved char_scene.png for {lib_id}")

        # Update thumbnail
        if structure == "quest":
            thumb_src = lib_dir / f"pose_{src_key}_0.png"
        else:
            thumb_src = lib_dir / "char_scene.png"
        if thumb_src.exists():
            thumb_dst = lib_dir / "thumb.png"
            if thumb_dst.exists():
                thumb_dst.unlink()
            shutil.copy2(str(thumb_src), str(thumb_dst))

        pose_urls = [
            {"name": f, "url": f"/api/character_library/{lib_id}/poses/{f}"}
            for f in pose_files
        ]
        _gen_status[lib_id] = {
            "status": "done",
            "error": "",
            "poses": pose_urls,
            "thumb_url": f"/api/character_library/{lib_id}/image",
        }
        print(f"  [CharGen] Done for {lib_id}: {len(pose_files)} poses")

    except Exception as e:
        _gen_status[lib_id] = {"status": "error", "error": str(e)[:200], "poses": []}
        print(f"  [CharGen] ERROR for {lib_id}: {e}")


@app.get("/characters", response_class=HTMLResponse)
async def characters_page(request: Request):
    """Character library management page."""
    config = load_config()
    library_chars = []
    if LIBRARY_DIR.exists():
        for d in sorted(LIBRARY_DIR.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
            meta_path = d / "meta.json"
            if not meta_path.exists():
                continue
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                thumb = d / "thumb.png"
                meta["image_url"] = f"/api/character_library/{d.name}/image" if thumb.exists() else ""
                # Count pose images
                if meta.get("structure") == "quest":
                    meta["pose_count"] = sum(1 for p in d.glob("pose_char_a_*.png"))
                else:
                    meta["pose_count"] = 1 if (d / "char_scene.png").exists() else 0
                library_chars.append(meta)
            except (json.JSONDecodeError, OSError):
                continue
    return templates.TemplateResponse(request, "characters.html", {
        "config": config,
        "library_chars": library_chars,
        "active_page": "characters",
    })


@app.post("/api/character_library/create")
async def api_library_create(
    name: str = Form(""),
    description: str = Form(""),
    gender: str = Form(""),
    structure: str = Form("quest"),
    qwen_speaker: str = Form(""),
    image: UploadFile | None = File(None),
):
    """Manually create a new character in the library."""
    if not name.strip():
        return JSONResponse({"ok": False, "error": "名称不能为空"}, status_code=400)
    if not description.strip():
        return JSONResponse({"ok": False, "error": "描述不能为空"}, status_code=400)

    lib_id = f"char_{int(time.time())}_{gender or 'custom'}"
    lib_dir = LIBRARY_DIR / lib_id
    lib_dir.mkdir(parents=True, exist_ok=True)

    # Save uploaded image as thumb if provided
    if image and image.filename:
        content = await image.read()
        (lib_dir / "thumb.png").write_bytes(content)

    meta = {
        "id": lib_id,
        "name": name.strip(),
        "description": description.strip(),
        "gender": gender.strip(),
        "structure": structure,
        "qwen_speaker": qwen_speaker.strip(),
        "source_run": "",
        "source_key": "char_a",
        "created": time.time(),
    }
    (lib_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    return {"ok": True, "id": lib_id, "meta": meta}


@app.put("/api/character_library/{lib_id}")
async def api_library_update(lib_id: str, request: Request):
    """Update character metadata (name, description, gender, structure, qwen_speaker)."""
    lib_dir = LIBRARY_DIR / lib_id
    if not lib_dir.exists():
        return JSONResponse({"ok": False, "error": "未找到"}, status_code=404)
    meta_path = lib_dir / "meta.json"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return JSONResponse({"ok": False, "error": "meta.json 读取失败"}, status_code=500)

    data = await request.json()
    for key in ("name", "description", "gender", "structure", "qwen_speaker"):
        if key in data:
            meta[key] = data[key]
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"ok": True, "meta": meta}


@app.post("/api/character_library/{lib_id}/image")
async def api_library_upload_image(lib_id: str, image: UploadFile = File(...)):
    """Upload or replace character thumbnail image."""
    lib_dir = LIBRARY_DIR / lib_id
    if not lib_dir.exists():
        return JSONResponse({"ok": False, "error": "未找到"}, status_code=404)
    content = await image.read()
    (lib_dir / "thumb.png").write_bytes(content)
    return {"ok": True, "image_url": f"/api/character_library/{lib_id}/image"}


@app.post("/api/character_library/{lib_id}/generate_images")
async def api_library_generate_images(lib_id: str):
    """Start AI pose atlas generation in background thread."""
    import threading
    lib_dir = LIBRARY_DIR / lib_id
    if not lib_dir.exists():
        return JSONResponse({"ok": False, "error": "未找到"}, status_code=404)
    meta_path = lib_dir / "meta.json"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return JSONResponse({"ok": False, "error": "meta.json 读取失败"}, status_code=500)

    description = meta.get("description", "")
    if not description:
        return JSONResponse({"ok": False, "error": "角色描述为空，无法生成图片"}, status_code=400)

    structure = meta.get("structure", "quest")
    config = load_config()
    mcp_tokens = config.get("mcp_tokens", "").strip()

    # Start background thread
    thread = threading.Thread(
        target=_generate_char_images,
        args=(lib_id, description, structure, mcp_tokens),
        daemon=True,
    )
    thread.start()

    return {"ok": True, "message": "图片生成中..."}


@app.get("/api/character_library/{lib_id}/generation_status")
async def api_library_gen_status(lib_id: str):
    """Poll image generation status."""
    status = _gen_status.get(lib_id, {"status": "idle", "poses": [], "error": ""})
    return status


@app.get("/api/character_library/{lib_id}/poses")
async def api_library_poses(lib_id: str):
    """List all pose images for a character."""
    lib_dir = LIBRARY_DIR / lib_id
    if not lib_dir.exists():
        return {"poses": []}
    meta_path = lib_dir / "meta.json"
    structure = "quest"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            structure = meta.get("structure", "quest")
        except (json.JSONDecodeError, OSError):
            pass

    poses = []
    if structure == "quest":
        for i in range(8):
            p = lib_dir / f"pose_char_a_{i}.png"
            if p.exists():
                poses.append({"name": p.name, "url": f"/api/character_library/{lib_id}/poses/{p.name}"})
    else:
        p = lib_dir / "char_scene.png"
        if p.exists():
            poses.append({"name": p.name, "url": f"/api/character_library/{lib_id}/poses/{p.name}"})
    return {"poses": poses}


@app.get("/api/character_library/{lib_id}/poses/{filename}")
async def api_library_pose_file(lib_id: str, filename: str):
    """Serve a pose image file."""
    lib_dir = LIBRARY_DIR / lib_id
    # Prevent path traversal
    if "/" in filename or "\\" in filename or ".." in filename:
        return JSONResponse({"error": "Invalid filename"}, status_code=400)
    pose_path = lib_dir / filename
    if not pose_path.exists():
        return JSONResponse({"error": "Not found"}, status_code=404)
    return FileResponse(str(pose_path), media_type="image/png")


# ===========================================================================
# Startup
# ===========================================================================

@app.on_event("startup")
async def startup():
    # Ensure configs dir exists
    (WEB_ROOT / "configs").mkdir(parents=True, exist_ok=True)
    (WEB_ROOT / "configs" / "character_library").mkdir(parents=True, exist_ok=True)
    # Create default config if missing
    if not (WEB_ROOT / "configs" / "default.json").exists():
        save_config(get_default_config())
