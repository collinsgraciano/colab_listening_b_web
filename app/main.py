"""FastAPI main application — all routes and API endpoints."""
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import (
    HTMLResponse, JSONResponse, StreamingResponse,
    FileResponse, RedirectResponse,
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
)
from .pipeline_service import get_service, STEP_ORDER
from .config_manager import detect_local_mcp_token

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
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "config": config,
        "runner": service,
        "active_page": "dashboard",
    })


@app.get("/config", response_class=HTMLResponse)
async def config_page(request: Request):
    config = load_config()
    presets = list_presets()
    # Group params by group
    grouped = {}
    for key, spec in PARAM_SPEC.items():
        g = spec["group"]
        if g not in grouped:
            grouped[g] = []
        grouped[g].append((key, spec, config.get(key, spec["default"])))
    # Sort groups by order
    sorted_groups = sorted(grouped.items(), key=lambda x: GROUP_META.get(x[0], {}).get("order", 99))
    return templates.TemplateResponse("config.html", {
        "request": request,
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

    return templates.TemplateResponse("topics.html", {
        "request": request,
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

    return templates.TemplateResponse("gallery.html", {
        "request": request,
        "run_name": name,
        "script": script,
        "images": images,
        "clips": clips,
        "audio": audio,
        "videos": videos,
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

    return templates.TemplateResponse("runs.html", {
        "request": request,
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
    # Final videos are in work_dir root; clips are in videos/ subdir
    video_path = output_dir / name / video_name
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
                structure = script.get("structure", "original")

                # Build character list based on structure
                if structure == "quest":
                    char_keys = ["char_a", "char_b", "char_c", "host"]
                    char_labels = {"char_a": "角色A", "char_b": "角色B", "char_c": "角色C", "host": "主持人"}
                    # Quest uses per-character pose images
                    has_chars = (img_dir / "pose_char_a_0.png").exists()
                else:
                    char_keys = ["char_a", "char_b"]
                    char_labels = {"char_a": "角色A", "char_b": "角色B"}
                    has_chars = (img_dir / "char_scene.png").exists()

                if not has_chars:
                    continue

                characters = []
                for key in char_keys:
                    desc = script.get(f"{key}_description", "")
                    gender = script.get(f"{key}_gender", "")
                    role = script.get(f"{key}_role", "")
                    if structure == "quest":
                        img_name = f"pose_{key}_0.png"
                        img_exists = (img_dir / img_name).exists()
                    else:
                        # Original/image: both share char_scene.png
                        img_name = "char_scene.png"
                        img_exists = (img_dir / img_name).exists()
                    characters.append({
                        "key": key,
                        "label": char_labels.get(key, key),
                        "description": desc,
                        "gender": gender,
                        "role": role,
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
    
    return {
        "images": images,
        "clips": clips,
        "audio": audio,
        "image_urls": {f: f"/api/runs/{name}/images/{f}" for f in images},
        "clip_urls": {f: f"/api/runs/{name}/video/{f}" for f in clips},
        "audio_urls": {f: f"/api/runs/{name}/audio/{f}" for f in audio},
    }


@app.get("/api/runs/{name}/audio/{audio_name}")
async def api_get_audio(name: str, audio_name: str):
    config = load_config()
    output_dir = Path(config.get("output_dir", "./output"))
    audio_path = output_dir / name / "audio" / audio_name
    if not audio_path.exists():
        return JSONResponse({"error": "Not found"}, status_code=404)
    return FileResponse(str(audio_path), media_type="audio/mpeg")


# ===========================================================================
# Startup
# ===========================================================================

@app.on_event("startup")
async def startup():
    # Ensure configs dir exists
    (WEB_ROOT / "configs").mkdir(parents=True, exist_ok=True)
    # Create default config if missing
    if not (WEB_ROOT / "configs" / "default.json").exists():
        save_config(get_default_config())
