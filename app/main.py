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
    FileResponse, RedirectResponse, PlainTextResponse, Response,
)
from fastapi.staticfiles import StaticFiles

# Suppress noisy Windows ConnectionResetError on video stream disconnect
logging.getLogger("asyncio").setLevel(logging.CRITICAL)

from .config_manager import (
    PARAM_SPEC, GROUP_META, get_default_config,
    load_config, save_config,
    list_presets, save_preset, load_preset, delete_preset,
    build_cli_args, detect_local_mcp_token,
    load_llm_providers, save_llm_providers,
    get_provider_options, resolve_provider,
    MODES, MODE_LABELS, MODE_SHORT_LABELS, get_active_mode, set_active_mode,
    load_mode_config, save_mode_config, load_all_mode_configs,
    effective_param_spec, param_effective_in_mode,
    RECYCLE_DIRNAME, LEGACY_RECYCLE_DIRNAME,
    find_run_dir, iter_run_dirs,
)
from .pipeline_service import get_service, STEP_ORDER
from .mode_test_service import get_mode_test_service
from .config_manager import detect_local_mcp_token
from .page_mcp import (
    PAGES as MCP_PAGES, SOURCE_LABELS as MCP_SOURCE_LABELS,
    PageMcpSession, resolve_page_tokens, load_page_tokens, save_page_tokens,
    mask_token,
)
from . import topics_ai
from . import script_library
import style_manager as style_lib
import subtitle_style_manager as subtitle_style_lib

# 共享基础设施（路径 / 模板 / SSE / 话题文件 IO / TTS 共享状态）
from .paths import (
    WEB_ROOT, TEMPLATES_DIR, STATIC_DIR, PIPELINE_DIR, LIBRARY_DIR,
    TRASH_META_FILENAME, SCRIPTS_FORM_PATH, CHARACTER_SETS_PATH, AI_TEST_CONFIG_PATH,
    QWEN_VOICE_CONFIG_PATH, CUSTOM_VOICES_DIR, VOICE_PREVIEWS_DIR,
    KOKORO_VOICE_CONFIG_PATH, MOSS_VOICE_CONFIG_PATH, MOSS_VOICES_DIR, MOSS_PREVIEWS_DIR,
)
from .templating import templates
from .sse import sse_line as _sse, SSE_HEADERS as _SSE_HEADERS
from .topics_io import (
    load_topics_data as _load_topics_data,
    load_used_topic_names as _load_used_topic_names,
)
from .tts_state import TTS_SYNTH_LOCK as _TTS_SYNTH_LOCK, PREVIEW_TEXTS as _PREVIEW_TEXTS
from .routers import config as config_routes
from .routers import health as health_routes
from .routers import mcp_tokens as mcp_tokens_routes
from .routers import mode_test as mode_test_routes
from .routers import run as run_routes
from .routers import runs as runs_routes
from .routers import scripts as scripts_routes
from .routers import styles as styles_routes
from .routers import subtitle_styles as subtitle_styles_routes
from .routers import topics as topics_routes
from .routers import characters as characters_routes
from .routers import voices_kokoro as voices_kokoro_routes
from .routers import voices_moss as voices_moss_routes
from .routers import voices_qwen as voices_qwen_routes
from .routers.voices_qwen import auto_freeze_pending_designed_voices

# FastAPI app
app = FastAPI(title="Listening Video Generator")


class _NoCacheStaticFiles(StaticFiles):
    """静态文件禁用浏览器缓存（本地开发工具，CSS/JS 迭代频繁，防止改版后浏览器继续用旧缓存）。"""

    def file_response(self, *args, **kwargs):
        resp = super().file_response(*args, **kwargs)
        resp.headers["Cache-Control"] = "no-cache"
        return resp


app.mount("/static", _NoCacheStaticFiles(directory=str(STATIC_DIR)), name="static")


# ===========================================================================
# Page routes
# ===========================================================================

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    service = get_service()
    config = load_config()
    mode_configs = load_all_mode_configs()
    # 快捷启动「画面风格」下拉选项（内置+自定义；各模式当前值不在选项中时兜底显示）
    style_options = style_lib.get_style_options()
    for mcfg in mode_configs.values():
        vs = mcfg.get("visual_style")
        if vs and vs not in style_options:
            style_options[vs] = f"{vs}（已失效，请重新选择）"
    return templates.TemplateResponse(request, "dashboard.html", {
        "config": config,
        "runner": service,
        "active_page": "dashboard",
        "mode_configs": mode_configs,
        "active_mode": get_active_mode(),
        "mode_labels": MODE_LABELS,
        "style_options": style_options,
    })


@app.get("/config", response_class=HTMLResponse)
async def config_page(request: Request, mode: str = ""):
    # ?mode= 切换 Tab：同步激活模式并渲染该模式配置
    if mode and mode in MODES:
        set_active_mode(mode)
    mode = get_active_mode()
    config = load_mode_config(mode)
    presets = list_presets()
    # Inject dynamic LLM provider options into PARAM_SPEC
    PARAM_SPEC["llm_provider"]["options"] = get_provider_options()
    # Inject visual style options (built-in + custom styles)
    style_opts = style_lib.get_style_options()
    cur_style = config.get("visual_style", "pixar3d")
    if cur_style and cur_style not in style_opts:
        # 自定义风格已删除：诚实显示，让用户重新选择
        style_opts = {cur_style: f"{cur_style}（已失效，请重新选择）", **style_opts}
    PARAM_SPEC["visual_style"]["options"] = style_opts
    # Inject subtitle style options (built-in + custom styles)
    sub_style_opts = subtitle_style_lib.get_style_options()
    cur_sub_style = config.get("subtitle_style", "")
    if cur_sub_style and cur_sub_style not in sub_style_opts:
        sub_style_opts = {cur_sub_style: f"{cur_sub_style}（已失效，请重新选择）", **sub_style_opts}
    PARAM_SPEC["subtitle_style"]["options"] = sub_style_opts
    # 按模式过滤：只渲染该模式实际消费的参数（PARAM_SPEC "modes" 标注）
    mode_spec = effective_param_spec(mode)
    # Group params by group
    grouped = {}
    for key, spec in mode_spec.items():
        if key == "structure":
            continue  # 结构由 Tab 决定，不渲染下拉
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
        "mode": mode,
        "mode_labels": MODE_LABELS,
        # 自定义 Provider 模型列表（不含 api_key 等敏感字段；去重保持顺序）
        "custom_providers": [
            {"id": p.get("id", ""), "name": p.get("name", ""),
             "models": list(dict.fromkeys(p.get("models") or []))}
            for p in load_llm_providers()
        ],
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
async def gallery_page(request: Request, name: str, mode: str = ""):
    config = load_config()
    output_dir = Path(config.get("output_dir", "./output"))
    # 运行卡片带 ?mode= 消歧；查不到时保持原「渲染空页面」行为
    run_dir = find_run_dir(output_dir, name, mode) or (output_dir / name)
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
        "yt_options": yt_meta.get("title_options", []),
        "active_page": "runs",
    })


@app.get("/runs", response_class=HTMLResponse)
async def runs_page(request: Request):
    config = load_config()
    output_dir = Path(config.get("output_dir", "./output"))
    runs = []
    for d in iter_run_dirs(output_dir):
        script_path = d / "script.json"
        videos_dir = d / "videos"
        thumbnail = d / "thumbnail.jpg"
        no_sub = videos_dir / "final_no_sub.mp4"
        meta_path = d / "subtitles" / "meta.json"
        has_4k = any(d.glob("*_4K.mp4"))
        # 重渲条件：无字幕底片 + 时间轴元数据 + 脚本齐备（仅重跑字幕烧录环节）
        recomposable = (no_sub.exists() and no_sub.stat().st_size >= 1_000_000
                        and meta_path.exists() and script_path.exists())
        run_info = {
            "name": d.name,
            "path": str(d),
            "created": d.stat().st_mtime,
            "has_script": script_path.exists(),
            "has_thumbnail": thumbnail.exists(),
            "thumbnail_url": f"/api/runs/{d.name}/thumbnail" if thumbnail.exists() else "",
            "uploaded": (d / "uploaded.flag").exists(),
            "has_4k": has_4k,
            "recomposable": recomposable,
            "structure": "",
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
        # 卡片模式徽标：脚本缺 structure 时回退所在模式文件夹名
        if run_info["structure"] not in MODES:
            run_info["structure"] = d.parent.name if d.parent.name in MODES else ""
        run_info["structure_label"] = MODE_SHORT_LABELS.get(run_info["structure"],
                                                            run_info["structure"])
        runs.append(run_info)

    # 回收站列表（_recycle_bin 下所有已删除运行；兼容旧版 .recycle_bin）
    trash_runs = []
    for recycle_root in (output_dir / RECYCLE_DIRNAME, output_dir / LEGACY_RECYCLE_DIRNAME):
        if not recycle_root.is_dir():
            continue
        for d in sorted(recycle_root.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
            if not d.is_dir():
                continue
            item = {
                "name": d.name,
                "original_name": d.name,
                "deleted_at": d.stat().st_mtime,
                "title": d.name,
                "structure_label": "",
            }
            meta_path = d / TRASH_META_FILENAME
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    item["deleted_at"] = meta.get("deleted_at", item["deleted_at"])
                    item["original_name"] = meta.get("original_name", d.name)
                except (json.JSONDecodeError, OSError):
                    pass
            structure = ""
            script_path = d / "script.json"
            if script_path.exists():
                try:
                    s = json.loads(script_path.read_text(encoding="utf-8"))
                    item["title"] = s.get("youtube_title", s.get("title", d.name))
                    structure = s.get("structure", "")
                except (json.JSONDecodeError, OSError):
                    pass
            item["structure_label"] = MODE_SHORT_LABELS.get(structure, structure)
            trash_runs.append(item)

    return templates.TemplateResponse(request, "runs.html", {
        "runs": runs,
        "trash_runs": trash_runs,
        "mode_labels": MODE_LABELS,
        "subtitle_style_options": subtitle_style_lib.get_style_options(),
        "active_page": "runs",
    })


# ===========================================================================
# Config API
# ===========================================================================

app.include_router(config_routes.router)

# ===========================================================================
# Pipeline Run API
# ===========================================================================

app.include_router(run_routes.router)

# ===========================================================================
# 模式效果测试（一次性生成迷你素材 + 零消耗本地合成）
# ===========================================================================

@app.get("/mode-test", response_class=HTMLResponse)
async def mode_test_page(request: Request):
    service = get_mode_test_service()
    return templates.TemplateResponse(request, "mode_test.html", {
        "active_page": "mode_test",
        "mode_labels": MODE_LABELS,
        "status": service.full_status(),
    })


app.include_router(mode_test_routes.router)

app.include_router(topics_routes.router)

# ===========================================================================
# Visual Styles — 画面风格管理
# ===========================================================================

@app.get("/styles", response_class=HTMLResponse)
async def styles_page(request: Request):
    config = load_config()
    current_id = config.get("visual_style", "pixar3d")
    styles = style_lib.list_styles()
    for s in styles:
        s["has_preview"] = style_lib.preview_path(s["id"]).exists()
    current = style_lib.get_style(current_id) or style_lib.get_style("pixar3d")
    return templates.TemplateResponse(request, "styles.html", {
        "styles": styles,
        "current_style": current,
        "current_id": current_id,
        "active_mode": get_active_mode(),
        "mode_labels": MODE_LABELS,
        "active_page": "styles",
    })


app.include_router(styles_routes.router)

# ===========================================================================
# Subtitle Styles — 字幕样式设计与预览
# ===========================================================================

@app.get("/subtitle-styles", response_class=HTMLResponse)
async def subtitle_styles_page(request: Request):
    ctx = _current_subtitle_style_ctx()
    styles = subtitle_style_lib.list_styles()
    return templates.TemplateResponse(request, "subtitle_styles.html", {
        "styles": styles,
        "current_id": ctx["current_id"],
        "current_style": ctx["current_style"],
        "config_font_size": ctx["config_font_size"],
        "fonts": subtitle_style_lib.get_font_options(),
        "backgrounds": subtitle_style_lib.list_backgrounds(),
        "mode_infos": subtitle_style_lib.list_mode_infos(),
        "active_mode": get_active_mode(),
        "mode_labels": MODE_LABELS,
        "active_page": "subtitle_styles",
    })


app.include_router(subtitle_styles_routes.router)

# ===========================================================================
# Scripts Library — batch generation & quality review
# ===========================================================================

@app.get("/scripts", response_class=HTMLResponse)
async def scripts_page(request: Request):
    config = load_config()
    return templates.TemplateResponse(request, "scripts.html", {
        "config": config,
        "active_page": "scripts",
    })


app.include_router(scripts_routes.router)

app.include_router(runs_routes.router)

# ===========================================================================
# MCP Token detection
# ===========================================================================

app.include_router(mcp_tokens_routes.router)

app.include_router(characters_routes.router)

# ===========================================================================
# Health check
# ===========================================================================

app.include_router(health_routes.router)

# ===========================================================================
# QwenTTS Voice Management
# ===========================================================================


app.include_router(voices_qwen_routes.router)

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


# Kokoro TTS Voices (本地引擎，固定音色清单 + 试听)
# ===========================================================================

app.include_router(voices_kokoro_routes.router)

@app.get("/kokoro_voices", response_class=HTMLResponse)
async def kokoro_voices_page(request: Request):
    """Kokoro voice management page."""
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
                library_chars.append(meta)
            except (json.JSONDecodeError, OSError):
                continue
    return templates.TemplateResponse(request, "kokoro_voices.html", {
        "config": config,
        "library_chars": library_chars,
        "active_page": "kokoro_voices",
    })


# ===========================================================================
# MOSS-TTS Voice Management
# ===========================================================================


app.include_router(voices_moss_routes.router)

@app.get("/moss_voices", response_class=HTMLResponse)
async def moss_voices_page(request: Request):
    """MOSS-TTS-Nano voice management page."""
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
                library_chars.append(meta)
            except (json.JSONDecodeError, OSError):
                continue
    return templates.TemplateResponse(request, "moss_voices.html", {
        "config": config,
        "library_chars": library_chars,
        "active_page": "moss_voices",
    })


# ===========================================================================
# AI Test — LLM Playground
# ===========================================================================


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
    messages = [m for m in data.get("messages", [])
                if isinstance(m, dict) and m.get("role")]
    system_prompt = data.get("system_prompt", "")
    provider = data.get("provider", "sensenova")
    model = data.get("model", "")
    try:
        temperature = float(data.get("temperature", 0.8))
    except (TypeError, ValueError):
        temperature = 0.8
    try:
        max_tokens = int(data.get("max_tokens", 8192))
    except (TypeError, ValueError):
        max_tokens = 8192
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
    if not model:
        return JSONResponse(
            {"error": "未指定模型（该 Provider 未配置模型列表，请在自定义 Provider 中添加）"},
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

    import queue as _queue

    def _produce(q: "_queue.Queue") -> None:
        """工作线程：阻塞读取上游 SSE 流。

        urllib 是同步 I/O，直接在 async generator（事件循环）里读会阻塞
        整个 Web 服务（包括运行中 pipeline 的日志 SSE），故移到线程中转。
        """
        import time as _time
        t0 = _time.time()
        usage_data = None
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                for raw_line in resp:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line or not line.startswith("data: "):
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
                            q.put(("token", content))
                    # Extract usage (some APIs send it in the last chunk)
                    if chunk.get("usage"):
                        usage_data = chunk["usage"]
            q.put(("done", {"elapsed": round(_time.time() - t0, 2),
                            "usage": usage_data}))
        except urllib.error.HTTPError as e:
            err = e.read().decode("utf-8", errors="replace")[:500]
            q.put(("error", f"HTTP {e.code}: {err}"))
        except Exception as e:  # noqa: BLE001 — 错误信息原样转发给前端
            q.put(("error", str(e)[:300]))
        finally:
            q.put(None)

    async def event_stream():
        q: _queue.Queue = _queue.Queue()
        threading.Thread(target=_produce, args=(q,), daemon=True).start()
        while True:
            item = await asyncio.to_thread(q.get, True, None)
            if item is None:
                break
            kind, payload = item
            if kind == "token":
                yield f"data: {json.dumps({'type': 'token', 'content': payload})}\n\n"
            elif kind == "done":
                yield f"data: {json.dumps({'type': 'done', 'elapsed': payload['elapsed'], 'usage': payload['usage']})}\n\n"
            elif kind == "error":
                yield f"data: {json.dumps({'type': 'error', 'error': payload})}\n\n"

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
                    meta["pose_count"] = sum(1 for p in d.glob("pose_char_a_*.png")
                                             if "_c" not in p.stem)
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


# ===========================================================================
# Startup
# ===========================================================================

@app.on_event("startup")
async def startup():
    # Ensure configs dir exists
    (WEB_ROOT / "configs").mkdir(parents=True, exist_ok=True)
    (WEB_ROOT / "configs" / "character_library").mkdir(parents=True, exist_ok=True)
    # 首次运行：从 legacy default.json 迁移生成 3 个模式配置文件
    load_all_mode_configs()
    # 设计音色自动冻结：内置英文女声 + 残留未冻结的设计音色 → 后台转为克隆音色（一次性）
    try:
        auto_freeze_pending_designed_voices()
    except Exception as e:  # noqa: BLE001 — 冻结失败不阻塞启动，下次启动重试
        print(f"[startup] 设计音色自动冻结入队失败: {e}")
