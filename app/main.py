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

# ===========================================================================
# Character reuse API
# ===========================================================================

@app.get("/api/character_sources")
async def api_character_sources():
    """List available previous runs for character reuse, with images and all characters."""
    config = load_config()
    output_dir = Path(config.get("output_dir", "./output"))
    sources = []
    for d in iter_run_dirs(output_dir):
        script_path = d / "script.json"
        if not script_path.exists():
            continue
        try:
            script = json.loads(script_path.read_text(encoding="utf-8"))
            img_dir = d / "images"

            # 结构识别：优先读 script.json 的 structure 字段（_step0_script 会写入），
            # 旧运行缺失时回退文件探测
            structure = script.get("structure", "")
            if structure not in ("original", "original_static",
                                 "original_cutout", "quest"):
                if (img_dir / "pose_char_a_0.png").exists():
                    # 姿势图集 → quest 或 cutout：以 char_c 图集存在与否细分
                    structure = ("quest" if (img_dir / "pose_char_c_0.png").exists()
                                 else "original_cutout")
                elif (img_dir / "char_scene.png").exists():
                    structure = "original"
                else:
                    continue  # no character images

            # Build character list based on detected structure
            if structure == "quest":
                char_keys = ["char_a", "char_b", "char_c", "host"]
                char_labels = {"char_a": "角色A", "char_b": "角色B", "char_c": "角色C", "host": "主持人"}
                img_name_for = lambda key: f"pose_{key}_0.png"
            elif structure == "original_cutout":
                # 仅当源运行存在独立主持人图集时才提供 host 卡
                # （绑定了 char_a/char_b 的源运行没有独立主持人可复用）
                char_keys = (["char_a", "char_b"]
                             + (["host"] if (img_dir / "pose_host_0.png").exists() else []))
                char_labels = {"char_a": "角色A", "char_b": "角色B", "host": "主持人"}
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
                    "moss_voice": script.get(f"{key}_moss_voice", ""),
                    "kokoro_voice": script.get(f"{key}_kokoro_voice", ""),
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
# Character Sets API (角色套装：整套角色配置命名保存 / 一键应用)
# ===========================================================================

_SET_FIELDS = ["character_source", "character_reuse", "character_fixes",
               "character_library", "character_voices", "character_zh_voices",
               "character_moss_voices", "character_kokoro_voices", "_ui_descs"]


def _load_char_sets() -> list:
    if not CHARACTER_SETS_PATH.exists():
        return []
    try:
        return json.loads(CHARACTER_SETS_PATH.read_text(encoding="utf-8")).get("sets", [])
    except (json.JSONDecodeError, OSError):
        return []


def _save_char_sets(sets: list) -> None:
    CHARACTER_SETS_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHARACTER_SETS_PATH.write_text(
        json.dumps({"sets": sets}, ensure_ascii=False, indent=2), encoding="utf-8")


@app.get("/api/character_sets")
async def api_char_sets_list():
    """List all saved character sets (whole-cast presets), newest first."""
    sets = sorted(_load_char_sets(), key=lambda s: s.get("created", 0), reverse=True)
    return {"sets": sets}


@app.post("/api/character_sets/save")
async def api_char_sets_save(request: Request):
    """Save the current character config of a mode as a named set.

    Same name + structure → update in place (keep id/created).
    """
    data = await request.json()
    name = (data.get("name", "") or "").strip()
    mode = data.get("mode", "") or get_active_mode()
    if not name:
        return JSONResponse({"ok": False, "error": "名称不能为空"}, status_code=400)
    if mode not in MODES:
        return JSONResponse({"ok": False, "error": f"未知模式: {mode}"}, status_code=400)

    sets = _load_char_sets()
    entry = next(
        (s for s in sets if s.get("name") == name and s.get("structure") == mode), None)
    if entry is None:
        entry = {"id": f"set_{int(time.time() * 1000)}", "name": name,
                 "structure": mode, "created": time.time()}
        sets.append(entry)
    for f in _SET_FIELDS:
        entry[f] = data.get(f, "")
    _save_char_sets(sets)
    return {"ok": True, "set": entry}


@app.delete("/api/character_sets/{set_id}")
async def api_char_sets_delete(set_id: str):
    sets = _load_char_sets()
    remaining = [s for s in sets if s.get("id") != set_id]
    if len(remaining) == len(sets):
        return JSONResponse({"ok": False, "error": "未找到"}, status_code=404)
    _save_char_sets(remaining)
    return {"ok": True}


# ===========================================================================
# Health check
# ===========================================================================

app.include_router(health_routes.router)

# ===========================================================================
# Character Library API
# ===========================================================================


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
    run_dir = find_run_dir(output_dir, run_name)
    script_path = run_dir / "script.json" if run_dir else None
    if not script_path or not script_path.exists():
        return JSONResponse({"ok": False, "error": "运行不存在"}, status_code=404)

    script = json.loads(script_path.read_text(encoding="utf-8"))
    desc = script.get(f"{char_key}_description", "")
    gender = script.get(f"{char_key}_gender", "")
    role = script.get(f"{char_key}_role", "")
    qwen_speaker = script.get(f"{char_key}_qwen_speaker", "")
    moss_voice = script.get(f"{char_key}_moss_voice", "")
    kokoro_voice = script.get(f"{char_key}_kokoro_voice", "")
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
        "qwen_speaker": qwen_speaker,
        "moss_voice": moss_voice,
        "kokoro_voice": kokoro_voice,
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


def _preview_cache_path(speaker: str, language: str, text: str) -> Path:
    """Deterministic cache path for a voice preview (speaker+language+text)."""
    import hashlib
    safe = "".join(c for c in speaker if c.isalnum() or c in "-_") or "voice"
    h = hashlib.md5(f"{speaker}|{language}|{text}".encode("utf-8")).hexdigest()[:8]
    return VOICE_PREVIEWS_DIR / f"{safe}__{language}__{h}.mp3"


def _purge_preview_cache(speaker: str) -> None:
    """Remove all cached previews of a voice (on voice deletion)."""
    if not VOICE_PREVIEWS_DIR.exists():
        return
    safe = "".join(c for c in speaker if c.isalnum() or c in "-_") or "voice"
    for p in VOICE_PREVIEWS_DIR.glob(f"{safe}__*.mp3"):
        p.unlink(missing_ok=True)


def _load_qwen_voice_config() -> dict:
    """Load qwen_voice_config.json with defaults."""
    defaults = {
        "default_male": "Ryan",
        "default_female": "Vivian",
        "default_host_female": "Serena",
        "custom_voices": [],
        "designed_voices": [],
        "candidate_voices": [],  # LLM 随机生成、待试听挑选的候选设计音色
        "builtin_voice_dismissed": [],  # 用户删除的冻结内置音色（启动时不再自动冻结复活）
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
    speaker = data.get("moss_voice", "")
    meta["moss_voice"] = speaker
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"ok": True, "meta": meta}


@app.put("/api/character_library/{lib_id}/kokoro_voice")
async def api_library_set_kokoro_voice(lib_id: str, request: Request):
    """Set Kokoro TTS voice for a library character."""
    lib_dir = LIBRARY_DIR / lib_id
    if not lib_dir.exists():
        return JSONResponse({"ok": False, "error": "未找到"}, status_code=404)
    meta_path = lib_dir / "meta.json"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return JSONResponse({"ok": False, "error": "meta.json 读取失败"}, status_code=500)
    data = await request.json()
    voice = data.get("kokoro_voice", "")
    meta["kokoro_voice"] = voice
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"ok": True, "meta": meta}


@app.post("/api/qwen_voices/preview")
async def api_qwen_preview(request: Request):
    """Preview a voice: serve cached audio if available, else generate & cache."""
    import sys as _sys
    _pipeline = str(PIPELINE_DIR)
    if _pipeline not in _sys.path:
        _sys.path.insert(0, _pipeline)

    data = await request.json()
    speaker = data.get("speaker", "Vivian")
    language = data.get("language", "english")
    text = data.get("text", "") or _PREVIEW_TEXTS.get(language, _PREVIEW_TEXTS["english"])
    regenerate = bool(data.get("regenerate", False))

    cache_path = _preview_cache_path(speaker, language, text)
    if cache_path.exists() and not regenerate:
        return FileResponse(str(cache_path), media_type="audio/mpeg")

    config = load_config()
    model_path = config.get("qwen_model_path", r"H:\models\Qwen3-TTS-12Hz-0.6B-CustomVoice")
    base_model_path = config.get("qwen_base_model_path", r"H:\models\Qwen3-TTS-12Hz-1.7B-Base")
    voicedesign_model_path = config.get("qwen_voicedesign_model_path", r"H:\models\Qwen3-TTS-12Hz-1.7B-VoiceDesign")
    device = config.get("qwen_device", "cuda:0")

    try:
        from qwen_tts_engine import QwenTTSEngine

        def _synth() -> None:
            engine = QwenTTSEngine(model_path, device, base_model_path, voicedesign_model_path)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            out_path = str(cache_path)
            # 共享合成锁：与批量试听任务串行化 GPU 合成，防并发冲突
            with _TTS_SYNTH_LOCK:
                if language == "chinese":
                    engine.synth_chinese(text, speaker, out_path, rate="+0%")
                else:
                    engine.synth_english(text, speaker, out_path, rate="+0%")

        await asyncio.to_thread(_synth)
        return FileResponse(str(cache_path), media_type="audio/mpeg",
                            filename="preview.mp3",
                            headers={"Content-Disposition": "attachment; filename=preview.mp3"})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.get("/api/qwen_voices/preview/{voice}")
async def api_qwen_preview_cached(voice: str, language: str = "english"):
    """Serve a cached preview if it exists (instant playback), 404 otherwise."""
    text = _PREVIEW_TEXTS.get(language, _PREVIEW_TEXTS["english"])
    p = _preview_cache_path(voice, language, text)
    if not p.exists():
        return JSONResponse({"ok": False, "error": "no cached preview"}, status_code=404)
    return FileResponse(str(p), media_type="audio/mpeg")


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
    # 同名候选音色清理（避免旧候选的试听缓存命中新音色）
    if any(c["name"] == name for c in config.get("candidate_voices", [])):
        config["candidate_voices"] = [c for c in config["candidate_voices"] if c["name"] != name]
        _purge_preview_cache(name)
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
    # Remove cached previews of this voice
    _purge_preview_cache(name)
    # Remove from config
    config["custom_voices"] = [v for v in config["custom_voices"] if v["name"] != name]
    # 删除冻结内置音色 → 记入 dismissed，防止下次启动自动冻结"复活"
    import sys as _sys
    _pipeline = str(PIPELINE_DIR)
    if _pipeline not in _sys.path:
        _sys.path.insert(0, _pipeline)
    from qwen_tts_engine import DESIGNED_VOICES_BUILTIN
    if target.get("frozen_from") == "designed" and \
            name in {v["name"] for v in DESIGNED_VOICES_BUILTIN}:
        dismissed = config.setdefault("builtin_voice_dismissed", [])
        if name not in dismissed:
            dismissed.append(name)
    _save_qwen_voice_config(config)
    return {"ok": True}


@app.post("/api/qwen_voices/designed")
async def api_qwen_designed_create(request: Request):
    """Create a VoiceDesign voice from a natural-language instruct description."""
    import sys as _sys
    _pipeline = str(PIPELINE_DIR)
    if _pipeline not in _sys.path:
        _sys.path.insert(0, _pipeline)

    data = await request.json()
    name = (data.get("name", "") or "").strip()
    gender = (data.get("gender", "") or "").strip()
    language = (data.get("language", "english") or "english").strip()
    instruct = (data.get("instruct", "") or "").strip()

    if not name or not instruct:
        return JSONResponse({"ok": False, "error": "缺少名称或音色描述"}, status_code=400)

    # Name conflicts: presets / builtin designed / existing designed & custom
    from qwen_tts_engine import QWEN_SPEAKERS, DESIGNED_VOICES_BUILTIN
    if name in {s["name"] for s in QWEN_SPEAKERS}:
        return JSONResponse({"ok": False, "error": f"名称与预设音色冲突: {name}"}, status_code=400)
    if name in {v["name"] for v in DESIGNED_VOICES_BUILTIN}:
        return JSONResponse({"ok": False, "error": f"名称与内置设计音色冲突: {name}"}, status_code=400)

    config = _load_qwen_voice_config()
    if any(v["name"] == name for v in config.get("designed_voices", [])):
        return JSONResponse({"ok": False, "error": f"设计音色已存在: {name}"}, status_code=400)
    # 同名克隆音色不再报错：重新创建即重新冻结，原地覆盖，不产生多个克隆版

    lang_short = "zh" if language == "chinese" else "en"
    config.setdefault("designed_voices", []).append({
        "name": name,
        "description": data.get("description", "") or f"设计音色 ({gender or '?'})",
        "gender": gender,
        "language": lang_short,
        "instruct": instruct,
        "created": time.time(),
    })
    # 同名候选音色清理（避免旧候选的试听缓存命中新音色）
    if any(c["name"] == name for c in config.get("candidate_voices", [])):
        config["candidate_voices"] = [c for c in config.get("candidate_voices", []) if c["name"] != name]
        _purge_preview_cache(name)
    _save_qwen_voice_config(config)
    # 保存即冻结：后台生成样本并转为同名克隆音色，音色从此稳定
    _enqueue_voice_freeze(name)
    return {"ok": True, "name": name, "freezing": True}


@app.delete("/api/qwen_voices/designed/{name}")
async def api_qwen_designed_delete(name: str):
    """Delete a user-created designed voice (builtin ones are protected)."""
    import sys as _sys
    _pipeline = str(PIPELINE_DIR)
    if _pipeline not in _sys.path:
        _sys.path.insert(0, _pipeline)

    from qwen_tts_engine import DESIGNED_VOICES_BUILTIN
    if name in {v["name"] for v in DESIGNED_VOICES_BUILTIN}:
        return JSONResponse({"ok": False, "error": "内置设计音色不可删除"}, status_code=400)

    config = _load_qwen_voice_config()
    if not any(v["name"] == name for v in config.get("designed_voices", [])):
        return JSONResponse({"ok": False, "error": "未找到"}, status_code=404)
    config["designed_voices"] = [v for v in config.get("designed_voices", []) if v["name"] != name]
    _save_qwen_voice_config(config)
    # Remove cached previews of this voice
    _purge_preview_cache(name)
    return {"ok": True}


# --- LLM 随机生成候选音色（试听 → 挑选保存）---

def _build_voice_llm_override() -> dict:
    """构建音色生成用的线程局部 LLM 配置（不改 os.environ，与运行中 pipeline 互不干扰）。"""
    cfg = load_config()
    p_type, base_url, api_key, model = resolve_provider(cfg)
    if not api_key:
        raise RuntimeError(
            "未配置 LLM API Key — 请先在「参数配置」页面填写当前模式的大模型配置")
    ov: dict[str, str] = {
        "LLM_PROVIDER": p_type,
        "LLM_RETRIES": str(cfg.get("llm_retries", 10)),
    }
    if p_type == "sensenova":
        ov["SENSENOVA_API_KEY"] = api_key
        ov["SENSENOVA_MODEL"] = model or "deepseek-v4-flash"
    else:
        ov["OPENAI_BASE_URL"] = base_url
        ov["OPENAI_API_KEY"] = api_key
        ov["OPENAI_MODEL"] = model or "grok-4.6"
    if cfg.get("llm_min_interval"):
        ov["LLM_MIN_INTERVAL"] = str(cfg["llm_min_interval"])
    return ov


@app.get("/api/qwen_voices/candidates")
async def api_qwen_candidates_list():
    """List current LLM-generated candidate voices (with preview cache state)."""
    config = _load_qwen_voice_config()
    candidates = []
    for c in config.get("candidate_voices", []):
        c = dict(c)
        language = "chinese" if c.get("language") == "zh" else "english"
        text = _PREVIEW_TEXTS.get(language, _PREVIEW_TEXTS["english"])
        c["preview_cached"] = _preview_cache_path(
            c.get("name", ""), language, text).exists()
        candidates.append(c)
    return {"candidates": candidates}


@app.post("/api/qwen_voices/candidates/generate")
async def api_qwen_candidates_generate(request: Request):
    """Generate a batch of random voice designs via LLM (replaces current candidates)."""
    import sys as _sys
    _pipeline = str(PIPELINE_DIR)
    if _pipeline not in _sys.path:
        _sys.path.insert(0, _pipeline)

    data = await request.json()
    try:
        count = int(data.get("count", 10))
    except (TypeError, ValueError):
        count = 10
    count = max(1, min(count, 20))
    language = (data.get("language", "english") or "english").strip()
    gender = (data.get("gender", "any") or "any").strip().lower()
    if gender not in ("female", "male", "any"):
        gender = "any"

    def _run() -> list[dict]:
        from llm_client import generate_random_voice_designs, set_llm_env_override
        from qwen_tts_engine import get_all_voices

        override = _build_voice_llm_override()
        avoid = {v.get("name", "") for v in get_all_voices()}
        for c in _load_qwen_voice_config().get("candidate_voices", []):
            avoid.add(c.get("name", ""))

        set_llm_env_override(override)
        try:
            return generate_random_voice_designs(count, sorted(avoid), language, gender)
        finally:
            set_llm_env_override(None)

    try:
        voices = await asyncio.to_thread(_run)
    except Exception as e:  # noqa: BLE001 — 前端展示错误信息
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

    config = _load_qwen_voice_config()
    # 清理被替换掉的旧候选的试听缓存（新一批名字不重，避免陈旧缓存命中）
    new_names = {v["name"] for v in voices}
    for old in config.get("candidate_voices", []):
        if old.get("name") and old["name"] not in new_names:
            _purge_preview_cache(old["name"])
    config["candidate_voices"] = voices
    _save_qwen_voice_config(config)
    return {"ok": True, "candidates": voices}


@app.post("/api/qwen_voices/candidates/{name}/save")
async def api_qwen_candidate_save(name: str):
    """Save a candidate voice as a user-designed voice."""
    import sys as _sys
    _pipeline = str(PIPELINE_DIR)
    if _pipeline not in _sys.path:
        _sys.path.insert(0, _pipeline)

    config = _load_qwen_voice_config()
    target = next((c for c in config.get("candidate_voices", []) if c["name"] == name), None)
    if not target:
        return JSONResponse({"ok": False, "error": "未找到候选音色"}, status_code=404)

    from qwen_tts_engine import QWEN_SPEAKERS, DESIGNED_VOICES_BUILTIN
    if name in {s["name"] for s in QWEN_SPEAKERS}:
        return JSONResponse({"ok": False, "error": f"名称与预设音色冲突: {name}"}, status_code=400)
    if name in {v["name"] for v in DESIGNED_VOICES_BUILTIN}:
        return JSONResponse({"ok": False, "error": f"名称与内置设计音色冲突: {name}"}, status_code=400)
    if any(v["name"] == name for v in config.get("designed_voices", [])):
        return JSONResponse({"ok": False, "error": f"设计音色已存在: {name}"}, status_code=400)
    # 同名克隆音色不再报错：重新保存即重新冻结，原地覆盖，不产生多个克隆版

    config.setdefault("designed_voices", []).append({
        "name": name,
        "description": target.get("description", ""),
        "gender": target.get("gender", ""),
        "language": target.get("language", "en"),
        "instruct": target.get("instruct", ""),
        "created": time.time(),
    })
    config["candidate_voices"] = [c for c in config.get("candidate_voices", []) if c["name"] != name]
    _save_qwen_voice_config(config)
    # 保存即冻结：后台生成样本并转为同名克隆音色，音色从此稳定
    _enqueue_voice_freeze(name)
    return {"ok": True, "freezing": True}


@app.delete("/api/qwen_voices/candidates/{name}")
async def api_qwen_candidate_delete(name: str):
    """Discard a candidate voice."""
    config = _load_qwen_voice_config()
    if not any(c["name"] == name for c in config.get("candidate_voices", [])):
        return JSONResponse({"ok": False, "error": "未找到"}, status_code=404)
    config["candidate_voices"] = [c for c in config.get("candidate_voices", []) if c["name"] != name]
    _save_qwen_voice_config(config)
    _purge_preview_cache(name)
    return {"ok": True}


@app.post("/api/qwen_voices/candidates/delete_batch")
async def api_qwen_candidates_delete_batch(request: Request):
    """Bulk discard candidate voices (一键删除不喜欢的)."""
    data = await request.json()
    names = {str(n) for n in (data.get("names") or []) if n}
    if not names:
        return JSONResponse({"ok": False, "error": "未选择任何音色"}, status_code=400)
    config = _load_qwen_voice_config()
    kept = []
    removed = 0
    for c in config.get("candidate_voices", []):
        if c.get("name") in names:
            _purge_preview_cache(c["name"])
            removed += 1
        else:
            kept.append(c)
    config["candidate_voices"] = kept
    _save_qwen_voice_config(config)
    return {"ok": True, "removed": removed}


# --- 一键生成所有候选试听音频（后台线程 + 轮询进度）---

_CANDIDATE_PREVIEW_JOB: dict = {
    "running": False,
    "total": 0,
    "completed": 0,
    "done_names": [],
    "failed": [],
}
_CANDIDATE_PREVIEW_LOCK = threading.Lock()


def _run_candidate_previews(candidates: list[dict]) -> None:
    """后台线程：逐个为候选音色生成试听音频（跳过已缓存，串行 GPU 合成）。"""
    import sys as _sys
    _pipeline = str(PIPELINE_DIR)
    if _pipeline not in _sys.path:
        _sys.path.insert(0, _pipeline)
    from qwen_tts_engine import QwenTTSEngine

    config = load_config()
    model_path = config.get("qwen_model_path", r"H:\models\Qwen3-TTS-12Hz-0.6B-CustomVoice")
    base_model_path = config.get("qwen_base_model_path", r"H:\models\Qwen3-TTS-12Hz-1.7B-Base")
    voicedesign_model_path = config.get("qwen_voicedesign_model_path", r"H:\models\Qwen3-TTS-12Hz-1.7B-VoiceDesign")
    device = config.get("qwen_device", "cuda:0")
    engine = QwenTTSEngine(model_path, device, base_model_path, voicedesign_model_path)

    try:
        for c in candidates:
            with _CANDIDATE_PREVIEW_LOCK:
                if not _CANDIDATE_PREVIEW_JOB["running"]:  # 已被新任务取代/停止
                    return
                done = set(_CANDIDATE_PREVIEW_JOB["done_names"])
                failed_names = {f["name"] for f in _CANDIDATE_PREVIEW_JOB["failed"]}
            name = c.get("name", "")
            if name in done or name in failed_names:
                continue  # 启动时已预填（已有缓存）或此前已失败

            # 候选可能已被删除/保存后删除 → 跳过，不计入完成
            current_names = {x.get("name") for x in
                             _load_qwen_voice_config().get("candidate_voices", [])}
            if name not in current_names:
                with _CANDIDATE_PREVIEW_LOCK:
                    _CANDIDATE_PREVIEW_JOB["total"] = max(0, _CANDIDATE_PREVIEW_JOB["total"] - 1)
                continue

            language = "chinese" if c.get("language") == "zh" else "english"
            text = _PREVIEW_TEXTS.get(language, _PREVIEW_TEXTS["english"])
            cache_path = _preview_cache_path(name, language, text)
            if cache_path.exists():
                with _CANDIDATE_PREVIEW_LOCK:
                    _CANDIDATE_PREVIEW_JOB["done_names"].append(name)
                    _CANDIDATE_PREVIEW_JOB["completed"] += 1
                continue

            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with _TTS_SYNTH_LOCK:
                    if language == "chinese":
                        engine.synth_chinese(text, name, str(cache_path), rate="+0%")
                    else:
                        engine.synth_english(text, name, str(cache_path), rate="+0%")
                with _CANDIDATE_PREVIEW_LOCK:
                    _CANDIDATE_PREVIEW_JOB["done_names"].append(name)
                    _CANDIDATE_PREVIEW_JOB["completed"] += 1
            except Exception as e:  # noqa: BLE001 — 记录失败继续下一个
                with _CANDIDATE_PREVIEW_LOCK:
                    _CANDIDATE_PREVIEW_JOB["failed"].append(
                        {"name": name, "error": str(e)[:200]})
    finally:
        with _CANDIDATE_PREVIEW_LOCK:
            _CANDIDATE_PREVIEW_JOB["running"] = False


@app.post("/api/qwen_voices/candidates/previews/generate")
async def api_qwen_candidates_previews_generate():
    """一键为所有候选音色后台生成试听音频（已缓存的直接计入完成）。"""
    import sys as _sys
    _pipeline = str(PIPELINE_DIR)
    if _pipeline not in _sys.path:
        _sys.path.insert(0, _pipeline)

    with _CANDIDATE_PREVIEW_LOCK:
        if _CANDIDATE_PREVIEW_JOB["running"]:
            return JSONResponse({"ok": False, "error": "试听音频生成中，请稍候"}, status_code=409)
        candidates = [dict(c) for c in _load_qwen_voice_config().get("candidate_voices", [])]
        if not candidates:
            return JSONResponse({"ok": False, "error": "暂无候选音色，请先生成"}, status_code=400)

        cached_names = []
        for c in candidates:
            language = "chinese" if c.get("language") == "zh" else "english"
            text = _PREVIEW_TEXTS.get(language, _PREVIEW_TEXTS["english"])
            if _preview_cache_path(c.get("name", ""), language, text).exists():
                cached_names.append(c.get("name", ""))

        _CANDIDATE_PREVIEW_JOB.update({
            "running": True,
            "total": len(candidates),
            "completed": len(cached_names),
            "done_names": cached_names,
            "failed": [],
        })

    t = threading.Thread(target=_run_candidate_previews, args=(candidates,), daemon=True)
    t.start()
    return {"ok": True, "total": len(candidates), "cached": len(cached_names)}


@app.get("/api/qwen_voices/candidates/previews/status")
async def api_qwen_candidates_previews_status():
    """批量试听生成任务进度（前端轮询）。"""
    with _CANDIDATE_PREVIEW_LOCK:
        return {
            "running": _CANDIDATE_PREVIEW_JOB["running"],
            "total": _CANDIDATE_PREVIEW_JOB["total"],
            "completed": _CANDIDATE_PREVIEW_JOB["completed"],
            "done_names": list(_CANDIDATE_PREVIEW_JOB["done_names"]),
            "failed": list(_CANDIDATE_PREVIEW_JOB["failed"]),
        }


# --- 保存即冻结：设计音色自动转同名克隆音色 ---

# 冻结样本文本：多句日常口语（~12-18 秒），作克隆参考音频，覆盖多样语音场景
_FREEZE_SAMPLE_TEXTS = {
    "english": (
        "Good morning! It's really nice to see you again. "
        "I've been thinking about our conversation from last week, "
        "and I wanted to share some new ideas with you. "
        "Would you like to grab a coffee this afternoon? "
        "I always enjoy our little chats, and there's so much more to catch up on."
    ),
    "chinese": (
        "早上好！很高兴又见到你。我一直在想我们上次聊天的内容，"
        "有些新的想法想跟你分享。今天下午要不要一起喝杯咖啡？"
        "我很喜欢和你聊天，还有好多话题想跟你聊呢。"
    ),
}

# 冻结任务：name -> {status: pending|running|done|error, error}
_VOICE_FREEZE_JOBS: dict[str, dict] = {}
_VOICE_FREEZE_QUEUE: list[dict] = []
_VOICE_FREEZE_LOCK = threading.Lock()
_VOICE_FREEZE_THREAD_ACTIVE = False


def _enqueue_voice_freeze(name: str) -> None:
    """入队冻结任务并确保 worker 线程在跑。"""
    global _VOICE_FREEZE_THREAD_ACTIVE
    with _VOICE_FREEZE_LOCK:
        _VOICE_FREEZE_QUEUE.append({"name": name})
        _VOICE_FREEZE_JOBS[name] = {"status": "pending", "error": ""}
        if not _VOICE_FREEZE_THREAD_ACTIVE:
            _VOICE_FREEZE_THREAD_ACTIVE = True
            t = threading.Thread(target=_run_voice_freeze_worker, daemon=True)
            t.start()


def _run_voice_freeze_worker() -> None:
    """串行处理冻结队列（GPU 合成独占，与试听任务共用 _TTS_SYNTH_LOCK）。"""
    global _VOICE_FREEZE_THREAD_ACTIVE
    try:
        while True:
            with _VOICE_FREEZE_LOCK:
                if not _VOICE_FREEZE_QUEUE:
                    _VOICE_FREEZE_THREAD_ACTIVE = False
                    return
                job = _VOICE_FREEZE_QUEUE.pop(0)
                name = job["name"]
                _VOICE_FREEZE_JOBS[name] = {"status": "running", "error": ""}
            try:
                _freeze_voice_impl(name)
                with _VOICE_FREEZE_LOCK:
                    _VOICE_FREEZE_JOBS[name] = {"status": "done", "error": ""}
            except Exception as e:  # noqa: BLE001 — 失败时音色保留在 designed_voices 仍可用
                with _VOICE_FREEZE_LOCK:
                    _VOICE_FREEZE_JOBS[name] = {"status": "error", "error": str(e)[:300]}
    except Exception:  # noqa: BLE001 — 兜底，绝不常驻死线程
        with _VOICE_FREEZE_LOCK:
            _VOICE_FREEZE_THREAD_ACTIVE = False


def _freeze_voice_impl(name: str) -> None:
    """用 VoiceDesign 生成一段多样本音频 → 注册为同名克隆音色（冻结音色身份）。

    设计音色靠 instruct 文字描述定义"音色空间区域"，每次采样结果都会漂移；
    冻结后以 ref_audio 声学特征为硬锚点，跨句/跨次运行音色稳定。
    成功：designed_voices → custom_voices（同名覆盖，不产生多版本）。
    失败：音色保留在 designed_voices（仍以设计模式可用，只是不稳定）。
    """
    import subprocess as _sp
    import sys as _sys
    _pipeline = str(PIPELINE_DIR)
    if _pipeline not in _sys.path:
        _sys.path.insert(0, _pipeline)
    from qwen_tts_engine import QwenTTSEngine

    config = _load_qwen_voice_config()
    meta = next((v for v in config.get("designed_voices", []) if v["name"] == name), None)
    if not meta:
        # 内置设计音色（Ella/Maya/Chloe/Hazel）不在 designed_voices，从内置表回退查找
        from qwen_tts_engine import DESIGNED_VOICES_BUILTIN
        meta = next((v for v in DESIGNED_VOICES_BUILTIN if v["name"] == name), None)
    if not meta:
        raise RuntimeError("音色已不存在（可能被删除），冻结中止")

    # 键名兼容：内置条目用 desc/lang，用户条目用 description/language
    description = meta.get("description", "") or meta.get("desc", "")
    lang_short = meta.get("language", "") or meta.get("lang", "en")
    gender = meta.get("gender", "")
    instruct = meta.get("instruct", "")

    language = "chinese" if lang_short == "zh" else "english"
    sample_text = _FREEZE_SAMPLE_TEXTS.get(language, _FREEZE_SAMPLE_TEXTS["english"])

    safe = "".join(c for c in name if c.isalnum() or c in "-_") or "voice"
    CUSTOM_VOICES_DIR.mkdir(parents=True, exist_ok=True)
    sample_mp3 = CUSTOM_VOICES_DIR / f"{safe}_freeze_sample.mp3"
    wav_path = CUSTOM_VOICES_DIR / f"{safe}_freeze.wav"

    cfg = load_config()
    engine = QwenTTSEngine(
        cfg.get("qwen_model_path", r"H:\models\Qwen3-TTS-12Hz-0.6B-CustomVoice"),
        cfg.get("qwen_device", "cuda:0"),
        cfg.get("qwen_base_model_path", r"H:\models\Qwen3-TTS-12Hz-1.7B-Base"),
        cfg.get("qwen_voicedesign_model_path", r"H:\models\Qwen3-TTS-12Hz-1.7B-VoiceDesign"),
    )
    with _TTS_SYNTH_LOCK:
        if language == "chinese":
            engine.synth_chinese(sample_text, name, str(sample_mp3), rate="+0%")
        else:
            engine.synth_english(sample_text, name, str(sample_mp3), rate="+0%")

    # mp3 → wav（克隆参考音频用 wav 最稳）
    _sp.run(
        ["ffmpeg", "-y", "-i", str(sample_mp3), "-ar", "24000", "-ac", "1", str(wav_path)],
        check=True, capture_output=True)
    sample_mp3.unlink(missing_ok=True)

    # 注册为克隆音色（同名覆盖）并从 designed_voices 移除
    config = _load_qwen_voice_config()
    config["custom_voices"] = [v for v in config.get("custom_voices", []) if v["name"] != name]
    config["custom_voices"].append({
        "name": name,
        "description": description or f"设计音色 ({gender or '?'})",
        "gender": gender,
        "language": lang_short,
        "ref_audio": str(wav_path),
        "ref_text": sample_text,
        "instruct": instruct,  # 保留以便将来重新冻结
        "frozen_from": "designed",
        "created": time.time(),
    })
    config["designed_voices"] = [v for v in config.get("designed_voices", []) if v["name"] != name]
    _save_qwen_voice_config(config)
    _purge_preview_cache(name)  # 旧试听缓存是设计模式产物，作废用克隆重生成


@app.get("/api/qwen_voices/freeze/status")
async def api_qwen_freeze_status():
    """冻结任务进度（前端轮询）。"""
    with _VOICE_FREEZE_LOCK:
        return {"jobs": {k: dict(v) for k, v in _VOICE_FREEZE_JOBS.items()}}


def _auto_freeze_pending_designed_voices() -> None:
    """启动时把尚未转换为克隆音色的设计音色入队后台冻结（一次性）。

    覆盖：内置 4 个英文女声（Ella/Maya/Chloe/Hazel）+ designed_voices 中
    残留的未冻结条目（此前冻结失败的会借此重试）。
    已冻结（custom_voices 同名存在）或已删除（builtin_voice_dismissed）的跳过。
    """
    import sys as _sys
    _pipeline = str(PIPELINE_DIR)
    if _pipeline not in _sys.path:
        _sys.path.insert(0, _pipeline)
    from qwen_tts_engine import DESIGNED_VOICES_BUILTIN

    config = _load_qwen_voice_config()
    custom_names = {v.get("name") for v in config.get("custom_voices", [])}
    dismissed = set(config.get("builtin_voice_dismissed", []))
    pending: list[str] = []
    for dv in DESIGNED_VOICES_BUILTIN:
        if dv["name"] not in custom_names and dv["name"] not in dismissed:
            pending.append(dv["name"])
    for dv in config.get("designed_voices", []):
        if dv.get("name") and dv["name"] not in custom_names:
            pending.append(dv["name"])
    if pending:
        print(f"[startup] 设计音色自动冻结入队: {', '.join(pending)}（后台生成样本约 30-60 秒/个）")
        for n in pending:
            _enqueue_voice_freeze(n)


# ===========================================================================
# Kokoro TTS Voices (本地引擎，固定音色清单 + 试听)
# ===========================================================================

def _kokoro_preview_cache_path(speaker: str, text: str) -> Path:
    import hashlib
    safe = "".join(c for c in speaker if c.isalnum() or c in "-_") or "voice"
    h = hashlib.md5(f"{speaker}|{text}".encode("utf-8")).hexdigest()[:8]
    return VOICE_PREVIEWS_DIR / f"kokoro__{safe}__{h}.mp3"


@app.get("/api/kokoro_voices/speakers")
async def api_kokoro_speakers():
    """List all Kokoro voices with cached flag (未缓存音色本机无法自动下载)."""
    import sys as _sys
    _pipeline = str(PIPELINE_DIR)
    if _pipeline not in _sys.path:
        _sys.path.insert(0, _pipeline)
    from tts_engine import get_all_kokoro_voices
    return {"speakers": get_all_kokoro_voices()}


async def _kokoro_preview_common(speaker: str, text: str, regenerate: bool):
    """Serve cached Kokoro preview if available, else synthesize & cache."""
    cache_path = _kokoro_preview_cache_path(speaker, text)
    if cache_path.exists() and not regenerate:
        return FileResponse(str(cache_path), media_type="audio/mpeg")

    import sys as _sys
    _pipeline = str(PIPELINE_DIR)
    if _pipeline not in _sys.path:
        _sys.path.insert(0, _pipeline)

    try:
        from tts_engine import TTSEngine

        def _synth() -> None:
            engine = TTSEngine()
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with _TTS_SYNTH_LOCK:
                engine.synth_english(text, speaker, str(cache_path), rate="+0%")

        await asyncio.to_thread(_synth)
        return FileResponse(str(cache_path), media_type="audio/mpeg",
                            filename="preview.mp3",
                            headers={"Content-Disposition": "attachment; filename=preview.mp3"})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.post("/api/kokoro_voices/preview")
async def api_kokoro_preview(request: Request):
    """Preview a Kokoro voice: serve cached audio if available, else generate & cache."""
    data = await request.json()
    speaker = data.get("speaker", "af_sarah")
    text = data.get("text", "") or _PREVIEW_TEXTS.get("english", "Hello, this is a voice preview test.")
    regenerate = bool(data.get("regenerate", False))
    return await _kokoro_preview_common(speaker, text, regenerate)


@app.get("/api/kokoro_voices/preview/{voice}")
async def api_kokoro_preview_cached(voice: str, language: str = "english"):
    """Serve a cached preview if it exists (instant playback), 404 otherwise."""
    text = _PREVIEW_TEXTS.get(language, _PREVIEW_TEXTS["english"])
    p = _kokoro_preview_cache_path(voice, text)
    if not p.exists():
        return JSONResponse({"ok": False, "error": "no cached preview"}, status_code=404)
    return FileResponse(str(p), media_type="audio/mpeg")


# Kokoro voice defaults config (性别自动分配的默认音色，与 Qwen/MOSS 对齐)


def _load_kokoro_voice_config() -> dict:
    """Load kokoro_voice_config.json with defaults (文件首次保存时创建)."""
    defaults = {
        "default_male": "am_adam",
        "default_female": "af_sarah",
        "default_host_female": "af_sky",
    }
    if KOKORO_VOICE_CONFIG_PATH.exists():
        try:
            saved = json.loads(KOKORO_VOICE_CONFIG_PATH.read_text(encoding="utf-8"))
            for k in defaults:
                if k in saved:
                    defaults[k] = saved[k]
        except (json.JSONDecodeError, OSError):
            pass
    return defaults


def _save_kokoro_voice_config(config: dict) -> None:
    KOKORO_VOICE_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    KOKORO_VOICE_CONFIG_PATH.write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


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


@app.get("/api/kokoro_voices/defaults")
async def api_kokoro_defaults_get():
    """Get Kokoro default voice configuration."""
    return _load_kokoro_voice_config()


@app.put("/api/kokoro_voices/defaults")
async def api_kokoro_defaults_put(request: Request):
    """Update Kokoro default voice configuration."""
    data = await request.json()
    config = _load_kokoro_voice_config()
    for key in ["default_male", "default_female", "default_host_female"]:
        if key in data:
            config[key] = data[key]
    _save_kokoro_voice_config(config)
    return {"ok": True, "config": config}


# ===========================================================================
# MOSS-TTS Voice Management
# ===========================================================================


def _moss_preview_cache_path(speaker: str, language: str, text: str) -> Path:
    import hashlib
    safe = "".join(c for c in speaker if c.isalnum() or c in "-_") or "voice"
    h = hashlib.md5(f"{speaker}|{language}|{text}".encode("utf-8")).hexdigest()[:8]
    return MOSS_PREVIEWS_DIR / f"{safe}__{language}__{h}.mp3"


def _purge_moss_preview_cache(speaker: str) -> None:
    if not MOSS_PREVIEWS_DIR.exists():
        return
    safe = "".join(c for c in speaker if c.isalnum() or c in "-_") or "voice"
    for p in MOSS_PREVIEWS_DIR.glob(f"{safe}__*.mp3"):
        p.unlink(missing_ok=True)


def _load_moss_voice_config() -> dict:
    defaults = {
        "default_male": "Adam",
        "default_female": "Ava",
        "default_host_female": "Bella",
        "custom_voices": [],
    }
    if MOSS_VOICE_CONFIG_PATH.exists():
        try:
            saved = json.loads(MOSS_VOICE_CONFIG_PATH.read_text(encoding="utf-8"))
            for k in defaults:
                if k in saved:
                    defaults[k] = saved[k]
        except (json.JSONDecodeError, OSError):
            pass
    return defaults


def _save_moss_voice_config(config: dict) -> None:
    MOSS_VOICE_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    MOSS_VOICE_CONFIG_PATH.write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


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


@app.get("/api/moss_voices/speakers")
async def api_moss_speakers():
    """List all available MOSS voices (preset + custom)."""
    import sys as _sys
    _pipeline = str(PIPELINE_DIR)
    if _pipeline not in _sys.path:
        _sys.path.insert(0, _pipeline)
    from moss_tts_engine import MOSS_PRESET_VOICES, get_all_moss_voices
    return {"speakers": get_all_moss_voices(), "presets": MOSS_PRESET_VOICES}


@app.post("/api/moss_voices/preview")
async def api_moss_preview(request: Request):
    """Preview a MOSS voice: serve cached audio if available, else generate & cache."""
    import sys as _sys
    _pipeline = str(PIPELINE_DIR)
    if _pipeline not in _sys.path:
        _sys.path.insert(0, _pipeline)

    data = await request.json()
    speaker = data.get("speaker", "Ava")
    language = data.get("language", "english")
    text = data.get("text", "") or _PREVIEW_TEXTS.get(language, _PREVIEW_TEXTS["english"])
    regenerate = bool(data.get("regenerate", False))

    cache_path = _moss_preview_cache_path(speaker, language, text)
    if cache_path.exists() and not regenerate:
        return FileResponse(str(cache_path), media_type="audio/mpeg")

    config = load_config()
    model_path = config.get("moss_model_path") or r"H:\models\MOSS-TTS-Nano-Model"
    tokenizer_path = config.get("moss_tokenizer_path") or r"H:\models\MOSS-Audio-Tokenizer-Nano"
    device = config.get("moss_device") or "cpu"
    repo_dir = config.get("moss_repo_dir") or r"H:\models\MOSS-TTS-Nano"
    try:
        moss_temperature = float(config.get("moss_tts_temperature") or 0.8)
    except (TypeError, ValueError):
        moss_temperature = 0.8
    try:
        moss_retry = int(config.get("moss_tts_retry") or 3)
    except (TypeError, ValueError):
        moss_retry = 3
    try:
        moss_top_p = float(config.get("moss_tts_top_p") or 0.95)
    except (TypeError, ValueError):
        moss_top_p = 0.95
    try:
        moss_top_k = int(config.get("moss_tts_top_k") or 25)
    except (TypeError, ValueError):
        moss_top_k = 25
    try:
        moss_rep_penalty = float(config.get("moss_tts_rep_penalty") or 1.2)
    except (TypeError, ValueError):
        moss_rep_penalty = 1.2
    try:
        moss_text_temperature = float(config.get("moss_tts_text_temperature") or 1.0)
    except (TypeError, ValueError):
        moss_text_temperature = 1.0
    moss_greedy = bool(config.get("moss_tts_greedy", False))

    try:
        from moss_tts_engine import MossTTSEngine

        def _synth() -> None:
            engine = MossTTSEngine(model_path, device, tokenizer_path, repo_dir,
                                   temperature=moss_temperature, retry=moss_retry,
                                   top_p=moss_top_p, top_k=moss_top_k,
                                   rep_penalty=moss_rep_penalty,
                                   text_temperature=moss_text_temperature,
                                   greedy=moss_greedy)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            out_path = str(cache_path)
            with _TTS_SYNTH_LOCK:
                if language == "chinese":
                    engine.synth_chinese(text, speaker, out_path, rate="+0%")
                else:
                    engine.synth_english(text, speaker, out_path, rate="+0%")

        await asyncio.to_thread(_synth)
        return FileResponse(str(cache_path), media_type="audio/mpeg",
                            filename="preview.mp3",
                            headers={"Content-Disposition": "attachment; filename=preview.mp3"})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.get("/api/moss_voices/preview/{voice}")
async def api_moss_preview_cached(voice: str, language: str = "english"):
    """Serve a cached preview if it exists (instant playback), 404 otherwise."""
    text = _PREVIEW_TEXTS.get(language, _PREVIEW_TEXTS["english"])
    p = _moss_preview_cache_path(voice, language, text)
    if not p.exists():
        return JSONResponse({"ok": False, "error": "no cached preview"}, status_code=404)
    return FileResponse(str(p), media_type="audio/mpeg")


@app.get("/api/moss_voices/defaults")
async def api_moss_defaults_get():
    """Get default voice configuration."""
    return _load_moss_voice_config()


@app.put("/api/moss_voices/defaults")
async def api_moss_defaults_put(request: Request):
    """Update default voice configuration."""
    data = await request.json()
    config = _load_moss_voice_config()
    for key in ["default_male", "default_female", "default_host_female"]:
        if key in data:
            config[key] = data[key]
    _save_moss_voice_config(config)
    return {"ok": True, "config": config}


@app.post("/api/moss_voices/custom")
async def api_moss_custom_create(
    name: str = Form(""),
    gender: str = Form(""),
    language: str = Form("english"),
    ref_text: str = Form(""),
    ref_audio: UploadFile = File(...),
):
    """Create a custom cloned voice from reference audio."""
    if not name or not ref_text:
        return JSONResponse({"ok": False, "error": "缺少名称或参考文字"}, status_code=400)

    safe_name = "".join(c for c in name if c.isalnum() or c in "-_") or "custom"
    MOSS_VOICES_DIR.mkdir(parents=True, exist_ok=True)
    audio_path = MOSS_VOICES_DIR / f"{safe_name}.wav"
    content = await ref_audio.read()
    audio_path.write_bytes(content)

    config = _load_moss_voice_config()
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
    _save_moss_voice_config(config)
    return {"ok": True, "name": name}


@app.delete("/api/moss_voices/custom/{name}")
async def api_moss_custom_delete(name: str):
    """Delete a custom voice."""
    config = _load_moss_voice_config()
    target = None
    for v in config["custom_voices"]:
        if v["name"] == name:
            target = v
            break
    if not target:
        return JSONResponse({"ok": False, "error": "未找到"}, status_code=404)
    audio_path = Path(target.get("ref_audio", ""))
    if audio_path.exists():
        audio_path.unlink()
    _purge_moss_preview_cache(name)
    config["custom_voices"] = [v for v in config["custom_voices"] if v["name"] != name]
    _save_moss_voice_config(config)
    return {"ok": True}


@app.put("/api/character_library/{lib_id}/moss_voice")
async def api_library_set_moss_voice(lib_id: str, request: Request):
    """Set MOSS TTS voice for a library character."""
    lib_dir = LIBRARY_DIR / lib_id
    if not lib_dir.exists():
        return JSONResponse({"ok": False, "error": "未找到"}, status_code=404)
    meta_path = lib_dir / "meta.json"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return JSONResponse({"ok": False, "error": "meta.json 读取失败"}, status_code=500)
    data = await request.json()
    meta["moss_voice"] = data.get("moss_voice", "")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"ok": True, "meta": meta}


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

# Background image generation status tracking
_gen_status: dict[str, dict] = {}  # lib_id → {status, poses, error, started_at}


def _generate_char_images(lib_id: str, description: str, structure: str):
    """Background thread: generate pose atlas via MCP, download, split into poses.

    用页面级独立 MCP 会话（人物素材库页专属 token → 激活模式 mcp_tokens →
    本地检测），与 pipeline 全局会话完全隔离，pipeline 运行中也可安全生成。
    """
    import shutil
    lib_dir = LIBRARY_DIR / lib_id
    if not lib_dir.exists():
        _gen_status[lib_id] = {"status": "error", "error": "角色目录不存在", "poses": []}
        return

    _gen_status[lib_id] = {"status": "generating", "error": "", "poses": [], "started_at": time.time()}

    try:
        # Add pipeline to sys.path for atlas_split imports
        _pipeline = str(PIPELINE_DIR)
        if _pipeline not in sys.path:
            sys.path.insert(0, _pipeline)
        from atlas_split import split_atlas

        # 独立 MCP 会话（不动 pipeline 全局会话）
        resolved = resolve_page_tokens("characters")
        if not resolved["tokens"]:
            _gen_status[lib_id] = {"status": "error",
                                   "error": "未配置 MCP Token（本页专属 / 模式配置 / 本地检测均为空）",
                                   "poses": []}
            return
        print(f"  [CharGen] MCP token: {MCP_SOURCE_LABELS[resolved['source']]} "
              f"×{len(resolved['tokens'])} ({mask_token(resolved['tokens'][0])})")
        mcp = PageMcpSession(resolved["tokens"]).initialize()

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
        result = mcp.call_tool("generate_image", gen_params)
        task_id = mcp.parse_task_id(result)
        if not task_id:
            _gen_status[lib_id] = {"status": "error", "error": "MCP 返回无 task_id", "poses": []}
            return

        data = mcp.poll_task(task_id, interval=10, max_wait=600)
        url = data.get("url", "")
        if not url:
            _gen_status[lib_id] = {"status": "error", "error": "MCP 生成失败：无图片 URL", "poses": []}
            return

        if structure == "quest":
            # Download atlas, split into 8 poses（分隔线检测 + 残边修剪，无缝时回退等分）
            atlas_path = str(lib_dir / f"pose_atlas_{src_key}.png")
            mcp.download_file(url, atlas_path)
            print(f"  [CharGen] Downloaded atlas for {lib_id}")

            pose_files = [f"pose_{src_key}_{idx}.png" for idx in range(8)]
            split_atlas(atlas_path, 4, 2,
                        [str(lib_dir / f) for f in pose_files],
                        log_prefix="[CharGen]")
            print(f"  [CharGen] Split into {len(pose_files)} poses for {lib_id}")
        else:
            # Original: save as char_scene.png
            atlas_path = str(lib_dir / "char_scene.png")
            mcp.download_file(url, atlas_path)
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


@app.post("/api/character_library/create")
async def api_library_create(
    name: str = Form(""),
    description: str = Form(""),
    gender: str = Form(""),
    structure: str = Form("quest"),
    qwen_speaker: str = Form(""),
    moss_voice: str = Form(""),
    kokoro_voice: str = Form(""),
    image: UploadFile | None = File(None),
):
    """Manually create a new character in the library.

    Empty description = 仅音色+性别角色 (voice+gender only — appearance is
    re-created by the LLM for every video; no images stored).
    """
    if not name.strip():
        return JSONResponse({"ok": False, "error": "名称不能为空"}, status_code=400)

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
        "moss_voice": moss_voice.strip(),
        "kokoro_voice": kokoro_voice.strip(),
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
    for key in ("name", "description", "gender", "structure", "qwen_speaker", "moss_voice", "kokoro_voice"):
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

    # Start background thread (token 由 worker 内 resolve_page_tokens 解析:
    # 本页专属 → 激活模式 mcp_tokens → 本地检测)
    thread = threading.Thread(
        target=_generate_char_images,
        args=(lib_id, description, structure),
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
    # 首次运行：从 legacy default.json 迁移生成 3 个模式配置文件
    load_all_mode_configs()
    # 设计音色自动冻结：内置英文女声 + 残留未冻结的设计音色 → 后台转为克隆音色（一次性）
    try:
        _auto_freeze_pending_designed_voices()
    except Exception as e:  # noqa: BLE001 — 冻结失败不阻塞启动，下次启动重试
        print(f"[startup] 设计音色自动冻结入队失败: {e}")
