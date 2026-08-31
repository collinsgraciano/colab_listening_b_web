"""HTML 页面路由 — 全部 14 个管理界面页面（渲染层，依赖最多）."""
import json
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from ..config_manager import (
    GROUP_META, MODES, MODE_LABELS, MODE_SHORT_LABELS, PARAM_SPEC,
    RECYCLE_DIRNAME, LEGACY_RECYCLE_DIRNAME,
    effective_param_spec, find_run_dir, get_active_mode, get_provider_options,
    iter_run_dirs, list_presets, load_all_mode_configs, load_config,
    load_llm_providers, load_mode_config, set_active_mode,
)
from ..library_io import list_library_chars
from ..mode_test_service import get_mode_test_service
from ..paths import TRASH_META_FILENAME
from ..pipeline_service import get_service
from ..templating import templates
import style_manager as style_lib
import subtitle_style_manager as subtitle_style_lib
from .ai_test import _load_ai_test_config
from .subtitle_styles import _current_subtitle_style_ctx

router = APIRouter()


# ===========================================================================
# Page routes
# ===========================================================================

@router.get("/", response_class=HTMLResponse)
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


@router.get("/config", response_class=HTMLResponse)
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


@router.get("/topics", response_class=HTMLResponse)
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


@router.get("/runs/{name}/gallery", response_class=HTMLResponse)
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


@router.get("/runs", response_class=HTMLResponse)
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


@router.get("/mode-test", response_class=HTMLResponse)
async def mode_test_page(request: Request):
    service = get_mode_test_service()
    return templates.TemplateResponse(request, "mode_test.html", {
        "active_page": "mode_test",
        "mode_labels": MODE_LABELS,
        "status": service.full_status(),
    })


@router.get("/styles", response_class=HTMLResponse)
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


@router.get("/subtitle-styles", response_class=HTMLResponse)
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


@router.get("/scripts", response_class=HTMLResponse)
async def scripts_page(request: Request):
    config = load_config()
    return templates.TemplateResponse(request, "scripts.html", {
        "config": config,
        "active_page": "scripts",
    })


@router.get("/voices", response_class=HTMLResponse)
async def voices_page(request: Request):
    """QwenTTS voice management page."""
    config = load_config()
    # Fetch library characters
    library_chars = list_library_chars()
    return templates.TemplateResponse(request, "voices.html", {
        "config": config,
        "library_chars": library_chars,
        "active_page": "voices",
    })


@router.get("/kokoro_voices", response_class=HTMLResponse)
async def kokoro_voices_page(request: Request):
    """Kokoro voice management page."""
    config = load_config()
    library_chars = list_library_chars()
    return templates.TemplateResponse(request, "kokoro_voices.html", {
        "config": config,
        "library_chars": library_chars,
        "active_page": "kokoro_voices",
    })


# ===========================================================================
@router.get("/moss_voices", response_class=HTMLResponse)
async def moss_voices_page(request: Request):
    """MOSS-TTS-Nano voice management page."""
    config = load_config()
    library_chars = list_library_chars()
    return templates.TemplateResponse(request, "moss_voices.html", {
        "config": config,
        "library_chars": library_chars,
        "active_page": "moss_voices",
    })


# ===========================================================================
@router.get("/ai_test", response_class=HTMLResponse)
async def ai_test_page(request: Request):
    config = load_config()
    ai_cfg = _load_ai_test_config()
    return templates.TemplateResponse(request, "ai_test.html", {
        "config": config,
        "active_page": "ai_test",
        "system_prompt": ai_cfg.get("system_prompt", ""),
    })


# ===========================================================================
@router.get("/characters", response_class=HTMLResponse)
async def characters_page(request: Request):
    """Character library management page."""
    config = load_config()
    library_chars = list_library_chars(include_pose_count=True)
    return templates.TemplateResponse(request, "characters.html", {
        "config": config,
        "library_chars": library_chars,
        "active_page": "characters",
    })


# ===========================================================================
