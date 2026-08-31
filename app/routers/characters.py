"""Characters API — 角色复用来源 / 角色套装 / 素材库 CRUD + AI 姿势图生成 + 三引擎音色绑定."""
import json
import sys
import time
from pathlib import Path

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from ..config_manager import (
    MODES, find_run_dir, get_active_mode, iter_run_dirs, load_config,
)
from ..library_io import write_library_meta
from ..page_mcp import (
    SOURCE_LABELS as MCP_SOURCE_LABELS,
    PageMcpSession, mask_token, resolve_page_tokens,
)
from ..paths import CHARACTER_SETS_PATH, LIBRARY_DIR, PIPELINE_DIR

router = APIRouter()


# ===========================================================================
# Character reuse API
# ===========================================================================

@router.get("/api/character_sources")
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


@router.get("/api/character_sets")
async def api_char_sets_list():
    """List all saved character sets (whole-cast presets), newest first."""
    sets = sorted(_load_char_sets(), key=lambda s: s.get("created", 0), reverse=True)
    return {"sets": sets}


@router.post("/api/character_sets/save")
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


@router.delete("/api/character_sets/{set_id}")
async def api_char_sets_delete(set_id: str):
    sets = _load_char_sets()
    remaining = [s for s in sets if s.get("id") != set_id]
    if len(remaining) == len(sets):
        return JSONResponse({"ok": False, "error": "未找到"}, status_code=404)
    _save_char_sets(remaining)
    return {"ok": True}


# ===========================================================================
# Character Library API
# ===========================================================================


@router.get("/api/character_library")
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


@router.post("/api/character_library/save")
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


@router.delete("/api/character_library/{lib_id}")
async def api_library_delete(lib_id: str):
    import shutil
    lib_dir = LIBRARY_DIR / lib_id
    if lib_dir.exists() and lib_dir.is_dir():
        if str(lib_dir.resolve()).startswith(str(LIBRARY_DIR.resolve())):
            shutil.rmtree(str(lib_dir))
            return {"ok": True}
    return JSONResponse({"ok": False, "error": "未找到"}, status_code=404)


@router.get("/api/character_library/{lib_id}/image")
async def api_library_image(lib_id: str):
    thumb = LIBRARY_DIR / lib_id / "thumb.png"
    if not thumb.exists():
        return JSONResponse({"error": "Not found"}, status_code=404)
    return FileResponse(str(thumb), media_type="image/png")


@router.put("/api/character_library/{lib_id}/voice")
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
    write_library_meta(lib_id, meta)
    return {"ok": True, "meta": meta}


@router.put("/api/character_library/{lib_id}/kokoro_voice")
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
    write_library_meta(lib_id, meta)
    return {"ok": True, "meta": meta}


# ===========================================================================
@router.put("/api/character_library/{lib_id}/moss_voice")
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
    write_library_meta(lib_id, meta)
    return {"ok": True, "meta": meta}


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


@router.post("/api/character_library/create")
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


@router.put("/api/character_library/{lib_id}")
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
    write_library_meta(lib_id, meta)
    return {"ok": True, "meta": meta}


@router.post("/api/character_library/{lib_id}/image")
async def api_library_upload_image(lib_id: str, image: UploadFile = File(...)):
    """Upload or replace character thumbnail image."""
    lib_dir = LIBRARY_DIR / lib_id
    if not lib_dir.exists():
        return JSONResponse({"ok": False, "error": "未找到"}, status_code=404)
    content = await image.read()
    (lib_dir / "thumb.png").write_bytes(content)
    return {"ok": True, "image_url": f"/api/character_library/{lib_id}/image"}


@router.post("/api/character_library/{lib_id}/generate_images")
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


@router.get("/api/character_library/{lib_id}/generation_status")
async def api_library_gen_status(lib_id: str):
    """Poll image generation status."""
    status = _gen_status.get(lib_id, {"status": "idle", "poses": [], "error": ""})
    return status


@router.get("/api/character_library/{lib_id}/poses")
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


@router.get("/api/character_library/{lib_id}/poses/{filename}")
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


