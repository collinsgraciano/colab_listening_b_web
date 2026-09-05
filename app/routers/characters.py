"""Characters API — 角色复用来源 / 角色套装 / 素材库 CRUD + AI 姿势图生成 + 三引擎音色绑定."""
import json
import time
from pathlib import Path

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from ..config_manager import (
    MODES, find_run_dir, get_active_mode, iter_run_dirs, load_config,
    resolve_provider, structure_family,
)
from ..library_io import list_library_chars, write_library_meta
from ..page_mcp import (
    SOURCE_LABELS as MCP_SOURCE_LABELS,
    PageMcpSession, mask_token, resolve_page_tokens,
)
from ..paths import CHARACTER_SETS_PATH, LIBRARY_DIR

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
                "has_host_bg": (img_dir / "host_bg.png").exists(),
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
            meta["has_host_bg"] = (d / "host_bg.png").exists()
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
    structure = structure_family(data.get("structure", "quest"))

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
        shutil.copy2(str(thumb_src), str(lib_dir / thumb_src.name))

    # 序列帧 clips（sprite_sequence 运行产出；与 structure 无关，cutout/quest 通用）
    clips_saved = 0
    run_clips_mp = src_img_dir / "sprite_clips.json"
    if run_clips_mp.exists():
        try:
            run_clips = json.loads(run_clips_mp.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            run_clips = {}
        char_clips = (run_clips.get("chars") or {}).get(char_key) or {}
        lib_actions = {}
        for action, frames in char_clips.items():
            names = []
            for j, fp in enumerate(frames):
                src = Path(fp)
                if not src.exists():
                    continue
                dst = lib_dir / f"clip_{action}_{j:02d}.png"
                shutil.copy2(str(src), str(dst))
                names.append(dst.name)
            if len(names) >= 4:
                lib_actions[action] = names
                clips_saved += 1
        if lib_actions:
            (lib_dir / "sprite_clips.json").write_text(
                json.dumps({"version": 1,
                            "fps": int(run_clips.get("fps", 12)),
                            "source": run_clips.get("source", "video_frames"),
                            "desc_snapshot": desc,
                            "actions": lib_actions},
                           ensure_ascii=False, indent=2),
                encoding="utf-8")

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
        "sprite_clips": clips_saved,
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


@router.get("/api/character_library/{lib_id}/host_bg")
async def api_library_host_bg(lib_id: str):
    """Serve the library entry's host studio background (主持人演播室背景预览)."""
    lib_dir = LIBRARY_DIR / lib_id
    if (not lib_dir.exists() or not lib_dir.is_dir()
            or not str(lib_dir.resolve()).startswith(str(LIBRARY_DIR.resolve()))):
        return JSONResponse({"error": "Not found"}, status_code=404)
    bg = lib_dir / "host_bg.png"
    if not bg.exists():
        return JSONResponse({"error": "Not found"}, status_code=404)
    return FileResponse(str(bg), media_type="image/png",
                        headers={"Cache-Control": "no-cache"})


@router.get("/api/character_library/{lib_id}/atlas")
async def api_library_atlas(lib_id: str):
    """Serve the character's full-size image for the atlas viewer modal.

    优先级：pose_atlas_{source_key}.png → 任意 pose_atlas_*.png →
    char_scene.png（original 结构）→ thumb.png。
    """
    lib_dir = LIBRARY_DIR / lib_id
    if not lib_dir.exists() or not lib_dir.is_dir():
        return JSONResponse({"error": "Not found"}, status_code=404)
    source_key = "char_a"
    meta_path = lib_dir / "meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            source_key = meta.get("source_key") or "char_a"
        except (json.JSONDecodeError, OSError):
            pass
    candidates = [lib_dir / f"pose_atlas_{source_key}.png"]
    candidates += sorted(lib_dir.glob("pose_atlas_*.png"))
    candidates.append(lib_dir / "char_scene.png")
    for p in candidates:
        if p.exists():
            return FileResponse(str(p), media_type="image/png")
    thumb = lib_dir / "thumb.png"
    if thumb.exists():
        return FileResponse(str(thumb), media_type="image/png")
    return JSONResponse({"error": "Not found"}, status_code=404)


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


# 参考图风格串（MCP 生成与外部工具提示词输出共用，保证逐字一致）
_CHAR_IMG_STYLE = ("3D cartoon style, Pixar-like, warm soft lighting, "
                   "cel-shaded with thin clean black outline, "
                   "vibrant saturated colors, smooth surfaces")


def _build_image_prompt(structure: str, description: str) -> str:
    """参考图生成 prompt（MCP 自动生成与 /gen_prompts 提示词输出共用）。"""
    if structure == "quest":
        # 4×2 grid atlas (8 poses)
        return (
            f"4x2 grid character pose sheet, eight poses of the same character, "
            f"{description}, "
            f"top row left to right: speaking with mouth open, listening with slight smile, "
            f"thinking with hand on chin, surprised with raised eyebrows, "
            f"bottom row left to right: nodding in agreement, waving right hand, "
            f"pointing forward, laughing with eyes closed, "
            f"medium waist-up shot, every pose fully inside its own cell with generous "
            f"empty margin on all sides, both shoulders fully visible, all arms and hands "
            f"completely within the cell borders, no body part cropped or cut off at the "
            f"cell edges, all eight poses same character same outfit, "
            f"plain white background, {_CHAR_IMG_STYLE}, "
            f"no props, no objects, no scene, no text"
        )
    # Original structure: single character scene image
    return (
        f"{description}, standing pose, medium waist-up shot, "
        f"both shoulders fully visible, all arms and hands completely within "
        f"the frame, no body part cropped or cut off at the image edges, "
        f"plain white background, {_CHAR_IMG_STYLE}, "
        f"no props, no objects, no scene, no text"
    )


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

        # Use "char_a" as source_key for all manually created characters
        src_key = "char_a"

        # Delete old pose images (for regeneration)
        for old in lib_dir.glob(f"pose_{src_key}_*.png"):
            old.unlink()
        old_atlas = lib_dir / f"pose_atlas_{src_key}.png"
        if old_atlas.exists():
            old_atlas.unlink()

        if structure == "quest":
            atlas_prompt = _build_image_prompt("quest", description)
            gen_params = {
                "prompt": atlas_prompt,
                "provider": "seedream",
                "image_size": "4992x3328",
                "output_format": "png",
                "is_segmentation": True,
            }
        else:
            # Original: save as char_scene.png
            atlas_prompt = _build_image_prompt("original", description)
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


# ===========================================================================
# AI Character Generation (LLM 一键生成通用角色：描述先行 → 用户审核后再生图)
# ===========================================================================

_ai_gen_status: dict = {"status": "idle", "count": 0, "error": "", "created": []}


def _generate_ai_characters(structure: str, count: int = 5, gender: str = "",
                            age: str = "", skin: str = "", extra: str = "") -> None:
    """后台线程：LLM 生成通用角色候选并直接入库（仅描述无图，生图由用户在卡片上触发）。

    gender/age/skin/extra 为用户勾选的筛选维度（空 = 随机），注入 prompt 硬约束。
    """
    import urllib.request

    _ai_gen_status.update({"status": "generating", "count": 0, "error": "", "created": []})
    try:
        p_type, base_url, api_key, model = resolve_provider(load_config())
        if not api_key:
            raise RuntimeError(f"未配置 {p_type} 的 API Key，请在参数配置页面填写")
        if not model:
            raise RuntimeError("未指定模型（该 Provider 未配置模型列表）")

        # 防重复：把现有角色清单交给 LLM 规避
        existing = list_library_chars()
        avoid_lines = "\n".join(
            f"- {c.get('name', '')}: {c.get('description', '')[:80]}"
            for c in existing[:30]) or "(none)"

        # 用户勾选的维度 → 硬约束（空维度 = 不约束，走随机多样化）
        constraint_lines = []
        if gender in ("female", "male"):
            constraint_lines.append(
                f'- ALL {count} characters MUST be {gender} — ignore the "mix genders" requirement.')
        if age:
            constraint_lines.append(
                f'- ALL characters MUST be {age} — ignore the "mix ages" requirement for the age band (still vary exact age).')
        if skin:
            constraint_lines.append(f'- ALL characters MUST have {skin}.')
        if extra:
            constraint_lines.append(
                f'- Additional user requirements (follow strictly, keep them out of "scenarios"): {extra}')
        constraints_block = (
            "\n\nUSER CONSTRAINTS (highest priority — override any requirement above):\n"
            + "\n".join(constraint_lines)) if constraint_lines else ""

        prompt = f"""You are a character designer for an English listening-practice video channel (audience: overseas Chinese ESL learners). The videos are everyday conversations set in common daily scenarios (coffee shop, pharmacy, airport, bank, restaurant, hotel, school, office, shopping mall, clinic, gas station, post office, gym, library...).

Design exactly {count} GENERIC, REUSABLE human characters that could plausibly appear in many of those daily scenarios.

Requirements:
- Mix genders (about half female, half male); mix ages (young adult / 30s / middle-aged / senior) and occupations
- "description": APPEARANCE ONLY in English, 25-45 words (age, hair, face, clothing, accessories, one distinctive trait). NEVER mention any location or scene — the character image is generated on a plain white background. Must be detailed enough to be the sole input for AI image generation.
- "role": the character's occupation/identity in 1-3 English words (e.g. "coffee shop barista")
- "scenarios": 3-5 short CHINESE scenario labels (2-6 Chinese characters each, e.g. "咖啡店点单", "药店买药") — the common scenarios this character fits
- Do NOT duplicate these existing characters:
{avoid_lines}{constraints_block}

Output valid JSON only (no markdown, no explanations):
{{"characters": [{{"name": "...", "gender": "female", "role": "...", "description": "...", "scenarios": ["..."]}}]}}"""

        from llm_client import _extract_json  # pipeline/ 已在 sys.path

        body = {
            "model": model,
            "messages": [
                {"role": "system",
                 "content": "You are an expert character designer. Output valid JSON only — no markdown, no explanations."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.9,
            "max_tokens": 4096,
        }
        if p_type != "openai":
            body["reasoning_effort"] = "low"

        print(f"  [CharAI] Requesting {count} generic characters from {model} ({p_type})...")
        req = urllib.request.Request(
            f"{base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            method="POST",
        )
        req.add_header("Authorization", f"Bearer {api_key}")
        req.add_header("Content-Type", "application/json")
        req.add_header("User-Agent", "CodelyLLM/1.0")
        with urllib.request.urlopen(req, timeout=180) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        content = result["choices"][0]["message"]["content"]
        chars = _extract_json(content).get("characters") or []
        print(f"  [CharAI] LLM returned {len(chars)} candidates")

        gender_map = {"female": "female", "male": "male",
                      "woman": "female", "man": "male", "f": "female", "m": "male"}
        saved = []
        base_ms = int(time.time() * 1000)
        for i, c in enumerate(chars):
            if not isinstance(c, dict):
                continue
            name = str(c.get("name", "")).strip()
            gender = gender_map.get(str(c.get("gender", "")).strip().lower(), "")
            desc = str(c.get("description", "")).strip()
            role = str(c.get("role", "")).strip()
            scenarios = [str(s).strip() for s in (c.get("scenarios") or [])
                         if str(s).strip()]
            if not name or not gender or not desc:
                continue
            lib_id = f"char_{base_ms + i}_{gender}"  # 毫秒+序号，批量入库不撞 ID
            lib_dir = LIBRARY_DIR / lib_id
            lib_dir.mkdir(parents=True, exist_ok=True)
            meta = {
                "id": lib_id,
                "name": f"{name} ({role})" if role else name,
                "description": desc,
                "gender": gender,
                "structure": structure,
                "role": role,
                "scenarios": scenarios,
                "origin": "ai",
                "qwen_speaker": "",
                "moss_voice": "",
                "kokoro_voice": "",
                "source_run": "",
                "source_key": "char_a",
                "created": time.time(),
            }
            (lib_dir / "meta.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            saved.append(lib_id)
        print(f"  [CharAI] Saved {len(saved)} characters to library")

        if not saved:
            raise RuntimeError("LLM 返回内容中没有有效角色（字段缺失或解析失败）")
        _ai_gen_status.update({"status": "done", "count": len(saved),
                               "created": saved, "error": ""})
    except Exception as e:  # noqa: BLE001 — 错误信息原样落状态供前端展示
        print(f"  [CharAI] ERROR: {e}")
        _ai_gen_status.update({"status": "error", "count": 0, "created": [],
                               "error": str(e)[:300]})


@router.post("/api/character_library/ai_generate")
async def api_library_ai_generate(request: Request):
    """LLM 一键生成通用角色（静态路径，须注册在 {lib_id} 通配路由之前防遮蔽）。"""
    if _ai_gen_status.get("status") == "generating":
        return JSONResponse({"ok": False, "error": "已有生成任务进行中，请稍候"}, status_code=409)
    try:
        data = await request.json()
    except Exception:
        data = {}
    structure = data.get("structure", "quest")
    if structure not in ("quest", "original"):
        structure = "quest"

    def _opt(val: object, limit: int) -> str:
        return str(val or "").strip().replace("\n", " ")[:limit]

    gender = data.get("gender") if data.get("gender") in ("female", "male") else ""
    age = _opt(data.get("age"), 60)
    skin = _opt(data.get("skin"), 60)
    extra = _opt(data.get("extra"), 300)

    import threading
    threading.Thread(target=_generate_ai_characters,
                     args=(structure, 5, gender, age, skin, extra),
                     daemon=True).start()
    return {"ok": True, "message": "LLM 生成中..."}


@router.get("/api/character_library/ai_generate_status")
async def api_library_ai_generate_status():
    """Poll AI character generation status."""
    return _ai_gen_status


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
    for key in ("name", "description", "gender", "structure", "qwen_speaker",
                "moss_voice", "kokoro_voice", "is_host"):
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


# ===========================================================================
# Sprite clip generation（序列帧素材：MCP 自动生成 / 外部工具提示词+回传）
# ===========================================================================

_CLIP_ACTIONS_BASE = ("talking_01", "talking_02", "talking_03", "idle_01", "idle_02")
_CLIP_ACTIONS_HOST = _CLIP_ACTIONS_BASE + ("wave",)

_clip_gen_status: dict[str, dict] = {}  # lib_id → {status, done, total, current_action, error}

# 有参考图时附加到视频 prompt 的一致性强约束（外部工具与 MCP 自动路线共用；
# 仅素材库路线附加，管线 sprite_seq._video_prompt 保持零改动）
_REF_CONSISTENCY = (
    "the character must be exactly the same person as the reference image, "
    "identical face, identical hairstyle, identical outfit and colors, "
    "strictly follow the reference image appearance, no design changes"
)


def _with_ref_consistency(prompt: str, has_ref: bool) -> str:
    """有参考图时在视频 prompt 末尾附加人物一致性约束。"""
    return f"{prompt}, {_REF_CONSISTENCY}" if has_ref and prompt else prompt


def _lib_ref_file(lib_dir: Path) -> str:
    """该角色当前参考图文件名（姿势图 → 场景图 → 空），供提示词与打开目录用。"""
    for name in ("pose_char_a_0.png", "char_scene.png"):
        if (lib_dir / name).exists():
            return name
    return ""


def _lib_is_host(meta: dict) -> bool:
    """该库角色是否按主持人处理（决定是否生成 wave 动作段）。"""
    return bool(meta.get("is_host")) or meta.get("source_key") == "host"


def _lib_clip_paths(lib_dir: Path, action: str) -> list[str]:
    """库内命名目标帧路径（clip_{action}_{j:02d}.png，talking 48 帧/其余 16 帧）。"""
    n = 48 if action.startswith("talking") else 16
    return [str(lib_dir / f"clip_{action}_{j:02d}.png") for j in range(n)]


def _produce_clips_from_video(video_path: str, lib_dir: Path, action: str,
                              label: str = "") -> bool:
    """本地处理：视频 → ffmpeg 抽帧(8fps) → 取样(48/16) → 抠图统一几何 → 存库内帧。

    复用 pipeline/sprite_seq.py 纯函数（与管线内序列帧产出规格逐字节一致），
    MCP 自动生成与外部视频回传两条路线共用。
    """
    import os
    import shutil
    import tempfile

    from PIL import Image
    from sprite_seq import (_extract_video_frames, _sample_frames,
                            _unify_clip_frames, frames_for_action)

    n_frames = frames_for_action(action)
    final_paths = _lib_clip_paths(lib_dir, action)
    # 不做「帧已齐早退」——重新上传必须替换旧素材；
    # MCP 自动路线的续传跳过由调用方 _generate_char_clips 自行判断。
    frames_dir = Path(tempfile.gettempdir()) / f"libclip_{lib_dir.name}_{action}"
    shutil.rmtree(str(frames_dir), ignore_errors=True)
    try:
        all_frames = _extract_video_frames(video_path, frames_dir, fps=8)
        if not all_frames:
            return False
        if len(all_frames) > n_frames:
            raw = [str(p) for p in _sample_frames(all_frames, n_frames)]
        else:
            raw = [str(p) for p in all_frames]
        if len(raw) < 4:
            print(f"    [LibClips] {label or action} too few frames ({len(raw)})")
            return False
        raw_imgs = []
        for p in raw:
            try:
                raw_imgs.append(Image.open(p))
            except Exception as e:
                print(f"    [LibClips] open frame error: {e}")
        unified = _unify_clip_frames(raw_imgs, label=label or action)
        if len(unified) < 4:
            return False
        if len(unified) != n_frames:
            idxs = [round(i * (len(unified) - 1) / (n_frames - 1))
                    for i in range(n_frames)]
            unified = [unified[k] for k in idxs]
        for j, frame in enumerate(unified):
            frame.save(final_paths[j], compress_level=2)
        print(f"    [LibClips] {label or action}: {n_frames} frames saved")
        return True
    finally:
        shutil.rmtree(str(frames_dir), ignore_errors=True)


def _refresh_clip_manifest(lib_dir: Path, description: str) -> int:
    """扫描库内已齐帧的 clip 文件重建 sprite_clips.json 并更新 meta.sprite_clips。

    返回登记的动作数；无任何完整动作时不写 manifest（与运行导入语义一致）。
    """
    import os

    actions: dict[str, list[str]] = {}
    for action in _CLIP_ACTIONS_HOST:  # host 是超集（含 wave）
        paths = _lib_clip_paths(lib_dir, action)
        if all(os.path.exists(p) for p in paths):
            actions[action] = [Path(p).name for p in paths]
    if actions:
        (lib_dir / "sprite_clips.json").write_text(
            json.dumps({"version": 1, "fps": 12, "source": "video_frames",
                        "desc_snapshot": description, "actions": actions},
                       ensure_ascii=False, indent=2),
            encoding="utf-8")
    meta_path = lib_dir / "meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return len(actions)
        meta["sprite_clips"] = len(actions)
        write_library_meta(lib_dir.name, meta)
    return len(actions)


def _generate_char_clips(lib_id: str, description: str, is_host: bool) -> None:
    """后台线程：MCP Seedance2 生成各动作白底视频 → 本地抽帧抠图入库。

    页面级独立 MCP 会话（与 pipeline 全局会话隔离）；单动作失败记日志继续；
    文件级续传（帧已齐的动作跳过）；无参考图时 talking_01 先行、其后动作
    用它的帧做一致性参考（与管线 sprite_seq._gen_char 语义一致）。
    """
    import os

    from image_gen import reupload_for_cdn
    from sprite_seq import _pick_ref_frame, _video_prompt
    from style_manager import DEFAULT_STYLE_PROMPT

    lib_dir = LIBRARY_DIR / lib_id
    actions = list(_CLIP_ACTIONS_HOST if is_host else _CLIP_ACTIONS_BASE)
    total = len(actions)
    if not lib_dir.exists():
        _clip_gen_status[lib_id] = {"status": "error", "error": "角色目录不存在",
                                    "done": 0, "total": total, "current_action": ""}
        return
    _clip_gen_status[lib_id] = {"status": "generating", "error": "", "done": 0,
                                "total": total, "current_action": "",
                                "started_at": time.time()}
    try:
        resolved = resolve_page_tokens("characters")
        if not resolved["tokens"]:
            _clip_gen_status[lib_id] = {
                "status": "error",
                "error": "未配置 MCP Token（本页专属 / 模式配置 / 本地检测均为空）",
                "done": 0, "total": total, "current_action": ""}
            return
        print(f"  [LibClips] MCP token: {MCP_SOURCE_LABELS[resolved['source']]} "
              f"×{len(resolved['tokens'])} ({mask_token(resolved['tokens'][0])})")
        mcp = PageMcpSession(resolved["tokens"]).initialize()

        # 参考图：姿势图 → 场景图 → 无（无图允许直接生成）
        ref_url = ""
        for name in ("pose_char_a_0.png", "char_scene.png"):
            p = lib_dir / name
            if p.exists():
                ref_url = reupload_for_cdn(str(p), name, call_tool_fn=mcp.call_tool)
                if ref_url:
                    print(f"  [LibClips] {lib_id} ref: {name}")
                else:
                    print(f"  [LibClips] WARNING: 参考图上传失败，回退 text_to_video")
                break

        ok, failed = 0, []
        for i, action in enumerate(actions):
            _clip_gen_status[lib_id].update({"current_action": action, "done": i})
            final_paths = _lib_clip_paths(lib_dir, action)
            if all(os.path.exists(p) for p in final_paths):
                print(f"  [LibClips] {lib_id}/{action} 已存在，跳过")
                ok += 1
                continue
            params = {"prompt": _with_ref_consistency(
                          _video_prompt(action, "" if ref_url else description,
                                        DEFAULT_STYLE_PROMPT),
                          bool(ref_url)),
                      "duration": 6, "ratio": "16:9", "resolution": "720p",
                      "generate_audio": False}
            if ref_url:
                params.update({"mode": "reference_image", "image_urls": ref_url})
            else:
                params.update({"mode": "text_to_video"})
            result = mcp.call_tool("generate_video", params)
            task_id = mcp.parse_task_id(result)
            video_path = ""
            if task_id:
                data = mcp.poll_task(task_id, interval=40, max_wait=900)
                url = data.get("url", "")
                tmp_video = lib_dir / f"_tmp_{action}.mp4"
                if (url and mcp.download_file(url, str(tmp_video))
                        and tmp_video.stat().st_size >= 500000):
                    video_path = str(tmp_video)
            if not video_path:
                print(f"  [LibClips] WARNING: {lib_id}/{action} 视频生成失败")
                failed.append(action)
                continue
            try:
                produced = _produce_clips_from_video(video_path, lib_dir, action,
                                                     label=f"{lib_id}/{action}")
            finally:
                try:
                    os.remove(video_path)
                except OSError:
                    pass
            if not produced:
                failed.append(action)
                continue
            ok += 1
            # 无参考图时用 talking_01 的帧做其后动作的一致性参考
            if not ref_url and action == _CLIP_ACTIONS_BASE[0]:
                ref = _pick_ref_frame(final_paths)
                if ref:
                    ref_url = reupload_for_cdn(ref, Path(ref).name,
                                               call_tool_fn=mcp.call_tool) or ""

        _refresh_clip_manifest(lib_dir, description)
        _clip_gen_status[lib_id] = {
            "status": "done" if ok else "error",
            "error": "" if ok else "全部动作生成失败" + (f"：{', '.join(failed)}" if failed else ""),
            "done": ok, "total": total, "current_action": ""}
        print(f"  [LibClips] Done {lib_id}: {ok}/{total} clips")
    except RuntimeError as e:
        msg = ("MCP Token 积分全部耗尽" if "ALL_MCP_TOKENS_EXHAUSTED" in str(e)
               else str(e)[:200])
        _clip_gen_status[lib_id] = {"status": "error", "error": msg,
                                    "done": 0, "total": total, "current_action": ""}
    except Exception as e:  # noqa: BLE001 — 错误信息原样落状态供前端展示
        print(f"  [LibClips] ERROR {lib_id}: {e}")
        _clip_gen_status[lib_id] = {"status": "error", "error": str(e)[:200],
                                    "done": 0, "total": total, "current_action": ""}


@router.post("/api/character_library/{lib_id}/generate_clips")
async def api_library_generate_clips(lib_id: str):
    """Start MCP sprite clip generation in background thread (页面级独立会话)."""
    import threading
    lib_dir = LIBRARY_DIR / lib_id
    if not lib_dir.exists():
        return JSONResponse({"ok": False, "error": "未找到"}, status_code=404)
    try:
        meta = json.loads((lib_dir / "meta.json").read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return JSONResponse({"ok": False, "error": "meta.json 读取失败"}, status_code=500)
    description = meta.get("description", "")
    if not description:
        return JSONResponse({"ok": False, "error": "角色描述为空，无法生成序列帧"}, status_code=400)
    if _clip_gen_status.get(lib_id, {}).get("status") == "generating":
        return JSONResponse({"ok": False, "error": "该角色已有序列帧生成任务进行中"}, status_code=409)
    is_host = _lib_is_host(meta)
    threading.Thread(target=_generate_char_clips,
                     args=(lib_id, description, is_host), daemon=True).start()
    return {"ok": True, "message": "序列帧生成中...", "is_host": is_host}


@router.get("/api/character_library/{lib_id}/clip_status")
async def api_library_clip_status(lib_id: str):
    """Poll sprite clip generation status."""
    return _clip_gen_status.get(lib_id, {"status": "idle", "done": 0, "total": 0,
                                         "current_action": "", "error": ""})


@router.get("/api/character_library/{lib_id}/gen_prompts")
async def api_library_gen_prompts(lib_id: str):
    """输出参考图 + 各动作视频的完整提示词（复制到外部工具用，与 MCP 路线逐字一致）。"""
    import os

    from sprite_seq import _video_prompt
    from style_manager import DEFAULT_STYLE_PROMPT

    lib_dir = LIBRARY_DIR / lib_id
    if not lib_dir.exists():
        return JSONResponse({"error": "Not found"}, status_code=404)
    try:
        meta = json.loads((lib_dir / "meta.json").read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return JSONResponse({"error": "meta.json 读取失败"}, status_code=500)
    description = meta.get("description", "")
    structure = meta.get("structure", "quest")
    is_host = _lib_is_host(meta)
    ref_file = _lib_ref_file(lib_dir)
    has_ref = bool(ref_file)
    clips = []
    for action in (_CLIP_ACTIONS_HOST if is_host else _CLIP_ACTIONS_BASE):
        paths = _lib_clip_paths(lib_dir, action)
        n_have = sum(1 for p in paths if os.path.exists(p))
        clips.append({
            "action": action,
            "prompt": (_with_ref_consistency(
                _video_prompt(action, "" if has_ref else description, DEFAULT_STYLE_PROMPT),
                has_ref) if description else ""),
            "exists": description != "" and n_have == len(paths),
            "frames": n_have,
            "sample_file": (Path(paths[0]).name if os.path.exists(paths[0]) else ""),
        })
    return {
        "description": description,
        "structure": structure,
        "is_host": is_host,
        "has_ref": has_ref,
        "clips": clips,
        "image": {
            "prompt": _build_image_prompt(structure, description) if description else "",
            "suggested": ("4992x3328 横版图集（4×2 八姿势）" if structure == "quest"
                          else "16:9 横版单人半身"),
            "sample_file": ref_file,
        },
        "video_suggested": "16:9、720p 及以上、6 秒以上、纯白背景、无文字水印、人物始终完整在画面中央",
    }


@router.post("/api/character_library/{lib_id}/import_pose_image")
async def api_library_import_pose_image(lib_id: str, image: UploadFile = File(...)):
    """上传外部生成的参考图回库：quest 图集自动切分 8 姿势图 / original 单图直存。"""
    import shutil

    lib_dir = LIBRARY_DIR / lib_id
    if not lib_dir.exists():
        return JSONResponse({"ok": False, "error": "未找到"}, status_code=404)
    try:
        meta = json.loads((lib_dir / "meta.json").read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return JSONResponse({"ok": False, "error": "meta.json 读取失败"}, status_code=500)
    filename = (image.filename or "").lower()
    if not filename.endswith((".png", ".jpg", ".jpeg", ".webp")):
        return JSONResponse({"ok": False, "error": "仅支持 png/jpg/webp 图片"}, status_code=400)
    content = await image.read()
    if len(content) < 10240:
        return JSONResponse({"ok": False, "error": "文件过小，可能不是有效图片"}, status_code=400)

    structure = meta.get("structure", "quest")
    tmp_path = lib_dir / "_tmp_upload.png"
    tmp_path.write_bytes(content)
    try:
        if structure == "quest":
            from atlas_split import split_atlas
            atlas_path = lib_dir / "pose_atlas_char_a.png"
            tmp_path.replace(atlas_path)
            pose_files = [str(lib_dir / f"pose_char_a_{i}.png") for i in range(8)]
            split_atlas(str(atlas_path), 4, 2, pose_files, log_prefix="[LibImport]")
            thumb_src = lib_dir / "pose_char_a_0.png"
        else:
            scene_path = lib_dir / "char_scene.png"
            tmp_path.replace(scene_path)
            thumb_src = scene_path
        if thumb_src.exists():
            shutil.copy2(str(thumb_src), str(lib_dir / "thumb.png"))
        print(f"  [LibImport] {lib_id}: pose image imported ({structure})")
        return {"ok": True, "structure": structure,
                "thumb_url": f"/api/character_library/{lib_id}/image"}
    except Exception as e:  # noqa: BLE001
        print(f"  [LibImport] ERROR {lib_id}: {e}")
        return JSONResponse({"ok": False, "error": f"处理失败: {str(e)[:150]}"}, status_code=500)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


@router.post("/api/character_library/{lib_id}/import_clip")
async def api_library_import_clip(lib_id: str, action: str = Form(""),
                                  video: UploadFile = File(...)):
    """上传外部生成的动作视频回库：本地抽帧→取样→抠图统一→入库（零积分）。"""
    lib_dir = LIBRARY_DIR / lib_id
    if not lib_dir.exists():
        return JSONResponse({"ok": False, "error": "未找到"}, status_code=404)
    try:
        meta = json.loads((lib_dir / "meta.json").read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return JSONResponse({"ok": False, "error": "meta.json 读取失败"}, status_code=500)
    allowed = _CLIP_ACTIONS_HOST if _lib_is_host(meta) else _CLIP_ACTIONS_BASE
    if action not in allowed:
        return JSONResponse({"ok": False, "error": f"未知动作: {action}"}, status_code=400)
    filename = (video.filename or "").lower()
    if not filename.endswith((".mp4", ".mov", ".webm", ".mkv")):
        return JSONResponse({"ok": False, "error": "仅支持 mp4/mov/webm/mkv 视频"}, status_code=400)
    content = await video.read()
    if len(content) < 102400:
        return JSONResponse({"ok": False, "error": "视频文件过小"}, status_code=400)

    tmp_video = lib_dir / f"_tmp_{action}_upload.mp4"
    tmp_video.write_bytes(content)
    try:
        ok = _produce_clips_from_video(str(tmp_video), lib_dir, action,
                                       label=f"{lib_id}/{action}")
    finally:
        try:
            tmp_video.unlink()
        except OSError:
            pass
    if not ok:
        return JSONResponse({"ok": False, "error": "视频处理失败（抽帧失败或有效内容过少）"}, status_code=500)
    n_actions = _refresh_clip_manifest(lib_dir, meta.get("description", ""))
    print(f"  [LibImport] {lib_id}/{action}: clip imported (manifest {n_actions} actions)")
    return {"ok": True, "action": action,
            "frames": 48 if action.startswith("talking") else 16,
            "manifest_actions": n_actions}


@router.post("/api/character_library/{lib_id}/open_folder")
async def api_library_open_folder(lib_id: str, file: str = ""):
    """在系统文件管理器中打开角色素材目录；file 非空时高亮选中该文件。

    仿 runs.py open_folder：仅接受目录内普通文件名（禁路径分隔符/点段）。
    """
    import os
    import subprocess
    import sys
    import threading

    lib_dir = (LIBRARY_DIR / lib_id).resolve()
    if (not str(lib_dir).startswith(str(LIBRARY_DIR.resolve()))
            or not lib_dir.is_dir()):
        return JSONResponse({"ok": False, "error": "Not found"}, status_code=404)
    if file and (os.sep in file or "/" in file or "\\" in file or ".." in file
                 or Path(file).name != file):
        return JSONResponse({"ok": False, "error": "Invalid file"}, status_code=400)
    target = lib_dir
    if file:
        candidate = lib_dir / file
        if candidate.is_file():
            target = candidate
    try:
        if sys.platform == "win32":
            from .runs import _list_explorer_windows, _raise_folder_window
            before = _list_explorer_windows()
            if target != lib_dir:
                subprocess.Popen(["explorer", "/select,", str(target)])
            else:
                os.startfile(str(lib_dir))  # noqa: S606
            threading.Thread(
                target=_raise_folder_window,
                args=(lib_dir.name, before), daemon=True).start()
        else:
            opener = "open" if sys.platform == "darwin" else "xdg-open"
            subprocess.Popen([opener, str(target)])
        return {"ok": True, "path": str(target)}
    except OSError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


