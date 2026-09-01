"""Runs API — 运行列表/素材文件/回收站/重渲/元数据刷新（Runs + Script editor + Gallery 三区合并）."""
import json
import os
import re
import sys
import threading
import time
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse

from ..config_manager import (
    LEGACY_RECYCLE_DIRNAME, MODES, RECYCLE_DIRNAME,
    find_run_dir, iter_run_dirs, load_config,
)
from ..paths import TRASH_META_FILENAME
from ..pipeline_service import get_service
from ..thumbnail_regen_service import get_thumb_regen_service
import subtitle_style_manager as subtitle_style_lib

router = APIRouter()


@router.get("/api/runs")
async def api_list_runs():
    config = load_config()
    output_dir = Path(config.get("output_dir", "./output"))
    runs = []
    for d in iter_run_dirs(output_dir):
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


@router.get("/api/runs/{name}/script")
async def api_get_script(name: str, mode: str = ""):
    config = load_config()
    output_dir = Path(config.get("output_dir", "./output"))
    run_dir = find_run_dir(output_dir, name, mode)
    script_path = run_dir / "script.json" if run_dir else None
    if not script_path or not script_path.exists():
        return JSONResponse({"error": "Not found"}, status_code=404)
    return json.loads(script_path.read_text(encoding="utf-8"))


# 缩略图文件名白名单（thumbnail.jpg + thumbnail_N.jpg），防路径穿越
_THUMB_NAME_RE = re.compile(r"thumbnail(?:_\d+)?\.jpg")


def _resolve_main_thumbnail(run_dir: Path) -> Path:
    """主缩略图：thumb_main.txt 指定 > thumbnail.jpg > 编号最小的 thumbnail_N.jpg。"""
    meta = run_dir / "thumb_main.txt"
    if meta.exists():
        try:
            name = meta.read_text(encoding="utf-8").strip()
            if _THUMB_NAME_RE.fullmatch(name) and (run_dir / name).exists():
                return run_dir / name
        except OSError:
            pass
    # thumbnail.jpg 被删时，自动回落到第一张 thumbnail_N.jpg
    if (run_dir / "thumbnail.jpg").exists():
        return run_dir / "thumbnail.jpg"
    for f in sorted(run_dir.glob("thumbnail_*.jpg"),
                    key=lambda p: int(p.stem.split("_")[1]) if p.stem.split("_")[1].isdigit() else 0):
        if _THUMB_NAME_RE.fullmatch(f.name):
            return f
    return run_dir / "thumbnail.jpg"  # 不存在时仍返回此路径（调用方 .exists() 判 False）


@router.get("/api/runs/{name}/thumbnail")
async def api_get_thumbnail(name: str, mode: str = ""):
    config = load_config()
    output_dir = Path(config.get("output_dir", "./output"))
    run_dir = find_run_dir(output_dir, name, mode)
    thumb = _resolve_main_thumbnail(run_dir) if run_dir else None
    if not thumb or not thumb.exists():
        return JSONResponse({"error": "Not found"}, status_code=404)
    return FileResponse(str(thumb), media_type="image/jpeg")


@router.get("/api/runs/{name}/thumbnails")
async def api_list_thumbnails(name: str, mode: str = ""):
    """列出运行的全部缩略图（thumbnail.jpg + thumbnail_N.jpg，旧图全保留）。"""
    config = load_config()
    output_dir = Path(config.get("output_dir", "./output"))
    run_dir = find_run_dir(output_dir, name, mode)
    if not run_dir:
        return JSONResponse({"error": "Not found"}, status_code=404)
    files = [f.name for f in run_dir.glob("thumbnail*.jpg")
             if _THUMB_NAME_RE.fullmatch(f.name)]

    def _thumb_sort_key(fname: str):
        # thumbnail.jpg 主图排最前，其余按编号升序
        if fname == "thumbnail.jpg":
            return (0, 0)
        num = fname[len("thumbnail_"):-len(".jpg")]
        return (1, int(num) if num.isdigit() else 0)

    files.sort(key=_thumb_sort_key)
    main_name = _resolve_main_thumbnail(run_dir).name
    return {"thumbnails": [
        {"name": f, "url": f"/api/runs/{name}/thumbnail/{f}", "is_main": f == main_name}
        for f in files
    ]}


@router.get("/api/runs/{name}/thumbnail/{filename}")
async def api_get_thumbnail_file(name: str, filename: str, mode: str = ""):
    """按文件名提供单张缩略图（白名单校验防路径穿越）。"""
    if not _THUMB_NAME_RE.fullmatch(filename):
        return JSONResponse({"error": "Invalid filename"}, status_code=400)
    config = load_config()
    output_dir = Path(config.get("output_dir", "./output"))
    run_dir = find_run_dir(output_dir, name, mode)
    thumb = run_dir / filename if run_dir else None
    if not thumb or not thumb.exists():
        return JSONResponse({"error": "Not found"}, status_code=404)
    return FileResponse(str(thumb), media_type="image/jpeg")


@router.post("/api/runs/{name}/thumbnail_regenerate")
async def api_regen_thumbnail(name: str, mode: str = ""):
    """为已有运行再生成一张缩略图（thumbnail_N.jpg 递增，旧图全保留；走 AI 生图）。

    独立子进程执行，不与主 pipeline / 模式测试互斥，可并行运行。
    """
    service = get_thumb_regen_service()
    ok, msg = service.start(name, mode=mode)
    if not ok:
        return JSONResponse({"ok": False, "error": msg}, status_code=409)
    return {"ok": True, "message": msg}


@router.get("/api/thumbnail_regen/status")
async def api_thumb_regen_status(since: int = 0):
    """缩略图重生成子进程任务状态 + 增量日志（形状对齐 /api/run/logs/since）。"""
    return get_thumb_regen_service().get_status(since=since)


@router.post("/api/runs/{name}/thumbnail_set_main")
async def api_set_main_thumbnail(name: str, request: Request, mode: str = ""):
    """设置「主图」：卡片封面显示所选缩略图（thumb_main.txt 记录，不改动任何文件）。"""
    data = await request.json()
    filename = str(data.get("filename", "")).strip()
    if not _THUMB_NAME_RE.fullmatch(filename):
        return JSONResponse({"ok": False, "error": f"非法缩略图文件名: {filename}"}, status_code=400)
    config = load_config()
    output_dir = Path(config.get("output_dir", "./output"))
    run_dir = find_run_dir(output_dir, name, mode)
    if not run_dir or not (run_dir / filename).exists():
        return JSONResponse({"ok": False, "error": "Not found"}, status_code=404)
    (run_dir / "thumb_main.txt").write_text(filename, encoding="utf-8")
    return {"ok": True, "main": filename}


@router.post("/api/runs/{name}/thumbnail_delete")
async def api_delete_thumbnail(name: str, request: Request, mode: str = ""):
    """删除一张缩略图（白名单校验防路径穿越）。

    删除的是主图时清除 thumb_main.txt 记录（回落 thumbnail.jpg 默认）；
    该文件正被重生成子进程写入时拒绝删除。
    """
    data = await request.json()
    filename = str(data.get("filename", "")).strip()
    if not _THUMB_NAME_RE.fullmatch(filename):
        return JSONResponse({"ok": False, "error": f"非法缩略图文件名: {filename}"}, status_code=400)

    # 正在生成中的同名文件不可删（子进程可能正写到一半）
    st = get_thumb_regen_service().get_status(0)["status"]
    if (st.get("status") == "running" and st.get("run_name") == name
            and st.get("mode") == (mode or "") and st.get("out_name") == filename):
        return JSONResponse({"ok": False, "error": "该缩略图正在生成中，请等待完成后再删除"},
                            status_code=409)

    config = load_config()
    output_dir = Path(config.get("output_dir", "./output"))
    run_dir = find_run_dir(output_dir, name, mode)
    if not run_dir or not (run_dir / filename).exists():
        return JSONResponse({"ok": False, "error": "Not found"}, status_code=404)
    try:
        (run_dir / filename).unlink()
    except OSError as e:
        return JSONResponse({"ok": False, "error": f"删除失败: {e}"}, status_code=500)

    # 删除的是主图 → 清除主图记录（_resolve_main_thumbnail 回落 thumbnail.jpg）
    meta = run_dir / "thumb_main.txt"
    was_main = False
    if meta.exists():
        try:
            was_main = meta.read_text(encoding="utf-8").strip() == filename
        except OSError:
            was_main = False
        if was_main:
            meta.unlink(missing_ok=True)
    return {"ok": True, "deleted": filename, "was_main": was_main}


@router.get("/api/runs/{name}/video/{video_name}")
async def api_get_video(name: str, video_name: str, mode: str = ""):
    config = load_config()
    output_dir = Path(config.get("output_dir", "./output"))
    run_dir = find_run_dir(output_dir, name, mode)
    if not run_dir:
        return JSONResponse({"error": "Not found"}, status_code=404)
    # Final videos are in work_dir root; clips are in clips/ subdir
    video_path = run_dir / video_name
    if not video_path.exists():
        video_path = run_dir / "clips" / video_name
    if not video_path.exists():
        video_path = run_dir / "videos" / video_name
    if not video_path.exists():
        return JSONResponse({"error": "Not found"}, status_code=404)
    return FileResponse(str(video_path), media_type="video/mp4")


@router.get("/api/runs/{name}/images/{image_name}")
async def api_get_image(name: str, image_name: str, mode: str = ""):
    config = load_config()
    output_dir = Path(config.get("output_dir", "./output"))
    run_dir = find_run_dir(output_dir, name, mode)
    img_path = run_dir / "images" / image_name if run_dir else None
    if not img_path or not img_path.exists():
        return JSONResponse({"error": "Not found"}, status_code=404)
    return FileResponse(str(img_path), media_type="image/png")


@router.get("/api/runs/{name}/images_list")
async def api_list_images(name: str, mode: str = ""):
    config = load_config()
    output_dir = Path(config.get("output_dir", "./output"))
    run_dir = find_run_dir(output_dir, name, mode)
    images_dir = run_dir / "images" if run_dir else None
    if not images_dir or not images_dir.exists():
        return {"images": []}
    images = sorted([f.name for f in images_dir.glob("*.png")])
    return {"images": images}


# 回收站：删除运行先移入 output/_recycle_bin，回收站内再次删除才真正删文件
# （RECYCLE_DIRNAME / LEGACY_RECYCLE_DIRNAME 定义见 config_manager.py）


def _move_run_to_recycle_bin(output_dir: Path, run_dir: Path) -> str:
    """把运行目录移入回收站（同盘 rename），重名自动追加 _N 后缀。返回回收站内的目录名。"""
    import shutil
    recycle_root = output_dir / RECYCLE_DIRNAME
    recycle_root.mkdir(exist_ok=True)
    dst = recycle_root / run_dir.name
    suffix = 1
    while dst.exists():
        dst = recycle_root / f"{run_dir.name}_{suffix}"
        suffix += 1
    shutil.move(str(run_dir), str(dst))
    meta = {
        "original_name": run_dir.name,
        "original_parent": run_dir.parent.name if run_dir.parent.name in MODES else "",
        "deleted_at": time.time(),
    }
    (dst / TRASH_META_FILENAME).write_text(
        json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    return dst.name


def _find_trash_item(output_dir: Path, name: str) -> Path | None:
    """在回收站（新旧两种目录）中按名定位已删除运行。"""
    for recycle_root in (output_dir / RECYCLE_DIRNAME, output_dir / LEGACY_RECYCLE_DIRNAME):
        src = recycle_root / name
        if src.is_dir() and str(src.resolve()).startswith(str(recycle_root.resolve())):
            return src
    return None


@router.delete("/api/runs/{name}")
async def api_delete_run(name: str, mode: str = ""):
    """删除运行 → 移入回收站（防误删，可在页面恢复）。"""
    config = load_config()
    output_dir = Path(config.get("output_dir", "./output"))
    run_dir = find_run_dir(output_dir, name, mode)
    if not run_dir or not run_dir.is_dir():
        return JSONResponse({"ok": False, "error": "Not found"}, status_code=404)
    # Safety: ensure it's within output_dir
    if not str(run_dir.resolve()).startswith(str(output_dir.resolve())):
        return JSONResponse({"ok": False, "error": "Cannot delete"}, status_code=400)
    try:
        _move_run_to_recycle_bin(output_dir, run_dir)
    except OSError as e:
        return JSONResponse({"ok": False, "error": f"移入回收站失败: {e}"}, status_code=500)
    return {"ok": True}


@router.post("/api/trash/{name}/restore")
async def api_trash_restore(name: str):
    """从回收站恢复运行（新条目回模式文件夹；旧条目按脚本结构归类）。"""
    import shutil
    config = load_config()
    output_dir = Path(config.get("output_dir", "./output"))
    src = _find_trash_item(output_dir, name)
    if not src:
        return JSONResponse({"ok": False, "error": "Not found"}, status_code=404)
    original = name
    parent = ""
    meta_path = src / TRASH_META_FILENAME
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            saved = meta.get("original_name") or ""
            if saved and "/" not in saved and "\\" not in saved:
                original = saved
            parent = meta.get("original_parent") or ""
        except (json.JSONDecodeError, OSError):
            pass
    if parent not in MODES:
        # 旧回收条目无 original_parent：按脚本 structure 归入模式文件夹
        parent = ""
        script_path = src / "script.json"
        if script_path.exists():
            try:
                s = json.loads(script_path.read_text(encoding="utf-8"))
                if s.get("structure") in MODES:
                    parent = s["structure"]
            except (json.JSONDecodeError, OSError):
                pass
    # 同名运行已存在时追加 _N 后缀，绝不覆盖现有数据
    dst = (output_dir / parent / original) if parent else (output_dir / original)
    suffix = 1
    while dst.exists():
        dst = (output_dir / parent / f"{original}_{suffix}" if parent
               else output_dir / f"{original}_{suffix}")
        suffix += 1
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
    except OSError as e:
        return JSONResponse({"ok": False, "error": f"恢复失败: {e}"}, status_code=500)
    (dst / TRASH_META_FILENAME).unlink(missing_ok=True)
    return {"ok": True, "restored_as": dst.name}


@router.delete("/api/trash/{name}")
async def api_trash_purge(name: str):
    """从回收站彻底删除运行（不可恢复）。"""
    import shutil
    config = load_config()
    output_dir = Path(config.get("output_dir", "./output"))
    run_dir = _find_trash_item(output_dir, name)
    if not run_dir:
        return JSONResponse({"ok": False, "error": "Not found"}, status_code=404)
    shutil.rmtree(str(run_dir))
    return {"ok": True}


@router.post("/api/trash/empty")
async def api_trash_empty():
    """清空回收站（彻底删除其中所有内容）。"""
    import shutil
    config = load_config()
    output_dir = Path(config.get("output_dir", "./output"))
    removed = 0
    for recycle_root in (output_dir / RECYCLE_DIRNAME, output_dir / LEGACY_RECYCLE_DIRNAME):
        if not recycle_root.is_dir():
            continue
        for entry in recycle_root.iterdir():
            try:
                if entry.is_dir():
                    shutil.rmtree(str(entry))
                else:
                    entry.unlink()
                removed += 1
            except OSError:
                continue
    return {"ok": True, "removed": removed}


@router.post("/api/runs/{name}/mark_uploaded")
async def api_mark_uploaded(name: str, mode: str = ""):
    """切换运行目录的 uploaded.flag 标记（记录视频已上传到 YouTube）。"""
    config = load_config()
    output_dir = Path(config.get("output_dir", "./output"))
    run_dir = find_run_dir(output_dir, name, mode)
    if not run_dir or not run_dir.is_dir():
        return JSONResponse({"ok": False, "error": "Not found"}, status_code=404)
    flag = run_dir / "uploaded.flag"
    if flag.exists():
        flag.unlink()
        return {"ok": True, "uploaded": False}
    flag.touch()
    return {"ok": True, "uploaded": True}


@router.post("/api/runs/{name}/recompose")
async def api_recompose(name: str, request: Request, mode: str = ""):
    """重选字幕样式重渲视频（复用 final_no_sub 底片，仅本地渲染）。"""
    data = await request.json()
    service = get_service()
    style_id = str(data.get("subtitle_style", "")).strip()
    if style_id and not subtitle_style_lib.get_style(style_id):
        return JSONResponse({"ok": False, "error": f"字幕样式不存在: {style_id}"}, status_code=400)
    try:
        font_size = max(20, min(int(data.get("font_size", 60) or 60), 200))
    except (TypeError, ValueError):
        font_size = 60
    ok, msg = service.recompose(
        name, subtitle_style=style_id, font_size=font_size,
        show_zh=bool(data.get("show_zh", True)),
        regen_4k=bool(data.get("regen_4k", False)), mode=mode)
    if not ok:
        return JSONResponse({"ok": False, "error": msg}, status_code=409)
    return {"ok": True, "message": msg, "status": service.status}


@router.post("/api/runs/{name}/generate_4k")
async def api_generate_4k(name: str, mode: str = ""):
    """为已完成运行生成（或重新生成）4K 版本（复用 Step 6 超分逻辑，本地渲染零积分）。"""
    service = get_service()
    ok, msg = service.generate_4k(name, mode=mode)
    if not ok:
        return JSONResponse({"ok": False, "error": msg}, status_code=409)
    return {"ok": True, "message": msg, "status": service.status}


@router.post("/api/runs/{name}/yt_meta_refresh")
async def api_yt_meta_refresh(name: str, mode: str = ""):
    """重新生成 youtube_metadata.json（脚本编辑后刷新标题/简介/章节，零 AI 成本）。"""
    service = get_service()
    ok, msg = service.refresh_youtube_metadata(name, mode=mode)
    if not ok:
        return JSONResponse({"ok": False, "error": msg}, status_code=409)
    meta_path = (Path(load_config().get("output_dir", "./output")) / name
                 / "youtube_metadata.json")
    meta = {}
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        pass
    return {"ok": True, "message": msg, "metadata": meta}


def _list_explorer_windows() -> set[int]:
    """枚举当前所有可见的资源管理器文件夹窗口（CabinetWClass）句柄。"""
    import ctypes
    user32 = ctypes.windll.user32
    found: set[int] = set()
    proto = ctypes.WINFUNCTYPE(ctypes.c_ssize_t, ctypes.c_ssize_t, ctypes.c_ssize_t)

    def _cb(hwnd, _lparam):
        buf = ctypes.create_unicode_buffer(64)
        user32.GetClassNameW(hwnd, buf, 64)
        if buf.value == "CabinetWClass" and user32.IsWindowVisible(hwnd):
            found.add(hwnd)
        return True

    user32.EnumWindows(proto(_cb), 0)
    return found


def _bring_window_to_front(hwnd) -> None:
    """把窗口强制带到前台获得焦点。先按一下 ALT 再 SetForegroundWindow，
    绕过 Windows 对非前台进程的焦点锁定限制；SW_RESTORE 兼容最小化状态。"""
    import ctypes
    user32 = ctypes.windll.user32
    user32.keybd_event(0x12, 0, 0, 0)   # ALT down
    user32.keybd_event(0x12, 0, 2, 0)   # ALT up（KEYEVENTF_KEYUP）
    user32.ShowWindow(hwnd, 9)          # SW_RESTORE
    user32.SetForegroundWindow(hwnd)


def _raise_folder_window(folder_name: str, before: set[int], timeout: float = 3.0) -> None:
    """等待新开的资源管理器窗口出现并置顶：优先找新建句柄，否则按标题匹配（兼容旧窗口新开标签页）。"""
    import ctypes
    deadline = time.time() + timeout
    while time.time() < deadline:
        windows = _list_explorer_windows()
        fresh = [h for h in windows if h not in before]
        if fresh:
            _bring_window_to_front(fresh[0])
            return
        for h in windows:
            buf = ctypes.create_unicode_buffer(256)
            ctypes.windll.user32.GetWindowTextW(h, buf, 256)
            if buf.value.strip() == folder_name:
                _bring_window_to_front(h)
                return
        time.sleep(0.15)


@router.post("/api/runs/{name}/open_folder")
async def api_open_folder(name: str, mode: str = ""):
    """在系统文件管理器中打开运行目录（视频及所有素材所在目录）。"""
    config = load_config()
    output_dir = Path(config.get("output_dir", "./output"))
    run_dir = find_run_dir(output_dir, name, mode)
    if not run_dir or not run_dir.is_dir():
        return JSONResponse({"ok": False, "error": "Not found"}, status_code=404)
    # Safety: ensure it's within output_dir
    if not str(run_dir.resolve()).startswith(str(output_dir.resolve())):
        return JSONResponse({"ok": False, "error": "Invalid path"}, status_code=400)
    try:
        if sys.platform == "win32":
            # 记录已有窗口，打开后把新资源管理器窗口强行带到前台（否则常被浏览器挡住）
            before = _list_explorer_windows()
            os.startfile(str(run_dir))
            threading.Thread(
                target=_raise_folder_window,
                args=(run_dir.name, before), daemon=True).start()
        else:
            import subprocess
            opener = "open" if sys.platform == "darwin" else "xdg-open"
            subprocess.Popen([opener, str(run_dir)])
        return {"ok": True, "path": str(run_dir)}
    except OSError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@router.get("/api/runs/{name}/script/edit")
async def api_get_script_edit(name: str, mode: str = ""):
    """Get script for editing (returns full JSON)."""
    config = load_config()
    output_dir = Path(config.get("output_dir", "./output"))
    run_dir = find_run_dir(output_dir, name, mode)
    script_path = run_dir / "script.json" if run_dir else None
    if not script_path or not script_path.exists():
        return JSONResponse({"error": "Script not found"}, status_code=404)
    return json.loads(script_path.read_text(encoding="utf-8"))


@router.post("/api/runs/{name}/script/save")
async def api_save_script(name: str, request: Request, mode: str = ""):
    """Save edited script JSON."""
    config = load_config()
    output_dir = Path(config.get("output_dir", "./output"))
    run_dir = find_run_dir(output_dir, name, mode)
    script_path = run_dir / "script.json" if run_dir else None
    if not script_path or not script_path.exists():
        return JSONResponse({"error": "Script not found"}, status_code=404)
    data = await request.json()
    script_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"ok": True}


@router.get("/api/runs/{name}/gallery")
async def api_gallery(name: str, mode: str = ""):
    """List all images for a run, grouped by type."""
    config = load_config()
    output_dir = Path(config.get("output_dir", "./output"))
    run_dir = find_run_dir(output_dir, name, mode)
    images_dir = run_dir / "images" if run_dir else None
    if not images_dir or not images_dir.exists():
        return {"images": [], "clips": []}

    images = sorted([f.name for f in images_dir.glob("*.png")])
    clips_dir = run_dir / "clips"
    clips = sorted([f.name for f in clips_dir.glob("*.mp4")]) if clips_dir.exists() else []
    audio_dir = run_dir / "audio"
    audio = sorted([f.name for f in audio_dir.glob("*.mp3")]) if audio_dir.exists() else []

    # Final videos in work_dir root (excluding intermediate files)
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


@router.get("/api/runs/{name}/audio/{audio_name}")
async def api_get_audio(name: str, audio_name: str, mode: str = ""):
    config = load_config()
    output_dir = Path(config.get("output_dir", "./output"))
    run_dir = find_run_dir(output_dir, name, mode)
    audio_path = run_dir / "audio" / audio_name if run_dir else None
    if not audio_path or not audio_path.exists():
        return JSONResponse({"error": "Not found"}, status_code=404)
    return FileResponse(str(audio_path), media_type="audio/mpeg")


@router.get("/api/runs/{name}/srt")
async def api_get_srt(name: str, mode: str = ""):
    """Serve the SRT subtitle file for step-mode review."""
    config = load_config()
    output_dir = Path(config.get("output_dir", "./output"))
    run_dir = find_run_dir(output_dir, name, mode)
    srt_path = run_dir / "subtitles" / "output.srt" if run_dir else None
    if not srt_path or not srt_path.exists():
        return JSONResponse({"error": "SRT not found"}, status_code=404)
    return PlainTextResponse(srt_path.read_text(encoding="utf-8"), media_type="text/plain")
