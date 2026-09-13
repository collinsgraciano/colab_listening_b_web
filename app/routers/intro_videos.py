"""🎬 片头库 API — sleep 模式 10 秒片头生成（本地动画 / MCP AI 视频）+ 库管理。

- 本地路线：pipeline/sleep/intro_video.build_local_intro —— Pillow 逐帧渲染
  sleep 主题动画（渐变背景 + 叶片漂动 + 频道名淡入），零积分；
- AI 路线：PageMcpSession generate_video（text_to_video / 10s / 16:9 / 720p /
  无音频）→ 下载 → finalize_ai_intro 标准化 + 频道名文字淡入叠加；
- 音频统一：BGM（bgm_music 库选一/随机）淡入淡出 + 可选频道名 TTS 播报
  （sleep 模式 tts_engine 合成，TTS_SYNTH_LOCK 内执行）；
- 产物 configs/intro_videos/{id}/intro.mp4，索引 configs/intro_library.json；
- POST /use 写入 sleep 模式配置 sleep_intro_video（pipeline_service 注入 CLI），
  空 = 回退默认片头（静态卡片 + 频道名播报）。
"""
import shutil
import threading
import time
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, JSONResponse

from ..config_manager import load_mode_config, save_mode_config
from ..intro_library import (INTRO_ID_RE, INTRO_VIDEOS_DIR, load_library,
                             resolve_video_path, save_library)
from ..page_mcp import PageMcpSession
from ..paths import WEB_ROOT
from ..tts_state import TTS_SYNTH_LOCK

router = APIRouter()

_gen_status: dict = {"status": "idle", "error": "", "intro_id": "", "logs": []}

_DEFAULT_AI_SCENE = (
    "A cozy dreamy night scene for a sleep English learning channel intro: "
    "soft fluffy clouds drifting under a starry night sky, warm moonlight "
    "through a window, a few gentle fireflies, soft pastel colors, slow calm "
    "camera drift, smooth 3D Pixar animation style, peaceful sleep atmosphere"
)


def _log(msg: str) -> None:
    _gen_status["logs"].append(f"[{time.strftime('%H:%M:%S')}] {msg}")
    print(f"  [IntroLibrary] {msg}")


def _sleep_cfg() -> dict:
    return load_mode_config("sleep")


# ---------------------------------------------------------------------------
# 片头生成
# ---------------------------------------------------------------------------

def _synth_announce(channel_name: str) -> str:
    """频道名 TTS 播报（sleep 引擎，char_a 男声旁白）。返回 mp3 路径。"""
    from sleep.audio_sleep import build_engine_and_voice_map
    cfg = _sleep_cfg()
    engine_name = str(cfg.get("tts_engine", "kokoro") or "kokoro")
    fake = {"structure": "sleep", "char_a_gender": "male",
            "char_b_gender": "female"}
    out = str(INTRO_VIDEOS_DIR / ".announce_tmp.mp3")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with TTS_SYNTH_LOCK:
        tts, vmap = build_engine_and_voice_map(engine_name, fake)
        voice = vmap.get("char_a") or "am_adam"
        tts.synth_english(channel_name, voice, out, rate="+0%")
    if not Path(out).exists() or Path(out).stat().st_size < 1000:
        raise RuntimeError("频道名播报 TTS 合成失败（无有效音频）")
    return out


def _resolve_bgm(bgm_choice: str) -> str:
    from sleep.intro_video import resolve_bgm
    cfg = _sleep_cfg()
    bgm_dir = str(cfg.get("bgm_music_dir", "") or "").strip() or str(WEB_ROOT / "bgm_music")
    return resolve_bgm(bgm_dir, bgm_choice)


def _generate_ai_video(scene_prompt: str, dest: Path) -> None:
    """MCP generate_video 原始场景视频（无文字无音频，文字由本地叠加保证准确）。"""
    # token 解析链：sleep 模式配置 → legacy default.json → 本机 CLI 检测
    # （sleep 模式文件 mcp_tokens 可能为空）
    from ..config_manager import resolve_mcp_tokens
    tokens = [t.strip() for t in resolve_mcp_tokens("sleep").splitlines()
              if t.strip()]
    if not tokens:
        raise RuntimeError("未配置 MCP Token（模式配置 / default.json / 本地检测均为空）"
                           "—— AI 片头需要 MCP，或改用本地动画路线")
    session = PageMcpSession(tokens).initialize()
    prompt = (scene_prompt.strip() or _DEFAULT_AI_SCENE) + \
        " No text, no letters, no words, no watermark in the scene."
    _log("MCP generate_video 提交中（10s / 720p / 16:9 / 无音频）...")
    result = session.call_tool("generate_video", {
        "mode": "text_to_video", "prompt": prompt,
        "duration": 10, "ratio": "16:9", "resolution": "720p",
        "generate_audio": False,
    })
    task_id = session.parse_task_id(result)
    if not task_id:
        raw = ""
        for item in result.get("result", {}).get("content", []):
            if item.get("type") == "text":
                raw = str(item.get("text", ""))[:300].replace("\n", " ")
                break
        raise RuntimeError(f"MCP 未返回任务 ID（响应: {raw}）")
    _log(f"Task: {task_id[:16]}... 轮询中（AI 视频通常 2-8 分钟）")
    data = session.poll_task(task_id, interval=15, max_wait=1800)
    url = data.get("url", "")
    if data.get("status") != "completed" or not url:
        raise RuntimeError(f"MCP 视频任务未完成: {data.get('status') or 'no status'}"
                           + (f" error={data.get('error')}" if data.get("error") else ""))
    if not session.download_file(url, str(dest)):
        raise RuntimeError("AI 视频下载落盘失败")


def _generate_worker(params: dict) -> None:
    intro_id = f"intro_{int(time.time() * 1000)}"
    _gen_status.update({"status": "running", "error": "", "intro_id": intro_id,
                        "logs": []})
    out_dir = INTRO_VIDEOS_DIR / intro_id
    final_path = out_dir / "intro.mp4"
    try:
        route = params["route"]
        channel = params["channel_name"]
        subtitle = params["subtitle"]
        out_dir.mkdir(parents=True, exist_ok=True)
        _log(f"生成片头（{'AI 视频' if route == 'ai' else '本地动画'}）: {channel}")
        bgm_path = _resolve_bgm(params["bgm"])
        _log(f"BGM: {Path(bgm_path).name}" if bgm_path else "BGM: 无可用音乐（静音）")
        announce_path = ""
        if params["announce"]:
            _log("合成频道名播报 TTS ...")
            announce_path = _synth_announce(channel)

        from sleep.sleep_cards import build_theme
        theme = build_theme(_sleep_cfg())
        if route == "ai":
            raw_path = out_dir / "raw.mp4"
            _generate_ai_video(params["scene_prompt"], raw_path)
            from sleep.intro_video import finalize_ai_intro
            finalize_ai_intro(str(raw_path), channel, subtitle, str(final_path),
                              theme, bgm_path=bgm_path,
                              bgm_volume_db=params["bgm_volume_db"],
                              announce_path=announce_path,
                              progress_cb=lambda p, m: _log(f"[{p}%] {m}"))
            try:
                raw_path.unlink()
            except OSError:
                pass
        else:
            from sleep.intro_video import build_local_intro
            build_local_intro(channel, subtitle, str(final_path), theme,
                              bgm_path=bgm_path,
                              bgm_volume_db=params["bgm_volume_db"],
                              announce_path=announce_path,
                              progress_cb=lambda p, m: _log(f"[{p}%] {m}"))

        from media_utils import get_duration
        entry = {"id": intro_id, "name": channel, "source": route,
                 "duration": round(get_duration(str(final_path)), 2),
                 "created": time.time(), "subtitle": subtitle,
                 "bgm": Path(bgm_path).name if bgm_path else "",
                 "announce": bool(announce_path)}
        lib = load_library()
        lib.insert(0, entry)
        save_library(lib)
        _log(f"片头已入库: {intro_id}")
        _gen_status.update({"status": "done", "intro_id": intro_id})
    except Exception as e:  # noqa: BLE001 — 错误原样落状态供前端展示
        print(f"  [IntroLibrary] ERROR: {e}")
        _log(f"ERROR: {e}")
        _gen_status.update({"status": "error", "error": str(e)[:300]})
        shutil.rmtree(out_dir, ignore_errors=True)


@router.post("/api/intro_videos/generate")
async def api_generate(request: Request):
    """启动片头生成（单槽 409 守卫；静态路径须在 {intro_id} 动态路由之前注册）。"""
    if _gen_status.get("status") == "running":
        return JSONResponse({"ok": False, "error": "已有片头生成任务进行中，请稍候"},
                            status_code=409)
    try:
        data = await request.json()
    except Exception:
        data = {}
    route = str(data.get("route", "local") or "local")
    if route not in ("local", "ai"):
        route = "local"
    cfg = _sleep_cfg()
    channel = str(data.get("channel_name", "") or "").strip()[:60] \
        or str(cfg.get("sleep_channel_name", "") or "").strip() \
        or "English with me"
    subtitle = str(data.get("subtitle", "") or "").strip()[:60]
    bgm = str(data.get("bgm", "") or "").strip()
    try:
        bgm_volume_db = max(-40.0, min(0.0, float(data.get("bgm_volume_db", -16))))
    except (TypeError, ValueError):
        bgm_volume_db = -16.0
    announce = bool(data.get("announce", False))
    scene_prompt = str(data.get("scene_prompt", "") or "").strip()[:600]

    threading.Thread(target=_generate_worker,
                     args=({"route": route, "channel_name": channel,
                            "subtitle": subtitle, "bgm": bgm,
                            "bgm_volume_db": bgm_volume_db,
                            "announce": announce,
                            "scene_prompt": scene_prompt},),
                     daemon=True).start()
    return {"ok": True, "message": "片头生成中（本地约 1-2 分钟 / AI 约 3-10 分钟）..."}


@router.get("/api/intro_videos/status")
async def api_status():
    return _gen_status


@router.get("/api/intro_videos/bgm_list")
async def api_bgm_list():
    from sleep.intro_video import list_bgm_files
    cfg = _sleep_cfg()
    bgm_dir = str(cfg.get("bgm_music_dir", "") or "").strip() or str(WEB_ROOT / "bgm_music")
    return {"files": list_bgm_files(bgm_dir), "dir": bgm_dir}


# ---------------------------------------------------------------------------
# 库管理
# ---------------------------------------------------------------------------

@router.get("/api/intro_videos")
async def api_list():
    used = str(_sleep_cfg().get("sleep_intro_video", "") or "").strip()
    intros = []
    for e in load_library():
        iid = str(e.get("id", ""))
        f = INTRO_VIDEOS_DIR / iid / "intro.mp4" if INTRO_ID_RE.match(iid) else None
        exists = bool(f and f.exists())
        intros.append({**e, "exists": exists, "used": used == iid,
                       "video_url": f"/api/intro_videos/{iid}/video?v="
                                    + (str(int(f.stat().st_mtime_ns)) if exists else "0")})
    return {"intros": intros, "used": used}


@router.post("/api/intro_videos/use")
async def api_use(request: Request):
    """绑定/解绑 sleep 模式片头（body {id: ""} = 回退默认片头）。"""
    try:
        data = await request.json()
    except Exception:
        data = {}
    iid = str(data.get("id", "") or "").strip()
    if iid:
        if not INTRO_ID_RE.match(iid):
            return JSONResponse({"ok": False, "error": "无效的片头 id"}, status_code=400)
        if not (INTRO_VIDEOS_DIR / iid / "intro.mp4").exists():
            return JSONResponse({"ok": False, "error": "片头文件不存在"}, status_code=404)
    cfg = _sleep_cfg()
    cfg["sleep_intro_video"] = iid
    save_mode_config("sleep", cfg)
    return {"ok": True, "used": iid}


@router.delete("/api/intro_videos/{intro_id}")
async def api_delete(intro_id: str):
    if not INTRO_ID_RE.match(intro_id):
        return JSONResponse({"ok": False, "error": "无效的片头 id"}, status_code=400)
    lib = load_library()
    remaining = [e for e in lib if e.get("id") != intro_id]
    if len(remaining) == len(lib):
        return JSONResponse({"ok": False, "error": "未找到该片头"}, status_code=404)
    save_library(remaining)
    shutil.rmtree(INTRO_VIDEOS_DIR / intro_id, ignore_errors=True)
    cfg = _sleep_cfg()
    if str(cfg.get("sleep_intro_video", "") or "") == intro_id:
        cfg["sleep_intro_video"] = ""
        save_mode_config("sleep", cfg)
    return {"ok": True}


@router.get("/api/intro_videos/{intro_id}/video")
async def api_video(intro_id: str):
    if not INTRO_ID_RE.match(intro_id):
        return JSONResponse({"ok": False, "error": "无效的片头 id"}, status_code=400)
    f = INTRO_VIDEOS_DIR / intro_id / "intro.mp4"
    if not f.exists():
        return JSONResponse({"ok": False, "error": "Not found"}, status_code=404)
    return FileResponse(str(f), media_type="video/mp4",
                        headers={"Cache-Control": "no-cache"})


def resolve_bound_intro(intro_sel: str) -> str:
    """供 pipeline_service：sleep_intro_video 配置值 → mp4 路径（无效回退空）。"""
    p = resolve_video_path(intro_sel)
    if not p:
        print(f"  [IntroLibrary] 片头绑定无效（库中不存在）: {intro_sel} —— 回退默认片头")
    return p
