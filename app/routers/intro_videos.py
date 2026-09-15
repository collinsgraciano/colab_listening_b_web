"""🎬 片头库 API — sleep 模式片头生成（本地动画 / MCP AI 视频，时长可设 4-15s）
+ 自定义片头上传 + 库管理 + LLM 随机片头提示词生成。

- 本地路线：pipeline/sleep/intro_video.build_local_intro —— Pillow 逐帧渲染
  sleep 主题动画（渐变背景 + 叶片漂动 + 频道名淡入），零积分；
- AI 路线：PageMcpSession generate_video（text_to_video / 时长可设 / 16:9 /
  720p / 无音频）→ 下载 → finalize_ai_intro 标准化（频道名由 AI 画进画面，
  本地不再叠加文字防重复）；
- 提示词：POST /gen_prompts 仅凭频道名让 LLM 一次生成 10 个随机片头场景
  提示词（prompt_en 含频道名入画 title moment + desc_zh 简体中文说明），
  前端点选填入 AI 画面描述；
- 上传：POST /upload 自带视频 → standardize_upload_intro 规格统一（保留
  原声与原时长，无音轨补静音）→ source=upload 入库；
- 音频统一：BGM（bgm_music 库选一/随机）淡入淡出 + 可选频道名 TTS 播报
  （sleep 模式 tts_engine 合成，TTS_SYNTH_LOCK 内执行）；
- 产物 configs/intro_videos/{id}/intro.mp4，索引 configs/intro_library.json；
- POST /use 写入 sleep 模式配置 sleep_intro_video（pipeline_service 注入 CLI），
  空 = 回退默认片头（静态卡片 + 频道名播报）。
"""
import json
import shutil
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from ..config_manager import (load_config, load_mode_config, resolve_provider,
                              save_mode_config)
from ..intro_library import (INTRO_ID_RE, INTRO_VIDEOS_DIR, load_library,
                             resolve_video_path, save_library)
from ..page_mcp import PageMcpSession
from ..paths import WEB_ROOT
from ..tts_state import TTS_SYNTH_LOCK

router = APIRouter()

_gen_status: dict = {"status": "idle", "error": "", "intro_id": "", "logs": []}
_prompt_status: dict = {"status": "idle", "prompts": [], "error": ""}

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


def _parse_duration(v) -> float:
    """片头时长（秒）：clamp [4, 15]（MCP Seedance 2.0 默认模型上限）；空/非法回退 10。"""
    if v is None or str(v).strip() == "":
        return 10.0
    try:
        dur = float(v)
    except (TypeError, ValueError):
        return 10.0
    return round(max(4.0, min(15.0, dur)), 1)


def _build_ai_prompt(scene_prompt: str, channel: str) -> str:
    """场景提示词 + 频道名入画约束（AI 画字需逐字母强调拼写，且为画面唯一文字）。"""
    base = scene_prompt.strip() or _DEFAULT_AI_SCENE
    name = (channel or "").strip() or "English with me"
    return (base + f' The channel name "{name}" appears in the scene, spelled '
            f'EXACTLY "{name}" letter-for-letter — it is the ONLY text allowed; '
            "once it appears, it stays continuously visible in the CENTER of "
            "the frame until the very end, never disappearing and never "
            "drifting away from the center; "
            "no other text, no captions, no watermarks.")


def _generate_ai_video(scene_prompt: str, dest: Path,
                       duration: float = 10.0, channel: str = "") -> None:
    """MCP generate_video 原始场景视频（频道名由 AI 画进画面，无音频，播报本地混）。"""
    # token 解析链：sleep 模式配置 → legacy default.json → 本机 CLI 检测
    # （sleep 模式文件 mcp_tokens 可能为空）
    from ..config_manager import resolve_mcp_tokens
    tokens = [t.strip() for t in resolve_mcp_tokens("sleep").splitlines()
              if t.strip()]
    if not tokens:
        raise RuntimeError("未配置 MCP Token（模式配置 / default.json / 本地检测均为空）"
                           "—— AI 片头需要 MCP，或改用本地动画路线")
    session = PageMcpSession(tokens).initialize()
    prompt = _build_ai_prompt(scene_prompt, channel)
    _log(f"MCP generate_video 提交中（{duration:g}s / 720p / 16:9 / 无音频）...")
    result = session.call_tool("generate_video", {
        "mode": "text_to_video", "prompt": prompt,
        "duration": duration, "ratio": "16:9", "resolution": "720p",
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
        dur = _parse_duration(params.get("duration"))
        out_dir.mkdir(parents=True, exist_ok=True)
        _log(f"生成片头（{'AI 视频' if route == 'ai' else '本地动画'}，{dur:g}s）: {channel}")
        # "none" = 用户显式不使用 BGM（成片时由 bgm_mix 统一混入，避免片头双重 BGM）；
        # resolve_bgm 对未知名本就返回 ""，这里显式短路让日志语义准确
        if str(params["bgm"]) == "none":
            bgm_path = ""
            _log("BGM: 不使用（成片生成时由 bgm_mix 统一混入）")
        else:
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
            _generate_ai_video(params["scene_prompt"], raw_path, dur, channel)
            from sleep.intro_video import finalize_ai_intro
            finalize_ai_intro(str(raw_path), channel, subtitle, str(final_path),
                              theme, bgm_path=bgm_path,
                              bgm_volume_db=params["bgm_volume_db"],
                              announce_path=announce_path,
                              duration=dur,
                              overlay_text=False,
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
                              duration=dur,
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
    announce = bool(data.get("announce", True))
    scene_prompt = str(data.get("scene_prompt", "") or "").strip()[:600]
    duration = _parse_duration(data.get("duration"))

    threading.Thread(target=_generate_worker,
                     args=({"route": route, "channel_name": channel,
                            "subtitle": subtitle, "bgm": bgm,
                            "bgm_volume_db": bgm_volume_db,
                            "announce": announce,
                            "scene_prompt": scene_prompt,
                            "duration": duration},),
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
# LLM 随机片头提示词（仅凭频道名 → 10 个 prompt_en + desc_zh 中文说明）
# ---------------------------------------------------------------------------

_PROMPT_SYSTEM = (
    "You are a creative director for cinematic AI-generated intro videos. "
    "Output valid JSON only — no markdown, no explanations."
)


def _build_prompts_prompt(channel: str) -> str:
    name = (channel or "").strip() or "English with me"
    return f"""Create 10 clearly different ambient intro video scene concepts for a sleep-relaxation English learning YouTube channel named "{name}". Audience: overseas Chinese ESL learners winding down before sleep.

Each concept is an AI text-to-video prompt. Mood: calm, dreamy and peaceful — perfect for falling asleep. Every concept MUST feature ONE title moment: the channel name "{name}" appears in the scene, spelled EXACTLY "{name}", and once it appears it stays in the CENTER of the frame continuously until the end.

For each concept output:
- "prompt_en": one rich English paragraph (70-120 words) describing ONE continuous very slow shot: the scene, lighting, color mood, art style (vary across concepts: soft 3D Pixar animation, dreamy pastel illustration, cinematic realism, watercolor, etc.), and a very slow gentle camera drift. Include the title moment: the channel name "{name}" appears within the first two seconds and then remains continuously visible in the CENTER of the frame for the ENTIRE rest of the video — once it is visible it must NEVER disappear or leave the center (no fading out, no drifting around or out of frame, no vanishing and re-appearing) — describe how it materializes and its lettering style (e.g. elegant glowing handwritten script traced by fireflies, soft 3D golden letters drifting out of the clouds, starlight gathering into letters). The channel name "{name}" is the ONLY text in the scene, spelled EXACTLY letter-for-letter — never any other words, letters, captions, subtitles or watermarks.
- "desc_zh": 1-2 句简体中文，概括这段画面长什么样（含频道名如何出现，让用户不看英文也能想象出视频的大致样子）。

The 10 concepts must span clearly different scenes/moods (for example: starry night sky with drifting clouds, cozy bedroom by a rainy window, moonlit forest, calm ocean waves at night, floating lanterns or dreamy clouds, snowy mountain cabin at dusk, sailboat gliding under moonlight, midnight garden full of fireflies) — never two similar ones.

Output valid JSON only:
{{"intros": [{{"prompt_en": "...", "desc_zh": "..."}}, {{"prompt_en": "...", "desc_zh": "..."}}, {{"prompt_en": "...", "desc_zh": "..."}}, {{"prompt_en": "...", "desc_zh": "..."}}, {{"prompt_en": "...", "desc_zh": "..."}}, {{"prompt_en": "...", "desc_zh": "..."}}, {{"prompt_en": "...", "desc_zh": "..."}}, {{"prompt_en": "...", "desc_zh": "..."}}, {{"prompt_en": "...", "desc_zh": "..."}}, {{"prompt_en": "...", "desc_zh": "..."}}]}}"""


def _llm_chat(base_url: str, api_key: str, model: str, p_type: str,
              prompt: str, proxy_url: str = "") -> str:
    """同步调 LLM 生成提示词，返回 content。独立函数便于测试 monkeypatch。

    与 channel_factory._llm_chat 同模式：gemini 走 SDK，其余 OpenAI 兼容
    /chat/completions（sensenova 等附 reasoning_effort=low）。
    """
    from llm_client import gemini_chat, llm_urlopen  # pipeline/ 已在 sys.path
    messages = [{"role": "system", "content": _PROMPT_SYSTEM},
                {"role": "user", "content": prompt}]
    if p_type == "gemini":
        return gemini_chat(api_key, model, messages, temperature=0.95,
                           max_tokens=8192, timeout=180, proxy_url=proxy_url)
    body = {"model": model, "messages": messages,
            "temperature": 0.95, "max_tokens": 8192}
    if p_type != "openai":
        body["reasoning_effort"] = "low"
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(body).encode("utf-8"), method="POST")
    req.add_header("Authorization", f"Bearer {api_key}")
    req.add_header("Content-Type", "application/json")
    req.add_header("User-Agent", "CodelyLLM/1.0")
    try:
        with llm_urlopen(req, 180, proxy_url) as resp:
            result = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode("utf-8", errors="replace")[:300]
        except Exception:  # noqa: BLE001
            pass
        raise RuntimeError(f"LLM API HTTP {e.code}: {detail}") from e
    choice = (result.get("choices") or [{}])[0]
    return str(choice.get("message", {}).get("content", ""))


def _normalize_prompts(raw_list) -> list[dict]:
    """LLM 返回列表 → [{"prompt_en", "desc_zh"}]（无 prompt_en 的条目丢弃）。"""
    out: list[dict] = []
    for raw in raw_list or []:
        if not isinstance(raw, dict):
            continue
        en = str(raw.get("prompt_en", "") or "").strip()
        zh = str(raw.get("desc_zh", "") or "").strip()
        if en:
            out.append({"prompt_en": en[:900], "desc_zh": zh[:160]})
    return out


def _prompts_worker(channel: str) -> None:
    _prompt_status.update({"status": "generating", "prompts": [], "error": ""})
    try:
        from llm_client import _extract_json, proxy_url_from_config  # pipeline/ 已在 sys.path
        cfg = load_config()
        p_type, base_url, api_key, model = resolve_provider(cfg)
        if not api_key:
            raise RuntimeError(f"未配置 {p_type} 的 API Key，请在参数配置页填写")
        if not model:
            raise RuntimeError("未指定模型（该 Provider 未配置模型列表）")
        _log(f"LLM 生成片头提示词：{model} ({p_type})，频道「{channel}」")
        content = _llm_chat(base_url, api_key, model, p_type,
                            _build_prompts_prompt(channel),
                            proxy_url=proxy_url_from_config(cfg))
        data = _extract_json(content)
        if isinstance(data, dict):
            raw_list = data.get("intros")
        elif isinstance(data, list):
            raw_list = data
        else:
            raw_list = None
        prompts = _normalize_prompts(raw_list)
        if not prompts:
            raise RuntimeError("LLM 返回内容中没有有效提示词，请重试或更换模型")
        _log(f"已生成 {len(prompts)} 条提示词")
        _prompt_status.update({"status": "done", "prompts": prompts, "error": ""})
    except Exception as e:  # noqa: BLE001 — 错误原样落状态供前端展示
        print(f"  [IntroLibrary] PROMPTS ERROR: {e}")
        _log(f"ERROR: {e}")
        _prompt_status.update({"status": "error", "prompts": [], "error": str(e)[:300]})


@router.post("/api/intro_videos/gen_prompts")
async def api_gen_prompts(request: Request):
    """LLM 生成 10 个随机片头提示词（单槽 409 守卫；静态路径须在 {intro_id} 动态路由之前）。"""
    if _prompt_status.get("status") == "generating":
        return JSONResponse({"ok": False, "error": "提示词生成进行中，请稍候"},
                            status_code=409)
    try:
        data = await request.json()
    except Exception:
        data = {}
    cfg = _sleep_cfg()
    channel = str(data.get("channel_name", "") or "").strip()[:60] \
        or str(cfg.get("sleep_channel_name", "") or "").strip() \
        or "English with me"
    threading.Thread(target=_prompts_worker, args=(channel,), daemon=True).start()
    return {"ok": True, "message": "提示词生成中（约 10-60 秒）..."}


@router.get("/api/intro_videos/prompts_status")
async def api_prompts_status():
    return _prompt_status


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


_UPLOAD_MAX_BYTES = 200 * 1024 * 1024
_UPLOAD_EXTS = (".mp4", ".mov", ".webm", ".mkv", ".m4v", ".avi")


@router.post("/api/intro_videos/upload")
async def api_upload(video: UploadFile = File(...), name: str = Form("")):
    """上传自定义片头 → 规格标准化入库（source=upload，保留原声与原时长）。

    纯本地 ffmpeg 处理，与生成槽/MCP 互斥无关，同步返回。
    """
    filename = (video.filename or "").lower()
    if not filename.endswith(_UPLOAD_EXTS):
        return JSONResponse({"ok": False,
                             "error": "仅支持 mp4/mov/webm/mkv/m4v/avi 视频文件"},
                            status_code=400)
    content = await video.read()
    if len(content) < 10240:
        return JSONResponse({"ok": False, "error": "视频文件过小"}, status_code=400)
    if len(content) > _UPLOAD_MAX_BYTES:
        return JSONResponse({"ok": False, "error": "视频过大（上限 200MB）"},
                            status_code=400)
    display = str(name or "").strip()[:60] \
        or Path(filename).stem.strip()[:60] or "自定义片头"
    intro_id = f"intro_{int(time.time() * 1000)}"
    out_dir = INTRO_VIDEOS_DIR / intro_id
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / f"_raw{Path(filename).suffix or '.mp4'}"
    final_path = out_dir / "intro.mp4"
    try:
        raw_path.write_bytes(content)
        from sleep.intro_video import standardize_upload_intro
        dur = standardize_upload_intro(str(raw_path), str(final_path))
    except Exception as e:  # noqa: BLE001 — 清理后原样返回错误
        shutil.rmtree(out_dir, ignore_errors=True)
        print(f"  [IntroLibrary] UPLOAD ERROR: {e}")
        return JSONResponse({"ok": False,
                             "error": f"片头标准化失败: {str(e)[:200]}"},
                            status_code=500)
    finally:
        try:
            raw_path.unlink()
        except OSError:
            pass
    entry = {"id": intro_id, "name": display, "source": "upload",
             "duration": round(dur, 2), "created": time.time()}
    lib = load_library()
    lib.insert(0, entry)
    save_library(lib)
    print(f"  [IntroLibrary] 片头上传入库: {intro_id} ({display}, {dur:.1f}s)")
    return {"ok": True, "id": intro_id, "duration": round(dur, 2)}


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
