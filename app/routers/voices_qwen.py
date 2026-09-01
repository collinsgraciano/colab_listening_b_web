"""QwenTTS Voice Management API — 音色列表/试听/克隆/设计/候选/冻结（页面路由在 pages.py）."""
import asyncio
import json
import threading
import time
from pathlib import Path

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from ..config_manager import load_config, resolve_provider
from ..paths import (
    CUSTOM_VOICES_DIR, QWEN_VOICE_CONFIG_PATH, VOICE_PREVIEWS_DIR,
)
from ..tts_state import TTS_SYNTH_LOCK as _TTS_SYNTH_LOCK, PREVIEW_TEXTS as _PREVIEW_TEXTS

router = APIRouter()


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
            # update 而非逐已知键回拷：保留未知键（如按模式分套的 modes）
            defaults.update(saved)
        except (json.JSONDecodeError, OSError):
            pass
    return defaults


def _save_qwen_voice_config(config: dict) -> None:
    QWEN_VOICE_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    QWEN_VOICE_CONFIG_PATH.write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


@router.get("/api/qwen_voices/speakers")
async def api_qwen_speakers():
    """List all available voices (preset + custom)."""
    from qwen_tts_engine import QWEN_SPEAKERS, get_all_voices
    return {"speakers": get_all_voices(), "presets": QWEN_SPEAKERS}


@router.post("/api/qwen_voices/preview")
async def api_qwen_preview(request: Request):
    """Preview a voice: serve cached audio if available, else generate & cache."""

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


@router.get("/api/qwen_voices/preview/{voice}")
async def api_qwen_preview_cached(voice: str, language: str = "english"):
    """Serve a cached preview if it exists (instant playback), 404 otherwise."""
    text = _PREVIEW_TEXTS.get(language, _PREVIEW_TEXTS["english"])
    p = _preview_cache_path(voice, language, text)
    if not p.exists():
        return JSONResponse({"ok": False, "error": "no cached preview"}, status_code=404)
    return FileResponse(str(p), media_type="audio/mpeg")


# 按模式分套默认音色的合法键（与 pipeline/qwen_tts_engine.QWEN_VOICE_DEFAULTS 同名）
_QWEN_DEFAULT_KEYS = ("default_male", "default_female", "default_host_female",
                      "default_male2", "default_female2",
                      "default_male3", "default_female3",
                      "default_host_male",
                      "default_male_zh", "default_female_zh",
                      "default_male_zh2", "default_female_zh2")


def _defaults_payload(saved: dict, builtin: dict) -> dict:
    """原配置 + 补全的 modes（每键 modes[mode] → 平铺 legacy → 内置默认）."""
    from tts_engine import VOICE_DEFAULT_MODES
    filled = {}
    for m in VOICE_DEFAULT_MODES:
        section = (saved.get("modes") or {}).get(m) or {}
        filled[m] = {k: section.get(k) or saved.get(k) or builtin[k] for k in builtin}
    return {**saved, "modes": filled}


@router.get("/api/qwen_voices/defaults")
async def api_qwen_defaults_get():
    """Get default voice configuration (含按模式分套的 modes)."""
    from qwen_tts_engine import QWEN_VOICE_DEFAULTS
    return _defaults_payload(_load_qwen_voice_config(), QWEN_VOICE_DEFAULTS)


@router.put("/api/qwen_voices/defaults")
async def api_qwen_defaults_put(request: Request):
    """Update default voice configuration（按模式写入 modes[mode]）."""
    from tts_engine import VOICE_DEFAULT_MODES
    data = await request.json()
    mode = data.get("mode")
    if mode not in VOICE_DEFAULT_MODES:
        return JSONResponse({"ok": False, "error": f"invalid mode: {mode}"},
                            status_code=400)
    config = _load_qwen_voice_config()
    section = config.setdefault("modes", {}).setdefault(mode, {})
    for key in _QWEN_DEFAULT_KEYS:
        if key in data:
            section[key] = data[key]
    _save_qwen_voice_config(config)
    return {"ok": True, "mode": mode, "config": section}


@router.post("/api/qwen_voices/custom")
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


@router.delete("/api/qwen_voices/custom/{name}")
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
    from qwen_tts_engine import DESIGNED_VOICES_BUILTIN
    if target.get("frozen_from") == "designed" and \
            name in {v["name"] for v in DESIGNED_VOICES_BUILTIN}:
        dismissed = config.setdefault("builtin_voice_dismissed", [])
        if name not in dismissed:
            dismissed.append(name)
    _save_qwen_voice_config(config)
    return {"ok": True}


@router.post("/api/qwen_voices/designed")
async def api_qwen_designed_create(request: Request):
    """Create a VoiceDesign voice from a natural-language instruct description."""

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


@router.delete("/api/qwen_voices/designed/{name}")
async def api_qwen_designed_delete(name: str):
    """Delete a user-created designed voice (builtin ones are protected)."""

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


@router.get("/api/qwen_voices/candidates")
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


@router.post("/api/qwen_voices/candidates/generate")
async def api_qwen_candidates_generate(request: Request):
    """Generate a batch of random voice designs via LLM (replaces current candidates)."""

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


@router.post("/api/qwen_voices/candidates/{name}/save")
async def api_qwen_candidate_save(name: str):
    """Save a candidate voice as a user-designed voice."""

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


@router.delete("/api/qwen_voices/candidates/{name}")
async def api_qwen_candidate_delete(name: str):
    """Discard a candidate voice."""
    config = _load_qwen_voice_config()
    if not any(c["name"] == name for c in config.get("candidate_voices", [])):
        return JSONResponse({"ok": False, "error": "未找到"}, status_code=404)
    config["candidate_voices"] = [c for c in config.get("candidate_voices", []) if c["name"] != name]
    _save_qwen_voice_config(config)
    _purge_preview_cache(name)
    return {"ok": True}


@router.post("/api/qwen_voices/candidates/delete_batch")
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


@router.post("/api/qwen_voices/candidates/previews/generate")
async def api_qwen_candidates_previews_generate():
    """一键为所有候选音色后台生成试听音频（已缓存的直接计入完成）。"""

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


@router.get("/api/qwen_voices/candidates/previews/status")
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


@router.get("/api/qwen_voices/freeze/status")
async def api_qwen_freeze_status():
    """冻结任务进度（前端轮询）。"""
    with _VOICE_FREEZE_LOCK:
        return {"jobs": {k: dict(v) for k, v in _VOICE_FREEZE_JOBS.items()}}


def auto_freeze_pending_designed_voices() -> None:
    """启动时把尚未转换为克隆音色的设计音色入队后台冻结（一次性）。

    覆盖：内置 4 个英文女声（Ella/Maya/Chloe/Hazel）+ designed_voices 中
    残留的未冻结条目（此前冻结失败的会借此重试）。
    已冻结（custom_voices 同名存在）或已删除（builtin_voice_dismissed）的跳过。
    """
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


