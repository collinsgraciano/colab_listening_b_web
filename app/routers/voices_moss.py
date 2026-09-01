"""MOSS-TTS Voice Management API — 音色列表/试听/克隆/默认配置（页面路由在 pages.py）."""
import asyncio
import json
import time
from pathlib import Path

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from ..config_manager import load_config
from ..paths import (
    MOSS_PREVIEWS_DIR, MOSS_VOICES_DIR, MOSS_VOICE_CONFIG_PATH,
)
from ..tts_state import TTS_SYNTH_LOCK as _TTS_SYNTH_LOCK, PREVIEW_TEXTS as _PREVIEW_TEXTS

router = APIRouter()


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
            # update 而非逐已知键回拷：保留未知键（如按模式分套的 modes）
            defaults.update(saved)
        except (json.JSONDecodeError, OSError):
            pass
    return defaults


def _save_moss_voice_config(config: dict) -> None:
    MOSS_VOICE_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    MOSS_VOICE_CONFIG_PATH.write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


@router.get("/api/moss_voices/speakers")
async def api_moss_speakers():
    """List all available MOSS voices (preset + custom)."""
    from moss_tts_engine import MOSS_PRESET_VOICES, get_all_moss_voices
    return {"speakers": get_all_moss_voices(), "presets": MOSS_PRESET_VOICES}


@router.post("/api/moss_voices/preview")
async def api_moss_preview(request: Request):
    """Preview a MOSS voice: serve cached audio if available, else generate & cache."""

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


@router.get("/api/moss_voices/preview/{voice}")
async def api_moss_preview_cached(voice: str, language: str = "english"):
    """Serve a cached preview if it exists (instant playback), 404 otherwise."""
    text = _PREVIEW_TEXTS.get(language, _PREVIEW_TEXTS["english"])
    p = _moss_preview_cache_path(voice, language, text)
    if not p.exists():
        return JSONResponse({"ok": False, "error": "no cached preview"}, status_code=404)
    return FileResponse(str(p), media_type="audio/mpeg")


# 按模式分套默认音色的合法键（与 pipeline/moss_tts_engine.MOSS_VOICE_DEFAULTS 同名）
_MOSS_DEFAULT_KEYS = ("default_male", "default_female", "default_host_female",
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


@router.get("/api/moss_voices/defaults")
async def api_moss_defaults_get():
    """Get default voice configuration (含按模式分套的 modes)."""
    from moss_tts_engine import MOSS_VOICE_DEFAULTS
    return _defaults_payload(_load_moss_voice_config(), MOSS_VOICE_DEFAULTS)


@router.put("/api/moss_voices/defaults")
async def api_moss_defaults_put(request: Request):
    """Update default voice configuration（按模式写入 modes[mode]）."""
    from tts_engine import VOICE_DEFAULT_MODES
    data = await request.json()
    mode = data.get("mode")
    if mode not in VOICE_DEFAULT_MODES:
        return JSONResponse({"ok": False, "error": f"invalid mode: {mode}"},
                            status_code=400)
    config = _load_moss_voice_config()
    section = config.setdefault("modes", {}).setdefault(mode, {})
    for key in _MOSS_DEFAULT_KEYS:
        if key in data:
            section[key] = data[key]
    _save_moss_voice_config(config)
    return {"ok": True, "mode": mode, "config": section}


@router.post("/api/moss_voices/custom")
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


@router.delete("/api/moss_voices/custom/{name}")
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


