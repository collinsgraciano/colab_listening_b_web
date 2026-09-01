"""Kokoro TTS Voices API — 本地引擎固定音色清单 + 试听 + 默认音色配置（页面路由在 pages.py）."""
import asyncio
import json
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, JSONResponse

from ..paths import KOKORO_VOICE_CONFIG_PATH, VOICE_PREVIEWS_DIR
from ..tts_state import TTS_SYNTH_LOCK as _TTS_SYNTH_LOCK, PREVIEW_TEXTS as _PREVIEW_TEXTS

router = APIRouter()


def _kokoro_preview_cache_path(speaker: str, text: str) -> Path:
    import hashlib
    safe = "".join(c for c in speaker if c.isalnum() or c in "-_") or "voice"
    h = hashlib.md5(f"{speaker}|{text}".encode("utf-8")).hexdigest()[:8]
    return VOICE_PREVIEWS_DIR / f"kokoro__{safe}__{h}.mp3"


@router.get("/api/kokoro_voices/speakers")
async def api_kokoro_speakers():
    """List all Kokoro voices with cached flag (未缓存音色本机无法自动下载)."""
    from tts_engine import get_all_kokoro_voices
    return {"speakers": get_all_kokoro_voices()}


async def _kokoro_preview_common(speaker: str, text: str, regenerate: bool):
    """Serve cached Kokoro preview if available, else synthesize & cache."""
    cache_path = _kokoro_preview_cache_path(speaker, text)
    if cache_path.exists() and not regenerate:
        return FileResponse(str(cache_path), media_type="audio/mpeg")


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


@router.post("/api/kokoro_voices/preview")
async def api_kokoro_preview(request: Request):
    """Preview a Kokoro voice: serve cached audio if available, else generate & cache."""
    data = await request.json()
    speaker = data.get("speaker", "af_sarah")
    text = data.get("text", "") or _PREVIEW_TEXTS.get("english", "Hello, this is a voice preview test.")
    regenerate = bool(data.get("regenerate", False))
    return await _kokoro_preview_common(speaker, text, regenerate)


@router.get("/api/kokoro_voices/preview/{voice}")
async def api_kokoro_preview_cached(voice: str, language: str = "english"):
    """Serve a cached preview if it exists (instant playback), 404 otherwise."""
    text = _PREVIEW_TEXTS.get(language, _PREVIEW_TEXTS["english"])
    p = _kokoro_preview_cache_path(voice, text)
    if not p.exists():
        return JSONResponse({"ok": False, "error": "no cached preview"}, status_code=404)
    return FileResponse(str(p), media_type="audio/mpeg")


# Kokoro voice defaults config (性别自动分配的默认音色，与 Qwen/MOSS 对齐)

# 按模式分套默认音色的合法键（与 pipeline/tts_engine.KOKORO_VOICE_DEFAULTS 同名）
_KOKORO_DEFAULT_KEYS = ("default_male", "default_female", "default_host_female",
                        "default_host_male",
                        "default_male2", "default_female2",
                        "default_male3", "default_female3")


def _load_kokoro_voice_config() -> dict:
    """Load kokoro_voice_config.json raw content (文件不存在/坏 JSON 返回 {})."""
    if KOKORO_VOICE_CONFIG_PATH.exists():
        try:
            return json.loads(KOKORO_VOICE_CONFIG_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_kokoro_voice_config(config: dict) -> None:
    KOKORO_VOICE_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    KOKORO_VOICE_CONFIG_PATH.write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


def _defaults_payload(saved: dict, builtin: dict) -> dict:
    """原配置 + 补全的 modes（每键 modes[mode] → 平铺 legacy → 内置默认）."""
    from tts_engine import VOICE_DEFAULT_MODES
    filled = {}
    for m in VOICE_DEFAULT_MODES:
        section = (saved.get("modes") or {}).get(m) or {}
        filled[m] = {k: section.get(k) or saved.get(k) or builtin[k] for k in builtin}
    return {**saved, "modes": filled}


@router.get("/api/kokoro_voices/defaults")
async def api_kokoro_defaults_get():
    """Get Kokoro default voice configuration (含按模式分套的 modes)."""
    from tts_engine import KOKORO_VOICE_DEFAULTS
    return _defaults_payload(_load_kokoro_voice_config(), KOKORO_VOICE_DEFAULTS)


@router.put("/api/kokoro_voices/defaults")
async def api_kokoro_defaults_put(request: Request):
    """Update Kokoro default voice configuration（按模式写入 modes[mode]）."""
    from tts_engine import VOICE_DEFAULT_MODES
    data = await request.json()
    mode = data.get("mode")
    if mode not in VOICE_DEFAULT_MODES:
        return JSONResponse({"ok": False, "error": f"invalid mode: {mode}"},
                            status_code=400)
    config = _load_kokoro_voice_config()
    section = config.setdefault("modes", {}).setdefault(mode, {})
    for key in _KOKORO_DEFAULT_KEYS:
        if key in data:
            section[key] = data[key]
    _save_kokoro_voice_config(config)
    return {"ok": True, "mode": mode, "config": section}


