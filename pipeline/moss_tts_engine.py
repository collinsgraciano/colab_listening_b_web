"""MOSS-TTS-Nano engine for English + Chinese TTS.

0.1B voice cloning TTS model, CPU-first.
Uses NanoTTSService from moss_tts_nano_runtime.py (local repo install).

Env vars (set by pipeline.py / pipeline_service.py):
  MOSS_MODEL_PATH       — model checkpoint (H:/models/MOSS-TTS-Nano-Model)
  MOSS_TOKENIZER_PATH   — audio tokenizer (H:/models/MOSS-Audio-Tokenizer-Nano)
  MOSS_DEVICE           — torch device (default: cpu)
  MOSS_REPO_DIR         — repo dir containing moss_tts_nano_runtime.py (H:/models/MOSS-TTS-Nano)
"""
import json
import os
import subprocess
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

_PARENT = str(Path(__file__).parent.resolve())
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from tts_engine import TTSEngine, _split_sentences, _rate_to_speed

# ---------------------------------------------------------------------------
# Built-in voice presets (from moss_tts_nano_runtime._DEFAULT_VOICE_FILES)
# ---------------------------------------------------------------------------

MOSS_PRESET_VOICES = [
    {"name": "Junhao",  "desc": "中文男声 A",   "gender": "male",   "lang": "zh"},
    {"name": "Zhiming", "desc": "中文男声 B",   "gender": "male",   "lang": "zh"},
    {"name": "Weiguo",  "desc": "中文男声 C",   "gender": "male",   "lang": "zh"},
    {"name": "Xiaoyu",  "desc": "中文女声 A",   "gender": "female", "lang": "zh"},
    {"name": "Yuewen",  "desc": "中文女声 B",   "gender": "female", "lang": "zh"},
    {"name": "Lingyu",  "desc": "中文女声 C",   "gender": "female", "lang": "zh"},
    {"name": "Trump",   "desc": "Trump 参考音色", "gender": "male",   "lang": "en"},
    {"name": "Ava",     "desc": "英文女声 A",   "gender": "female", "lang": "en"},
    {"name": "Bella",   "desc": "英文女声 B",   "gender": "female", "lang": "en"},
    {"name": "Adam",    "desc": "英文男声 A",   "gender": "male",   "lang": "en"},
    {"name": "Nathan",  "desc": "英文男声 B",   "gender": "male",   "lang": "en"},
    {"name": "Sakura",  "desc": "日语女声 A",   "gender": "female", "lang": "ja"},
    {"name": "Yui",     "desc": "日语女声 B",   "gender": "female", "lang": "ja"},
    {"name": "Aoi",     "desc": "日语女声 C",   "gender": "female", "lang": "ja"},
    {"name": "Hina",    "desc": "日语女声 D",   "gender": "female", "lang": "ja"},
    {"name": "Mei",     "desc": "日语女声 E",   "gender": "female", "lang": "ja"},
]

_PRESET_NAMES = {s["name"] for s in MOSS_PRESET_VOICES}

# Default auto-assign by gender (built-in preset names)
_DEFAULT_MALE = "Adam"
_DEFAULT_FEMALE = "Ava"
_DEFAULT_HOST_FEMALE = "Bella"


# ---------------------------------------------------------------------------
# Config file helpers
# ---------------------------------------------------------------------------

def _voice_config_path() -> Path:
    return Path(__file__).parent.parent / "configs" / "moss_voice_config.json"


def _load_voice_config() -> dict:
    """Load moss_voice_config.json, return defaults if not exists."""
    defaults = {
        "default_male": _DEFAULT_MALE,
        "default_female": _DEFAULT_FEMALE,
        "default_host_female": _DEFAULT_HOST_FEMALE,
        "custom_voices": [],
    }
    path = _voice_config_path()
    if path.exists():
        try:
            saved = json.loads(path.read_text(encoding="utf-8"))
            for k in defaults:
                if k in saved:
                    defaults[k] = saved[k]
        except (json.JSONDecodeError, OSError):
            pass
    return defaults


def get_all_moss_voices() -> list[dict]:
    """Return preset + custom voices for UI dropdowns."""
    config = _load_voice_config()
    voices = []
    # Built-in presets
    for s in MOSS_PRESET_VOICES:
        voices.append({**s, "preset": True})
    # Custom cloned voices
    for cv in config.get("custom_voices", []):
        voices.append({
            "name": cv["name"],
            "desc": cv.get("description", ""),
            "gender": cv.get("gender", ""),
            "lang": cv.get("language", "english"),
            "custom": True,
        })
    return voices


def get_moss_voice_meta(name: str) -> dict | None:
    """Look up a voice by name. Returns {name, ...} or None.

    Custom voices take priority over presets (frozen clone wins).
    """
    # Check custom voices first
    config = _load_voice_config()
    for cv in config.get("custom_voices", []):
        if cv["name"] == name:
            return cv
    # Check built-in presets
    if name in _PRESET_NAMES:
        return {"name": name, "is_preset": True}
    return None


def build_moss_voice_map(script: dict) -> dict:
    """Build voice_map for MOSS-TTS.

    Priority: script['char_X_moss_voice'] (from library binding) > auto by gender.
    Default voices are read from moss_voice_config.json (preset names or custom voice names).
    """
    config = _load_voice_config()
    voice_map = {}
    for key in ["char_a", "char_b", "char_c", "host"]:
        # Priority 1: moss_voice from script (set by library binding)
        voice = script.get(f"{key}_moss_voice", "")
        if voice:
            voice_map[key] = voice
            continue
        # Priority 2: auto by gender -> use default from config
        gender = script.get(f"{key}_gender", "").lower()
        if not gender:
            continue
        if key == "host":
            voice_map[key] = config["default_host_female"] if gender == "female" else config["default_male"]
        else:
            voice_map[key] = config["default_female"] if gender == "female" else config["default_male"]
    return voice_map


# ---------------------------------------------------------------------------
# Engine class
# ---------------------------------------------------------------------------

class MossTTSEngine(TTSEngine):
    """MOSS-TTS-Nano: 0.1B voice cloning TTS, CPU-first.

    Uses NanoTTSService from moss_tts_nano_runtime.py.
    Built-in presets resolved by service; custom voices use ref_audio.
    Inherits _loudnorm, get_duration from TTSEngine.
    Overrides synth_english and synth_chinese.
    """

    _service = None
    _device = None

    def __init__(self, model_path: str = "", device: str = "cpu",
                 tokenizer_path: str = "", repo_dir: str = ""):
        self._model_path = model_path
        self._tokenizer_path = tokenizer_path
        self._repo_dir = repo_dir
        MossTTSEngine._device = device

    @classmethod
    def _get_service(cls, model_path: str, tokenizer_path: str,
                     device: str, repo_dir: str = ""):
        """Lazy-init NanoTTSService (class-level cache).

        Adds repo_dir to sys.path, imports NanoTTSService, creates instance.
        """
        if cls._service is None:
            if "HF_ENDPOINT" not in os.environ and sys.platform == "win32":
                os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
            # Add MOSS repo dir to sys.path for imports
            if repo_dir and repo_dir not in sys.path:
                sys.path.insert(0, repo_dir)
            # Patch torchaudio to use soundfile instead of torchcodec (Windows DLL fix)
            try:
                import torchaudio_sf_patch  # noqa: F401
            except ImportError:
                pass
            from moss_tts_nano_runtime import NanoTTSService
            print(f"  [MOSS-TTS] Loading NanoTTSService (checkpoint={model_path}, device={device})...")
            cls._service = NanoTTSService(
                checkpoint_path=model_path,
                audio_tokenizer_path=tokenizer_path,
                device=device,
                dtype="auto",
            )
            print(f"  [MOSS-TTS] NanoTTSService ready.")
        return cls._service

    def _synth(self, text: str, voice: str, language: str,
               out_path: str, rate: str) -> float:
        """Core synthesis: resolve voice -> NanoTTSService.synthesize -> WAV->MP3 -> loudnorm."""
        import numpy as np
        import soundfile as sf

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)

        # Resolve voice -> preset name OR custom ref_audio
        voice_meta = get_moss_voice_meta(voice)
        is_preset = False
        custom_ref_audio = None
        custom_ref_text = None
        if not voice_meta:
            print(f"  [MOSS-TTS] Unknown voice '{voice}', falling back to {_DEFAULT_MALE}")
            voice = _DEFAULT_MALE
            is_preset = True
        elif voice_meta.get("is_preset"):
            is_preset = True
        else:
            custom_ref_audio = voice_meta.get("ref_audio", "")
            custom_ref_text = voice_meta.get("ref_text", "")
            if not custom_ref_audio:
                raise RuntimeError(
                    f"MOSS-TTS voice '{voice}' has no ref_audio configured. "
                    "Please set up reference audio in MOSS 音色管理 page."
                )

        service = self._get_service(
            self._model_path, self._tokenizer_path,
            self._device or "cpu", self._repo_dir)

        all_audio = []
        sr = 24000  # MOSS default output sample rate

        for sentence in _split_sentences(text):
            sentence = sentence.strip()
            if not sentence:
                continue
            try:
                tmp_wav = out_path.replace('.mp3', f'_tmp_{len(all_audio)}.wav')
                kwargs = {
                    "text": sentence,
                    "mode": "voice_clone",
                    "output_audio_path": tmp_wav,
                }
                if is_preset:
                    kwargs["voice"] = voice
                else:
                    kwargs["prompt_audio_path"] = custom_ref_audio
                    kwargs["prompt_text"] = custom_ref_text or None

                result = service.synthesize(**kwargs)
                audio, sr = sf.read(result["audio_path"])
                all_audio.append(audio)
                try:
                    os.remove(result["audio_path"])
                except OSError:
                    pass
            except Exception as e:
                print(f"    [MOSS-TTS skip] {sentence[:40]}: {e}")

        if not all_audio:
            raise RuntimeError(f"MOSS-TTS produced no audio for: {text[:50]}")

        final_audio = all_audio[0] if len(all_audio) == 1 else np.concatenate(all_audio)

        # Write WAV, convert to MP3 with atempo + downsample to 24kHz
        wav_path = out_path.replace('.mp3', '_tmp.wav')
        sf.write(wav_path, final_audio, sr)

        speed = _rate_to_speed(rate)
        af = f"atempo={speed:.4f}" if 0.5 <= speed <= 2.0 and speed != 1.0 else "anull"
        subprocess.run(
            ["ffmpeg", "-y", "-i", wav_path,
             "-af", af,
             "-c:a", "libmp3lame", "-b:a", "128k",
             "-ar", "24000", "-ac", "1", out_path],
            check=True, capture_output=True
        )
        os.remove(wav_path)

        self._loudnorm(out_path)
        return self.get_duration(out_path)

    def synth_english(self, text: str, voice: str, out_path: str,
                      rate: str = "+0%") -> float:
        """Synthesize English text with MOSS-TTS-Nano."""
        return self._synth(text, voice, "english", out_path, rate)

    def synth_chinese(self, text: str, voice: str, out_path: str,
                      rate: str = "-10%", max_retries: int = 5,
                      timeout: int = 30) -> float:
        """Synthesize Chinese text with MOSS-TTS-Nano."""
        return self._synth(text, voice, "chinese", out_path, rate)
