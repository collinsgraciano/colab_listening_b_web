"""MOSS-TTS-Nano engine for English + Chinese TTS.

0.1B voice cloning TTS model, CPU-first.
Uses NanoTTSService from moss_tts_nano_runtime.py (local repo install).

Env vars (set by pipeline.py / pipeline_service.py):
  MOSS_MODEL_PATH       — model checkpoint (H:/models/MOSS-TTS-Nano-Model)
  MOSS_TOKENIZER_PATH   — audio tokenizer (H:/models/MOSS-Audio-Tokenizer-Nano)
  MOSS_DEVICE           — torch device (default: cpu)
  MOSS_REPO_DIR         — repo dir containing moss_tts_nano_runtime.py (H:/models/MOSS-TTS-Nano)
  MOSS_TTS_TEMPERATURE  — audio sampling temperature, lower = more stable (default: 0.8)
  MOSS_TTS_RETRY        — per-sentence retry attempts with re-seed (default: 3)
  MOSS_TTS_GAP_MS       — inter-sentence silence in ms (default: 120)
"""
import json
import math
import os
import re
import subprocess
import sys
import zlib
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    except Exception:
        pass  # stdout may be redirected/captured (web/pytest) — reconfigure unavailable

_PARENT = str(Path(__file__).parent.resolve())
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from tts_engine import TTSEngine, _rate_to_speed

# ---------------------------------------------------------------------------
# Built-in voice presets (from moss_tts_nano_runtime._DEFAULT_VOICE_FILES)
# Filtered at import time to only include voices whose audio files exist.
# ---------------------------------------------------------------------------

_ALL_PRESET_VOICES = [
    {"name": "Junhao",  "file": "zh_1.wav",  "desc": "中文男声 A",   "gender": "male",   "lang": "zh"},
    {"name": "Zhiming", "file": "zh_2.wav",  "desc": "中文男声 B",   "gender": "male",   "lang": "zh"},
    {"name": "Weiguo",  "file": "zh_5.wav",  "desc": "中文男声 C",   "gender": "male",   "lang": "zh"},
    {"name": "Xiaoyu",  "file": "zh_3.wav",  "desc": "中文女声 A",   "gender": "female", "lang": "zh"},
    {"name": "Yuewen",  "file": "zh_4.wav",  "desc": "中文女声 B",   "gender": "female", "lang": "zh"},
    {"name": "Lingyu",  "file": "zh_6.wav",  "desc": "中文女声 C",   "gender": "female", "lang": "zh"},
    {"name": "Trump",   "file": "en_1.wav",  "desc": "Trump 参考音色", "gender": "male",   "lang": "en"},
    {"name": "Ava",     "file": "en_2.wav",  "desc": "英文女声 A",   "gender": "female", "lang": "en"},
    {"name": "Bella",   "file": "en_3.wav",  "desc": "英文女声 B",   "gender": "female", "lang": "en"},
    {"name": "Adam",    "file": "en_4.wav",  "desc": "英文男声 A",   "gender": "male",   "lang": "en"},
    {"name": "Nathan",  "file": "en_5.wav",  "desc": "英文男声 B",   "gender": "male",   "lang": "en"},
    {"name": "Sakura",  "file": "jp_1.mp3",  "desc": "日语女声 A",   "gender": "female", "lang": "ja"},
    {"name": "Yui",     "file": "jp_2.wav",  "desc": "日语女声 B",   "gender": "female", "lang": "ja"},
    {"name": "Aoi",     "file": "jp_3.wav",  "desc": "日语女声 C",   "gender": "female", "lang": "ja"},
    {"name": "Hina",    "file": "jp_4.wav",  "desc": "日语女声 D",   "gender": "female", "lang": "ja"},
    {"name": "Mei",     "file": "jp_5.wav",  "desc": "日语女声 E",   "gender": "female", "lang": "ja"},
]


def _get_preset_audio_dir() -> Path:
    """Resolve the assets/audio directory from MOSS_REPO_DIR env var."""
    repo_dir = os.environ.get("MOSS_REPO_DIR", r"H:\models\MOSS-TTS-Nano")
    return Path(repo_dir) / "assets" / "audio"


def _filter_available_presets() -> list[dict]:
    """Return only presets whose audio files exist on disk (one summary line)."""
    audio_dir = _get_preset_audio_dir()
    result = []
    missing = []
    for v in _ALL_PRESET_VOICES:
        if (audio_dir / v["file"]).exists():
            result.append({k: v[k] for k in ("name", "desc", "gender", "lang")})
        else:
            missing.append(v["name"])
    if missing:
        print(f"  [MOSS-TTS] {len(result)} preset voices available "
              f"({len(missing)} skipped, ref audio not installed)")
    return result


MOSS_PRESET_VOICES = _filter_available_presets()

_PRESET_NAMES = {s["name"] for s in MOSS_PRESET_VOICES}

# Default auto-assign by gender (built-in preset names)
_DEFAULT_MALE = "Adam"
_DEFAULT_FEMALE = "Ava"
_DEFAULT_HOST_FEMALE = "Bella"


# ---------------------------------------------------------------------------
# Sentence splitting / text normalization / audio post-processing
# ---------------------------------------------------------------------------

# Negative lookbehinds block splits right after common abbreviations
# ("a.m.", "p.m.", "e.g.", "U.S.", "Mr.", "Dr.", ...) so "7:30 a.m. on"
# stays one sentence. CJK punctuation splits without needing whitespace.
_ABBR_LOOKBEHIND = (
    r'(?<![A-Za-z]\.[A-Za-z]\.)'
    r'(?<!\bMr\.)'
    r'(?<!\bMrs\.)'
    r'(?<!\bMs\.)'
    r'(?<!\bDr\.)'
    r'(?<!\bProf\.)'
    r'(?<!\bSt\.)'
    r'(?<!\bvs\.)'
    r'(?<!\betc\.)'
    r'(?<!\bJr\.)'
    r'(?<!\bInc\.)'
)
_SENT_SPLIT_RE = re.compile(_ABBR_LOOKBEHIND + r'(?<=[.!?])\s+|(?<=[。！？；])\s*')

_CJK_CHAR_RE = re.compile(r'[\u3400-\u4dbf\u4e00-\u9fff]')


def _split_sentences_moss(text: str) -> list[str]:
    """Split text into sentences, handling English + Chinese punctuation.

    Abbreviation periods (a.m./Mr./...) do not trigger a split; fragments
    that start lowercase (unknown abbreviations) are merged back.
    """
    parts = [p for p in _SENT_SPLIT_RE.split(text.strip()) if p and p.strip()]
    merged: list[str] = []
    for part in parts:
        part = part.strip()
        if merged and part[:1].isascii() and part[:1].islower():
            # lowercase start = artificial split (e.g. "approx. three days")
            merged[-1] = merged[-1].rstrip() + " " + part
        else:
            merged.append(part)
    return merged


def _is_chinese_text(text: str) -> bool:
    """True when the text is predominantly Chinese."""
    cjk = len(_CJK_CHAR_RE.findall(text))
    letters = len(re.findall(r'[A-Za-z]', text))
    return cjk > letters


_normalize_fn = None
_normalize_tried = False


def _get_text_normalizer():
    """Import normalize_tts_text from the MOSS repo (self-contained script).

    Must be called after the repo dir is on sys.path (see _get_service).
    Returns None (falls back to raw text) if unavailable.
    """
    global _normalize_fn, _normalize_tried
    if not _normalize_tried:
        _normalize_tried = True
        try:
            from tts_robust_normalizer_single_script import normalize_tts_text
            _normalize_fn = normalize_tts_text
        except Exception as e:
            print(f"  [MOSS-TTS] Robust text normalizer unavailable, using raw text ({e})")
    return _normalize_fn


def _clean_sentence_audio(audio, sr: int):
    """Compress pathological pauses and apply fades for natural pacing.

    MOSS-TTS often emits 1-3s of near-silence between/after speech bursts.
    Frame-based (20ms) RMS detection: mid pauses > 0.6s shrink to 0.25s,
    leading silence to 30ms, trailing silence to 80ms. 10ms fades prevent
    clicks at concatenation joins.
    """
    import numpy as np
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if audio.size < int(0.3 * sr):
        return audio

    frame = max(1, int(0.02 * sr))  # 20ms frames
    n = audio.size // frame
    if n < 3:
        return audio
    frames = audio[:n * frame].reshape(n, frame)
    rms = np.sqrt((frames.astype(np.float64) ** 2).mean(axis=1))
    silent = rms < 0.008  # ≈ -42dB, above MOSS noise floor (-55dB+)

    # Locate silent runs (start, end) in frame indices
    runs = []
    start = None
    for i, s in enumerate(silent):
        if s and start is None:
            start = i
        elif not s and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, n))

    # Decide how much of each silent run to keep (in frames)
    def _keep_frames(a: int, b: int) -> int:
        dur_s = (b - a) * 0.02
        if a == 0:
            return max(0, int(0.03 / 0.02))       # leading -> 30ms
        if b >= n:
            return max(0, int(0.08 / 0.02))       # trailing -> 80ms
        if dur_s > 0.6:
            return max(0, int(0.25 / 0.02))       # long mid pause -> 250ms
        return b - a                               # natural short pause

    keep_parts = []
    cur = 0
    for a, b in runs:
        keep_parts.append((cur, a))
        keep_parts.append((a, min(b, a + _keep_frames(a, b))))
        cur = b
    keep_parts.append((cur, n))

    kept = [audio[a * frame: b * frame] for a, b in keep_parts if b > a]
    if not kept:
        return audio
    cleaned = np.concatenate(kept)
    tail = audio[n * frame:]
    if tail.size:
        cleaned = np.concatenate([cleaned, tail])

    # 25ms fade-in suppresses residual onset clicks/bursts; 10ms fade-out
    fade_in = min(int(0.025 * sr), cleaned.size // 2)
    fade_out = min(int(0.01 * sr), cleaned.size // 2)
    cleaned = cleaned.copy()
    if fade_in > 0:
        cleaned[:fade_in] *= np.linspace(0.0, 1.0, fade_in, dtype=np.float32)
    if fade_out > 0:
        cleaned[-fade_out:] *= np.linspace(1.0, 0.0, fade_out, dtype=np.float32)
    return cleaned


def _has_leading_burst(audio, sr: int) -> bool:
    """Detect an unnatural full-energy onset in the first ~120ms.

    MOSS 冷启动后第一次推理 / 个别采样劣化会在句首产生爆音或含混，听感为
    "杂音"：第 2-4 帧（20-80ms）就冲到全句峰值附近、完全没有自然起音包络
    （正常起音前 40ms 至少比峰值低 ~35%）。所有历史劣化样本（4/5 的
    welcome 首句）均呈现 [≤12%, >75%, >75%, >75%] 形态；正常样本无一命中。
    """
    import numpy as np
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    win = int(0.02 * sr)
    n = audio.size // win
    if n < 10:
        return False
    frames = audio[:n * win].reshape(n, win).astype(np.float64)
    rms = np.sqrt((frames ** 2).mean(axis=1))
    p95 = float(np.percentile(rms, 95))
    if p95 <= 1e-6:
        return False
    head = rms[:4] / p95
    # 帧1(0-20ms)极低 + 帧2-4(20-80ms)直接冲到 75% 峰值以上 = 无起音瞬态
    return bool(head[0] <= 0.15 and head[1] >= 0.75
                and head[2] >= 0.75 and head[3] >= 0.75)


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
        "default_male_zh": "Junhao",
        "default_female_zh": "Xiaoyu",
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


def get_moss_zh_default(gender: str) -> str:
    """Default Chinese preset voice for Chinese lines, by gender.

    Chinese lines spoken with an English reference voice sound accented —
    default to native Chinese presets (Junhao male / Xiaoyu female).
    """
    config = _load_voice_config()
    if str(gender).lower() == "female":
        return config.get("default_female_zh", "Xiaoyu")
    return config.get("default_male_zh", "Junhao")


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
                 tokenizer_path: str = "", repo_dir: str = "",
                 temperature: float | None = None, retry: int | None = None):
        self._model_path = model_path
        self._tokenizer_path = tokenizer_path
        self._repo_dir = repo_dir
        MossTTSEngine._device = device
        # Generation quality knobs (env-overridable; set by pipeline_service / web UI)
        if temperature is None:
            temperature = os.environ.get("MOSS_TTS_TEMPERATURE", "0.8")
        try:
            self._temperature = max(0.0, min(1.5, float(temperature)))
        except (TypeError, ValueError):
            self._temperature = 0.8
        if retry is None:
            retry = os.environ.get("MOSS_TTS_RETRY", "3")
        try:
            self._retry = max(1, int(retry))
        except (TypeError, ValueError):
            self._retry = 3
        try:
            self._gap_ms = max(0, int(os.environ.get("MOSS_TTS_GAP_MS", "120")))
        except (TypeError, ValueError):
            self._gap_ms = 120

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
            cls._warm_up()
            print(f"  [MOSS-TTS] NanoTTSService ready.")
        return cls._service

    @classmethod
    def _warm_up(cls):
        """加载后立即做一次丢弃式预热合成。

        冷启动后第一次推理容易在句首产生爆音/含混（历史 cutout 运行中
        welcome.mp3 恒为首个合成文件且多次出现起始满能量包络）。预热消耗
        一次推理，使真实首句不再是"第一次生成"。失败不阻断主流程
        （真实调用仍有 retry + 起始爆音校验兜底）。
        """
        import tempfile
        import time as _time
        voice = next((v for v in (_DEFAULT_MALE, _DEFAULT_FEMALE, "Bella")
                      if v in _PRESET_NAMES),
                     next(iter(sorted(_PRESET_NAMES)), ""))
        if not voice:
            return
        tmp = os.path.join(tempfile.gettempdir(), f"moss_warmup_{os.getpid()}.wav")
        t0 = _time.time()
        try:
            cls._service.synthesize(
                text="Hello.",
                mode="voice_clone",
                voice=voice,
                output_audio_path=tmp,
                max_new_frames=100,
            )
            print(f"  [MOSS-TTS] Warm-up done in {_time.time() - t0:.1f}s "
                  f"(voice={voice}, cold-start first inference consumed).")
        except Exception as e:
            print(f"  [MOSS-TTS] Warm-up skipped ({type(e).__name__}: {str(e)[:80]})")
        finally:
            try:
                os.remove(tmp)
            except OSError:
                pass

    def _synth(self, text: str, voice: str, language: str,
               out_path: str, rate: str) -> float:
        """Core synthesis: resolve voice -> per-sentence synthesize (retry +
        validation) -> trim/fade/concat with gaps -> WAV->MP3 -> loudnorm."""
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

        normalizer = _get_text_normalizer()

        pieces = []
        sr = 24000  # MOSS default output sample rate
        for sentence in _split_sentences_moss(text):
            sentence = sentence.strip()
            if not sentence:
                continue
            # Robust normalization: numbers, symbols, ellipsis, quotes...
            if normalizer is not None:
                try:
                    normalized = str(normalizer(sentence) or "").strip()
                    if normalized:
                        sentence = normalized
                except Exception:
                    pass
            base_seed = zlib.crc32(f"{voice}|{sentence}".encode("utf-8"))
            audio, sr = self._synthesize_sentence(
                service, sentence, is_preset, voice,
                custom_ref_audio, custom_ref_text, language,
                base_seed, out_path, len(pieces))
            if pieces and self._gap_ms > 0:
                pieces.append(np.zeros(int(self._gap_ms / 1000.0 * sr), dtype=np.float32))
            pieces.append(audio)

        if not pieces:
            raise RuntimeError(f"MOSS-TTS produced no audio for: {text[:50]}")

        final_audio = pieces[0] if len(pieces) == 1 else np.concatenate(pieces)

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

    def _synthesize_sentence(self, service, sentence: str, is_preset: bool,
                             voice: str, custom_ref_audio, custom_ref_text,
                             language: str, base_seed: int,
                             out_path: str, idx: int):
        """Synthesize one sentence with retry + output validation.

        Each retry uses a different seed (re-sampling). Raises RuntimeError
        when every attempt fails — sentences are never silently dropped.
        """
        import numpy as np
        import soundfile as sf

        # Duration estimate at natural pace: EN ~2.4 words/s, ZH ~4.0 chars/s
        if language == "chinese" or _is_chinese_text(sentence):
            est_dur = max(len(_CJK_CHAR_RE.findall(sentence)) / 4.0, 0.5)
        else:
            est_dur = max(len(sentence.split()) / 2.4, 0.5)
        # One frame = 80ms (48kHz / 3840 downsample); scale to avoid truncation
        max_frames = max(375, math.ceil(est_dur / 0.08) + 75)

        last_err = None
        for attempt in range(1, self._retry + 1):
            tmp_wav = out_path.replace('.mp3', f'_tmp_{idx}_{attempt}.wav')
            seed = (base_seed + (attempt - 1) * 7919) & 0x7FFFFFFF
            try:
                kwargs = {
                    "text": sentence,
                    "mode": "voice_clone",
                    "output_audio_path": tmp_wav,
                    "seed": seed,
                    "max_new_frames": max_frames,
                    "audio_temperature": self._temperature,
                    "audio_top_p": 0.95,
                    "audio_top_k": 25,
                    "audio_repetition_penalty": 1.2,
                }
                if is_preset:
                    kwargs["voice"] = voice
                else:
                    kwargs["prompt_audio_path"] = custom_ref_audio
                    kwargs["prompt_text"] = custom_ref_text or None

                result = service.synthesize(**kwargs)
                audio, sr = sf.read(result["audio_path"], dtype="float32")
                try:
                    os.remove(result["audio_path"])
                except OSError:
                    pass
            except Exception as e:
                last_err = e
                print(f"    [MOSS-TTS retry {attempt}/{self._retry}] synthesize failed: {str(e)[:80]}")
                continue

            # Compress pathological pauses FIRST so validation sees real speech
            audio = _clean_sentence_audio(audio, sr)

            # Validation 1: near-silent / empty output
            if audio.size == 0 or float(np.mean(np.abs(audio))) < 0.002:
                last_err = RuntimeError("near-silent audio output")
                print(f"    [MOSS-TTS retry {attempt}/{self._retry}] near-silent audio: {sentence[:40]}")
                continue

            # Validation 1b: leading non-speech burst (cold-start artifact).
            # Last attempt accepts with a warning — never worse than before.
            if _has_leading_burst(audio, sr):
                if attempt < self._retry:
                    last_err = RuntimeError("leading burst artifact")
                    print(f"    [MOSS-TTS retry {attempt}/{self._retry}] leading burst: {sentence[:40]}")
                    continue
                print(f"    [MOSS-TTS WARN] leading burst persisted after "
                      f"{self._retry} attempts: {sentence[:40]}")

            # Validation 2: implausible duration (truncation / babbling)
            actual = audio.size / float(sr)
            if est_dur >= 0.8:
                ratio = actual / est_dur
                if ratio < 0.4 or ratio > 2.3:
                    last_err = RuntimeError(
                        f"implausible duration {actual:.1f}s (est {est_dur:.1f}s)")
                    if attempt < self._retry:
                        print(f"    [MOSS-TTS retry {attempt}/{self._retry}] "
                              f"duration {actual:.1f}s vs est {est_dur:.1f}s: {sentence[:40]}")
                        continue
                    print(f"    [MOSS-TTS WARN] accepted borderline duration "
                          f"{actual:.1f}s (est {est_dur:.1f}s) after {self._retry} attempts: {sentence[:40]}")
            return audio, sr

        raise RuntimeError(
            f"MOSS-TTS failed after {self._retry} attempts: {str(last_err)[:120]} "
            f"(sentence: {sentence[:40]})")

    def synth_english(self, text: str, voice: str, out_path: str,
                      rate: str = "+0%") -> float:
        """Synthesize English text with MOSS-TTS-Nano."""
        return self._synth(text, voice, "english", out_path, rate)

    def synth_chinese(self, text: str, voice: str, out_path: str,
                      rate: str = "-10%", max_retries: int = 5,
                      timeout: int = 30) -> float:
        """Synthesize Chinese text with MOSS-TTS-Nano."""
        return self._synth(text, voice, "chinese", out_path, rate)
