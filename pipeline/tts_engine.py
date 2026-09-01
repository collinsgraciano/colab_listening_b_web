"""Standalone TTS engine for English listening videos.

- English: Kokoro TTS (American English, pitch-preserving speed control)
- Chinese: Kokoro TTS (Mandarin Chinese, zf_xiaoxiao.pt female / zf_xiaobei.pt male)
- All audio normalized with FFmpeg loudnorm

No dependency on any external project.
"""
import os
import sys
import re
import subprocess
import json
from pathlib import Path
from dataclasses import dataclass

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    except Exception:
        pass  # stdout may be redirected/captured (web/pytest) — reconfigure unavailable

# HF_ENDPOINT: Windows local uses hf-mirror (HuggingFace blocked in China),
# Colab/Linux uses direct huggingface.co. Allow override via env var.
if "HF_ENDPOINT" not in os.environ:
    if sys.platform == "win32":
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    else:
        os.environ['HF_ENDPOINT'] = 'https://huggingface.co'

# If Kokoro model is already cached, use offline mode to avoid slow/failed
# network HEAD requests (hf-mirror.com can be unreachable).
_kokoro_cache = Path(os.path.expanduser(
    "~/.cache/huggingface/hub/models--hexgrad--Kokoro-82M/snapshots"))
if _kokoro_cache.exists() and any(_kokoro_cache.rglob("kokoro-v1_0.pth")):
    os.environ["HF_HUB_OFFLINE"] = "1"


@dataclass
class TTSResult:
    audio_path: str
    duration_sec: float


# Known words that cause Kokoro to silently produce no audio.
# Map each to a phonetically similar spelling that Kokoro can handle.
_PHONETIC_FIXES = {
    "Mia": "Maya",
    "mia": "Maya",
    "Wi-Fi": "WiFi",
    "wi-fi": "WiFi",
    "Wi-fi": "WiFi",
}


def _apply_phonetic_fixes(text: str) -> str:
    """Replace known problem words with Kokoro-compatible phonetic spellings."""
    for bad, good in _PHONETIC_FIXES.items():
        text = text.replace(bad, good)
    return text


def _rate_to_speed(rate: str) -> float:
    """Convert rate string to Kokoro speed float.

    '+0%' = 1.0, '-15%' = 0.85, '+10%' = 1.1
    """
    if not rate or rate == "+0%":
        return 1.0
    try:
        pct = int(rate.replace('%', '').replace('+', ''))
        return 1.0 + pct / 100.0
    except (ValueError, TypeError):
        return 1.0


# Natural inter-sentence pause inserted between Kokoro sentence chunks.
# Concatenating split sentences back-to-back sounds rushed otherwise; 0.12s
# keeps narration pacing natural without double-gapping (chunks carry their
# own trailing silence). Internal prosody constant, not a config option.
_SENTENCE_GAP_SEC = 0.12

# Negative lookbehinds block splits right after common abbreviations
# ("Mr.", "Dr.", "a.m."-style ...) so "I saw Dr. Smith yesterday." stays
# one sentence. Ported from moss_tts_engine (battle-tested there).
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
_SENT_SPLIT_RE = re.compile(_ABBR_LOOKBEHIND + r'(?<=[.!?])\s+')


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences for Kokoro processing.

    Abbreviation periods (Dr./Mr./a.m./...) do not trigger a split;
    fragments that start lowercase (unknown abbreviations) are merged back.
    """
    parts = _SENT_SPLIT_RE.split(text.strip())
    merged: list[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if merged and part[:1].isascii() and part[:1].islower():
            # lowercase start = artificial split (e.g. "approx. three days")
            merged[-1] = merged[-1].rstrip() + " " + part
        else:
            merged.append(part)
    return merged


def _clean_text_for_kokoro(text: str) -> str:
    """Remove characters that cause Kokoro/misaki phonemizer issues.

    ONLY for English pipeline (lang_code='a'). Chinese pipeline (lang_code='z')
    must NOT use this function as it strips non-ASCII characters.
    """
    text = text.replace('-', ' ').replace('\u2014', ' ').replace('\u2013', ' ')
    text = re.sub(r'[^\x00-\x7F]', ' ', text)
    text = re.sub(r'[^\w\s.,!?\']', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


_UNITS = ["zero", "one", "two", "three", "four", "five", "six", "seven",
          "eight", "nine", "ten", "eleven", "twelve"]
_TENS = {1: "ten", 2: "twenty", 3: "thirty", 4: "forty", 5: "fifty",
         6: "sixty", 7: "seventy", 8: "eighty", 9: "ninety"}


def _two_digit_words(n: int) -> str:
    """10-99 → English words (26 → 'twenty six'); assumes 10 <= n <= 99."""
    t, u = divmod(n, 10)
    return _TENS[t] if u == 0 else f"{_TENS[t]} {_UNITS[u]}"


def _normalize_tts_text_en(text: str) -> str:
    """Normalize numbers/symbols Kokoro's G2P reads unreliably.

    Runs BEFORE _clean_text_for_kokoro, which strips $ % : — converting
    '$20' to '20' loses 'dollars' and '10%' loses 'percent'. Plain integers
    are kept as digits (misaki reads those fine); only symbol semantics are
    added: currency, percent, clock times, and 4-digit years.
    """
    # Currency with cents: $19.99 → 19 dollars and 99 cents
    def _money_cents(m: re.Match) -> str:
        d, c = m.group(1), m.group(2)
        dollars = "dollar" if d == "1" else "dollars"
        cents = "cent" if c.lstrip("0") == "1" else "cents"
        return f"{d} {dollars} and {c} {cents}"

    text = re.sub(r'\$(\d+)\.(\d{2})(?!\d)', _money_cents, text)
    # Plain currency: $20 → 20 dollars
    text = re.sub(
        r'\$(\d+)(?!\d|\.\d)',
        lambda m: f"{m.group(1)} dollar{'s' if m.group(1) != '1' else ''}",
        text)
    # Clock times: 8:30 → eight 30 (read 'eight thirty'), 8:05 → eight oh 5,
    # 9:00 → nine o'clock. Hours 13-23 keep digits (read fine as-is).
    def _clock(m: re.Match) -> str:
        h, mm = int(m.group(1)), m.group(2)
        hour = _UNITS[h] if 0 <= h <= 12 else str(h)
        if mm == "00":
            return f"{hour} o'clock"
        if 1 <= int(mm) <= 9:
            return f"{hour} oh {_UNITS[int(mm)]}"
        return f"{hour} {mm}"

    text = re.sub(r'(?<!\d)(\d{1,2}):(\d{2})(?!\d)', _clock, text)
    # 4-digit years: 2026 → twenty twenty six, 1999 → nineteen ninety nine.
    # 2000-2009 stay digits (read 'two thousand ...' which is fine).
    def _year(m: re.Match) -> str:
        s = m.group(0)
        first = {"19": "nineteen", "20": "twenty"}.get(s[:2])
        if not first or int(s[2:]) < 10:
            return s
        return f"{first} {_two_digit_words(int(s[2:]))}"

    text = re.sub(r'(?<!\d)(19|20)\d{2}(?!\d)', _year, text)
    # Percent: 50% → 50 percent
    text = re.sub(r'(\d+(?:\.\d+)?)\s*%', r'\1 percent', text)
    return text


# ---------------------------------------------------------------------------
# Voice file resolution (download from HF mirror if not cached)
# ---------------------------------------------------------------------------

def _find_voice_in_cache(filename: str) -> str | None:
    """Search HF cache for a Kokoro voice .pt file by filename."""
    cache_dir = Path(os.path.expanduser("~/.cache/huggingface/hub"))
    if not cache_dir.exists():
        return None
    for root, _dirs, files in os.walk(cache_dir):
        if filename in files:
            path = os.path.join(root, filename)
            if os.path.getsize(path) > 1000:
                return path
    return None


def _download_voice(filename: str) -> str:
    """Download a Kokoro voice .pt file from HF mirror into cache."""
    cache_voices = Path(os.path.expanduser(
        "~/.cache/huggingface/hub/models--hexgrad--Kokoro-82M/voices"))
    cache_voices.mkdir(parents=True, exist_ok=True)
    dest = str(cache_voices / filename)
    if os.path.exists(dest) and os.path.getsize(dest) > 1000:
        return dest
    urls = [
        f"https://www.modelscope.cn/models/AI-ModelScope/Kokoro-82M/resolve/master/voices/{filename}",
        f"https://hf-mirror.com/hexgrad/Kokoro-82M/resolve/main/voices/{filename}",
        f"https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/voices/{filename}",
    ]
    for url in urls:
        print(f"  [Download] {url}")
        try:
            import urllib.request
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = resp.read()
            if len(data) > 1000:
                Path(dest).write_bytes(data)
                print(f"  [OK] Downloaded voice: {filename} ({len(data)//1024}KB)")
                return dest
        except Exception as e:
            print(f"  [Warn] Download failed: {str(e)[:80]}")
    raise FileNotFoundError(
        f"Voice '{filename}' could not be downloaded. "
        f"Please manually place the .pt file at: {dest}")


def _resolve_voice(voice: str) -> str:
    """Resolve a voice name/path to a valid .pt file, downloading if needed."""
    if os.path.exists(voice) and os.path.getsize(voice) > 1000:
        return voice
    filename = os.path.basename(voice)
    cached = _find_voice_in_cache(filename)
    if cached:
        return cached
    return _download_voice(filename)


class TTSEngine:
    """Standalone TTS engine: Kokoro (English + Chinese) + loudnorm."""

    _kokoro_pipeline = None
    _kokoro_zh_pipeline = None

    @classmethod
    def _get_kokoro(cls):
        """Lazy-init and cache the English KPipeline (lang_code='a').
        Raises RuntimeError if model fails to load.
        """
        if cls._kokoro_pipeline is None:
            from kokoro import KPipeline
            try:
                cls._kokoro_pipeline = KPipeline(lang_code='a')
            except Exception as e:
                raise RuntimeError(f"Kokoro English model failed to load: {e}") from e
        return cls._kokoro_pipeline

    @classmethod
    def _get_kokoro_zh(cls):
        """Lazy-init and cache the Mandarin Chinese KPipeline (lang_code='z').
        Raises RuntimeError if model fails to load.
        """
        if cls._kokoro_zh_pipeline is None:
            from kokoro import KPipeline
            try:
                cls._kokoro_zh_pipeline = KPipeline(lang_code='z')
            except Exception as e:
                raise RuntimeError(f"Kokoro Chinese model failed to load: {e}") from e
        return cls._kokoro_zh_pipeline

    @staticmethod
    def get_duration(audio_path: str) -> float:
        """Get audio duration in seconds via ffprobe."""
        from media_utils import get_duration as _get_dur
        return _get_dur(audio_path)

    @staticmethod
    def _loudnorm(input_path: str, output_path: str | None = None):
        """Apply loudnorm + volume normalization to target -16 dB RMS.

        Two-step: (1) loudnorm linear pass, (2) if still too quiet, apply
        volume boost via volumedetect to hit target RMS.
        """
        if output_path is None:
            output_path = input_path.replace(".mp3", "_norm.mp3")

        # Step 1: loudnorm
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", input_path,
             "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
             "-c:a", "libmp3lame", "-b:a", "128k", output_path],
            capture_output=True, timeout=30,
        )
        if result.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
            os.replace(output_path, input_path)
        else:
            # Fallback: simple volume boost (+6dB)
            fallback = input_path.replace(".mp3", "_vol.mp3")
            subprocess.run(
                ["ffmpeg", "-y", "-i", input_path,
                 "-af", "volume=6dB",
                 "-c:a", "libmp3lame", "-b:a", "128k", fallback],
                capture_output=True, timeout=30,
            )
            if os.path.exists(fallback) and os.path.getsize(fallback) > 1000:
                os.replace(fallback, input_path)

        # Step 2: Check if still too quiet, apply extra boost
        try:
            detect = subprocess.run(
                ["ffmpeg", "-i", input_path, "-af", "volumedetect",
                 "-f", "null", "-"],
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=15)
            stderr = detect.stderr
            import re
            m = re.search(r"mean_volume:\s*(-?\d+\.?\d*)\s*dB", stderr)
            if m:
                mean_db = float(m.group(1))
                if mean_db < -20:
                    # Need extra boost: target -16 dB
                    boost = -16 - mean_db
                    boosted = input_path.replace(".mp3", "_boost.mp3")
                    subprocess.run(
                        ["ffmpeg", "-y", "-i", input_path,
                         "-af", f"volume={boost}dB,alimiter=limit=0.95",
                         "-c:a", "libmp3lame", "-b:a", "128k", boosted],
                        capture_output=True, timeout=30)
                    if os.path.exists(boosted) and os.path.getsize(boosted) > 1000:
                        os.replace(boosted, input_path)
                        print(f"    [loudnorm] Extra boost: +{boost:.1f}dB (was {mean_db:.1f}dB)")
        except Exception:
            pass

    @staticmethod
    def _normalize_wav(wav_path: str):
        """Loudness-normalize a WAV in place (PCM domain, pre-MP3-encode).

        Same targets/fallbacks as _loudnorm but applied before the single
        lossy encode, so each Kokoro clip goes through MP3 encoding exactly
        once (was: MP3 → loudnorm re-encode → optional boost re-encode).
        """
        tmp = wav_path.replace(".wav", "_norm.wav")
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", wav_path,
             "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
             "-c:a", "pcm_s16le", "-ar", "24000", tmp],
            capture_output=True, timeout=30,
        )
        if result.returncode == 0 and os.path.exists(tmp) and os.path.getsize(tmp) > 1000:
            os.replace(tmp, wav_path)
        else:
            # Fallback: simple volume boost (+6dB)
            fallback = wav_path.replace(".wav", "_vol.wav")
            subprocess.run(
                ["ffmpeg", "-y", "-i", wav_path,
                 "-af", "volume=6dB",
                 "-c:a", "pcm_s16le", "-ar", "24000", fallback],
                capture_output=True, timeout=30,
            )
            if os.path.exists(fallback) and os.path.getsize(fallback) > 1000:
                os.replace(fallback, wav_path)

        # Boost again if still too quiet (target ~-16 dB), same as _loudnorm
        try:
            detect = subprocess.run(
                ["ffmpeg", "-i", wav_path, "-af", "volumedetect",
                 "-f", "null", "-"],
                capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=15)
            m = re.search(r"mean_volume:\s*(-?\d+\.?\d*)\s*dB", detect.stderr)
            if m:
                mean_db = float(m.group(1))
                if mean_db < -20:
                    boost = -16 - mean_db
                    boosted = wav_path.replace(".wav", "_boost.wav")
                    subprocess.run(
                        ["ffmpeg", "-y", "-i", wav_path,
                         "-af", f"volume={boost}dB,alimiter=limit=0.95",
                         "-c:a", "pcm_s16le", "-ar", "24000", boosted],
                        capture_output=True, timeout=30)
                    if os.path.exists(boosted) and os.path.getsize(boosted) > 1000:
                        os.replace(boosted, wav_path)
                        print(f"    [loudnorm] Extra boost: +{boost:.1f}dB (was {mean_db:.1f}dB)")
        except Exception:
            pass

    @staticmethod
    def _kokoro_synth_chunk(pipeline, text: str, voice: str,
                            speed: float) -> list:
        """Try synthesizing text with Kokoro, progressively splitting on failure.

        Kokoro's misaki phonemizer can crash with NoneType+str on certain
        words or phrases. When that happens, split the text into smaller
        chunks (by comma, then by word groups) so problematic segments are
        isolated rather than losing the entire sentence.
        """
        def _try(chunk: str) -> list:
            chunk = chunk.strip()
            if not chunk:
                return []
            try:
                parts = []
                for _gs, _ps, audio in pipeline(chunk, voice=voice, speed=speed):
                    if audio is not None and len(audio) > 0:
                        parts.append(audio)
                return parts
            except Exception:
                return []

        # Attempt 1: full text
        audio = _try(text)
        if audio:
            return audio

        # Attempt 2: split by comma
        comma_parts = [p.strip() for p in text.split(',') if p.strip()]
        if len(comma_parts) > 1:
            audio = []
            for part in comma_parts:
                audio.extend(_try(part))
            if audio:
                print(f"    [Kokoro] Recovered via comma-split: {text[:50]}")
                return audio

        # Attempt 3: word groups of 3
        words = text.split()
        if len(words) > 1:
            audio = []
            for i in range(0, len(words), 3):
                chunk = ' '.join(words[i:i + 3])
                audio.extend(_try(chunk))
            if audio:
                print(f"    [Kokoro] Recovered via word-split: {text[:50]}")
                return audio

        print(f"    [Kokoro skip] Could not synthesize: {text[:60]}")
        return []

    def synth_english(self, text: str, voice: str, out_path: str,
                      rate: str = "+0%") -> float:
        """Synthesize English text with Kokoro TTS. Returns duration in seconds.

        Args:
            text: English text to speak
            voice: Kokoro voice ID (af_sarah, am_adam, af_sky, etc.) or local .pt path
            out_path: output MP3 file path
            rate: speed adjustment ('+0%'=normal, '-15%'=slow, '+10%'=fast)
        """
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        speed = _rate_to_speed(rate)
        pipeline = self._get_kokoro()
        # Resolve voice name to local .pt path and pass the path directly:
        # hf_hub_download only finds voices in the standard HF snapshot layout,
        # so voices downloaded to other locations (or any offline lookup) would
        # fail with network retries swallowed as empty audio. Path mode
        # (load_single_voice supports '*.pt') bypasses that entirely — same
        # pattern as _synth_chinese_kokoro below.
        if not voice.endswith(".pt"):
            voice = _resolve_voice(f"{voice}.pt")

        import soundfile as sf
        import numpy as np
        all_audio = []

        for sentence in _split_sentences(text):
            sentence = sentence.strip()
            if not sentence:
                continue
            # Apply phonetic fixes FIRST (before cleaning removes hyphens),
            # then normalize numbers/symbols, then strip problem characters.
            fixed = _apply_phonetic_fixes(sentence)
            normalized = _normalize_tts_text_en(fixed)
            cleaned = _clean_text_for_kokoro(normalized)
            parts = self._kokoro_synth_chunk(pipeline, cleaned, voice, speed)
            if not parts:
                continue
            # Natural inter-sentence pause, only between sentences that
            # actually produced audio (single-sentence lines unaffected)
            if all_audio and _SENTENCE_GAP_SEC > 0:
                all_audio.append(
                    np.zeros(int(24000 * _SENTENCE_GAP_SEC), dtype=np.float32))
            all_audio.extend(parts)

        if not all_audio:
            raise RuntimeError(f"Kokoro produced no audio for: {text[:50]}")

        final_audio = all_audio[0] if len(all_audio) == 1 else np.concatenate(all_audio)

        # Write WAV (24kHz) → loudness-normalize in PCM domain → single MP3
        # encode (loudnorm used to re-encode the MP3, up to 3 lossy passes)
        wav_path = out_path.replace('.mp3', '_tmp.wav')
        sf.write(wav_path, final_audio, 24000)
        self._normalize_wav(wav_path)
        subprocess.run(
            ["ffmpeg", "-y", "-i", wav_path, "-c:a", "libmp3lame", "-b:a", "128k",
             "-ar", "24000", "-ac", "1", out_path],
            check=True, capture_output=True
        )
        os.remove(wav_path)

        return self.get_duration(out_path)

    def synth_chinese(self, text: str, voice: str, out_path: str,
                      rate: str = "-10%", max_retries: int = 5, timeout: int = 30) -> float:
        """Synthesize Chinese text. Primary: edge-tts, Fallback: Kokoro.

        Args:
            text: Chinese text to speak
            voice: edge-tts voice ID (e.g. zh-CN-XiaoxiaoNeural, zh-CN-YunxiNeural)
            out_path: output MP3 file path
            rate: speed adjustment ('-10%' = slightly slow)
            max_retries: max retry attempts for edge-tts (default 5)
            timeout: timeout per attempt in seconds (default 30)
        """
        import asyncio
        import edge_tts
        from edge_tts.exceptions import NoAudioReceived

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        edge_voice = voice if voice else "zh-CN-XiaoxiaoNeural"

        async def _gen_edge():
            comm = edge_tts.Communicate(text, edge_voice, rate=rate)
            await asyncio.wait_for(comm.save(out_path), timeout=timeout)

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        last_error = None
        for attempt in range(max_retries):
            try:
                if loop.is_running():
                    asyncio.run_coroutine_threadsafe(_gen_edge(), loop).result(timeout=timeout + 5)
                else:
                    loop.run_until_complete(_gen_edge())
                if os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
                    # Success — apply loudnorm and return
                    self._loudnorm(out_path)
                    return self.get_duration(out_path)
                last_error = f"Empty audio (attempt {attempt+1})"
            except Exception as e:
                last_error = e
                print(f"    [edge-tts retry {attempt+1}/{max_retries}] {edge_voice} failed: {str(e)[:60]}")
                if attempt < max_retries - 1:
                    import time
                    time.sleep(2)
                continue

        # All edge-tts retries failed — fallback to Kokoro
        print(f"    [edge-tts] All {max_retries} retries failed, falling back to Kokoro zf_xiaoxiao.pt")
        try:
            return self._synth_chinese_kokoro(text, out_path, rate)
        except Exception as e:
            raise RuntimeError(f"Both edge-tts and Kokoro failed for Chinese TTS: {last_error} / {e}")

    def _synth_chinese_kokoro(self, text: str, out_path: str, rate: str = "-10%") -> float:
        """Kokoro Chinese TTS fallback. Uses zf_xiaoxiao.pt."""
        speed = _rate_to_speed(rate)
        pipeline = self._get_kokoro_zh()
        voice_path = _resolve_voice("zf_xiaoxiao.pt")

        import soundfile as sf
        import numpy as np
        all_audio = []

        try:
            for gs, ps, audio in pipeline(text, voice=voice_path, speed=speed):
                if audio is not None and len(audio) > 0:
                    all_audio.append(audio)
        except Exception as e:
            print(f"    [Kokoro zh skip] {str(e)[:80]}")

        if not all_audio:
            raise RuntimeError(f"Kokoro produced no Chinese audio for: {text[:50]}")

        final_audio = all_audio[0] if len(all_audio) == 1 else np.concatenate(all_audio)

        wav_path = out_path.replace('.mp3', '_tmp.wav')
        sf.write(wav_path, final_audio, 24000)
        self._normalize_wav(wav_path)
        subprocess.run(
            ["ffmpeg", "-y", "-i", wav_path, "-c:a", "libmp3lame", "-b:a", "128k",
             "-ar", "24000", "-ac", "1", out_path],
            check=True, capture_output=True
        )
        os.remove(wav_path)
        return self.get_duration(out_path)


# Voice configuration helpers

# Kokoro-82M 官方英文音色（af/am=美式，bf/bm=英式）。
# 中文台词走 edge-tts（zh_voice 绑定 / get_zh_voice），故中文音色不在此列
# （英文 KPipeline 无法识别 zf_/zm_ 音色名）。
KOKORO_VOICES = [
    {"name": "af_bella", "gender": "female", "desc": "美式·热情女声"},
    {"name": "af_heart", "gender": "female", "desc": "美式·温暖女声"},
    {"name": "af_nicole", "gender": "female", "desc": "美式·轻柔女声"},
    {"name": "af_sarah", "gender": "female", "desc": "美式·清晰女声"},
    {"name": "af_sky", "gender": "female", "desc": "美式·知性女声"},
    {"name": "am_adam", "gender": "male", "desc": "美式·沉稳男声"},
    {"name": "am_michael", "gender": "male", "desc": "美式·浑厚男声"},
    {"name": "am_fenrir", "gender": "male", "desc": "美式·低沉男声"},
    {"name": "am_puck", "gender": "male", "desc": "美式·活泼男声"},
    {"name": "bf_alice", "gender": "female", "desc": "英式·优雅女声"},
    {"name": "bf_emma", "gender": "female", "desc": "英式·柔和女声"},
    {"name": "bf_isabella", "gender": "female", "desc": "英式·清亮女声"},
    {"name": "bf_lily", "gender": "female", "desc": "英式·端庄女声"},
    {"name": "bm_fable", "gender": "male", "desc": "英式·叙事男声"},
    {"name": "bm_george", "gender": "male", "desc": "英式·醇厚男声"},
    {"name": "bm_lewis", "gender": "male", "desc": "英式·干练男声"},
    {"name": "af_alloy", "gender": "female", "desc": "美式·中性女声"},
    {"name": "af_aoede", "gender": "female", "desc": "美式·柔美女声"},
    {"name": "af_jessica", "gender": "female", "desc": "美式·活泼女声"},
    {"name": "af_kore", "gender": "female", "desc": "美式·沉稳女声"},
    {"name": "af_nova", "gender": "female", "desc": "美式·清新女声"},
    {"name": "af_river", "gender": "female", "desc": "美式·自然女声"},
    {"name": "am_echo", "gender": "male", "desc": "美式·清亮男声"},
    {"name": "am_eric", "gender": "male", "desc": "美式·随和男声"},
    {"name": "am_liam", "gender": "male", "desc": "美式·年轻男声"},
    {"name": "am_onyx", "gender": "male", "desc": "美式·深沉男声"},
    {"name": "am_santa", "gender": "male", "desc": "美式·老者男声"},
    {"name": "bm_daniel", "gender": "male", "desc": "英式·沉稳男声"},
]


def get_all_kokoro_voices() -> list[dict]:
    """All Kokoro English voices with a cached flag (.pt in HF cache).

    cached 音色排前：本机 HF 镜像不可达时未缓存音色无法自动下载，
    只能手动放置 .pt 或从 ModelScope 镜像下载。
    """
    out = []
    for v in KOKORO_VOICES:
        item = dict(v)
        item["cached"] = _find_voice_in_cache(f"{v['name']}.pt") is not None
        out.append(item)
    out.sort(key=lambda v: (not v["cached"], v["name"]))
    return out


# ---------------------------------------------------------------------------
# 默认音色按模式分套 + 同性别错开（共用 helper，qwen/moss 引擎 import 复用）
# ---------------------------------------------------------------------------

# config["modes"] 的合法键（四种视频结构）
VOICE_DEFAULT_MODES = ("original", "original_static", "original_cutout", "quest")

KOKORO_VOICE_DEFAULTS = {
    "default_male": "am_adam",
    "default_female": "af_sarah",
    "default_host_female": "af_sky",
    "default_male2": "am_liam",      # 第二默认男声（同性别冲突时第二个角色用）
    "default_female2": "af_nova",
    "default_male3": "am_echo",      # 第三默认（仅 quest 三个同性别角色时用）
    "default_female3": "af_jessica",
}


def _same_gender_ranks(entries: list[tuple[str, str, bool]]) -> dict[str, int]:
    """未绑定音色的角色按同性别分组，组内按传入顺序给 0/1/2 序号.

    entries: [(key, gender, bound), ...]；bound=True 或 gender 为空的不参与分组。
    返回 {key: rank}，rank 0=第一默认, 1=第二默认, 2=第三默认。
    """
    ranks: dict[str, int] = {}
    counts: dict[str, int] = {}
    for key, gender, bound in entries:
        if bound or not gender:
            continue
        rank = counts.get(gender, 0)
        ranks[key] = rank
        counts[gender] = rank + 1
    return ranks


def _gender_default(defaults: dict, base: str, rank: int) -> str:
    """按 rank 取第一/第二/第三默认音色，缺键逐级回退（X3→X2→X，X2→X）."""
    if rank >= 2:
        v = defaults.get(f"{base}3") or defaults.get(f"{base}2") or defaults.get(base)
        if v:
            return v
    if rank >= 1:
        v = defaults.get(f"{base}2") or defaults.get(base)
        if v:
            return v
    return defaults.get(base, "")


def _resolve_mode_defaults(saved: dict, structure: str | None, builtin: dict) -> dict:
    """默认音色解析：builtin ← 配置平铺键(legacy) ← config["modes"][structure] 覆盖."""
    out = dict(builtin)
    for k in builtin:
        v = saved.get(k)
        if v:
            out[k] = v
    if structure:
        section = (saved.get("modes") or {}).get(structure) or {}
        for k in builtin:
            v = section.get(k)
            if v:
                out[k] = v
    return out


def _load_kokoro_voice_config() -> dict:
    """Load configs/kokoro_voice_config.json 原始内容（与 Web 音色页配置共享）.

    缺文件/坏 JSON 返回 {}（默认值由 _resolve_mode_defaults 兜底）。
    """
    config_path = Path(__file__).parent.parent / "configs" / "kokoro_voice_config.json"
    if config_path.exists():
        try:
            return json.loads(config_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def gender_default_ranks(script: dict) -> dict[str, int]:
    """char_a/b/c 未绑定 kokoro 音色者的同性别分组序号（0=第一默认,1=第二,2=第三）."""
    entries = []
    for key in ("char_a", "char_b", "char_c"):
        fallback = {"char_a": "male", "char_b": "female"}.get(key, "")
        gender = (script.get(f"{key}_gender") or fallback).lower()
        bound = bool((script.get(f"{key}_kokoro_voice") or "").strip())
        entries.append((key, gender, bound))
    return _same_gender_ranks(entries)


def build_voice_map(script: dict, structure: str | None = None) -> dict:
    """Build voice_map from char_a/char_b/char_c/host genders.

    Priority 1: script['{key}_kokoro_voice'] (素材库 / Kokoro 音色绑定).
    Priority 2: by gender — configs/kokoro_voice_config.json 按模式分套的默认音色；
    未绑定的同性别角色依次取第一/第二/第三默认（char_a→char_b→char_c），
    性别不同时各自取第一默认。structure 决定用哪个模式的默认套
    （缺省读 script['structure']，再兜底 original）。
    """
    if not structure:
        structure = script.get("structure") or "original"
    defaults = _resolve_mode_defaults(_load_kokoro_voice_config(), structure,
                                      KOKORO_VOICE_DEFAULTS)
    ranks = gender_default_ranks(script)
    voice_map = {}
    for key in ["char_a", "char_b", "char_c", "host"]:
        bound = script.get(f"{key}_kokoro_voice", "").strip()
        if bound:
            voice_map[key] = bound
    for key in ["char_a", "char_b", "char_c"]:
        if key in voice_map:
            continue
        fallback = {"char_a": "male", "char_b": "female"}.get(key, "")
        gender = (script.get(f"{key}_gender") or fallback).lower()
        if not gender:
            continue
        base = "default_male" if gender == "male" else "default_female"
        voice_map[key] = _gender_default(defaults, base, ranks.get(key, 0))
    # Quest mode host (节目主) — appears in welcome/hook/outro segments
    if "host" not in voice_map:
        host_gender = script.get("host_gender", "").lower()
        if host_gender:
            voice_map["host"] = defaults["default_male"] if host_gender == "male" else defaults["default_host_female"]
        else:
            print(f"  [TTS] WARNING: host_gender not set, defaulting to {defaults['default_host_female']} (female). "
                  "Check CHARACTER_OVERRIDES or script.json host_gender.")
            voice_map["host"] = defaults["default_host_female"]
    return voice_map


def get_zh_voice(speaker: str, script: dict) -> str:
    """Get Chinese edge-tts voice based on speaker gender.

    male   -> zh-CN-YunxiNeural
    female -> zh-CN-XiaoxiaoNeural
    """
    if speaker == "char_a":
        gender = script.get("char_a_gender", "male").lower()
    else:
        gender = script.get("char_b_gender", "female").lower()
    if gender == "male":
        return "zh-CN-YunxiNeural"
    return "zh-CN-XiaoxiaoNeural"
