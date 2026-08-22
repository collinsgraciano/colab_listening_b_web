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
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

# HF_ENDPOINT: Windows local uses hf-mirror (HuggingFace blocked in China),
# Colab/Linux uses direct huggingface.co. Allow override via env var.
if "HF_ENDPOINT" not in os.environ:
    if sys.platform == "win32":
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    else:
        os.environ['HF_ENDPOINT'] = 'https://huggingface.co'


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


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences for Kokoro processing."""
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p for p in parts if p.strip()]


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

    def synth_english(self, text: str, voice: str, out_path: str,
                      rate: str = "+0%") -> float:
        """Synthesize English text with Kokoro TTS. Returns duration in seconds.

        Args:
            text: English text to speak
            voice: Kokoro voice ID (af_sarah, am_adam, af_sky, etc.)
            out_path: output MP3 file path
            rate: speed adjustment ('+0%'=normal, '-15%'=slow, '+10%'=fast)
        """
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        speed = _rate_to_speed(rate)
        pipeline = self._get_kokoro()

        import soundfile as sf
        import numpy as np
        all_audio = []

        for sentence in _split_sentences(text):
            sentence = sentence.strip()
            if not sentence:
                continue
            # Apply phonetic fixes FIRST (before cleaning removes hyphens)
            fixed = _apply_phonetic_fixes(sentence)
            cleaned = _clean_text_for_kokoro(fixed)
            try:
                for gs, ps, audio in pipeline(cleaned, voice=voice, speed=speed):
                    if audio is not None and len(audio) > 0:
                        all_audio.append(audio)
            except Exception as e:
                print(f"    [Kokoro skip] {cleaned[:40]}: {e}")

        if not all_audio:
            raise RuntimeError(f"Kokoro produced no audio for: {text[:50]}")

        final_audio = all_audio[0] if len(all_audio) == 1 else np.concatenate(all_audio)

        # Write WAV (24kHz), then convert to MP3 via ffmpeg
        wav_path = out_path.replace('.mp3', '_tmp.wav')
        sf.write(wav_path, final_audio, 24000)
        subprocess.run(
            ["ffmpeg", "-y", "-i", wav_path, "-c:a", "libmp3lame", "-b:a", "128k",
             "-ar", "24000", "-ac", "1", out_path],
            check=True, capture_output=True
        )
        os.remove(wav_path)

        # Apply loudnorm
        self._loudnorm(out_path)

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
        subprocess.run(
            ["ffmpeg", "-y", "-i", wav_path, "-c:a", "libmp3lame", "-b:a", "128k",
             "-ar", "24000", "-ac", "1", out_path],
            check=True, capture_output=True
        )
        os.remove(wav_path)
        self._loudnorm(out_path)
        return self.get_duration(out_path)


# Voice configuration helpers
def build_voice_map(script: dict) -> dict:
    """Build voice_map from char_a_gender/char_b_gender.

    Male -> am_adam, Female -> af_sarah.
    """
    voice_map = {}
    char_a_gender = script.get("char_a_gender", "male").lower()
    char_b_gender = script.get("char_b_gender", "female").lower()
    voice_map["char_a"] = "am_adam" if char_a_gender == "male" else "af_sarah"
    voice_map["char_b"] = "am_adam" if char_b_gender == "male" else "af_sarah"
    # Quest mode third character (staff/interviewer) — only present in quest scripts
    char_c_gender = script.get("char_c_gender", "").lower()
    if char_c_gender:
        voice_map["char_c"] = "am_adam" if char_c_gender == "male" else "af_sarah"
    # Quest mode host (节目主) — appears in welcome/hook/outro segments
    host_gender = script.get("host_gender", "").lower()
    if host_gender:
        voice_map["host"] = "am_adam" if host_gender == "male" else "af_sky"
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
