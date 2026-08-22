"""VoxCPM TTS engine via Cloudflare Worker.

Generates English speech using VoxCPM model hosted on a Cloudflare Worker.
Voice characteristics are controlled via natural language descriptions
(the ``control`` parameter), so the LLM can design per-character voices.

To ensure voice consistency across sentences, a short reference clip is
generated per voice description on first use, then uploaded back to the
Worker as a reference audio anchor for all subsequent calls.

Chinese TTS and loudnorm are inherited from :class:`TTSEngine`.
"""
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import requests

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

# Add parent to path so we can import tts_engine
_PARENT = str(Path(__file__).parent.resolve())
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from tts_engine import TTSEngine


class VoxCPMEngine(TTSEngine):
    """VoxCPM TTS engine via Cloudflare Worker.

    Inherits ``synth_chinese``, ``_loudnorm``, ``get_duration`` from TTSEngine.
    Overrides ``synth_english`` to call the VoxCPM Worker API.

    Flow per voice:
        1. Generate a short reference clip (one-time, per voice description)
        2. Upload reference clip to Worker as anchor
        3. For each sentence: submit /generate (with reference) -> SSE poll -> download WAV
        4. Concatenate sentence WAVs -> ffmpeg MP3 -> atempo speed -> loudnorm
    """

    # --- Generation parameters ---
    CFG_VALUE = 2.0        # Gradio slider range 1.0–3.0, default 2.0
    DO_NORMALIZE = True
    DENOISE = True

    # Short reference text for voice anchoring (varied to avoid all-same intro)
    _REF_TEXTS = [
        "Hello, nice to meet you. How are you doing today?",
        "Hey there, it's great to see you. What's up?",
        "Hi, I'm so glad you're here. Let me tell you something.",
    ]

    def __init__(self, worker_url: str, api_key: str = ""):
        self.worker_url = worker_url.rstrip("/")
        self.api_key = api_key
        self._session = requests.Session()
        # Cache: voice_description -> (gradio_path, ref_text)
        self._reference_cache: dict[str, tuple[str, str]] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _headers(self) -> dict:
        h = {}
        if self.api_key:
            h["X-API-Key"] = self.api_key
        return h

    def test_connection(self) -> bool:
        """Quick health check against the Worker."""
        try:
            r = self._session.get(
                f"{self.worker_url}/health",
                headers=self._headers(),
                timeout=30,
            )
            r.raise_for_status()
            print(f"  [VoxCPM] Worker OK: {r.text[:80]}")
            return True
        except Exception as e:
            print(f"  [VoxCPM] Health check failed: {e}")
            return False

    def _upload_reference_wav(self, wav_path: str) -> str:
        """Upload a local WAV to the Worker and return the Gradio path."""
        with open(wav_path, "rb") as f:
            files = {"files": (Path(wav_path).name, f, "audio/wav")}
            r = self._session.post(
                f"{self.worker_url}/upload",
                files=files,
                headers=self._headers(),
                timeout=120,
            )
        r.raise_for_status()
        result = r.json()
        if isinstance(result, list):
            if not result:
                raise RuntimeError("Gradio upload returned empty list")
            return result[0]
        if isinstance(result, dict):
            if "path" in result:
                return result["path"]
            if "files" in result:
                return result["files"][0]
        raise RuntimeError(f"Cannot parse upload result: {result}")

    def _submit(self, text: str, control: str,
                reference_wav_path: str | None = None,
                use_prompt_text: bool = False,
                prompt_text: str = "",
                cfg_value: int = None,
                do_normalize: bool = None,
                denoise: bool = None) -> str:
        """Submit a generation task, return event_id."""
        # Gradio expects FileData object (not bare string) for audio input
        ref_data = None
        if reference_wav_path:
            ref_data = {"path": reference_wav_path,
                        "meta": {"_type": "gradio.FileData"}}
        data = [
            text,
            control,
            ref_data,
            use_prompt_text,
            prompt_text,
            cfg_value if cfg_value is not None else self.CFG_VALUE,
            do_normalize if do_normalize is not None else self.DO_NORMALIZE,
            denoise if denoise is not None else self.DENOISE,
        ]
        r = self._session.post(
            f"{self.worker_url}/generate",
            json={"data": data},
            headers={**self._headers(), "Content-Type": "application/json"},
            timeout=120,
        )
        r.raise_for_status()
        result = r.json()
        if "event_id" not in result:
            raise RuntimeError(f"No event_id in VoxCPM response: {result}")
        return result["event_id"]

    @staticmethod
    def _parse_sse(event_text: str):
        """Parse a single SSE event block -> (event_type, data)."""
        event_type = None
        data_lines = []
        for line in event_text.splitlines():
            if line.startswith("event:"):
                event_type = line[6:].strip()
            elif line.startswith("data:"):
                data_lines.append(line[5:].strip())
        data_text = "\n".join(data_lines)
        data = None
        if data_text:
            try:
                data = json.loads(data_text)
            except Exception:
                data = data_text
        return event_type, data

    def _wait(self, event_id: str, timeout: int = 600):
        """Block on SSE stream until 'complete' or 'error'."""
        url = f"{self.worker_url}/events/{event_id}"
        start = time.time()
        with self._session.get(
            url,
            headers={
                **self._headers(),
                "Accept": "text/event-stream",
                "Cache-Control": "no-cache",
            },
            stream=True,
            timeout=(30, timeout),
        ) as r:
            r.raise_for_status()
            buf = ""
            for raw in r.iter_lines(decode_unicode=True):
                if time.time() - start > timeout:
                    raise TimeoutError("VoxCPM generation timed out")
                if raw is None:
                    continue
                if raw == "":
                    if buf.strip():
                        etype, edata = self._parse_sse(buf)
                        if etype == "complete":
                            return edata
                        if etype == "error":
                            raise RuntimeError(
                                f"VoxCPM error: {edata}\nRaw SSE:\n{buf}"
                            )
                        buf = ""
                    continue
                buf += raw + "\n"
        raise RuntimeError("SSE ended without 'complete' event")

    @staticmethod
    def _find_wav_path(obj):
        """Recursively search Gradio result for a .wav file path."""
        if isinstance(obj, str):
            if obj.endswith(".wav") or "/tmp/gradio/" in obj or "\\tmp\\gradio\\" in obj:
                return obj
            return None
        if isinstance(obj, dict):
            for k in ("path", "filepath", "file_path"):
                if k in obj and isinstance(obj[k], str):
                    return obj[k]
            for v in obj.values():
                found = VoxCPMEngine._find_wav_path(v)
                if found:
                    return found
            return None
        if isinstance(obj, (list, tuple)):
            for item in obj:
                found = VoxCPMEngine._find_wav_path(item)
                if found:
                    return found
        return None

    def _download(self, result, wav_path: str):
        """Download generated WAV from the Worker."""
        remote = self._find_wav_path(result)
        if not remote:
            raise RuntimeError(f"No WAV path in VoxCPM result: {result}")
        r = self._session.get(
            f"{self.worker_url}/download",
            params={"path": remote},
            headers=self._headers(),
            stream=True,
            timeout=300,
        )
        r.raise_for_status()
        Path(wav_path).parent.mkdir(parents=True, exist_ok=True)
        with open(wav_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    # ------------------------------------------------------------------
    # Reference voice management
    # ------------------------------------------------------------------

    def _ensure_reference(self, voice: str, cache_dir: Path) -> tuple[str, str]:
        """Ensure a reference WAV exists for the given voice description.

        Generates one short sample on first use, uploads it to the Worker,
        and caches the Gradio path for all subsequent calls.

        Returns:
            (gradio_path, reference_text)
        """
        if voice in self._reference_cache:
            return self._reference_cache[voice]

        cache_dir.mkdir(parents=True, exist_ok=True)
        # Deterministic filename based on voice hash
        import hashlib
        voice_hash = int(hashlib.md5(voice.encode()).hexdigest()[:8], 16)
        ref_wav = str(cache_dir / f"ref_{voice_hash}.wav")

        # Check if we already downloaded a reference (resume support)
        if os.path.exists(ref_wav) and os.path.getsize(ref_wav) > 1000:
            print(f"  [VoxCPM] Found cached reference for voice {voice_hash}")
        else:
            # Generate reference clip (no reference audio, just control description)
            ref_text = self._REF_TEXTS[voice_hash % len(self._REF_TEXTS)]
            last_err = None
            max_attempts = 20
            for attempt in range(max_attempts):
                if attempt > 0:
                    wait = min(15 * attempt, 60)
                    print(f"  [VoxCPM] Retry {attempt}/{max_attempts-1} after {wait}s...")
                    time.sleep(wait)
                try:
                    print(f"  [VoxCPM] Generating reference voice for: {voice[:60]}...")
                    eid = self._submit(
                        ref_text, voice,
                        cfg_value=self.CFG_VALUE,
                        do_normalize=self.DO_NORMALIZE,
                        denoise=False,
                    )
                    result = self._wait(eid, timeout=600)
                    self._download(result, ref_wav)
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
                    print(f"  [VoxCPM] Reference generation failed (attempt {attempt+1}/{max_attempts}): {e}")
            if last_err:
                raise last_err

        # Upload reference to Worker
        gradio_path = self._upload_reference_wav(ref_wav)
        print(f"  [VoxCPM] Reference uploaded: {gradio_path[:60]}")

        # Read the ref_text back (deterministic from hash)
        ref_text = self._REF_TEXTS[voice_hash % len(self._REF_TEXTS)]

        self._reference_cache[voice] = (gradio_path, ref_text)
        return gradio_path, ref_text

    # ------------------------------------------------------------------
    # Sentence splitting
    # ------------------------------------------------------------------

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split text into sentences for cleaner generation."""
        parts = re.split(r'(?<=[.!?])\s+', text.strip())
        return [p for p in parts if p.strip()]

    # ------------------------------------------------------------------
    # Public API (same signature as TTSEngine.synth_english)
    # ------------------------------------------------------------------

    def synth_english(self, text: str, voice: str, out_path: str,
                      rate: str = "+0%", max_retries: int = 3) -> float:
        """Synthesize English text via VoxCPM.

        Uses a per-voice reference clip to ensure timbre consistency across
        sentences. Long text is split into sentences, each generated
        separately, then concatenated.

        Args:
            text: English text to speak.
            voice: VoxCPM voice description (the ``control`` parameter).
            out_path: output MP3 path.
            rate: speed adjustment ('-15%' = slow; applied via FFmpeg atempo).
            max_retries: max retry attempts on failure.

        Returns:
            Audio duration in seconds.
        """
        from media_utils import get_duration as _get_dur

        # Convert rate string to atempo factor
        speed = 1.0
        if rate and rate != "+0%":
            try:
                pct = int(rate.replace("%", "").replace("+", ""))
                speed = 1.0 + pct / 100.0
            except (ValueError, TypeError):
                speed = 1.0

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)

        # Ensure reference voice is ready
        ref_cache_dir = Path(out_path).parent / "_voxcpm_refs"
        ref_path, ref_text = self._ensure_reference(voice, ref_cache_dir)

        # Split into sentences for cleaner generation
        sentences = self._split_sentences(text)
        if not sentences:
            sentences = [text]

        # If single sentence, generate directly
        if len(sentences) == 1:
            return self._generate_single(
                sentences[0], voice, ref_path, ref_text,
                out_path, speed, max_retries)

        # Multiple sentences: generate each, then concatenate
        temp_dir = Path(out_path).parent / "_voxcpm_tmp"
        temp_dir.mkdir(parents=True, exist_ok=True)

        wav_paths = []
        for si, sent in enumerate(sentences):
            sent_wav = str(temp_dir / f"sent_{si}.wav")
            success = False
            for attempt in range(max_retries):
                try:
                    eid = self._submit(
                        sent, voice,
                        reference_wav_path=ref_path,
                        use_prompt_text=True,
                        prompt_text=ref_text,
                    )
                    result = self._wait(eid, timeout=600)
                    self._download(result, sent_wav)
                    wav_paths.append(sent_wav)
                    success = True
                    break
                except Exception as e:
                    print(f"    [VoxCPM] Sentence {si} retry {attempt+1}/{max_retries}: {str(e)[:100]}")
                    if attempt < max_retries - 1:
                        time.sleep(3)
            if not success:
                raise RuntimeError(f"VoxCPM failed on sentence {si} after {max_retries} retries")

        # Concatenate sentence WAVs
        combined_wav = str(temp_dir / "combined.wav")
        if len(wav_paths) == 1:
            # Single sentence but went through multi-path
            import shutil
            shutil.copy(wav_paths[0], combined_wav)
        else:
            # Build concat demuxer file
            concat_list = str(temp_dir / "concat.txt")
            with open(concat_list, "w", encoding="utf-8") as f:
                for wp in wav_paths:
                    f.write(f"file '{wp}'\n")
            subprocess.run(
                ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
                 "-i", concat_list, "-c", "copy", combined_wav],
                check=True, capture_output=True,
            )

        # Apply atempo + convert to MP3
        af_parts = []
        if 0.5 <= speed <= 2.0 and speed != 1.0:
            af_parts.append(f"atempo={speed:.4f}")
        af = ",".join(af_parts) if af_parts else "anull"

        subprocess.run(
            ["ffmpeg", "-y", "-i", combined_wav,
             "-af", af,
             "-c:a", "libmp3lame", "-b:a", "128k",
             "-ar", "24000", "-ac", "1", out_path],
            check=True, capture_output=True,
        )

        # Cleanup temp files
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

        # loudnorm (inherited from TTSEngine)
        self._loudnorm(out_path)
        return self.get_duration(out_path)

    def _generate_single(self, sentence: str, voice: str,
                         ref_path: str, ref_text: str,
                         out_path: str, speed: float,
                         max_retries: int) -> float:
        """Generate a single sentence to MP3."""
        wav_path = out_path.replace(".mp3", "_voxcpm.wav")

        last_err = None
        for attempt in range(max_retries):
            try:
                eid = self._submit(
                    sentence, voice,
                    reference_wav_path=ref_path,
                    use_prompt_text=True,
                    prompt_text=ref_text,
                )
                result = self._wait(eid, timeout=600)
                self._download(result, wav_path)

                af_parts = []
                if 0.5 <= speed <= 2.0 and speed != 1.0:
                    af_parts.append(f"atempo={speed:.4f}")
                af = ",".join(af_parts) if af_parts else "anull"

                subprocess.run(
                    ["ffmpeg", "-y", "-i", wav_path,
                     "-af", af,
                     "-c:a", "libmp3lame", "-b:a", "128k",
                     "-ar", "24000", "-ac", "1", out_path],
                    check=True, capture_output=True,
                )
                if os.path.exists(wav_path):
                    os.remove(wav_path)

                self._loudnorm(out_path)
                return self.get_duration(out_path)

            except Exception as e:
                last_err = e
                print(f"    [VoxCPM retry {attempt+1}/{max_retries}] {str(e)[:120]}")
                if os.path.exists(wav_path):
                    os.remove(wav_path)
                if attempt < max_retries - 1:
                    time.sleep(5)

        raise RuntimeError(
            f"VoxCPM failed after {max_retries} retries: {last_err}"
        )
