"""Shared media utilities — FFmpeg helpers, font constants, and common
video composition building blocks used by all structure variants.

Consolidates previously duplicated code from:
  - pipeline._get_audio_duration
  - video_compose._get_duration / _probe_resolution / _has_audio
  - tts_engine.TTSEngine.get_duration
  - video_compose + quest: concat, subtitle burn, loudnorm
  - pipeline._safe_dirname + video_compose inline _re.sub
"""
import os
import re
import sys
import subprocess
import shutil
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

# ---------------------------------------------------------------------------
# Font paths (auto-detect Windows vs Linux/Colab)
# ---------------------------------------------------------------------------
import platform

_IS_WINDOWS = platform.system() == "Windows"

if _IS_WINDOWS:
    FONT_EN = r"C:\Windows\Fonts\msyhbd.ttc"
    FONT_ZH = r"C:\Windows\Fonts\msyh.ttc"
    FONT_PH = r"C:\Windows\Fonts\cambria.ttc"
else:
    FONT_EN = "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc"
    FONT_ZH = "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"
    # DejaVu has complete IPA glyph coverage — Noto CJK renders many IPA
    # characters (ɪ ə ʃ ʒ ɡ...) as tofu boxes
    FONT_PH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    if not os.path.exists(FONT_EN):
        FONT_EN = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    if not os.path.exists(FONT_ZH):
        FONT_ZH = "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"
    if not os.path.exists(FONT_PH):
        FONT_PH = FONT_EN

# ---------------------------------------------------------------------------
# Target canvas — every segment is normalized to this size for concat safety
# ---------------------------------------------------------------------------
TARGET_W, TARGET_H = 1280, 720
VF_NORM = (
    f"scale={TARGET_W}:{TARGET_H}:force_original_aspect_ratio=decrease,"
    f"pad={TARGET_W}:{TARGET_H}:(ow-iw)/2:(oh-ih)/2"
)


# ---------------------------------------------------------------------------
# ffprobe helpers
# ---------------------------------------------------------------------------

def get_duration(path: str) -> float:
    """Get media duration in seconds via ffprobe.

    Consolidates pipeline._get_audio_duration, video_compose._get_duration,
    and tts_engine.TTSEngine.get_duration.
    """
    if not path or not os.path.exists(path):
        return 0.0
    try:
        return float(subprocess.check_output(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(path)],
            text=True, encoding="utf-8", errors="replace",
        ).strip())
    except Exception:
        return 0.0


def probe_resolution(video_path: str) -> tuple[int, int]:
    """Get video stream resolution via ffprobe (fallback TARGET_W x TARGET_H).

    Overlays are rendered at this size so they match the video canvas exactly.
    """
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "csv=p=0",
             str(video_path)], text=True, encoding="utf-8", errors="replace").strip()
        w_str, h_str = out.split("x")
        w, h = int(w_str), int(h_str)
        if w > 0 and h > 0:
            return w, h
    except Exception:
        pass
    return TARGET_W, TARGET_H


def has_audio(video_path: str) -> bool:
    """Check if a video file has an audio stream."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-select_streams", "a",
             "-show_entries", "stream=codec_type", "-of", "csv=p=0",
             str(video_path)],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10,
        )
        return "audio" in r.stdout.strip()
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Filename helpers
# ---------------------------------------------------------------------------

def safe_filename(yt_title: str, fallback: str = "final_video") -> str:
    """Sanitize a YouTube title into a filesystem-safe name.

    Consolidates pipeline._safe_dirname and video_compose inline _re.sub.
    """
    name = re.sub(r'[\U0001F000-\U0001FFFF]', '', yt_title)  # remove emoji
    name = re.sub(r"[\\/:*?\"'<>|]", '', name).strip()  # ' breaks FFmpeg concat demuxer
    name = re.sub(r'\s+', '_', name)[:80]
    if not name:
        name = re.sub(r'[^\w\s-]', '', fallback).strip().replace(' ', '_')
    return name or "final_video"


# ---------------------------------------------------------------------------
# FFmpeg command runners
# ---------------------------------------------------------------------------

def run_ffmpeg_with_fallback(cmd: list[str], fallback_cmd: list[str],
                             out_path: str, label: str = "segment",
                             timeout: int = 300) -> bool:
    """Run an FFmpeg command; on failure, try fallback_cmd.

    Returns True if out_path was produced (> 1KB), False otherwise.
    Both commands must produce the same out_path.
    """
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"  FFmpeg TIMEOUT ({timeout}s) on {label}, trying fallback...")
        r = None

    if r is not None and r.returncode == 0 and os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
        return True

    if r is not None:
        stderr_tail = r.stderr[-300:] if r.stderr else ""
        print(f"  FFmpeg error {label}: {stderr_tail}")

    # Try fallback
    try:
        r2 = subprocess.run(fallback_cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"  Fallback also timed out for {label}")
        return False

    if r2.returncode == 0 and os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
        return True

    stderr_tail = r2.stderr[-300:] if r2.stderr else ""
    print(f"  Fallback also failed for {label}: {stderr_tail}")
    return False


def make_silent_fallback_cmd(scene_img: str, duration: float,
                             out_path: str) -> list[str]:
    """Build a fallback FFmpeg command: silent static-image segment.

    Used by run_ffmpeg_with_fallback when the primary command fails.
    Produces a segment matching the standard format (libx264/yuv420p/24fps/
    aac/44100Hz/stereo) so it can be concat-demuxed.
    """
    return [
        "ffmpeg", "-y", "-loop", "1", "-i", scene_img,
        "-f", "lavfi", "-i", "anullsrc=stereo:44100",
        "-t", f"{duration:.3f}", "-vf", f"{VF_NORM},fps=24",
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
        out_path,
    ]


# ---------------------------------------------------------------------------
# Segment concat
# ---------------------------------------------------------------------------

def concat_segments(segment_paths: list[str], output_path: str,
                    tmp_dir: str | Path = None) -> str:
    """Concatenate segment files via FFmpeg concat demuxer (-c copy).

    All segments must have uniform format (libx264/yuv420p/24fps/aac/44100Hz/stereo).
    Returns the output path on success, raises RuntimeError on failure.
    """
    tmp_dir = Path(tmp_dir).resolve() if tmp_dir else Path(output_path).resolve().parent / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    concat_list = tmp_dir / "concat.txt"
    with open(concat_list, "w", encoding="utf-8") as f:
        for s in segment_paths:
            p = str(Path(s).resolve()).replace("'", "'\\''")
            f.write(f"file '{p}'\n")

    result = subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list),
        "-c", "copy", output_path,
    ], capture_output=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Concat failed: {result.stderr.decode(errors='replace')[-2000:]}")

    return output_path


# ---------------------------------------------------------------------------
# Subtitle rendering + overlay burn
# ---------------------------------------------------------------------------

def burn_subtitles(no_sub_path: str, timeline: list[dict], script: dict,
                   work_dir: str, srt_dir: str, pad: float = 0.4,
                   progress_cb=None, show_zh: bool = True,
                   subtitle_seg_types: tuple[str, ...] = ("dialogue", "welcome", "hook_intro", "outro"),
                   en_font_size: int = 60, zh_font_size: int = 50) -> str:
    """Render dialogue subtitles via Pillow and burn them onto the video.

    Extracts subtitle entries from timeline segments whose type is in
    subtitle_seg_types, renders transparent PNG overlays sized to match the
    actual video canvas, and applies them via FFmpeg filter_complex overlay
    with timed enable.

    Returns the final video path with subtitles burned in.
    """
    def _cb(pct, msg):
        if progress_cb:
            progress_cb(pct, msg)

    work = Path(work_dir)
    tmp_dir = work / "tmp_segments"

    final_path = str(work / f"{safe_filename(script.get('youtube_title', script.get('title', 'final_video')))}.mp4")

    # Extract subtitle entries from timeline
    import re as _re

    def _split_subtitles(en: str, zh: str, audio_d: float, t_start: float):
        """Split long narration into per-sentence subtitle entries.

        First splits by sentence-ending punctuation, then further splits
        very long sentences by commas/semicolons so text doesn't cram.
        Returns list of subtitle entry dicts.
        """
        EN_MAX_CHARS = 80
        EN_MAX_WORDS = 12
        ZH_MAX_CHARS = 30

        def _further_split_en(text: str) -> list[str]:
            """Split EN by [.!?] first, then by [,;] if still too long."""
            parts = _re.split(r'(?<=[.!?])\s+', text.strip())
            result = []
            for p in parts:
                p = p.strip()
                if not p:
                    continue
                if len(p) > EN_MAX_CHARS or len(p.split()) > EN_MAX_WORDS:
                    sub_parts = _re.split(r'(?<=[,;])\s+', p)
                    result.extend(s.strip() for s in sub_parts if s.strip())
                else:
                    result.append(p)
            return result if result else [text.strip()]

        def _further_split_zh(text: str) -> list[str]:
            """Split ZH by [。！？] first, then by [，；] if still too long."""
            parts = _re.split(r'(?<=[。！？])\s*', text.strip())
            result = []
            for p in parts:
                p = p.strip()
                if not p:
                    continue
                if len(p) > ZH_MAX_CHARS:
                    sub_parts = _re.split(r'(?<=[，；])\s*', p)
                    result.extend(s.strip() for s in sub_parts if s.strip())
                else:
                    result.append(p)
            return result if result else [text.strip()]

        en_parts = _further_split_en(en) if en else []
        zh_parts = _further_split_zh(zh) if zh else []
        n = max(len(en_parts), len(zh_parts), 1)
        total_chars = sum(len(s) for s in en_parts) or 1
        entries = []
        cursor = t_start
        for i in range(n):
            sent_en = en_parts[i] if i < len(en_parts) else ""
            sent_zh = zh_parts[i] if i < len(zh_parts) else ""
            sent_frac = (len(sent_en) if sent_en else len(sent_zh) or 1) / total_chars
            sent_dur = audio_d * sent_frac if i < n - 1 else audio_d - (cursor - t_start)
            entries.append({
                "start": cursor,
                "end": cursor + sent_dur,
                "en": sent_en,
                "zh": sent_zh,
            })
            cursor += sent_dur
        return entries

    subtitle_entries = []
    t_cursor = 0.0
    for seg in timeline:
        dur = seg["duration"]
        seg_type = seg.get("type", "")
        if seg_type in subtitle_seg_types:
            en = seg.get("subtitle_en", "")
            zh = seg.get("subtitle_zh", "")
            audio_d = seg.get("audio_dur", dur - pad)
            if en or zh:
                subtitle_entries.extend(
                    _split_subtitles(en, zh, audio_d, t_cursor)
                )
        t_cursor += dur

    # Drop entries that have no visible text (happens when show_zh=False
    # and the entry's EN side is empty because ZH had more sentences)
    subtitle_entries = [
        e for e in subtitle_entries
        if e["en"] or (e["zh"] if show_zh else "")
    ]

    if not subtitle_entries:
        shutil.copy2(no_sub_path, final_path)
        return final_path

    from PIL import Image, ImageDraw, ImageFont

    w, h = probe_resolution(no_sub_path)
    sub_overlay_dir = tmp_dir / "subtitles"
    sub_overlay_dir.mkdir(exist_ok=True)

    BOTTOM_MARGIN = 36  # clear of frame edge + YouTube player UI
    EN_SIZE = en_font_size
    ZH_SIZE = zh_font_size
    MAX_SUB_W = w - 80
    LINE_GAP = 6

    def _wrap_fixed(text, font, max_w, is_cjk=False):
        """Wrap text into lines at a fixed font size (no shrinking).

        For CJK text, wrap per-character; for Latin, wrap per-word.
        Returns list of (line_text, line_width, line_height).
        """
        lines = []
        if is_cjk:
            cur = ""
            for ch in text:
                test = cur + ch
                bbox = draw.textbbox((0, 0), test, font=font)
                if bbox[2] - bbox[0] <= max_w or not cur:
                    cur = test
                else:
                    lines.append(cur)
                    cur = ch
            if cur:
                lines.append(cur)
        else:
            words = text.split()
            cur = ""
            for word in words:
                test = (cur + " " + word).strip()
                bbox = draw.textbbox((0, 0), test, font=font)
                if bbox[2] - bbox[0] <= max_w or not cur:
                    cur = test
                else:
                    lines.append(cur)
                    cur = word
            if cur:
                lines.append(cur)
        result = []
        for ln in lines:
            bbox = draw.textbbox((0, 0), ln, font=font)
            result.append((ln, bbox[2] - bbox[0], bbox[3] - bbox[1]))
        return result

    for i, entry in enumerate(subtitle_entries):
        overlay_path = str(sub_overlay_dir / f"sub_{i:03d}.png")
        bg = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(bg)

        en_text = entry["en"]
        zh_text = entry["zh"] if show_zh else ""

        # Fixed font size with word wrapping
        en_lines = []
        if en_text:
            en_font = ImageFont.truetype(FONT_EN, EN_SIZE)
            en_lines = _wrap_fixed(en_text, en_font, MAX_SUB_W, is_cjk=False)

        zh_lines = []
        if zh_text:
            zh_font = ImageFont.truetype(FONT_ZH, ZH_SIZE)
            zh_lines = _wrap_fixed(zh_text, zh_font, MAX_SUB_W, is_cjk=True)

        en_total_h = sum(lh for _, _, lh in en_lines) + LINE_GAP * max(0, len(en_lines) - 1) if en_lines else 0
        zh_total_h = sum(lh for _, _, lh in zh_lines) + LINE_GAP * max(0, len(zh_lines) - 1) if zh_lines else 0

        # Stack bottom-up: ZH lowest, EN above it
        if en_lines and zh_lines:
            zh_block_y = h - BOTTOM_MARGIN - zh_total_h
            en_block_y = zh_block_y - 15 - en_total_h
        elif en_lines:
            en_block_y = h - BOTTOM_MARGIN - en_total_h
            zh_block_y = 0
        else:
            en_block_y = 0
            zh_block_y = h - BOTTOM_MARGIN - zh_total_h

        # Render EN lines
        cur_y = en_block_y
        for ln_text, ln_w, ln_h in en_lines:
            draw.text(((w - ln_w) // 2, cur_y), ln_text, font=en_font,
                      fill=(255, 255, 255, 255), stroke_width=5,
                      stroke_fill=(0, 0, 0, 255))
            cur_y += ln_h + LINE_GAP

        # Render ZH lines
        cur_y = zh_block_y
        for ln_text, ln_w, ln_h in zh_lines:
            draw.text(((w - ln_w) // 2, cur_y), ln_text, font=zh_font,
                      fill=(255, 215, 0, 255), stroke_width=4,
                      stroke_fill=(0, 0, 0, 255))
            cur_y += ln_h + LINE_GAP

        bg.save(overlay_path, "PNG")
        entry["overlay_path"] = overlay_path

    # Burn subtitles in batches to avoid Windows command-line length limit
    # (WinError 206, ~32767 chars for CreateProcess). Each batch applies a
    # subset of overlay PNGs in a separate FFmpeg pass; the video output
    # chains from one pass to the next.  Audio is copied through untouched.
    BATCH_SIZE = 40
    total_batches = (len(subtitle_entries) + BATCH_SIZE - 1) // BATCH_SIZE
    current_input = no_sub_path
    temp_files: list[str] = []

    for batch_idx in range(total_batches):
        batch_start = batch_idx * BATCH_SIZE
        batch_end = min(batch_start + BATCH_SIZE, len(subtitle_entries))
        batch = subtitle_entries[batch_start:batch_end]
        is_last = (batch_idx == total_batches - 1)

        if is_last:
            out_path = final_path
        else:
            out_path = str(tmp_dir / f"sub_batch_{batch_idx:03d}.mp4")
            temp_files.append(out_path)

        # Build filter chain for this batch only (labels restart from 0)
        filter_parts = []
        prev_label = "0:v"
        for j, entry in enumerate(batch):
            start = entry["start"]
            end = entry["end"]
            filter_parts.append(
                f"[{prev_label}][{j+1}:v]overlay=0:0:enable='between(t,{start:.3f},{end:.3f})'[v{j}]"
            )
            prev_label = f"v{j}"

        filter_complex = ";".join(filter_parts)
        final_label = prev_label

        input_args = ["-i", current_input]
        for entry in batch:
            input_args.extend(["-i", entry["overlay_path"]])

        cmd = ["ffmpeg", "-y"] + input_args + [
            "-filter_complex", filter_complex,
            "-map", f"[{final_label}]",
            "-map", "0:a:0",
            "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-r", "24",
            "-c:a", "copy",
            out_path,
        ]

        if progress_cb:
            pct = 90 + int((batch_idx / max(total_batches, 1)) * 5)
            progress_cb(pct, f"Burning subtitles batch {batch_idx + 1}/{total_batches}")

        try:
            subprocess.run(cmd, check=True, capture_output=True,
                            cwd=str(srt_dir), timeout=1800)
        except subprocess.TimeoutExpired:
            raise RuntimeError("Subtitle overlay burn timed out after 1800s")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Subtitle overlay burn failed (batch {batch_idx + 1}/{total_batches}): "
                f"{e.stderr.decode(errors='replace')[-500:] if e.stderr else e}")

        current_input = out_path

    # Clean up intermediate temp files
    for tf in temp_files:
        try:
            os.remove(tf)
        except OSError:
            pass

    return final_path


# ---------------------------------------------------------------------------
# Final loudnorm pass
# ---------------------------------------------------------------------------

def apply_final_loudnorm(video_path: str, vid_dir: str,
                        progress_cb=None) -> str:
    """Apply final loudnorm normalization to the composed video.

    Tries loudnorm first; if it fails, falls back to volume boost.
    Returns the path to the normalized video (may be the same as input).
    """
    def _cb(pct, msg):
        if progress_cb:
            progress_cb(pct, msg)

    norm_path = str(Path(vid_dir) / "final_video_norm.mp4")
    norm_result = subprocess.run(
        ["ffmpeg", "-y", "-i", video_path,
         "-c:v", "copy",  # video passthrough — fast, no re-encode
         "-c:a", "aac", "-b:a", "128k",
         "-af", "loudnorm=I=-14:TP=-1.5:LRA=11",
         norm_path],
        capture_output=True, timeout=600,
    )
    if (norm_result.returncode == 0 and os.path.exists(norm_path)
            and os.path.getsize(norm_path) > 1000):
        os.replace(norm_path, video_path)
        return video_path

    # Fallback: simple volume boost
    if os.path.exists(norm_path):
        try:
            os.remove(norm_path)
        except OSError:
            pass

    vol_path = str(Path(vid_dir) / "final_video_vol.mp4")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path,
             "-c:v", "copy",
             "-c:a", "aac", "-b:a", "128k",
             "-af", "volume=6dB",
             vol_path],
            capture_output=True, timeout=600,
        )
        if os.path.exists(vol_path) and os.path.getsize(vol_path) > 1000:
            os.replace(vol_path, video_path)
    except Exception:
        pass  # Keep original if both fail

    return video_path


# ---------------------------------------------------------------------------
# SRT building (shared by all timeline variants)
# ---------------------------------------------------------------------------

def _format_srt_time(seconds: float) -> str:
    """Convert seconds to SRT timestamp: HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def build_srt(timeline: list[dict], skip_types: set[str] | None = None,
              gap: float = 0.0) -> str:
    """Build SRT from a timeline list. Timestamps match video exactly.

    Segments whose type is in skip_types produce no SRT entry (text is on
    static images, not subtitles). Segments with empty subtitle_en are also
    skipped.

    Consolidates timeline.build_srt_from_timeline,
    quest.build_srt_from_timeline_quest.
    """
    if skip_types is None:
        skip_types = {"listen_en", "listen_zh", "practice",
                      "practice_intro", "vocab", "quiz",
                      "dialogue_slow"}

    srt_lines = []
    idx = 1
    current_time = 0.0

    for seg in timeline:
        dur = seg["duration"]
        start = current_time
        end = start + dur

        text_en = seg.get("subtitle_en", "")
        text_zh = seg.get("subtitle_zh", "")
        seg_type = seg.get("type", "")

        if seg_type in skip_types:
            current_time = end + gap
            continue

        if not text_en:
            current_time = end + gap
            continue

        audio_dur = seg.get("audio_dur", dur)
        srt_end = start + audio_dur

        srt_lines.append(str(idx))
        srt_lines.append(f"{_format_srt_time(start)} --> {_format_srt_time(srt_end)}")
        srt_lines.append(text_en)
        if text_zh:
            srt_lines.append(text_zh)
        srt_lines.append("")
        idx += 1
        current_time = end + gap

    return "\n".join(srt_lines)
