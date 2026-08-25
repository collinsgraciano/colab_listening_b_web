"""Standalone FFmpeg video composition for listening practice videos.

Adapts compose_listening to take parameters directly (no meta.json, no DATA_DIR).
Includes inline Pillow rendering functions for static frames and subtitle overlays.

No dependency on any external project.
"""
import os
import sys
import subprocess
import shutil
from pathlib import Path

from media_utils import (
    FONT_EN, FONT_ZH, FONT_PH, TARGET_W, TARGET_H, VF_NORM,
    get_duration as _get_duration,
    probe_resolution as _probe_resolution,
    has_audio as _has_audio,
    safe_filename, concat_segments, burn_subtitles, apply_final_loudnorm,
    make_silent_fallback_cmd,
)


# ---------------------------------------------------------------------------
# Pillow rendering functions
# ---------------------------------------------------------------------------

def _render_static_frame(en_text, phonetic, zh_text, scene_img_path,
                          out_path, idx, total, w=1280, h=720):
    """Render a static PNG with English + phonetic + Chinese text over scene."""
    from PIL import Image, ImageDraw, ImageFont

    bg = Image.open(scene_img_path).convert("RGBA").resize((w, h))
    draw_on_bg = ImageDraw.Draw(bg)

    # Sentence number (top-right, yellow on black circle)
    num_font = ImageFont.truetype(FONT_EN, 36)
    num_text = f"{idx+1}/{total}"
    num_bbox = draw_on_bg.textbbox((0, 0), num_text, font=num_font)
    num_tw, num_th = num_bbox[2]-num_bbox[0], num_bbox[3]-num_bbox[1]
    num_pad = 12
    num_radius = max(num_tw, num_th) // 2 + num_pad
    cx, cy = w - num_radius - 20, num_radius + 20

    def _fit_font(text, font_path, start_size=56, min_size=16, max_w=w-80):
        size = start_size
        font = ImageFont.truetype(font_path, size)
        while size > min_size:
            bbox = draw_on_bg.textbbox((0, 0), text, font=font)
            if bbox[2] - bbox[0] <= max_w:
                break
            size -= 2
            font = ImageFont.truetype(font_path, size)
        return size, font

    en_fit, _ = _fit_font(en_text, FONT_EN)
    ph_fit = _fit_font(phonetic, FONT_PH)[0] if phonetic else en_fit
    zh_fit = _fit_font(zh_text, FONT_ZH)[0] if zh_text else en_fit
    final_size = min(en_fit, ph_fit, zh_fit)

    en_font = ImageFont.truetype(FONT_EN, final_size)
    en_bbox = draw_on_bg.textbbox((0, 0), en_text, font=en_font)
    en_w, en_h = en_bbox[2]-en_bbox[0], en_bbox[3]-en_bbox[1]
    target_h = en_h

    def _match_height(text, font_path, target_h, base_size, max_w=w-40):
        size = base_size
        font = ImageFont.truetype(font_path, size)
        bb = draw_on_bg.textbbox((0, 0), text, font=font)
        actual_h = bb[3] - bb[1]
        if actual_h > 0:
            size = int(size * target_h / actual_h)
            size = max(size, 14)
        font = ImageFont.truetype(font_path, size)
        while size > 14:
            bb = draw_on_bg.textbbox((0, 0), text, font=font)
            if bb[2] - bb[0] <= max_w:
                break
            size -= 2
            font = ImageFont.truetype(font_path, size)
        return size, font

    ph_font_final = None
    zh_font_final = None
    ph_w_final = ph_h_final = 0
    zh_w_final = zh_h_final = 0

    if phonetic:
        ph_size, ph_font_final = _match_height(phonetic, FONT_PH, target_h, final_size)
        bbox = draw_on_bg.textbbox((0, 0), phonetic, font=ph_font_final)
        ph_w_final, ph_h_final = bbox[2]-bbox[0], bbox[3]-bbox[1]
    if zh_text:
        zh_size, zh_font_final = _match_height(zh_text, FONT_ZH, target_h, final_size)
        bbox = draw_on_bg.textbbox((0, 0), zh_text, font=zh_font_final)
        zh_w_final, zh_h_final = bbox[2]-bbox[0], bbox[3]-bbox[1]

    gap = 15
    total_text_h = en_h + (ph_h_final + gap if phonetic else 0) + (zh_h_final + gap if zh_text else 0)
    box_padding = 30
    box_h = int(total_text_h + box_padding * 2)
    box_y = (h - box_h) // 2

    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ov_draw = ImageDraw.Draw(overlay)
    ov_draw.rectangle([0, box_y, w, box_y + box_h], fill=(0, 0, 0, 175))
    bg = Image.alpha_composite(bg, overlay)
    draw = ImageDraw.Draw(bg)

    draw.ellipse([cx-num_radius, cy-num_radius, cx+num_radius, cy+num_radius], fill=(0, 0, 0, 200))
    draw.text((cx-num_tw//2, cy-num_th//2-2), num_text, font=num_font, fill=(255, 220, 0, 255))

    en_y = box_y + box_padding
    draw.text(((w-en_w)//2, en_y), en_text, font=en_font, fill=(255, 255, 255, 255))

    if phonetic and ph_font_final:
        ph_y = en_y + en_h + gap
        draw.text(((w-ph_w_final)//2, ph_y), phonetic, font=ph_font_final, fill=(130, 200, 255, 255))

    if zh_text and zh_font_final:
        zh_y = en_y + en_h + gap + (ph_h_final + gap if phonetic else 0)
        draw.text(((w-zh_w_final)//2, zh_y), zh_text, font=zh_font_final, fill=(255, 220, 0, 255))

    bg.convert("RGB").save(out_path, "PNG")


def _render_title_card(title_en, title_zh, scene_zh, scene_img_path,
                       out_path, w=1280, h=720):
    """Render a title card overlay PNG with TRANSPARENT background (for video overlay).

    Text: 大字英文標題 (center) + 繁中標題 (below), thick stroke for readability.
    """
    from PIL import Image, ImageDraw, ImageFont

    STROKE = 8
    MARGIN = 80  # safe margin from frame edge

    bg = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(bg)

    en_y = 0
    en_h = 0
    if title_en:
        display_text = title_en.upper()  # measure what we actually draw
        en_size = 120
        en_font = ImageFont.truetype(FONT_EN, en_size)
        while en_size > 40:
            bbox = draw.textbbox((0, 0), display_text, font=en_font)
            # Account for stroke_width: rendered width = bbox + 2*stroke
            rendered_w = (bbox[2]-bbox[0]) + STROKE * 2
            if rendered_w <= w - MARGIN:
                break
            en_size -= 2
            en_font = ImageFont.truetype(FONT_EN, en_size)
        bbox = draw.textbbox((0, 0), display_text, font=en_font)
        en_w = bbox[2]-bbox[0]
        en_h = bbox[3]-bbox[1]
        en_y = int(h * 0.30)
        draw.text(((w-en_w)//2, en_y), display_text, font=en_font,
                  fill=(255, 255, 255, 255), stroke_width=STROKE, stroke_fill=(0, 0, 0, 255))

    if title_zh:
        ZH_STROKE = 6
        zh_size = 120
        zh_font = ImageFont.truetype(FONT_ZH, zh_size)
        while zh_size > 28:
            bbox = draw.textbbox((0, 0), title_zh, font=zh_font)
            rendered_w = (bbox[2]-bbox[0]) + ZH_STROKE * 2
            if rendered_w <= w - MARGIN:
                break
            zh_size -= 2
            zh_font = ImageFont.truetype(FONT_ZH, zh_size)
        bbox = draw.textbbox((0, 0), title_zh, font=zh_font)
        zh_w = bbox[2]-bbox[0]
        zh_y = en_y + en_h + 30 if title_en else int(h * 0.45)
        draw.text(((w-zh_w)//2, zh_y), title_zh, font=zh_font,
                  fill=(255, 220, 0, 255), stroke_width=ZH_STROKE, stroke_fill=(0, 0, 0, 255))

    bg.save(out_path, "PNG")


def _render_practice_intro(intro_en, intro_zh, scene_img_path,
                            out_path, w=1280, h=720):
    """Render practice intro / outro overlay PNG — large text, multi-line English layout.

    Uses proper word-wrapping to fit long sentences into multiple lines.
    """
    from PIL import Image, ImageDraw, ImageFont

    STROKE = 7
    MARGIN = 80

    bg = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(bg)

    # Word-wrap: split into lines that fit within w - MARGIN at given font size
    def _wrap_text(text, font_path, start_size, min_size, max_w, stroke):
        """Find font size + line breaks so every line fits within max_w."""
        size = start_size
        while size >= min_size:
            font = ImageFont.truetype(font_path, size)
            available_w = max_w - stroke * 2
            words = text.split()
            lines = []
            cur_line = ""
            for word in words:
                test = (cur_line + " " + word).strip()
                bbox = draw.textbbox((0, 0), test, font=font)
                if bbox[2]-bbox[0] <= available_w or not cur_line:
                    cur_line = test
                else:
                    lines.append(cur_line)
                    cur_line = word
            if cur_line:
                lines.append(cur_line)
            # Check all lines fit
            all_fit = True
            for line in lines:
                bbox = draw.textbbox((0, 0), line, font=font)
                if bbox[2]-bbox[0] > available_w:
                    all_fit = False
                    break
            if all_fit and lines:
                return size, font, lines
            size -= 2
        # Fallback: min size, force-split by width
        font = ImageFont.truetype(font_path, min_size)
        available_w = max_w - stroke * 2
        words = text.split()
        lines = []
        cur_line = ""
        for word in words:
            test = (cur_line + " " + word).strip()
            bbox = draw.textbbox((0, 0), test, font=font)
            if bbox[2]-bbox[0] <= available_w or not cur_line:
                cur_line = test
            else:
                lines.append(cur_line)
                cur_line = word
        if cur_line:
            lines.append(cur_line)
        return min_size, font, lines

    en_lines = []
    en_font = None
    if intro_en:
        en_size, en_font, en_lines = _wrap_text(
            intro_en, FONT_EN, 96, 36, w - MARGIN, STROKE)

    # Calculate total height for vertical centering
    line_heights = []
    for line_text in en_lines:
        bbox = draw.textbbox((0, 0), line_text, font=en_font)
        line_heights.append(bbox[3]-bbox[1])

    zh_h = 0
    zh_font_final = None
    zh_w = 0
    if intro_zh:
        zh_size = 80
        zh_font_final = ImageFont.truetype(FONT_ZH, zh_size)
        while zh_size > 28:
            bbox = draw.textbbox((0, 0), intro_zh, font=zh_font_final)
            rendered_w = (bbox[2]-bbox[0]) + 5 * 2
            if rendered_w <= w - MARGIN:
                break
            zh_size -= 2
            zh_font_final = ImageFont.truetype(FONT_ZH, zh_size)
        bbox = draw.textbbox((0, 0), intro_zh, font=zh_font_final)
        zh_w = bbox[2]-bbox[0]
        zh_h = bbox[3]-bbox[1]

    total_h = sum(line_heights) + max(0, len(line_heights) - 1) * 20
    if intro_zh:
        total_h += 30 + zh_h
    start_y = max(int(h * 0.25), (h - total_h) // 2)

    current_y = start_y
    for i, line_text in enumerate(en_lines):
        bbox = draw.textbbox((0, 0), line_text, font=en_font)
        en_w = bbox[2]-bbox[0]
        en_h = bbox[3]-bbox[1]
        draw.text(((w-en_w)//2, current_y), line_text, font=en_font,
                  fill=(255, 255, 255, 255), stroke_width=STROKE, stroke_fill=(0, 0, 0, 255))
        current_y += en_h + 20

    if intro_zh and zh_font_final:
        zh_y = current_y + 10
        draw.text(((w-zh_w)//2, zh_y), intro_zh, font=zh_font_final,
                  fill=(255, 220, 0, 255), stroke_width=5, stroke_fill=(0, 0, 0, 255))

    bg.save(out_path, "PNG")


# ---------------------------------------------------------------------------
# Main compose function
# ---------------------------------------------------------------------------

def compose_listening(
    work_dir: str,
    clip_paths: list[str],
    timeline: list[dict],
    script: dict,
    narration: dict,
    normal_paths: list[str],
    zh_paths: list[str],
    scene_img: str,
    srt_dir: str,
    pad: float = 0.4,
    progress_cb=None,
    group_info: list[dict] | None = None,
    line_to_group: dict | None = None,
    subtitle_font_size: int = 60,
    subtitle_style: dict | None = None,
    show_zh: bool = True,
) -> str:
    """Compose final listening practice video.

    Args:
        work_dir: Working directory for temp files and output.
        clip_paths: List of video clip paths (index 0 = scene/HOOK, 1+ = dialogue groups).
                    May contain None for failed clips (handled gracefully).
        timeline: Timeline segments from build_listening_timeline (with audio_dur added).
        script: Lesson script dict.
        narration: {"intro": path, "outro": path, "practice_intro": path}.
        normal_paths: English dialogue audio paths.
        zh_paths: Chinese dialogue audio paths.
        scene_img: Scene background image path.
        srt_dir: Directory for SRT file (cwd for FFmpeg subtitle burn).
        pad: Audio pad between segments (seconds).
        progress_cb: callback(percent, message).
        group_info: [{clip_path, audio_path, total_dur, lines}] per group.
        line_to_group: {line_idx: group_idx} mapping for group-based dialogue.
        subtitle_font_size: Legacy EN subtitle font size (used when subtitle_style is None).
        subtitle_style: 字幕样式 dict（字幕样式设计器）；None → 历史行为。

    Returns:
        Path to final video.
    """
    def _cb(pct, msg):
        if progress_cb:
            progress_cb(pct, msg)

    work = Path(work_dir)
    tmp_dir = work / "tmp_segments"
    static_dir = work / "static_frames"
    vid_dir = work / "videos"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    static_dir.mkdir(parents=True, exist_ok=True)
    vid_dir.mkdir(parents=True, exist_ok=True)

    dialogue = script.get("dialogue", [])
    n = len(dialogue)
    HOOK_CLIP = clip_paths[0] if clip_paths else None
    DIALOGUE_CLIPS = clip_paths[1:] if len(clip_paths) > 1 else []

    # --- Render static frames for Ch3 ---
    if os.path.exists(scene_img):
        _cb(5, f"Rendering {n} static frames...")
        for i, line in enumerate(dialogue):
            # EN-only frame (listen_en): English + phonetic, NO Chinese yet —
            # the translation must stay hidden until the listen_zh segment
            p_en = str(static_dir / f"en_{i}.png")
            _render_static_frame(
                line.get("text", ""), line.get("phonetic", ""),
                "", scene_img, p_en, i, n)
            # EN+ZH frame (listen_zh): reveals the Traditional Chinese translation
            p = str(static_dir / f"zh_{i}.png")
            _render_static_frame(
                line.get("text", ""), line.get("phonetic", ""),
                line.get("zh", ""), scene_img, p, i, n)
        _cb(10, "Static frames done.")

    # --- Build each segment ---
    segments = []
    seg_idx = 0
    total_segs = len(timeline)
    processed_groups = set()  # track which groups have been rendered as a single segment
    skipped_segs = 0  # count skipped segments for accurate progress

    for seg in timeline:
        seg_type = seg["type"]
        duration = seg["duration"]
        audio_idx = seg.get("audio_index", 0)

        # Group-based dialogue: skip lines that belong to an already-processed group
        if seg_type == "dialogue" and group_info and line_to_group:
            gi = line_to_group.get(audio_idx)
            if gi is not None and gi in processed_groups:
                # This line's group already rendered — skip
                skipped_segs += 1
                continue
            if gi is not None:
                processed_groups.add(gi)
                # Render entire group as ONE segment (no -ss slicing)
                ginfo = group_info[gi]
                group_audio = ginfo["audio_path"]
                group_clip = ginfo["clip_path"]
                group_dur = ginfo["total_dur"]
                out_path = str(tmp_dir / f"seg_{seg_idx:03d}.mp4")
                seg_idx += 1
                # group_dur = actual concat audio duration (includes pad silence)
                # Slow down clip to match audio duration via setpts (correct direction: >1 = slower)
                fade_af = f"afade=t=in:st=0:d=0.05,afade=t=out:st={max(0, group_dur-0.05):.2f}:d=0.05"
                if group_clip and os.path.exists(group_clip) and group_audio and os.path.exists(group_audio):
                    vid_dur = _get_duration(group_clip)
                    if vid_dur > 0 and group_dur > 0 and abs(vid_dur - group_dur) > 0.01:
                        # CORRECT: group_dur/vid_dur (>1 = slow down, <1 = speed up)
                        # e.g. clip=13s, audio=14.3s → 14.3/13=1.0898 → PTS stretched → slow down
                        # fps=24 then duplicates frames evenly throughout (no freeze)
                        vf = f"setpts={group_dur/vid_dur:.4f}*PTS,{VF_NORM},fps=24"
                    else:
                        vf = f"{VF_NORM},fps=24"
                    cmd = ["ffmpeg", "-y", "-i", group_clip, "-i", group_audio,
                           "-t", f"{group_dur:.3f}", "-vf", vf,
                           "-map", "0:v:0", "-map", "1:a:0",
                           "-c:v", "libx264", "-pix_fmt", "yuv420p",
                           "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
                           "-af", fade_af,
                           out_path]
                else:
                    # clip or audio missing — silent static segment keeps the timeline intact
                    cmd = ["ffmpeg", "-y", "-loop", "1", "-i", scene_img,
                           "-f", "lavfi", "-i", "anullsrc=stereo:44100",
                           "-t", f"{group_dur:.3f}", "-vf", f"{VF_NORM},fps=24",
                           "-map", "0:v:0", "-map", "1:a:0",
                           "-c:v", "libx264", "-pix_fmt", "yuv420p",
                           "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
                           out_path]
                try:
                    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600)
                except subprocess.TimeoutExpired:
                    print(f"  FFmpeg TIMEOUT (600s) on group seg {gi}, using fallback")
                    r = None
                if r is None or r.returncode != 0 or not os.path.exists(out_path) or os.path.getsize(out_path) < 1000:
                    print(f"  FFmpeg error group seg {gi}: {r.stderr[-200:] if r else 'timeout'}")
                    # Fallback: silent segment with scene image to maintain timeline
                    fallback_cmd = ["ffmpeg", "-y", "-loop", "1", "-i", scene_img,
                                   "-f", "lavfi", "-i", "anullsrc=stereo:44100",
                                   "-t", f"{group_dur:.3f}", "-vf", f"{VF_NORM},fps=24",
                                   "-map", "0:v:0", "-map", "1:a:0",
                                   "-c:v", "libx264", "-pix_fmt", "yuv420p",
                                   "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
                                   out_path]
                    try:
                        r2 = subprocess.run(fallback_cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)
                    except subprocess.TimeoutExpired:
                        print(f"  Fallback also timed out for group seg {gi}")
                        continue
                    if r2.returncode != 0:
                        print(f"  Fallback also failed: {r2.stderr[-200:]}")
                        continue
                segments.append(out_path)
                _cb(int(seg_idx / (total_segs - skipped_segs) * 80), f"  Segment {seg_idx}/{total_segs - skipped_segs} (group {gi})")
                continue
        d_idx = seg.get("dialogue_idx", -1)
        out_path = str(tmp_dir / f"seg_{seg_idx:03d}.mp4")
        seg_idx += 1

        # Determine audio file and audio_dur
        audio_file = None
        audio_dur = seg.get("audio_dur", duration - pad)

        if seg_type == "title_card":
            audio_file = None
            audio_dur = duration
        elif seg_type == "practice_intro":
            audio_file = narration.get("practice_intro")
            if audio_file and os.path.exists(audio_file):
                audio_dur = _get_duration(audio_file)
            else:
                audio_dur = duration - pad
        elif seg_type == "dialogue":
            audio_file = normal_paths[audio_idx] if audio_idx < len(normal_paths) else None
            audio_dur = seg.get("audio_dur", duration - pad)
        elif seg_type == "listen_en":
            audio_file = normal_paths[audio_idx] if audio_idx < len(normal_paths) else None
            audio_dur = seg.get("audio_dur", duration - pad)
        elif seg_type == "listen_zh":
            audio_file = zh_paths[audio_idx] if audio_idx < len(zh_paths) and zh_paths[audio_idx] else None
            audio_dur = seg.get("audio_dur", duration - pad)
        elif seg_type == "outro":
            audio_file = narration.get("outro")
            audio_dur = seg.get("audio_dur", duration - pad)

        # Determine video source
        if seg_type in ("listen_en", "listen_zh", "practice"):
            # listen_en / practice (silence after EN reads): EN-only frame.
            # listen_zh: EN+ZH frame (translation revealed).
            # Backward compat: fall back to zh_{i}.png if en_{i}.png missing.
            if seg_type == "listen_zh":
                frame_name = f"zh_{d_idx}.png" if d_idx >= 0 else ""
            else:
                frame_name = f"en_{d_idx}.png" if d_idx >= 0 else ""
            video_src = str(static_dir / frame_name) if frame_name else scene_img
            if not os.path.exists(video_src):
                video_src = str(static_dir / f"zh_{d_idx}.png") if d_idx >= 0 else scene_img
            if not os.path.exists(video_src):
                video_src = scene_img
            is_static = True
        elif seg_type == "title_card":
            video_src = HOOK_CLIP if HOOK_CLIP and os.path.exists(HOOK_CLIP) else scene_img
            is_static = False
        elif seg_type == "practice_intro":
            video_src = HOOK_CLIP if HOOK_CLIP and os.path.exists(HOOK_CLIP) else scene_img
            is_static = False
        elif seg_type == "outro":
            video_src = HOOK_CLIP if HOOK_CLIP and os.path.exists(HOOK_CLIP) else scene_img
            is_static = False
        else:
            idx = min(audio_idx, len(DIALOGUE_CLIPS)-1) if DIALOGUE_CLIPS else 0
            video_src = DIALOGUE_CLIPS[idx] if DIALOGUE_CLIPS and DIALOGUE_CLIPS[idx] and os.path.exists(DIALOGUE_CLIPS[idx]) else (HOOK_CLIP if HOOK_CLIP and os.path.exists(HOOK_CLIP) else scene_img)
            is_static = False

        fade_af = f"afade=t=in:st=0:d=0.05,afade=t=out:st={max(0, audio_dur-0.05):.2f}:d=0.05"

        # --- Build FFmpeg command ---
        if is_static:
            # Static image: loop image + audio (PNGs are already 1280x720;
            # VF_NORM is a no-op guard against future size mismatches)
            if audio_file and os.path.exists(audio_file):
                cmd = ["ffmpeg", "-y", "-loop", "1", "-i", video_src, "-i", audio_file,
                       "-t", f"{duration:.3f}", "-map", "0:v:0", "-map", "1:a:0",
                       "-c:v", "libx264", "-pix_fmt", "yuv420p", "-vf", VF_NORM, "-r", "24",
                       "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
                       "-af", f"{fade_af},apad=whole_dur={duration:.3f}",
                       out_path]
            else:
                cmd = ["ffmpeg", "-y", "-loop", "1", "-i", video_src,
                       "-f", "lavfi", "-i", "anullsrc=stereo:44100",
                       "-t", f"{duration:.3f}", "-map", "0:v:0", "-map", "1:a:0",
                       "-c:v", "libx264", "-pix_fmt", "yuv420p", "-vf", VF_NORM, "-r", "24",
                       "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
                       out_path]

        elif seg_type == "title_card":
            # Title card: video clip + title overlay
            title_en = seg.get("subtitle_en", "")
            title_zh = seg.get("subtitle_zh", "")
            title_overlay = str(static_dir / "title_overlay.png")
            _render_title_card(title_en, title_zh, "", scene_img, title_overlay)
            vid_dur = _get_duration(video_src) if os.path.exists(video_src) else 0
            vf = f"setpts={duration/vid_dur:.4f}*PTS,{VF_NORM},fps=24" if vid_dur > 0 else f"{VF_NORM},fps=24"
            fade_af_title = f"afade=t=in:st=0:d=0.05,afade=t=out:st={max(0, duration-0.05):.2f}:d=0.05"
            if _has_audio(video_src):
                cmd = ["ffmpeg", "-y", "-i", video_src, "-i", title_overlay,
                       "-t", f"{duration:.3f}",
                       "-filter_complex", f"[0:v]{vf}[bg];[bg][1:v]overlay=0:0[v]",
                       "-map", "[v]", "-map", "0:a:0",
                       "-c:v", "libx264", "-pix_fmt", "yuv420p",
                       "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
                       "-af", fade_af_title,
                       out_path]
            else:
                cmd = ["ffmpeg", "-y", "-i", video_src, "-i", title_overlay,
                       "-f", "lavfi", "-i", "anullsrc=stereo:44100",
                       "-t", f"{duration:.3f}",
                       "-filter_complex", f"[0:v]{vf}[bg];[bg][1:v]overlay=0:0[v]",
                       "-map", "[v]", "-map", "2:a",
                       "-c:v", "libx264", "-pix_fmt", "yuv420p",
                       "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
                       out_path]

        elif seg_type == "practice_intro":
            # Practice intro: video clip + text overlay + narration ONLY (no video audio)
            intro_en = seg.get("subtitle_en", "")
            intro_zh = seg.get("subtitle_zh", "")
            intro_overlay = str(static_dir / "practice_intro_overlay.png")
            _render_practice_intro(intro_en, intro_zh, scene_img, intro_overlay)
            vid_dur = _get_duration(video_src) if os.path.exists(video_src) else 0
            vf = f"setpts={audio_dur/vid_dur:.4f}*PTS,{VF_NORM},fps=24" if vid_dur > 0 and audio_dur > 0 else f"{VF_NORM},fps=24"
            out_dur = audio_dur + pad
            fade_af_pi = f"afade=t=in:st=0:d=0.05,afade=t=out:st={max(0, audio_dur-0.05):.2f}:d=0.05"
            narration_audio = narration.get("practice_intro")
            if narration_audio and os.path.exists(narration_audio):
                # Use narration audio ONLY (discard video's audio entirely)
                # Audio filters go INSIDE filter_complex (not -af, which conflicts with filter_complex)
                cmd = ["ffmpeg", "-y", "-i", video_src, "-i", narration_audio, "-i", intro_overlay,
                       "-t", f"{out_dur:.3f}",
                       "-filter_complex", f"[0:v]{vf}[bg];[bg][2:v]overlay=0:0[v];[1:a]afade=t=in:st=0:d=0.05,afade=t=out:st={max(0, audio_dur-0.05):.2f}:d=0.05,apad=whole_dur={out_dur:.3f}[a]",
                       "-map", "[v]", "-map", "[a]",
                       "-c:v", "libx264", "-pix_fmt", "yuv420p",
                       "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
                       out_path]
            else:
                # No narration: use silence (no video audio)
                cmd = ["ffmpeg", "-y", "-i", video_src, "-i", intro_overlay,
                       "-f", "lavfi", "-i", "anullsrc=stereo:44100",
                       "-t", f"{out_dur:.3f}",
                       "-filter_complex", f"[0:v]{vf}[bg];[bg][1:v]overlay=0:0[v]",
                       "-map", "[v]", "-map", "2:a",
                       "-c:v", "libx264", "-pix_fmt", "yuv420p",
                       "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
                       out_path]

        elif seg_type == "outro":
            # Outro: video clip + text overlay + narration ONLY (no video audio)
            outro_en = seg.get("subtitle_en", "")
            outro_zh = seg.get("subtitle_zh", "")
            outro_overlay = str(static_dir / "outro_overlay.png")
            _render_practice_intro(outro_en, outro_zh, scene_img, outro_overlay)
            vid_dur = _get_duration(video_src) if os.path.exists(video_src) else 0
            out_dur = audio_dur + pad
            vf = f"setpts={audio_dur/vid_dur:.4f}*PTS,{VF_NORM},fps=24" if vid_dur > 0 and audio_dur > 0 else f"{VF_NORM},fps=24"
            fade_af_outro = f"afade=t=in:st=0:d=0.05,afade=t=out:st={max(0, audio_dur-0.05):.2f}:d=0.05"
            outro_audio = narration.get("outro")
            if outro_audio and os.path.exists(outro_audio):
                # Use narration audio ONLY (discard video's audio entirely)
                # Audio filters go INSIDE filter_complex (not -af, which conflicts with filter_complex)
                cmd = ["ffmpeg", "-y", "-i", video_src, "-i", outro_audio, "-i", outro_overlay,
                       "-t", f"{out_dur:.3f}",
                       "-filter_complex", f"[0:v]{vf}[bg];[bg][2:v]overlay=0:0[v];[1:a]afade=t=in:st=0:d=0.05,afade=t=out:st={max(0, audio_dur-0.05):.2f}:d=0.05,apad=whole_dur={out_dur:.3f}[a]",
                       "-map", "[v]", "-map", "[a]",
                       "-c:v", "libx264", "-pix_fmt", "yuv420p",
                       "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
                       out_path]
            else:
                # No narration: use silence (no video audio)
                cmd = ["ffmpeg", "-y", "-i", video_src, "-i", outro_overlay,
                       "-f", "lavfi", "-i", "anullsrc=stereo:44100",
                       "-t", f"{out_dur:.3f}",
                       "-filter_complex", f"[0:v]{vf}[bg];[bg][1:v]overlay=0:0[v]",
                       "-map", "[v]", "-map", "2:a",
                       "-c:v", "libx264", "-pix_fmt", "yuv420p",
                       "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
                       out_path]

        else:
            # Dialogue (no-grouping fallback): video clip + audio
            # 方案 B grouped dialogue is handled above; this is the non-grouped path.
            vid_dur = _get_duration(video_src) if os.path.exists(video_src) else 0
            if vid_dur > 0 and audio_dur > 0:
                vf = f"setpts={audio_dur/vid_dur:.4f}*PTS,{VF_NORM},fps=24"
            else:
                vf = f"{VF_NORM},fps=24"
            if audio_file and os.path.exists(audio_file):
                cmd = ["ffmpeg", "-y", "-i", video_src, "-i", audio_file,
                       "-t", f"{duration:.3f}", "-vf", vf,
                       "-map", "0:v:0", "-map", "1:a:0",
                       "-c:v", "libx264", "-pix_fmt", "yuv420p",
                       "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
                       "-af", f"{fade_af},apad=whole_dur={duration:.3f}",
                       out_path]
            else:
                cmd = ["ffmpeg", "-y", "-i", video_src,
                       "-f", "lavfi", "-i", "anullsrc=stereo:44100",
                       "-t", f"{duration:.3f}", "-vf", vf,
                       "-map", "0:v:0", "-map", "1:a:0",
                       "-c:v", "libx264", "-pix_fmt", "yuv420p",
                       "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
                       out_path]

        try:
            r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)
        except subprocess.TimeoutExpired:
            print(f"  FFmpeg TIMEOUT (300s) on seg {seg_idx} ({seg_type}), using fallback")
            r = None
        if r is None or r.returncode != 0 or not os.path.exists(out_path) or os.path.getsize(out_path) < 1000:
            print(f"  FFmpeg error seg {seg_idx}: {r.stderr[-200:] if r else 'timeout'}")
            # Fallback: silent segment with scene image
            fallback_cmd = ["ffmpeg", "-y", "-loop", "1", "-i", scene_img,
                           "-f", "lavfi", "-i", "anullsrc=stereo:44100",
                           "-t", f"{duration:.3f}", "-vf", f"{VF_NORM},fps=24",
                           "-map", "0:v:0", "-map", "1:a:0",
                           "-c:v", "libx264", "-pix_fmt", "yuv420p",
                           "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
                           out_path]
            try:
                r2 = subprocess.run(fallback_cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)
            except subprocess.TimeoutExpired:
                print(f"  Fallback also timed out, skipping segment {seg_idx}")
                continue
            if r2.returncode != 0 or not os.path.exists(out_path) or os.path.getsize(out_path) < 1000:
                print(f"  Fallback also failed, skipping segment: {r2.stderr[-200:]}")
                continue
        segments.append(out_path)
        _cb(int(seg_idx / (total_segs - skipped_segs) * 80),
            f"  Segment {seg_idx}/{total_segs - skipped_segs} ({seg_type})")

    # --- Concat all segments ---
    _cb(80, "Concatenating segments...")
    no_sub = str(vid_dir / "final_no_sub.mp4")
    concat_segments(segments, no_sub, tmp_dir=tmp_dir)

    # --- Burn subtitles via Pillow overlay ---
    _cb(90, "Burning subtitles (Pillow overlay)...")
    final_path = burn_subtitles(no_sub, timeline, script, str(work), srt_dir, pad, _cb, show_zh=show_zh, en_font_size=subtitle_font_size, zh_font_size=int(subtitle_font_size * 0.85), style=subtitle_style)

    # Cleanup
    shutil.rmtree(tmp_dir, ignore_errors=True)

    # Final loudnorm pass
    _cb(95, "Final loudnorm pass (normalize volume)...")
    apply_final_loudnorm(final_path, str(vid_dir))
    size_mb = os.path.getsize(final_path) / (1024 * 1024)
    _cb(100, f"Listening video done: {final_path} ({size_mb:.1f}MB)")

    return final_path


def compose_image(
    work_dir: str,
    dialogue_images: list[str],
    pose_images: list[list[str]],
    background_img: str,
    timeline: list[dict],
    script: dict,
    narration: dict,
    normal_paths: list[str],
    zh_paths: list[str],
    scene_img: str,
    srt_dir: str,
    pad: float = 0.4,
    progress_cb=None,
    animation: str = "landing",
    subtitle_font_size: int = 60,
    subtitle_style: dict | None = None,
    show_zh: bool = True,
    target_w: int = TARGET_W,
    target_h: int = TARGET_H,
) -> str:
    """Compose final listening practice video using images (no video clips).

    Unified function merging compose_static + compose_stop_motion. The
    *animation* parameter controls how dialogue segments are rendered:

    - ``"none"``: static image per line, no movement (former ``static`` mode).
    - ``"landing"``: static image per line with FFmpeg landing-transform
      micro-animation (scale decay + pan + sine bounce, 0.3s) (former
      ``static_animated`` mode).
    - ``"stop_motion"``: multi-pose character animation via PIL frame
      rendering with optical-flow morphing and landing transforms
      (former ``stop_motion`` mode). Requires *pose_images* and
      *background_img*.

    Non-dialogue segments (title_card, practice, outro, etc.) are identical
    across all animation modes.

    Args:
        work_dir: Working directory for temp files and output.
        dialogue_images: Per-line dialogue image paths (used when animation
            is "none" or "landing").
        pose_images: Per-line list of pose image paths (used when animation
            is "stop_motion"). Pass ``[]`` when not used.
        background_img: Background image for stop_motion compositing
            (usually same as scene_img).
        timeline: Timeline segments from build_listening_timeline.
        script: Lesson script dict.
        narration: {"intro": path, "outro": path, "practice_intro": path}.
        normal_paths: English dialogue audio paths.
        zh_paths: Chinese dialogue audio paths.
        scene_img: Scene background image path.
        srt_dir: Directory for SRT file (cwd for FFmpeg subtitle burn).
        pad: Audio pad between segments (seconds).
        progress_cb: callback(percent, message).
        animation: "none", "landing", or "stop_motion".
        target_w / target_h: Output canvas size. Defaults to 1280x720;
            shorts mode passes 1080x1920 (vertical). stop_motion always
            renders at the default canvas.

    Returns:
        Path to final video.
    """
    def _cb(pct, msg):
        if progress_cb:
            progress_cb(pct, msg)

    import tempfile

    work = Path(work_dir)
    tmp_dir = work / "tmp_segments"
    static_dir = work / "static_frames"
    vid_dir = work / "videos"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    static_dir.mkdir(parents=True, exist_ok=True)
    vid_dir.mkdir(parents=True, exist_ok=True)

    dialogue = script.get("dialogue", [])
    n = len(dialogue)

    is_stop_motion = (animation == "stop_motion")

    # Resolution-aware normalization filter + canvas for this compose call
    # (defaults are the module constants; shorts passes 1080x1920)
    tw, th = target_w, target_h
    vf_norm = (
        f"scale={tw}:{th}:force_original_aspect_ratio=decrease,"
        f"pad={tw}:{th}:(ow-iw)/2:(oh-ih)/2"
    )

    # Skip Ch3 static frame rendering when the timeline has no listen
    # segments (shorts timeline has none — saves n×2 Pillow renders)
    has_listen_segs = any(seg.get("type", "").startswith("listen_")
                          for seg in timeline)

    # --- Stop-motion: preprocess poses + subtitle overlays ---
    processed_poses: list[list] = []
    subtitle_overlays: dict[int, object] = {}
    frames_dir = None
    bg_img_rgba = None
    if is_stop_motion:
        from stop_motion import remove_bg, normalize_pose, _render_subtitle_overlay
        from PIL import Image as _PILImage

        frames_dir = Path(tempfile.gettempdir()) / f"sm_frames_{work.name}"
        frames_dir.mkdir(parents=True, exist_ok=True)

        _cb(2, "Loading background...")
        bg_img_rgba = _PILImage.open(background_img).convert("RGBA").resize(
            (TARGET_W, TARGET_H))

        _cb(5, f"Processing {sum(len(p) for p in pose_images)} character poses...")
        for i, line_poses in enumerate(pose_images):
            line_processed = []
            for j, p_path in enumerate(line_poses):
                if not os.path.exists(p_path):
                    print(f"  WARNING: pose image {p_path} not found, skipping")
                    continue
                raw = _PILImage.open(p_path)
                alpha = remove_bg(raw)
                normalized = normalize_pose(alpha)
                line_processed.append(normalized)
            if not line_processed:
                line_processed = [_PILImage.new("RGBA", (TARGET_W, TARGET_H), (0, 0, 0, 0))]
            processed_poses.append(line_processed)
        _cb(8, "Pose processing done.")

        # Pre-render subtitle overlays for dialogue segments
        for i, line in enumerate(dialogue):
            en = line.get("text", "")
            zh = line.get("zh", "")
            subtitle_overlays[i] = _render_subtitle_overlay(en, zh)

    # --- Render static frames for Ch3 (same as compose_listening) ---
    if os.path.exists(scene_img) and has_listen_segs:
        _cb(10 if is_stop_motion else 5, f"Rendering {n} static frames...")
        for i, line in enumerate(dialogue):
            p_en = str(static_dir / f"en_{i}.png")
            _render_static_frame(
                line.get("text", ""), line.get("phonetic", ""),
                "", scene_img, p_en, i, n, w=tw, h=th)
            p = str(static_dir / f"zh_{i}.png")
            _render_static_frame(
                line.get("text", ""), line.get("phonetic", ""),
                line.get("zh", ""), scene_img, p, i, n, w=tw, h=th)
        _cb(12 if is_stop_motion else 10, "Static frames done.")

    # --- Build each segment ---
    segments = []
    seg_idx = 0
    total_segs = len(timeline)

    for seg in timeline:
        seg_type = seg["type"]
        duration = seg["duration"]
        audio_idx = seg.get("audio_index", 0)
        d_idx = seg.get("dialogue_idx", -1)
        out_path = str(tmp_dir / f"seg_{seg_idx:03d}.mp4")
        seg_idx += 1

        # Determine audio file and audio_dur
        audio_file = None
        audio_dur = seg.get("audio_dur", duration - pad)

        if seg_type == "title_card":
            audio_file = None
            audio_dur = duration
        elif seg_type == "practice_intro":
            audio_file = narration.get("practice_intro")
            if audio_file and os.path.exists(audio_file):
                audio_dur = _get_duration(audio_file)
            else:
                audio_dur = duration - pad
        elif seg_type == "dialogue":
            audio_file = normal_paths[audio_idx] if audio_idx < len(normal_paths) else None
            audio_dur = seg.get("audio_dur", duration - pad)
        elif seg_type == "listen_en":
            audio_file = normal_paths[audio_idx] if audio_idx < len(normal_paths) else None
            audio_dur = seg.get("audio_dur", duration - pad)
        elif seg_type == "listen_zh":
            audio_file = zh_paths[audio_idx] if audio_idx < len(zh_paths) and zh_paths[audio_idx] else None
            audio_dur = seg.get("audio_dur", duration - pad)
        elif seg_type == "outro":
            audio_file = narration.get("outro")
            audio_dur = seg.get("audio_dur", duration - pad)

        fade_af = f"afade=t=in:st=0:d=0.05,afade=t=out:st={max(0, audio_dur-0.05):.2f}:d=0.05"

        # --- Build FFmpeg command based on segment type ---
        if seg_type in ("listen_en", "listen_zh", "practice"):
            # Static Pillow frame + audio/silence (same as compose_listening)
            if seg_type == "listen_zh":
                frame_name = f"zh_{d_idx}.png" if d_idx >= 0 else ""
            else:
                frame_name = f"en_{d_idx}.png" if d_idx >= 0 else ""
            video_src = str(static_dir / frame_name) if frame_name else scene_img
            if not os.path.exists(video_src):
                video_src = str(static_dir / f"zh_{d_idx}.png") if d_idx >= 0 else scene_img
            if not os.path.exists(video_src):
                video_src = scene_img

            if audio_file and os.path.exists(audio_file):
                cmd = ["ffmpeg", "-y", "-loop", "1", "-i", video_src, "-i", audio_file,
                       "-t", f"{duration:.3f}", "-map", "0:v:0", "-map", "1:a:0",
                       "-c:v", "libx264", "-pix_fmt", "yuv420p", "-vf", vf_norm, "-r", "24",
                       "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
                       "-af", f"{fade_af},apad=whole_dur={duration:.3f}",
                       out_path]
            else:
                cmd = ["ffmpeg", "-y", "-loop", "1", "-i", video_src,
                       "-f", "lavfi", "-i", "anullsrc=stereo:44100",
                       "-t", f"{duration:.3f}", "-map", "0:v:0", "-map", "1:a:0",
                       "-c:v", "libx264", "-pix_fmt", "yuv420p", "-vf", vf_norm, "-r", "24",
                       "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
                       out_path]

        elif seg_type == "title_card":
            # Static scene image + title overlay (transparent PNG) + silence
            # scene_img may be at frontier's native size — normalize before
            # overlaying the title PNG, else text gets cropped
            title_en = seg.get("subtitle_en", "")
            title_zh = seg.get("subtitle_zh", "")
            title_overlay = str(static_dir / "title_overlay.png")
            _render_title_card(title_en, title_zh, "", scene_img, title_overlay,
                               w=tw, h=th)
            cmd = ["ffmpeg", "-y", "-loop", "1", "-i", scene_img,
                   "-i", title_overlay,
                   "-f", "lavfi", "-i", "anullsrc=stereo:44100",
                   "-t", f"{duration:.3f}",
                   "-filter_complex", f"[0:v]{vf_norm}[bg];[bg][1:v]overlay=0:0[v]",
                   "-map", "[v]", "-map", "2:a",
                   "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "24",
                   "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
                   out_path]

        elif seg_type == "practice_intro":
            # Static scene image + text overlay + narration audio
            intro_en = seg.get("subtitle_en", "")
            intro_zh = seg.get("subtitle_zh", "")
            intro_overlay = str(static_dir / "practice_intro_overlay.png")
            _render_practice_intro(intro_en, intro_zh, scene_img, intro_overlay,
                                   w=tw, h=th)
            out_dur = audio_dur + pad
            narration_audio = narration.get("practice_intro")
            if narration_audio and os.path.exists(narration_audio):
                cmd = ["ffmpeg", "-y", "-loop", "1", "-i", scene_img,
                       "-i", narration_audio, "-i", intro_overlay,
                       "-t", f"{out_dur:.3f}",
                       "-filter_complex",
                       f"[0:v]{vf_norm}[bg];[bg][2:v]overlay=0:0[v];"
                       f"[1:a]afade=t=in:st=0:d=0.05,afade=t=out:st={max(0, audio_dur-0.05):.2f}:d=0.05,apad=whole_dur={out_dur:.3f}[a]",
                       "-map", "[v]", "-map", "[a]",
                       "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "24",
                       "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
                       out_path]
            else:
                cmd = ["ffmpeg", "-y", "-loop", "1", "-i", scene_img,
                       "-i", intro_overlay,
                       "-f", "lavfi", "-i", "anullsrc=stereo:44100",
                       "-t", f"{out_dur:.3f}",
                       "-filter_complex", f"[0:v]{vf_norm}[bg];[bg][1:v]overlay=0:0[v]",
                       "-map", "[v]", "-map", "2:a",
                       "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "24",
                       "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
                       out_path]

        elif seg_type == "outro":
            # Static scene image + text overlay + narration audio
            outro_en = seg.get("subtitle_en", "")
            outro_zh = seg.get("subtitle_zh", "")
            outro_overlay = str(static_dir / "outro_overlay.png")
            _render_practice_intro(outro_en, outro_zh, scene_img, outro_overlay,
                                   w=tw, h=th)
            out_dur = audio_dur + pad
            outro_audio = narration.get("outro")
            if outro_audio and os.path.exists(outro_audio):
                cmd = ["ffmpeg", "-y", "-loop", "1", "-i", scene_img,
                       "-i", outro_audio, "-i", outro_overlay,
                       "-t", f"{out_dur:.3f}",
                       "-filter_complex",
                       f"[0:v]{vf_norm}[bg];[bg][2:v]overlay=0:0[v];"
                       f"[1:a]afade=t=in:st=0:d=0.05,afade=t=out:st={max(0, audio_dur-0.05):.2f}:d=0.05,apad=whole_dur={out_dur:.3f}[a]",
                       "-map", "[v]", "-map", "[a]",
                       "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "24",
                       "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
                       out_path]
            else:
                cmd = ["ffmpeg", "-y", "-loop", "1", "-i", scene_img,
                       "-i", outro_overlay,
                       "-f", "lavfi", "-i", "anullsrc=stereo:44100",
                       "-t", f"{out_dur:.3f}",
                       "-filter_complex", f"[0:v]{vf_norm}[bg];[bg][1:v]overlay=0:0[v]",
                       "-map", "[v]", "-map", "2:a",
                       "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "24",
                       "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
                       out_path]

        else:
            # Dialogue segment — branch on animation mode
            if is_stop_motion and d_idx >= 0 and d_idx < len(processed_poses):
                # --- Stop-motion: PIL frame rendering + optical flow ---
                from stop_motion import _render_dialogue_segment as _render_sm
                line_poses = processed_poses[d_idx]
                sub_overlay = subtitle_overlays.get(d_idx)
                out_frames_dir = frames_dir / f"dialogue_{d_idx}"
                out_frames_dir.mkdir(parents=True, exist_ok=True)

                _render_sm(bg_img_rgba, line_poses, sub_overlay,
                           duration, audio_dur, 24, out_frames_dir, d_idx)

                frame_pattern = str(out_frames_dir / "frame-%04d.png")
                if audio_file and os.path.exists(audio_file):
                    cmd = [
                        "ffmpeg", "-y",
                        "-framerate", "24", "-i", frame_pattern,
                        "-i", audio_file,
                        "-t", f"{duration:.3f}", "-map", "0:v:0", "-map", "1:a:0",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "24",
                        "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
                        "-af", f"{fade_af},apad=whole_dur={duration:.3f}",
                        out_path,
                    ]
                else:
                    cmd = [
                        "ffmpeg", "-y",
                        "-framerate", "24", "-i", frame_pattern,
                        "-f", "lavfi", "-i", "anullsrc=stereo:44100",
                        "-t", f"{duration:.3f}", "-map", "0:v:0", "-map", "1:a:0",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "24",
                        "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
                        out_path,
                    ]
                # Run now (don't defer to common try/continue below)
                try:
                    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600)
                    if r.returncode != 0:
                        print(f"  FFmpeg error (stop_motion dialogue {d_idx}): {r.stderr[-200:]}")
                except subprocess.TimeoutExpired:
                    print(f"  FFmpeg timeout on stop_motion dialogue {d_idx}")
                except Exception as e:
                    print(f"  Error on stop_motion dialogue {d_idx}: {e}")
                # Cleanup frames to save disk
                shutil.rmtree(out_frames_dir, ignore_errors=True)

                # Skip the common FFmpeg try/except below
                if os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
                    segments.append(out_path)
                    _cb(int(seg_idx / total_segs * 80),
                        f"  Segment {seg_idx}/{total_segs} (dialogue {d_idx})")
                continue

            # --- None / Landing: static image per line via FFmpeg ---
            idx = min(audio_idx, len(dialogue_images) - 1) if dialogue_images else 0
            d_img = dialogue_images[idx] if dialogue_images and idx < len(dialogue_images) else scene_img
            if not os.path.exists(d_img):
                d_img = scene_img

            # Build VF: landing-transform micro-animation when animation=="landing",
            # else plain vf_norm.  Landing transform: scale 1.04x → crop to
            # target canvas with time-varying pan (±14px x, ±10px sine y,
            # 0.3s decay, alternating direction per line) — gives
            # stop-motion "settle" feel.
            if animation == "landing":
                _dir = 1 if (idx % 2 == 0) else -1
                _ld = 0.3  # landing duration (seconds)
                _up_w = int(tw * 1.04)
                _up_h = int(th * 1.04)
                vf_dialogue = (
                    f"scale={tw}:{th}:force_original_aspect_ratio=decrease,"
                    f"pad={tw}:{th}:(ow-iw)/2:(oh-ih)/2,"
                    f"scale={_up_w}:{_up_h},"
                    f"crop=w={tw}:h={th}:"
                    f"x='(iw-{tw})/2+{_dir}*14*max(0,1-t/{_ld})':"
                    f"y='(ih-{th})/2-10*sin(min(1,t/{_ld})*PI)'"
                )
            else:
                vf_dialogue = vf_norm

            if audio_file and os.path.exists(audio_file):
                cmd = ["ffmpeg", "-y", "-loop", "1", "-i", d_img, "-i", audio_file,
                       "-t", f"{duration:.3f}", "-map", "0:v:0", "-map", "1:a:0",
                       "-c:v", "libx264", "-pix_fmt", "yuv420p", "-vf", vf_dialogue, "-r", "24",
                       "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
                       "-af", f"{fade_af},apad=whole_dur={duration:.3f}",
                       out_path]
            else:
                cmd = ["ffmpeg", "-y", "-loop", "1", "-i", d_img,
                       "-f", "lavfi", "-i", "anullsrc=stereo:44100",
                       "-t", f"{duration:.3f}", "-map", "0:v:0", "-map", "1:a:0",
                       "-c:v", "libx264", "-pix_fmt", "yuv420p", "-vf", vf_dialogue, "-r", "24",
                       "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
                       out_path]

        try:
            r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)
        except subprocess.TimeoutExpired:
            print(f"  FFmpeg TIMEOUT (300s) on seg {seg_idx} ({seg_type}), using fallback")
            r = None
        if r is None or r.returncode != 0 or not os.path.exists(out_path) or os.path.getsize(out_path) < 1000:
            print(f"  FFmpeg error seg {seg_idx}: {r.stderr[-200:] if r else 'timeout'}")
            # Fallback: silent segment with scene image
            fallback_cmd = ["ffmpeg", "-y", "-loop", "1", "-i", scene_img,
                           "-f", "lavfi", "-i", "anullsrc=stereo:44100",
                           "-t", f"{duration:.3f}", "-map", "0:v:0", "-map", "1:a:0",
                           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-vf", vf_norm, "-r", "24",
                           "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
                           out_path]
            try:
                r2 = subprocess.run(fallback_cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)
            except subprocess.TimeoutExpired:
                print(f"  Fallback also timed out, skipping segment {seg_idx}")
                continue
            if r2.returncode != 0 or not os.path.exists(out_path) or os.path.getsize(out_path) < 1000:
                print(f"  Fallback also failed, skipping segment: {r2.stderr[-200:]}")
                continue
        segments.append(out_path)
        _cb(int(seg_idx / total_segs * 80),
            f"  Segment {seg_idx}/{total_segs} ({seg_type})")

    # --- Concat all segments ---
    _cb(80, "Concatenating segments...")
    no_sub = str(vid_dir / "final_no_sub.mp4")
    concat_segments(segments, no_sub, tmp_dir=tmp_dir)

    # --- Burn subtitles via Pillow overlay (same as compose_listening) ---
    _cb(90, "Burning subtitles (Pillow overlay)...")
    final_path = burn_subtitles(no_sub, timeline, script, str(work), srt_dir, pad, _cb, show_zh=show_zh, en_font_size=subtitle_font_size, zh_font_size=int(subtitle_font_size * 0.85), style=subtitle_style)

    # Cleanup
    shutil.rmtree(tmp_dir, ignore_errors=True)
    if frames_dir is not None:
        shutil.rmtree(frames_dir, ignore_errors=True)

    # Final loudnorm pass
    _cb(95, "Final loudnorm pass (normalize volume)...")
    apply_final_loudnorm(final_path, str(vid_dir))
    size_mb = os.path.getsize(final_path) / (1024 * 1024)
    _cb(100, f"Image listening video done: {final_path} ({size_mb:.1f}MB)")

    return final_path
