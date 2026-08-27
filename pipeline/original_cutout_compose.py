"""Original Cutout compose — 4-chapter timeline + quest-style character cutout rendering.

Combines:
- Original 4-chapter timeline (host welcome/hook_intro → dialogue → practice_intro
  → Ch3 practice → host outro)，开场/结尾为 quest 式主持人定格动画
- Quest-style stop-motion character cutout animation for dialogue segments
  (per-character pose atlas, audio-driven pose switching,
   optical flow morphing, landing transforms, multi-character on screen)
- Original static Pillow frames for Ch3 practice segments (listen_en/listen_zh)

Reuses:
- quest.video_compose_quest._render_sm_segment — dialogue cutout renderer
- video_compose._render_static_frame / _render_title_card / _render_practice_intro — static frames
- media_utils — concat, subtitle burn, loudnorm, FFmpeg helpers
"""
import os
import sys
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

_PARENT = str(Path(__file__).parent.resolve())
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from video_compose import (
    _render_static_frame,
    _render_title_card,
    _render_practice_intro,
    _render_intro_cards,
    _build_intro_overlay_chain,
)
from media_utils import (
    VF_NORM,
    get_duration as _get_duration,
    concat_segments, burn_subtitles, apply_final_loudnorm,
    make_silent_fallback_cmd,
)


def _run_ffmpeg(cmd: list[str], label: str, out_path: str,
                scene_img: str, duration: float) -> None:
    """Run FFmpeg command with static-image fallback."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=600)
        if r.returncode != 0:
            print(f"  [Cutout] FFmpeg error ({label}): {r.stderr[-200:]}")
    except subprocess.TimeoutExpired:
        print(f"  [Cutout] FFmpeg timeout on {label}")
    except Exception as e:
        print(f"  [Cutout] Error on {label}: {e}")
    if not os.path.exists(out_path) or os.path.getsize(out_path) < 1000:
        # Fallback: silent static segment
        print(f"  [Cutout] Fallback for {label}")
        fb = make_silent_fallback_cmd(scene_img, duration, out_path)
        try:
            subprocess.run(fb, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=300)
        except Exception:
            pass


def _render_host_segment(seg_type: str, seg_idx: int, host_poses: list[str] | None,
                         host_bg: str, audio_file: str | None, out_path: str,
                         duration: float, sm_root: Path, render_fps: int,
                         fade_af: str, stop_check=None) -> bool:
    """Quest 式主持人段：姿势图集抠像定格动画（welcome/hook_intro/outro 出镜）。

    Returns True if rendered successfully.
    """
    h_poses = host_poses or []
    if not h_poses:
        return False
    char_layers = [{"poses": h_poses, "is_speaker": True}]
    frames_dir = sm_root / f"{seg_type}_{seg_idx}"
    direction = 1 if seg_idx % 2 == 0 else -1

    from quest.video_compose_quest import _render_sm_segment

    return _render_sm_segment(
        char_layers, host_bg, audio_file, out_path, duration,
        frames_dir, sm_root,
        render_fps=render_fps,
        seed=hash(seg_type) % 1000 + seg_idx,
        direction=direction, fade_af=fade_af,
        stop_check=stop_check,
    )


def _prepare_segment(
    seg_idx: int,
    seg: dict,
    timeline: list[dict],
    dialogue: list[dict],
    narration: dict,
    normal_paths: list[str],
    zh_paths: list[str],
    char_pose_map: dict[str, list[str]],
    scene_bgs: list[str],
    scene_img: str,
    pad: float,
    render_fps: int,
    tmp_dir: Path,
    sm_root: Path,
    static_dir: Path,
    host_poses: list[str] | None = None,
    host_bg: str = "",
    stop_check=None,
) -> tuple[str | None, str]:
    """Prepare params and render a single segment.

    Returns (out_path or None, seg_type).
    """
    seg_type = seg["type"]
    duration = seg["duration"]
    audio_idx = seg.get("audio_index", 0)
    d_idx = seg.get("dialogue_idx", -1)
    out_path = str(tmp_dir / f"seg_{seg_idx:03d}.mp4")

    # Skip if already rendered (resume support)
    if os.path.exists(out_path) and os.path.getsize(out_path) >= 1000:
        return out_path, seg_type

    audio_file = None
    audio_dur = seg.get("audio_dur", duration - pad)

    if seg_type == "dialogue":
        audio_file = normal_paths[audio_idx] if audio_idx < len(normal_paths) else None
    elif seg_type in ("welcome", "hook_intro"):
        audio_file = narration.get("welcome" if seg_type == "welcome" else "hook")
    elif seg_type == "listen_en":
        audio_file = normal_paths[audio_idx] if audio_idx < len(normal_paths) else None
    elif seg_type == "listen_zh":
        audio_file = zh_paths[audio_idx] if audio_idx < len(zh_paths) and zh_paths[audio_idx] else None
    elif seg_type == "practice_intro":
        audio_file = narration.get("practice_intro")
    elif seg_type == "outro":
        audio_file = narration.get("outro")
    elif seg_type == "title_card":
        audio_file = narration.get("intro")

    fade_af = f"afade=t=in:st=0:d=0.05,afade=t=out:st={max(0, audio_dur-0.05):.2f}:d=0.05"

    # --- Quest-style host segments (Ch1 welcome / Ch2 hook) ---
    if seg_type in ("welcome", "hook_intro"):
        bg_for_host = host_bg or scene_img
        if not _render_host_segment(seg_type, seg_idx, host_poses, bg_for_host,
                                    audio_file, out_path, duration, sm_root,
                                    render_fps, fade_af, stop_check=stop_check):
            # 无图集或渲染失败 → 静态演播室底图回退
            if audio_file and os.path.exists(audio_file):
                cmd = ["ffmpeg", "-y", "-loop", "1", "-i", bg_for_host,
                       "-i", audio_file,
                       "-t", f"{duration:.3f}", "-map", "0:v:0", "-map", "1:a:0",
                       "-c:v", "libx264", "-pix_fmt", "yuv420p",
                       "-vf", f"{VF_NORM},fps=25",
                       "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
                       "-af", f"{fade_af},apad=whole_dur={duration:.3f}",
                       out_path]
            else:
                cmd = ["ffmpeg", "-y", "-loop", "1", "-i", bg_for_host,
                       "-f", "lavfi", "-i", "anullsrc=stereo:44100",
                       "-t", f"{duration:.3f}", "-map", "0:v:0", "-map", "1:a:0",
                       "-c:v", "libx264", "-pix_fmt", "yuv420p",
                       "-vf", f"{VF_NORM},fps=25",
                       "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
                       out_path]
            _run_ffmpeg(cmd, seg_type, out_path, scene_img, duration)

    # --- Dialogue segment: quest-style character cutout stop-motion ---
    elif seg_type == "dialogue":
        line_data = dialogue[audio_idx] if audio_idx < len(dialogue) else {}
        speaker = line_data.get("speaker", "char_a")
        other = "char_b" if speaker == "char_a" else "char_a"

        char_layers = []
        if speaker in char_pose_map and char_pose_map[speaker]:
            char_layers.append({"poses": char_pose_map[speaker], "is_speaker": True})
        if other in char_pose_map and char_pose_map[other]:
            char_layers.append({"poses": char_pose_map[other], "is_speaker": False})

        if not char_layers:
            # No pose images — fallback to static scene
            _run_ffmpeg(
                ["ffmpeg", "-y", "-loop", "1", "-i", scene_img,
                 "-i", audio_file] if audio_file else
                ["ffmpeg", "-y", "-loop", "1", "-i", scene_img,
                 "-f", "lavfi", "-i", "anullsrc=stereo:44100"],
                f"dialogue_{audio_idx}", out_path, scene_img, duration)
            return out_path if os.path.exists(out_path) else None, seg_type

        # Scene background rotation
        dialogue_seg_count = sum(1 for s in timeline[:timeline.index(seg)] if s["type"] == "dialogue")
        n_scene_bgs = max(1, len(scene_bgs))
        bg_idx = (dialogue_seg_count // 5) % n_scene_bgs
        line_bg = scene_bgs[bg_idx]

        frames_dir = sm_root / f"dialogue_{audio_idx}"
        direction = 1 if audio_idx % 2 == 0 else -1

        # Import quest's stop-motion renderer
        from quest.video_compose_quest import _render_sm_segment

        success = _render_sm_segment(
            char_layers, line_bg, audio_file, out_path, duration,
            frames_dir, sm_root,
            render_fps=render_fps,
            seed=audio_idx * 7 + 13,
            direction=direction, fade_af=fade_af,
            stop_check=stop_check,
        )
        if not success:
            _run_ffmpeg(
                ["ffmpeg", "-y", "-loop", "1", "-i", scene_img,
                 "-f", "lavfi", "-i", "anullsrc=stereo:44100",
                 "-t", f"{duration:.3f}", "-vf", f"{VF_NORM},fps=25",
                 "-map", "0:v:0", "-map", "1:a:0",
                 "-c:v", "libx264", "-pix_fmt", "yuv420p",
                 "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
                 out_path],
                f"dialogue_{audio_idx}", out_path, scene_img, duration)

    # --- Static frame segments (Ch3 practice) ---
    elif seg_type in ("listen_en", "listen_zh", "practice"):
        if seg_type == "listen_zh":
            frame_name = f"zh_{d_idx}.png" if d_idx >= 0 else ""
        else:
            frame_name = f"en_{d_idx}.png" if d_idx >= 0 else ""
        video_src = str(static_dir / frame_name) if frame_name else scene_img
        if not os.path.exists(video_src):
            video_src = scene_img

        if audio_file and os.path.exists(audio_file):
            cmd = ["ffmpeg", "-y", "-loop", "1", "-i", video_src, "-i", audio_file,
                   "-t", f"{duration:.3f}", "-map", "0:v:0", "-map", "1:a:0",
                   "-c:v", "libx264", "-pix_fmt", "yuv420p", "-vf", f"{VF_NORM},fps=25",
                   "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
                   "-af", f"{fade_af},apad=whole_dur={duration:.3f}",
                   out_path]
        else:
            cmd = ["ffmpeg", "-y", "-loop", "1", "-i", video_src,
                   "-f", "lavfi", "-i", "anullsrc=stereo:44100",
                   "-t", f"{duration:.3f}", "-map", "0:v:0", "-map", "1:a:0",
                   "-c:v", "libx264", "-pix_fmt", "yuv420p", "-vf", f"{VF_NORM},fps=25",
                   "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
                   out_path]
        _run_ffmpeg(cmd, seg_type, out_path, scene_img, duration)

    # --- Title card ---
    elif seg_type == "title_card":
        title_en = seg.get("subtitle_en", "")
        title_zh = seg.get("subtitle_zh", "")
        scene_zh = seg.get("scene_zh", "")
        title_overlay = str(static_dir / "title_overlay.png")
        _render_title_card(title_en, title_zh, scene_zh, scene_img, title_overlay)

        intro_audio = narration.get("intro")
        if intro_audio and os.path.exists(intro_audio):
            intro_dur = _get_duration(intro_audio)
            out_dur = intro_dur + pad
            cmd = ["ffmpeg", "-y", "-loop", "1", "-i", scene_img,
                   "-i", intro_audio, "-i", title_overlay,
                   "-t", f"{out_dur:.3f}",
                   "-filter_complex",
                   f"[0:v]{VF_NORM}[bg];[bg][2:v]overlay=0:0[v];"
                   f"[1:a]afade=t=in:st=0:d=0.05,afade=t=out:st={max(0, intro_dur-0.05):.2f}:d=0.05,apad=whole_dur={out_dur:.3f}[a]",
                   "-map", "[v]", "-map", "[a]",
                   "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "25",
                   "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
                   out_path]
        else:
            cmd = ["ffmpeg", "-y", "-loop", "1", "-i", scene_img,
                   "-i", title_overlay,
                   "-f", "lavfi", "-i", "anullsrc=stereo:44100",
                   "-t", f"{duration:.3f}",
                   "-filter_complex", f"[0:v]{VF_NORM}[bg];[bg][1:v]overlay=0:0[v]",
                   "-map", "[v]", "-map", "2:a",
                   "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "25",
                   "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
                   out_path]
        _run_ffmpeg(cmd, "title_card", out_path, scene_img, duration)

    # --- Practice intro ---
    elif seg_type == "practice_intro":
        intro_en = seg.get("subtitle_en", "")
        intro_zh = seg.get("subtitle_zh", "")
        out_dur = audio_dur + pad
        # 引导文字逐句拆成多张卡片，按时间窗依次显示（避免整段文字溢出画面）
        card_paths, card_windows = _render_intro_cards(intro_en, intro_zh, static_dir, out_dur)
        card_input_args = [arg for c in card_paths for arg in ("-i", c)]
        narration_audio = narration.get("practice_intro")
        if narration_audio and os.path.exists(narration_audio):
            overlay_chain, _ = _build_intro_overlay_chain("bg", card_paths, card_windows, start_idx=2)
            cmd = ["ffmpeg", "-y", "-loop", "1", "-i", scene_img,
                   "-i", narration_audio,
                   *card_input_args,
                   "-t", f"{out_dur:.3f}",
                   "-filter_complex",
                   f"[0:v]{VF_NORM}[bg];{overlay_chain};"
                   f"[1:a]afade=t=in:st=0:d=0.05,afade=t=out:st={max(0, audio_dur-0.05):.2f}:d=0.05,apad=whole_dur={out_dur:.3f}[a]",
                   "-map", "[v]", "-map", "[a]",
                   "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "25",
                   "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
                   out_path]
        else:
            overlay_chain, _ = _build_intro_overlay_chain("bg", card_paths, card_windows, start_idx=1)
            anull_idx = 1 + len(card_paths)
            cmd = ["ffmpeg", "-y", "-loop", "1", "-i", scene_img,
                   *card_input_args,
                   "-f", "lavfi", "-i", "anullsrc=stereo:44100",
                   "-t", f"{out_dur:.3f}",
                   "-filter_complex", f"[0:v]{VF_NORM}[bg];{overlay_chain}",
                   "-map", "[v]", "-map", f"{anull_idx}:a",
                   "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "25",
                   "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
                   out_path]
        _run_ffmpeg(cmd, "practice_intro", out_path, scene_img, duration)

    # --- Outro ---
    elif seg_type == "outro":
        bg_for_host = host_bg or scene_img
        if not _render_host_segment("outro", seg_idx, host_poses, bg_for_host,
                                    audio_file, out_path, duration, sm_root,
                                    render_fps, fade_af, stop_check=stop_check):
            # 回退：静态场景图 + 文字叠加
            outro_en = seg.get("subtitle_en", "")
            outro_zh = seg.get("subtitle_zh", "")
            outro_overlay = str(static_dir / "outro_overlay.png")
            _render_practice_intro(outro_en, outro_zh, scene_img, outro_overlay)
            out_dur = audio_dur + pad
            outro_audio = narration.get("outro")
            if outro_audio and os.path.exists(outro_audio):
                cmd = ["ffmpeg", "-y", "-loop", "1", "-i", scene_img,
                       "-i", outro_audio, "-i", outro_overlay,
                       "-t", f"{out_dur:.3f}",
                       "-filter_complex",
                       f"[0:v]{VF_NORM}[bg];[bg][2:v]overlay=0:0[v];"
                       f"[1:a]afade=t=in:st=0:d=0.05,afade=t=out:st={max(0, audio_dur-0.05):.2f}:d=0.05,apad=whole_dur={out_dur:.3f}[a]",
                       "-map", "[v]", "-map", "[a]",
                       "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "25",
                       "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
                       out_path]
            else:
                cmd = ["ffmpeg", "-y", "-loop", "1", "-i", scene_img,
                       "-i", outro_overlay,
                       "-f", "lavfi", "-i", "anullsrc=stereo:44100",
                       "-t", f"{duration:.3f}",
                       "-filter_complex", f"[0:v]{VF_NORM}[bg];[bg][1:v]overlay=0:0[v]",
                       "-map", "[v]", "-map", "2:a",
                       "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "25",
                       "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
                       out_path]
            _run_ffmpeg(cmd, "outro", out_path, scene_img, duration)

    # --- Fallback: static scene image ---
    else:
        if audio_file and os.path.exists(audio_file):
            cmd = ["ffmpeg", "-y", "-loop", "1", "-i", scene_img, "-i", audio_file,
                   "-t", f"{duration:.3f}", "-map", "0:v:0", "-map", "1:a:0",
                   "-c:v", "libx264", "-pix_fmt", "yuv420p", "-vf", f"{VF_NORM},fps=25",
                   "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
                   "-af", f"{fade_af},apad=whole_dur={duration:.3f}",
                   out_path]
        else:
            cmd = ["ffmpeg", "-y", "-loop", "1", "-i", scene_img,
                   "-f", "lavfi", "-i", "anullsrc=stereo:44100",
                   "-t", f"{duration:.3f}", "-map", "0:v:0", "-map", "1:a:0",
                   "-c:v", "libx264", "-pix_fmt", "yuv420p", "-vf", f"{VF_NORM},fps=25",
                   "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
                   out_path]
        _run_ffmpeg(cmd, f"fallback_{seg_type}", out_path, scene_img, duration)

    if os.path.exists(out_path) and os.path.getsize(out_path) >= 1000:
        return out_path, seg_type
    return None, seg_type


def compose_original_cutout(
    work_dir: str,
    char_pose_map: dict[str, list[str]],
    scene_bg_list: list[str] | None = None,
    host_poses: list[str] | None = None,
    host_bg: str = "",
    timeline: list[dict] = None,
    script: dict = None,
    narration: dict = None,
    normal_paths: list[str] = None,
    zh_paths: list[str] = None,
    scene_img: str = "",
    srt_dir: str = "",
    pad: float = 0.4,
    render_fps: int = 12,
    workers: int = 1,
    show_zh: bool = True,
    subtitle_font_size: int = 60,
    subtitle_style: dict | None = None,
    ch3_zh_always: bool = False,
    progress_cb=None,
    stop_check=None,
) -> str:
    """Compose final video: original 4-chapter timeline + quest-style cutout dialogue.

    Args:
        work_dir: Working directory.
        char_pose_map: {"char_a": [pose_0..7 paths], "char_b": [...]}.
        scene_bg_list: Background images for dialogue segments. Falls back to [scene_img].
        host_poses: Host pose atlas for welcome/hook_intro/outro segments
                    (quest-style). Bound character poses or separate host atlas.
        host_bg: TV-studio background for host segments. Falls back to scene_img.
        timeline: Timeline segments from build_listening_timeline.
        script: Lesson script dict.
        narration: {"welcome"/"hook"/"outro"/"practice_intro" paths} (host form)
                   or legacy {"intro", "outro", "practice_intro"}.
        normal_paths: English TTS paths.
        zh_paths: Chinese TTS paths.
        scene_img: Scene background image path.
        srt_dir: Directory for SRT file.
        pad: Audio pad between segments.
        render_fps: Stop-motion render framerate.
        workers: Render threads (1=single, 2+=multi, 0=auto).
        show_zh: Show Chinese subtitles.
        subtitle_font_size: English subtitle font size.
        subtitle_style: Subtitle style dict or None.
        progress_cb: callback(percent, message).
        stop_check: Function returning True to stop.

    Returns:
        Path to final video.
    """
    def _cb(pct, msg):
        if progress_cb:
            progress_cb(pct, msg)

    timeline = timeline or []
    script = script or {}
    narration = narration or {}
    normal_paths = normal_paths or []
    zh_paths = zh_paths or []
    scene_bgs = scene_bg_list or [scene_img]

    work = Path(work_dir)
    tmp_dir = work / "tmp_segments"
    static_dir = work / "static_frames"
    vid_dir = work / "videos"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    static_dir.mkdir(parents=True, exist_ok=True)
    vid_dir.mkdir(parents=True, exist_ok=True)

    dialogue = script.get("dialogue", [])
    n = len(dialogue)

    sm_root = Path(tempfile.gettempdir()) / f"sm_frames_{work.name}"
    sm_root.mkdir(parents=True, exist_ok=True)

    total_segs = len(timeline)
    segments: list[str | None] = [None] * total_segs

    # --- Render static frames for Ch3 ---
    if os.path.exists(scene_img):
        _cb(5, f"Rendering {n} static frames...")
        for i, line in enumerate(dialogue):
            if stop_check and stop_check():
                print("  [Cutout] Stop requested during static frame rendering...")
                return ""
            p_en = str(static_dir / f"en_{i}.png")
            # 中文常显：EN 帧直接带中文翻译（内容同 zh 帧）
            _render_static_frame(
                line.get("text", ""), line.get("phonetic", ""),
                line.get("zh", "") if ch3_zh_always else "",
                scene_img, p_en, i, n)
            p = str(static_dir / f"zh_{i}.png")
            _render_static_frame(
                line.get("text", ""), line.get("phonetic", ""),
                line.get("zh", ""), scene_img, p, i, n)
        _cb(10, "Static frames done.")

    # --- Resolve workers ---
    if workers == 0:
        workers = os.cpu_count() or 2
    print(f"  [Cutout] Rendering {total_segs} segments with {workers} thread(s)...")

    if workers <= 1:
        # --- Single-thread ---
        for seg_idx, seg in enumerate(timeline):
            if stop_check and stop_check():
                print("  [Cutout] Stop requested, aborting segment rendering...")
                break
            out_path, seg_type = _prepare_segment(
                seg_idx, seg, timeline, dialogue, narration,
                normal_paths, zh_paths, char_pose_map, scene_bgs,
                scene_img, pad, render_fps, tmp_dir, sm_root, static_dir,
                host_poses=host_poses, host_bg=host_bg or scene_img,
                stop_check=stop_check,
            )
            if out_path:
                segments[seg_idx] = out_path
            print(f"  Segment {seg_idx + 1}/{total_segs} ({seg_type}) done", flush=True)
            _cb(int((seg_idx + 1) / total_segs * 80),
                f"  Segment {seg_idx + 1}/{total_segs} done")
    else:
        # --- Multi-thread ---
        # 不使用 with 语句 — __exit__ 会 shutdown(wait=True) 阻塞至所有任务完成，
        # 导致 stop_check 无法生效。手动管理 pool，stop 时立即取消未开始的任务。
        from concurrent.futures import ThreadPoolExecutor, as_completed
        pool = ThreadPoolExecutor(max_workers=workers)
        futs = {
            pool.submit(
                _prepare_segment,
                seg_idx, seg, timeline, dialogue, narration,
                normal_paths, zh_paths, char_pose_map, scene_bgs,
                scene_img, pad, render_fps, tmp_dir, sm_root, static_dir,
                host_poses, host_bg or scene_img,
                stop_check,
            ): seg_idx
            for seg_idx, seg in enumerate(timeline)
        }
        done_count = 0
        _stopped = False
        for fut in as_completed(futs):
            seg_idx = futs[fut]
            try:
                out_path, seg_type = fut.result()
            except Exception as e:
                print(f"  [Cutout] Segment {seg_idx+1} error: {e}", flush=True)
                out_path, seg_type = None, "error"
            if out_path:
                segments[seg_idx] = out_path
            done_count += 1
            print(f"  Segment {done_count}/{total_segs} ({seg_type}) done", flush=True)
            _cb(int(done_count / total_segs * 80),
                f"  Segment {done_count}/{total_segs} done")
            if stop_check and stop_check():
                print("  [Cutout] Stop requested, cancelling remaining segments...")
                for f in futs:
                    f.cancel()
                _stopped = True
                break
        # 立即关闭 pool，不等待未完成的任务
        pool.shutdown(wait=False, cancel_futures=True)
        if _stopped:
            # 收集已完成的结果，丢弃仍在运行的
            for fut, seg_idx in futs.items():
                if fut.done() and not fut.cancelled():
                    try:
                        out_path, _ = fut.result()
                        if out_path:
                            segments[seg_idx] = out_path
                    except Exception:
                        pass

    # If stopped, skip concat/subtitles/loudnorm
    if stop_check and stop_check():
        print("  [Cutout] Compose interrupted, segments saved for resume.")
        return ""

    # --- Insert silent placeholders for failed segments ---
    missing = [i for i, s in enumerate(segments) if s is None]
    for i in missing:
        seg = timeline[i]
        ph_path = str(tmp_dir / f"ph_{i:03d}.mp4")
        ph_cmd = ["ffmpeg", "-y", "-loop", "1", "-i", scene_img,
                  "-f", "lavfi", "-i", "anullsrc=stereo:44100",
                  "-t", f"{seg['duration']:.3f}", "-vf", f"{VF_NORM},fps=25",
                  "-map", "0:v:0", "-map", "1:a:0",
                  "-c:v", "libx264", "-pix_fmt", "yuv420p",
                  "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
                  ph_path]
        print(f"  [Cutout] WARNING: segment {i + 1} ({seg['type']}) failed — "
              f"inserting silent placeholder ({seg['duration']:.2f}s)")
        _run_ffmpeg(ph_cmd, f"placeholder_{i}", ph_path, scene_img, seg["duration"])
        if os.path.exists(ph_path) and os.path.getsize(ph_path) >= 1000:
            segments[i] = ph_path
    if missing:
        print(f"  [Cutout] {len(missing)} failed segment(s) handled with placeholders")

    segments = [s for s in segments if s is not None]

    # --- Concat all segments ---
    _cb(80, "Concatenating segments...")
    no_sub = str(vid_dir / "final_no_sub.mp4")
    concat_segments(segments, no_sub, tmp_dir=tmp_dir)

    # --- Burn subtitles ---
    _cb(90, "Burning subtitles (Pillow overlay)...")
    final_path = burn_subtitles(
        no_sub, timeline, script, str(work), srt_dir, pad, _cb,
        show_zh=show_zh,
        en_font_size=subtitle_font_size,
        zh_font_size=int(subtitle_font_size * 0.85),
        out_fps=25, style=subtitle_style,
    )

    shutil.rmtree(tmp_dir, ignore_errors=True)

    # --- Final loudnorm ---
    _cb(95, "Final loudnorm pass (normalize volume)...")
    apply_final_loudnorm(final_path, str(vid_dir))

    size_mb = os.path.getsize(final_path) / (1024 * 1024)
    _cb(100, f"Original Cutout video done: {final_path} ({size_mb:.1f}MB)")
    return final_path
