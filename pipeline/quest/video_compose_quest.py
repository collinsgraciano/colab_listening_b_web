"""Quest FFmpeg video composition — task-hook slow listening, stop-motion mode.

Structure:
  Ch1: Welcome       (host stop-motion + subtitle)
  Ch2: Hook / Intro  (host stop-motion + subtitle)
  Ch3: Slow Dialogue (4 phases: buildup->core->reveal->review, per-line
      stop-motion with randomized pose schedule + subtitles)
  Ch4: Outro & CTA   (host stop-motion + subtitle)

All segments (welcome/hook/dialogue/outro) render as stop-motion with
subtitles — no title card, no overlay cards. The host character (节目主)
appears on-screen in Welcome, Hook, and Outro segments. Dialogue segments
use each speaker's pose atlas with multi-character support via on_screen.
"""
import os
import sys
import math
import subprocess
import shutil
import tempfile
import random
import threading
from pathlib import Path

_PARENT = str(Path(__file__).parent.parent.resolve())
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from video_compose import (
    _render_title_card,
    _render_practice_intro,
)
from PIL import Image
from media_utils import (
    FONT_EN, FONT_ZH, VF_NORM,
    concat_segments, burn_subtitles, apply_final_loudnorm,
    make_silent_fallback_cmd, write_sync_report,
)


def _wrap_lines(draw, text, font_path, size, max_w, min_size=24):
    """Word-wrap text at the largest font size (>= min_size) that fits max_w.

    Returns (font, lines).
    """
    from PIL import ImageFont
    while size >= min_size:
        font = ImageFont.truetype(font_path, size)
        words = text.split()
        lines = []
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
        ok = all(draw.textbbox((0, 0), ln, font=font)[2] -
                 draw.textbbox((0, 0), ln, font=font)[0] <= max_w for ln in lines)
        if ok and lines:
            return font, lines
        size -= 2
    font = ImageFont.truetype(font_path, min_size)
    return font, [text]


def _fit_font(draw, text, font_path, start_size, min_size, max_w):
    """Shrink font size until text fits max_w. Returns (font, w, h)."""
    from PIL import ImageFont
    size = start_size
    font = ImageFont.truetype(font_path, size)
    while size > min_size:
        bbox = draw.textbbox((0, 0), text, font=font)
        if bbox[2] - bbox[0] <= max_w:
            break
        size -= 2
        font = ImageFont.truetype(font_path, size)
    bbox = draw.textbbox((0, 0), text, font=font)
    return font, bbox[2] - bbox[0], bbox[3] - bbox[1]


def _render_hook_frame(hook_en, question_en, question_zh, scene_img_path,
                       out_path, w=1280, h=720):
    """Render the listening-task hook card PNG (transparent bg for overlay).

    Layout: task badge -> narrator text (wrapped, small) -> question EN (big
    white) -> question ZH (gold) -> bottom hint bar. All plain text (no emoji —
    CJK fonts render emoji as tofu).
    """
    from PIL import Image, ImageDraw, ImageFont

    MARGIN = 70
    frame = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(frame)

    # Semi-transparent dark panel
    panel = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    pd = ImageDraw.Draw(panel)
    pd.rectangle([40, 30, w - 40, h - 110], fill=(0, 0, 0, 185))
    frame = Image.alpha_composite(frame, panel)
    draw = ImageDraw.Draw(frame)

    cur_y = 55

    # Badge: LISTENING TASK · 聽力任務
    badge_txt = "LISTENING TASK · 聽力任務"
    badge_font, bw, bh = _fit_font(draw, badge_txt, FONT_EN, 46, 28, w - 2 * MARGIN)
    draw.text(((w - bw) // 2, cur_y), badge_txt, font=badge_font,
              fill=(255, 220, 0, 255), stroke_width=4, stroke_fill=(0, 0, 0, 255))
    cur_y += bh + 28

    # Narrator text (small, wrapped) — auto-shrink until <= 9 lines
    if hook_en:
        for start in (40, 36, 32, 28, 24):
            n_font, n_lines = _wrap_lines(draw, hook_en, FONT_EN, start,
                                          w - 2 * MARGIN, min_size=24)
            if len(n_lines) <= 9:
                break
        for ln in n_lines:
            lb = draw.textbbox((0, 0), ln, font=n_font)
            lw, lh = lb[2] - lb[0], lb[3] - lb[1]
            draw.text(((w - lw) // 2, cur_y), ln, font=n_font,
                      fill=(235, 235, 235, 255))
            cur_y += lh + 6
        cur_y += 22

    # Question EN (big white)
    if question_en:
        q_font, qw, qh = _fit_font(draw, question_en, FONT_EN, 64, 34, w - 2 * MARGIN)
        draw.text(((w - qw) // 2, cur_y), question_en, font=q_font,
                  fill=(255, 255, 255, 255), stroke_width=6, stroke_fill=(0, 0, 0, 255))
        cur_y += qh + 18

    # Question ZH (gold)
    if question_zh:
        z_font, zw, zh = _fit_font(draw, question_zh, FONT_ZH, 50, 26, w - 2 * MARGIN)
        draw.text(((w - zw) // 2, cur_y), question_zh, font=z_font,
                  fill=(255, 220, 0, 255), stroke_width=4, stroke_fill=(0, 0, 0, 255))
        cur_y += zh + 14

    frame.save(out_path, "PNG")


def _render_outro_frame(question_en, question_zh, scene_img_path,
                        out_path, w=1280, h=720):
    """Render the closing answer/CTA card PNG (transparent bg for overlay)."""
    from PIL import Image, ImageDraw

    MARGIN = 70
    frame = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    panel = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    pd = ImageDraw.Draw(panel)
    pd.rectangle([40, 40, w - 40, h - 40], fill=(0, 0, 0, 190))
    frame = Image.alpha_composite(frame, panel)
    draw = ImageDraw.Draw(frame)

    # Vertical stack, centered as a block
    blocks = []

    t1 = "你的答案是什麼？"
    blocks.append((t1, FONT_ZH, 56, (255, 220, 0, 255), 5))
    if question_en:
        blocks.append((question_en, FONT_EN, 58, (255, 255, 255, 255), 6))
    if question_zh:
        blocks.append((question_zh, FONT_ZH, 44, (255, 220, 0, 255), 4))
    blocks.append(("在評論區用英文寫下你的答案！", FONT_ZH, 44, (255, 255, 255, 255), 4))
    blocks.append(("哪怕一句話也很棒！", FONT_ZH, 38, (200, 200, 200, 255), 3))
    blocks.append(("Like · Subscribe · 每天慢速聽英文", FONT_ZH, 36, (130, 200, 255, 255), 3))

    # Measure all blocks with fitted fonts
    fitted = []
    for text, fpath, start_size, color, stroke in blocks:
        font, tw, th = _fit_font(draw, text, fpath, start_size, 24, w - 2 * MARGIN)
        fitted.append((text, font, tw, th, color, stroke))

    GAP = 26
    total_h = sum(f[3] for f in fitted) + GAP * (len(fitted) - 1)
    cur_y = max(50, (h - total_h) // 2)
    for text, font, tw, th, color, stroke in fitted:
        draw.text(((w - tw) // 2, cur_y), text, font=font,
                  fill=color, stroke_width=stroke, stroke_fill=(0, 0, 0, 255))
        cur_y += th + GAP

    frame.save(out_path, "PNG")


def _compute_audio_rms_segments(audio_file: str, n_segments: int = 20) -> list[float]:
    """Compute RMS energy segments from an audio file.

    Returns a list of n_segments float values (0..1) representing the
    relative energy at evenly-spaced points in the audio.
    Used to schedule pose switches at natural speech pauses.
    """
    import subprocess as _sp
    import json as _json
    try:
        r = _sp.run(
            ["ffmpeg", "-i", audio_file, "-af",
             f"astats=metadata=1:reset={1.0/n_segments},ametadata=print:key=lavfi.astats.Overall.RMS_level",
             "-f", "null", "-"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)
        lines = r.stderr.split("\n")
        rms_vals = []
        for line in lines:
            if "RMS_level" in line:
                try:
                    val = float(line.split("=")[-1].strip())
                    rms_vals.append(max(0, val))
                except ValueError:
                    pass
        if rms_vals:
            mx = max(rms_vals)
            return [v / mx if mx > 0 else 0 for v in rms_vals]
    except Exception:
        pass
    return [0.5] * n_segments


def _build_pose_schedule(n_poses: int, total_frames: int, fps: float,
                         seed: int, is_speaker: bool = True,
                         audio_file: str | None = None) -> list[tuple[int, int]]:
    """Build a pose schedule: list of (pose_idx, start_frame).

    Speaker: switches pose at audio-driven pauses (low RMS points), or
             every 2.0-5.0s if no audio. Consecutive poses always differ.
    Listener: always static — caller uses fixed pose.
    """
    rng = random.Random(seed)
    if n_poses <= 1 or not is_speaker:
        return [(0, 0)]

    # Try audio-driven scheduling
    if audio_file and os.path.exists(audio_file):
        rms = _compute_audio_rms_segments(audio_file, n_segments=max(20, total_frames // max(1, fps)))
        # Find low-energy points (pauses) — at least 2s apart
        min_hold_frames = round(2.0 * fps)
        schedule = [(0, 0)]
        last_switch = 0
        for i, val in enumerate(rms):
            frame = round(i * total_frames / len(rms))
            if frame - last_switch < min_hold_frames:
                continue
            if frame >= total_frames:
                break
            # Switch at low-energy points (below 40% of max)
            if val < 0.4:
                current = schedule[-1][0]
                candidates = [j for j in range(n_poses) if j != current]
                schedule.append((rng.choice(candidates), frame))
                last_switch = frame
        return schedule

    # Fallback: random 2.0-5.0s holds
    schedule = [(0, 0)]
    frame = 0
    while frame < total_frames:
        hold = rng.uniform(2.0, 5.0)
        frame += max(1, round(hold * fps))
        if frame < total_frames:
            current = schedule[-1][0]
            candidates = [i for i in range(n_poses) if i != current]
            schedule.append((rng.choice(candidates), frame))
    return schedule


_CLIP_BASE = "talking"
_CLIP_IDLE = "idle"
_CLIP_INSERTS = ("gesture", "wave")


def _clip_variants(clips: dict, base: str) -> list:
    """动作变体键列表（base 与 base_NN 前缀，如 talking_01/02/03），按名排序稳定。

    兼容旧单键 manifest（键名恰为 base）。无匹配返回空列表。
    """
    return sorted(k for k in clips if k == base or k.startswith(base + "_"))


def _build_clip_schedule(clips: dict, total_frames: int, fps: float,
                         audio_file: str | None, seed: int,
                         is_speaker: bool = True) -> list:
    """构建序列帧动作调度：[(action, start_frame)]。

    说话者：talking 变体组按行轮换（seed 含行号），在音频低能量停顿点插入
    gesture/wave 一个循环后回到组内下一个变体；无音频时按 3-5s 随机间隔。
    倾听者：整段 idle 变体循环（clip 自带微动作，替代呼吸正弦）。
    旧单键 manifest（talking/idle）退化为原行为。
    """
    if not clips:
        return []
    talking_group = _clip_variants(clips, _CLIP_BASE)
    idle_group = _clip_variants(clips, _CLIP_IDLE)
    inserts = [a for a in _CLIP_INSERTS if a in clips]
    if not is_speaker:
        pool = idle_group or talking_group or [sorted(clips)[0]]
        return [(pool[seed % len(pool)], 0)]
    if not talking_group:
        # 未知名动作键：按字典序取第一个当基础循环
        return [(sorted(clips)[0], 0)]
    base_idx = seed % len(talking_group)
    schedule = [(talking_group[base_idx], 0)]
    if not inserts:
        return schedule
    rng = random.Random(seed)
    loop_frames = 16  # 与 FRAMES_PER_CLIP 对齐的插入循环长度
    tail_guard = round(1.2 * fps)
    switch_frames = []
    if audio_file and os.path.exists(audio_file):
        rms = _compute_audio_rms_segments(
            audio_file, n_segments=max(20, total_frames // max(1, round(fps))))
        min_hold_frames = round(2.0 * fps)
        last = 0
        for i, val in enumerate(rms):
            frame = round(i * total_frames / len(rms))
            if frame - last < min_hold_frames or frame >= total_frames - tail_guard:
                continue
            if val < 0.4:
                switch_frames.append(frame)
                last = frame
    else:
        frame = 0
        while True:
            frame += round(rng.uniform(3.0, 5.0) * fps)
            if frame >= total_frames - tail_guard:
                break
            switch_frames.append(frame)

    k = 0
    cycle = 0
    for sf in switch_frames:
        if sf - schedule[-1][1] < round(1.6 * fps):
            continue
        schedule.append((inserts[k % len(inserts)], sf))
        k += 1
        cycle += 1
        # 插入一个完整循环（loop_frames @ 12fps）后回到组内下一个 talking 变体
        back = sf + max(round(1.5 * fps), round(loop_frames * fps / 12.0))
        if back < total_frames - tail_guard:
            schedule.append((talking_group[(base_idx + cycle) % len(talking_group)],
                             back))
    return schedule


def _atomic_save(img, target: Path) -> None:
    """原子写 PNG 缓存：先写进程/线程唯一的 .tmp 再 os.replace。

    防止并发 worker 线程在读缓存时读到写到一半的半截文件
    （曾导致某句对白整段渲染失败、成片插入无声背景占位段）。
    """
    tmp = target.with_name(f"{target.stem}.{os.getpid()}_{threading.get_ident()}.tmp.png")
    img.save(str(tmp))
    os.replace(str(tmp), str(target))


def _render_sm_segment(
    char_layers: list[dict],
    bg_img_path: str,
    audio_file: str | None,
    out_path: str,
    duration: float,
    frames_dir: Path,
    cache_dir: Path,
    render_fps: int = 12,
    overlay_path: str | None = None,
    seed: int = 0,
    direction: int = 1,
    fade_af: str = "",
    stop_check=None,
) -> bool:
    """渲染定格动画段。无论成功还是中途异常，frames_dir 一律清理（防残留）。"""
    frames_dir.mkdir(parents=True, exist_ok=True)
    try:
        return _render_sm_segment_inner(
            char_layers, bg_img_path, audio_file, out_path, duration,
            frames_dir, cache_dir,
            render_fps=render_fps, overlay_path=overlay_path, seed=seed,
            direction=direction, fade_af=fade_af, stop_check=stop_check,
        )
    finally:
        shutil.rmtree(frames_dir, ignore_errors=True)


def _render_sm_segment_inner(
    char_layers: list[dict],
    bg_img_path: str,
    audio_file: str | None,
    out_path: str,
    duration: float,
    frames_dir: Path,
    cache_dir: Path,
    render_fps: int = 12,
    overlay_path: str | None = None,
    seed: int = 0,
    direction: int = 1,
    fade_af: str = "",
    stop_check=None,
) -> bool:
    """Render a stop-motion video segment with multi-character support.

    Phase 1 enhancements:
    - Speaker: optical flow morph between pose switches (5-frame transition)
    - Speaker: audio-driven pose switching (at low-RMS pauses)
    - Listener: subtle breathing (±2px sine, 3.3s period) + occasional blink
    - Landing transform still applies at each pose hold start

    Each entry in char_layers is a dict:
        {"poses": list[str], "is_speaker": bool}

    Returns True on success, False on failure.
    """
    from stop_motion import (
        remove_bg, normalize_pose, render_frame,
        compute_landing, POSE_CENTER_Y,
    )
    from PIL import Image as PILImage

    frames_dir.mkdir(parents=True, exist_ok=True)

    # Process poses for each character layer (normalize + cache)
    processed_layers: list[dict] = []
    for layer in char_layers:
        pose_paths = layer.get("poses", [])
        is_speaker = layer.get("is_speaker", False)
        processed = []
        _uncached = [p for p in pose_paths
                     if os.path.exists(p)
                     and not (cache_dir / f"cutout_{Path(p).stem}.png").exists()]
        if _uncached:
            print(f"    processing {len(_uncached)} pose(s)...", flush=True)
        for p_path in pose_paths:
            if not os.path.exists(p_path):
                continue
            cache_path = cache_dir / f"cutout_{Path(p_path).stem}.png"
            norm = None
            if cache_path.exists():
                try:
                    norm = PILImage.open(cache_path).convert("RGBA")
                except Exception:
                    # 缓存可能损坏或被并发线程写到一半：删掉后走重新处理
                    try:
                        cache_path.unlink(missing_ok=True)
                    except Exception:
                        pass
                    norm = None
            if norm is None:
                try:
                    raw = PILImage.open(p_path)
                    alpha = remove_bg(raw)
                    norm = normalize_pose(alpha)
                    _atomic_save(norm, cache_path)
                except Exception as e:
                    print(f"  [Quest] pose process error for {p_path}: {e}")
                    continue
            processed.append(norm)
        if not processed:
            processed = [PILImage.new("RGBA", (1280, 720), (0, 0, 0, 0))]
        # 序列帧 clips（sprite_sequence 模式）：生成期已统一 remove_bg+归一化，
        # 此处直接加载，无需 pose 的 cutout 缓存链
        clips_in = layer.get("clips") or {}
        processed_clips: dict[str, list] = {}
        for act, frame_paths in clips_in.items():
            cframes = []
            for fp in frame_paths or []:
                if not os.path.exists(fp):
                    continue
                try:
                    cframes.append(PILImage.open(fp).convert("RGBA"))
                except Exception as e:
                    print(f"  [Quest] clip frame error {fp}: {e}")
            if cframes:
                processed_clips[act] = cframes
        processed_layers.append({
            "poses": processed,
            "clips": processed_clips,
            "clip_fps": int(layer.get("clip_fps", 12)),
            "is_speaker": is_speaker,
        })

    # Load background
    bg_img = PILImage.open(bg_img_path).convert("RGBA").resize((1280, 720))

    # Determine positions based on number of characters
    n_chars = len(processed_layers)
    if n_chars == 0:
        positions = []
    elif n_chars == 1:
        positions = [1280 * 0.5]
    elif n_chars == 2:
        # 2 characters: speaker left, listener right
        positions = []
        for layer in processed_layers:
            if layer["is_speaker"]:
                positions.append(1280 * 0.35)
            else:
                positions.append(1280 * 0.65)
    else:
        # 3+ characters: speaker at left, listeners spread across center-right
        positions = []
        listener_idx = 0
        n_listeners = sum(1 for l in processed_layers if not l["is_speaker"])
        for layer in processed_layers:
            if layer["is_speaker"]:
                positions.append(1280 * 0.28)
            else:
                # Distribute listeners from 0.50 to 0.85
                t = (listener_idx + 1) / (n_listeners + 1)
                positions.append(1280 * (0.50 + t * 0.35))
                listener_idx += 1

    # Build pose schedule for each character layer
    # Use ceil to ensure enough input frames to fill the (frame-aligned) duration
    total_frames = max(1, math.ceil(duration * render_fps))
    rng = random.Random(seed)
    schedules = []
    for i, layer in enumerate(processed_layers):
        af = audio_file if layer["is_speaker"] else None
        s = _build_pose_schedule(
            len(layer["poses"]), total_frames, render_fps,
            seed + i * 100, is_speaker=layer["is_speaker"],
            audio_file=af)
        schedules.append(s)

    # Build clip schedules（sprite_sequence 模式专用；无 clips/take_mode 层为空表）
    clip_schedules = []
    for i, layer in enumerate(processed_layers):
        clips = layer.get("clips") or {}
        if not clips or layer.get("take_mode"):
            clip_schedules.append([])
            continue
        clip_schedules.append(_build_clip_schedule(
            clips, total_frames, render_fps,
            audio_file if layer["is_speaker"] else None,
            seed + i * 100, is_speaker=layer["is_speaker"]))

    # Pre-generate optical flow morph frames for speaker pose transitions
    morph_cache: dict[int, list] = {}  # layer_idx -> {transition_idx: [frames]}
    try:
        from stop_motion import generate_morph_frames as _gen_morph
        _HAS_MORPH = True
    except ImportError:
        _HAS_MORPH = False

    for li, layer in enumerate(processed_layers):
        if (not layer["is_speaker"]) or (not _HAS_MORPH) or layer.get("clips"):
            continue
        poses = layer["poses"]
        sched = schedules[li]
        if len(poses) < 2 or len(sched) < 2:
            continue
        morph_cache[li] = {}
        for ti in range(len(sched) - 1):
            pa_idx = sched[ti][0]
            pb_idx = sched[ti + 1][0]
            pa = poses[pa_idx]
            pb = poses[pb_idx]
            if pa.size == pb.size:
                try:
                    frames = _gen_morph(pa, pb, n_frames=5)
                    morph_cache[li][ti] = frames
                except Exception:
                    pass

    MORPH_DUR = 0.2  # seconds for 5-frame transition
    MORPH_N = 5
    cy = POSE_CENTER_Y

    for fidx in range(total_frames):
        if stop_check and stop_check():
            print("    [SM] Stop requested, aborting frame render...", flush=True)
            return False
        if fidx % 10 == 0:
            print(f"    frame {fidx}/{total_frames}", flush=True)
        t = fidx / render_fps
        canvas = bg_img.copy()

        for li, layer in enumerate(processed_layers):
            poses = layer["poses"]
            schedule = schedules[li]
            x = positions[li] if li < len(positions) else 1280 * 0.5

            # --- 序列帧播放分支（sprite_sequence；clips 加载成功才走此路）---
            layer_clips = layer.get("clips") or {}

            # --- 整句单 take 分支（take_mode：序列帧新模式）---
            if layer.get("take_mode") and layer_clips:
                from stop_motion import transform_pose, paste_with_shadow
                if layer["is_speaker"]:
                    # 说话者：一个 talking take 铺满整句（无循环/无插入/无 landing），
                    # 帧随时间均匀取样 → 动作速度 ≈ 源视频速度（台词 4-8s 带 0.75-1.5x）
                    group = (layer_clips.get(layer.get("take_action") or "")
                             or layer_clips.get(_CLIP_BASE))
                    if not group:
                        gkeys = _clip_variants(layer_clips, _CLIP_BASE) \
                            or sorted(layer_clips)
                        group = layer_clips[gkeys[0]]
                    idx = min(int(fidx / max(1, total_frames) * len(group)),
                              len(group) - 1)
                    sprite = transform_pose(group[idx])
                    paste_with_shadow(canvas, sprite, x, cy, centered=True)
                else:
                    # 倾听者：idle 变体按 seed 选其一，循环播放
                    gkeys = _clip_variants(layer_clips, _CLIP_IDLE) \
                        or sorted(layer_clips)
                    frames = layer_clips[gkeys[(seed + li * 100) % len(gkeys)]]
                    idx = int((fidx / render_fps) * layer.get("clip_fps", 12)) \
                        % len(frames)
                    paste_with_shadow(canvas, frames[idx], x, cy, centered=True)
                continue

            cschedule = clip_schedules[li] if li < len(clip_schedules) else []
            if layer_clips and cschedule:
                from stop_motion import transform_pose, paste_with_shadow
                act_idx = 0
                for si, (act, start_f) in enumerate(cschedule):
                    if fidx >= start_f:
                        act_idx = si
                    else:
                        break
                act, start_f = cschedule[act_idx]
                cframes = layer_clips.get(act) or []
                if cframes:
                    local_t = (fidx - start_f) / render_fps
                    # 游戏式采样：播放帧率(clip_fps)与输出帧率(render_fps)解耦
                    cidx = int(local_t * layer.get("clip_fps", 12)) % len(cframes)
                    sprite = cframes[cidx]
                    if layer["is_speaker"]:
                        landing = compute_landing(local_t, direction=direction)
                        sprite = transform_pose(sprite, scale=1.0 + landing["scale"],
                                                rotation=landing["rotation"])
                        paste_with_shadow(canvas, sprite, x + landing["x"],
                                          cy + landing["y"], centered=True)
                    else:
                        paste_with_shadow(canvas, sprite, x, cy, centered=True)
                    continue
            # --- 以下为原姿势切换路径（无 clips 时行为逐字节不变）---

            # Find which pose stage we're in
            stage_idx = 0
            for si, (pi, start_frame) in enumerate(schedule):
                if fidx >= start_frame:
                    stage_idx = si
                else:
                    break

            pose_idx = schedule[stage_idx][0]
            stage_start_frame = schedule[stage_idx][1]
            local_t = (fidx - stage_start_frame) / render_fps

            # Check if we're in a morph transition (near end of current stage)
            in_morph = False
            if (li in morph_cache and stage_idx < len(schedule) - 1
                    and stage_idx in morph_cache[li]):
                next_start = schedule[stage_idx + 1][1]
                frames_to_next = next_start - fidx
                morph_frames_needed = round(MORPH_DUR * render_fps)
                if frames_to_next <= morph_frames_needed and frames_to_next > 0:
                    morph_frames = morph_cache[li][stage_idx]
                    morph_progress = 1.0 - (frames_to_next / morph_frames_needed)
                    morph_idx = min(MORPH_N - 1, int(morph_progress * MORPH_N))
                    pose = morph_frames[morph_idx]
                    in_morph = True

            if not in_morph:
                pose = poses[pose_idx]

            if layer["is_speaker"]:
                if in_morph:
                    # During morph: no landing transform, just show the interpolated frame
                    scale = 1.0
                    px = x
                    py = cy
                    rot = 0.0
                else:
                    landing = compute_landing(local_t, direction=direction)
                    scale = 1.0 + landing["scale"]
                    px = x + landing["x"]
                    py = cy + landing["y"]
                    rot = landing["rotation"]
            else:
                # Listener: subtle breathing motion (sine wave, ~3.3s period)
                breath_y = 2.0 * math.sin(t * 2 * math.pi / 3.3)
                # Occasional blink: replace with "surprised" pose briefly every 4-6s
                blink_cycle = 5.0 + (seed % 100) / 50.0  # 5.0-7.0s period
                blink_phase = (t % blink_cycle) / blink_cycle
                if 0.85 < blink_phase < 0.92 and len(poses) > 3:
                    pose = poses[3]  # "surprised" pose as blink substitute
                else:
                    pose = poses[1] if len(poses) > 1 else poses[0]
                scale = 1.0
                px = x
                py = cy + breath_y
                rot = 0.0

            from stop_motion import transform_pose, paste_with_shadow
            sprite = transform_pose(pose, scale=scale, rotation=rot)
            paste_with_shadow(canvas, sprite, px, py, centered=True)

        frame = canvas.convert("RGB")

        # Overlay PNG (subtitle cards, etc.)
        if overlay_path and os.path.exists(overlay_path):
            ov = PILImage.open(overlay_path).convert("RGBA")
            frame_rgba = frame.convert("RGBA")
            frame_rgba.alpha_composite(ov, (0, 0))
            frame = frame_rgba.convert("RGB")

        frame.save(str(frames_dir / f"frame-{fidx:04d}.png"), compress_level=2)

    # Encode frames to mp4 — output at 25fps to minimize duration quantization
    # error (ceil(duration*25)/25 max drift = 0.04s vs 0.125s at 8fps).
    # Input framerate stays at render_fps for natural stop-motion timing.
    frame_pattern = str(frames_dir / "frame-%04d.png")

    if audio_file and os.path.exists(audio_file):
        cmd = ["ffmpeg", "-y",
               "-framerate", str(render_fps), "-i", frame_pattern,
               "-i", audio_file,
               "-t", f"{duration:.3f}", "-map", "0:v:0", "-map", "1:a:0",
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-vf", "fps=25",
               "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
               "-af", f"{fade_af},apad=whole_dur={duration:.3f}",
               out_path]
    else:
        cmd = ["ffmpeg", "-y",
               "-framerate", str(render_fps), "-i", frame_pattern,
               "-f", "lavfi", "-i", "anullsrc=stereo:44100",
               "-t", f"{duration:.3f}", "-map", "0:v:0", "-map", "1:a:0",
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-vf", "fps=25",
               "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
               out_path]

    try:
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600)
        if r.returncode != 0:
            print(f"  [Quest] FFmpeg error: {r.stderr[-200:]}")
            return False
    except Exception as e:
        print(f"  [Quest] Error: {e}")
        return False
    finally:
        shutil.rmtree(frames_dir, ignore_errors=True)

    return True


def _run_fallback(cmd, out_path, scene_img, duration, render_fps):
    """Run ffmpeg cmd, fallback to static image on failure."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)
    except subprocess.TimeoutExpired:
        print("  FFmpeg TIMEOUT, using static fallback")
        r = None
    if r is None or r.returncode != 0 or not os.path.exists(out_path) or os.path.getsize(out_path) < 1000:
        fallback_cmd = ["ffmpeg", "-y", "-loop", "1", "-i", scene_img,
                        "-f", "lavfi", "-i", "anullsrc=stereo:44100",
                        "-t", f"{duration:.3f}", "-vf", f"{VF_NORM},fps=25",
                        "-map", "0:v:0", "-map", "1:a:0",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
                        out_path]
        try:
            r2 = subprocess.run(fallback_cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)
        except subprocess.TimeoutExpired:
            print("  Fallback also timed out, skipping")
            return
        if r2.returncode != 0 or not os.path.exists(out_path) or os.path.getsize(out_path) < 1000:
            print(f"  Fallback also failed: {r2.stderr[-200:] if r2.stderr else 'unknown'}")


def _prepare_segment(seg_idx, seg, timeline, dialogue, narration,
                     normal_paths, host_poses, host_bg_path, scene_bgs,
                     char_pose_map, pose_images, scene_img, pad, render_fps,
                     tmp_dir, sm_root, char_clip_map=None, sprite_clip_fps=12,
                     sprite_take_mode=False):
    """Prepare params and render a single segment. Returns (out_path or None, seg_type)."""
    seg_type = seg["type"]
    duration = seg["duration"]
    audio_idx = seg.get("audio_index", 0)
    out_path = str(tmp_dir / f"seg_{seg_idx:03d}.mp4")

    # Skip if already rendered (resume support)
    if os.path.exists(out_path) and os.path.getsize(out_path) >= 1000:
        print(f"  Segment {seg_idx + 1} cached, skipping...", flush=True)
        return out_path, seg_type

    audio_file = None
    audio_dur = seg.get("audio_dur", duration - pad)

    if seg_type == "dialogue":
        audio_file = normal_paths[audio_idx] if audio_idx < len(normal_paths) else None
    elif seg_type == "welcome":
        audio_file = narration.get("welcome")
    elif seg_type == "hook_intro":
        audio_file = narration.get("hook")
    elif seg_type == "outro":
        audio_file = narration.get("outro")

    fade_af = f"afade=t=in:st=0:d=0.05,afade=t=out:st={max(0, audio_dur-0.05):.2f}:d=0.05"

    if seg_type in ("welcome", "hook_intro", "outro"):
        h_poses = host_poses or [scene_img]
        host_layer: dict = {"poses": h_poses, "is_speaker": True}
        _host_clips = (char_clip_map or {}).get("host")
        if _host_clips:
            host_layer["clips"] = _host_clips
            host_layer["clip_fps"] = sprite_clip_fps
            if sprite_take_mode:
                host_layer["take_mode"] = True
                if seg_type == "outro":
                    host_layer["take_action"] = "wave"  # 送别挥手
        char_layers = [host_layer]
        frames_dir = sm_root / f"{seg_type}_{seg_idx}"
        direction = 1 if seg_idx % 2 == 0 else -1
        success = _render_sm_segment(
            char_layers, host_bg_path, audio_file, out_path, duration,
            frames_dir, sm_root,
            render_fps=render_fps,
            seed=hash(seg_type) % 1000 + seg_idx,
            direction=direction, fade_af=fade_af,
        )
        if not success:
            cmd = ["ffmpeg", "-y", "-loop", "1", "-i", host_bg_path,
                   "-f", "lavfi", "-i", "anullsrc=stereo:44100",
                   "-t", f"{duration:.3f}", "-vf", f"{VF_NORM},fps=25",
                   "-map", "0:v:0", "-map", "1:a:0",
                   "-c:v", "libx264", "-pix_fmt", "yuv420p",
                   "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
                   out_path]
            _run_fallback(cmd, out_path, scene_img, duration, render_fps)

    elif seg_type == "dialogue":
        # Pre-compute dialogue position for scene_bg rotation
        # 节奏自适应（与 original_cutout_compose 同款）：cadence = 总对话段数 // 背景数，
        # 全部场景背景都会出场（48 行 × 8 背景 → 每 6 行轮换）
        dialogue_seg_count = sum(1 for s in timeline[:timeline.index(seg)] if s["type"] == "dialogue")
        total_dialogue = sum(1 for s in timeline if s["type"] == "dialogue")
        n_scene_bgs = max(1, len(scene_bgs))
        cadence = max(1, total_dialogue // n_scene_bgs)
        bg_idx = (dialogue_seg_count // cadence) % n_scene_bgs
        line_bg = scene_bgs[bg_idx]

        line_data = dialogue[audio_idx] if audio_idx < len(dialogue) else {}
        speaker = line_data.get("speaker", "char_a")
        # 环境镜头已废弃：空/缺失 on_screen 回退为说话人（兼容旧脚本库）
        on_screen = line_data.get("on_screen") or [speaker]

        if char_pose_map:
            _clip_map = char_clip_map or {}
            char_layers = []
            for char_key in on_screen:
                poses = char_pose_map.get(char_key, [])
                if not poses:
                    poses = char_pose_map.get("char_a", [line_bg])
                layer: dict = {
                    "poses": poses,
                    "is_speaker": (char_key == speaker),
                }
                # sprite_sequence：该角色有序列帧时附加（渲染层自动优先播放）
                char_clips = _clip_map.get(char_key)
                if char_clips:
                    layer["clips"] = char_clips
                    layer["clip_fps"] = sprite_clip_fps
                    if sprite_take_mode and char_key == speaker:
                        # 整句单 take：talking 变体按行轮换（相邻行不重样）
                        tkeys = sorted(k for k in char_clips
                                       if k == "talking"
                                       or k.startswith("talking_"))
                        layer["take_mode"] = True
                        layer["take_action"] = tkeys[audio_idx % len(tkeys)]
                char_layers.append(layer)
        else:
            idx = min(audio_idx, len(pose_images) - 1) if pose_images else 0
            line_poses = pose_images[idx] if pose_images and idx < len(pose_images) else [line_bg]
            char_layers = [{"poses": line_poses, "is_speaker": True}]

        frames_dir = sm_root / f"dialogue_{audio_idx}"
        direction = 1 if audio_idx % 2 == 0 else -1
        success = _render_sm_segment(
            char_layers, line_bg, audio_file, out_path, duration,
            frames_dir, sm_root,
            render_fps=render_fps,
            seed=audio_idx * 7 + 13,
            direction=direction, fade_af=fade_af,
        )
        if not success:
            cmd = ["ffmpeg", "-y", "-loop", "1", "-i", scene_img,
                   "-f", "lavfi", "-i", "anullsrc=stereo:44100",
                   "-t", f"{duration:.3f}", "-vf", f"{VF_NORM},fps=25",
                   "-map", "0:v:0", "-map", "1:a:0",
                   "-c:v", "libx264", "-pix_fmt", "yuv420p",
                   "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
                   out_path]
            _run_fallback(cmd, out_path, scene_img, duration, render_fps)

    else:
        cmd = ["ffmpeg", "-y", "-loop", "1", "-i", scene_img,
               "-f", "lavfi", "-i", "anullsrc=stereo:44100",
               "-t", f"{duration:.3f}", "-vf", f"{VF_NORM},fps=25",
               "-map", "0:v:0", "-map", "1:a:0",
               "-c:v", "libx264", "-pix_fmt", "yuv420p",
               "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
               out_path]
        _run_fallback(cmd, out_path, scene_img, duration, render_fps)

    if os.path.exists(out_path) and os.path.getsize(out_path) >= 1000:
        return out_path, seg_type
    return None, seg_type


def compose_quest(
    work_dir: str,
    pose_images: list[list[str]],
    timeline: list[dict],
    script: dict,
    narration: dict,
    normal_paths: list[str],
    scene_img: str,
    srt_dir: str,
    pad: float = 0.4,
    host_poses: list[str] | None = None,
    char_pose_map: dict[str, list[str]] | None = None,
    host_bg: str | None = None,
    scene_bg_list: list[str] | None = None,
    char_clip_map: dict | None = None,
    sprite_clip_fps: int = 12,
    sprite_take_mode: bool = False,
    render_fps: int = 12,
    show_zh: bool = True,
    workers: int = 1,
    subtitle_font_size: int = 60,
    subtitle_style: dict | None = None,
    progress_cb=None,
    stop_check=None,
) -> str:
    """Compose the final quest video — stop-motion with multi-character + multi-scene.

    Args:
        host_bg: Background image for host segments (TV studio). Falls back to scene_img.
        scene_bg_list: List of background images for dialogue segments.
            Different backgrounds are used for different groups of lines for
            visual variety. Falls back to [scene_img] if None.
        workers: Number of render threads (1=single-thread, 2+=multi-thread, 0=auto=cpu_count).
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

    question_en = script.get("listening_question_en", "")
    question_zh = script.get("listening_question_zh", "")
    hook_en = script.get("hook_intro_en", "")
    host_bg_path = host_bg or scene_img
    scene_bgs = scene_bg_list or [scene_img]
    dialogue = script.get("dialogue", [])

    sm_root = Path(tempfile.gettempdir()) / f"sm_frames_{work.name}"
    sm_root.mkdir(parents=True, exist_ok=True)

    total_segs = len(timeline)
    segments: list[str | None] = [None] * total_segs

    # Resolve workers
    if workers == 0:
        workers = os.cpu_count() or 2
    print(f"  [Quest] Rendering {total_segs} segments with {workers} thread(s)...")

    if workers <= 1:
        # --- Single-thread (original behavior) ---
        for seg_idx, seg in enumerate(timeline):
            if stop_check and stop_check():
                print("  [Quest] Stop requested, aborting segment rendering...")
                break
            out_path, seg_type = _prepare_segment(
                seg_idx, seg, timeline, dialogue, narration,
                normal_paths, host_poses, host_bg_path, scene_bgs,
                char_pose_map, pose_images, scene_img, pad, render_fps,
                tmp_dir, sm_root, char_clip_map, sprite_clip_fps,
                sprite_take_mode,
            )
            if out_path:
                segments[seg_idx] = out_path
            print(f"  Segment {seg_idx + 1}/{total_segs} ({seg_type}) done")
            _cb(int((seg_idx + 1) / total_segs * 80),
                f"  Segment {seg_idx + 1}/{total_segs} ({seg_type}) done")
    else:
        # --- Multi-thread ---
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {
                pool.submit(
                    _prepare_segment,
                    seg_idx, seg, timeline, dialogue, narration,
                    normal_paths, host_poses, host_bg_path, scene_bgs,
                    char_pose_map, pose_images, scene_img, pad, render_fps,
                    tmp_dir, sm_root, char_clip_map, sprite_clip_fps,
                    sprite_take_mode,
                ): seg_idx
                for seg_idx, seg in enumerate(timeline)
            }
            done_count = 0
            for fut in as_completed(futs):
                seg_idx = futs[fut]
                out_path, seg_type = fut.result()
                if out_path:
                    segments[seg_idx] = out_path
                done_count += 1
                print(f"  Segment {done_count}/{total_segs} ({seg_type}) done")
                _cb(int(done_count / total_segs * 80),
                    f"  Segment {done_count}/{total_segs} ({seg_type}) done")
                if stop_check and stop_check():
                    print("  [Quest] Stop requested, cancelling remaining segments...")
                    for f in futs:
                        f.cancel()
                    break

    # If stopped, skip concat/subtitles/loudnorm — segments are preserved for resume
    if stop_check and stop_check():
        print("  [Quest] Compose interrupted, segments saved for resume.")
        return ""

    # 渲染失败的段插入静音占位段（精确时长，fps=25），保住时间轴完整性。
    # 丢弃该段会让之后所有字幕整体错位数秒；占位段用 ph_ 前缀命名，
    # 不会污染 _prepare_segment 的断点续传缓存（重跑时会重新渲染真实段）。
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
        print(f"  [Quest] WARNING: segment {i + 1} ({seg['type']}) failed — "
              f"inserting silent placeholder ({seg['duration']:.2f}s)")
        _run_fallback(ph_cmd, ph_path, scene_img, seg["duration"], render_fps)
        if os.path.exists(ph_path) and os.path.getsize(ph_path) >= 1000:
            segments[i] = ph_path
    if missing:
        print(f"  [Quest] {len(missing)} failed segment(s) handled with placeholders to preserve sync")

    # Filter out remaining None segments (placeholder also failed)
    segments = [s for s in segments if s is not None]

    # --- Concat all segments ---
    _cb(80, "Concatenating segments...")
    no_sub = str(vid_dir / "final_no_sub.mp4")
    concat_segments(segments, no_sub, tmp_dir=tmp_dir)

    # --- Burn subtitles via Pillow overlay (dialogue entries only) ---
    _cb(90, "Burning subtitles (Pillow overlay)...")
    final_path = burn_subtitles(no_sub, timeline, script, str(work), srt_dir, pad, _cb, show_zh=show_zh, en_font_size=subtitle_font_size, zh_font_size=int(subtitle_font_size * 0.85), out_fps=25, style=subtitle_style)

    shutil.rmtree(tmp_dir, ignore_errors=True)

    # --- Final loudnorm pass ---
    _cb(95, "Final loudnorm pass (normalize volume)...")
    apply_final_loudnorm(final_path, str(vid_dir))

    # --- Sync QA report (planned timeline vs detected audio pauses) ---
    try:
        _cb(97, "Writing sync QA report...")
        report = write_sync_report(final_path, timeline, str(work), pad)
        print(f"  [Quest] Sync QA: {report.get('status')} "
              f"duration_drift={report.get('duration_drift_ms', '?')}ms "
              f"max_boundary_dev={report.get('max_boundary_dev_ms', '?')}ms "
              f"({report.get('boundaries_matched', 0)}/{report.get('voiced_segments', 0)} boundaries)")
    except Exception as e:
        print(f"  [Quest] Sync QA report failed (non-fatal): {e}")

    size_mb = os.path.getsize(final_path) / (1024 * 1024)
    _cb(100, f"Quest video done: {final_path} ({size_mb:.1f}MB)")
    return final_path
