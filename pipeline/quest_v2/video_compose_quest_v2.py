"""Quest V2 FFmpeg video composition — quest + 唇形同步 + 观感修复。

与 quest 的差异（quest 原模块零改动，本模块独立实现）：
  1. 唇形同步：每个角色配对「张嘴/闭嘴」双态姿势图（pose_{char}_{j}.png /
     pose_{char}_{j}_c.png），说话人按逐帧音频 RMS 包络用 Image.blend 实时
     混合 —— 两版仅嘴部不同，全图混合视觉上只有嘴在动（PNG-Tuber 双态口型）。
  2. 位置稳定：角色按身份固定站位（canonical 顺序分槽位），不再随
     speaker/listener 动态互换左右；说话者以 scale×1.06 + 前移 10px 强调。
  3. 听者去除假"眨眼"（原用 surprised 姿势替代），仅保留呼吸浮动。
  4. 姿势调度数据源改用 audio_envelope（PCM+numpy 逐帧 RMS，
     替代 astats stderr 解析 hack）。
  5. seed 固定映射（hash() 随机化导致续跑不可复现的修复）。

结构同 quest：Ch1 Welcome → Ch2 Hook → Ch3 慢速对话（4幕）→ Ch4 Outro & CTA，
全部段落停格渲染 + 全局字幕烧录 + loudnorm。
"""
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_PARENT = str(Path(__file__).parent.parent.resolve())
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from media_utils import (
    VF_NORM,
    concat_segments, burn_subtitles, apply_final_loudnorm,
    write_sync_report,
)
from audio_envelope import extract_rms_envelope, mouth_blend_weights

# 复用 quest 的兜底编码助手（只引用，不改动 quest 模块）
from quest.video_compose_quest import _run_fallback

# ---------------------------------------------------------------------------
# 站位（按角色身份固定）+ 说话者强调
# ---------------------------------------------------------------------------
_CHAR_RANK = {"char_a": 0, "char_b": 1, "char_c": 2, "host": 3}
_SLOTS = {
    1: [0.5],
    2: [0.35, 0.65],
    3: [0.30, 0.55, 0.80],
}
SPEAKER_SCALE = 1.06   # 说话者放大强调
SPEAKER_DY = -10.0     # 说话者前移（上移）
LISTENER_SCALE = 0.98


def _char_positions_v2(char_keys: list[str]) -> list[float]:
    """按 canonical 角色顺序分配固定站位（相对画布宽度的比例）。

    同一组 on_screen 角色，无论谁是 speaker，站位恒定 —— 修复 quest 中
    换说话人时两角色左右瞬移的问题（违反 180° 轴线原则）。
    返回与 char_keys 顺序对齐的位置列表。
    """
    keys = list(char_keys)
    n = len(keys)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: _CHAR_RANK.get(keys[i], 99))
    if n in _SLOTS:
        slots = _SLOTS[n]
    elif n > 3:
        slots = [0.25 + i * (0.60 / (n - 1)) for i in range(n)]
    else:
        slots = [0.5]
    positions = [0.0] * n
    for rank, key_idx in enumerate(order):
        positions[key_idx] = slots[rank]
    return positions


# ---------------------------------------------------------------------------
# 姿势调度（数据源：audio_envelope）
# ---------------------------------------------------------------------------

def _build_pose_schedule_v2(n_poses: int, total_frames: int, fps: float,
                            seed: int, is_speaker: bool = True,
                            audio_file: str | None = None) -> list[tuple[int, int]]:
    """姿势调度：说话人在音频低能量点（停顿）换姿势，听者静止。

    与 quest 版差异：数据源改用 audio_envelope（PCM+numpy 逐帧 RMS），
    阈值语义不变（低于段内最大值 40% 视为停顿，最短持有 2s）。
    """
    rng = random.Random(seed)
    if n_poses <= 1 or not is_speaker:
        return [(0, 0)]

    if audio_file and os.path.exists(audio_file):
        env = extract_rms_envelope(audio_file, fps)
        if env:
            min_hold_frames = round(2.0 * fps)
            schedule = [(0, 0)]
            last_switch = 0
            for i, val in enumerate(env):
                frame = i
                if frame >= total_frames:
                    break
                if frame - last_switch < min_hold_frames:
                    continue
                if val < 0.4:
                    current = schedule[-1][0]
                    candidates = [j for j in range(n_poses) if j != current]
                    schedule.append((rng.choice(candidates), frame))
                    last_switch = frame
            return schedule

    # 兜底：随机 2.0-5.0s 持有
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


# ---------------------------------------------------------------------------
# 停格段渲染（唇同步 + 固定站位 + 无假眨眼）
# ---------------------------------------------------------------------------

def _render_sm_segment_v2(
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
    lip_sync: bool = True,
) -> bool:
    """渲染一段停格视频（quest_v2 核心渲染器）。

    char_layers 每项：
        {"poses": list[str], "poses_closed": list[str](可空),
         "is_speaker": bool, "char_key": str}

    唇同步：说话人每个姿势有张嘴(open)/闭嘴(closed)配对图，按逐帧音频
    包络权重 Image.blend 混合；配对不齐则整体放弃（宁缺毋滥，回退现状）。

    Returns True on success, False on failure.
    """
    from stop_motion import (
        remove_bg, normalize_pose, compute_landing, POSE_CENTER_Y,
        transform_pose, paste_with_shadow, generate_morph_frames,
    )
    from PIL import Image as PILImage

    frames_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)  # 抠图缓存目录自建（防御式）

    # 统计待抠图数量（含 closed 配对），打印与 quest 一致的进度提示
    all_paths: list[str] = []
    for layer in char_layers:
        all_paths += [p for p in layer.get("poses", []) if os.path.exists(p)]
        all_paths += [p for p in (layer.get("poses_closed") or []) if os.path.exists(p)]
    _uncached = [p for p in all_paths
                 if not (cache_dir / f"cutout_{Path(p).stem}.png").exists()]
    if _uncached:
        print(f"    rembg {len(_uncached)} pose(s)...", flush=True)

    def _cutout(p_path: str):
        cache_path = cache_dir / f"cutout_{Path(p_path).stem}.png"
        if cache_path.exists():
            return PILImage.open(cache_path).convert("RGBA")
        try:
            raw = PILImage.open(p_path)
            alpha = remove_bg(raw)
            norm = normalize_pose(alpha)
            norm.save(str(cache_path))
            return norm
        except Exception as e:
            print(f"  [QuestV2] rembg error for {p_path}: {e}")
            return None

    # 处理各角色姿势（rembg + 归一化 + 缓存）；closed 配对必须与 open 等长
    processed_layers: list[dict] = []
    for layer in char_layers:
        pose_paths = [p for p in layer.get("poses", []) if os.path.exists(p)]
        closed_paths = [p for p in (layer.get("poses_closed") or [])
                        if os.path.exists(p)]
        processed = [img for img in (_cutout(p) for p in pose_paths)
                     if img is not None]
        processed_closed = [img for img in (_cutout(p) for p in closed_paths)
                            if img is not None]
        if len(processed_closed) != len(processed):
            processed_closed = []  # 配对不齐 → 放弃该层唇同步
        if not processed:
            processed = [PILImage.new("RGBA", (1280, 720), (0, 0, 0, 0))]
        processed_layers.append({
            "poses": processed,
            "poses_closed": processed_closed,
            "is_speaker": layer.get("is_speaker", False),
        })

    # 背景
    bg_img = PILImage.open(bg_img_path).convert("RGBA").resize((1280, 720))

    # 站位：按角色身份固定（未知 key 按传入顺序占槽）
    char_keys = [layer.get("char_key") or f"slot{i}"
                 for i, layer in enumerate(char_layers)]
    positions = _char_positions_v2(char_keys)

    total_frames = max(1, math.ceil(duration * render_fps))
    schedules = []
    for i, layer in enumerate(processed_layers):
        af = audio_file if layer["is_speaker"] else None
        s = _build_pose_schedule_v2(
            len(layer["poses"]), total_frames, render_fps,
            seed + i * 100, is_speaker=layer["is_speaker"], audio_file=af)
        schedules.append(s)

    # 光流 morph 帧预计算（仅说话人的姿势过渡，open 集）
    morph_cache: dict[int, dict] = {}
    for li, layer in enumerate(processed_layers):
        if not layer["is_speaker"]:
            continue
        poses = layer["poses"]
        sched = schedules[li]
        if len(poses) < 2 or len(sched) < 2:
            continue
        morph_cache[li] = {}
        for ti in range(len(sched) - 1):
            pa, pb = poses[sched[ti][0]], poses[sched[ti + 1][0]]
            if pa.size == pb.size:
                try:
                    morph_cache[li][ti] = generate_morph_frames(pa, pb, n_frames=5)
                except Exception:
                    pass

    # 唇同步：段级一次性提取包络与混合权重（首个有 closed 配对的说话人）
    blend_weights: list[float] | None = None
    lipsync_char = ""
    if lip_sync and audio_file and os.path.exists(audio_file):
        for li, layer in enumerate(processed_layers):
            if layer["is_speaker"] and len(layer.get("poses_closed", [])) >= 1:
                env = extract_rms_envelope(audio_file, render_fps)
                if env:
                    blend_weights = mouth_blend_weights(env, fps=render_fps)
                    lipsync_char = char_keys[li]
                break

    MORPH_DUR = 0.2  # seconds for 5-frame transition
    MORPH_N = 5
    cy = POSE_CENTER_Y
    open_frames = 0

    for fidx in range(total_frames):
        if fidx % 10 == 0:
            print(f"    frame {fidx}/{total_frames}", flush=True)
        t = fidx / render_fps
        canvas = bg_img.copy()

        for li, layer in enumerate(processed_layers):
            poses = layer["poses"]
            closed_poses = layer.get("poses_closed") or []
            schedule = schedules[li]
            # positions 为画布宽度比例（quest 原版存像素，此处统一比例→像素）
            x = (positions[li] * 1280) if li < len(positions) else 1280 * 0.5

            # 当前姿势阶段
            stage_idx = 0
            for si, (pi, start_frame) in enumerate(schedule):
                if fidx >= start_frame:
                    stage_idx = si
                else:
                    break
            pose_idx = schedule[stage_idx][0]
            stage_start_frame = schedule[stage_idx][1]
            local_t = (fidx - stage_start_frame) / render_fps

            # 光流 morph 过渡帧
            in_morph = False
            if (li in morph_cache and stage_idx < len(schedule) - 1
                    and stage_idx in morph_cache[li]):
                next_start = schedule[stage_idx + 1][1]
                frames_to_next = next_start - fidx
                morph_frames_needed = round(MORPH_DUR * render_fps)
                if 0 < frames_to_next <= morph_frames_needed:
                    morph_frames = morph_cache[li][stage_idx]
                    morph_progress = 1.0 - (frames_to_next / morph_frames_needed)
                    morph_idx = min(MORPH_N - 1, int(morph_progress * MORPH_N))
                    pose = morph_frames[morph_idx]
                    in_morph = True

            # 唇同步混合：非 morph 帧 + 配对就绪（权重越界帧=尾静音，闭嘴）
            if (not in_morph and blend_weights is not None
                    and layer["is_speaker"] and closed_poses
                    and pose_idx < len(closed_poses)):
                w = blend_weights[fidx] if fidx < len(blend_weights) else 0.0
                if w > 0.0:
                    pose = PILImage.blend(closed_poses[pose_idx],
                                          poses[pose_idx], w)
                    open_frames += 1
                else:
                    pose = closed_poses[pose_idx]
            elif not in_morph:
                pose = poses[pose_idx]

            # 变换：说话者强调（放大+前移），听者呼吸浮动（无假眨眼）
            if layer["is_speaker"]:
                if in_morph:
                    scale = SPEAKER_SCALE
                    px = x
                    py = cy + SPEAKER_DY
                    rot = 0.0
                else:
                    landing = compute_landing(local_t, direction=direction)
                    scale = SPEAKER_SCALE * (1.0 + landing["scale"])
                    px = x + landing["x"]
                    py = cy + SPEAKER_DY + landing["y"]
                    rot = landing["rotation"]
            else:
                breath_y = 2.0 * math.sin(t * 2 * math.pi / 3.3)
                pose = poses[1] if len(poses) > 1 else poses[0]
                scale = LISTENER_SCALE
                px = x
                py = cy + breath_y
                rot = 0.0

            sprite = transform_pose(pose, scale=scale, rotation=rot)
            paste_with_shadow(canvas, sprite, px, py, centered=True)

        frame = canvas.convert("RGB")
        if overlay_path and os.path.exists(overlay_path):
            ov = PILImage.open(overlay_path).convert("RGBA")
            frame_rgba = frame.convert("RGBA")
            frame_rgba.alpha_composite(ov, (0, 0))
            frame = frame_rgba.convert("RGB")
        frame.save(str(frames_dir / f"frame-{fidx:04d}.png"), compress_level=2)

    if blend_weights is not None:
        print(f"    lipsync on: {lipsync_char} "
              f"open {open_frames}/{total_frames} frames")

    # 编码：输出 25fps 最小化时长量化误差（同 quest）
    frame_pattern = str(frames_dir / "frame-%04d.png")
    if audio_file and os.path.exists(audio_file):
        cmd = ["ffmpeg", "-y",
               "-framerate", str(render_fps), "-i", frame_pattern,
               "-i", audio_file,
               "-t", f"{duration:.3f}", "-map", "0:v:0", "-map", "1:a:0",
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "25",
               "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
               "-af", f"{fade_af},apad=whole_dur={duration:.3f}",
               out_path]
    else:
        cmd = ["ffmpeg", "-y",
               "-framerate", str(render_fps), "-i", frame_pattern,
               "-f", "lavfi", "-i", "anullsrc=stereo:44100",
               "-t", f"{duration:.3f}", "-map", "0:v:0", "-map", "1:a:0",
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "25",
               "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
               out_path]

    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=600)
        if r.returncode != 0:
            print(f"  [QuestV2] FFmpeg error: {r.stderr[-200:]}")
            return False
    except Exception as e:
        print(f"  [QuestV2] Error: {e}")
        return False
    finally:
        shutil.rmtree(frames_dir, ignore_errors=True)
    return True


# ---------------------------------------------------------------------------
# 段准备（seed 固定映射 + closed 配对装配）
# ---------------------------------------------------------------------------

_SEG_SEED = {"welcome": 11, "hook_intro": 22, "outro": 33}


def _prepare_segment_v2(seg_idx, seg, timeline, dialogue, narration,
                        normal_paths, host_poses, host_closed, host_bg_path,
                        scene_bgs, char_pose_map, pose_images, scene_img,
                        pad, render_fps, tmp_dir, sm_root, lip_sync=True):
    """准备并渲染单个段落。Returns (out_path or None, seg_type)。"""
    seg_type = seg["type"]
    duration = seg["duration"]
    audio_idx = seg.get("audio_index", 0)
    out_path = str(tmp_dir / f"seg_{seg_idx:03d}.mp4")

    # 断点续传缓存（与 quest 同名同位置，互相兼容）
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
        h_closed = host_closed or []
        if len(h_closed) < len(h_poses):
            h_closed = []
        char_layers = [{"poses": h_poses, "poses_closed": h_closed,
                        "is_speaker": True, "char_key": "host"}]
        frames_dir = sm_root / f"{seg_type}_{seg_idx}"
        direction = 1 if seg_idx % 2 == 0 else -1
        success = _render_sm_segment_v2(
            char_layers, host_bg_path, audio_file, out_path, duration,
            frames_dir, sm_root, render_fps=render_fps,
            seed=_SEG_SEED.get(seg_type, 44) + seg_idx,
            direction=direction, fade_af=fade_af, lip_sync=lip_sync,
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
        # 场景背景轮换（同 quest：每 5 行换一张）
        dialogue_seg_count = sum(1 for s in timeline[:timeline.index(seg)]
                                 if s["type"] == "dialogue")
        n_scene_bgs = max(1, len(scene_bgs))
        bg_idx = (dialogue_seg_count // 5) % n_scene_bgs
        line_bg = scene_bgs[bg_idx]

        line_data = dialogue[audio_idx] if audio_idx < len(dialogue) else {}
        speaker = line_data.get("speaker", "char_a")
        on_screen = line_data.get("on_screen") or [speaker]

        if char_pose_map:
            char_layers = []
            for char_key in on_screen:
                entry = char_pose_map.get(char_key) or {}
                poses = entry.get("poses") or []
                closed = entry.get("closed") or []
                if not poses:
                    fallback = char_pose_map.get("char_a") or {}
                    poses = fallback.get("poses") or [line_bg]
                    closed = []
                char_layers.append({
                    "poses": poses,
                    "poses_closed": closed,
                    "is_speaker": (char_key == speaker),
                    "char_key": char_key,
                })
        else:
            idx = min(audio_idx, len(pose_images) - 1) if pose_images else 0
            line_poses = (pose_images[idx]
                          if pose_images and idx < len(pose_images) else [line_bg])
            char_layers = [{"poses": line_poses, "poses_closed": [],
                            "is_speaker": True, "char_key": ""}]

        frames_dir = sm_root / f"dialogue_{audio_idx}"
        direction = 1 if audio_idx % 2 == 0 else -1
        success = _render_sm_segment_v2(
            char_layers, line_bg, audio_file, out_path, duration,
            frames_dir, sm_root, render_fps=render_fps,
            seed=audio_idx * 7 + 13,
            direction=direction, fade_af=fade_af, lip_sync=lip_sync,
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


# ---------------------------------------------------------------------------
# 主合成入口
# ---------------------------------------------------------------------------

def compose_quest_v2(
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
    char_pose_map: dict | None = None,
    host_bg: str | None = None,
    scene_bg_list: list[str] | None = None,
    render_fps: int = 12,
    show_zh: bool = True,
    workers: int = 1,
    subtitle_font_size: int = 60,
    subtitle_style: dict | None = None,
    progress_cb=None,
    stop_check=None,
    lip_sync: bool = True,
    host_closed: list[str] | None = None,
) -> str:
    """Compose the final quest_v2 video — quest + lipsync + 观感修复.

    Args:
        char_pose_map: 每角色 -> {"poses": [...], "closed": [...]}（closed 可空，
            兼容旧式 list 值——视为无闭嘴配对）。
        host_closed: host 的闭嘴配对图列表（与 host_poses 等长才生效）。
        lip_sync: 关闭则忽略所有 closed 配对（回退 quest 同款渲染）。
        workers: 渲染线程数（1=单线程, 2+=多线程, 0=自动=cpu_count）。
    """
    def _cb(pct, msg):
        if progress_cb:
            progress_cb(pct, msg)

    work = Path(work_dir)
    tmp_dir = work / "tmp_segments"
    vid_dir = work / "videos"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    vid_dir.mkdir(parents=True, exist_ok=True)

    host_bg_path = host_bg or scene_img
    scene_bgs = scene_bg_list or [scene_img]
    dialogue = script.get("dialogue", [])

    # char_pose_map 值归一化为 {"poses": [...], "closed": [...]}
    norm_map: dict[str, dict] = {}
    for k, v in (char_pose_map or {}).items():
        if isinstance(v, dict):
            norm_map[k] = {"poses": list(v.get("poses") or []),
                           "closed": list(v.get("closed") or [])}
        elif isinstance(v, list):
            norm_map[k] = {"poses": list(v), "closed": []}
    char_pose_map = norm_map

    host_closed = list(host_closed or [])
    if host_poses and len(host_closed) < len(host_poses):
        host_closed = []

    sm_root = Path(tempfile.gettempdir()) / f"sm_frames_{work.name}"
    sm_root.mkdir(parents=True, exist_ok=True)

    total_segs = len(timeline)
    segments: list[str | None] = [None] * total_segs

    if workers == 0:
        workers = os.cpu_count() or 2
    print(f"  [QuestV2] Rendering {total_segs} segments with {workers} thread(s)..."
          + ("" if lip_sync else " [lipsync OFF]"))

    # --- 预处理全部 rembg 抠图（单线程，避免 onnxruntime 并发问题）---
    if workers > 1:
        from stop_motion import remove_bg, normalize_pose
        from PIL import Image as _PIL
        _all_pose_paths: set[str] = set()
        for entry in char_pose_map.values():
            _all_pose_paths.update(entry["poses"])
            _all_pose_paths.update(entry["closed"])
        if host_poses:
            _all_pose_paths.update(host_poses)
        if host_closed:
            _all_pose_paths.update(host_closed)
        for line_poses in (pose_images or []):
            _all_pose_paths.update(line_poses)

        _rembg_count = 0
        for p_path in sorted(_all_pose_paths):
            if not os.path.exists(p_path):
                continue
            cache_path = sm_root / f"cutout_{Path(p_path).stem}.png"
            if cache_path.exists():
                continue
            try:
                raw = _PIL.open(p_path)
                alpha = remove_bg(raw)
                norm = normalize_pose(alpha)
                norm.save(str(cache_path))
                _rembg_count += 1
            except Exception as e:
                print(f"  [QuestV2] rembg error for {p_path}: {e}")
        if _rembg_count:
            print(f"  [QuestV2] Pre-processed {_rembg_count} pose cutouts (single-thread rembg)")

    if workers <= 1:
        # --- 单线程（与 quest 行为一致）---
        for seg_idx, seg in enumerate(timeline):
            if stop_check and stop_check():
                print("  [QuestV2] Stop requested, aborting segment rendering...")
                break
            out_path, seg_type = _prepare_segment_v2(
                seg_idx, seg, timeline, dialogue, narration,
                normal_paths, host_poses, host_closed, host_bg_path, scene_bgs,
                char_pose_map, pose_images, scene_img, pad, render_fps,
                tmp_dir, sm_root, lip_sync=lip_sync,
            )
            if out_path:
                segments[seg_idx] = out_path
            print(f"  Segment {seg_idx + 1}/{total_segs} ({seg_type}) done")
            _cb(int((seg_idx + 1) / total_segs * 80),
                f"  Segment {seg_idx + 1}/{total_segs} ({seg_type}) done")
    else:
        # --- 多线程 ---
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {
                pool.submit(
                    _prepare_segment_v2,
                    seg_idx, seg, timeline, dialogue, narration,
                    normal_paths, host_poses, host_closed, host_bg_path, scene_bgs,
                    char_pose_map, pose_images, scene_img, pad, render_fps,
                    tmp_dir, sm_root, lip_sync=lip_sync,
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
                    print("  [QuestV2] Stop requested, cancelling remaining segments...")
                    for f in futs:
                        f.cancel()
                    break

    if stop_check and stop_check():
        print("  [QuestV2] Compose interrupted, segments saved for resume.")
        return ""

    # 渲染失败的段插入静音占位段（保住时间轴完整性，同 quest）
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
        print(f"  [QuestV2] WARNING: segment {i + 1} ({seg['type']}) failed — "
              f"inserting silent placeholder ({seg['duration']:.2f}s)")
        _run_fallback(ph_cmd, ph_path, scene_img, seg["duration"], render_fps)
        if os.path.exists(ph_path) and os.path.getsize(ph_path) >= 1000:
            segments[i] = ph_path
    if missing:
        print(f"  [QuestV2] {len(missing)} failed segment(s) handled with placeholders to preserve sync")

    segments = [s for s in segments if s is not None]

    # --- Concat ---
    _cb(80, "Concatenating segments...")
    no_sub = str(vid_dir / "final_no_sub.mp4")
    concat_segments(segments, no_sub, tmp_dir=tmp_dir)

    # --- Pillow 字幕烧录 ---
    _cb(90, "Burning subtitles (Pillow overlay)...")
    final_path = burn_subtitles(
        no_sub, timeline, script, str(work), srt_dir, pad, _cb,
        show_zh=show_zh, en_font_size=subtitle_font_size,
        zh_font_size=int(subtitle_font_size * 0.85), out_fps=25,
        style=subtitle_style)

    shutil.rmtree(tmp_dir, ignore_errors=True)

    # --- 最终响度归一 ---
    _cb(95, "Final loudnorm pass (normalize volume)...")
    apply_final_loudnorm(final_path, str(vid_dir))

    # --- 音画同步 QA 报告 ---
    try:
        _cb(97, "Writing sync QA report...")
        report = write_sync_report(final_path, timeline, str(work), pad)
        print(f"  [QuestV2] Sync QA: {report.get('status')} "
              f"duration_drift={report.get('duration_drift_ms', '?')}ms "
              f"max_boundary_dev={report.get('max_boundary_dev_ms', '?')}ms "
              f"({report.get('boundaries_matched', 0)}/{report.get('voiced_segments', 0)} boundaries)")
    except Exception as e:
        print(f"  [QuestV2] Sync QA report failed (non-fatal): {e}")

    size_mb = os.path.getsize(final_path) / (1024 * 1024)
    _cb(100, f"QuestV2 video done: {final_path} ({size_mb:.1f}MB)")
    return final_path
