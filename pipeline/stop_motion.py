"""Stop-motion animation renderer — Phase 2 (layer separation) + Phase 3 (optical flow).

Inspired by the image-motion-animation skill's render_semantic_cartoon.py and
morph_rgba_pair.py. Renders multi-pose character animation over separated
background layers with Pillow, using alpha-aware bidirectional optical flow
for flow-safe pose transitions.

Architecture:
  1. Character pose images (generated as cutouts, transparent) are normalized
     to a shared canvas with center anchoring.
  2. For each dialogue segment, poses switch at semantic boundaries with
     a landing transform (scale/pan/bounce decay).
  3. Adjacent poses of the same speaker/framing are classified flow-safe
     and interpolated with OpenCV DISOpticalFlow. Incompatible pairs use
     hard-cut at midpoint.
  4. Frames are rendered as PNGs → FFmpeg encodes each segment → concat.
"""
from __future__ import annotations

import math
import os
import subprocess
from pathlib import Path

from PIL import Image, ImageFilter

from media_utils import TARGET_W, TARGET_H

# Try importing cv2/numpy for optical flow (Phase 3). Falls back gracefully.
try:
    import cv2
    import numpy as np
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DELIVERY_FPS = 24       # final output fps
LANDING_DURATION = 0.3  # seconds — landing transform decay
LANDING_X = 14          # px — horizontal offset
LANDING_Y = 10          # px — vertical sine bounce
LANDING_SCALE = 0.018   # scale delta
LANDING_ROTATION = 1.4  # degrees

# Pose canvas for normalization (transparent, larger than target to avoid clipping)
POSE_CANVAS_W = TARGET_W
POSE_CANVAS_H = TARGET_H
POSE_TARGET_H = TARGET_H - 100  # leave room for subtitle area at bottom
# For half-body close-ups, center vertically; keep character fully in-frame
POSE_CENTER_Y = TARGET_H // 2 + 60  # slightly below center to leave room for subtitles
POSE_BOTTOM = POSE_CENTER_Y  # kept for backward compat

# Shadow
SHADOW_RGB = (145, 141, 133)
SHADOW_OPACITY = 0.40
SHADOW_BLUR = 1.2
SHADOW_OFFSET_X = 8
SHADOW_OFFSET_Y = 10


# ---------------------------------------------------------------------------
# Phase 2: background handling (images are pre-cutout at generation time) + pose normalization
# ---------------------------------------------------------------------------


def _has_transparency(img: Image.Image, threshold: float = 0.05) -> bool:
    """Check if an image already has significant transparency.

    Handles P-mode (palette transparency from is_segmentation=true) and RGBA.
    Returns True if >threshold fraction of pixels are fully transparent.
    """
    import numpy as _np
    if img.mode == "P" and "transparency" in img.info:
        rgba = img.convert("RGBA")
    elif img.mode == "RGBA":
        rgba = img
    else:
        return False
    alpha = _np.array(rgba.getchannel("A"))
    return bool((_np.count_nonzero(alpha == 0) / alpha.size) > threshold)


def remove_bg(img: Image.Image) -> Image.Image:
    """Return the image as RGBA with transparency.

    Pose images are generated as cutouts (is_segmentation=true), so the fast
    path is a plain RGBA conversion. For legacy opaque inputs falls back to
    luminance-threshold white removal.
    """
    # Fast path: image already has transparency (from is_segmentation=true)
    if _has_transparency(img):
        return img.convert("RGBA")
    # 新增选项：MODNet AI 抠图（engine=modnet，或 auto 且权重可用时）
    try:
        import matting as _matting
        engine = _matting.resolve_engine()
        if engine in ("auto", "modnet"):
            if _matting.matting_available():
                return _matting.matting_remove_bg(img)
            if engine == "modnet":
                print("  [Matting] MODNet 权重不可用，回退白度抠图")
    except Exception as e:
        print(f"  [Matting] MODNet 抠图失败，回退白度抠图: {e}")
    return _remove_white_bg_fallback(img)


def _remove_white_bg_fallback(img: Image.Image, threshold: int = 238) -> Image.Image:
    """Threshold-based white background removal (legacy opaque inputs)."""
    rgba = img.convert("RGBA")
    data = rgba.load()
    w, h = rgba.size
    for y in range(h):
        for x in range(w):
            r, g, b, a = data[x, y]
            if r >= threshold and g >= threshold and b >= threshold:
                data[x, y] = (r, g, b, 0)
            elif r >= threshold - 30 and g >= threshold - 30 and b >= threshold - 30:
                whiteness = min(r, g, b)
                alpha = max(0, int(255 * (1 - (whiteness - (threshold - 30)) / 30)))
                data[x, y] = (r, g, b, alpha)
    return rgba


def normalize_pose(img: Image.Image,
                   canvas_w: int = POSE_CANVAS_W,
                   canvas_h: int = POSE_CANVAS_H,
                   target_h: int = POSE_TARGET_H,
                   bottom: int = POSE_BOTTOM) -> Image.Image:
    """Scale an RGBA character image into a shared transparent canvas.

    For half-body close-ups, the character is centered vertically rather
    than bottom-anchored, since there are no feet to align.
    """
    source = img.convert("RGBA")
    # Scale to target height, preserving aspect ratio
    scale = target_h / source.height
    tw = round(source.width * scale)
    th = target_h
    if tw > canvas_w:
        # If too wide, scale by width instead
        scale = canvas_w / source.width
        tw = canvas_w
        th = round(source.height * scale)
    sprite = source.resize((tw, th), Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    left = round((canvas_w - sprite.width) / 2)
    # Center vertically for half-body close-ups
    top = round((canvas_h - sprite.height) / 2)
    canvas.alpha_composite(sprite, (left, top))
    return canvas


# ---------------------------------------------------------------------------
# Phase 3: Alpha-aware bidirectional optical flow
# ---------------------------------------------------------------------------

def _flow_gray(rgba: "np.ndarray") -> "np.ndarray":
    """Convert RGBA to grayscale on a neutral composite for optical flow."""
    alpha = rgba[:, :, 3:4].astype("float32") / 255.0
    rgb = rgba[:, :, :3].astype("float32")
    composite = rgb * alpha + 232.0 * (1.0 - alpha)
    return cv2.cvtColor(composite.astype("uint8"), cv2.COLOR_RGB2GRAY)


def _estimate_flow(first: "np.ndarray", second: "np.ndarray"):
    """Estimate bidirectional DISOpticalFlow between two RGBA numpy arrays."""
    estimator = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    estimator.setFinestScale(1)
    forward = estimator.calc(_flow_gray(first), _flow_gray(second), None)
    backward = estimator.calc(_flow_gray(second), _flow_gray(first), None)
    return forward, backward


def _warp(source: "np.ndarray", flow: "np.ndarray", amount: float) -> "np.ndarray":
    """Warp source RGBA by *amount* along *flow*."""
    rows, cols = source.shape[:2]
    grid_x, grid_y = np.meshgrid(
        np.arange(cols, dtype="float32"), np.arange(rows, dtype="float32")
    )
    return cv2.remap(
        source,
        grid_x - flow[:, :, 0] * amount,
        grid_y - flow[:, :, 1] * amount,
        cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def _morph(first: "np.ndarray", second: "np.ndarray",
           forward, backward, progress: float) -> "np.ndarray":
    """Alpha-aware bidirectional morph at *progress* (0→1)."""
    if progress <= 0:
        return first
    if progress >= 1:
        return second
    a = _warp(first, forward, progress).astype("float32") / 255.0
    b = _warp(second, backward, 1.0 - progress).astype("float32") / 255.0
    a[:, :, :3] *= a[:, :, 3:4]
    b[:, :, :3] *= b[:, :, 3:4]
    mixed = a * (1.0 - progress) + b * progress
    alpha = mixed[:, :, 3:4]
    mixed[:, :, :3] = np.divide(mixed[:, :, :3], np.maximum(alpha, 1e-6))
    return np.clip(mixed * 255.0, 0, 255).astype("uint8")


def generate_morph_frames(pose_a: Image.Image, pose_b: Image.Image,
                           n_frames: int = 5) -> list[Image.Image]:
    """Generate intermediate frames between two poses via optical flow.

    Returns list of RGBA Images including endpoints. Falls back to [a, b]
    (hard-cut) if cv2 is not available.
    """
    if not _HAS_CV2 or n_frames < 2:
        return [pose_a, pose_b]
    arr_a = np.asarray(pose_a).copy()
    arr_b = np.asarray(pose_b).copy()
    forward, backward = _estimate_flow(arr_a, arr_b)
    frames = []
    for i in range(n_frames):
        p = i / (n_frames - 1)
        arr = _morph(arr_a, arr_b, forward, backward, p)
        frames.append(Image.fromarray(arr))
    return frames


# ---------------------------------------------------------------------------
# Frame renderer (ported from render_semantic_cartoon.py)
# ---------------------------------------------------------------------------


def transform_pose(source: Image.Image, scale: float = 1.0,
                   rotation: float = 0.0) -> Image.Image:
    """Resize and rotate a pose image."""
    if scale != 1.0:
        w = round(source.width * scale)
        h = round(source.height * scale)
        source = source.resize((w, h), Image.Resampling.LANCZOS)
    if rotation:
        source = source.rotate(rotation, resample=Image.Resampling.BICUBIC,
                               expand=True)
    return source


def paste_with_shadow(canvas: Image.Image, image: Image.Image,
                      x: float, y: float, *, centered: bool = True) -> None:
    """Paste a character image onto canvas with soft shadow.

    When centered=True (half-body close-ups), y is the vertical CENTER
    of the image. When centered=False (full-body), y is the BOTTOM edge
    (legacy bottom-anchored mode).
    """
    left = round(x - image.width / 2)
    if centered:
        top = round(y - image.height / 2)
    else:
        top = round(y - image.height)
    # Shadow
    alpha = image.getchannel("A").filter(ImageFilter.GaussianBlur(SHADOW_BLUR))
    shadow = Image.new("RGBA", image.size, (*SHADOW_RGB, 255))
    shadow.putalpha(alpha.point(lambda v: round(v * SHADOW_OPACITY)))
    canvas.alpha_composite(shadow, (left + SHADOW_OFFSET_X, top + SHADOW_OFFSET_Y))
    canvas.alpha_composite(image, (left, top))


def render_frame(background: Image.Image, character: Image.Image | None,
                 x: float, y: float, scale: float = 1.0,
                 rotation: float = 0.0, *, centered: bool = True) -> Image.Image:
    """Render a single frame: background + character (with shadow).

    Args:
        background: Pre-loaded RGBA background image (already at TARGET_W×TARGET_H).
        character: RGBA character pose (already normalized to canvas). None = no character.
        x: Character center X position.
        y: Character center Y position (when centered=True) or bottom (when False).
        scale: Character scale factor.
        rotation: Character rotation in degrees.
        centered: If True, y is the vertical center (half-body mode).
                  If False, y is the bottom edge (full-body mode).

    Returns:
        RGB Image at TARGET_W×TARGET_H.
    """
    canvas = background.copy()
    if character is not None:
        pose = transform_pose(character, scale=scale, rotation=rotation)
        paste_with_shadow(canvas, pose, x, y, centered=centered)
    return canvas.convert("RGB")


def compute_landing(local_time: float, landing_dur: float = LANDING_DURATION,
                    direction: int = 1) -> dict:
    """Compute landing transform parameters at *local_time* since pose start.

    Returns dict with scale, x_offset, y_offset, rotation deltas.
    """
    if landing_dur <= 0 or local_time >= landing_dur:
        return {"scale": 0.0, "x": 0.0, "y": 0.0, "rotation": 0.0}
    landing = max(0.0, 1.0 - local_time / landing_dur)
    phase = min(1.0, local_time / max(landing_dur, 1e-6))
    return {
        "scale": landing * LANDING_SCALE,
        "x": direction * landing * LANDING_X,
        "y": -math.sin(phase * math.pi) * LANDING_Y,
        "rotation": direction * landing * LANDING_ROTATION,
    }


# ---------------------------------------------------------------------------
# Subtitle rendering (reuse from video_compose)
# ---------------------------------------------------------------------------





# ---------------------------------------------------------------------------
# Dialogue segment renderer (multi-pose + optical flow)
# ---------------------------------------------------------------------------

def _render_dialogue_segment(
    background: Image.Image,
    poses: list[Image.Image],
    duration: float,
    audio_dur: float,
    fps: int,
    out_dir: Path,
    line_idx: int,
) -> None:
    """Render PNG frames for a dialogue segment with multi-pose stop-motion.

    Pose switching logic:
      - If 1 pose: hold throughout with landing transform at start.
      - If 2 poses: pose A for first half, pose B for second half.
        If cv2 available and both poses are normalized: optical flow transition
        at the midpoint (5 intermediate frames). Otherwise hard-cut.
      - If 3+ poses: distribute evenly, hard-cut between stages.
    """
    total_frames = round(duration * fps)
    n_poses = len(poses)
    direction = 1 if line_idx % 2 == 0 else -1

    # Character position (centered for half-body close-ups)
    char_x = TARGET_W / 2
    char_bottom = POSE_CENTER_Y

    # Determine pose schedule
    if n_poses == 1:
        schedule = [(0.0, 0)]  # (start_time, pose_index)
    elif n_poses == 2:
        mid = duration * 0.5
        schedule = [(0.0, 0), (mid, 1)]
    else:
        # Even distribution
        step = duration / n_poses
        schedule = [(i * step, i) for i in range(n_poses)]

    # Pre-generate morph frames for flow-safe transitions
    morph_cache: dict[int, list[Image.Image]] = {}
    if _HAS_CV2 and n_poses >= 2:
        for i in range(len(schedule) - 1):
            pose_a_idx = schedule[i][1]
            pose_b_idx = schedule[i + 1][1]
            # Only morph if same framing (both poses exist and are similar size)
            pose_a = poses[pose_a_idx]
            pose_b = poses[pose_b_idx]
            if pose_a.size == pose_b.size:
                try:
                    morph_frames = generate_morph_frames(pose_a, pose_b, n_frames=5)
                    morph_cache[i] = morph_frames
                except Exception as e:
                    print(f"  [StopMotion] Morph failed for line {line_idx} transition {i}: {e}")

    # Morph transition duration
    morph_dur = 0.2  # seconds for 5-frame transition
    morph_n = 5

    for frame_idx in range(total_frames):
        t = frame_idx / fps  # current time in segment

        # Find which pose stage we're in
        stage_idx = 0
        for si, (stage_start, pose_idx) in enumerate(schedule):
            if t >= stage_start:
                stage_idx = si
            else:
                break

        stage_start, pose_idx = schedule[stage_idx]
        local_time = t - stage_start

        # Check if we're in a morph transition
        current_pose = poses[pose_idx]
        in_morph = False

        if stage_idx < len(schedule) - 1 and morph_cache.get(stage_idx):
            next_stage_start = schedule[stage_idx + 1][0]
            time_to_next = next_stage_start - t
            if time_to_next < morph_dur:
                # In morph transition
                morph_progress = 1.0 - (time_to_next / morph_dur)
                morph_frames = morph_cache[stage_idx]
                morph_idx = min(morph_n - 1, int(morph_progress * morph_n))
                current_pose = morph_frames[morph_idx]
                in_morph = True

        # Compute landing transform (only at start of each stage, not during morph)
        if in_morph:
            landing = {"scale": 0.0, "x": 0.0, "y": 0.0, "rotation": 0.0}
        else:
            landing = compute_landing(local_time, direction=direction)

        scale = 1.0 + landing["scale"]
        x = char_x + landing["x"]
        bottom = char_bottom + landing["y"]
        rotation = landing["rotation"]

        # Render frame
        frame = render_frame(background, current_pose, x, bottom, scale, rotation, centered=True)

        frame.save(out_dir / f"frame-{frame_idx:04d}.png", compress_level=2)


def _run_cmd(cmd: list[str], label: str, out_path: str) -> None:
    """Run FFmpeg command with fallback."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600)
        if r.returncode != 0:
            print(f"  [StopMotion] FFmpeg error ({label}): {r.stderr[-200:]}")
    except subprocess.TimeoutExpired:
        print(f"  [StopMotion] FFmpeg timeout on {label}")
    except Exception as e:
        print(f"  [StopMotion] Error on {label}: {e}")

    if not os.path.exists(out_path) or os.path.getsize(out_path) < 1000:
        # Fallback: silent static segment
        print(f"  [StopMotion] Fallback for {label}")
