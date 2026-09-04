"""游戏角色式序列帧素材生成（sprite_sequence 动画模式）。

三条 AI 素材生产路线（--sprite-seq-source 配置选择）：
  A. atlas_grid   — 4×4 网格动作图集 → split_atlas_sequence 切 16 帧
                    （与现有姿势图集管线同构，成本最低）
  B. video_frames — Seedance2 参考图生成白底动作视频 → ffmpeg 抽帧 → 抠图 → 取样 16 帧
                    （帧间一致性最好，积分消耗较高）
  C. sprite_sheet — MCP generate_sprite_animation（4×4=16 帧 sprite sheet，支持参考图）

产出规格：每角色 4 个动作循环（talking/idle/gesture/wave）× 16 帧，
文件 clip_{char}_{action}_{j:02d}.png —— 已统一 remove_bg + 整组 union bbox
对齐 + 共同比例缩放居中到 POSE 画布（渲染层直接加载，无需再处理）。
清单 images/sprite_clips.json 记录 {char: {action: [帧路径]}} + fps + source。
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from mcp_client import call_tool, parse_task_id, poll_task, download_file
from atlas_split import split_atlas_sequence
from stop_motion import remove_bg, POSE_CANVAS_W, POSE_CANVAS_H, POSE_TARGET_H
from style_manager import DEFAULT_STYLE_PROMPT
import sensenova_image
from image_gen import reupload_for_cdn

SPRITE_ACTIONS = ("talking", "idle", "gesture", "wave")
FIRST_ACTION = "talking"
FRAMES_PER_CLIP = 16
CLIP_FPS = 12           # 播放帧率（与成片 25fps 解耦，游戏式采样）
MANIFEST_NAME = "sprite_clips.json"

# 动作提示词：强调"同一人物连续微动作"（flip book 式），供三条路线共用
_ACTION_PHRASES = {
    "talking": ("talking and conversing, mouth moving with expressive friendly "
                "expressions, natural small hand gestures while speaking"),
    "idle": ("standing relaxed, calm breathing idle loop, subtle body sway, "
             "gentle listening expression"),
    "gesture": ("making an emphatic explanatory gesture, one hand raised "
                "presenting, confident friendly expression"),
    "wave": ("waving hello in a friendly greeting, right hand raised waving"),
}

# 带参考图时的一致性前缀（编辑模式保持外观一致）
_REF_PREFIX = ("Keep the character's face, hairstyle, outfit, proportions and "
               "colors EXACTLY consistent with the reference image. ")


def _grid_prompt(action: str, char_desc: str, style_prompt: str) -> str:
    return (
        f"4x4 grid sprite sheet, 16 consecutive animation frames of ONE continuous "
        f"looping motion cycle, the exact same character performing the same action "
        f"in every frame, only small smooth movement differences between adjacent "
        f"frames like a flip book animation, "
        f"{char_desc}, {_ACTION_PHRASES[action]}, "
        f"medium waist-up shot, every frame fully inside its own cell with generous "
        f"empty margin on all sides, both shoulders fully visible, all arms and hands "
        f"completely within the cell borders, no body part cropped or cut off at the "
        f"cell edges, "
        f"all 16 frames same character same outfit same camera framing same character "
        f"size, plain white background, {style_prompt}, "
        f"no props, no objects, no scene, no text, no grid lines, no numbers"
    )


def _video_prompt(action: str, char_desc: str, style_prompt: str) -> str:
    return (
        f"{char_desc}, {_ACTION_PHRASES[action]}, plain pure white background, "
        f"static camera, medium waist-up shot, character stays centered and fully "
        f"in frame at all times, consistent appearance throughout the whole video, "
        f"smooth continuous looping motion, {style_prompt}, "
        f"no text, no watermark, no other people"
    )


# ---------------------------------------------------------------------------
# 路线 A：4×4 网格动作图集
# ---------------------------------------------------------------------------

def _gen_grid_atlas(prompt: str, atlas_path: str, ref_local: str = "",
                    stop_check=None) -> str:
    """生成 4×4 动作网格图，返回本地路径（空串=失败）。

    ref_local：参考图本地路径（一致性链）。sensenova 走 /v1/images/edits
    （本地路径内部转 Data-URL）；MCP 无参考时 is_segmentation=True 透明图集，
    带参考走编辑模式（输出可能为白底，统一在 _unify_clip_frames 抠图）。
    """
    try:
        if sensenova_image.get_image_provider() == "sensenova":
            size = sensenova_image.SIZE_MAP.get("atlas_seq", "4096x4096")
            if ref_local:
                url = sensenova_image.edit_image(ref_local, prompt, size=size,
                                                 prompt_extend=False)
            else:
                url = sensenova_image.text_to_image(prompt, size=size,
                                                    prompt_extend=False)
            return atlas_path if sensenova_image.download_image(url, atlas_path) else ""
        gen_params = {"prompt": prompt, "provider": "seedream",
                      "image_size": "4992x4992", "output_format": "png"}
        if ref_local:
            ref_url = reupload_for_cdn(ref_local, Path(ref_local).name)
            if ref_url:
                gen_params["image_urls"] = ref_url
            else:
                # 参考图上传失败 → 退回纯文本（is_segmentation 透明路径）
                gen_params["is_segmentation"] = True
        else:
            gen_params["is_segmentation"] = True
        result = call_tool("generate_image", gen_params)
        task_id = parse_task_id(result)
        if not task_id:
            if result and "result" in result:
                print(f"    [SpriteSeq] WARNING: no task_id, response: "
                      f"{json.dumps(result['result'].get('content', []), ensure_ascii=False)[:500]}")
            return ""
        data = poll_task(task_id, interval=10, max_wait=600, stop_check=stop_check)
        url = data.get("url", "")
        if not url:
            print(f"    [SpriteSeq] WARNING: no URL, status={data.get('status')}")
            return ""
        return atlas_path if download_file(url, atlas_path) else ""
    except RuntimeError:
        raise
    except Exception as e:
        print(f"    [SpriteSeq] grid atlas ERROR: {e}")
        return ""


# ---------------------------------------------------------------------------
# 路线 B：AI 视频抽帧
# ---------------------------------------------------------------------------

def _gen_action_video(prompt: str, video_path: str, ref_url: str = "",
                      stop_check=None) -> str:
    """生成白底动作视频（Seedance2），返回本地路径（空串=失败）。"""
    try:
        params = {"prompt": prompt, "duration": 5, "ratio": "16:9",
                  "resolution": "720p", "generate_audio": False}
        if ref_url:
            params.update({"mode": "reference_image", "image_urls": ref_url})
        else:
            params.update({"mode": "text_to_video"})
        result = call_tool("generate_video", params)
        task_id = parse_task_id(result)
        if not task_id:
            if result and "result" in result:
                print(f"    [SpriteSeq] WARNING: no task_id, response: "
                      f"{json.dumps(result['result'].get('content', []), ensure_ascii=False)[:500]}")
            return ""
        data = poll_task(task_id, interval=40, max_wait=900, stop_check=stop_check)
        if data.get("status") == "stopped":
            return ""
        url = data.get("url", "")
        if not url:
            print(f"    [SpriteSeq] WARNING: no video URL, status={data.get('status')}")
            return ""
        if not download_file(url, video_path):
            return ""
        # clip_gen 同款大小校验（防静默失败的小文件）
        if os.path.getsize(video_path) < 500000:
            print(f"    [SpriteSeq] WARNING: video too small "
                  f"({os.path.getsize(video_path)//1024}KB)")
            return ""
        return video_path
    except RuntimeError:
        raise
    except Exception as e:
        print(f"    [SpriteSeq] action video ERROR: {e}")
        return ""


def _extract_video_frames(video_path: str, out_dir: Path, fps: int = 8) -> list:
    """ffmpeg 抽帧，返回按帧号排序的 PNG 路径列表。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-i", video_path, "-vf", f"fps={fps}",
           str(out_dir / "f%03d.png")]
    try:
        subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=300)
    except Exception as e:
        print(f"    [SpriteSeq] ffmpeg extract ERROR: {e}")
        return []
    return sorted(out_dir.glob("f*.png"))


def _sample_frames(frame_paths: list, n: int = FRAMES_PER_CLIP) -> list:
    """均匀取样 n 帧（覆盖整个动作循环，避免只取开头）。"""
    if len(frame_paths) <= n:
        return list(frame_paths)
    idxs = []
    for i in range(n):
        k = round(i * (len(frame_paths) - 1) / (n - 1))
        if not idxs or k != idxs[-1]:
            idxs.append(k)
    return [frame_paths[k] for k in idxs]


# ---------------------------------------------------------------------------
# 路线 C：MCP generate_sprite_animation（4×4=16 帧 sprite sheet）
# ---------------------------------------------------------------------------

def _gen_sprite_sheet(prompt: str, sheet_path: str, ref_url: str = "",
                      stop_check=None) -> str:
    """生成 4×4 sprite sheet，返回本地路径（空串=失败）。"""
    try:
        params = {"prompt": prompt, "output_format": "png"}
        if ref_url:
            params["image_urls"] = ref_url
        result = call_tool("generate_sprite_animation", params)
        task_id = parse_task_id(result)
        if not task_id:
            if result and "result" in result:
                print(f"    [SpriteSeq] WARNING: no task_id, response: "
                      f"{json.dumps(result['result'].get('content', []), ensure_ascii=False)[:500]}")
            return ""
        data = poll_task(task_id, interval=10, max_wait=900, stop_check=stop_check)
        url = data.get("url", "")
        if not url:
            print(f"    [SpriteSeq] WARNING: no sheet URL, status={data.get('status')}")
            return ""
        return sheet_path if download_file(url, sheet_path) else ""
    except RuntimeError:
        raise
    except Exception as e:
        print(f"    [SpriteSeq] sprite sheet ERROR: {e}")
        return ""


# ---------------------------------------------------------------------------
# 帧统一处理：整组 union bbox 对齐 + 共同比例缩放 + 居中到 POSE 画布
# ---------------------------------------------------------------------------

def _alpha_bbox(img: Image.Image):
    """内容 bbox（alpha ≥ 8）；空内容返回 None。"""
    a = np.asarray(img.getchannel("A"))
    ys, xs = np.where(a >= 8)
    if len(xs) == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def _unify_clip_frames(raw_frames: list, label: str = "") -> list:
    """统一一组序列帧的几何基准。

    逐帧独立 normalize 会因每帧内容 bbox 不同导致播放时人物忽大忽小，
    必须整组共用同一 union bbox 裁剪 + 同一缩放比例 + 同一画布位置
    （union 内的相对运动 = 真实动作，完整保留）。
    """
    frames = []
    for f in raw_frames:
        try:
            frames.append(remove_bg(f.convert("RGBA")))
        except Exception as e:
            print(f"    [SpriteSeq] remove_bg error ({label}): {e}")
            frames.append(f.convert("RGBA"))
    boxes = [_alpha_bbox(f) for f in frames]
    valid = [b for b in boxes if b]
    if not valid:
        return frames
    x0 = max(0, min(b[0] for b in valid))
    y0 = max(0, min(b[1] for b in valid))
    x1 = max(b[2] for b in valid)
    y1 = max(b[3] for b in valid)
    uw, uh = x1 - x0, y1 - y0
    if uw <= 0 or uh <= 0:
        return frames
    # 目标高度 POSE_TARGET_H（与姿势图集视觉尺度一致），宽向留 8% 边
    scale = min(POSE_TARGET_H / uh, (POSE_CANVAS_W * 0.92) / uw)
    scale = min(scale, 4.0)
    out = []
    for f, box in zip(frames, boxes):
        if box:
            crop = f.crop((x0, y0, min(f.width, x1), min(f.height, y1)))
        else:
            crop = f
        if abs(scale - 1.0) > 1e-6:
            crop = crop.resize((max(1, round(crop.width * scale)),
                                max(1, round(crop.height * scale))),
                               Image.Resampling.LANCZOS)
        canvas = Image.new("RGBA", (POSE_CANVAS_W, POSE_CANVAS_H), (0, 0, 0, 0))
        canvas.alpha_composite(crop, (round((POSE_CANVAS_W - crop.width) / 2),
                                      round((POSE_CANVAS_H - crop.height) / 2)))
        out.append(canvas)
    print(f"    [SpriteSeq] unified {len(out)} frames ({label}) "
          f"union={uw}x{uh} scale={scale:.3f}")
    return out


# ---------------------------------------------------------------------------
# 单动作产出与编排
# ---------------------------------------------------------------------------

def clip_frame_paths(img_dir, char_key: str, action: str) -> list:
    """该角色该动作的 16 帧目标路径。"""
    return [str(Path(img_dir) / f"clip_{char_key}_{action}_{j:02d}.png")
            for j in range(FRAMES_PER_CLIP)]


def _clip_complete(img_dir, char_key: str, action: str) -> bool:
    return all(os.path.exists(p) for p in clip_frame_paths(img_dir, char_key, action))


def _split_grid_frames(atlas_path: str, work_dir: Path, cols: int = 4,
                       rows: int = 4) -> list:
    """切分网格图集为原始帧路径（split_atlas_sequence，统一 union 对齐）。"""
    out_paths = [str(work_dir / f"cell_{k:02d}.png") for k in range(cols * rows)]
    sizes = split_atlas_sequence(atlas_path, cols, rows, out_paths)
    return [p for p, s in zip(out_paths, sizes) if s[0] > 4 and s[1] > 4]


def _produce_clip(char_key: str, action: str, char_desc: str,
                  img_dir: Path, source: str, style_prompt: str,
                  ref_frame: str | None, stop_check=None) -> list:
    """产出单个动作 clip，返回 16 帧最终路径（空列表=失败）。

    返回前完成：切分/抽帧 → _unify_clip_frames 统一几何 → 覆盖保存目标帧。
    """
    work_dir = Path(tempfile.gettempdir()) / f"sprite_work_{char_key}_{action}"
    work_dir.mkdir(parents=True, exist_ok=True)
    final_paths = clip_frame_paths(img_dir, char_key, action)
    if all(os.path.exists(p) for p in final_paths):
        return final_paths

    if source == "atlas_grid":
        prompt = _grid_prompt(action, char_desc, style_prompt)
        atlas = str(work_dir / "atlas.png")
        got = _gen_grid_atlas(prompt, atlas, ref_local=ref_frame or "",
                              stop_check=stop_check)
        if not got and ref_frame:
            # 带参考失败 → 纯文本重试
            print(f"    [SpriteSeq] {char_key}/{action} ref-guided grid failed, retry text-only")
            got = _gen_grid_atlas(prompt, atlas, ref_local="", stop_check=stop_check)
        if not got:
            return []
        raw_paths = _split_grid_frames(atlas, work_dir)
    elif source == "video_frames":
        prompt = _video_prompt(action, char_desc, style_prompt)
        video = str(work_dir / "action.mp4")
        ref_url = ""
        if ref_frame and os.path.exists(ref_frame):
            ref_url = reupload_for_cdn(ref_frame, Path(ref_frame).name)
        got = _gen_action_video(prompt, video, ref_url=ref_url, stop_check=stop_check)
        if not got:
            return []
        all_frames = _extract_video_frames(video, work_dir / "frames", fps=8)
        raw_paths = [str(p) for p in _sample_frames(all_frames)]
    elif source == "sprite_sheet":
        prompt = _grid_prompt(action, char_desc, style_prompt)
        sheet = str(work_dir / "sheet.png")
        ref_url = ""
        if ref_frame and os.path.exists(ref_frame):
            ref_url = reupload_for_cdn(ref_frame, Path(ref_frame).name)
        got = _gen_sprite_sheet(prompt, sheet, ref_url=ref_url, stop_check=stop_check)
        if not got and ref_url:
            print(f"    [SpriteSeq] {char_key}/{action} ref-guided sheet failed, retry text-only")
            got = _gen_sprite_sheet(prompt, sheet, ref_url="", stop_check=stop_check)
        if not got:
            return []
        raw_paths = _split_grid_frames(sheet, work_dir)
    else:
        print(f"    [SpriteSeq] Unknown source '{source}'")
        return []

    if len(raw_paths) < 4:
        print(f"    [SpriteSeq] {char_key}/{action} too few frames ({len(raw_paths)})")
        return []

    raw_imgs = []
    for p in raw_paths:
        try:
            raw_imgs.append(Image.open(p))
        except Exception as e:
            print(f"    [SpriteSeq] open frame error {p}: {e}")
    unified = _unify_clip_frames(raw_imgs, label=f"{char_key}/{action}")
    if len(unified) < 4:
        return []

    # 帧数对齐 FRAMES_PER_CLIP：不足时循环补齐，超出时均匀取样
    if len(unified) != FRAMES_PER_CLIP:
        idxs = [round(i * (len(unified) - 1) / (FRAMES_PER_CLIP - 1))
                for i in range(FRAMES_PER_CLIP)]
        unified = [unified[k] for k in idxs]

    for j, frame in enumerate(unified):
        frame.save(final_paths[j], compress_level=2)
    print(f"    [SpriteSeq] {char_key}/{action}: {FRAMES_PER_CLIP} frames saved "
          f"(source={source})")
    return final_paths


def _pick_ref_frame(frame_paths: list) -> str | None:
    """从一组已统一帧中选内容最多的一帧作一致性参考图。"""
    best, best_area = None, -1
    for p in frame_paths:
        try:
            img = Image.open(p)
        except Exception:
            continue
        box = _alpha_bbox(img)
        if not box:
            continue
        area = (box[2] - box[0]) * (box[3] - box[1])
        if area > best_area:
            best, best_area = p, area
    return best


def generate_sprite_clips(script, img_dir, source: str = "atlas_grid",
                          tts_thread=None, max_workers: int = 2,
                          style_prompt: str = DEFAULT_STYLE_PROMPT,
                          char_keys=None, stop_check=None) -> dict | None:
    """生成全部角色的 4 动作序列帧素材，返回 manifest dict（完全失败返回 None）。

    resume：目标帧文件齐 16 张的动作直接登记跳过；单动作失败记日志继续。
    """
    img_dir = Path(img_dir)
    if source not in ("atlas_grid", "video_frames", "sprite_sheet"):
        print(f"  [SpriteSeq] Unknown source '{source}', fallback to atlas_grid")
        source = "atlas_grid"

    all_chars = [
        ("char_a", script.get("char_a_description", "friendly young man")),
        ("char_b", script.get("char_b_description", "friendly young woman")),
        ("char_c", script.get("char_c_description", "friendly staff member")),
        ("host", script.get("host_description",
                            "friendly young woman with short brown hair, wearing a "
                            "smart blue blazer, warm smile, professional TV host appearance")),
    ]
    chars = [c for c in all_chars if char_keys is None or c[0] in char_keys]
    if not chars:
        return None

    manifest = {"version": 1, "fps": CLIP_FPS, "source": source, "chars": {}}
    todo: dict[str, list] = {}
    for char_key, _desc in chars:
        entry = {}
        missing = []
        for action in SPRITE_ACTIONS:
            if _clip_complete(img_dir, char_key, action):
                entry[action] = clip_frame_paths(img_dir, char_key, action)
            else:
                missing.append(action)
        manifest["chars"][char_key] = entry
        if missing:
            todo[char_key] = missing

    n_done = sum(len(v) for v in manifest["chars"].values())
    n_total = len(chars) * len(SPRITE_ACTIONS)
    if not todo:
        print(f"  [SpriteSeq] All {n_total} clips already exist, skipping")
        _write_manifest(img_dir, manifest)
        return manifest
    print(f"  [SpriteSeq] Generating {n_total - n_done}/{n_total} clips "
          f"(source={source}, actions={list(SPRITE_ACTIONS)})...")

    def _gen_char(char_key: str, char_desc: str, actions: list) -> dict:
        produced: dict[str, list] = {}
        # 一致性锚点：优先姿势图集帧；否则用先生成的 talking 帧充当参考
        identity_ref = None
        pose_ref = img_dir / f"pose_{char_key}_0.png"
        if pose_ref.exists():
            identity_ref = str(pose_ref)
        talking_frames: list = []
        for action in actions:
            if stop_check and stop_check():
                break
            ref = identity_ref
            if ref is None and action != FIRST_ACTION and talking_frames:
                ref = _pick_ref_frame(talking_frames)
            try:
                frames = _produce_clip(char_key, action, char_desc, img_dir,
                                       source, style_prompt, ref,
                                       stop_check=stop_check)
            except RuntimeError as e:
                if "ALL_MCP_TOKENS_EXHAUSTED" in str(e):
                    if tts_thread:
                        tts_thread.join(timeout=5)
                    sys.exit(1)
                raise
            if not frames:
                print(f"    [SpriteSeq] WARNING: {char_key}/{action} failed, "
                      f"该动作将回退姿势图集")
                continue
            produced[action] = frames
            if action == FIRST_ACTION and not talking_frames:
                talking_frames = frames
        return produced

    from concurrent.futures import ThreadPoolExecutor, as_completed
    results: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
        futs = {pool.submit(_gen_char, ck, cd, todo[ck]): ck
                for ck, cd in chars if ck in todo}
        for fut in as_completed(futs):
            char_key = futs[fut]
            try:
                results[char_key] = fut.result()
            except Exception as e:
                print(f"  [SpriteSeq] ERROR {char_key}: {e}")

    for char_key, produced in results.items():
        manifest["chars"][char_key].update(produced)
    _write_manifest(img_dir, manifest)

    n_ok = sum(len(v) for v in manifest["chars"].values())
    print(f"  [SpriteSeq] Done — {n_ok}/{n_total} clips ready "
          f"(missing actions fall back to pose atlas)")
    return manifest if n_ok else None


def _write_manifest(img_dir: Path, manifest: dict) -> None:
    try:
        with open(img_dir / MANIFEST_NAME, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=1)
    except Exception as e:
        print(f"  [SpriteSeq] WARNING: manifest write failed: {e}")


def load_clip_map(img_dir) -> tuple:
    """读取 manifest，返回 ({char: {action: [帧路径]}}, fps)；文件缺失返回 ({}, CLIP_FPS)。"""
    path = Path(img_dir) / MANIFEST_NAME
    if not path.exists():
        return {}, CLIP_FPS
    try:
        with open(path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception as e:
        print(f"  [SpriteSeq] WARNING: manifest read failed: {e}")
        return {}, CLIP_FPS
    chars = manifest.get("chars") or {}
    clip_map = {}
    for char_key, actions in chars.items():
        valid = {}
        for action, frames in (actions or {}).items():
            paths = [p for p in (frames or []) if os.path.exists(p)]
            if len(paths) >= 4:
                valid[action] = paths
        if valid:
            clip_map[char_key] = valid
    fps = int(manifest.get("fps") or CLIP_FPS)
    return clip_map, fps

