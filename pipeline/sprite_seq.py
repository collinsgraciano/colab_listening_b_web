"""游戏角色式序列帧素材生成（sprite_sequence 动画模式）。

素材生产路线（唯一）：Seedance2 参考图生成白底动作视频 → ffmpeg 抽帧 →
抠图 → 取样 16 帧（帧间一致性最好）。

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
from stop_motion import remove_bg, POSE_CANVAS_W, POSE_CANVAS_H, POSE_TARGET_H
from style_manager import DEFAULT_STYLE_PROMPT
from image_gen import reupload_for_cdn

SPRITE_ACTIONS = ("talking_01", "idle_01", "wave",
                  "talking_02", "talking_03", "idle_02")
FIRST_ACTION = "talking_01"
FRAMES_PER_CLIP = 16
TALKING_FRAMES = 48     # talking take：6s 源视频 @8fps 全帧（整句单 take 铺放）
CLIP_FPS = 12           # 循环型动作播放帧率（与成片 25fps 解耦，游戏式采样）
MANIFEST_NAME = "sprite_clips.json"
SOURCE = "video_frames"


def frames_for_action(action: str) -> int:
    """talking 变体整句单 take 需要全帧（6s@8fps=48）；循环型动作 16 帧足够。"""
    return TALKING_FRAMES if action.startswith("talking") else FRAMES_PER_CLIP

# 动作提示词：强调"同一人物连续微动作"（flip book 式）。
# talking ×3 / idle ×2 为变体：新模式（take_mode）说话者整句播一个 take、
# 倾听者按行轮换 idle，跨行不重样；变体差异用 prompt 手势/体态侧别区分
# （Seedance2 无 seed 暴露）。wave 仅供主持人 outro 送别。
_ACTION_PHRASES = {
    "talking_01": ("talking and conversing, mouth moving with expressive friendly "
                   "expressions, natural small hand gestures while speaking"),
    "talking_02": ("explaining with both hands gesturing in front of the chest, "
                   "slightly more animated friendly expressions, mouth moving"),
    "talking_03": ("leaning slightly forward, one hand raised in a soft presenting "
                   "gesture, warm engaged expression, mouth moving"),
    "idle_01": ("standing relaxed, calm breathing idle loop, subtle body sway, "
                "gentle listening expression"),
    "idle_02": ("standing relaxed with a warm smile, gently shifting weight from "
                "one side to the other, subtle head tilts while listening"),
    "wave": ("waving hello in a friendly greeting, right hand raised waving"),
}


def _video_prompt(action: str, char_desc: str, style_prompt: str) -> str:
    """char_desc 传空串时省略外观描述段（有参考图场景以图为准，文字描述反干扰一致性）；
    管线内各调用方恒传非空描述，输出逐字节不变。"""
    desc = f"{char_desc}, " if char_desc else ""
    return (
        f"{desc}{_ACTION_PHRASES[action]}, plain pure white background, "
        f"static camera, medium waist-up shot, character stays centered and fully "
        f"in frame at all times, consistent appearance throughout the whole video, "
        f"smooth continuous looping motion, {style_prompt}, "
        f"no text, no watermark, no other people"
    )


def _gen_action_video(prompt: str, video_path: str, ref_url: str = "",
                      stop_check=None) -> str:
    """生成白底动作视频（Seedance2），返回本地路径（空串=失败）。"""
    try:
        params = {"prompt": prompt, "duration": 6, "ratio": "16:9",
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

def clip_frame_paths(img_dir, char_key: str, action: str,
                     count: int | None = None) -> list:
    """该角色该动作的目标帧路径（count 缺省按动作类型取 48/16）。"""
    n = count if count is not None else frames_for_action(action)
    return [str(Path(img_dir) / f"clip_{char_key}_{action}_{j:02d}.png")
            for j in range(n)]


def _clip_complete(img_dir, char_key: str, action: str,
                   count: int | None = None) -> bool:
    return all(os.path.exists(p)
               for p in clip_frame_paths(img_dir, char_key, action, count))


def _produce_clip(char_key: str, action: str, char_desc: str,
                  img_dir: Path, style_prompt: str,
                  ref_frame: str | None, stop_check=None) -> list:
    """产出单个动作 clip，返回帧路径列表（空列表=失败）。

    talking 变体存全帧（48）供整句单 take 铺放；循环型动作取样 16 帧。
    返回前完成：抽帧 → _unify_clip_frames 统一几何 → 覆盖保存目标帧。
    """
    n_frames = frames_for_action(action)
    work_dir = Path(tempfile.gettempdir()) / f"sprite_work_{char_key}_{action}"
    work_dir.mkdir(parents=True, exist_ok=True)
    final_paths = clip_frame_paths(img_dir, char_key, action, n_frames)
    if all(os.path.exists(p) for p in final_paths):
        return final_paths

    prompt = _video_prompt(action, char_desc, style_prompt)
    video = str(work_dir / "action.mp4")
    ref_url = ""
    if ref_frame and os.path.exists(ref_frame):
        ref_url = reupload_for_cdn(ref_frame, Path(ref_frame).name)
    got = _gen_action_video(prompt, video, ref_url=ref_url, stop_check=stop_check)
    if not got:
        return []
    all_frames = _extract_video_frames(video, work_dir / "frames", fps=8)
    if len(all_frames) <= n_frames:
        raw_paths = [str(p) for p in all_frames]
    else:
        raw_paths = [str(p) for p in _sample_frames(all_frames, n_frames)]

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

    # 帧数对齐 n_frames：不足时循环补齐，超出时均匀取样
    if len(unified) != n_frames:
        idxs = [round(i * (len(unified) - 1) / (n_frames - 1))
                for i in range(n_frames)]
        unified = [unified[k] for k in idxs]

    for j, frame in enumerate(unified):
        frame.save(final_paths[j], compress_level=2)
    print(f"    [SpriteSeq] {char_key}/{action}: {n_frames} frames saved")
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


def _load_prior_manifest(img_dir) -> tuple[dict, set]:
    """读运行目录已有 manifest（存在时），返回 (chars dict, from_library 集合)。

    from_library 由 app/pipeline_service._merge_run_clip_manifest 写入：
    这些角色的序列帧素材以素材库为权威来源。
    """
    path = Path(img_dir) / MANIFEST_NAME
    if not path.exists():
        return {}, set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            m = json.load(f)
        return (m.get("chars") or {}), set(m.get("from_library") or [])
    except (json.JSONDecodeError, OSError):
        return {}, set()


def generate_sprite_clips(script, img_dir,
                          tts_thread=None, max_workers: int = 2,
                          style_prompt: str = DEFAULT_STYLE_PROMPT,
                          char_keys=None, stop_check=None) -> dict | None:
    """生成全部角色的动作变体序列帧素材，返回 manifest dict（完全失败返回 None）。

    resume：目标帧文件齐该动作应有帧数（talking 48 / 其余 16）直接登记跳过；
    单动作失败记日志继续。
    from_library 角色（绑定素材库序列帧角色）：缺失动作不自动补齐（用户决策
    「缺什么用什么」，零积分），已有动作沿用原 manifest 帧清单——帧数随上传
    视频时长可变，固定 48/16 判定会截断长视频素材。
    """
    img_dir = Path(img_dir)
    prior_chars, library_chars = _load_prior_manifest(img_dir)
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

    manifest = {"version": 1, "fps": CLIP_FPS, "source": SOURCE, "chars": {}}
    if library_chars:
        manifest["from_library"] = sorted(library_chars)
    todo: dict[str, list] = {}
    for char_key, _desc in chars:
        prior = prior_chars.get(char_key) or {}
        entry = {}
        missing = []
        for action in SPRITE_ACTIONS:
            prior_frames = [p for p in (prior.get(action) or [])
                            if os.path.exists(p)]
            if len(prior_frames) >= 4:
                entry[action] = prior_frames
            elif _clip_complete(img_dir, char_key, action):
                entry[action] = clip_frame_paths(img_dir, char_key, action)
            else:
                missing.append(action)
        manifest["chars"][char_key] = entry
        if missing and char_key in library_chars:
            print(f"  [SpriteSeq] {char_key}: 素材库角色，缺失动作不自动补齐 "
                  f"({', '.join(missing)})")
        elif missing:
            todo[char_key] = missing

    n_done = sum(len(v) for v in manifest["chars"].values())
    n_total = len(chars) * len(SPRITE_ACTIONS)
    if not todo:
        print(f"  [SpriteSeq] All {n_total} clips already exist, skipping")
        _write_manifest(img_dir, manifest)
        return manifest
    print(f"  [SpriteSeq] Generating {n_total - n_done}/{n_total} clips "
          f"(actions={list(SPRITE_ACTIONS)})...")

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
                                       style_prompt, ref,
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
