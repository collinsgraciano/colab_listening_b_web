"""游戏角色式序列帧素材生成（sprite_sequence 动画模式）。

素材生产路线（唯一）：Seedance2 参考图生成白底动作视频 → ffmpeg 全程抽帧
(24fps) → 抠图（帧间一致性最好）。

产出规格：帧数 = 源视频实际时长 × 24fps（全动作一致，不固定帧数），
文件 clip_{char}_{action}_{j:02d}.png —— 已统一 remove_bg + 整组 union bbox
对齐 + 共同比例缩放居中到 POSE 画布（渲染层直接加载，无需再处理）。
take 型动作（talking/wave）整句均匀铺放；循环型动作（idle）以 manifest
fps=24 原速循环。清单 images/sprite_clips.json 记录 {char: {action: [帧路径]}}
+ fps + source。
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

# 旧全集（6 段变体）仅留作参考：自 2026-09-11 起三 sprite 模式统一最小动作集
# （用户决策：普通角色 talking+idle 各 1 段、独立主持人 talking+wave 各 1 段，
# 缺素材不回退直接报错；需要更多变体在素材库手动单动作补生成）。
LEGACY_FULL_ACTIONS = ("talking_01", "idle_01", "wave",
                       "talking_02", "talking_03", "idle_02")
FIRST_ACTION = "talking_01"
CLIP_FPS = 24           # 循环型动作播放帧率 = 抽帧密度 → 循环以原速播放（用户决策 2026-09-05）
MANIFEST_NAME = "sprite_clips.json"
SOURCE = "video_frames"
EXTRACT_FPS = 24        # 源视频抽帧密度（用户决策 2026-09-05：8fps→24fps 提升流畅度）


def actions_for_char(char_key: str) -> tuple:
    """该角色需生成的最小动作集（用户决策 2026-09-11）。

    普通角色 talking+idle 各 1 段；独立主持人 talking+wave（几乎一直讲话
    不需要 idle，outro 送别用 wave）。被绑为主持人的普通角色不生成 wave
    ——outro 送别按用户决策改播 talking take。
    """
    if char_key == "host":
        return ("talking_01", "wave")
    return ("talking_01", "idle_01")

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
    "wave": ("waving hello in a friendly greeting while talking, right hand "
             "raised waving, mouth moving with a warm smile, "
             "as if warmly welcoming the audience"),
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

def _action_frame_paths(img_dir, char_key: str, action: str) -> list:
    """该角色该动作磁盘上实际存在的帧（clip_{char}_{action}_*.png，帧号数值序）。

    帧数由源视频实际时长决定（24fps 全程抽帧，不固定 144/16）；
    帧号两位/三位混排（j≥100 自然进位），必须按数值排序。
    """
    import re

    def _key(p: Path) -> int:
        m = re.search(r"_(\d+)\.png$", p.name)
        return int(m.group(1)) if m else 0

    return [str(p) for p in
            sorted(Path(img_dir).glob(f"clip_{char_key}_{action}_*.png"), key=_key)]


def _clip_complete(img_dir, char_key: str, action: str) -> bool:
    return len(_action_frame_paths(img_dir, char_key, action)) >= 4


def _produce_clip(char_key: str, action: str, char_desc: str,
                  img_dir: Path, style_prompt: str,
                  ref_frame: str | None, stop_check=None) -> list:
    """产出单个动作 clip，返回实际帧路径列表（空列表=失败）。

    全动作全程抽帧（EXTRACT_FPS，帧数=源视频实际时长×fps），不做固定帧数
    取样/补齐——take 靠均匀铺放、循环靠 fps=24 原速播放，任意帧数都正确。
    返回前完成：抽帧 → _unify_clip_frames 统一几何 → 清该动作旧帧（替换
    语义，防不同帧数残留混入）→ 保存。
    """
    work_dir = Path(tempfile.gettempdir()) / f"sprite_work_{char_key}_{action}"
    work_dir.mkdir(parents=True, exist_ok=True)

    prompt = _video_prompt(action, char_desc, style_prompt)
    video = str(work_dir / "action.mp4")
    ref_url = ""
    if ref_frame and os.path.exists(ref_frame):
        ref_url = reupload_for_cdn(ref_frame, Path(ref_frame).name)
    got = _gen_action_video(prompt, video, ref_url=ref_url, stop_check=stop_check)
    if not got:
        return []
    all_frames = _extract_video_frames(video, work_dir / "frames", fps=EXTRACT_FPS)
    raw_paths = [str(p) for p in all_frames]

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

    # 替换语义：清该动作旧帧再按实际帧数写盘（不同帧数残留会污染 manifest）
    for old in Path(img_dir).glob(f"clip_{char_key}_{action}_*.png"):
        try:
            old.unlink()
        except OSError:
            pass
    saved = []
    for j, frame in enumerate(unified):
        p = Path(img_dir) / f"clip_{char_key}_{action}_{j:02d}.png"
        frame.save(str(p), compress_level=2)
        saved.append(str(p))
    print(f"    [SpriteSeq] {char_key}/{action}: {len(saved)} frames saved")
    return saved


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


def _load_library_flag(img_dir) -> set:
    """读运行目录已有 manifest 的 from_library 集合（不存在/损坏返回空集）。

    from_library 由 app/pipeline_service._merge_run_clip_manifest 写入：
    这些角色的序列帧素材以素材库为权威来源。
    """
    path = Path(img_dir) / MANIFEST_NAME
    if not path.exists():
        return set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            return set(json.load(f).get("from_library") or [])
    except (json.JSONDecodeError, OSError):
        return set()


def _all_chars_from_script(script: dict) -> list:
    """序列帧角色清单：char_a/b/c 恒在（与旧行为一致），story 扩展角色
    （char_d 家人 / char_e 嘉宾）有描述才追加，host 恒在——实际生成集合
    由调用方 char_keys 过滤决定。"""
    chars = [
        ("char_a", script.get("char_a_description", "friendly young man")),
        ("char_b", script.get("char_b_description", "friendly young woman")),
        ("char_c", script.get("char_c_description", "friendly staff member")),
    ]
    # quest 脚本无 char_d/e 字段 → 列表与原行为逐字节一致
    for _sk in ("char_d", "char_e"):
        _sd = str(script.get(f"{_sk}_description", "")).strip()
        if _sd:
            chars.append((_sk, _sd))
    chars.append(("host", script.get("host_description",
                                     "friendly young woman with short brown hair, wearing a "
                                     "smart blue blazer, warm smile, professional TV host appearance")))
    return chars


def generate_sprite_clips(script, img_dir,
                          tts_thread=None, max_workers: int = 2,
                          style_prompt: str = DEFAULT_STYLE_PROMPT,
                          char_keys=None, stop_check=None) -> dict | None:
    """生成全部角色的动作变体序列帧素材，返回 manifest dict（完全失败返回 None）。

    resume：磁盘上该动作已有 ≥4 帧（帧数随源视频时长可变）直接登记跳过；
    单动作失败记日志继续。
    from_library 角色（绑定素材库序列帧角色）：缺失动作不自动补齐（用户决策
    「缺什么用什么」，零积分）；帧登记一律以磁盘 glob 为准——帧数随上传视频
    时长可变，固定帧数判定会截断长视频素材。
    """
    img_dir = Path(img_dir)
    library_chars = _load_library_flag(img_dir)
    all_chars = _all_chars_from_script(script)
    chars = [c for c in all_chars if char_keys is None or c[0] in char_keys]
    if not chars:
        return None

    manifest = {"version": 1, "fps": CLIP_FPS, "source": SOURCE, "chars": {}}
    if library_chars:
        manifest["from_library"] = sorted(library_chars)
    todo: dict[str, list] = {}
    for char_key, _desc in chars:
        char_actions = actions_for_char(char_key)
        entry = {}
        missing = []
        for action in char_actions:
            existing = _action_frame_paths(img_dir, char_key, action)
            if len(existing) >= 4:
                entry[action] = existing
            else:
                missing.append(action)
        manifest["chars"][char_key] = entry
        if missing and char_key in library_chars:
            print(f"  [SpriteSeq] {char_key}: 素材库角色，缺失动作不自动补齐 "
                  f"({', '.join(missing)})")
        elif missing:
            todo[char_key] = missing

    n_done = sum(len(v) for v in manifest["chars"].values())
    n_total = sum(len(actions_for_char(ck)) for ck, _ in chars)
    if not todo:
        print(f"  [SpriteSeq] All {n_total} clips already exist, skipping")
        _write_manifest(img_dir, manifest)
        return manifest
    print(f"  [SpriteSeq] Generating {n_total - n_done}/{n_total} clips "
          f"(minimal set: char=talking+idle, host=talking+wave)...")

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
                      f"缺失动作将在素材校验时导致运行停止")
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
          f"(missing actions will fail validation — run stops)")
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


def _has_clip_family(clip_map: dict, char_key: str, family: str) -> bool:
    """该角色是否已有 family 动作族的任一变体（talking / talking_NN 均算）。"""
    actions = clip_map.get(char_key) or {}
    return any(k == family or k.startswith(family + "_") for k in actions)


def ensure_sprite_families(img_dir, required: dict[str, tuple[str, ...]]) -> None:
    """序列帧素材硬校验（三 sprite 模式，用户决策 2026-09-11）。

    required = {char_key: 需要的动作族}，族内任一变体即满足（兼容素材库
    手动生成的 talking_02 等扩展变体）。校验与渲染同源（load_clip_map：
    manifest + 磁盘 ≥4 帧过滤），不通过直接抛 RuntimeError 停止运行
    ——不再回退姿势图集动画。
    """
    clip_map, _fps = load_clip_map(img_dir)
    missing: dict[str, list[str]] = {}
    for char_key, families in required.items():
        lack = [fam for fam in families
                if not _has_clip_family(clip_map, char_key, fam)]
        if lack:
            missing[char_key] = lack
    if not missing:
        return
    lines = "\n".join(f"  {ck}: 缺少 {', '.join(lack)}"
                      for ck, lack in missing.items())
    raise RuntimeError(
        "序列帧素材不完整 —— 按当前设置直接停止（不回退姿势图集动画）：\n"
        f"{lines}\n"
        "补齐方式：① 重新运行（续传会自动补生成缺失动作，已完成素材零积分跳过）；"
        "② 素材库绑定角色请先在「人物素材库」对应卡片补生成缺失动作后再运行。")
