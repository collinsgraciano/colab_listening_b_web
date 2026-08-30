"""数字人说话引擎：JoyVASA（音频→动作）+ LivePortrait（动作→渲染）本地 ONNX 推理。

original_cutout 模式的新动画类型（animation=digital_human）：
与 stop_motion（姿势图集定格动画）/ landing / none 并存，均为可选。

移植自 Timeline Studio（MIT）src/workers/{joyvasa,liveportrait}.worker.js，
模型经 ModelScope 镜像下载并 SHA256 校验，落盘 H:\\models\\digital_human\\：
  joyvasa/      joyvasa-audio.onnx(378MB) + denoiser(33MB) + conditioning/schedule/template
  liveportrait/ appearance(3.2MB) + motion_extractor(112MB) + stitching(0.2MB)
                + generator_preview_fp16(210MB,256px) / generator_quality_fp16(210MB,512px)

流程（每句对话）：
  TTS 音频(16k mono) → JoyVASA 50步扩散 → 100帧×73动作系数（≤4s）
  → LivePortrait 渲染关键帧（CPU 约 4.4s/帧@256px，故按 neural_fps 抽帧）
  → ffmpeg minterpolate 插帧到输出帧率 → MODNet 逐帧抠像 → RGBA 帧
  → 贴到场景背景 → 与音频封装为片段 mp4

已知限制（v1）：
  - JoyVASA 窗口固定 4 秒（64000 样本@16kHz）；音频超出部分静止保持末帧
  - CPU 推理较慢：256px 约 4.4s/帧、512px 约 5.5s/帧；18 行对话
    默认 neural_fps=3 时整段约 15-25 分钟（一次性成本，可接受）
  - 仅说话角色出镜（stop_motion 模式的听者姿势不参与）
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import threading
from pathlib import Path

import numpy as np
from PIL import Image

MODELS_ROOT = Path(os.environ.get("DH_MODELS_ROOT", r"H:\models\digital_human"))

# ---------------------------------------------------------------------------
# JoyVASA：音频 → 动作系数（移植 joyvasa.worker.js，固定种子保证可复现）
# ---------------------------------------------------------------------------


def _reflect_pad(x: np.ndarray, amount: int) -> np.ndarray:
    out = np.zeros(len(x) + amount * 2, dtype=np.float32)
    out[amount:amount + len(x)] = x
    for i in range(amount):
        out[amount - 1 - i] = x[i + 1]
        out[amount + len(x) + i] = x[len(x) - 2 - i]
    return out


def _make_gaussian(seed: int = 0x6A09E667):
    """xorshift32 + Box-Muller（与 JS 版一致，固定种子确定性输出）。"""
    state = seed & 0xFFFFFFFF
    spare: list[float | None] = [None]

    def uniform():
        nonlocal state
        state ^= (state << 13) & 0xFFFFFFFF
        state ^= state >> 17
        state ^= (state << 5) & 0xFFFFFFFF
        return (state + 1) / 4294967297

    def gauss():
        if spare[0] is not None:
            v = spare[0]
            spare[0] = None
            return v
        r = math.sqrt(-2 * math.log(uniform()))
        a = 2 * math.pi * uniform()
        spare[0] = r * math.sin(a)
        return r * math.cos(a)

    return gauss


def _rot_mat(pitch: float, yaw: float, roll: float) -> np.ndarray:
    x, y, z = map(math.radians, (pitch, yaw, roll))
    rx = np.array([[1, 0, 0], [0, math.cos(x), -math.sin(x)], [0, math.sin(x), math.cos(x)]], np.float32)
    ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]], np.float32)
    rz = np.array([[math.cos(z), -math.sin(z), 0], [math.sin(z), math.cos(z), 0], [0, 0, 1]], np.float32)
    return (rz @ ry @ rx).T.copy()  # JS rotationMatrix 返回转置布局


def _headpose_degree(logits: np.ndarray) -> float:
    e = np.exp(logits - logits.max())
    return float((e @ np.arange(len(logits))) / e.sum() * 3 - 97.5)


class _JoyVasa:
    """音频特征 + 扩散去噪 → [100,73] 动作系数（懒加载单例）。"""

    _inst = None
    _lock = threading.Lock()

    @classmethod
    def get(cls):
        with cls._lock:
            if cls._inst is not None:
                return cls._inst
            import onnxruntime as ort
            d = MODELS_ROOT / "joyvasa"
            inst = cls.__new__(cls)
            inst.audio = ort.InferenceSession(str(d / "joyvasa-audio.onnx"),
                                              providers=["CPUExecutionProvider"])
            inst.denoiser = ort.InferenceSession(str(d / "joyvasa-denoiser.onnx"),
                                                 providers=["CPUExecutionProvider"])
            inst.conditioning = np.frombuffer(
                (d / "joyvasa-conditioning.bin").read_bytes(), dtype=np.float32).copy()
            inst.schedule = np.frombuffer(
                (d / "joyvasa-schedule.bin").read_bytes(), dtype=np.float32).copy()
            cls._inst = inst
            return inst

    @staticmethod
    def prepare_window(samples: np.ndarray) -> np.ndarray:
        src = np.zeros(64000, dtype=np.float32)
        n = min(len(samples), 64000)
        src[:n] = samples[:n]
        return _reflect_pad(_reflect_pad(src, 20), 20)  # [64080]

    def generate_motion(self, samples: np.ndarray) -> np.ndarray:
        window = self.prepare_window(samples)
        audio_features = np.asarray(
            self.audio.run(None, {"audio_padded": window[None]})[0],
            dtype=np.float32).reshape(100, 256)

        cond = self.conditioning
        start_audio = cond[0:2560].reshape(10, 256)
        start_motion = cond[2560:3290].reshape(10, 73)
        null_audio = cond[3290:3546]

        audio_batch = np.zeros((2, 100, 256), dtype=np.float32)
        audio_batch[0] = np.tile(null_audio, (100, 1))
        audio_batch[1] = audio_features
        prev_motion = np.zeros((2, 10, 73), dtype=np.float32)
        prev_motion[0] = prev_motion[1] = start_motion
        prev_audio = np.zeros((2, 10, 256), dtype=np.float32)
        prev_audio[0] = prev_audio[1] = start_audio

        g = _make_gaussian()
        motion = np.array([g() for _ in range(100 * 73)], dtype=np.float32).reshape(100, 73)
        sched = self.schedule
        for step in range(50, 0, -1):
            pred = self.denoiser.run(None, {
                "motion": np.stack([motion, motion]),
                "audio": audio_batch,
                "previous_motion": prev_motion,
                "previous_audio": prev_audio,
                "step": np.array([step, step], dtype=np.int64),
            })[0]  # [2,110,73]
            uncond, condp = pred[0, 10:], pred[1, 10:]
            target = uncond + 1.15 * (condp - uncond)
            alpha, alpha_bar = sched[step * 2], sched[step * 2 + 1]
            alpha_bar_prev = sched[(step - 1) * 2 + 1]
            beta = 1 - alpha
            c0 = (1 - alpha_bar_prev) * math.sqrt(alpha) / (1 - alpha_bar)
            c1 = beta * math.sqrt(alpha_bar_prev) / (1 - alpha_bar)
            sigma = math.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar) * beta)
            noise = (np.array([g() for _ in range(100 * 73)], dtype=np.float32).reshape(100, 73)
                     if step > 1 else None)
            motion = (c0 * motion + c1 * target + (sigma * noise if noise is not None else 0)
                      ).astype(np.float32)
        return motion


# ---------------------------------------------------------------------------
# LivePortrait：角色图 + 动作系数 → 说话帧（移植 liveportrait.worker.js）
# ---------------------------------------------------------------------------


class _LivePortrait:
    """外观/动作提取 + stitching + 生成器（懒加载单例）。"""

    _inst: dict[str, "_LivePortrait"] = {}
    _lock = threading.Lock()

    @classmethod
    def get(cls, quality: str = "preview"):
        with cls._lock:
            if quality in cls._inst:
                return cls._inst[quality]
            import onnxruntime as ort
            d = MODELS_ROOT / "liveportrait"
            gen_name = ("generator_quality_fp16.onnx" if quality == "quality"
                        else "generator_preview_fp16.onnx")
            inst = cls.__new__(cls)
            inst.quality = quality
            inst.sessions = {
                "appearance": ort.InferenceSession(
                    str(d / "appearance_feature_extractor.onnx"), providers=["CPUExecutionProvider"]),
                "motion": ort.InferenceSession(
                    str(d / "motion_extractor.onnx"), providers=["CPUExecutionProvider"]),
                "stitching": ort.InferenceSession(
                    str(d / "stitching.onnx"), providers=["CPUExecutionProvider"]),
                "generator": ort.InferenceSession(
                    str(d / gen_name), providers=["CPUExecutionProvider"]),
            }
            inst._out_names = [o.name for o in inst.sessions["motion"].get_outputs()]
            inst._cache: tuple | None = None  # (key, feature, motion_flat)
            cls._inst[quality] = inst
            return inst

    @staticmethod
    def preprocess(portrait: Image.Image) -> np.ndarray:
        """中心方形裁剪（上偏 12.5%）→ 256×256 → alpha 预乘 CHW。"""
        w, h = portrait.size
        side = min(w, h)
        sx = max(0, (w - side) / 2)
        sy = max(0, (h - side) / 2 - side * 0.125)
        crop = portrait.crop((int(sx), int(sy), int(sx) + side, int(sy) + side))
        crop = crop.resize((256, 256), Image.BILINEAR)
        rgba = np.asarray(crop.convert("RGBA"), dtype=np.float32) / 255.0
        alpha = rgba[..., 3]
        chw = np.zeros((3, 256, 256), dtype=np.float32)
        for c in range(3):
            chw[c] = rgba[..., c] * alpha
        return chw[None]

    def _transform_keypoints(self, mo: dict) -> np.ndarray:
        R = _rot_mat(_headpose_degree(mo["pitch"]), _headpose_degree(mo["yaw"]),
                     _headpose_degree(mo["roll"]))
        out = np.zeros((21, 3), dtype=np.float32)
        for p in range(21):
            off = p * 3
            for a in range(3):
                out[p, a] = mo["scale"][0] * (
                    mo["kp"][off] * R[0, a] + mo["kp"][off + 1] * R[1, a]
                    + mo["kp"][off + 2] * R[2, a] + mo["exp"][off + a])
            out[p, 0] += mo["t"][0]
            out[p, 1] += mo["t"][1]
        return out

    def prepare_portrait(self, portrait: Image.Image, key: str = ""):
        """提取外观特征与源动作（同角色跨片段复用）。"""
        if self._cache and self._cache[0] == key:
            return self._cache[1], self._cache[2]
        img = self.preprocess(portrait)
        feature = self.sessions["appearance"].run(None, {"img": img})[0]
        raw = self.sessions["motion"].run(None, {"img": img})
        mo = {name: np.asarray(arr, dtype=np.float32).reshape(-1)
              for name, arr in zip(self._out_names, raw)}
        self._cache = (key, feature, mo)
        return feature, mo

    def decode_frame(self, coefficients: np.ndarray, frame: int, template: dict) -> dict:
        coefficients = coefficients.reshape(-1)
        off = frame * 73
        exp = coefficients[off:off + 63] * np.array(template["std_exp"], np.float32) \
            + np.array(template["mean_exp"], np.float32)

        def interp(key: str, value: float, index: int = 0) -> float:
            mx, mn = template[f"max_{key}"], template[f"min_{key}"]
            if isinstance(mx, list):
                mx, mn = mx[index], mn[index]
            return value * (mx - mn) + mn

        return {
            "exp": exp.reshape(21, 3),
            "scale": interp("scale", coefficients[off + 63]),
            "translation": np.array(
                [interp("t", coefficients[off + 64 + i], i) for i in range(3)], np.float32),
            "rotation": _rot_mat(interp("pitch", coefficients[off + 67]),
                                 interp("yaw", coefficients[off + 68]),
                                 interp("roll", coefficients[off + 69])),
        }

    def render_frame(self, feature, mo: dict, source_kp: np.ndarray,
                     driving_kp: np.ndarray) -> np.ndarray:
        stitch_in = np.concatenate([source_kp.reshape(-1), driving_kp.reshape(-1)]
                                   )[None].astype(np.float32)
        delta = self.sessions["stitching"].run(None, {"input": stitch_in})[0].reshape(-1)
        driving = driving_kp.reshape(-1).copy()
        driving += delta[:63]
        for p in range(21):
            driving[p * 3] += delta[63]
            driving[p * 3 + 1] += delta[64]
        out = self.sessions["generator"].run(None, {
            "feature_3d": feature,
            "kp_source": source_kp[None].astype(np.float32),
            "kp_driving": driving.reshape(21, 3)[None].astype(np.float32),
        })[0]
        return out  # [1,3,H,W]


def _build_driving_keypoints(mo: dict, driving: dict, initial_driving: dict) -> np.ndarray:
    """相对旋转/表情合成（唇点 6,12,14,17,19,20 直接取驱动表情）。"""
    source_rot = _rot_mat(_headpose_degree(mo["pitch"]), _headpose_degree(mo["yaw"]),
                          _headpose_degree(mo["roll"]))
    rel_rot = driving["rotation"] @ initial_driving["rotation"].T @ source_rot
    lip = {6, 12, 14, 17, 19, 20}
    out = np.zeros((21, 3), dtype=np.float32)
    for p in range(21):
        off = p * 3
        for a in range(3):
            rel_exp = mo["exp"][off + a] + driving["exp"][p, a] - initial_driving["exp"][p, a]
            expression = driving["exp"][p, a] if p in lip else rel_exp
            canonical = (mo["kp"][off] * rel_rot[0, a] + mo["kp"][off + 1] * rel_rot[1, a]
                         + mo["kp"][off + 2] * rel_rot[2, a])
            out[p, a] = mo["scale"][0] * (driving["scale"] / initial_driving["scale"]) \
                * (canonical + expression)
        out[p, 0] += mo["t"][0] + driving["translation"][0] - initial_driving["translation"][0]
        out[p, 1] += mo["t"][1] + driving["translation"][1] - initial_driving["translation"][1]
    return out


# ---------------------------------------------------------------------------
# 高层接口：音频解码 / 说话帧渲染 / 片段封装
# ---------------------------------------------------------------------------


def _load_audio_16k(audio_path: str) -> np.ndarray:
    """任意音频 → 16kHz mono float32（最长 4 秒进模型）。"""
    import tempfile
    tmp = Path(tempfile.gettempdir()) / f"dh_audio_{os.getpid()}_{threading.get_ident()}.f32"
    try:
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", audio_path,
                        "-ac", "1", "-ar", "16000", "-f", "f32le", str(tmp)],
                       check=True, capture_output=True)
        return np.frombuffer(tmp.read_bytes(), dtype=np.float32)
    finally:
        tmp.unlink(missing_ok=True)


def render_talking_frames(
    char_img_path: str,
    audio_path: str,
    frames_dir: str | Path,
    quality: str = "preview",
    neural_fps: int = 3,
    out_fps: int = 25,
    matting_engine: str = "auto",
    log=print,
) -> tuple[list[str], float]:
    """渲染一段"开口说话"的 RGBA 抠像帧序列。

    返回 (帧路径列表, 音频时长秒)。音频超过 4 秒部分保持末帧（JoyVASA 窗口限制）。
    """
    frames_dir = Path(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    # 1) 音频 → 动作
    samples = _load_audio_16k(audio_path)
    audio_dur = len(samples) / 16000.0
    jv = _JoyVasa.get()
    motion = jv.generate_motion(samples)
    log(f"    [DH] motion 100 帧 × 73 系数（音频 {audio_dur:.2f}s）")

    # 2) 角色图 → 关键帧
    lp = _LivePortrait.get(quality)
    portrait = Image.open(char_img_path)
    feature, mo = lp.prepare_portrait(portrait, key=char_img_path)
    source_kp = lp._transform_keypoints(mo)
    template = json.loads(
        (MODELS_ROOT / "joyvasa" / "joyvasa-motion-template.json").read_text(encoding="utf-8"))
    initial_driving = lp.decode_frame(motion, 0, template)

    motion_dur = min(audio_dur, 4.0)
    kf_count = max(2, min(100, int(motion_dur * neural_fps)))
    kf_paths: list[str] = []
    for i in range(kf_count):
        frame_idx = min(int(i * 100 / kf_count), 99)
        driving = lp.decode_frame(motion, frame_idx, template)
        raw_kp = _build_driving_keypoints(mo, driving, initial_driving)
        out = lp.render_frame(feature, mo, source_kp, raw_kp)
        img = (np.clip(out[0].transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)
        p = str(frames_dir / f"kf_{i:04d}.png")
        Image.fromarray(img).save(p)
        kf_paths.append(p)
    log(f"    [DH] 渲染 {kf_count} 关键帧（{'256' if quality == 'preview' else '512'}px）")

    # 3) minterpolate 插帧到 out_fps
    interp_dir = frames_dir / "interp"
    interp_dir.mkdir(exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error",
         "-framerate", f"{neural_fps}", "-i", str(frames_dir / "kf_%04d.png"),
         "-vf", (f"minterpolate=fps={out_fps}:mi_mode=mci:mc_mode=aobmc:vsbmc=1"
                 f"{',scale=512:512' if quality == 'preview' else ',scale=1024:1024'}"),
         str(interp_dir / "f_%05d.png")],
        check=True, capture_output=True)
    interp_paths = sorted(str(p) for p in interp_dir.glob("f_*.png"))

    # 4) 按音频时长截帧（音频不足帧序列时循环截断；超出 4s 部分重复末帧）
    target_frames = max(1, int(round(audio_dur * out_fps)))
    if len(interp_paths) >= target_frames:
        final_paths = interp_paths[:target_frames]
    else:
        final_paths = interp_paths + [interp_paths[-1]] * (target_frames - len(interp_paths))

    # 5) MODNet 逐帧抠像 → RGBA（无 MODNet 时保持 RGB，由 render_dh_segment 整帧兜底）
    import matting as _matting
    use_matting = (_matting.matting_available()
                   and _matting.resolve_engine() in ("auto", "modnet"))
    rgba_paths: list[str] = []
    rgba_dir = frames_dir / "rgba"
    rgba_dir.mkdir(exist_ok=True)
    for i, p in enumerate(final_paths):
        rp = str(rgba_dir / f"r_{i:05d}.png")
        if use_matting:
            _matting.matting_remove_bg(Image.open(p)).save(rp)
        else:
            Image.open(p).convert("RGB").save(rp)
        rgba_paths.append(rp)
    return rgba_paths, audio_dur


def render_dh_segment(
    portrait_path: str,
    bg_path: str,
    audio_path: str | None,
    out_path: str,
    duration: float,
    frames_dir: str | Path,
    quality: str = "preview",
    neural_fps: int = 3,
    out_fps: int = 25,
    canvas_w: int = 1920,
    canvas_h: int = 1080,
    fade_af: str = "",
    log=print,
) -> bool:
    """数字人对话片段：说话帧贴场景背景 + 音频 → segment mp4。

    Returns True on success.
    """
    try:
        if not os.path.exists(portrait_path):
            log(f"    [DH] portrait missing: {portrait_path}")
            return False
        if not audio_path or not os.path.exists(audio_path):
            log(f"    [DH] audio missing: {audio_path}")
            return False

        rgba_paths, audio_dur = render_talking_frames(
            portrait_path, audio_path, frames_dir,
            quality=quality, neural_fps=neural_fps, out_fps=out_fps, log=log)

        # 合成：角色贴背景（居中、底部留字幕区），背景归一化到画布
        comp_dir = Path(frames_dir) / "comp"
        comp_dir.mkdir(exist_ok=True)
        bg = Image.open(bg_path).convert("RGB").resize((canvas_w, canvas_h), Image.LANCZOS)
        # 无 MODNet 抠像时帧为不透明 → 整帧 cover 铺满画布（兜底，避免贴方块）
        full_frame = bool(rgba_paths) and Image.open(rgba_paths[0]).mode == "RGB"
        char_target_h = int(canvas_h * 0.62)
        for i, rp in enumerate(rgba_paths):
            cut = Image.open(rp)
            if full_frame:
                scale = max(canvas_w / cut.width, canvas_h / cut.height)
                cut = cut.resize((int(cut.width * scale), int(cut.height * scale)),
                                 Image.LANCZOS)
                px = (canvas_w - cut.width) // 2
                py = (canvas_h - cut.height) // 2
                frame = cut.convert("RGB")
            else:
                frame = bg.copy()
                scale = char_target_h / cut.height
                cut = cut.resize((int(cut.width * scale), char_target_h), Image.LANCZOS)
                px = (canvas_w - cut.width) // 2
                py = canvas_h - char_target_h - int(canvas_h * 0.06)  # 底部留字幕区
                frame.paste(cut, (px, py), cut)
            frame.save(str(comp_dir / f"c_{i:05d}.png"))

        # 封装：帧序列 + 音频（音画同长，天然同步）
        cmd = ["ffmpeg", "-y", "-v", "error",
               "-framerate", f"{out_fps}", "-i", str(comp_dir / "c_%05d.png"),
               "-i", audio_path,
               "-t", f"{duration:.3f}",
               "-map", "0:v:0", "-map", "1:a:0",
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "19",
               "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
               out_path]
        if fade_af:
            cmd += ["-af", f"{fade_af},apad=whole_dur={duration:.3f}"]
        else:
            cmd += ["-af", f"apad=whole_dur={duration:.3f}"]
        subprocess.run(cmd, check=True, capture_output=True)
        return os.path.exists(out_path) and os.path.getsize(out_path) > 1000
    except Exception as e:
        log(f"    [DH] segment render failed: {e}")
        return False


def models_available(quality: str = "preview") -> bool:
    """数字人模型是否齐备（audio/denoiser/生成器等）。"""
    jv = MODELS_ROOT / "joyvasa"
    lp = MODELS_ROOT / "liveportrait"
    gen = ("generator_quality_fp16.onnx" if quality == "quality"
           else "generator_preview_fp16.onnx")
    required = [
        jv / "joyvasa-audio.onnx", jv / "joyvasa-denoiser.onnx",
        jv / "joyvasa-conditioning.bin", jv / "joyvasa-schedule.bin",
        jv / "joyvasa-motion-template.json",
        lp / "appearance_feature_extractor.onnx", lp / "motion_extractor.onnx",
        lp / "stitching.onnx", lp / gen,
    ]
    return all(p.exists() for p in required)
