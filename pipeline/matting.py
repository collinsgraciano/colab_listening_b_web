"""MODNet AI 人像抠图引擎（ONNX，本地推理）。

新增抠图选项：与 stop_motion._remove_white_bg_fallback（白度阈值抠图）并存，
通过 MATTING_ENGINE 环境变量选择：
- auto（默认）：有权重用 MODNet，无权重自动回退白度抠图
- modnet：强制 MODNet（权重缺失/推理失败时回退白度并告警，保证管线不中断）
- white_threshold：原白度抠图

权重：H:\\models\\modnet\\modnet.onnx（Apache-2.0，~25MB），
可用 MODNET_MODEL_PATH 环境变量覆盖路径。

输入任意背景的角色图均可抠（白底 prompt 不再是硬约束），
预处理长边缩放至 512（MODNet 官方推理口径），CPU 单图亚秒级。
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

DEFAULT_MODEL_PATH = r"H:\models\modnet\modnet.onnx"

_session = None
_session_lock = threading.Lock()
_load_attempted = False


def _model_path() -> Path:
    return Path(os.environ.get("MODNET_MODEL_PATH", "").strip() or DEFAULT_MODEL_PATH)


def matting_available() -> bool:
    """MODNet 是否可用（权重文件存在且 onnxruntime 可导入）。"""
    if not _model_path().exists():
        return False
    try:
        import onnxruntime  # noqa: F401
        return True
    except ImportError:
        return False


def _get_session():
    """懒加载单例 onnxruntime session（CPU EP）。"""
    global _session, _load_attempted
    with _session_lock:
        if _session is not None or _load_attempted:
            return _session
        _load_attempted = True
        path = _model_path()
        if not path.exists():
            return None
        try:
            import onnxruntime as ort
            _session = ort.InferenceSession(
                str(path), providers=["CPUExecutionProvider"])
        except Exception as e:  # 权重损坏/版本不兼容 → 走白度回退
            print(f"  [Matting] MODNet session 初始化失败，回退白度抠图: {e}")
            _session = None
        return _session


def resolve_engine() -> str:
    """读取抠图引擎配置：auto / modnet / white_threshold。"""
    return os.environ.get("MATTING_ENGINE", "auto").strip().lower() or "auto"


def matting_alpha(img: Image.Image, ref_size: int = 512) -> np.ndarray:
    """计算 alpha matte（[0,1] float32，与原图同尺寸）。"""
    session = _get_session()
    if session is None:
        raise RuntimeError("MODNet 不可用（权重缺失或初始化失败）")

    w, h = img.size
    im = np.asarray(img.convert("RGB"), dtype=np.float32)
    im = (im - 127.5) / 127.5
    # 长边缩放到 ref_size，宽高对齐 32 的倍数（MODNet 输入口径）
    if w >= h:
        rw, rh = ref_size, max(32, int(h / w * ref_size))
    else:
        rh, rw = ref_size, max(32, int(w / h * ref_size))
    rw -= rw % 32
    rh -= rh % 32
    resized = np.asarray(
        Image.fromarray(((im * 127.5 + 127.5).clip(0, 255)).astype(np.uint8))
        .resize((rw, rh), Image.BILINEAR), dtype=np.float32)
    resized = (resized - 127.5) / 127.5
    x = resized[None].transpose(0, 3, 1, 2).astype(np.float32)  # 1,3,H,W

    y = session.run(None, {"input": x})[0][0, 0]  # H,W
    matte = Image.fromarray((np.clip(y, 0, 1) * 255).astype(np.uint8))
    return np.asarray(matte.resize((w, h), Image.BILINEAR), dtype=np.float32) / 255.0


def matting_remove_bg(img: Image.Image, feather: float = 0.0) -> Image.Image:
    """任意背景图片 → RGBA（alpha 来自 MODNet）。"""
    alpha = matting_alpha(img)
    rgba = img.convert("RGBA")
    if feather > 0:
        # 轻微羽化边缘（可选，0=关闭）
        alpha_img = Image.fromarray((alpha * 255).astype(np.uint8))
        alpha_img = alpha_img.filter(ImageFilter.GaussianBlur(radius=feather))
        alpha = np.asarray(alpha_img, dtype=np.float32) / 255.0
    rgba.putalpha(Image.fromarray((alpha * 255).astype(np.uint8)))
    return rgba
