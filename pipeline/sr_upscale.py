"""Real-ESRGAN animevideov3 AI 视频超分引擎（torch CUDA fp16）。

4K 步骤的新引擎选项（upscale_engine=ai）：与原 ffmpeg lanczos 插值路径并存，
默认仍为 ffmpeg（行为零变化，AI 为显式 opt-in）。

- 模型：realesr-animevideov3（BSD-3-Clause，2.4MB，专为卡通/动画视频设计）
  权重 H:\\models\\upscaling\\realesr-animevideov3.pth，
  可用 UPSCALE_MODEL_PATH 环境变量覆盖
- 推理：SRVGGNetCompact（basicsr 原版架构独立复制，免 basicsr 依赖），
  torch CUDA + fp16 —— 实测 RTX 3060 Laptop 约 108ms/帧（720p→4x），
  全片约 30-60 分钟；显存峰值 <1GB
- 流式处理：ffmpeg rawvideo 双管道（解码 pipe → 逐帧超分 → 编码 pipe），
  中间帧不落盘（5120×2880 PNG ×1.8万帧 ≈ 数百 GB，必须流式）
- 输出：4x 超分后 bicubic(antialias) 收敛到 3840×2160，音频流原样复制

依赖：torch（CUDA 构建）；无 CUDA 时 upscale_video_ai 抛错由调用方回退。
"""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

import numpy as np

DEFAULT_MODEL_PATH = r"H:\models\upscaling\realesr-animevideov3.pth"

_model = None
_model_lock = None


def model_available() -> bool:
    """AI 超分是否可用（权重存在且 torch+CUDA 可用）。"""
    path = Path(_model_path())
    if not path.exists():
        return False
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _model_path() -> str:
    import os
    return os.environ.get("UPSCALE_MODEL_PATH", "").strip() or DEFAULT_MODEL_PATH


class SRVGGNetCompact:
    """basicsr srvgg_arch.py 原版架构（BSD-3）：Conv/PReLU 交错 + PixelShuffle + nearest 残差。"""

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4):
        import torch.nn as nn
        self.upscale = upscale
        body = nn.ModuleList()
        body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        body.append(nn.PReLU(num_parameters=num_feat))
        for _ in range(num_conv):
            body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            body.append(nn.PReLU(num_parameters=num_feat))
        body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        self.body = body
        self.upsampler = nn.PixelShuffle(upscale)
        self._nn = nn

    def load_weights(self, ckpt_path: str) -> None:
        import torch
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        sd = ckpt.get("params_ema") or ckpt.get("params") or ckpt
        # checkpoint key 带 "body." 前缀（body.0.weight...），本类直接持有 ModuleList
        sd2 = {k[5:] if k.startswith("body.") else k: v for k, v in sd.items()}
        result = self.body.load_state_dict(sd2, strict=False)
        if result.missing_keys:
            raise RuntimeError(f"超分权重不匹配: missing={list(result.missing_keys)[:5]}")

    def to(self, device, dtype=None):
        self.body = self.body.to(device) if dtype is None else self.body.to(device, dtype)
        return self

    def eval(self):
        self.body.eval()
        return self

    def __call__(self, x):
        import torch.nn.functional as F
        out = x
        for layer in self.body:
            out = layer(out)
        out = self.upsampler(out)
        base = F.interpolate(x, scale_factor=self.upscale, mode="nearest")
        return out + base


def _get_model():
    """懒加载单例模型（CUDA fp16）。"""
    global _model, _model_lock
    import threading
    import torch
    if _model_lock is None:
        _model_lock = threading.Lock()
    with _model_lock:
        if _model is not None:
            return _model
        if not torch.cuda.is_available():
            raise RuntimeError("AI 超分需要 CUDA（当前 torch 无可用 GPU）")
        m = SRVGGNetCompact(3, 3, num_feat=64, num_conv=16, upscale=4)
        m.load_weights(_model_path())
        m = m.to("cuda", torch.float16).eval()
        for p in m.body.parameters():
            p.requires_grad_(False)
        _model = m
        return _model


def _probe_video(path: str) -> tuple[int, int, float]:
    """返回 (width, height, fps)。"""
    out = subprocess.check_output(
        ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,avg_frame_rate", "-of", "csv=p=0",
         path], text=True, encoding="utf-8", errors="replace").strip()
    parts = out.replace("x", ",").split(",")
    w, h = int(parts[0]), int(parts[1])
    num, den = parts[2].split("/")
    fps = float(num) / float(den) if float(den) else 25.0
    return w, h, round(fps, 3)


def upscale_video_ai(src: str, dst: str, out_w: int = 3840, out_h: int = 2160,
                     timeout: int = 3600, log=print) -> None:
    """AI 4x 超分整段视频（流式双管道），输出 out_w×out_h + 原音频。

    超时/失败时清理半成品 dst 并抛 RuntimeError，由调用方决定回退。
    """
    import torch

    model = _get_model()
    w, h, fps = _probe_video(src)
    sw, sh = w * 4, h * 4  # 模型 4x 输出尺寸
    frame_bytes = w * h * 3
    dst_path = Path(dst)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    # 管道 A：解码为 rawvideo rgb24
    dec = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", src,
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    # 管道 B：rawvideo（超分后 bicubic 收敛到目标分辨率）→ h264 + 原音频
    enc = subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{out_w}x{out_h}",
         "-r", f"{fps}", "-i", "-",
         "-i", src,
         "-map", "0:v", "-map", "1:a?",
         "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-threads", "0",
         "-c:a", "copy", str(dst_path)],
        stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    t0 = time.time()
    frames = 0
    try:
        while True:
            if time.time() - t0 > max(60, timeout):
                raise TimeoutError(f"AI 超时超时（>{timeout}s），已完成 {frames} 帧")
            buf = dec.stdout.read(frame_bytes)
            if not buf or len(buf) < frame_bytes:
                break
            frame = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)
            with torch.no_grad():
                x = torch.from_numpy(frame.astype(np.float32) / 255).permute(2, 0, 1)[None].cuda().half()
                y = model(x)[0].float()
                # 4x 超分结果收敛到目标分辨率（bicubic+antialias，等效高质量降采样）
                if y.shape[-2] != out_h or y.shape[-1] != out_w:
                    y = torch.nn.functional.interpolate(
                        y[None], size=(out_h, out_w), mode="bicubic", antialias=True)[0]
                out = (y.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
            enc.stdin.write(out.tobytes())
            frames += 1
            if frames % 500 == 0:
                elapsed = time.time() - t0
                log(f"    [SR] {frames} 帧 | {elapsed/max(frames,1)*1000:.0f} ms/帧 | 已用 {elapsed/60:.1f} min")
        enc.stdin.close()
        dec.stdout.close()
        enc.wait(timeout=max(60, timeout))
        dec.wait(timeout=30)
        if enc.returncode != 0 or not dst_path.exists() or dst_path.stat().st_size < 1000:
            raise RuntimeError(f"AI 超分编码失败（encoder returncode={enc.returncode}）")
        log(f"    [SR] 完成：{frames} 帧，{(time.time()-t0)/60:.1f} min")
    except Exception:
        enc.kill()
        dec.kill()
        dst_path.unlink(missing_ok=True)
        raise
