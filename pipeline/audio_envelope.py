"""逐帧音频 RMS 包络提取 — quest_v2 唇形同步与姿势调度共用。

设计（替代 quest 模块的 astats stderr 解析 hack）：
  ffmpeg 解码为 16kHz 单声道 f32le PCM → numpy 按渲染帧率分箱计算 RMS
  → 段内最大值归一化(0..1) → 滞后阈值 + 最短持有/补隙去抖
  → 连续 0..1 张嘴混合权重（供 Image.blend 双态口型使用）。
"""
import math
import subprocess

import numpy as np

SR = 16000  # 解码采样率（对口型足够，解码快）


def extract_rms_envelope(audio_path: str, fps: float,
                         sr: int = SR) -> list[float]:
    """逐帧 RMS 包络（0..1，按段内最大值归一化）。失败返回空列表。

    Args:
        audio_path: 音频文件路径。
        fps: 渲染帧率（包络每帧一个值，与停格渲染帧一一对应）。
        sr: 解码采样率。
    """
    if fps <= 0 or not audio_path:
        return []
    try:
        r = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", audio_path,
             "-f", "f32le", "-ac", "1", "-ar", str(sr), "-"],
            capture_output=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return []
    if r.returncode != 0 or not r.stdout:
        return []
    samples = np.frombuffer(r.stdout, dtype=np.float32)
    if samples.size == 0:
        return []

    total_frames = max(1, math.ceil(samples.size / sr * fps))
    spf = sr / fps  # 每帧采样数
    env = np.zeros(total_frames, dtype=np.float64)
    for i in range(total_frames):
        s = int(i * spf)
        e = min(samples.size, max(s + 1, int((i + 1) * spf)))
        seg = samples[s:e].astype(np.float64)
        env[i] = float(np.sqrt(np.mean(seg * seg))) if seg.size else 0.0
    mx = float(env.max())
    if mx <= 1e-6:
        return [0.0] * total_frames
    return [float(v) / mx for v in env]


def mouth_blend_weights(envelope: list[float],
                        open_th: float = 0.30, close_th: float = 0.18,
                        fps: float = 12, min_hold: float = 0.08) -> list[float]:
    """张嘴混合权重（0..1 连续值，与 envelope 等长）。

    流程：滞后阈值二值化（防高频抖动）→ 补短隙/删短开（自然音节节律）
    → 3 帧滑动平均软化切换 → 开口期按包络幅度调制（嘴动有大小变化）。
    """
    n = len(envelope)
    if n == 0:
        return []

    # 1) 滞后阈值二值化
    flags = [False] * n
    state = False
    for i, v in enumerate(envelope):
        if not state and v >= open_th:
            state = True
        elif state and v < close_th:
            state = False
        flags[i] = state

    # 2) 去抖：补短隙 + 删短开
    min_frames = max(1, round(min_hold * fps))
    i = 0
    while i < n:
        if not flags[i]:
            j = i
            while j < n and not flags[j]:
                j += 1
            # 前后都张嘴的短隙（< min_frames）补为张嘴（语速快的连读）
            if i > 0 and j < n and (j - i) < min_frames:
                for k in range(i, j):
                    flags[k] = True
            i = j
        else:
            j = i
            while j < n and flags[j]:
                j += 1
            # 过短张嘴段（噪声毛刺）删除
            if (j - i) < min_frames:
                for k in range(i, j):
                    flags[k] = False
            i = j

    # 3) 3 帧滑动平均软化（12fps 下 ≈0.25s 平滑过渡）
    win = 3
    soft = []
    for i in range(n):
        lo = max(0, i - win + 1)
        seg = flags[lo:i + 1]
        soft.append(sum(seg) / len(seg))

    # 4) 开口期按包络幅度调制：响=大张，轻=微张（基底 0.65 保证开口可见）
    weights = []
    for i in range(n):
        f = soft[i]
        if f <= 0.0:
            weights.append(0.0)
        else:
            mag = min(1.0, envelope[i] / max(open_th, 1e-6))
            weights.append(min(1.0, (0.65 + 0.35 * mag) * f))
    return weights
