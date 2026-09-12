"""sleep 模式 10 秒片头：本地 Pillow 动画渲染 + MCP AI 视频标准化。

供 Web 层（app/routers/intro_videos.py）调用，产物入库供 sleep 运行绑定：
- 本地路线 build_local_intro：Pillow 逐帧渲染（25fps × 10s）——渐变背景 +
  叶片漂动 + 白色圆角卡 + 频道名手写体淡入 + 副题淡入 + EN 徽标；
- AI 路线 finalize_ai_intro：调用方先用 PageMcpSession 生成/下载原始视频，
  本函数标准化（scale/pad 1280x720 → 25fps）+ 透明文字 PNG 淡入叠加。

音频统一由 _audio_chain 构建：BGM（bgm_music 库选一）裁 10s 淡入淡出 +
可选频道名 TTS 播报（合成由调用方完成，这里只混音）；两路线产物规格一致：
1280x720 / 25fps / yuv420p / aac 44100 立体声 / 10s（与 sleep 块 concat 兼容）。
"""
import math
import os
import random
import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from media_utils import FONT_ZH, TARGET_H, TARGET_W, VF_NORM, get_duration
from sleep.sleep_cards import _draw_badge, _fit_font, _handwrite_path, _hex_rgb

INTRO_DURATION = 10.0
INTRO_FPS = 25
FFMPEG_TIMEOUT = 600


def _run_ffmpeg(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace",
                          timeout=FFMPEG_TIMEOUT)


def _log(msg: str, progress_cb=None, pct: int | None = None) -> None:
    print(f"  [Intro] {msg}")
    if progress_cb and pct is not None:
        progress_cb(pct, msg)


# ---------------------------------------------------------------------------
# BGM 库
# ---------------------------------------------------------------------------

_BGM_EXTS = (".mp3", ".wav", ".m4a", ".flac", ".ogg")


def list_bgm_files(bgm_dir: str) -> list[str]:
    """bgm_music 目录下的音频文件名（排序；目录无效返回空）。"""
    base = Path(bgm_dir) if bgm_dir else None
    if not base or not base.is_dir():
        return []
    return sorted(f.name for f in base.iterdir()
                  if f.is_file() and f.suffix.lower() in _BGM_EXTS)


def resolve_bgm(bgm_dir: str, choice: str) -> str:
    """BGM 选择 → 绝对路径。choice 为空/random = 随机一首；无库/无文件返回空。"""
    files = list_bgm_files(bgm_dir)
    if not files:
        return ""
    name = str(choice or "").strip()
    if not name or name == "random":
        name = random.choice(files)
    if name not in files:
        return ""
    return str(Path(bgm_dir) / name)


# ---------------------------------------------------------------------------
# 音频链：BGM 主控 + 可选播报混音
# ---------------------------------------------------------------------------

def _audio_chain(bgm_path: str, announce_path: str, duration: float,
                 bgm_volume_db: float, start_idx: int) -> tuple[str, list[str]]:
    """块音频 filter_complex。返回 (fg 尾段字符串, ffmpeg 输入参数列表)。

    主控流恒为 10s（BGM 裁剪 or anullsrc），播报 0.6s 后淡入叠加，
    amix duration=first 保证输出恰为片头时长（输出端仍加 -t 兜底）。
    anullsrc 必须以 -f lavfi 输入（否则被当作文件名导致整块失败）。
    """
    inputs: list[str] = []
    chains: list[str] = []
    idx = start_idx
    if bgm_path and os.path.exists(bgm_path):
        inputs += ["-i", bgm_path]
        chains.append(
            f"[{idx}:a]aresample=44100,aformat=channel_layouts=stereo,"
            f"atrim=0:{duration:.3f},asetpts=PTS-STARTPTS,"
            f"volume={bgm_volume_db:.1f}dB,"
            f"afade=t=in:st=0:d=0.8,"
            f"afade=t=out:st={max(0.0, duration - 1.2):.3f}:d=1.2[bgm]")
    else:
        inputs += ["-f", "lavfi", "-i",
                   f"anullsrc=r=44100:cl=stereo:d={duration:.3f}"]
        chains.append(f"[{idx}:a]anull[bgm]")
    idx += 1
    if announce_path and os.path.exists(announce_path):
        inputs += ["-i", announce_path]
        chains.append(f"[{idx}:a]aresample=44100,aformat=channel_layouts=stereo,"
                      f"adelay=600|600[ann]")
        chains.append("[bgm][ann]amix=inputs=2:duration=first:normalize=0[aout]")
    else:
        chains.append("[bgm]anull[aout]")
    return ";".join(chains), inputs


def _verify_intro(path: str, duration: float) -> None:
    if not os.path.exists(path) or os.path.getsize(path) < 100_000:
        raise RuntimeError(f"Intro video invalid: {path}")
    real = get_duration(path)
    if abs(real - duration) > 1.5:
        raise RuntimeError(f"Intro duration {real:.2f}s != {duration:.2f}s")


# ---------------------------------------------------------------------------
# 本地路线：Pillow 逐帧渲染
# ---------------------------------------------------------------------------

def _leaf_sprite(lw: int, lh: int, angle: float, color: tuple) -> Image.Image:
    leaf = Image.new("RGBA", (lw * 3, lh * 3), (0, 0, 0, 0))
    ld = ImageDraw.Draw(leaf)
    cx0, cy0 = lw, lh
    ld.ellipse([cx0, cy0, cx0 + lw, cy0 + lh], fill=color,
               outline=(255, 255, 255, 210), width=2)
    ld.line([cx0 + lw // 2, cy0 + 2, cx0 + lw // 2, cy0 + lh - 2],
            fill=(255, 255, 255, 170), width=2)
    return leaf.rotate(angle, expand=True, resample=Image.BICUBIC)


def _text_sprite(text: str, font_path: str, size: int, color: tuple,
                 max_w: int, glow_radius: int = 10,
                 glow_color: tuple = (255, 255, 255, 160)) -> Image.Image:
    """文字 → 透明 sprite（柔光晕 + 锐利文字），供逐帧淡入粘贴。"""
    tmp = Image.new("RGBA", (8, 8))
    td = ImageDraw.Draw(tmp)
    font = _fit_font(td, text, font_path, size, 28, max_w)
    box = td.textbbox((0, 0), text, font=font)
    tw, th = box[2] - box[0], box[3] - box[1]
    pad = glow_radius * 4 + 8
    sprite = Image.new("RGBA", (tw + pad * 2, th + pad * 2), (0, 0, 0, 0))
    sd = ImageDraw.Draw(sprite)
    origin = (pad - box[0], pad - box[1])
    if glow_radius > 0:
        glow = Image.new("RGBA", sprite.size, (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow)
        gd.text(origin, text, font=font, fill=glow_color)
        glow = glow.filter(ImageFilter.GaussianBlur(glow_radius))
        sprite = Image.alpha_composite(sprite, glow)
        sd = ImageDraw.Draw(sprite)
    sd.text(origin, text, font=font, fill=color)
    return sprite


def _faded(sprite: Image.Image, alpha: float) -> Image.Image:
    if alpha >= 0.999:
        return sprite
    s = sprite.copy()
    s.putalpha(s.getchannel("A").point(lambda v: int(v * alpha)))
    return s


def _build_static_base(theme: dict, w: int, h: int) -> tuple[Image.Image, dict]:
    """渐变背景（无叶片，叶片动画叠加）+ 白色圆角卡 + EN 徽标。"""
    top = _hex_rgb(theme["bg_top"], (234, 244, 226))
    bottom = _hex_rgb(theme["bg_bottom"], (217, 237, 207))
    base = Image.new("RGB", (w, h))
    for y in range(h):
        t = y / max(1, h - 1)
        base.paste(tuple(int(top[c] + (bottom[c] - top[c]) * t) for c in range(3)),
                   [0, y, w, y + 1])
    img = base.convert("RGBA")
    draw = ImageDraw.Draw(img)
    card = {"x0": int(w * 0.028), "y0": int(h * 0.042),
            "x1": int(w * 0.972), "y1": int(h * 0.972)}
    draw.rounded_rectangle([card["x0"], card["y0"], card["x1"], card["y1"]],
                           radius=36, fill=_hex_rgb(theme["card"], (255, 255, 255)),
                           outline=_hex_rgb(theme["card_border"], (143, 176, 201)),
                           width=2)
    _draw_badge(draw, card, theme, "EN")
    return img, card


def _render_intro_frames(frames_dir: Path, channel_name: str, subtitle: str,
                         theme: dict, duration: float, fps: int,
                         progress_cb=None, stop_check=None) -> None:
    """逐帧渲染动画：叶片漂动 + 频道名淡入浮动 + 副题淡入。"""
    w, h = TARGET_W, TARGET_H
    base, card = _build_static_base(theme, w, h)
    ca = (*_hex_rgb(theme["leaf_a"], (201, 226, 242)), 215)
    cb = (*_hex_rgb(theme["leaf_b"], (191, 224, 200)), 215)
    leaves = []
    for (lx, ly, lw, lh, ang, col, phase, period, amp) in (
            (70, 645, 95, 46, -35, ca, 0.0, 7.0, 14),
            (28, 600, 82, 42, -10, ca, 1.3, 9.0, 10),
            (115, 690, 88, 44, -60, ca, 2.1, 8.0, 12),
            (1212, 655, 98, 48, 30, cb, 0.7, 7.5, 12),
            (1256, 608, 82, 40, 60, cb, 1.9, 9.5, 10),
            (1168, 698, 88, 42, 10, cb, 2.7, 8.5, 14)):
        leaves.append({"sprite": _leaf_sprite(lw, lh, ang, col),
                       "x": lx - lw, "y": ly - lh,
                       "phase": phase, "period": period, "amp": amp})

    hw_path = _handwrite_path(theme)
    channel = (channel_name or "English with me").strip() or "English with me"
    title = _text_sprite(channel, hw_path, 96,
                         _hex_rgb(theme["channel_text"], (90, 107, 82)),
                         max_w=w - 420, glow_radius=8)
    sub = _text_sprite(subtitle or "閉上眼睛 · 輕鬆聽", FONT_ZH, 40,
                       _hex_rgb(theme["zh_text"], (58, 58, 58)),
                       max_w=w - 480, glow_radius=4)

    total = int(duration * fps)
    frames_dir.mkdir(parents=True, exist_ok=True)
    for fi in range(total):
        if stop_check and stop_check():
            raise RuntimeError("stopped")
        t = fi / fps
        img = base.copy()
        for leaf in leaves:
            dx = leaf["amp"] * math.sin(2 * math.pi * t / leaf["period"] + leaf["phase"])
            dy = (leaf["amp"] * 0.6) * math.sin(
                2 * math.pi * t / (leaf["period"] * 1.4) + leaf["phase"] * 1.7)
            img.paste(leaf["sprite"], (int(leaf["x"] + dx), int(leaf["y"] + dy)),
                      leaf["sprite"])
        cx = w / 2
        a_title = min(1.0, max(0.0, (t - 0.4) / 1.2))
        if a_title > 0:
            sy = h * 0.36 - title.height / 2 + 8 * math.sin(2 * math.pi * t / 6)
            sp = _faded(title, a_title)
            img.paste(sp, (int(cx - sp.width / 2), int(sy)), sp)
        a_sub = min(1.0, max(0.0, (t - 1.8) / 1.0))
        if a_sub > 0:
            sy = h * 0.36 + title.height / 2 + 28
            sp = _faded(sub, a_sub)
            img.paste(sp, (int(cx - sp.width / 2), int(sy)), sp)
        img.convert("RGB").save(str(frames_dir / f"frame_{fi:04d}.png"), "PNG")
        if fi % 50 == 0 or fi == total - 1:
            _log(f"Rendered frame {fi + 1}/{total}", progress_cb,
                 int(5 + fi / total * 60))


def build_local_intro(channel_name: str, subtitle: str, out_path: str,
                      theme: dict, bgm_path: str = "",
                      bgm_volume_db: float = -16.0, announce_path: str = "",
                      duration: float = INTRO_DURATION,
                      progress_cb=None, stop_check=None) -> str:
    """本地渲染 10s 片头（Pillow 帧 + FFmpeg 合成）。返回 out_path。"""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames_dir = out.parent / "intro_frames_tmp"
    shutil.rmtree(frames_dir, ignore_errors=True)
    try:
        _log(f"Rendering local intro frames ({duration:.0f}s @ {INTRO_FPS}fps)...",
             progress_cb, 5)
        _render_intro_frames(frames_dir, channel_name, subtitle, theme,
                             duration, INTRO_FPS, progress_cb, stop_check)
        fg, inputs = _audio_chain(bgm_path, announce_path, duration,
                                  bgm_volume_db, start_idx=1)
        cmd = ["ffmpeg", "-y", "-framerate", str(INTRO_FPS),
               "-i", str(frames_dir / "frame_%04d.png")]
        cmd += inputs
        cmd += ["-filter_complex", fg, "-map", "0:v:0", "-map", "[aout]",
                "-t", f"{duration:.3f}",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(INTRO_FPS),
                "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
                str(out)]
        _log("Encoding intro video...", progress_cb, 70)
        r = _run_ffmpeg(cmd)
        if r.returncode != 0:
            raise RuntimeError(f"FFmpeg intro encode failed: {(r.stderr or '')[-300:]}")
    finally:
        shutil.rmtree(frames_dir, ignore_errors=True)
    _verify_intro(str(out), duration)
    _log(f"Local intro saved: {out.name}", progress_cb, 100)
    return str(out)


# ---------------------------------------------------------------------------
# AI 路线：原始视频标准化 + 文字叠加
# ---------------------------------------------------------------------------

def _render_overlay_text(channel_name: str, subtitle: str, theme: dict) -> str:
    """AI 视频用透明文字层（白字 + 深色柔光晕，居中略偏上）。返回 PNG 路径。"""
    w, h = TARGET_W, TARGET_H
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    channel = (channel_name or "English with me").strip() or "English with me"
    title = _text_sprite(channel, _handwrite_path(theme), 104, (255, 255, 255, 255),
                         max_w=w - 360, glow_radius=14, glow_color=(0, 0, 0, 170))
    sub = _text_sprite(subtitle or "閉上眼睛 · 輕鬆聽", FONT_ZH, 42,
                       (255, 255, 255, 235), max_w=w - 420,
                       glow_radius=8, glow_color=(0, 0, 0, 150))
    img.paste(title, (int(w / 2 - title.width / 2), int(h * 0.34 - title.height / 2)),
              title)
    img.paste(sub, (int(w / 2 - sub.width / 2),
                    int(h * 0.34 + title.height / 2 + 24)), sub)
    out = str(Path(theme.get("_tmp_dir", ".")) / "intro_text_overlay.png")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    img.save(out, "PNG")
    return out


def finalize_ai_intro(src_video: str, channel_name: str, subtitle: str,
                      out_path: str, theme: dict, bgm_path: str = "",
                      bgm_volume_db: float = -16.0, announce_path: str = "",
                      duration: float = INTRO_DURATION,
                      progress_cb=None) -> str:
    """AI 原始视频 → 统一规格 + 频道名文字淡入叠加 + 音频。返回 out_path。"""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    theme = dict(theme)
    theme["_tmp_dir"] = str(out.parent)
    text_png = _render_overlay_text(channel_name, subtitle, theme)
    fg_audio, a_inputs = _audio_chain(bgm_path, announce_path, duration,
                                      bgm_volume_db, start_idx=2)
    fg = ("[0:v]scale=1280:720:force_original_aspect_ratio=decrease,"
          "pad=1280:720:(ow-iw)/2:(oh-ih)/2,fps=25[bg];"
          "[1:v]format=rgba,fade=t=in:st=0.5:d=0.9:alpha=1[txt];"
          "[bg][txt]overlay=0:0[v];") + fg_audio
    cmd = ["ffmpeg", "-y", "-i", src_video, "-loop", "1", "-i", text_png]
    cmd += a_inputs
    cmd += ["-filter_complex", fg, "-map", "[v]", "-map", "[aout]",
            "-t", f"{duration:.3f}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(INTRO_FPS),
            "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2",
            str(out)]
    _log("Standardizing AI intro video...", progress_cb, 80)
    r = _run_ffmpeg(cmd)
    try:
        os.remove(text_png)
    except OSError:
        pass
    if r.returncode != 0:
        raise RuntimeError(f"FFmpeg intro finalize failed: {(r.stderr or '')[-300:]}")
    _verify_intro(str(out), duration)
    _log(f"AI intro saved: {out.name}", progress_cb, 100)
    return str(out)


__all__ = ["INTRO_DURATION", "INTRO_FPS", "build_local_intro",
           "finalize_ai_intro", "list_bgm_files", "resolve_bgm"]
