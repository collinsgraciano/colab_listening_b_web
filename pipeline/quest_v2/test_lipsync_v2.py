"""quest_v2 唇形同步独立验证脚本（不消耗 MCP 积分，无需 LLM/TTS）。

覆盖：
  1. audio_envelope: 合成「语音/静音」交替音频 → 包络节律 + 混合权重去抖
  2. 站位稳定纯函数: 角色身份固定槽位（换 speaker 不横跳）
  3. _render_sm_segment_v2: 假姿势对（仅嘴部不同）+ 合成音频 → 渲染 →
     抽帧断言「说话帧嘴部亮、静音帧嘴部暗」，背景区不变
  4. 降级路径: poses_closed 为空 → 正常渲染不崩
  5. quest 原模式回归: compose_quest 签名无 lip_sync（零改动验证）

运行：python pipeline/quest_v2/test_lipsync_v2.py
"""
import math
import os
import struct
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

_PARENT = str(Path(__file__).parent.parent.resolve())
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from audio_envelope import extract_rms_envelope, mouth_blend_weights
from quest_v2.video_compose_quest_v2 import _char_positions_v2, _render_sm_segment_v2

SR = 16000
PASS, FAIL = "OK", "FAIL"
_results = []


def check(name: str, cond: bool, detail: str = ""):
    status = PASS if cond else FAIL
    _results.append((name, status, detail))
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))


# ---------------------------------------------------------------------------
# 音频合成（wave + math，无第三方依赖）
# ---------------------------------------------------------------------------

def synth_audio(path: str, pattern: list[tuple[float, bool]]):
    """合成 test 音频：pattern = [(时长, 是否发声)]，发声=220Hz 调幅正弦。"""
    samples = []
    for dur, on in pattern:
        n = int(dur * SR)
        for i in range(n):
            if on:
                v = 0.6 * math.sin(2 * math.pi * 220 * i / SR) * (
                    0.7 + 0.3 * math.sin(2 * math.pi * 3 * i / SR))
            else:
                v = 0.0
            samples.append(v)
    with wave.open(path, "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(b"".join(
            struct.pack("<h", int(max(-1.0, min(1.0, v)) * 32767))
            for v in samples))


# ---------------------------------------------------------------------------
# 1) 包络 + 混合权重
# ---------------------------------------------------------------------------

def test_envelope(audio_path: str, fps: int = 8):
    # pattern: 1s 语音 / 0.5s 静音 × 3 = 4.5s
    env = extract_rms_envelope(audio_path, fps)
    n = len(env)
    check("envelope 长度≈4.5s×fps", abs(n - 4.5 * fps) <= 2, f"n={n}")

    # t=0.5（语音）应为高能量；t=1.25（静音）应≈0
    i_speech = int(0.5 * fps)
    i_silence = int(1.25 * fps)
    check("语音帧包络高", env[i_speech] > 0.5, f"env[{i_speech}]={env[i_speech]:.3f}")
    check("静音帧包络≈0", env[i_silence] < 0.05, f"env[{i_silence}]={env[i_silence]:.3f}")

    w = mouth_blend_weights(env, fps=fps)
    check("权重长度一致", len(w) == n, f"len={len(w)}")
    check("语音帧张嘴权重高", w[i_speech] > 0.5, f"w[{i_speech}]={w[i_speech]:.3f}")
    check("静音帧闭嘴权重=0", w[i_silence] == 0.0, f"w[{i_silence}]={w[i_silence]:.3f}")

    # 去抖：无高频振荡 —— 权重向上穿越 0.5 的次数应≈语音段数（3 段 + 容差），
    # 而非逐帧跳变幅度（边界处一次较大跳变是自然的嘴部开合）
    up_crossings = sum(1 for i in range(1, n)
                       if w[i - 1] < 0.5 <= w[i])
    check("无高频振荡（0.5 上穿次数少）", up_crossings <= 6,
          f"up_crossings={up_crossings}（语音段=3）")


# ---------------------------------------------------------------------------
# 2) 站位稳定纯函数
# ---------------------------------------------------------------------------

def test_positions():
    p = _char_positions_v2(["char_a", "char_b"])
    check("2人: char_a 左 / char_b 右", p == [0.35, 0.65], f"{p}")
    # 乱序传入 on_screen（LLM 可能给 ["char_b","char_a"]）→ 身份固定
    p2 = _char_positions_v2(["char_b", "char_a"])
    check("乱序输入仍按身份固定", p2 == [0.65, 0.35], f"{p2}")
    p3 = _char_positions_v2(["char_a", "char_c"])
    check("a+c 组合: a 左 / c 右", p3 == [0.35, 0.65], f"{p3}")
    p4 = _char_positions_v2(["char_a", "char_b", "char_c"])
    check("3人槽位", p4 == [0.30, 0.55, 0.80], f"{p4}")
    p5 = _char_positions_v2(["char_c"])
    check("单人居中", p5 == [0.5], f"{p5}")
    # 同一组角色两次调用一致（speaker 换人不影响 —— 函数与 speaker 无关）
    check("确定性", _char_positions_v2(["char_a", "char_b"]) == p)


# ---------------------------------------------------------------------------
# 3) 渲染级唇同步验证
# ---------------------------------------------------------------------------

def _make_fake_pose(path: str, mouth_open: bool):
    """假角色姿势：透明背景 + 蓝色身体 + 嘴部区域（张嘴=亮红 / 闭嘴=暗色）。"""
    from PIL import Image, ImageDraw
    img = Image.new("RGBA", (500, 600), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rectangle([100, 80, 400, 560], fill=(60, 120, 220, 255))  # 身体
    # 眼睛（不变区域）
    d.ellipse([160, 150, 200, 190], fill=(255, 255, 255, 255))
    d.ellipse([300, 150, 340, 190], fill=(255, 255, 255, 255))
    # 嘴部（唯一变化区域）
    if mouth_open:
        d.rectangle([170, 230, 330, 300], fill=(255, 60, 60, 255))
    else:
        d.rectangle([170, 230, 330, 300], fill=(60, 20, 20, 255))
    img.save(path)


def _make_fake_bg(path: str):
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (1280, 720), (30, 34, 44))
    d = ImageDraw.Draw(img)
    for y in range(0, 720, 40):
        d.line([(0, y), (1280, y)], fill=(40, 44, 55))
    img.save(path)


def test_render_lipsync(tmp: Path):
    from PIL import Image
    import numpy as np

    pose_open = str(tmp / "pose_open.png")
    pose_closed = str(tmp / "pose_closed.png")
    _make_fake_pose(pose_open, mouth_open=True)
    _make_fake_pose(pose_closed, mouth_open=False)
    bg = str(tmp / "bg.png")
    _make_fake_bg(bg)
    audio = str(tmp / "audio.wav")
    synth_audio(audio, [(1.0, True), (0.5, False)] * 3)

    duration = 4.5
    fade_af = f"afade=t=in:st=0:d=0.05,afade=t=out:st={duration-0.05:.2f}:d=0.05"
    out = str(tmp / "seg_lipsync.mp4")
    ok = _render_sm_segment_v2(
        [{"poses": [pose_open], "poses_closed": [pose_closed],
          "is_speaker": True, "char_key": "char_a"}],
        bg, audio, out, duration,
        tmp / "frames_a", tmp / "cache",
        render_fps=8, seed=7, direction=1, fade_af=fade_af, lip_sync=True)
    check("渲染成功（带唇同步）", ok and os.path.getsize(out) > 1000)

    # 抽帧：t=0.5（语音→张嘴）vs t=1.25（静音→闭嘴）
    f_open = str(tmp / "f_open.png")
    f_closed = str(tmp / "f_closed.png")
    for t, dst in ((0.5, f_open), (1.25, f_closed)):
        subprocess.run(["ffmpeg", "-y", "-ss", str(t), "-i", out,
                        "-frames:v", "1", dst],
                       capture_output=True, timeout=60)
    a = np.asarray(Image.open(f_open).convert("RGB")).astype(int)
    b = np.asarray(Image.open(f_closed).convert("RGB")).astype(int)
    check("两帧存在", a.shape == b.shape and a.shape[:2] == (720, 1280))

    # 嘴部区域（单人槽位 0.5×1280=640，缩放后嘴部约在 (540,290)-(750,400)）
    mouth_a = a[290:400, 540:750]
    mouth_b = b[290:400, 540:750]
    red_a = mouth_a[:, :, 0].mean()
    red_b = mouth_b[:, :, 0].mean()
    check("语音帧嘴部红通道显著高于静音帧", red_a - red_b > 20,
          f"open={red_a:.1f} closed={red_b:.1f}")

    # 背景角落不变（变化只发生在角色嘴部）
    corner_diff = np.abs(a[:100, :100] - b[:100, :100]).mean()
    check("背景角落无变化", corner_diff < 3.0, f"diff={corner_diff:.2f}")


# ---------------------------------------------------------------------------
# 4) 降级路径（无 closed 配对）
# ---------------------------------------------------------------------------

def test_render_degraded(tmp: Path):
    pose_open = str(tmp / "pose_open.png")
    bg = str(tmp / "bg.png")
    audio = str(tmp / "audio.wav")
    out = str(tmp / "seg_degraded.mp4")
    duration = 2.0
    fade_af = f"afade=t=in:st=0:d=0.05,afade=t=out:st={duration-0.05:.2f}:d=0.05"
    ok = _render_sm_segment_v2(
        [{"poses": [pose_open], "poses_closed": [],
          "is_speaker": True, "char_key": "char_a"}],
        bg, audio, out, duration,
        tmp / "frames_b", tmp / "cache",
        render_fps=8, seed=7, direction=1, fade_af=fade_af, lip_sync=True)
    check("降级渲染成功（无 closed 配对不崩）", ok and os.path.getsize(out) > 1000)

    # lip_sync=False 显式关闭也应正常
    out2 = str(tmp / "seg_off.mp4")
    ok2 = _render_sm_segment_v2(
        [{"poses": [pose_open], "poses_closed": [pose_open],
          "is_speaker": True, "char_key": "char_a"}],
        bg, audio, out2, duration,
        tmp / "frames_c", tmp / "cache",
        render_fps=8, seed=7, direction=1, fade_af=fade_af, lip_sync=False)
    check("lip_sync=False 渲染成功", ok2 and os.path.getsize(out2) > 1000)


# ---------------------------------------------------------------------------
# 5) quest 原模式回归（零改动验证）
# ---------------------------------------------------------------------------

def test_quest_untouched():
    import inspect
    from quest.video_compose_quest import compose_quest, _render_sm_segment
    sig = inspect.signature(compose_quest)
    check("quest compose_quest 签名无 lip_sync（未被改动）",
          "lip_sync" not in sig.parameters,
          f"params={list(sig.parameters)}")
    sig2 = inspect.signature(_render_sm_segment)
    check("quest _render_sm_segment 签名无 lip_sync",
          "lip_sync" not in sig2.parameters)

    from structures import STRUCTURES, list_structures
    check("structures 注册表含 quest_v2", "quest_v2" in list_structures())
    check("quest_v2 复用 quest 脚本生成器",
          STRUCTURES["quest_v2"]["generate_script"]
          is STRUCTURES["quest"]["generate_script"])
    check("quest_v2 独立 compose", STRUCTURES["quest_v2"]["compose"]
          is not STRUCTURES["quest"]["compose"])

    # argparse 接受 quest_v2 / --no-lip-sync
    import pipeline as pl
    old_argv = sys.argv[:]
    try:
        sys.argv = ["pipeline.py", "--structure", "quest_v2"]
        args = pl._parse_args()
        check("argparse 接受 quest_v2", args.structure == "quest_v2")
        check("lip_sync 默认开启", args.lip_sync is True)
        sys.argv = ["pipeline.py", "--structure", "quest_v2", "--no-lip-sync"]
        args2 = pl._parse_args()
        check("--no-lip-sync 关闭", args2.lip_sync is False)
        sys.argv = ["pipeline.py", "--structure", "quest"]
        args3 = pl._parse_args()
        check("quest 仍可解析", args3.structure == "quest")
    finally:
        sys.argv = old_argv


# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("quest_v2 唇形同步验证")
    print("=" * 60)
    tmp = Path(tempfile.mkdtemp(prefix="lipsync_test_"))
    try:
        audio = str(tmp / "audio.wav")
        synth_audio(audio, [(1.0, True), (0.5, False)] * 3)
        print("\n[1] audio_envelope 包络与混合权重")
        test_envelope(audio)
        print("\n[2] 站位稳定纯函数")
        test_positions()
        print("\n[3] 渲染级唇同步（假姿势 + 合成音频）")
        test_render_lipsync(tmp)
        print("\n[4] 降级路径")
        test_render_degraded(tmp)
        print("\n[5] quest 原模式回归")
        test_quest_untouched()
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)

    n_fail = sum(1 for _, s, _ in _results if s == FAIL)
    print("\n" + "=" * 60)
    print(f"结果: {len(_results) - n_fail}/{len(_results)} 通过"
          + (f"，{n_fail} 失败" if n_fail else ""))
    print("=" * 60)
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
