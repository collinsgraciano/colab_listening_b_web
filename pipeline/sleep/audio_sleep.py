"""sleep 模式音频准备：每 4 次常速 TTS 派生整组 9 段音频。

每组生成：a_m/b_m（男声常速）、a_f/b_f（女声常速，中间产物）、
a_slow/b_slow（atempo 派生慢速，保音高）、combo（A男+B女 0.15s 间隔拼接）；
另有 intro（频道名播报）/ outro（结束语），均男声。
跨引擎（kokoro/qwen/moss）复用 tts_pipeline 同款引擎装配 + kokoro 单句回退。
音频文件保持引擎原生格式，块合成时统一 aresample/立体声（video_compose_sleep）。
"""
import os
import subprocess
from pathlib import Path

from media_utils import get_duration


def _sleep_meta_sig(tts_engine: str, slow_rate: float) -> str:
    return f"sleep|{tts_engine}|{slow_rate:.3f}"


def _check_sleep_cache(audio_dir: Path, sig: str) -> None:
    """引擎/慢速参数变化时清除旧 sleep 音频缓存（.tts_meta.json 签名机制）。"""
    meta_path = audio_dir / ".tts_meta.json"
    current = meta_path.read_text(encoding="utf-8") if meta_path.exists() else ""
    if current == sig:
        return
    if current.startswith("sleep|"):
        removed = 0
        for f in audio_dir.glob("pair_*.mp3"):
            f.unlink()
            removed += 1
        for name in ("intro_sleep.mp3", "outro_sleep.mp3"):
            p = audio_dir / name
            if p.exists():
                p.unlink()
                removed += 1
        if removed:
            print(f"  [Sleep] TTS 参数变化（{current} → {sig}），清除 {removed} 个旧音频缓存")
    meta_path.write_text(sig, encoding="utf-8")


def _atempo_file(src: str, dst: str, rate: float) -> float:
    """常速音频 atempo 变慢（rate<1 变慢），保音高。返回新时长。"""
    cmd = ["ffmpeg", "-y", "-i", src,
           "-filter:a", f"atempo={rate:.4f}",
           "-c:a", "libmp3lame", "-b:a", "128k", dst]
    r = subprocess.run(cmd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=120)
    if r.returncode != 0 or not os.path.exists(dst):
        raise RuntimeError(f"atempo failed for {Path(src).name}: {r.stderr[-200:]}")
    return get_duration(dst)


def _combo_file(src_a: str, src_b: str, dst: str, gap: float = 0.15) -> float:
    """A男 + 微间隔 + B女 常速拼接（统一 44100 立体声）。返回时长。"""
    fg = (f"[0:a]aresample=44100,aformat=channel_layouts=stereo[0a];"
          f"[1:a]aresample=44100,aformat=channel_layouts=stereo[1a];"
          f"anullsrc=r=44100:cl=stereo:d={gap:.3f}[sil];"
          f"[0a][sil][1a]concat=n=3:v=0:a=1[out]")
    cmd = ["ffmpeg", "-y", "-i", src_a, "-i", src_b,
           "-filter_complex", fg, "-map", "[out]",
           "-c:a", "libmp3lame", "-b:a", "128k", dst]
    r = subprocess.run(cmd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=120)
    if r.returncode != 0 or not os.path.exists(dst):
        raise RuntimeError(f"combo concat failed: {r.stderr[-200:]}")
    return get_duration(dst)


def pair_steps(audio_dir: Path, i: int) -> dict:
    """第 i 组（1-based）各步骤音频路径表。"""
    base = f"pair_{i:04d}"
    return {step: str(audio_dir / f"{base}_{step}.mp3")
            for step in ("a_m", "a_f", "b_m", "b_f", "a_slow", "b_slow", "combo")}


def load_sleep_audio_results(audio_dir: Path, num_pairs: int) -> dict | None:
    """从已存在文件重建 sleep 音频结果（resume / 完整性校验）。

    全部组 5 步骤（a_m/a_slow/b_m/b_slow/combo）+ intro/outro 齐全才返回，
    否则 None（交给正常生成流程按文件续传）。
    """
    audio_dir = Path(audio_dir)
    intro = audio_dir / "intro_sleep.mp3"
    outro = audio_dir / "outro_sleep.mp3"
    if not (intro.exists() and outro.exists()):
        return None
    pair_paths, pair_durs = {}, {}
    for i in range(1, num_pairs + 1):
        paths = pair_steps(audio_dir, i)
        need = ("a_m", "a_slow", "b_m", "b_slow", "combo")
        if not all(os.path.exists(paths[s]) for s in need):
            return None
        pair_paths[str(i).zfill(4)] = {s: paths[s] for s in need}
        pair_durs[str(i).zfill(4)] = {s: get_duration(paths[s]) for s in need}
    return {
        "intro": str(intro), "intro_dur": get_duration(str(intro)),
        "outro": str(outro), "outro_dur": get_duration(str(outro)),
        "pair_paths": pair_paths, "pair_durs": pair_durs,
    }


def prepare_sleep_audio(script: dict, audio_dir: Path, num_pairs: int,
                        tts_engine: str = "kokoro", slow_rate: float = 0.8,
                        channel_name: str = "", outro_text: str = "",
                        stop_check=None) -> dict:
    """生成全部 sleep 音频（文件级续传）。返回 results dict（见 load_*）。"""
    audio_dir = Path(audio_dir)
    audio_dir.mkdir(parents=True, exist_ok=True)
    _check_sleep_cache(audio_dir, _sleep_meta_sig(tts_engine, slow_rate))

    # --- 引擎装配（照抄 tts_pipeline 三分支 + kokoro 回退）---
    if tts_engine == "qwen":
        from qwen_tts_engine import QwenTTSEngine, build_qwen_voice_map
        tts = QwenTTSEngine(
            os.environ.get("QWEN_MODEL_PATH",
                           r"H:\models\Qwen3-TTS-12Hz-0.6B-CustomVoice"),
            os.environ.get("QWEN_DEVICE", "cuda:0"),
            os.environ.get("QWEN_BASE_MODEL_PATH",
                           r"H:\models\Qwen3-TTS-12Hz-1.7B-Base"),
            os.environ.get("QWEN_VOICEDSIGN_MODEL_PATH",
                           r"H:\models\Qwen3-TTS-12Hz-1.7B-VoiceDesign"))
        voice_map = build_qwen_voice_map(script, "sleep")
    elif tts_engine == "moss":
        from moss_tts_engine import MossTTSEngine, build_moss_voice_map
        tts = MossTTSEngine(
            os.environ.get("MOSS_MODEL_PATH") or r"H:\models\MOSS-TTS-Nano-Model",
            os.environ.get("MOSS_DEVICE") or "cpu",
            os.environ.get("MOSS_TOKENIZER_PATH") or r"H:\models\MOSS-Audio-Tokenizer-Nano",
            os.environ.get("MOSS_REPO_DIR") or r"H:\models\MOSS-TTS-Nano")
        voice_map = build_moss_voice_map(script, "sleep")
    else:
        from tts_engine import TTSEngine, build_voice_map
        tts = TTSEngine()
        voice_map = build_voice_map(script, "sleep")
    male_voice = voice_map.get("char_a", "am_adam")
    female_voice = voice_map.get("char_b", "af_sarah")

    _fb = {"eng": None, "map": None}

    def _kokoro():
        from tts_engine import TTSEngine, build_voice_map
        if _fb["eng"] is None:
            _fb["eng"] = TTSEngine()
        if _fb["map"] is None:
            _fb["map"] = build_voice_map(script, "sleep")
        return _fb["eng"], _fb["map"]

    def _synth(text: str, voice: str, path: str) -> float:
        try:
            return tts.synth_english(text, voice, path, rate="+0%")
        except Exception as e:
            if tts_engine == "kokoro":
                raise
            print(f"  [Sleep][Fallback] {Path(path).name} {type(e).__name__}: {e}")
            eng, kmap = _kokoro()
            fb_voice = kmap.get("char_a") if voice == male_voice else kmap.get("char_b")
            return eng.synth_english(text, fb_voice or "af_sarah", path, rate="+0%")

    narration_voice = male_voice
    intro = audio_dir / "intro_sleep.mp3"
    outro = audio_dir / "outro_sleep.mp3"
    if not intro.exists():
        _synth(channel_name or "English with me", narration_voice, str(intro))
    if not outro.exists():
        _synth(outro_text or "Thanks for listening. See you next time!",
               narration_voice, str(outro))

    dialogue = script.get("dialogue", [])
    rows_a = dialogue[0::2]
    rows_b = dialogue[1::2]
    total = min(num_pairs, len(rows_a), len(rows_b))

    pair_paths, pair_durs = {}, {}
    for i in range(1, total + 1):
        if stop_check and stop_check():
            print("  [Sleep] Stop requested, aborting audio prep.", flush=True)
            raise RuntimeError("stopped")
        paths = pair_steps(audio_dir, i)
        text_a = rows_a[i - 1].get("text", "")
        text_b = rows_b[i - 1].get("text", "")
        # 5 个消费步骤齐全 → 整组跳过（中间产物 a_f/b_f 可缺席，缺则按需补）
        if all(os.path.exists(paths[s]) for s in ("a_m", "a_slow", "b_m", "b_slow", "combo")):
            pair_paths[str(i).zfill(4)] = paths
            pair_durs[str(i).zfill(4)] = {
                s: get_duration(paths[s]) for s in paths if s != "a_f" and s != "b_f"}
            continue
        if not os.path.exists(paths["a_m"]):
            _synth(text_a, male_voice, paths["a_m"])
        if not os.path.exists(paths["b_m"]):
            _synth(text_b, male_voice, paths["b_m"])
        if not os.path.exists(paths["a_f"]):
            _synth(text_a, female_voice, paths["a_f"])
        if not os.path.exists(paths["b_f"]):
            _synth(text_b, female_voice, paths["b_f"])
        if not os.path.exists(paths["a_slow"]):
            _atempo_file(paths["a_f"], paths["a_slow"], slow_rate)
        if not os.path.exists(paths["b_slow"]):
            _atempo_file(paths["b_f"], paths["b_slow"], slow_rate)
        if not os.path.exists(paths["combo"]):
            _combo_file(paths["a_m"], paths["b_f"], paths["combo"])
        pair_paths[str(i).zfill(4)] = paths
        pair_durs[str(i).zfill(4)] = {
            s: get_duration(paths[s]) for s in ("a_m", "a_slow", "b_m", "b_slow", "combo")}
        if i % 10 == 0 or i == total:
            print(f"  [Sleep] Audio pairs {i}/{total} done")

    return {
        "intro": str(intro), "intro_dur": get_duration(str(intro)),
        "outro": str(outro), "outro_dur": get_duration(str(outro)),
        "pair_paths": pair_paths, "pair_durs": pair_durs,
    }
