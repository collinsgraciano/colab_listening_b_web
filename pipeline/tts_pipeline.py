"""Batch TTS generation for all dialogue/narration audio."""
import json
import os
from pathlib import Path

from tts_engine import TTSEngine, build_voice_map, get_zh_voice
from media_utils import get_duration


def _audio_exists(path: str) -> bool:
    """Check if audio file exists and is non-empty (>100 bytes)."""
    return bool(path) and os.path.exists(path) and os.path.getsize(path) > 100


def _invalidate_cache_if_rate_changed(audio_dir, tts_rate, quest, tts_engine, extra_sig=""):
    """如果 tts_rate / 引擎参数配置变化，清除 audio_dir 中所有 .mp3 缓存文件。"""
    audio_path = Path(audio_dir)
    meta_path = audio_path / ".tts_meta.json"
    sig = {
        "tts_rate": tts_rate or "",
        "quest": quest,
        "engine": tts_engine,
    }
    if extra_sig:
        sig["extra"] = extra_sig
    current_sig = json.dumps(sig)
    if meta_path.exists():
        try:
            saved = meta_path.read_text(encoding="utf-8")
            if saved == current_sig:
                return  # 签名一致，缓存有效
        except OSError:
            pass
    # 签名不同或首次运行 — 清除旧缓存
    if audio_path.exists():
        cleared = 0
        for f in audio_path.glob("*.mp3"):
            try:
                f.unlink()
                cleared += 1
            except OSError:
                pass
        if cleared:
            print(f"  [TTS] Rate config changed, cleared {cleared} cached audio files.")
    audio_path.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(current_sig, encoding="utf-8")


def generate_tts(script, dialogue, audio_dir, results, quest=False, host_narration=False,
                 tts_rate=None, tts_engine="kokoro", stop_check=None):
    """Generate all TTS audio. Runs in a thread.

    Produces narration and dialogue EN/ZH audio.
    Writes results into the *results* dict (shared with caller).

    Args:
        host_narration: True 时按 quest 风格生成旁白 welcome/hook/outro
                        （original_cutout 主持人开场/结尾用），但仍生成中文对话。
        tts_rate: Override dialogue English TTS rate (e.g. '-15%', '0%').
                  If None, uses mode default (quest: '0%', non-quest: '-15%').
        tts_engine: 'kokoro' (default, local Kokoro TTS) or 'qwen'
                    (Qwen3-TTS local GPU).
        stop_check: Optional callable returning True to abort early.
    """
    # --- Engine + voice map setup ---
    if tts_engine == "qwen":
        from qwen_tts_engine import QwenTTSEngine, build_qwen_voice_map

        model_path = os.environ.get("QWEN_MODEL_PATH", r"H:\models\Qwen3-TTS-12Hz-0.6B-CustomVoice")
        base_model_path = os.environ.get("QWEN_BASE_MODEL_PATH", r"H:\models\Qwen3-TTS-12Hz-1.7B-Base")
        voicedesign_model_path = os.environ.get(
            "QWEN_VOICEDSIGN_MODEL_PATH", r"H:\models\Qwen3-TTS-12Hz-1.7B-VoiceDesign")
        device = os.environ.get("QWEN_DEVICE", "cuda:0")
        tts = QwenTTSEngine(model_path, device, base_model_path, voicedesign_model_path)
        voice_map = build_qwen_voice_map(script)
        narration_voice = voice_map.get("narration", voice_map.get("host", "Serena"))
        voice_label = "qwen"
    elif tts_engine == "moss":
        from moss_tts_engine import MossTTSEngine, build_moss_voice_map, get_moss_zh_default

        model_path = os.environ.get("MOSS_MODEL_PATH") or r"H:\models\MOSS-TTS-Nano-Model"
        tokenizer_path = os.environ.get("MOSS_TOKENIZER_PATH") or r"H:\models\MOSS-Audio-Tokenizer-Nano"
        device = os.environ.get("MOSS_DEVICE") or "cpu"
        repo_dir = os.environ.get("MOSS_REPO_DIR") or r"H:\models\MOSS-TTS-Nano"
        tts = MossTTSEngine(model_path, device, tokenizer_path, repo_dir)
        voice_map = build_moss_voice_map(script)
        narration_voice = voice_map.get("narration", voice_map.get("host", "Bella"))
        voice_label = "moss"
    else:
        tts = TTSEngine()
        voice_map = build_voice_map(script)
        narration_voice = voice_map.get("narration", voice_map.get("host", "af_sky"))
        voice_label = "kokoro"

    # Narration voice: check script for custom binding (from character_voices UI)
    _narration_voice = script.get("narration_qwen_speaker", "").strip()
    if _narration_voice:
        narration_voice = _narration_voice
    elif (script.get("host_character") or "").strip():
        # 形象绑定角色出镜时，旁白音色联动该角色（显式旁白绑定仍最高优先）
        narration_voice = voice_map.get(script["host_character"].strip(), narration_voice)
    narration_zh_voice = script.get("narration_zh_voice", "").strip() or narration_voice

    # 检测 tts_rate / 引擎参数变化，必要时清除旧缓存
    extra_sig = ""
    if tts_engine == "moss":
        # MOSS 温度/句间停顿影响输出 — 参数变化时自动重新生成
        extra_sig = (f"{os.environ.get('MOSS_TTS_TEMPERATURE', '0.8')}"
                     f"|{os.environ.get('MOSS_TTS_GAP_MS', '120')}")
    _invalidate_cache_if_rate_changed(audio_dir, tts_rate, quest, tts_engine,
                                      f"{extra_sig}|host:{int(host_narration)}")

    narration = {}
    if quest or host_narration:
        texts = [
            ("welcome", script.get("welcome_en", "")),
            ("hook", script.get("hook_intro_en") or script.get("story_hook", "")),
            ("outro", script.get("outro", "That's all for today. Keep practicing!")),
        ]
        if host_narration and not quest:
            # original_cutout：主持人开场沿用 quest 三段旁白，但 Ch3 桥接词仍需要
            texts.append(("practice_intro",
                          script.get("practice_intro_en",
                                     "Now let's practice. Listen and repeat each sentence.")))
        quest_narration_rate = tts_rate if tts_rate else "-10%"
        for name, text in texts:
            if stop_check and stop_check():
                print("  [TTS] Stop requested, aborting narration.", flush=True)
                results["fatal_error"] = "stopped"
                return
            if text:
                path = str(audio_dir / f"{name}.mp3")
                if _audio_exists(path):
                    dur = get_duration(path)
                    narration[name] = path
                    print(f"  [TTS] {name}: {dur:.1f}s (cached)")
                    continue
                dur = tts.synth_english(text, narration_voice, path, rate=quest_narration_rate)
                narration[name] = path
                print(f"  [TTS] {name}: {dur:.1f}s")
    else:
        intro_text = script.get("story_hook", "")
        outro_text = script.get("outro", "That's all for today. Keep practicing!")
        practice_intro_text = script.get("practice_intro_en", "Now let's practice. Listen and repeat each sentence.")

        narration_rate = tts_rate if tts_rate else "+0%"
        for name, text in [("intro", intro_text), ("outro", outro_text), ("practice_intro", practice_intro_text)]:
            if stop_check and stop_check():
                print("  [TTS] Stop requested, aborting narration.", flush=True)
                results["fatal_error"] = "stopped"
                return
            if text:
                path = str(audio_dir / f"{name}.mp3")
                if _audio_exists(path):
                    dur = get_duration(path)
                    narration[name] = path
                    print(f"  [TTS] {name}: {dur:.1f}s (cached)")
                    continue
                dur = tts.synth_english(text, narration_voice, path, rate=narration_rate)
                narration[name] = path
                print(f"  [TTS] {name}: {dur:.1f}s")

    # Dialogue English (character voices; tts_rate overrides mode default)
    dialogue_rate = tts_rate if tts_rate else ("0%" if quest else "-15%")
    normal_paths = []
    dialogue_durations = []
    for i, line in enumerate(dialogue):
        if stop_check and stop_check():
            print("  [TTS] Stop requested, aborting dialogue EN.", flush=True)
            results["fatal_error"] = "stopped"
            return
        text = line.get("text", "")
        speaker = line.get("speaker", "char_a")
        voice = voice_map.get(speaker, voice_map.get("char_a", "af_sarah"))
        path = str(audio_dir / f"dialogue_{i}.mp3")
        if _audio_exists(path):
            dur = get_duration(path)
            normal_paths.append(path)
            dialogue_durations.append(dur)
            print(f"  [TTS] dialogue_{i}: {dur:.1f}s (cached)")
            continue
        dur = tts.synth_english(text, voice, path, rate=dialogue_rate)
        normal_paths.append(path)
        dialogue_durations.append(dur)
        print(f"  [TTS] dialogue_{i}: {dur:.1f}s ({voice_label})")

    # Dialogue Chinese (quest skips entirely)
    zh_paths = []
    if not quest:
        zh_rate = tts_rate if tts_rate else "-10%"
        for i, line in enumerate(dialogue):
            if stop_check and stop_check():
                print("  [TTS] Stop requested, aborting dialogue ZH.", flush=True)
                results["fatal_error"] = "stopped"
                return
            text = line.get("zh", "")
            if not text:
                zh_paths.append("")
                continue
            speaker = line.get("speaker", "char_a")
            # 中文音频音色选择：
            # Qwen 引擎 → 优先 zh_voice 绑定，回退 voice_map（角色英文音色）
            # Kokoro 引擎 → get_zh_voice（优先 zh_voice 绑定，回退性别 edge-tts 默认）
            # MOSS 引擎 → 优先 moss_voice 绑定，否则按性别用中文预设（英文参考音说中文带口音）
            zh_bind = script.get(f"{speaker}_zh_voice", "").strip()
            if tts_engine == "qwen":
                voice = zh_bind or voice_map.get(speaker, voice_map.get("char_a", "Vivian"))
            elif tts_engine == "moss":
                moss_bind = script.get(f"{speaker}_moss_voice", "").strip()
                if moss_bind:
                    voice = moss_bind
                else:
                    gender = script.get(f"{speaker}_gender", "").lower()
                    voice = get_moss_zh_default(gender)
            else:
                voice = zh_bind or get_zh_voice(speaker, script)
            path = str(audio_dir / f"zh_{i}.mp3")
            if _audio_exists(path):
                dur = get_duration(path)
                zh_paths.append(path)
                print(f"  [TTS] zh_{i}: {dur:.1f}s (cached)")
                continue
            dur = tts.synth_chinese(text, voice, path, rate=zh_rate)
            zh_paths.append(path)
            print(f"  [TTS] zh_{i}: {dur:.1f}s")

    results["narration"] = narration
    results["normal_paths"] = normal_paths
    results["dialogue_durations"] = dialogue_durations
    results["zh_paths"] = zh_paths
    results["vocab_paths"] = []
    results["slow_paths"] = []
    results["slow_durations"] = []
    results["quiz_paths"] = []
    print("  [TTS] All TTS generation complete.")
