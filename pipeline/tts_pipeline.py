"""Batch TTS generation for all dialogue/narration audio."""
import json
import os
from pathlib import Path

from tts_engine import TTSEngine, build_voice_map, get_zh_voice
from media_utils import get_duration


def _audio_exists(path: str) -> bool:
    """Check if audio file exists and is non-empty (>100 bytes)."""
    return bool(path) and os.path.exists(path) and os.path.getsize(path) > 100


# ---------------------------------------------------------------------------
# TTS 语速默认值（唯一权威定义）+ 解析
# kind: dialogue=对话英文 / zh=中文台词 / narration=旁白
# structure: quest / original / original_static / original_cutout（'_' 为兜底行）
# ---------------------------------------------------------------------------
TTS_RATE_DEFAULTS = {
    "dialogue":  {"quest": "0%", "original_cutout": "0%", "_": "-15%"},
    "zh":        {"original_cutout": "0%", "_": "-10%"},   # quest 不生成中文
    "narration": {"quest": "-10%", "original_cutout": "0%", "_": "+0%"},
}


def resolve_tts_rate(kind: str, structure: str,
                     override: str | None = None,
                     legacy: str | None = None) -> str:
    """解析某类内容的实际语速。

    优先级: 分项 override > 旧全局 legacy(--tts-rate) > 模式默认。
    """
    if override:
        return str(override)
    if legacy:
        return str(legacy)
    table = TTS_RATE_DEFAULTS.get(kind, {})
    return table.get(structure) or table.get("_", "+0%")


def _invalidate_cache_if_rate_changed(audio_dir, rates, quest, tts_engine, extra_sig=""):
    """如果语速 / 引擎参数配置变化，清除 audio_dir 中所有 .mp3 缓存文件。

    rates 为解析后的实际语速 dict（含模式默认），任何来源的变化都会触发清缓存。
    """
    audio_path = Path(audio_dir)
    meta_path = audio_path / ".tts_meta.json"
    sig = {
        "rates": rates,
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
                 tts_rate=None, tts_rate_en=None, tts_rate_zh=None, tts_rate_narration=None,
                 tts_engine="kokoro", stop_check=None,
                 include_zh=True, structure=None):
    """Generate all TTS audio. Runs in a thread.

    Produces narration and dialogue EN/ZH audio.
    Writes results into the *results* dict (shared with caller).

    Args:
        host_narration: True 时按 quest 风格生成旁白 welcome/hook/outro
                        （original_cutout 主持人开场/结尾用），但仍生成中文对话。
        include_zh: 生成对话中文音频 zh_{i}.mp3（Ch3 中文跟读专用）。
                    ch3_zh_repeats=0 时时间轴无 listen_zh 段，传 False 跳过以省时。
        tts_rate: Legacy global override — applies to ALL content types when set.
                  Per-type args take precedence.
        tts_rate_en / tts_rate_zh / tts_rate_narration: Per-type rate override
                  (e.g. '-15%'). If None, uses mode default (TTS_RATE_DEFAULTS).
        structure: Mode id for rate defaults ('quest'/'original_cutout'/...).
                   Derived from quest/host_narration flags when omitted.
        tts_engine: 'kokoro' (default, local Kokoro TTS) or 'qwen'
                    (Qwen3-TTS local GPU).
        stop_check: Optional callable returning True to abort early.
    """
    # --- 语速解析：分项覆盖 > 旧全局覆盖 > 模式默认（唯一来源 TTS_RATE_DEFAULTS）---
    _structure = structure or ("quest" if quest
                               else ("original_cutout" if host_narration else "original"))
    rates = {
        "dialogue": resolve_tts_rate("dialogue", _structure, tts_rate_en, tts_rate),
        "zh": resolve_tts_rate("zh", _structure, tts_rate_zh, tts_rate),
        "narration": resolve_tts_rate("narration", _structure, tts_rate_narration, tts_rate),
    }
    # --- Engine + voice map setup ---
    zh_ranks: dict = {}  # 同性别冲突时中文默认音色错开用（qwen/moss）
    if tts_engine == "qwen":
        from qwen_tts_engine import (QwenTTSEngine, build_qwen_voice_map,
                                     pick_zh_preset_fallback, gender_default_ranks)

        model_path = os.environ.get("QWEN_MODEL_PATH", r"H:\models\Qwen3-TTS-12Hz-0.6B-CustomVoice")
        base_model_path = os.environ.get("QWEN_BASE_MODEL_PATH", r"H:\models\Qwen3-TTS-12Hz-1.7B-Base")
        voicedesign_model_path = os.environ.get(
            "QWEN_VOICEDSIGN_MODEL_PATH", r"H:\models\Qwen3-TTS-12Hz-1.7B-VoiceDesign")
        device = os.environ.get("QWEN_DEVICE", "cuda:0")
        tts = QwenTTSEngine(model_path, device, base_model_path, voicedesign_model_path)
        voice_map = build_qwen_voice_map(script, _structure)
        zh_ranks = gender_default_ranks(script)
        narration_voice = voice_map.get("narration", voice_map.get("host", "Serena"))
        voice_label = "qwen"
    elif tts_engine == "moss":
        from moss_tts_engine import (MossTTSEngine, build_moss_voice_map,
                                     get_moss_zh_default, gender_default_ranks)

        model_path = os.environ.get("MOSS_MODEL_PATH") or r"H:\models\MOSS-TTS-Nano-Model"
        tokenizer_path = os.environ.get("MOSS_TOKENIZER_PATH") or r"H:\models\MOSS-Audio-Tokenizer-Nano"
        device = os.environ.get("MOSS_DEVICE") or "cpu"
        repo_dir = os.environ.get("MOSS_REPO_DIR") or r"H:\models\MOSS-TTS-Nano"
        tts = MossTTSEngine(model_path, device, tokenizer_path, repo_dir)
        voice_map = build_moss_voice_map(script, _structure)
        zh_ranks = gender_default_ranks(script)
        narration_voice = voice_map.get("narration", voice_map.get("host", "Bella"))
        voice_label = "moss"
    else:
        tts = TTSEngine()
        voice_map = build_voice_map(script, _structure)
        narration_voice = voice_map.get("narration", voice_map.get("host", "af_sky"))
        voice_label = "kokoro"

    # Narration voice: check script for custom binding (from character_voices UI)
    # kokoro 引擎读 narration_kokoro_voice（Kokoro 音色名），
    # 其余引擎读 narration_qwen_speaker（qwen 音色名跨引擎通用会导致解析失败）
    if tts_engine == "kokoro":
        _narration_voice = script.get("narration_kokoro_voice", "").strip()
    else:
        _narration_voice = script.get("narration_qwen_speaker", "").strip()
    if _narration_voice:
        narration_voice = _narration_voice
    elif (script.get("host_character") or "").strip():
        # 形象绑定角色出镜时，旁白音色联动该角色（显式旁白绑定仍最高优先）
        narration_voice = voice_map.get(script["host_character"].strip(), narration_voice)
    narration_zh_voice = script.get("narration_zh_voice", "").strip() or narration_voice

    # --- 跨引擎回退：qwen/moss 单句失败 → 降级 Kokoro 合成该句，整线不报废 ---
    # kokoro 为本地最终兜底，自身失败直接抛出。
    _fb_state = {"engine": None, "voice_map": None}

    def _kokoro():
        if _fb_state["engine"] is None:
            print("  [TTS][Fallback] Initializing Kokoro fallback engine...")
            _fb_state["engine"] = TTSEngine()
        if _fb_state["voice_map"] is None:
            _fb_state["voice_map"] = build_voice_map(script, _structure)
        return _fb_state["engine"], _fb_state["voice_map"]

    def _synth_line(method, text, voice, path, rate, speaker="char_a"):
        """单句合成 + 引擎级回退。primary=qwen/moss 失败时用 Kokoro 重试一次。"""
        try:
            return getattr(tts, method)(text, voice, path, rate=rate)
        except Exception as e:
            if tts_engine == "kokoro":
                raise
            print(f"  [TTS][Fallback] {Path(path).name} {type(e).__name__}: {e}")
            eng, kmap = _kokoro()
            if method == "synth_chinese":
                fb_voice = get_zh_voice(speaker, script)
            else:
                fb_voice = kmap.get(speaker) or kmap.get("host", "af_sky")
            print(f"  [TTS][Fallback] Retrying with Kokoro voice '{fb_voice}'")
            return getattr(eng, method)(text, fb_voice, path, rate=rate)

    # 检测 tts_rate / 引擎参数变化，必要时清除旧缓存
    extra_sig = ""
    if tts_engine == "moss":
        # MOSS 温度/句间停顿/采样参数影响输出 — 参数变化时自动重新生成
        extra_sig = (f"{os.environ.get('MOSS_TTS_TEMPERATURE', '0.8')}"
                     f"|{os.environ.get('MOSS_TTS_GAP_MS', '120')}"
                     f"|{os.environ.get('MOSS_TTS_TOP_P', '0.95')}"
                     f"|{os.environ.get('MOSS_TTS_TOP_K', '25')}"
                     f"|{os.environ.get('MOSS_TTS_REP_PENALTY', '1.2')}"
                     f"|{os.environ.get('MOSS_TTS_TEXT_TEMPERATURE', '1.0')}"
                     f"|{os.environ.get('MOSS_TTS_GREEDY', '0')}")
    _invalidate_cache_if_rate_changed(audio_dir, rates, quest, tts_engine,
                                      f"{extra_sig}|host:{int(host_narration)}")

    narration = {}
    if quest or host_narration:
        # hook 文本别名：quest 脚本产 hook_intro_en（长钩子）；
        # listening 脚本只有 story_hook（短钩子），混用脚本库时语义自动降级兼容
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
        quest_narration_rate = rates["narration"]
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
                dur = _synth_line("synth_english", text, narration_voice, path,
                                  quest_narration_rate, speaker="host")
                narration[name] = path
                print(f"  [TTS] {name}: {dur:.1f}s")
    else:
        outro_text = script.get("outro", "That's all for today. Keep practicing!")
        practice_intro_text = script.get("practice_intro_en", "Now let's practice. Listen and repeat each sentence.")

        narration_rate = rates["narration"]
        # 注：不再生成 intro.mp3（story_hook 仅作标题卡文字，时间轴从不消费其音频）
        for name, text in [("outro", outro_text), ("practice_intro", practice_intro_text)]:
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
                dur = _synth_line("synth_english", text, narration_voice, path,
                                  narration_rate, speaker="host")
                narration[name] = path
                print(f"  [TTS] {name}: {dur:.1f}s")

    # Dialogue English (character voices; per-type override > legacy global > mode default)
    dialogue_rate = rates["dialogue"]
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
        dur = _synth_line("synth_english", text, voice, path, dialogue_rate, speaker)
        normal_paths.append(path)
        dialogue_durations.append(dur)
        print(f"  [TTS] dialogue_{i}: {dur:.1f}s ({voice_label})")

    # Dialogue Chinese (quest skips entirely; ch3_zh_repeats=0 时无 listen_zh 段，同样跳过)
    zh_paths = []
    if not quest and include_zh:
        zh_rate = rates["zh"]
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
            # Qwen 引擎 → 优先 zh_voice 绑定，回退 voice_map；预设为外语时校正为中文性别预设
            # Kokoro 引擎 → get_zh_voice（优先 zh_voice 绑定，回退性别 edge-tts 默认）
            # MOSS 引擎 → 优先 moss_voice 绑定，否则按性别用中文预设（英文参考音说中文带口音）
            zh_bind = script.get(f"{speaker}_zh_voice", "").strip()
            if tts_engine == "qwen":
                voice = zh_bind or pick_zh_preset_fallback(
                    voice_map.get(speaker, voice_map.get("char_a", "Vivian")),
                    script.get(f"{speaker}_gender", ""),
                    rank=zh_ranks.get(speaker, 0), structure=_structure)
            elif tts_engine == "moss":
                moss_bind = script.get(f"{speaker}_moss_voice", "").strip()
                if moss_bind:
                    voice = moss_bind
                else:
                    gender = script.get(f"{speaker}_gender", "").lower()
                    voice = get_moss_zh_default(gender, rank=zh_ranks.get(speaker, 0),
                                                structure=_structure)
            else:
                voice = zh_bind or get_zh_voice(speaker, script)
            path = str(audio_dir / f"zh_{i}.mp3")
            if _audio_exists(path):
                dur = get_duration(path)
                zh_paths.append(path)
                print(f"  [TTS] zh_{i}: {dur:.1f}s (cached)")
                continue
            dur = _synth_line("synth_chinese", text, voice, path, zh_rate, speaker)
            zh_paths.append(path)
            print(f"  [TTS] zh_{i}: {dur:.1f}s")

    results["narration"] = narration
    results["normal_paths"] = normal_paths
    results["dialogue_durations"] = dialogue_durations
    results["zh_paths"] = zh_paths
    # vocab/slow/quiz 曾属旧版结构；timeline_enrich/media_utils 仍预留对应段类型，
    # 此处置空占位以保持 tts_results 形状稳定
    results["vocab_paths"] = []
    results["slow_paths"] = []
    results["slow_durations"] = []
    results["quiz_paths"] = []
    print("  [TTS] All TTS generation complete.")
