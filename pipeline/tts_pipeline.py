"""Batch TTS generation for all dialogue/narration audio."""
import os

from tts_engine import TTSEngine, build_voice_map, get_zh_voice
from media_utils import get_duration


def _audio_exists(path: str) -> bool:
    """Check if audio file exists and is non-empty (>100 bytes)."""
    return bool(path) and os.path.exists(path) and os.path.getsize(path) > 100


def generate_tts(script, dialogue, audio_dir, results, quest=False, tts_rate=None,
                 tts_engine="kokoro", shorts=False):
    """Generate all TTS audio. Runs in a thread.

    Produces narration and dialogue EN/ZH audio.
    Writes results into the *results* dict (shared with caller).

    Args:
        tts_rate: Override dialogue English TTS rate (e.g. '-15%', '0%').
                  If None, uses mode default (quest: '0%', non-quest: '-15%').
        tts_engine: 'kokoro' (default, local Kokoro TTS) or 'voxcpm'
                    (VoxCPM via Cloudflare Worker, LLM-designed voices).
        shorts: Shorts mode — narration is outro CTA only, no Chinese dialogue.
    """
    # --- Engine + voice map setup ---
    if tts_engine == "voxcpm":
        from voxcpm_tts import VoxCPMEngine
        from llm_client import design_voxcpm_voices

        worker_url = os.environ.get("VOXCPM_WORKER_URL", "")
        api_key = os.environ.get("VOXCPM_API_KEY", "")
        if not worker_url:
            raise RuntimeError(
                "VOXCPM_WORKER_URL not set. Pass --voxcpm-worker-url or set env var."
            )

        # Reuse cached voice descriptions if present (resume support)
        cached_voices = script.get("voxcpm_voices")
        if cached_voices and isinstance(cached_voices, dict) and cached_voices.get("char_a"):
            print("  [TTS] Reusing cached VoxCPM voice descriptions from script.")
            voice_map = cached_voices
        else:
            print("  [TTS] Designing VoxCPM voices via LLM...")
            voice_map = design_voxcpm_voices(script)
            script["voxcpm_voices"] = voice_map  # persist for resume

        tts = VoxCPMEngine(worker_url, api_key)
        if not tts.test_connection():
            raise RuntimeError("VoxCPM Worker health check failed. Check VOXCPM_WORKER_URL.")
        narration_voice = voice_map.get("host", voice_map.get("narrator", "Warm narrator voice, clear, moderate pace."))
        voice_label = "voxcpm"
    elif tts_engine == "qwen":
        from qwen_tts_engine import QwenTTSEngine, build_qwen_voice_map

        model_path = os.environ.get("QWEN_MODEL_PATH", r"H:\models\Qwen3-TTS-12Hz-0.6B-CustomVoice")
        base_model_path = os.environ.get("QWEN_BASE_MODEL_PATH", r"H:\models\Qwen3-TTS-12Hz-1.7B-Base")
        device = os.environ.get("QWEN_DEVICE", "cuda:0")
        tts = QwenTTSEngine(model_path, device, base_model_path)
        voice_map = build_qwen_voice_map(script)
        narration_voice = voice_map.get("host", "Serena")
        voice_label = "qwen"
    else:
        tts = TTSEngine()
        voice_map = build_voice_map(script)
        narration_voice = voice_map.get("host", "af_sky")
        voice_label = "kokoro"

    narration = {}
    if quest:
        welcome_text = script.get("welcome_en", "")
        hook_text = script.get("hook_intro_en", "")
        outro_text = script.get("outro", "That's all for today. Keep practicing!")
        for name, text, rate in [("welcome", welcome_text, "-10%"),
                                 ("hook", hook_text, "-10%"),
                                 ("outro", outro_text, "-10%")]:
            if text:
                path = str(audio_dir / f"{name}.mp3")
                if _audio_exists(path):
                    dur = get_duration(path)
                    narration[name] = path
                    print(f"  [TTS] {name}: {dur:.1f}s (cached)")
                    continue
                dur = tts.synth_english(text, narration_voice, path, rate=rate)
                narration[name] = path
                print(f"  [TTS] {name}: {dur:.1f}s")
    elif shorts:
        # Shorts: only the outro CTA is narrated (title card is silent)
        outro_text = script.get("outro", "Follow for more easy English every day!")
        path = str(audio_dir / "outro.mp3")
        if _audio_exists(path):
            dur = get_duration(path)
            narration["outro"] = path
            print(f"  [TTS] outro: {dur:.1f}s (cached)")
        else:
            dur = tts.synth_english(outro_text, narration_voice, path, rate="+0%")
            narration["outro"] = path
            print(f"  [TTS] outro: {dur:.1f}s")
    else:
        intro_text = script.get("story_hook", "")
        outro_text = script.get("outro", "That's all for today. Keep practicing!")
        practice_intro_text = script.get("practice_intro_en", "Now let's practice. Listen and repeat each sentence.")

        for name, text in [("intro", intro_text), ("outro", outro_text), ("practice_intro", practice_intro_text)]:
            if text:
                path = str(audio_dir / f"{name}.mp3")
                if _audio_exists(path):
                    dur = get_duration(path)
                    narration[name] = path
                    print(f"  [TTS] {name}: {dur:.1f}s (cached)")
                    continue
                dur = tts.synth_english(text, narration_voice, path, rate="+0%")
                narration[name] = path
                print(f"  [TTS] {name}: {dur:.1f}s")

    # Dialogue English (character voices; tts_rate overrides mode default)
    dialogue_rate = tts_rate if tts_rate else ("0%" if quest else "-15%")
    normal_paths = []
    dialogue_durations = []
    for i, line in enumerate(dialogue):
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

    # Dialogue Chinese (quest/shorts skip entirely)
    zh_paths = []
    if not quest and not shorts:
        for i, line in enumerate(dialogue):
            text = line.get("zh", "")
            if not text:
                zh_paths.append("")
                continue
            speaker = line.get("speaker", "char_a")
            if tts_engine == "qwen":
                voice = voice_map.get(speaker, voice_map.get("char_a", "Vivian"))
            else:
                voice = get_zh_voice(speaker, script)
            path = str(audio_dir / f"zh_{i}.mp3")
            if _audio_exists(path):
                dur = get_duration(path)
                zh_paths.append(path)
                print(f"  [TTS] zh_{i}: {dur:.1f}s (cached)")
                continue
            dur = tts.synth_chinese(text, voice, path, rate="-10%")
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
