"""Batch TTS generation for all dialogue/narration audio."""
import os

from tts_engine import TTSEngine, build_voice_map, get_zh_voice


def generate_tts(script, dialogue, audio_dir, results, quest=False, tts_rate=None,
                 tts_engine="kokoro"):
    """Generate all TTS audio. Runs in a thread.

    Produces narration and dialogue EN/ZH audio.
    Writes results into the *results* dict (shared with caller).

    Args:
        tts_rate: Override dialogue English TTS rate (e.g. '-15%', '0%').
                  If None, uses mode default (quest: '0%', non-quest: '-15%').
        tts_engine: 'kokoro' (default, local Kokoro TTS) or 'voxcpm'
                    (VoxCPM via Cloudflare Worker, LLM-designed voices).
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
                dur = tts.synth_english(text, narration_voice, path, rate=rate)
                narration[name] = path
                print(f"  [TTS] {name}: {dur:.1f}s")
    else:
        intro_text = script.get("story_hook", "")
        outro_text = script.get("outro", "That's all for today. Keep practicing!")
        practice_intro_text = script.get("practice_intro_en", "Now let's practice. Listen and repeat each sentence.")

        for name, text in [("intro", intro_text), ("outro", outro_text), ("practice_intro", practice_intro_text)]:
            if text:
                path = str(audio_dir / f"{name}.mp3")
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
        dur = tts.synth_english(text, voice, path, rate=dialogue_rate)
        normal_paths.append(path)
        dialogue_durations.append(dur)
        print(f"  [TTS] dialogue_{i}: {dur:.1f}s ({voice_label})")

    # Dialogue Chinese (quest skips entirely)
    zh_paths = []
    if not quest:
        for i, line in enumerate(dialogue):
            text = line.get("zh", "")
            if not text:
                zh_paths.append("")
                continue
            speaker = line.get("speaker", "char_a")
            voice = get_zh_voice(speaker, script)
            path = str(audio_dir / f"zh_{i}.mp3")
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
