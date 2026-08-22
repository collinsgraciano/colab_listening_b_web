"""Timeline enrichment: fill audio_dur and duration for all segment types."""
import os


def enrich_timeline(timeline: list[dict], tts, pad: float,
                     dialogue_durations: list[float], zh_paths: list[str],
                     narration: dict,
                     vocab_durations: list[float] | None = None,
                     quiz_durations: list[float] | None = None,
                     slow_durations: list[float] | None = None) -> None:
    """Fill seg['audio_dur'] and pad-inclusive seg['duration'] for all segment types.

    Shared by original/static and quest structures.
    Mutates the timeline in place.
    """
    def _safe_index(durs, idx, default):
        return durs[idx] if durs and idx < len(durs) else default

    def _narration_dur(key, fallback):
        path = narration.get(key, "")
        return tts.get_duration(path) if path and os.path.exists(path) else fallback

    for seg in timeline:
        seg_type = seg.get("type", "")
        audio_idx = seg.get("audio_index", 0)

        if seg_type == "vocab":
            ad = _safe_index(vocab_durations, audio_idx, 4.0)
        elif seg_type == "quiz":
            ad = _safe_index(quiz_durations, audio_idx, 6.0)
        elif seg_type == "dialogue_slow":
            ad = _safe_index(slow_durations, audio_idx, 4.0)
        elif seg_type in ("dialogue", "listen_en"):
            ad = _safe_index(dialogue_durations, audio_idx, 3.0)
        elif seg_type == "listen_zh":
            if audio_idx < len(zh_paths) and zh_paths[audio_idx]:
                ad = tts.get_duration(zh_paths[audio_idx])
            else:
                ad = _safe_index(dialogue_durations, audio_idx, 3.0)
        elif seg_type == "practice":
            seg["audio_dur"] = 0
            continue
        elif seg_type == "title_card":
            seg["audio_dur"] = seg["duration"]
            continue
        elif seg_type == "practice_intro":
            ad = _narration_dur("practice_intro", seg["duration"] - pad)
        elif seg_type == "hook_intro":
            ad = _narration_dur("hook", seg["duration"] - pad)
        elif seg_type == "welcome":
            ad = _narration_dur("welcome", seg["duration"] - pad)
        elif seg_type == "outro":
            ad = _narration_dur("outro", seg["duration"] - pad)
        else:
            continue

        seg["audio_dur"] = ad
        seg["duration"] = ad + pad
