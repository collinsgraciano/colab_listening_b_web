"""Quest timeline builder + SRT generator for task-hook slow listening videos.

Structure:
  Ch1: Welcome         (host on-screen, welcome line + subtitle)
  Ch2: Hook / Intro    (host on-screen, narrator intro + subtitle)
  Ch3: Slow Dialogue   (4 phases: buildup -> core -> reveal -> review, subtitles)
  Ch4: Outro & CTA     (host on-screen, narrator outro + subtitle)

No title card — welcome/hook/outro use subtitle display like dialogue lines.
"""
import sys
from pathlib import Path

_PARENT = str(Path(__file__).parent.parent.resolve())
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from media_utils import build_srt


def build_quest_timeline(script: dict, dialogue_durations: list[float],
                         pad: float = 0.4) -> list[dict]:
    """Build a timeline for the quest (task-hook slow listening) lesson.

    Args:
        script: LLM-generated quest script (dialogue lines carry "phase").
        dialogue_durations: per-line TTS durations (slow speed).
        pad: silence pad between dialogue lines (long by default — thinking time).

    Returns:
        List of timeline segment dicts: title_card, hook_intro, dialogue×N, outro.
    """
    dialogue = script.get("dialogue", [])
    outro = script.get("outro", "That's all for today. Keep practicing!")
    outro_zh = script.get("outro_zh", "")

    timeline = []

    def _add(seg_type, dur, sub_en="", sub_zh="", audio_idx=0, image_idx=0, d_idx=-1,
             speaker=""):
        timeline.append({
            "type": seg_type,
            "duration": dur,
            "subtitle_en": sub_en,
            "subtitle_zh": sub_zh,
            "speaker": speaker,
            "audio_index": audio_idx,
            "image_idx": image_idx,
            "dialogue_idx": d_idx,
        })

    # 1. Welcome (host on-screen, ~4s)
    welcome_en = script.get("welcome_en", "")
    if welcome_en:
        _add("welcome", 4.0, welcome_en, script.get("welcome_zh", ""),
             audio_idx=-1, image_idx=0, speaker="host")

    # 2. Hook / intro (host on-screen, narrator)
    hook_en = script.get("hook_intro_en", "")
    if hook_en:
        _add("hook_intro", 10.0, hook_en, script.get("hook_intro_zh", ""),
             audio_idx=-1, image_idx=0, speaker="host")

    # 3. Slow dialogue — all lines in order (phases stay contiguous)
    for i, line in enumerate(dialogue):
        dur = dialogue_durations[i] if i < len(dialogue_durations) else 3.0
        _add("dialogue", dur, line.get("text", ""), line.get("zh", ""),
             audio_idx=i, image_idx=i + 1,
             speaker=line.get("speaker", ""))
        seg = timeline[-1]
        seg["phase"] = line.get("phase", "buildup")

    # 4. Outro & CTA (host on-screen, narrator)
    if outro:
        outro_dur = min(max(len(outro) * 0.08, 6.0), 10.0)
        _add("outro", outro_dur, outro, outro_zh, audio_idx=0, image_idx=0,
             speaker="host")

    return timeline


def build_srt_from_timeline_quest(timeline: list[dict], gap: float = 0.0) -> str:
    """Build SRT from quest timeline. Timestamps match video exactly.

    All segments (welcome, hook_intro, dialogue, outro) get subtitles —
    they all display as sentence-by-sentence subtitles like dialogue lines.
    """
    return build_srt(timeline, skip_types=set(), gap=gap)
