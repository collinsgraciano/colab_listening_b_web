"""Unified structure dispatch: maps --structure to the correct
LLM client, timeline builder, SRT builder, and compose function.

This replaces scattered if/elif chains in pipeline.py with a single lookup.
Each structure variant registers its four domain functions here.
"""
import sys
from pathlib import Path

_PARENT = str(Path(__file__).parent.resolve())
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

# --- Original ---
from llm_client import generate_listening_script, generate_shorts_script
from timeline import (build_listening_timeline, build_srt_from_timeline,
                      build_shorts_timeline)
from video_compose import compose_listening, compose_image

# --- Quest (lazy import) ---
def _quest_script(topic, cefr, lessons_dir, num_lines):
    from quest.llm_client_quest import generate_quest_script
    return generate_quest_script(topic, cefr, lessons_dir=lessons_dir, num_lines=num_lines)

def _quest_timeline(script, dialogue_durations, pad):
    from quest.timeline_quest import build_quest_timeline
    return build_quest_timeline(script, dialogue_durations, pad=pad)

def _quest_srt(timeline, gap=0.0):
    from quest.timeline_quest import build_srt_from_timeline_quest
    return build_srt_from_timeline_quest(timeline, gap=gap)

def _quest_compose(**kwargs):
    from quest.video_compose_quest import compose_quest
    return compose_quest(**kwargs)


def _quest_v2_compose(**kwargs):
    from quest_v2.video_compose_quest_v2 import compose_quest_v2
    return compose_quest_v2(**kwargs)


STRUCTURES = {
    "original": {
        "generate_script": generate_listening_script,
        "build_timeline": lambda script, ddur, pad, pd: build_listening_timeline(
            script, ddur, pad=pad, practice_duration=pd),
        "build_srt": lambda tl, gap=0.0: build_srt_from_timeline(tl, gap=gap),
        "compose": compose_listening,
        "needs_video_clips": True,
        "needs_zh_tts": True,
        "needs_dialogue_images": False,
    },
    "image": {
        "generate_script": generate_listening_script,
        "build_timeline": lambda script, ddur, pad, pd: build_listening_timeline(
            script, ddur, pad=pad, practice_duration=pd),
        "build_srt": lambda tl, gap=0.0: build_srt_from_timeline(tl, gap=gap),
        "compose": compose_image,
        "needs_video_clips": False,
        "needs_zh_tts": True,
        "needs_dialogue_images": True,
    },
    "quest": {
        "generate_script": _quest_script,
        "build_timeline": lambda script, ddur, pad, pd: _quest_timeline(
            script, ddur, pad=pad),
        "build_srt": _quest_srt,
        "compose": _quest_compose,
        "needs_video_clips": False,
        "needs_zh_tts": False,
        "needs_dialogue_images": True,
    },
    # quest_v2 = quest 内容管线（脚本/时间轴/SRT 完全复用）+ 唇形同步渲染
    "quest_v2": {
        "generate_script": _quest_script,
        "build_timeline": lambda script, ddur, pad, pd: _quest_timeline(
            script, ddur, pad=pad),
        "build_srt": _quest_srt,
        "compose": _quest_v2_compose,
        "needs_video_clips": False,
        "needs_zh_tts": False,
        "needs_dialogue_images": True,
    },
    "shorts": {
        "generate_script": generate_shorts_script,
        "build_timeline": lambda script, ddur, pad, pd: build_shorts_timeline(
            script, ddur, pad=pad),
        "build_srt": lambda tl, gap=0.0: build_srt_from_timeline(tl, gap=gap),
        "compose": compose_image,
        "needs_video_clips": False,
        "needs_zh_tts": False,
        "needs_dialogue_images": True,
    },
}


def get_structure(name: str) -> dict:
    """Return the structure dispatch dict for the given structure name."""
    return STRUCTURES[name]


def list_structures() -> list[str]:
    """Return all available structure names."""
    return list(STRUCTURES.keys())
