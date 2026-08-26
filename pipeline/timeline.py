"""Standalone timeline builder + SRT generator for listening practice videos.

No external dependencies. Extracted from listening video subtitle module.
"""
import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from media_utils import build_srt


def _format_srt_time(seconds: float) -> str:
    """Convert seconds to SRT timestamp: HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def build_listening_timeline(script: dict, dialogue_durations: list[float],
                             practice_duration: float = 3.0,
                             pad: float = 0.4) -> list[dict]:
    """Build a timeline for the listening-practice lesson type.

    Structure:
      1. Title card (5s)
      2. Full dialogue (all lines, normal speed)
      3. Practice intro (4s)
      4. Per line: 1EN -> sil -> 1EN -> sil -> 1EN -> sil -> 1ZH -> 1EN -> sil (9 segments)
      5. Outro (narration)
    """
    dialogue = script.get("dialogue", [])
    n = len(dialogue)
    intro = script.get("story_hook", script.get("intro", ""))
    intro_zh = script.get("intro_zh", "")
    outro = script.get("outro", "That's all for today. Keep practicing!")
    outro_zh = script.get("outro_zh", "")

    timeline = []

    def _add(seg_type, dur, sub_en, sub_zh, audio_idx=0, image_idx=0, d_idx=-1):
        timeline.append({
            "type": seg_type,
            "duration": dur,
            "subtitle_en": sub_en,
            "subtitle_zh": sub_zh,
            "speaker": "",
            "audio_index": audio_idx,
            "image_idx": image_idx,
            "dialogue_idx": d_idx,
        })

    # 1. Title card
    title_en = script.get("title", "")
    title_zh = script.get("title_zh", intro_zh)
    scene_zh = script.get("scene_zh", "")
    if title_en:
        _add("title_card", 5.0, title_en, title_zh, audio_idx=0, image_idx=0)
        timeline[-1]["scene_zh"] = scene_zh

    # 2. Full dialogue
    for i, line in enumerate(dialogue):
        dur = dialogue_durations[i] if i < len(dialogue_durations) else 3.0
        _add("dialogue", dur, line.get("text", ""), line.get("zh", ""),
             audio_idx=i, image_idx=i + 1)

    # 3. Practice intro
    practice_intro_en = script.get("practice_intro_en", "Now let's practice. Listen and repeat each sentence.")
    practice_intro_zh = script.get("practice_intro_zh", "現在來練習。請跟著朗讀每一句。")
    _add("practice_intro", 4.0, practice_intro_en, practice_intro_zh, audio_idx=-1, image_idx=0)

    # 4. Per-line practice (9 segments per line)
    for i in range(n):
        line = dialogue[i]
        en_text = line.get("text", "")
        zh_text = line.get("zh", "")
        dur = dialogue_durations[i] if i < len(dialogue_durations) else 3.0

        _add("listen_en", dur, en_text, "", audio_idx=i, image_idx=-1, d_idx=i)
        _add("practice", practice_duration, "", "", audio_idx=-1, image_idx=-1, d_idx=i)
        _add("listen_en", dur, en_text, "", audio_idx=i, image_idx=-1, d_idx=i)
        _add("practice", practice_duration, "", "", audio_idx=-1, image_idx=-1, d_idx=i)
        _add("listen_en", dur, en_text, "", audio_idx=i, image_idx=-1, d_idx=i)
        _add("practice", practice_duration, "", "", audio_idx=-1, image_idx=-1, d_idx=i)
        _add("listen_zh", dur, en_text, zh_text, audio_idx=i, image_idx=-1, d_idx=i)
        _add("listen_en", dur, en_text, "", audio_idx=i, image_idx=-1, d_idx=i)
        _add("practice", practice_duration, "", "", audio_idx=-1, image_idx=-1, d_idx=i)

    # 5. Outro
    if outro:
        outro_dur = min(max(len(outro) * 0.08, 3.0), 6.0)
        _add("outro", outro_dur, outro, outro_zh, audio_idx=0, image_idx=0)

    return timeline


def rewrite_title_card_as_host_segments(timeline: list[dict], script: dict) -> None:
    """original_cutout 专用：移除 title_card，改为 quest 式主持人两段开场。

    开场/结尾均为主持人出镜定格动画；welcome/hook_intro 的实际时长由
    enrich_timeline 按 narration 音频自动补全，这里只放占位时长。
    就地修改 timeline。
    """
    def _make(seg_type, dur, sub_en, sub_zh):
        return {
            "type": seg_type,
            "duration": dur,
            "subtitle_en": sub_en,
            "subtitle_zh": sub_zh,
            "speaker": "host",
            "audio_index": -1,
            "image_idx": 0,
            "dialogue_idx": -1,
        }

    host_segs = []
    welcome_en = (script.get("welcome_en") or "").strip()
    if welcome_en:
        host_segs.append(_make("welcome", 4.0, welcome_en, script.get("welcome_zh", "")))
    hook_en = (script.get("story_hook") or "").strip()
    if hook_en:
        host_segs.append(_make("hook_intro", 10.0, hook_en, script.get("intro_zh", "")))

    # 定位并移除 title_card；主持人段插在原位置（无标题卡则插到最前）
    pos = 0
    for i, seg in enumerate(timeline):
        if seg.get("type") == "title_card":
            pos = i
            del timeline[i]
            break
    timeline[pos:pos] = host_segs


def build_srt_from_timeline(timeline: list[dict], gap: float = 0.0) -> str:
    """Build SRT from a timeline list. Timestamps match video exactly.

    Ch3 segments (listen_en, listen_zh, practice, title_card, practice_intro, outro)
    are SKIPPED — text is on static images, not subtitles.
    """
    return build_srt(timeline, gap=gap)
