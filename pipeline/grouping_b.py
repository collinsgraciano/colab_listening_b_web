"""Dialogue grouping for 方案 B — merge consecutive dialogue lines regardless
of speaker into a single video clip (≤ clip_duration seconds each).

Unlike Plan A (which only merges same-speaker consecutive lines), Plan B merges
ALL consecutive lines whose combined TTS audio fits within the clip_duration,
even if speakers alternate (A→B→A→B...).

Multi-character reference: the group's video clip is generated with BOTH
character reference images passed as image_urls (comma-separated), so Seedance2
knows what both characters look like.

No external dependencies. Pure Python.
"""
import sys
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass  # stdout may be redirected/captured (web/pytest) — reconfigure unavailable


def build_dialogue_groups(dialogue: list[dict], dialogue_durations: list[float],
                          clip_duration: float = 15.0) -> list[dict]:
    """Group consecutive dialogue lines by combined audio <= clip_duration.

    Unlike Plan A, speaker identity is NOT a grouping criterion — consecutive
    lines are merged as long as total audio fits within clip_duration.

    Args:
        dialogue: list of dialogue line dicts (each has 'speaker').
        dialogue_durations: per-line TTS audio durations in seconds.
        clip_duration: max combined audio duration per group (Seedance2 limit).

    Returns:
        list of dicts: [{"lines": [0,1,2], "total_audio": 8.5, "speakers": ["char_a","char_b","char_a"]}, ...]
    """
    groups = []
    cur_lines: list[int] = []
    cur_total = 0.0
    cur_speakers: list[str] = []

    for i, line in enumerate(dialogue):
        speaker = line.get("speaker", "char_a")
        dur = dialogue_durations[i] if i < len(dialogue_durations) else 3.0

        if cur_total + dur <= clip_duration:
            cur_lines.append(i)
            cur_total += dur
            cur_speakers.append(speaker)
        else:
            if cur_lines:
                groups.append({
                    "lines": list(cur_lines),
                    "total_audio": cur_total,
                    "speakers": list(cur_speakers),
                })
            cur_lines = [i]
            cur_total = dur
            cur_speakers = [speaker]

    if cur_lines:
        groups.append({
            "lines": list(cur_lines),
            "total_audio": cur_total,
            "speakers": list(cur_speakers),
        })

    return groups


def merge_group_prompt(group: dict, dialogue: list[dict]) -> str:
    """Merge all video_prompts from a group's lines into one continuous prompt.

    Args:
        group: a group dict from build_dialogue_groups (has 'lines').
        dialogue: full dialogue list.

    Returns:
        Combined single prompt describing the whole multi-line conversation.
    """
    parts = []
    for i in group["lines"]:
        line = dialogue[i]
        p = line.get("video_prompt", "") or line.get("image_prompt", "")
        if p:
            parts.append(p)
    return " ".join(parts)


if __name__ == "__main__":
    # Quick self-test: alternating speakers
    dlg = [
        {"speaker": "char_a"}, {"speaker": "char_b"}, {"speaker": "char_a"},
        {"speaker": "char_b"}, {"speaker": "char_a"}, {"speaker": "char_b"},
    ]
    durs = [3.1, 2.8, 2.5, 2.2, 3.0, 3.4]
    groups = build_dialogue_groups(dlg, durs, 15.0)
    print("Plan B groups (alternating speakers):")
    for g in groups:
        print(f"  lines={g['lines']} speakers={g['speakers']} total={g['total_audio']:.1f}s")
    print(f"Total groups: {len(groups)} (Plan A would give 6, Plan B gives fewer)")
