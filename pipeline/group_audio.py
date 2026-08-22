"""Group audio concatenation: merge per-line TTS into one file per dialogue group."""
import os
import subprocess

from media_utils import get_duration as _get_audio_duration


def build_group_info(groups: list[dict], normal_paths: list[str],
                      dialogue_durations: list[float], audio_dir,
                      clip_paths: list[str], pad: float) -> tuple[list[dict], dict]:
    """Concat each group's TTS audio into one file (pad between lines) and build
    the group_info / line_to_group structures used by compose."""
    group_info = []
    for gi, group in enumerate(groups):
        clip_idx = gi + 1  # scene is index 0
        clip_path = clip_paths[clip_idx] if clip_idx < len(clip_paths) else None

        group_audio_path = str(audio_dir / f"group_audio_{gi}.mp3")
        lines_audio = [normal_paths[li] for li in group["lines"]
                       if li < len(normal_paths) and os.path.exists(normal_paths[li])]
        kept_lines = [li for li in group["lines"]
                      if li < len(normal_paths) and os.path.exists(normal_paths[li])]
        n_lines = len(lines_audio)
        if not lines_audio:
            continue

        if n_lines == 1:
            subprocess.run(
                ["ffmpeg", "-y", "-i", lines_audio[0],
                 "-c:a", "libmp3lame", "-b:a", "128k", "-ar", "44100", "-ac", "2",
                 group_audio_path],
                capture_output=True, timeout=30)
        else:
            inputs = []
            for la in lines_audio:
                inputs.extend(["-i", la])
            filter_parts = []
            for j in range(n_lines):
                line_idx = kept_lines[j]
                line_dur = dialogue_durations[line_idx] if line_idx < len(dialogue_durations) else 3.0
                pad_dur = line_dur + pad
                filter_parts.append(f"[{j}:a]apad=whole_dur={pad_dur:.3f}[a{j}]")
            concat_inputs = "".join(f"[a{j}]" for j in range(n_lines))
            filter_parts.append(f"{concat_inputs}concat=n={n_lines}:v=0:a=1[a]")
            filter_complex = ";".join(filter_parts)
            subprocess.run(
                ["ffmpeg", "-y"] + inputs + ["-filter_complex", filter_complex,
                 "-map", "[a]",
                 "-c:a", "libmp3lame", "-b:a", "128k", "-ar", "44100", "-ac", "2",
                 group_audio_path],
                capture_output=True, timeout=60)

        if os.path.exists(group_audio_path) and os.path.getsize(group_audio_path) > 1000:
            group_total_dur = _get_audio_duration(group_audio_path)
        else:
            group_total_dur = group["total_audio"]
            group_audio_path = normal_paths[group["lines"][0]] if normal_paths else None

        group_info.append({
            "clip_path": clip_path,
            "audio_path": group_audio_path,
            "total_dur": group_total_dur,
            "lines": list(group["lines"]),
        })
        print(f"  [Group {gi}] audio concat: {group_total_dur:.1f}s, lines={group['lines']}")

    line_to_group = {}
    for gi, g in enumerate(groups):
        for li in g["lines"]:
            line_to_group[li] = gi
    return group_info, line_to_group
