"""Group audio concatenation: merge per-line TTS into one file per dialogue group."""
import math
import os
import subprocess

from media_utils import get_duration as _get_audio_duration


def _snap(duration: float, fps: int | None) -> float:
    """Snap duration up to the nearest frame boundary (see timeline_enrich)."""
    if not fps or fps <= 0:
        return duration
    f = 1.0 / fps
    return math.ceil(round(duration / f, 6)) * f


def build_group_info(groups: list[dict], normal_paths: list[str],
                      dialogue_durations: list[float], audio_dir,
                      clip_paths: list[str], pad: float,
                      fps: int | None = None) -> tuple[list[dict], dict]:
    """Concat each group's TTS audio into one file (pad between lines) and build
    the group_info / line_to_group structures used by compose.

    Args:
        fps: 编码输出帧率。传入时每行的 pad 目标时长先做帧量化（与
             enrich_timeline 的 snap_to_frame 完全同公式），保证组音频总长 ==
             该组在 timeline 中各段 duration 之和（帧格精确，消除跨段漂移）。
    """
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

        # 每行目标时长（含尾 pad），帧量化后与 timeline 各段 duration 同源同值；
        # expected_total 即组窗口的期望长度
        targets = []
        for j in range(n_lines):
            line_idx = kept_lines[j]
            line_dur = dialogue_durations[line_idx] if line_idx < len(dialogue_durations) else 3.0
            targets.append(_snap(line_dur + pad, fps))
        expected_total = sum(targets)

        def _af_chain(idx_target: float) -> str:
            # loudnorm 统一组内响度；apad/atrim 把长度钉死到期望帧格值，
            # 消除 mp3 编码引入的毫秒级抖动
            return (f"loudnorm=I=-16:TP=-1.5:LRA=11,"
                    f"aresample=44100,"
                    f"apad=whole_dur={idx_target:.3f},atrim=end={idx_target:.3f}")

        inputs = []
        for la in lines_audio:
            inputs.extend(["-i", la])
        filter_parts = []
        for j in range(n_lines):
            filter_parts.append(f"[{j}:a]{_af_chain(targets[j])}[a{j}]")
        concat_inputs = "".join(f"[a{j}]" for j in range(n_lines))
        filter_parts.append(f"{concat_inputs}concat=n={n_lines}:v=0:a=1[a]")
        subprocess.run(
            ["ffmpeg", "-y"] + inputs + ["-filter_complex", ";".join(filter_parts),
             "-map", "[a]",
             "-c:a", "libmp3lame", "-b:a", "128k", "-ar", "44100", "-ac", "2",
             group_audio_path],
            capture_output=True, timeout=120)

        if os.path.exists(group_audio_path) and os.path.getsize(group_audio_path) > 1000:
            if fps:
                # 帧量化模式下以构造值为准（ffprobe 只作异常兜底）
                group_total_dur = expected_total
                measured = _get_audio_duration(group_audio_path)
                if abs(measured - expected_total) > 0.05:
                    print(f"  [Group {gi}] WARNING: audio len {measured:.3f}s "
                          f"!= expected {expected_total:.3f}s")
            else:
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
