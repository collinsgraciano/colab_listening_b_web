"""sleep 模式时间轴：intro → N×[5 朗读段 + 4 静音段] → outro，gap 恒 0.0。

段 dict 与 listening 时间轴同构（type/duration/subtitle_en/subtitle_zh +
sleep 专属键 pair/step），供 save_youtube_metadata 章节与 compose 块构建消费。
pair 段 duration = 音频实际时长（pad=0，气口全部由显式 gap 段承担），
保证时间轴累计时长 == 合成视频时长。
"""
from media_utils import build_srt

# 每组内的步骤顺序与配对静音时长键（与 audio_sleep 文件后缀一致）
PAIR_STEPS = ("a_m", "a_slow", "b_m", "b_slow", "combo")
SLEEP_SEG_TYPES = ("intro", "pair", "gap", "outro")


def build_sleep_timeline(script: dict, audio: dict, num_pairs: int,
                         gap_short: float = 1.0, gap_long: float = 2.0,
                         pair_gap: float = 3.0) -> list[dict]:
    """由 prepare_sleep_audio 结果构建线性时间轴。

    每组段序（gap 与参考视频一致）：
    a_m → g_short → a_slow → g_long → b_m → g_short → b_slow → g_long
    → combo → pair_gap。
    """
    timeline: list[dict] = []
    intro_dur = float(audio.get("intro_dur", 0.0))
    timeline.append({"type": "intro", "duration": round(intro_dur, 3),
                     "subtitle_en": "", "subtitle_zh": "", "pair": 0, "step": ""})

    pair_durs = audio.get("pair_durs", {})
    dialogue = script.get("dialogue", [])
    rows_a, rows_b = dialogue[0::2], dialogue[1::2]
    total = min(num_pairs, len(pair_durs), len(rows_a), len(rows_b))
    for i in range(1, total + 1):
        key = str(i).zfill(4)
        durs = pair_durs[key]
        text_a = rows_a[i - 1].get("text", "")
        text_b = rows_b[i - 1].get("text", "")
        zh_a = rows_a[i - 1].get("zh", "")
        zh_b = rows_b[i - 1].get("zh", "")
        step_texts = {"a_m": (text_a, zh_a), "a_slow": (text_a, zh_a),
                      "b_m": (text_b, zh_b), "b_slow": (text_b, zh_b),
                      "combo": (f"{text_a} {text_b}", f"{zh_a} {zh_b}")}
        for si, step in enumerate(PAIR_STEPS):
            timeline.append({
                "type": "pair", "step": step, "pair": i,
                "duration": round(float(durs[step]), 3),
                "subtitle_en": step_texts[step][0],
                "subtitle_zh": step_texts[step][1],
            })
            is_last = (si == len(PAIR_STEPS) - 1)
            gap = pair_gap if is_last else (gap_long if step in ("a_slow", "b_slow") else gap_short)
            timeline.append({"type": "gap", "step": "", "pair": i,
                             "duration": round(float(gap), 3),
                             "subtitle_en": "", "subtitle_zh": ""})

    outro_dur = float(audio.get("outro_dur", 0.0))
    timeline.append({"type": "outro", "duration": round(outro_dur, 3),
                     "subtitle_en": "", "subtitle_zh": "", "pair": 0, "step": ""})
    return timeline


def timeline_total(timeline: list[dict]) -> float:
    return round(sum(seg.get("duration", 0.0) for seg in timeline), 3)


def build_sleep_srt(timeline: list[dict]) -> str:
    """sidecar 字幕（不上屏烧录）：pair 段的 EN/ZH 进 SRT，其余跳过。"""
    return build_srt(timeline, skip_types={"intro", "gap", "outro"}, gap=0.0)
