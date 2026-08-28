"""参考频道同款样式：沉浸式英文動畫風格的標題+簡介（繁中）。

標題格式（≤95 字符，YouTube 硬限 100）：
【🎬沉浸式英文動畫】{emoji}{主題+英文}｜{內容點1・內容點2・內容點3}｜{吸引句}｜🌱{CEFR} 初級英文｜…

簡介主體由 LLM 在 Step 0 生成（youtube_description_ref：📌 影片簡介 → 你會聽到 →
逐句跟讀 → CEFR → 收尾），本模組負責：注入真實 時間軸 章节（繁中 label）、
拼接链接区块；hashtag 由 save_youtube_metadata 统一追加。
"""
import re

STYLE_ID = "reference_v1"
LABEL = "参考频道同款"
STRUCTURES = ("original", "original_static", "original_cutout")

# ======================= 可编辑配置区 =======================
# 在此填你自己频道的链接（留空则同款簡介不含链接区块）。例：
#   PLAYLISTS = [("沉浸式英文動畫", "https://www.youtube.com/playlist?list=PLxxxx")]
#   RELATED_VIDEOS = [("飯店住宿全流程", "https://youtu.be/xxxx")]
PLAYLISTS: list[tuple[str, str]] = []
RELATED_VIDEOS: list[tuple[str, str]] = []
# ============================================================

# 章节 → 繁中标签（同款風格為繁中）。
# save_youtube_metadata 的 marks 元素是英文 label（seg_labels 的值），故以英文 label 为键；
# 同时兜底 seg_type 键，防未来调用方直接传 seg_type。
_SEG_ZH_LABELS = {
    "Welcome & Hook": "主持人開場",
    "Immersive Dialogue": "沉浸式情境對話",
    "Shadowing Practice": "逐句跟讀練習",
    "Outro": "重點整理",
    "Welcome": "主持人開場",
    "Intro · Listening Task": "聽力任務說明",
    "Slow Dialogue": "慢速對話",
    "Outro · Answer & CTA": "重點整理",
    "welcome": "主持人開場",
    "hook_intro": "聽力任務說明",
    "dialogue": "沉浸式情境對話",
    "practice_intro": "逐句跟讀練習",
    "outro": "重點整理",
}

# LLM 简介里自带的时间轴段（占位时间戳）→ 整段替换为真实 時間軸
_TIMELINE_SECTION = re.compile(
    r"-{4,}\s*\n\s*時間軸[::][^\n]*\n"
    r"(?:[ \t]*[-•*]?[ \t]*(?:[\dx]{1,2}:)?[\dx]{1,2}:[\dx]{2}[^\n]*\n?)+"
)


def _fmt_ts(seconds: float) -> str:
    return f"{int(seconds // 60)}:{int(seconds % 60):02d}"


def _build_timeline_lines(marks: list[tuple[float, str]]) -> list[str]:
    """同款 時間軸 行列表（YouTube 硬规则：首章 00:00、相邻 >=10s，违规则 YouTube 全禁章节）。"""
    lines = ["時間軸:"]
    kept_t: float | None = None
    for i, (t, seg) in enumerate(marks):
        if i == 0:
            t = 0.0
        elif kept_t is not None and t - kept_t < 10.0:
            continue
        kept_t = t
        lines.append(f"{_fmt_ts(t)} {_SEG_ZH_LABELS.get(seg, seg)}")
    return lines


def build(script: dict, ctx: dict) -> dict | None:
    title = str(script.get("youtube_title_ref", "")).strip()
    desc = str(script.get("youtube_description_ref", "")).strip()
    if not title or not desc:
        return None  # 脚本缺同款字段（旧脚本）→ 跳过该样式

    timeline_block = "------------\n" + "\n".join(
        _build_timeline_lines(ctx.get("marks", []))) + "\n------------"
    if _TIMELINE_SECTION.search(desc):
        desc = _TIMELINE_SECTION.sub(timeline_block + "\n", desc, count=1)
    else:
        desc = desc.rstrip() + "\n\n" + timeline_block

    links: list[str] = []
    if PLAYLISTS:
        links.append("✨下面也有多個用心的影片也可以點進去學習唷!")
        for name, url in PLAYLISTS:
            links.append(f"{name}\n{url}")
    if RELATED_VIDEOS:
        for name, url in RELATED_VIDEOS:
            links.append(f"{name} {url}")
    if links:
        desc = desc.rstrip() + "\n\n" + "\n".join(links)

    return {"title": title, "description": desc}
