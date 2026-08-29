"""Listening 脚本程序化质检门禁 — 纯 Python 检查，无 LLM 依赖。

为三个 Original 类模式（original / original_static / original_cutout）的
listening 脚本提供与 quest 对等的 Phase D 程序化门禁。

七类检查：结构 / 字段 / 自然度 / 重复 / 一致性（性别·场景·风格·引语） / 翻译 / 元数据。
输出结构与 quest 完全一致：{"passed", "n_errors", "n_warnings", "issues", "summary"}。
severity="error" 计入阻断（触发修补轮）；"warning" 仅报告并喂给 LLM 评审。

可独立运行：python quality_gate_listening.py <script.json> [num_lines]
"""
import os
import re
import sys
from pathlib import Path

# 复用 quest 门禁的简体检测（stdlib-only，零风险）；报告格式化在下方本地实现
try:
    from quest.quality_gate import _is_simplified
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent))
    from quest.quality_gate import _is_simplified

# ── 词表 ──────────────────────────────────────────────────────────────

# 核心填充词（自然度密度检查用；与 quest 同表，listening 行数少阈值更低）
FILLERS = [
    "well", "you know", "hmm", "actually", "oh", "um", "i mean",
    "let me think", "you see", "sort of", "kind of", "yeah", "haha",
    "sure", "great", "perfect", "awesome", "no problem",
]

BACK_CHANNELS = [
    "oh really", "that makes sense", "interesting", "wow", "i see",
    "right", "exactly", "nice", "got it", "of course", "sounds good",
]

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "to", "of", "in", "it",
    "that", "this", "and", "or", "for", "with", "you", "your", "they",
    "their", "not", "but", "have", "has", "do", "does", "did", "be",
    "been", "at", "on", "by", "as", "from", "he", "she", "his", "her",
    "we", "us", "our", "i", "me", "my", "if", "so", "what", "why",
    "how", "when", "there", "here", "just", "very", "really",
}

# CEFR → 每行期望词数区间（与 quest 一致；上限按 max_line_words clamp）
_CEFR_WORD_RANGE = {
    "A1": (4, 8), "A2": (6, 10), "B1": (8, 13), "B2": (10, 16),
}

# 每行最大词数默认值（实际值 = param > env LISTENING_MAX_LINE_WORDS > 此默认）
_MAX_LINE_WORDS_DEFAULT = 10

# 旁白字段（逐句检查每句词数 ≤ max_line_words；warning 级）
_NARRATION_FIELDS = ("welcome_en", "story_hook", "outro", "practice_intro_en")

# 性别指示词（一致性检查用）
_FEMALE_WORDS = {"woman", "women", "girl", "lady", "she", "her", "hers", "female"}
_MALE_WORDS = {"man", "men", "boy", "guy", "he", "him", "his", "male"}

# 元数据目标
_META_TARGETS = {
    "youtube_tags": (15, 20),
    "thumbnail_icons": (4, 5),
    "youtube_title_chars": (40, 95),
}


def _words(text: str) -> list[str]:
    return text.split()


def _content_words(text: str) -> list[str]:
    return [w.strip(".,!?;:'\"()").lower() for w in text.split()
            if len(w.strip(".,!?;:'\"()")) > 3
            and w.strip(".,!?;:'\"()").lower() not in STOPWORDS]


def _norm_quote(text: str) -> str:
    """归一化台词用于 title_quote 逐字比对：小写 + 去首尾标点 + 压缩空白。"""
    return " ".join(text.lower().strip(".,!?;:'\"()…“”‘’ «»").split())


def _gender_words(text: str) -> tuple[set[str], set[str]]:
    """返回文本中命中的（女性词, 男性词）集合。"""
    tokens = {w.strip(".,!?;:'\"()").lower() for w in text.split()}
    return (tokens & _FEMALE_WORDS, tokens & _MALE_WORDS)


def _scene_tokens(scene: str) -> list[str]:
    """scene 提取有区分度的词元（≥4 字符实词，如 "coffee shop" → [coffee, shop]）。"""
    return [w for w in _content_words(scene)]


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

# 各结构模式行级必需的视觉 prompt 字段（裁剪后 poses 已彻底废弃）：
# original 只生成 video_prompt（Step3 片段）；original_static 只生成 image_prompt（逐行图）；
# original_cutout 不生成任何 prompt 文本（姿势图全部程序化生成）。
PROMPT_FIELDS = {
    "original": ("video_prompt",),
    "original_static": ("image_prompt",),
    "original_cutout": (),
}


def _resolve_structure(script: dict, structure: str = "") -> str:
    """解析结构模式：显式参数优先，其次 script 内记录，缺省 original（旧脚本最严兼容）。"""
    s = (structure or str(script.get("structure", "")) or "original").strip()
    return s if s in PROMPT_FIELDS else "original"


def _resolve_max_line_words(max_line_words: int | None) -> int:
    """每行最大词数：param → env LISTENING_MAX_LINE_WORDS → 默认 10，clamp [4,20]。

    env 直读 os.environ（本模块 stdlib-only 独立可跑）；Web 批量生成的
    线程局部 override 由调用方（llm_review）解析后经 param 传入。
    """
    if max_line_words is None:
        raw = (os.environ.get("LISTENING_MAX_LINE_WORDS", "") or "").strip()
        try:
            max_line_words = int(raw) if raw else _MAX_LINE_WORDS_DEFAULT
        except ValueError:
            max_line_words = _MAX_LINE_WORDS_DEFAULT
    return max(4, min(20, int(max_line_words)))


def run_listening_quality_gate(script: dict, num_lines: int | None = None,
                               structure: str = "",
                               max_line_words: int | None = None) -> dict:
    """对 listening 脚本跑全部程序化检查，返回报告。"""
    issues: list[dict] = []

    def add(check: str, severity: str, detail: str, lines: list | None = None):
        issues.append({"check": check, "severity": severity,
                       "detail": detail, "lines": lines or []})

    structure = _resolve_structure(script, structure)
    prompt_fields = PROMPT_FIELDS[structure]
    max_line_words = _resolve_max_line_words(max_line_words)
    dialogue = script.get("dialogue", [])
    if num_lines is None:
        num_lines = script.get("_requested_num_lines", 0) or len(dialogue)
    cefr = (script.get("cefr") or "A2").upper()

    # ── 1. 结构 ──────────────────────────────────────────────────────
    if len(dialogue) != num_lines:
        add("structure", "error",
            f"总行数 {len(dialogue)} != 要求 {num_lines}")
    bad_speakers = [i for i, l in enumerate(dialogue)
                    if l.get("speaker", "") not in ("char_a", "char_b")]
    if bad_speakers:
        add("structure", "error",
            f"speaker 非法（只允许 char_a/char_b）: lines {bad_speakers[:10]}",
            bad_speakers[:10])
    consecutive = sum(1 for i in range(1, len(dialogue))
                      if dialogue[i].get("speaker") == dialogue[i - 1].get("speaker"))
    consec_max = max(6, round(num_lines * 0.25)) if num_lines else 6
    if consecutive > consec_max:
        add("structure", "warning",
            f"同角色连续发言 {consecutive} 次（建议 ≤{consec_max}）")

    # ── 2. 字段完整性 ────────────────────────────────────────────────
    for i, line in enumerate(dialogue):
        for f in ("text", "zh", "phonetic", *prompt_fields):
            if not str(line.get(f, "")).strip():
                add("fields", "error", f"line {i} 字段 '{f}' 为空", [i])
        ph = str(line.get("phonetic", "")).strip()
        if ph and not (ph.startswith("/") and ph.endswith("/")):
            add("fields", "warning", f"line {i} phonetic 未用 /slashes/ 包裹", [i])

    # ── 3. 自然度 ────────────────────────────────────────────────────
    texts = [l.get("text", "") for l in dialogue]
    all_lower = " ".join(texts).lower()
    filler_total = sum(all_lower.count(f) for f in FILLERS)
    back_total = sum(all_lower.count(b) for b in BACK_CHANNELS)
    filler_target = max(4, round(num_lines * 0.12)) if num_lines else 4
    if filler_total < filler_target:
        add("naturalness", "warning",
            f"核心填充词 {filler_total} 个（建议 ≥{filler_target}，约每 8 行 1 个）")
    if back_total < 2:
        add("naturalness", "warning",
            f"back-channeling {back_total} 个（建议 ≥2）")

    wc = [len(_words(t)) for t in texts]
    if wc:
        avg = sum(wc) / len(wc)
        lo, hi = _CEFR_WORD_RANGE.get(cefr, (6, 10))
        hi = min(hi, max_line_words)
        if not (lo <= avg <= hi):
            add("naturalness", "warning",
                f"平均每行 {avg:.1f} 词，CEFR {cefr} 建议 {lo}-{hi}")
        long_lines = [i for i, w in enumerate(wc) if w > max_line_words]
        if long_lines:
            add("naturalness", "error",
                f"{len(long_lines)} 行超长（>{max_line_words} 词，字幕将超过两行）",
                long_lines[:10])
    streak = 0
    for i, w in enumerate(wc):
        streak = streak + 1 if w <= 5 else 0
        if streak >= 3:
            add("naturalness", "warning",
                f"连续 3+ 短行（≤5 词）止于 line {i}", [i - 2, i])
            streak = 0

    # 旁白逐句词数检查（warning：旁白不在对话行 patch 修复范围，设 error 会
    # 误触 QA 循环的空转保护提前终止；实际显示由渲染兜底保证 ≤2 行）
    for field in _NARRATION_FIELDS:
        ntext = str(script.get(field, "") or "").strip()
        if not ntext:
            continue
        for si, sent in enumerate(re.split(r"(?<=[.!?])\s+", ntext)):
            if sent.strip() and len(_words(sent)) > max_line_words:
                add("narration", "warning",
                    f"旁白 {field} 第 {si + 1} 句超长（>{max_line_words} 词）")

    # ── 4. 重复 ──────────────────────────────────────────────────────
    seen: dict[str, int] = {}
    for i, t in enumerate(texts):
        key = " ".join(t.lower().split())
        if key in seen:
            add("duplicates", "error",
                f"完全重复行 line {seen[key]} == line {i}: \"{t[:60]}\"",
                [seen[key], i])
        else:
            seen[key] = i

    word_sets = [set(_content_words(t)) for t in texts]
    near_dups = []
    for i in range(len(texts)):
        for j in range(i + 3, len(texts)):
            a, b = word_sets[i], word_sets[j]
            if not a or not b:
                continue
            jac = len(a & b) / len(a | b)
            if jac > 0.8:
                near_dups.append((i, j, round(jac, 2)))
    if near_dups:
        add("duplicates", "warning",
            f"近似重复 {len(near_dups)} 组（词集 Jaccard>0.8）",
            [i for i, _, _ in near_dups[:10]])

    # ── 5. 一致性（listening 特有：性别 / 场景 / 风格 / 引语） ────────
    # 5a. 性别一致性：每个 speaker 的描述 + 其所有行的 prompts/poses 文本
    scene = str(script.get("scene", "")).strip()
    scene_toks = _scene_tokens(scene)
    scene_missing: list[int] = []
    gender_conflict: list[int] = []
    gender_field_mismatch: list[str] = []

    for char_key in ("char_a", "char_b"):
        desc = str(script.get(f"{char_key}_description", ""))
        field_gender = str(script.get(f"{char_key}_gender", "")).strip().lower()
        own_lines = [i for i, l in enumerate(dialogue)
                     if l.get("speaker") == char_key]
        corpus = desc
        for i in own_lines:
            l = dialogue[i]
            corpus += " " + " ".join(str(l.get(f, "")) for f in prompt_fields)
        fw, mw = _gender_words(corpus)
        line_conflicts = []
        for i in own_lines:
            l = dialogue[i]
            ltext = " ".join(str(l.get(f, "")) for f in prompt_fields)
            lf, lm = _gender_words(ltext)
            if lf and lm:
                line_conflicts.append(i)
        if line_conflicts:
            gender_conflict.extend(line_conflicts)
            add("consistency", "error",
                f"{char_key} 的 prompts 内男女指示词混用: lines {line_conflicts[:10]}",
                line_conflicts[:10])
        if field_gender in ("male", "female") and fw and mw:
            pass  # 全语料混用已由逐行检查覆盖
        elif field_gender == "male" and fw and not mw:
            gender_field_mismatch.append(
                f"{char_key}_gender=male 但描述/prompts 全是女性指示词 {sorted(fw)}")
        elif field_gender == "female" and mw and not fw:
            gender_field_mismatch.append(
                f"{char_key}_gender=female 但描述/prompts 全是男性指示词 {sorted(mw)}")

    for msg in gender_field_mismatch:
        add("consistency", "warning", f"性别字段与文本矛盾: {msg}")
    if gender_conflict:
        summary_gender = len(gender_conflict)
    else:
        summary_gender = 0

    # 5b. 场景一致性：逐行 prompt 文本是否提及 scene 词元（无 prompt 字段的结构跳过）
    if scene_toks and prompt_fields:
        for i, l in enumerate(dialogue):
            ptext = " ".join(str(l.get(f, "")) for f in prompt_fields).lower()
            if not any(tok in ptext for tok in scene_toks):
                scene_missing.append(i)
        if scene_missing:
            ratio = len(scene_missing) / max(1, len(dialogue))
            if ratio > 0.3:
                add("consistency", "warning",
                    f"scene '{scene}' 有 {len(scene_missing)}/{len(dialogue)} 行 "
                    f"prompts 未提及（场景漂移风险）", scene_missing[:10])
    # 5c. 风格短语：所有 prompts 应包含激活的视觉风格描述（配置为空则跳过）
    style_prompt = ""
    try:
        from style_manager import get_active_style_prompt
        style_prompt = str(get_active_style_prompt() or "").strip()
    except Exception:
        style_prompt = ""
    if style_prompt and prompt_fields:
        style_missing = []
        for i, l in enumerate(dialogue):
            ptext = " ".join(str(l.get(f, "")) for f in prompt_fields).lower()
            if style_prompt.lower() not in ptext:
                style_missing.append(i)
        if len(style_missing) > max(1, round(len(dialogue) * 0.2)):
            add("consistency", "warning",
                f"{len(style_missing)}/{len(dialogue)} 行 prompts 缺视觉风格短语"
                f"（\"{style_prompt[:50]}…\"）", style_missing[:10])

    # 5d. title_quote 引语必须逐字来自台词
    quote = str(script.get("title_quote", "")).strip()
    if quote:
        norm = _norm_quote(quote)
        if norm and not any(_norm_quote(t) == norm or norm in _norm_quote(t)
                            for t in texts):
            add("consistency", "warning",
                f"title_quote 未逐字命中任何台词: \"{quote[:60]}\"")

    # ── 6. 翻译质量 ──────────────────────────────────────────────────
    simp_lines = []
    for i, l in enumerate(dialogue):
        zh = l.get("zh", "")
        if zh:
            is_simp, ch = _is_simplified(zh)
            if is_simp:
                simp_lines.append(i)
    if simp_lines:
        add("translation", "warning",
            f"{len(simp_lines)} 行 zh 疑似含简体字（可 opencc 程序化修复）",
            simp_lines[:10])

    title_zh = str(script.get("title_zh", "")).strip()
    if len(title_zh) > 6:
        add("translation", "warning",
            f"title_zh \"{title_zh}\" 超过 6 字（缩略图空间有限）")

    # ── 7. 元数据 ────────────────────────────────────────────────────
    if not str(script.get("welcome_en", "")).strip():
        add("metadata", "error", "welcome_en 为空（主持人开场白必需）")
    if not str(script.get("title", "")).strip():
        add("metadata", "warning", "title 为空")
    tags = script.get("youtube_tags", [])
    if tags:
        lo, hi = _META_TARGETS["youtube_tags"]
        if not (lo <= len(tags) <= hi):
            add("metadata", "warning",
                f"youtube_tags {len(tags)} 个（建议 {lo}-{hi}）")
    icons = script.get("thumbnail_icons", [])
    if icons:
        lo, hi = _META_TARGETS["thumbnail_icons"]
        if not (lo <= len(icons) <= hi):
            add("metadata", "warning",
                f"thumbnail_icons {len(icons)} 个（建议 {lo}-{hi}）")
    yt_title = str(script.get("youtube_title", ""))
    if yt_title:
        lo, hi = _META_TARGETS["youtube_title_chars"]
        if not (lo <= len(yt_title) <= hi):
            add("metadata", "warning",
                f"youtube_title {len(yt_title)} 字符（建议 {lo}-{hi}）")
    yt_title_ref = str(script.get("youtube_title_ref", ""))
    if yt_title_ref:
        lo, hi = _META_TARGETS["youtube_title_chars"]
        if not (lo <= len(yt_title_ref) <= hi):
            add("metadata", "warning",
                f"youtube_title_ref {len(yt_title_ref)} 字符（建议 {lo}-{hi}）")

    # ── 汇总 ────────────────────────────────────────────────────────
    n_errors = sum(1 for i in issues if i["severity"] == "error")
    n_warnings = sum(1 for i in issues if i["severity"] == "warning")
    return {
        "passed": n_errors == 0,
        "n_errors": n_errors,
        "n_warnings": n_warnings,
        "issues": issues,
        "summary": {
            "lines": len(dialogue),
            "num_lines_requested": num_lines,
            "avg_words": round(sum(wc) / len(wc), 2) if wc else 0,
            "fillers": filler_total,
            "back_channels": back_total,
            "consecutive_same_speaker": consecutive,
            "simplified_zh_lines": len(simp_lines),
            "scene_missing_lines": len(scene_missing),
            "gender_conflict_lines": summary_gender,
            "cefr": cefr,
        },
    }


def format_report(report: dict) -> str:
    """把报告格式化为控制台多行文本（summary 字段适配 listening）。"""
    lines = [f"[QA] passed={report['passed']}  "
             f"errors={report['n_errors']}  warnings={report['n_warnings']}"]
    for i in report["issues"]:
        tag = "ERROR" if i["severity"] == "error" else "WARN "
        lines.append(f"  [{tag}] {i['check']}: {i['detail']}"
                     + (f"  lines={i['lines']}" if i["lines"] else ""))
    s = report["summary"]
    lines.append(f"  summary: {s['lines']} lines "
                 f"(req {s['num_lines_requested']}), "
                 f"avg {s['avg_words']} words, fillers={s['fillers']}, "
                 f"back_channels={s['back_channels']}, "
                 f"consec_speaker={s['consecutive_same_speaker']}, "
                 f"simp_zh={s['simplified_zh_lines']}, "
                 f"scene_missing={s['scene_missing_lines']}, "
                 f"gender_conflict={s['gender_conflict_lines']}, "
                 f"cefr={s['cefr']}")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    import json
    if len(sys.argv) < 2:
        print("Usage: python quality_gate_listening.py <script.json> [num_lines]")
        sys.exit(2)
    data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    n = int(sys.argv[2]) if len(sys.argv) > 2 else None
    print(format_report(run_listening_quality_gate(data, n)))
