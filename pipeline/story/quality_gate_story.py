"""Story 模式程序化质检门禁 — 纯 Python 检查，无 LLM 依赖。

与 quest/quality_gate.py 同构的报告契约：
{"passed", "n_errors", "n_warnings", "issues", "summary"}。
severity="error" 触发 QA 修复轮；"warning" 仅报告。

检查维度：结构（行数/阶段配比/顺序/说话人白名单）/ 字段（on_screen/zh）/
自然度 / 重复 / 故事完整性（埋题-回收链）/ 翻译 / 元数据 / 冷开场约束。

可独立运行：python quality_gate_story.py <script.json> [num_lines]
"""
import json
import re
import sys
from pathlib import Path

# ── 类型/阶段定义（与 llm_client_story 保持一致；本地复制避免循环导入）──

STORY_KINDS = ("plot", "chat", "solo")

PHASES = {
    "plot": ("opening", "setup", "conflict", "twist", "resolution", "finale"),
    "chat": ("opening", "activity", "finale"),
    "solo": ("opening", "body", "finale"),
}

PHASE_RATIOS = {
    "plot": {"opening": 0.10, "setup": 0.20, "conflict": 0.30,
             "twist": 0.15, "resolution": 0.15, "finale": 0.10},
    "chat": {"opening": 0.10, "activity": 0.78, "finale": 0.12},
    "solo": {"opening": 0.10, "body": 0.78, "finale": 0.12},
}

SPEAKERS = {
    "plot": ("char_a", "char_b", "char_c", "char_d", "char_e"),
    "chat": ("char_a", "char_e"),
    "solo": ("char_a",),
}

# char_e 仅在 has_guest / chat 类型时可用（plot 的 NPC 嘉宾可选）
CTA_TYPES = ("recall_question", "open_question", "golden_words",
             "simple_cta", "cliffhanger")

# ── 词表（与 quest 门禁同源）──────────────────────────────────────────

FILLERS = [
    "well", "you know", "hmm", "actually", "oh", "um", "i mean",
    "let me think", "you see", "sort of", "kind of", "yeah", "haha",
]

BACK_CHANNELS = [
    "oh really", "that makes sense", "interesting", "wow", "i see",
    "right", "exactly", "perfect", "nice", "got it", "of course",
]

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "to", "of", "in", "it",
    "that", "this", "and", "or", "for", "with", "you", "your", "they",
    "their", "not", "but", "have", "has", "do", "does", "did", "be",
    "been", "at", "on", "by", "as", "from", "he", "she", "his", "her",
    "we", "us", "our", "i", "me", "my", "if", "so", "what", "why",
    "how", "when", "there", "here", "just", "very", "really",
}

_SIMP_CHARS = "简们这说吗现过给还让认记谁买卖办杂志证计发为关会见学习亿块钱标题签"

_CEFR_WORD_RANGE = {
    "A1": (4, 8), "A2": (6, 10), "B1": (8, 13), "B2": (10, 16),
}

_CHAR_ORDER = {"char_a": 0, "char_b": 1, "char_c": 2, "char_d": 3, "char_e": 4}

_META_TARGETS = {
    "youtube_tags": (15, 20),
    "thumbnail_icons": (4, 5),
}

_MAX_LINE_WORDS_DEFAULT = 10

# story 冷开场约束：这些旁白字段必须为空（welcome/hook/outro 由对话行承担）
_NARRATION_FIELDS_MUST_EMPTY = ("welcome_en", "hook_intro_en", "outro")


def _resolve_max_line_words(max_line_words: int | None) -> int:
    if max_line_words is None:
        import os
        raw = (os.environ.get("QUEST_MAX_LINE_WORDS", "") or "").strip()
        try:
            max_line_words = int(raw) if raw else _MAX_LINE_WORDS_DEFAULT
        except ValueError:
            max_line_words = _MAX_LINE_WORDS_DEFAULT
    return max(4, min(20, int(max_line_words)))


def _words(text: str) -> list[str]:
    return text.split()


def _content_words(text: str) -> list[str]:
    return [w.strip(".,!?;:'\"()").lower() for w in text.split()
            if len(w.strip(".,!?;:'\"()")) > 3
            and w.strip(".,!?;:'\"()").lower() not in STOPWORDS]


def _is_simplified(zh: str) -> tuple[bool, str]:
    try:
        from opencc import OpenCC
        cc = OpenCC("s2t")
        converted = cc.convert(zh)
        if converted != zh:
            for a, b in zip(zh, converted):
                if a != b:
                    return True, a
        return False, ""
    except ImportError:
        for c in zh:
            if c in _SIMP_CHARS:
                return True, c
        return False, ""


def split_phase_lines_story(num_lines: int, kind: str) -> dict[str, int]:
    """按类型配比拆分行数，保证 opening≥2、finale≥2、各阶段≥1。"""
    ratios = PHASE_RATIOS.get(kind, PHASE_RATIOS["plot"])
    counts = {ph: max(1, round(num_lines * r)) for ph, r in ratios.items()}
    # 最小值保证
    for m in ("opening", "finale"):
        if m in counts and counts[m] < 2:
            counts[m] = 2
    total = sum(counts.values())
    # 修正总行数：从最大阶段增删
    while total > num_lines:
        big = max(counts, key=lambda p: counts[p])
        if counts[big] > 2:
            counts[big] -= 1
            total -= 1
        else:
            break
    if total < num_lines:
        big = max(counts, key=lambda p: counts[p])
        counts[big] += num_lines - total
    return counts


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def run_quality_gate(script: dict, num_lines: int | None = None,
                     max_line_words: int | None = None) -> dict:
    """对 story 脚本跑全部程序化检查，返回报告。"""
    issues: list[dict] = []

    def add(check: str, severity: str, detail: str, lines: list | None = None):
        issues.append({"check": check, "severity": severity,
                       "detail": detail, "lines": lines or []})

    dialogue = script.get("dialogue", [])
    if num_lines is None:
        num_lines = script.get("_requested_num_lines", 0) or len(dialogue)
    kind = str(script.get("story_kind", "plot")).strip().lower()
    if kind not in STORY_KINDS:
        add("structure", "error", f"未知 story_kind '{kind}'")
        kind = "plot"
    cefr = (script.get("cefr") or "A2").upper()
    max_line_words = _resolve_max_line_words(max_line_words)
    phase_seq = PHASES[kind]
    phase_order = {ph: i for i, ph in enumerate(phase_seq)}
    speakers_allowed = list(SPEAKERS[kind])
    has_guest = bool(script.get("has_guest", False)) or kind == "chat"

    # ── 1. 结构 ──────────────────────────────────────────────────────
    if len(dialogue) != num_lines:
        add("structure", "error",
            f"总行数 {len(dialogue)} != 要求 {num_lines}")
    ref = split_phase_lines_story(num_lines, kind) if num_lines else {}
    counts = {ph: 0 for ph in phase_seq}
    for line in dialogue:
        ph = line.get("phase", "")
        if ph in counts:
            counts[ph] += 1
        else:
            add("structure", "error",
                f"未知 phase '{ph}'（line text: {line.get('text', '')[:40]}）")
    for ph, target in ref.items():
        if counts[ph] != target:
            add("structure", "error",
                f"{ph} 行数 {counts[ph]} != 目标 {target}")
    for ph, cnt in counts.items():
        if cnt == 0 and num_lines >= 20:
            add("structure", "error", f"阶段 {ph} 为空")

    seq = [phase_order.get(line.get("phase", ""), -1) for line in dialogue]
    for i in range(1, len(seq)):
        if seq[i] < seq[i - 1]:
            add("structure", "error",
                f"阶段顺序回退于 line {i}（{dialogue[i-1].get('phase')} → "
                f"{dialogue[i].get('phase')}）", [i - 1, i])
            break

    for i, line in enumerate(dialogue):
        speaker = line.get("speaker", "")
        if speaker not in speakers_allowed:
            add("structure", "error",
                f"line {i} speaker 非法: '{speaker}'（{kind} 允许 "
                f"{speakers_allowed}）", [i])

    # ── 2. 字段完整性 ────────────────────────────────────────────────
    for i, line in enumerate(dialogue):
        for f in ("speaker", "text", "phase"):
            if not str(line.get(f, "")).strip():
                add("fields", "error", f"line {i} 字段 '{f}' 为空", [i])
        if not str(line.get("zh", "")).strip():
            add("fields", "error", f"line {i} 字段 'zh' 为空", [i])
        os_ = line.get("on_screen", None)
        if os_ is None:
            add("fields", "warning", f"line {i} 缺 'on_screen'，将默认为说话人", [i])
        elif isinstance(os_, list):
            bad = [k for k in os_ if k not in _CHAR_ORDER]
            if bad:
                add("fields", "error", f"line {i} on_screen 非法键 {bad}", [i])
            elif os_ and os_ != sorted(set(os_), key=lambda k: _CHAR_ORDER[k]):
                add("fields", "warning",
                    f"line {i} on_screen 顺序/重复不规范: {os_}（可程序化修复）", [i])

    empty_os = [i for i, l in enumerate(dialogue) if l.get("on_screen") == []]
    if empty_os:
        add("fields", "error",
            f"on_screen 为空（不允许环境镜头）: lines {empty_os[:10]}", empty_os[:10])

    # NPC 一致性：char_e 台词但脚本声明无嘉宾
    e_speaks = any(l.get("speaker") == "char_e" for l in dialogue)
    e_on_screen = any("char_e" in (l.get("on_screen") or []) for l in dialogue)
    if kind == "solo" and (e_speaks or e_on_screen):
        add("structure", "error", "solo 类型不允许 char_e（NPC/嘉宾）出场")
    elif not has_guest and (e_speaks or e_on_screen):
        add("structure", "error",
            "对话使用了 char_e 但脚本未声明 has_guest=true")
    if has_guest:
        if not str(script.get("char_e_description", "")).strip():
            add("fields", "error", "has_guest=true 但缺少 char_e_description")
        if str(script.get("char_e_gender", "")).lower() not in ("male", "female"):
            add("fields", "error",
                f"char_e_gender 非法: '{script.get('char_e_gender', '')}'")

    # ── 3. 自然度 ────────────────────────────────────────────────────
    texts = [l.get("text", "") for l in dialogue]
    all_lower = " ".join(texts).lower()
    filler_total = sum(all_lower.count(f) for f in FILLERS)
    back_total = sum(all_lower.count(b) for b in BACK_CHANNELS)
    filler_target = max(8, round(num_lines * 0.16)) if num_lines else 8
    if filler_total < filler_target:
        add("naturalness", "warning",
            f"核心填充词 {filler_total} 个（建议 ≥{filler_target}，约每 6 行 1 个）")
    if back_total < max(3, filler_target // 4):
        add("naturalness", "warning",
            f"back-channeling {back_total} 个（建议 ≥{max(3, filler_target // 4)}）")

    wc = [len(_words(t)) for t in texts]
    streak = 0
    for i, w in enumerate(wc):
        streak = streak + 1 if w <= 5 else 0
        if streak >= 3:
            add("naturalness", "warning",
                f"连续 3+ 短行（≤5 词）止于 line {i}", [i - 2, i])
            streak = 0

    if kind != "solo":
        consecutive = sum(1 for i in range(1, len(dialogue))
                          if dialogue[i].get("speaker") == dialogue[i - 1].get("speaker")
                          and dialogue[i].get("phase") == dialogue[i - 1].get("phase"))
        consec_min = max(3, round(num_lines * 0.04)) if num_lines else 3
        consec_max = max(8, round(num_lines * 0.12)) if num_lines else 8
        if not (consec_min <= consecutive <= consec_max):
            add("naturalness", "warning",
                f"同角色连续发言 {consecutive} 次（建议 {consec_min}-{consec_max}）")

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
        for j in range(i + 4, len(texts)):
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

    # ── 5. 故事完整性：埋题-回收链 ────────────────────────────────────
    question = script.get("listening_question_en", "")
    if not question.strip():
        add("story", "error", "缺少 listening_question_en（结尾回收问题）")
    cta_type = str(script.get("cta_type", "")).strip().lower()
    if cta_type not in CTA_TYPES:
        add("story", "error", f"cta_type 非法: '{cta_type}'（允许 {CTA_TYPES}）")

    finale_lines = [i for i, l in enumerate(dialogue)
                    if l.get("phase") == "finale"]
    finale_text = " ".join(texts[i] for i in finale_lines).lower()
    if len(finale_lines) < 2:
        add("story", "error", f"finale 阶段行数不足（{len(finale_lines)}，需 ≥2）")
    if "comment" not in finale_text and cta_type in ("recall_question",
                                                     "open_question"):
        add("story", "warning",
            "finale 未出现 comment 引导（同款结尾均为 write it in the comments）")

    answer = str(script.get("answer_en", "") or "").strip()
    if kind == "plot" and cta_type == "recall_question" and answer:
        ans_words = _content_words(answer)
        # 埋题：opening/setup 应自然展示该细节（warning，交给 LLM 评审细判）。
        # 注意：结尾【不重述答案】是同款行为（观众写答案），不检查 answer 出现。
        if ans_words:
            early_text = " ".join(texts[i] for i, l in enumerate(dialogue)
                                  if l.get("phase") in ("opening", "setup")).lower()
            plant_hit = sum(1 for w in ans_words if w in early_text)
            if plant_hit < max(1, len(ans_words) // 2):
                add("story", "warning",
                    f"opening/setup 似乎未埋入答案细节（实词命中 "
                    f"{plant_hit}/{len(ans_words)}）")
    if question.strip():
        # 回收：finale 必须把问题说给观众（同款结尾核心动作）
        q_words = set(_content_words(question))
        q_hit = sum(1 for w in q_words if w in finale_text)
        if q_words and q_hit < max(1, len(q_words) // 2):
            add("story", "error",
                f"finale 缺少回收问题（关键词命中 {q_hit}/{len(q_words)}）")

    # ── 6. 翻译质量 ──────────────────────────────────────────────────
    simp_lines = []
    for i, l in enumerate(dialogue):
        zh = l.get("zh", "")
        if zh:
            is_simp, _ch = _is_simplified(zh)
            if is_simp:
                simp_lines.append(i)
    if simp_lines:
        add("translation", "warning",
            f"{len(simp_lines)} 行 zh 疑似含简体字（可 opencc 程序化修复）",
            simp_lines[:10])

    for kw in script.get("key_words", []):
        en = str(kw.get("en", "")).lower()
        zh = str(kw.get("zh", "")).strip()
        if not en or not zh:
            continue
        en_uses = sum(1 for t in texts if en in t.lower())
        zh_all = "".join(l.get("zh", "") for l in dialogue)
        zh_uses = zh_all.count(zh)
        if en_uses >= 3 and zh_uses < max(1, en_uses // 2):
            add("translation", "warning",
                f"关键词 '{en}' EN 出现 {en_uses} 次但 zh '{zh}' 仅 {zh_uses} 次（术语一致性）")

    # ── 7. 元数据 + 冷开场约束 ───────────────────────────────────────
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
    scenes = script.get("scene_images", [])
    if len(scenes) < 8:
        add("metadata", "warning", f"scene_images {len(scenes)} 个（建议 ≥8）")
    for field in _NARRATION_FIELDS_MUST_EMPTY:
        if str(script.get(field, "") or "").strip():
            add("structure", "error",
                f"story 冷开场约束: {field} 必须为空（旁白由对话行承担）")

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
            "story_kind": kind,
            "phase_counts": counts,
            "avg_words": round(sum(wc) / len(wc), 2) if wc else 0,
            "fillers": filler_total,
            "back_channels": back_total,
            "empty_on_screen": len(empty_os),
            "consecutive_same_speaker": consecutive if kind != "solo" else 0,
            "simplified_zh_lines": len(simp_lines),
            "cefr": cefr,
            "cta_type": cta_type,
        },
    }


def format_report(report: dict) -> str:
    """把报告格式化为控制台多行文本。"""
    lines = [f"[QA] passed={report['passed']}  "
             f"errors={report['n_errors']}  warnings={report['n_warnings']}"]
    for i in report["issues"]:
        tag = "ERROR" if i["severity"] == "error" else "WARN "
        lines.append(f"  [{tag}] {i['check']}: {i['detail']}"
                     + (f"  lines={i['lines']}" if i["lines"] else ""))
    s = report["summary"]
    lines.append(f"  summary: {s['lines']} lines "
                 f"(req {s['num_lines_requested']}), kind={s['story_kind']}, "
                 f"phases={s['phase_counts']}, avg {s['avg_words']} words, "
                 f"fillers={s['fillers']}, empty_os={s['empty_on_screen']}, "
                 f"simp_zh={s['simplified_zh_lines']}, cta={s['cta_type']}")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if len(sys.argv) < 2:
        print("Usage: python quality_gate_story.py <script.json> [num_lines]")
        sys.exit(2)
    data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    n = int(sys.argv[2]) if len(sys.argv) > 2 else None
    print(format_report(run_quality_gate(data, n)))
