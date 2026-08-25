"""Quest 脚本程序化质检门禁 — 纯 Python 检查，无 LLM 依赖。

七类检查：结构 / 字段 / 自然度 / 重复 / 故事 / 翻译 / 元数据。
输出结构化报告 dict：{"passed", "n_errors", "n_warnings", "issues", "summary"}。
severity="error" 阻断通过（触发修复轮）；"warning" 仅报告并喂给 critique。

可独立运行：python quality_gate.py <script.json> [num_lines]
"""
import json
import sys
from pathlib import Path

# ── 词表 ──────────────────────────────────────────────────────────────

# 核心填充词（自然度密度检查用，不含 so/like 这类高频歧义词）
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

# 常见简体专用字（opencc 缺失时的降级检测）
_SIMP_CHARS = "简们这说吗现过给还让认记谁买卖办杂志证计发为关会见学习亿块钱标题签"

# CEFR → 每行期望词数区间
_CEFR_WORD_RANGE = {
    "A1": (4, 8), "A2": (6, 10), "B1": (8, 13), "B2": (10, 16),
}

# on_screen 规范排序
_CHAR_ORDER = {"char_a": 0, "char_b": 1, "char_c": 2}

# 元数据词数/数量目标
_META_TARGETS = {
    "hook_intro_en": (70, 110),
    "outro": (80, 110),
    "youtube_tags": (15, 20),
    "thumbnail_icons": (4, 5),
}


def _words(text: str) -> list[str]:
    return text.split()


def _content_words(text: str) -> list[str]:
    return [w.strip(".,!?;:'\"()").lower() for w in text.split()
            if len(w.strip(".,!?;:'\"()")) > 3
            and w.strip(".,!?;:'\"()").lower() not in STOPWORDS]


def _is_simplified(zh: str) -> tuple[bool, str]:
    """检测简体字。优先 opencc（zh != s2t(zh) 即含简体），缺失则启发式字符表。"""
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


def _split_phase_lines_ref(num_lines: int) -> tuple[int, int, int, int]:
    """与 llm_client_quest.split_phase_lines 相同的参考配比（本地复制避免循环导入）。"""
    n_buildup = round(num_lines * 0.30)
    n_core = round(num_lines * 0.48)
    n_reveal = round(num_lines * 0.10)
    n_review = num_lines - n_buildup - n_core - n_reveal
    n = [n_buildup, n_core, n_reveal, n_review]
    for i in range(4):
        if n[i] < 1:
            max_idx = n.index(max(n))
            if max_idx != i and n[max_idx] > 1:
                n[max_idx] -= 1
                n[i] = 1
    total = sum(n)
    if total > num_lines:
        while sum(n) > num_lines:
            max_idx = n.index(max(n))
            if n[max_idx] > 1:
                n[max_idx] -= 1
            else:
                break
    elif total < num_lines:
        n[1] += (num_lines - total)
    return tuple(n)


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def run_quality_gate(script: dict, num_lines: int | None = None) -> dict:
    """对 quest 脚本跑全部程序化检查，返回报告。"""
    issues: list[dict] = []

    def add(check: str, severity: str, detail: str, lines: list | None = None):
        issues.append({"check": check, "severity": severity,
                       "detail": detail, "lines": lines or []})

    dialogue = script.get("dialogue", [])
    if num_lines is None:
        num_lines = script.get("_requested_num_lines", 0) or len(dialogue)
    cefr = (script.get("cefr") or "A2").upper()

    # ── 1. 结构 ──────────────────────────────────────────────────────
    if len(dialogue) != num_lines:
        add("structure", "error",
            f"总行数 {len(dialogue)} != 要求 {num_lines}")
    ref = dict(zip(("buildup", "core", "reveal", "review"),
                   _split_phase_lines_ref(num_lines))) if num_lines else {}
    counts = {"buildup": 0, "core": 0, "reveal": 0, "review": 0}
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

    order = {"buildup": 0, "core": 1, "reveal": 2, "review": 3}
    seq = [order.get(line.get("phase", ""), -1) for line in dialogue]
    for i in range(1, len(seq)):
        if seq[i] < seq[i - 1]:
            add("structure", "error",
                f"阶段顺序回退于 line {i}（{dialogue[i-1].get('phase')} → {dialogue[i].get('phase')}）",
                [i - 1, i])
            break

    for i, line in enumerate(dialogue):
        speaker = line.get("speaker", "")
        phase = line.get("phase", "")
        if phase in ("buildup", "reveal", "review") and speaker not in ("char_a", "char_b"):
            add("structure", "error",
                f"line {i} ({phase}) speaker 非法: '{speaker}'", [i])
        elif phase == "core" and speaker not in ("char_a", "char_b", "char_c"):
            add("structure", "error",
                f"line {i} (core) speaker 非法: '{speaker}'", [i])

    core_lines = [i for i, l in enumerate(dialogue) if l.get("phase") == "core"]
    c_speaks = [i for i in core_lines if dialogue[i].get("speaker") == "char_c"]
    if core_lines and not c_speaks:
        add("structure", "error", "core 阶段没有任何 char_c 台词")
    elif core_lines and len(c_speaks) < max(1, len(core_lines) // 6):
        add("structure", "warning",
            f"core 阶段 char_c 台词仅 {len(c_speaks)}/{len(core_lines)} 行（建议 ≥1/5）")

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

    # 环境镜头（on_screen=[]）已废弃：每行画面必须至少有说话人
    empty_os = [i for i, l in enumerate(dialogue) if l.get("on_screen") == []]
    if empty_os:
        add("fields", "error",
            f"on_screen 为空（不允许环境镜头）: lines {empty_os[:10]}", empty_os[:10])

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
        if not (lo <= avg <= hi):
            add("naturalness", "warning",
                f"平均每行 {avg:.1f} 词，CEFR {cefr} 建议 {lo}-{hi}")
        long_lines = [i for i, w in enumerate(wc) if w > hi + 8]
        if long_lines:
            add("naturalness", "warning",
                f"{len(long_lines)} 行超长（>{hi + 8} 词）", long_lines[:10])

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

    # ── 5. 故事完整性 ────────────────────────────────────────────────
    question = script.get("listening_question_en", "")
    if not question.strip():
        add("story", "error", "缺少 listening_question_en")
    else:
        q_words = set(_content_words(question))
        buildup_text = " ".join(t for t, l in zip(texts, dialogue)
                                if l.get("phase") == "buildup").lower()
        hit = sum(1 for w in q_words if w in buildup_text)
        if q_words and hit < max(2, len(q_words) // 2):
            add("story", "warning",
                f"buildup 阶段似乎未自然提出听力问题（关键词命中 {hit}/{len(q_words)}）")

    answer = script.get("answer_en", "")
    if not answer.strip():
        add("story", "warning",
            "缺少 answer_en（无法做答案泄露/揭晓检查；新管线应携带该字段）")
    else:
        ans_words = _content_words(answer)
        if ans_words:
            # 泄露：buildup/core 单行命中 ≥60% 答案实词
            leak_threshold = max(2, -(-len(ans_words) * 3 // 5))
            leaks = []
            for i, l in enumerate(dialogue):
                if l.get("phase") in ("buildup", "core"):
                    tw = set(_content_words(l.get("text", "")))
                    if sum(1 for w in ans_words if w in tw) >= leak_threshold:
                        leaks.append(i)
            if leaks:
                add("story", "error",
                    f"答案疑似在 reveal 前泄露（line 命中 ≥60% 答案实词）", leaks[:10])
            # 揭晓：reveal 全文命中 ≥50%
            reveal_text = set(_content_words(
                " ".join(t for t, l in zip(texts, dialogue)
                         if l.get("phase") == "reveal")))
            reveal_hit = sum(1 for w in ans_words if w in reveal_text)
            if reveal_hit < max(1, len(ans_words) // 2):
                add("story", "error",
                    f"reveal 阶段未明确揭晓答案（实词命中 {reveal_hit}/{len(ans_words)}）")
            # 复习：review 命中 ≥30%
            review_text = set(_content_words(
                " ".join(t for t, l in zip(texts, dialogue)
                         if l.get("phase") == "review")))
            review_hit = sum(1 for w in ans_words if w in review_text)
            if review_hit < max(1, len(ans_words) // 3):
                add("story", "warning",
                    f"review 阶段答案复现不足（实词命中 {review_hit}/{len(ans_words)}）")

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

    # ── 7. 元数据 ────────────────────────────────────────────────────
    hook = script.get("hook_intro_en", "")
    if hook:
        n = len(_words(hook))
        lo, hi = _META_TARGETS["hook_intro_en"]
        if not (lo <= n <= hi):
            add("metadata", "warning", f"hook_intro_en {n} 词（建议 {lo}-{hi}）")
    outro = script.get("outro", "")
    if outro:
        n = len(_words(outro))
        lo, hi = _META_TARGETS["outro"]
        if not (lo <= n <= hi):
            add("metadata", "warning", f"outro {n} 词（建议 {lo}-{hi}）")
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
    if not str(script.get("welcome_en", "")).strip():
        add("metadata", "error", "welcome_en 为空")

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
            "phase_counts": counts,
            "avg_words": round(sum(wc) / len(wc), 2) if wc else 0,
            "fillers": filler_total,
            "back_channels": back_total,
            "empty_on_screen": len(empty_os),
            "consecutive_same_speaker": consecutive,
            "simplified_zh_lines": len(simp_lines),
            "cefr": cefr,
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
                 f"(req {s['num_lines_requested']}), phases={s['phase_counts']}, "
                 f"avg {s['avg_words']} words, fillers={s['fillers']}, "
                 f"empty_os={s['empty_on_screen']}, simp_zh={s['simplified_zh_lines']}")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if len(sys.argv) < 2:
        print("Usage: python quality_gate.py <script.json> [num_lines]")
        sys.exit(2)
    data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    n = int(sys.argv[2]) if len(sys.argv) > 2 else None
    print(format_report(run_quality_gate(data, n)))
