"""Listening 脚本 LLM 复审 + 定向修补 — 对齐 quest Phase E 的轮次机制。

为三个 Original 类模式的 listening 脚本提供：
  1. 程序化预修复（opencc 繁化 / speaker 钳制）
  2. LLM 双评审（story judge + language judge）
  3. 门禁 error + judge issues 合并为定向 patch → 保持行数的局部重写
  4. 轮次循环：跑满 LISTENING_QA_MAX_ROUNDS 轮（默认 3，0 = 禁用），每轮必跑双评审；
     轮满仍有 error 则继续修到 0 error，硬上限 _QA_HARD_CAP 轮（超限接受 best effort）

报告结构与 quest 一致，存入 script["_qa"] = {"rounds": [...], "final": {...}}。
只 import llm_client 内部助手，不反向依赖 quest 子包（通用小助手本地精简复制）。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from llm_client import (
    _chat,
    _env_get,
    _extract_json,
    resolve_max_line_words,
)
from quality_gate_listening import (
    run_listening_quality_gate, format_report, PROMPT_FIELDS,
)


def _prompt_fields(structure: str) -> tuple[str, ...]:
    """当前结构模式行级必需的 prompt 字段（未知结构回退 original）。"""
    return PROMPT_FIELDS.get(structure if structure in PROMPT_FIELDS
                             else "original", PROMPT_FIELDS["original"])


def _prompt_field_reqs(pf: tuple[str, ...], scene: str,
                       style_block: str) -> str:
    """按结构生成重写 prompt 的行级 prompt 字段要求文本（cutout 为空）。"""
    if not pf:
        return ""
    reqs = []
    if "image_prompt" in pf:
        reqs.append(f'- "image_prompt": the speaking character\'s FULL physical '
                    f"description + role + the {scene} location + action matching "
                    f"the text{style_block and ' + the style phrase'}")
    if "video_prompt" in pf:
        reqs.append('- "video_prompt": same character description + action + '
                    "the line's text spoken naturally + match the reference "
                    "image exactly")
    return "\n" + "\n".join(reqs)

_MAX_PATCHES = 8
_PATCH_MERGE_GAP = 5
_MAX_SPAN = 20
_QA_HARD_CAP = 10  # 有 error 时的修复轮数硬上限，防止 LLM 修不动时无限循环

_WRITER_SYSTEM = (
    "You are an expert ESL script writer for slow-listening videos for "
    "overseas Chinese beginners. Natural American English, textbook-free. "
    "Output valid JSON only."
)


def _env_int(name: str, default: int) -> int:
    # 经 _env_get 读取：批量生成线程的线程局部 override 才能生效
    try:
        return int(_env_get(name, "") or default)
    except ValueError:
        return default


# ---------------------------------------------------------------------------
# 程序化预修复（无 LLM）
# ---------------------------------------------------------------------------

def _apply_fixes(script: dict):
    """确定性修复：speaker 钳制 + zh 简体→繁体（opencc 可选）。"""
    for line in script.get("dialogue", []):
        sp = line.get("speaker", "")
        if sp not in ("char_a", "char_b"):
            line["speaker"] = "char_a"

    try:
        from opencc import OpenCC
    except ImportError:
        return
    cc = OpenCC("s2t")

    def conv(s):
        return cc.convert(s) if isinstance(s, str) and s else s

    for field in ("title_zh", "scene_zh", "intro_zh", "welcome_zh",
                  "outro_zh", "youtube_title", "youtube_description",
                  "thumbnail_subtitle"):
        script[field] = conv(script.get(field, ""))
    for line in script.get("dialogue", []):
        line["zh"] = conv(line.get("zh", ""))


# ---------------------------------------------------------------------------
# LLM 评审（story judge + language judge）
# ---------------------------------------------------------------------------

def _dialogue_compact(script: dict) -> str:
    items = []
    for i, l in enumerate(script.get("dialogue", [])):
        items.append({"i": i, "s": l.get("speaker", ""), "t": l.get("text", ""),
                      "z": l.get("zh", "")})
    return json.dumps(items, ensure_ascii=False, separators=(",", ":"))


def _gate_issues_digest(report: dict) -> str:
    lines = []
    for i in report.get("issues", []):
        tag = "E" if i["severity"] == "error" else "W"
        lines.append(f"[{tag}] {i['check']}: {i['detail']}")
    return "\n".join(lines[:60]) or "(none)"


def _build_judge_prompt(kind: str, script: dict, report: dict) -> str:
    dialogue_json = _dialogue_compact(script)
    meta_block = f"""Title: {script.get('title', '')}
Scene: {script.get('scene', '')} ({script.get('scene_zh', '')})
CEFR level: {script.get('cefr', 'A2')}
char_a: {script.get('char_a_role', '')} — {script.get('char_a_description', '')} ({script.get('char_a_gender', '')})
char_b: {script.get('char_b_role', '')} — {script.get('char_b_description', '')} ({script.get('char_b_gender', '')})
"""
    machine = f"""Machine checks already found (do NOT repeat these):
{_gate_issues_digest(report)}
"""
    if kind == "story":
        task = """You are a STORY COHERENCE judge for an ESL slow-listening video script (a real-life scene dialogue for listening practice).
Find ONLY story-level problems the machine checks above cannot detect:
1. Topic regression: a later line re-introduces a topic/question already settled earlier (as if new).
2. Facts contradict across lines (prices, names, times, quantities, places).
3. The story arc is incomplete or abrupt: no clear beginning/development/resolution, or an ending that feels cut off.
4. Scene drift in the STORY itself (actions imply a different location than the stated scene).
5. Character role confusion: a speaker acts contrary to their role (e.g. the customer gives the waiter's lines).
Report each problem as a line range [start, end] (0-indexed, inclusive)."""
    elif kind == "engagement":
        task = """You are an ENGAGEMENT judge for an ESL slow-listening video script (audience: overseas Chinese learners who want vivid, natural conversation).
The script may be technically correct but DULL. Find the DULLEST stretches (2-6 consecutive lines) that feel like a flat transaction or textbook Q&A, and say how to make them vivid:
- add a genuine human reaction (amusement, surprise, relief, mild annoyance)
- add a light joke, a personal remark, or a back-channel ("oh really?", "that makes sense")
- raise a small conflict or unexpected beat if the story has none
- replace textbook phrasing with the way people actually talk (contractions, fragments)
- vary line length (very short reactions vs fuller sentences)
Rules: do NOT break story coherence, do NOT exceed the per-line word limit, do NOT change the scene or the facts. Report each dull stretch as a line range [start, end] (0-indexed, inclusive)."""
    else:
        task = """You are a LANGUAGE QUALITY judge for an ESL slow-listening video script (target: overseas Chinese beginners).
Find ONLY language-level problems the machine checks above cannot detect:
1. Robotic stretches: 3+ consecutive lines with flat Q&A rhythm or identical sentence structure.
2. Unnatural English: textbook phrasing no native speaker would say; register above the CEFR level (hard words/grammar).
3. zh translation errors: mistranslation, wrong meaning, stiff machine-like 繁體中文, inconsistent terminology for the same English term.
4. Repetitive openers or verbal tics the machine didn't flag.
Report each problem as a line range [start, end] (0-indexed, inclusive)."""

    return f"""{meta_block}{machine}
{task}

Dialogue (compact JSON, i = line index, s = speaker, t = English text, z = Traditional Chinese):
{dialogue_json}

Output JSON ONLY:
{{"issues": [{{"type": "...", "lines": [start, end], "problem": "...", "fix_hint": "how to fix"}}]}}

If everything is fine, output {{"issues": []}}. Max 10 issues, most important first."""


def _parse_judge_issues(raw, n_lines: int) -> list[dict]:
    issues = []
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        lines = item.get("lines", [])
        if isinstance(lines, int):
            lines = [lines, lines]
        if not isinstance(lines, list) or len(lines) < 2:
            continue
        try:
            s, e = int(lines[0]), int(lines[1])
        except (TypeError, ValueError):
            continue
        s, e = max(0, min(s, e)), min(n_lines - 1, max(s, e))
        if s > e:
            continue
        if not str(item.get("problem", "")).strip():
            continue
        issues.append({
            "type": str(item.get("type", "other")),
            "lines": [s, e],
            "problem": str(item["problem"])[:300],
            "fix_hint": str(item.get("fix_hint", ""))[:300],
        })
    return issues[:10]


def _engagement_qa_enabled() -> bool:
    """C（张力评审）开关：SCRIPT_ENGAGEMENT_QA=1/true/yes/on（线程局部优先）。"""
    return _env_get("SCRIPT_ENGAGEMENT_QA", "").strip().lower() in (
        "1", "true", "yes", "on")


def _critique_listening(script: dict, report: dict) -> list[dict]:
    """Run story judge + language judge (+engagement judge if enabled)."""
    kinds = ["story", "language"]
    if _engagement_qa_enabled():
        kinds.append("engagement")
    combined = []
    for kind in kinds:
        try:
            content = _chat(
                [{"role": "system",
                  "content": "You are a strict script quality judge for ESL videos. "
                             "Output valid JSON only."},
                 {"role": "user",
                  "content": _build_judge_prompt(kind, script, report)}],
                temperature=0.3, max_tokens=8192, reasoning_effort="low")
            result = _extract_json(content)
        except (RuntimeError, json.JSONDecodeError, ValueError) as e:
            print(f"  [QA] {kind} judge failed: {e}")
            continue
        raw = (result.get("issues", []) if isinstance(result, dict)
               else (result if isinstance(result, list) else []))
        found = _parse_judge_issues(raw, len(script.get("dialogue", [])))
        if kind == "engagement":
            found = found[:3]  # 张力重写每轮最多 3 段，控制成本与重写抖动
        print(f"    {kind} judge: {len(found)} issues")
        combined.extend(found)
    return combined


# ---------------------------------------------------------------------------
# Patch 合并 + 定向修补（保持行数的局部重写）
# ---------------------------------------------------------------------------

def _merge_patches(report: dict, judge_issues: list[dict],
                   script: dict, structure: str = "original"
                   ) -> list[tuple[int, int, list[str]]]:
    """Merge gate errors + judge issues into repair patches (s, e, hints)."""
    dialogue = script.get("dialogue", [])
    n = len(dialogue)
    spans: list[tuple[int, int, str]] = []

    for issue in report.get("issues", []):
        if issue["severity"] != "error" or not issue.get("lines"):
            continue
        ls = issue["lines"]
        if "完全重复行" in issue["detail"]:
            later = max(ls)
            spans.append((max(0, later - 1), min(n - 1, later + 1),
                          "Exact duplicate of an earlier line — rewrite with "
                          "different wording and new content."))
        elif "speaker 非法" in issue["detail"]:
            for l in ls:
                spans.append((l, l, "Fix the speaker (only char_a/char_b allowed)."))
        elif "为空" in issue["detail"]:
            fields_desc = "text/zh/phonetic"
            pf = _prompt_fields(structure)
            if pf:
                fields_desc += "/" + "/".join(pf)
            for l in ls:
                spans.append((l, l,
                              "This line has empty required fields — fill in "
                              f"{fields_desc} completely."))
        elif "男女指示词混用" in issue["detail"]:
            for l in ls:
                spans.append((l, l,
                              "Gender words conflict in this line's prompts — make all "
                              "pronouns/appearance words match the speaker's gender."))

    for issue in judge_issues:
        s, e = issue["lines"]
        hint = f"[{issue['type']}] {issue['problem']} → {issue['fix_hint']}"
        spans.append((s, e, hint))

    if not spans:
        return []

    spans.sort(key=lambda t: (t[0], t[1]))
    merged: list[list] = []
    for s, e, hint in spans:
        if (merged and s - merged[-1][1] <= _PATCH_MERGE_GAP
                and max(merged[-1][1], e) - merged[-1][0] + 1 <= _MAX_SPAN):
            merged[-1][1] = max(merged[-1][1], e)
            if hint not in merged[-1][2]:
                merged[-1][2].append(hint)
        else:
            merged.append([s, e, [hint]])
    return [(s, e, hints) for s, e, hints in merged[:_MAX_PATCHES]]


def _safe_extract(text: str):
    """Extract JSON from LLM response; return dict or list (best effort)."""
    import re
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return _extract_json(text)


def _as_line_list(result) -> list[dict]:
    """Coerce judge/writer output into a list of line dicts."""
    if isinstance(result, list):
        return [x for x in result if isinstance(x, dict)]
    if isinstance(result, dict):
        for key in ("lines", "dialogue", "data"):
            if isinstance(result.get(key), list):
                return [x for x in result[key] if isinstance(x, dict)]
    return []


def _fmt_lines(ls) -> str:
    return "\n".join(f"  {l.get('speaker', '?')}: {l.get('text', '')}" for l in ls)


def _batch_translate_zh(lines: list[tuple[int, str]]) -> dict[int, str]:
    """Translate English dialogue lines to Traditional Chinese in one batch call."""
    if not lines:
        return {}
    items = [{"i": i, "text": text} for i, text in lines]
    prompt = f"""Translate each English sentence to Traditional Chinese (繁體中文).
Return a JSON array, same length and order, each item with "i" and "zh":
{json.dumps(items, ensure_ascii=False)}

Output: [{{"i": 0, "zh": "繁中翻譯"}}, ...]
JSON array ONLY, no markdown."""
    try:
        content = _chat(
            [{"role": "system",
              "content": "You are a professional translator. Translate English "
                         "to Traditional Chinese (繁體中文). Output valid JSON array only."},
             {"role": "user", "content": prompt}],
            temperature=0.3, max_tokens=4096, reasoning_effort="low")
        result = _safe_extract(content)
    except (RuntimeError, json.JSONDecodeError, ValueError) as e:
        print(f"  [LLM] Fallback zh translation failed: {e}")
        return {}

    zh_map: dict[int, str] = {}
    if isinstance(result, list):
        for item in result:
            if isinstance(item, dict):
                idx = item.get("i", -1)
                zh = str(item.get("zh", "")).strip()
                if idx >= 0 and zh:
                    zh_map[idx] = zh
    elif isinstance(result, dict):
        for idx_str, zh in result.items():
            try:
                idx = int(idx_str)
                if str(zh).strip():
                    zh_map[idx] = str(zh).strip()
            except (ValueError, TypeError):
                continue
    for i, _ in lines:
        if i not in zh_map:
            zh_map[i] = "（翻譯待補）"
    return zh_map


def _repair_patches(script: dict, patches: list[tuple[int, int, list[str]]],
                    cefr: str, structure: str = "original"):
    """Rewrite each patch in place, preserving the exact line count."""
    dialogue = script["dialogue"]
    scene = script.get("scene", "")
    char_a_desc = script.get("char_a_description", "")
    char_b_desc = script.get("char_b_description", "")
    char_a_role = script.get("char_a_role", "")
    char_b_role = script.get("char_b_role", "")
    pf = _prompt_fields(structure)
    cap = resolve_max_line_words()

    for s, e, hints in sorted(patches, key=lambda p: -p[0]):
        k = e - s + 1
        orig = dialogue[s:e + 1]
        ctx_before = dialogue[max(0, s - 3):s]
        ctx_after = dialogue[e + 1:e + 4]

        style_prompt = ""
        try:
            from style_manager import get_active_style_prompt
            style_prompt = str(get_active_style_prompt() or "").strip()
        except Exception:
            style_prompt = ""
        style_block = ""
        if style_prompt and pf:
            style_block = (f"\nVisual style phrase (copy VERBATIM into "
                           f"{' and '.join(pf)}): \"{style_prompt}\"\n")

        orig_lines = [{"speaker": l.get("speaker"), "text": l.get("text"),
                       "phonetic": l.get("phonetic"), "zh": l.get("zh"),
                       **{f: l.get(f) for f in pf}}
                      for l in orig]
        orig_json = json.dumps(orig_lines,
                               ensure_ascii=False, separators=(",", ":"))

        prompt = f"""Rewrite EXACTLY {k} dialogue lines (indices {s}-{e}) of an ongoing ESL conversation at a {scene}.

Scene: {scene}
char_a: {char_a_role} — {char_a_desc}
char_b: {char_b_role} — {char_b_desc}
{style_block}
Lines BEFORE (context, do not rewrite):
{_fmt_lines(ctx_before)}

Lines AFTER (context, do not rewrite):
{_fmt_lines(ctx_after)}

Current lines to rewrite:
{orig_json}

Problems to fix:
{chr(10).join(f"- {h}" for h in hints)}

Requirements:
- Keep EXACTLY {k} lines, same order, speakers only "char_a" or "char_b".
- CEFR {cefr}: natural conversational English, contractions, fillers ("well", "um", "you know"), varied length, each line AT MOST {cap} words (HARD LIMIT — subtitles must fit 2 lines).
- Every line keeps ALL fields:
  - "text": the English sentence
  - "phonetic": IPA transcription in /slashes/
  - "zh": Traditional Chinese (繁體中文) translation{_prompt_field_reqs(pf, scene, style_block)}
- Seamless continuity with the before/after context lines.

Output: JSON array of exactly {k} objects. JSON ONLY."""

        messages = [{"role": "system", "content": _WRITER_SYSTEM},
                    {"role": "user", "content": prompt}]
        lines = None
        try:
            content = _chat(messages, temperature=0.6, max_tokens=4096,
                            reasoning_effort="low")
            lines = _as_line_list(_safe_extract(content))
            if not lines or len(lines) != k:
                got = len(lines) if lines else 0
                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content":
                                 f"You returned {got} lines; exactly {k} are "
                                 f"required. Return the corrected JSON array of "
                                 f"exactly {k} line objects. JSON ONLY."})
                content = _chat(messages, temperature=0.6, max_tokens=4096,
                                reasoning_effort="low")
                lines = _as_line_list(_safe_extract(content))
        except (RuntimeError, json.JSONDecodeError, ValueError) as ex:
            print(f"    patch L{s}-{e} failed: {ex}")
            continue

        if lines and len(lines) == k:
            for new, old in zip(lines, orig):
                sp = new.get("speaker", "")
                if sp not in ("char_a", "char_b"):
                    new["speaker"] = old.get("speaker", "char_a")
                for f in ("phonetic",):
                    if not new.get(f):
                        new[f] = old.get(f, "")
            dialogue[s:e + 1] = lines
            print(f"    patch L{s}-{e}: rewritten ({len(hints)} hints)")
        else:
            print(f"    patch L{s}-{e}: kept original "
                  f"(got {len(lines) if lines else 0}/{k})")

    # zh fallback for lines emptied by repair
    empty = [(i, l.get("text", "")) for i, l in enumerate(dialogue)
             if not str(l.get("zh", "")).strip()]
    if empty:
        zh_map = _batch_translate_zh(empty)
        for i, zh in zh_map.items():
            dialogue[i]["zh"] = zh


# ---------------------------------------------------------------------------
# 主入口：轮次循环
# ---------------------------------------------------------------------------

def run_listening_qa(script: dict, num_lines: int,
                     structure: str = "original") -> dict:
    """Gate + critique/repair loop. Mutates script in place, stores script["_qa"].

    轮次语义：跑满 qa_rounds 轮（每轮必跑双评审，发现问题即修复）；轮满仍有
    error 则继续修到 0 error，硬上限 _QA_HARD_CAP 轮；有 error 但无可行动
    patch 时提前接受 best effort（脚本未变，再循环结果相同）。
    """
    qa_rounds = _env_int("LISTENING_QA_MAX_ROUNDS", 3)
    if qa_rounds <= 0:
        return script
    hard_cap = max(_QA_HARD_CAP, qa_rounds)
    # 每行最大词数（字幕两行约束）：线程局部 override 优先，传给门禁与修复 prompt
    cap = resolve_max_line_words()

    print(f"  [QA] listening quality loop: min {qa_rounds} rounds, "
          f"error-repair up to {hard_cap} rounds")
    _apply_fixes(script)
    rounds_log = []
    for rnd in range(1, hard_cap + 1):
        report = run_listening_quality_gate(script, num_lines,
                                            structure=structure,
                                            max_line_words=cap)
        print(f"  [QA] Round {rnd}: gate errors={report['n_errors']} "
              f"warnings={report['n_warnings']}, running LLM judges...")
        judge_issues = _critique_listening(script, report)
        rounds_log.append({
            "round": rnd,
            "gate": report,
            "judge_issues": judge_issues,
        })
        patches = _merge_patches(report, judge_issues, script,
                                 structure=structure)
        if patches:
            print(f"  [QA] Round {rnd}: repairing {len(patches)} patches: "
                  + ", ".join(f"L{p[0]}-{p[1]}" for p in patches))
            _repair_patches(script, patches, script.get("cefr", "A2"),
                            structure=structure)
            _apply_fixes(script)
        if rnd >= qa_rounds and report["n_errors"] == 0:
            print(f"  [QA] Round {rnd}: done (0 errors, {rnd} rounds run)")
            break
        if report["n_errors"] > 0 and not patches:
            print(f"  [QA] Round {rnd}: errors remain but no actionable "
                  f"patches, accepting best effort")
            break

    final = run_listening_quality_gate(script, num_lines, structure=structure,
                                       max_line_words=cap)
    script["_qa"] = {"rounds": rounds_log, "final": final}
    print(format_report(final))
    if not final["passed"]:
        print(f"  [QA] WARNING: {final['n_errors']} errors remain after "
              f"{len(rounds_log)} QA rounds (accepted best effort)")
    return script
