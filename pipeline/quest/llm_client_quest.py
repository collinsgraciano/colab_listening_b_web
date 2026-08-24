"""Quest LLM client — multi-turn generation with auto critique-repair loop.

5-phase pipeline (quality-first, ~20-28 LLM calls for 250 lines):
  A. Outline (characters / question / answer / key_words) + Beat sheet
     (per-beat plan: lines / event / question_status / char_c_speaks)
  B. True multi-turn session: beats written in groups (<=30 lines per call),
     full history kept in messages; exact line budgets with in-place
     corrective turns (multi-turn advantage: short follow-up fixes the batch)
  C. Metadata (narration + youtube fields) with target validation + retry
  D. Programmatic quality gate (quality_gate.py): structure / fields /
     naturalness / duplicates / story / translation / metadata
  E. LLM critique (story judge + language judge) -> targeted patch rewrites
     (line-count preserving), loop until gate passes or QUEST_QA_MAX_ROUNDS.

Env knobs: QUEST_BEAT_LINES (default 10), QUEST_QA_MAX_ROUNDS (default 3).
Reuses _chat, _extract_json from parent llm_client.
"""
import json
import math
import os
import sys
import time
from pathlib import Path

_PARENT = str(Path(__file__).parent.parent.resolve())
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from llm_client import (
    _chat,
    _extract_json,
    _load_used_listening_summaries,
    _build_character_override_prompt,
    _get_character_overrides,
)

try:
    from quest.quality_gate import run_quality_gate, format_report
except ImportError:
    from quality_gate import run_quality_gate, format_report


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


# ---------------------------------------------------------------------------
# Public entry — orchestrates phases A-E
# ---------------------------------------------------------------------------

def generate_quest_script(topic: str, cefr: str = "A1",
                          lessons_dir: str = None,
                          num_lines: int = 48) -> dict:
    """Generate a quest lesson script via multi-round LLM calls."""
    n_buildup, n_core, n_reveal, n_review = split_phase_lines(num_lines)
    used_summaries = _load_used_listening_summaries(lessons_dir)
    beat_lines = _env_int("QUEST_BEAT_LINES", 10)
    qa_rounds = _env_int("QUEST_QA_MAX_ROUNDS", 3)

    # ── Phase A: outline + beat sheet ───────────────────────────────────
    print("  [LLM] Phase A: story outline...")
    outline_prompt = _build_outline_prompt(topic, cefr, used_summaries)
    outline = _chat_and_parse(
        outline_prompt, temperature=0.9, max_tokens=4096, reasoning_effort="medium",
        label="outline",
        system="You are an expert ESL video director designing slow-listening stories for overseas Chinese beginners. Output valid JSON only.")

    print(f"  [LLM] Phase A: beat sheet (beat_lines={beat_lines})...")
    beats = _generate_beat_sheet(
        outline, n_buildup, n_core, n_reveal, n_review, beat_lines)
    total_beats = sum(len(v) for v in beats.values())
    print(f"    beats: " + ", ".join(f"{ph}={len(beats[ph])}" for ph in beats))

    # ── Phase B: multi-turn dialogue session ────────────────────────────
    print("  [LLM] Phase B: multi-turn dialogue session...")
    all_dialogue = _generate_dialogue_session(outline, beats, cefr)
    print(f"    generated {len(all_dialogue)} dialogue lines")

    # Per-line defaults + zh fallback
    for line in all_dialogue:
        if not line.get("on_screen"):
            line["on_screen"] = [line.get("speaker", "char_a")]
    empty_zh = [(i, d.get("text", "")) for i, d in enumerate(all_dialogue)
                if not str(d.get("zh", "")).strip()]
    if empty_zh:
        print(f"  [LLM] {len(empty_zh)} lines have empty zh, batch translating...")
        zh_map = _batch_translate_zh(empty_zh)
        for i, zh in zh_map.items():
            all_dialogue[i]["zh"] = zh

    # ── Phase C: metadata with target validation ────────────────────────
    print("  [LLM] Phase C: narration + metadata...")
    meta = _generate_metadata_validated(topic, cefr, outline, all_dialogue)

    # ── Assemble final script ───────────────────────────────────────────
    script = {
        "lesson_type": "listening",
        "title": meta.get("title", topic.upper()),
        "cefr": cefr,
        "title_zh": meta.get("title_zh", ""),
        "scene_zh": meta.get("scene_zh", ""),
        "story_hook": outline.get("story_concept", ""),
        "story_concept": outline.get("story_concept", ""),
        "answer_en": outline.get("answer_en", ""),
        "common_misconception_en": outline.get("common_misconception_en", ""),
        "intro_zh": meta.get("intro_zh", ""),
        "welcome_en": meta.get("welcome_en", "Welcome to English listening channel."),
        "welcome_zh": meta.get("welcome_zh", "歡迎來到英文聽力頻道。"),
        "hook_intro_en": meta.get("hook_intro_en", ""),
        "hook_intro_zh": meta.get("hook_intro_zh", ""),
        "listening_question_en": outline.get("listening_question_en", ""),
        "listening_question_zh": outline.get("listening_question_zh", ""),
        "key_words": outline.get("key_words", []),
        "outro": meta.get("outro", "That's all for today. Keep practicing!"),
        "outro_zh": meta.get("outro_zh", ""),
        "practice_intro_en": "Now let's practice. Listen and repeat each sentence.",
        "practice_intro_zh": "現在來練習。請跟著朗讀每一句。",
        "char_a_description": outline.get("char_a_description", ""),
        "char_b_description": outline.get("char_b_description", ""),
        "char_c_description": outline.get("char_c_description", ""),
        "char_a_gender": outline.get("char_a_gender", "male"),
        "char_b_gender": outline.get("char_b_gender", "female"),
        "char_c_gender": outline.get("char_c_gender", "female"),
        "char_a_role": outline.get("char_a_role", ""),
        "char_b_role": outline.get("char_b_role", ""),
        "char_c_role": outline.get("char_c_role", "staff"),
        "char_a_personality": outline.get("char_a_personality", "curious and energetic"),
        "char_b_personality": outline.get("char_b_personality", "calm and knowledgeable"),
        "host_description": outline.get("host_description", ""),
        "host_gender": outline.get("host_gender", ""),
        "host_bg_prompt": meta.get("host_bg_prompt", "a bright modern TV studio set, no people"),
        "youtube_title": meta.get("youtube_title", ""),
        "youtube_title_en": meta.get("youtube_title_en", ""),
        "youtube_description": meta.get("youtube_description", ""),
        "youtube_description_en": meta.get("youtube_description_en", ""),
        "youtube_tags": meta.get("youtube_tags", []),
        "thumbnail_prompt": meta.get("thumbnail_prompt", ""),
        "scene": outline.get("scene", topic.lower()),
        "scene_images": meta.get("scene_images", []),
        "thumbnail_expression": meta.get("thumbnail_expression", "surprised and excited"),
        "thumbnail_action": meta.get("thumbnail_action", "gesturing naturally"),
        "thumbnail_subtitle": meta.get("thumbnail_subtitle", "慢速聽力"),
        "thumbnail_icons": meta.get("thumbnail_icons", []),
        "dialogue": all_dialogue,
        "_requested_num_lines": num_lines,
    }

    # Ensure per-line phase/speaker defaults (beat generation already sets them)
    for i, line in enumerate(script["dialogue"]):
        line.setdefault("zh", "")
        line.setdefault("on_screen", [line.get("speaker", "char_a")])
        if not line.get("on_screen"):
            line["on_screen"] = [line.get("speaker", "char_a")]
        if not line.get("phase"):
            if i < n_buildup:
                line["phase"] = "buildup"
            elif i < n_buildup + n_core:
                line["phase"] = "core"
            elif i < n_buildup + n_core + n_reveal:
                line["phase"] = "reveal"
            else:
                line["phase"] = "review"

    # Force-override character genders from CHARACTER_OVERRIDES (set by
    # pipeline_service when using character reuse / library / fixes).
    overrides = _get_character_overrides()
    for key in ("char_a", "char_b", "char_c", "host"):
        if key in overrides:
            gender = overrides[key].get("gender", "")
            if gender:
                script[f"{key}_gender"] = gender
                print(f"  [LLM] Override {key}_gender = {gender}")

    # ── Phase D+E: quality gate + critique/repair loop ──────────────────
    _apply_programmatic_fixes(script, num_lines)
    rounds_log = []
    had_judge_issues = True  # run judges at least once (round 1)
    for rnd in range(1, qa_rounds + 1):
        report = run_quality_gate(script, num_lines)
        judge_issues = []
        if report["n_errors"] or had_judge_issues:
            print(f"  [QA] Round {rnd}: gate errors={report['n_errors']} "
                  f"warnings={report['n_warnings']}, running LLM judges...")
            judge_issues = _critique_script(script, report)
            had_judge_issues = bool(judge_issues)
        else:
            had_judge_issues = False
        rounds_log.append({
            "round": rnd,
            "gate": report,
            "judge_issues": judge_issues,
        })
        if report["passed"] and not judge_issues:
            print(f"  [QA] Round {rnd}: PASSED (0 errors, judges clean)")
            break
        patches = _merge_patches(report, judge_issues, script)
        if not patches:
            print(f"  [QA] Round {rnd}: no actionable patches, accepting best effort")
            break
        print(f"  [QA] Round {rnd}: repairing {len(patches)} patches: "
              + ", ".join(f"L{p[0]}-{p[1]}" for p in patches))
        _repair_patches(script, patches, outline, cefr)
        _apply_programmatic_fixes(script, num_lines)

    final = run_quality_gate(script, num_lines)
    script["_qa"] = {"rounds": rounds_log, "final": final}
    print(format_report(final))
    if not final["passed"]:
        print(f"  [QA] WARNING: {final['n_errors']} errors remain after "
              f"{qa_rounds} QA rounds (accepted best effort)")
    return script


# ---------------------------------------------------------------------------
# Line splitting + shared chat helper
# ---------------------------------------------------------------------------

def split_phase_lines(num_lines: int) -> tuple[int, int, int, int]:
    """Split total lines into (buildup, core, reveal, review) by ratio ~30/48/10/12.

    Mirrors the bubble-tea reference video (~18min, 213 lines): long buildup
    where the listening question arises naturally, longest core with mixed
    speakers (education + transactions), a short reveal, and a review.
    48 -> (14, 23, 5, 6).  250 -> (75, 120, 25, 30).
    """
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


def _chat_and_parse(prompt: str = "", temperature: float = 0.8,
                    max_tokens: int = 8192, reasoning_effort: str = "low",
                    label: str = "", system: str = "",
                    messages: list[dict] | None = None,
                    retries: int | None = None) -> dict:
    """Call _chat, extract JSON, retry on failure.

    Either pass (prompt, system) or a pre-built messages list.
    Retry count: explicit retries arg > LLM_RETRIES env (default 10).
    """
    last_error = None
    max_retries = retries if retries is not None else _env_int("LLM_RETRIES", 10)
    if messages is None:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
    for attempt in range(max_retries):
        try:
            content = _chat(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                reasoning_effort=reasoning_effort,
            )
            return _extract_json(content)
        except (json.JSONDecodeError, RuntimeError, KeyError) as e:
            last_error = e
            print(f"  [LLM {label} retry {attempt+1}/{max_retries}] "
                  f"{type(e).__name__}: {str(e)[:600]}")
            if attempt < max_retries - 1:
                time.sleep(5)
    raise RuntimeError(f"LLM {label} failed after {max_retries} retries: {last_error}")


# ---------------------------------------------------------------------------
# Phase A: beat sheet
# ---------------------------------------------------------------------------

def _allocate_beats(n_lines: int, beat_lines: int) -> list[int]:
    """Deterministically split n_lines into beat budgets (each 1..beat_lines)."""
    if n_lines <= 0:
        return []
    n_beats = max(1, math.ceil(n_lines / max(1, beat_lines)))
    base, rem = divmod(n_lines, n_beats)
    return [base + 1 if i < rem else base for i in range(n_beats)]


def _generate_beat_sheet(outline: dict, n_buildup: int, n_core: int,
                         n_reveal: int, n_review: int,
                         beat_lines: int) -> dict[str, list[dict]]:
    """Generate per-phase beat plans via chained per-phase LLM calls.

    Smaller requests avoid empty-content failures on large outputs; each
    phase sees the previous phase's beat events for continuity. A phase that
    fails falls back to programmatic beats built from its summary.
    Beat budgets are always enforced programmatically.
    """
    alloc = {
        "buildup": _allocate_beats(n_buildup, beat_lines),
        "core": _allocate_beats(n_core, beat_lines),
        "reveal": _allocate_beats(n_reveal, beat_lines),
        "review": _allocate_beats(n_review, beat_lines),
    }
    beats: dict[str, list[dict]] = {}
    prev_events: list[str] = []
    for ph in ("buildup", "core", "reveal", "review"):
        prompt = _build_beat_sheet_prompt(outline, ph, alloc, prev_events)
        raw: list = []
        try:
            result = _chat_and_parse(
                prompt, temperature=0.7, max_tokens=4096, reasoning_effort="low",
                label=f"beat_{ph}", retries=4,
                system="You are a story architect planning ESL slow-listening video scripts. Output valid JSON only.")
            if isinstance(result, dict):
                raw = result.get(ph, [])
            elif isinstance(result, list):
                raw = result
            raw = [b for b in raw if isinstance(b, dict)]
        except RuntimeError as e:
            print(f"  [LLM] Beat sheet '{ph}' failed ({str(e)[:600]}); "
                  f"using programmatic fallback")

        budgets = alloc[ph]
        # Force beat count == len(budgets): trim or pad with generic beats
        while len(raw) > len(budgets):
            raw.pop()
        while len(raw) < len(budgets):
            raw.append({"event": f"{ph} beat {len(raw)+1}: "
                        + outline.get(f"{ph}_summary", ph)[:120]})
        # Force per-beat line budgets (LLM "lines" values not trusted)
        for b, lines in zip(raw, budgets):
            b["lines"] = lines
            b["phase"] = ph
            b.setdefault("event", outline.get(f"{ph}_summary", ph))
            b.setdefault("location", outline.get("scene", ""))
            b.setdefault("emotion", "warm and curious")
            b.setdefault("key_words", [])
            b.setdefault("char_c_speaks", False)
        # question_status clamps: answer only allowed in reveal/review
        for i, b in enumerate(raw):
            qs = str(b.get("question_status", "")).lower()
            if ph in ("buildup", "core"):
                if i == 0 and ph == "buildup" and not qs:
                    qs = "raised"
                if qs == "answered":
                    qs = "hint"
            elif ph == "reveal":
                qs = "answered" if i == 0 else (qs or "confirmed")
            else:  # review
                qs = qs or "confirmed"
            b["question_status"] = qs
        # buildup must contain exactly one "raised" beat (early)
        if ph == "buildup" and raw:
            raised = [i for i, b in enumerate(raw) if b["question_status"] == "raised"]
            if not raised:
                raw[0]["question_status"] = "raised"
            elif len(raised) > 1:
                for i in raised[1:]:
                    raw[i]["question_status"] = "deferred"
        # char_c must speak in core at least every 3rd beat
        if ph == "core":
            flags = [bool(b.get("char_c_speaks")) for b in raw]
            if not any(flags):
                for i in range(0, len(raw), 3):
                    raw[i]["char_c_speaks"] = True
        beats[ph] = raw
        prev_events = [f"{ph[0]}{i+1}: {b.get('event', '')}" for i, b in enumerate(raw)]
    return beats


def _build_beat_sheet_prompt(outline: dict, phase: str, alloc: dict,
                             prev_events: list[str]) -> str:
    budgets = alloc[phase]
    budget_desc = ", ".join(f"beat{i+1}={n} lines" for i, n in enumerate(budgets))
    prev_hint = ""
    if prev_events:
        prev_hint = ("\nBeats already planned for EARLIER phases (continue this "
                     "story thread, never re-introduce settled topics):\n  "
                     + "\n  ".join(prev_events) + "\n")
    phase_rules = {
        "buildup": ('This phase introduces the story: friends discuss going to the place. '
                    'Exactly ONE beat (early, beats 1-3) is question_status "raised" — the listening '
                    'question comes up naturally and the answer is withheld. Others are "deferred"/"hint".'),
        "core": ('This phase: teaching + transaction with staff (char_c). question_status ONLY '
                 '"deferred"/"hint" — the answer must NOT appear. Mark char_c_speaks=true on at '
                 'least 1 of every 3 consecutive beats.'),
        "reveal": ('This phase reveals the ANSWER. Beat 1 question_status "answered", the rest '
                   '"confirmed". Include a surprised reaction to the common misconception.'),
        "review": ('This phase confirms the answer naturally (asked again, answered correctly), '
                   'positive emotions, mention coming back, natural goodbye. question_status "confirmed".'),
    }
    return f"""Plan the beats for the "{phase}" phase of an ESL slow-listening story.

Story concept: {outline.get("story_concept", "")}
Listening question: {outline.get("listening_question_en", "")}
True answer (revealed ONLY in reveal phase): {outline.get("answer_en", "")}
Common misconception: {outline.get("common_misconception_en", "")}
Scene: {outline.get("scene", "")}
Key words: {", ".join(w.get("en", "") for w in outline.get("key_words", []))}
Phase summary: {outline.get(f"{phase}_summary", "")}
{prev_hint}
EXACT line budgets (do NOT change): {budget_desc}

RULES:
1. Each beat is ONE story event at ONE location. Events must be progressive and
   distinct — never repeat or re-introduce an earlier topic.
2. {phase_rules[phase]}
3. Distribute key_words across beats (each used 2-3 times overall).
4. Each beat: 1-2 sentence "event" + emotional tone.

Output JSON ONLY (line counts will be enforced by us):
{{"{phase}": [{{"beat_id": "{phase[0]}1", "location": "...", "event": "...", "emotion": "...", "key_words": ["..."], "question_status": "...", "char_c_speaks": false}}]}}"""


# ---------------------------------------------------------------------------
# Phase B: multi-turn dialogue session
# ---------------------------------------------------------------------------

_PHASE_RULES = {
    "buildup": (
        'ONLY char_a and char_b speak. They discuss going to the place.\n'
        'The LISTENING QUESTION arises naturally — one character is curious, the other says "you\'ll see" or "I\'ll tell you later".\n'
        'Do NOT reveal the answer. The question should arise naturally but NO answer is given.\n'
        'Use at least 2 key_words. Include filler words ("well", "you know", "hmm", "actually").'
    ),
    "core": (
        'char_a, char_b, AND char_c all speak. Include BOTH teaching (explaining options/menu/process) AND transaction (ordering, price, payment).\n'
        'char_c (staff) must appear at least once per 5 lines.\n'
        'Do NOT reveal the answer to the listening question. Characters discuss the topic but the answer remains unknown.\n'
        'Include a small surprise or interesting detail.\n'
        'Use at least 3 key_words.'
    ),
    "reveal": (
        'ONLY char_a and char_b. One friend reveals the ANSWER to the listening question.\n'
        'The other is surprised ("Really? I thought...").\n'
        'Use simple English to explain the fun fact. Use at least 2 key_words.'
    ),
    "review": (
        'ONLY char_a and char_b. Confirm the answer (one asks again, other answers correctly).\n'
        'Reuse key_words in new sentences. Express positive emotions.\n'
        'Mention coming back. End with natural goodbye.'
    ),
}


_DIALOGUE_QUALITY_RULES = """\
NATURAL DIALOGUE QUALITY RULES (MANDATORY — the #1 priority):

1. SENTENCE LENGTH VARIETY: Mix short reactions (3-5 words) with medium (6-10) and longer (11-18). NEVER write 3 consecutive lines that are all 4-6 words — it sounds robotic.

2. FILLER WORDS (at least 1 per 3-4 lines): "well", "you know", "hmm", "actually", "oh", "I mean", "let me think", "you see", "sort of", "kind of"

3. BACK-CHANNELING (every 5-8 lines): "Oh really?", "That makes sense", "Hmm, interesting", "Wow", "I see", "Right", "Exactly", "Nice"

4. SAME SPEAKER CAN SPEAK CONSECUTIVELY (2-3 lines): Sometimes a character explains something across 2 lines without interruption. Do this naturally.

5. EMOTIONAL VARIETY: Express curiosity, excitement, nervousness, surprise, satisfaction, humor. Characters are NOT flat.

6. NATURAL FLOW: Avoid the Q&A pattern (ask→answer→ask→answer). Characters can change topic, remember something, express doubt, make a joke, share personal experience.

7. CHARACTER PERSONALITY: char_a is curious/energetic (asks follow-ups, gets excited), char_b is calm/knowledgeable (explains patiently, teases playfully), char_c is helpful/friendly.

8. AVOID REPETITIVE PATTERNS: Don't start consecutive lines with "Yes," or "No,". Don't repeat the same sentence structure.

9. STORY CONTINUITY: This is an ONGOING conversation. Never restart a topic that earlier beats already covered. The listening question has ALREADY been raised — do not re-ask it as if new.

GOOD dialogue (from reference video — natural, varied):
  char_a: "Mina, do you want to try something fun after class today?"   (10 words)
  char_b: "Sure. What is it?"                                            (3 words, short reaction)
  char_a: "Bubble tea. There is a new shop close to the campus."         (10 words)
  char_b: "Bubble tea? I have never had that before."                    (7 words, surprised)
  char_a: "Really? You are going to love it."                            (6 words, enthusiastic)
  char_b: "What is it exactly?"                                         (4 words, curious follow-up)
  char_a: "It is a cold tea drink. You mix it with milk, sugar, and other things."  (14 words, explanatory)
  char_b: "And what about the bubble part? Are there real bubbles in it?"            (11 words, questioning)
  char_a: "That's a good question. Let me ask you first. Why do you think it's called bubble tea?"  (17 words, playfully deflects)

BAD dialogue (DO NOT write like this — too robotic):
  char_a: "Hi, Sarah. How are you?"     (boring greeting, textbook-style)
  char_b: "I am good. You?"             (too short, no personality)
  char_a: "I am very thirsty."          (robotic statement)
  char_b: "Let's get a drink then."     (textbook transition)
"""

_WRITER_SYSTEM = (
    "You are an expert ESL dialogue writer. Write natural conversational "
    "English for slow-listening videos. Output valid JSON only.")


def _phase_temperature(phase: str) -> float:
    return {"buildup": 0.85, "core": 0.7, "reveal": 0.8, "review": 0.75}.get(phase, 0.8)


def _pack_groups(phase_beats: list[dict], max_lines: int = 30) -> list[list[dict]]:
    """Greedily pack consecutive beats of ONE phase into groups (<=max_lines)."""
    groups, cur, cur_lines = [], [], 0
    for b in phase_beats:
        if cur and cur_lines + b["lines"] > max_lines:
            groups.append(cur)
            cur, cur_lines = [], 0
        cur.append(b)
        cur_lines += b["lines"]
    if cur:
        groups.append(cur)
    return groups


def _generate_dialogue_session(outline: dict, beats: dict, cefr: str) -> list[dict]:
    """True multi-turn session: full history persists in messages."""
    groups: list[list[dict]] = []
    for ph in ("buildup", "core", "reveal", "review"):
        groups.extend(_pack_groups(beats.get(ph, [])))

    messages: list[dict] = [{"role": "system", "content": _WRITER_SYSTEM}]
    all_lines: list[dict] = []
    for gi, group in enumerate(groups):
        phases = sorted({b["phase"] for b in group})
        budget = sum(b["lines"] for b in group)
        prompt = _build_group_prompt(outline, group, cefr, all_lines,
                                     first=(gi == 0))
        messages.append({"role": "user", "content": prompt})
        lines = _session_turn(messages, budget,
                              temperature=_phase_temperature(phases[0]))
        _post_validate_lines(lines, group, all_lines)
        all_lines.extend(lines)
        print(f"    group {gi+1}/{len(groups)} [{'+'.join(phases)}]: "
              f"{len(lines)}/{budget} lines")
    return all_lines


def _build_group_prompt(outline: dict, group: list[dict], cefr: str,
                        all_lines: list[dict], first: bool) -> str:
    import json as _json
    phases = sorted({b["phase"] for b in group})
    question = outline.get("listening_question_en", "")
    scene = outline.get("scene", "")
    key_words = ", ".join(w.get("en", "") for w in outline.get("key_words", []))
    budget = sum(b["lines"] for b in group)

    head = ""
    if first:
        head = f"""Story: {outline.get("story_concept", "")}
Scene: {scene}
Characters:
- char_a: {outline.get("char_a_description", "")} — personality: {outline.get("char_a_personality", "")}
- char_b: {outline.get("char_b_description", "")} — personality: {outline.get("char_b_personality", "")}
- char_c: {outline.get("char_c_description", "")} (staff, speaks ONLY in core phase)
Key words: {key_words}

{_DIALOGUE_QUALITY_RULES}
"""
    else:
        head = "Continue the same conversation.\n"

    prev_hint = ""
    if all_lines:
        tail = all_lines[-5:]
        prev_hint = ("Last lines written (continue seamlessly from here, "
                     "same scene and story):\n"
                     + "\n".join(f"  {l.get('speaker','?')}: {l.get('text','')}"
                                 for l in tail) + "\n")

    beats_desc = ""
    for b in group:
        beats_desc += (f"- {b.get('beat_id','?')} ({b['lines']} lines) @ "
                       f"{b.get('location', scene)}: {b.get('event','')} "
                       f"[emotion: {b.get('emotion','')}; question_status: "
                       f"{b.get('question_status','')}"
                       + ("; char_c speaks" if b.get("char_c_speaks") else "")
                       + (f"; key_words: {', '.join(b.get('key_words', []))}"
                          if b.get("key_words") else "") + "]\n")

    rules = "\n".join(f"PHASE {ph} RULES:\n{_PHASE_RULES[ph]}" for ph in phases)

    answer_block = ""
    if "reveal" in phases:
        answer_block = (f"\nThe ANSWER to reveal NOW: {outline.get('answer_en', '')}\n"
                        f"Common misconception to correct: "
                        f"{outline.get('common_misconception_en', '')}\n")
    else:
        answer_block = ("\nDo NOT reveal or state the answer to the listening "
                        "question — it is revealed later.\n")

    kw_hint = ""
    if "core" in phases:
        kw_hint = ('on_screen rules:\n'
                   '- mix of ["char_a","char_c"], ["char_b","char_c"], ["char_a","char_b"]\n'
                   '- Use [] (empty, environment/menu/object shot) for 1-2 lines in this batch\n')
    else:
        kw_hint = ('on_screen rules:\n'
                   '- mostly ["char_a","char_b"]\n'
                   '- Use [] (empty) for at most 1 line (environment shot)\n')

    phase_field = phases[0]
    return f"""{head}{prev_hint}
Write the next beats of dialogue ({budget} lines total):
{beats_desc}
Listening question (already raised earlier): {question}
{answer_block}
CEFR {cefr} level. Each line 5-15 words, vary length naturally.
{rules}

{kw_hint}
Output: JSON array of EXACTLY {budget} objects:
[{{"speaker":"char_a","text":"English sentence","phase":"{phase_field}","zh":"繁體中文翻譯","on_screen":["char_a","char_b"]}}]
Every line MUST include "zh" (Traditional Chinese 繁體中文) and "on_screen".
NO markdown, NO explanation. JSON array ONLY."""


def _session_turn(messages: list[dict], budget: int,
                  temperature: float = 0.8) -> list[dict]:
    """One session turn: append user msg (caller), call LLM, verify count.

    On mismatch, sends a corrective follow-up turn (multi-turn advantage).
    Returns the best list of line dicts.
    """
    content = _chat(messages, temperature=temperature, max_tokens=8192,
                    reasoning_effort="low")
    lines = _as_line_list(_safe_extract(content))

    for attempt in range(2):
        if lines and len(lines) == budget:
            messages.append({"role": "assistant", "content": content})
            return lines
        got = len(lines) if isinstance(lines, list) else 0
        messages.append({"role": "assistant", "content": content})
        messages.append({
            "role": "user",
            "content": (f"Your reply contained {got} dialogue lines but the "
                        f"budget is EXACTLY {budget}. Rewrite this batch and "
                        f"return a JSON array of exactly {budget} line objects. "
                        "Same fields (speaker/text/phase/zh/on_screen). "
                        "JSON array ONLY.")})
        content = _chat(messages, temperature=temperature, max_tokens=8192,
                        reasoning_effort="low")
        lines = _as_line_list(_safe_extract(content))

    messages.append({"role": "assistant", "content": content})
    if not isinstance(lines, list):
        raise RuntimeError("Dialogue session turn failed to produce JSON array")
    return lines


def _safe_extract(content: str):
    try:
        return _extract_json(content)
    except Exception:
        return None


def _as_line_list(result) -> list[dict] | None:
    if isinstance(result, list):
        return [x for x in result if isinstance(x, dict)]
    if isinstance(result, dict) and isinstance(result.get("dialogue"), list):
        return [x for x in result["dialogue"] if isinstance(x, dict)]
    return None


def _post_validate_lines(lines: list[dict], group: list[dict],
                         all_lines: list[dict]):
    """Force beat-allocated phase, clamp speakers, fill on_screen defaults."""
    phases = [b["phase"] for b in group]
    for i, line in enumerate(lines):
        # phase follows the beat allocation (position -> beat budget)
        acc = 0
        for b in group:
            acc += b["lines"]
            if i < acc:
                line["phase"] = b["phase"]
                break
        else:
            line["phase"] = phases[-1]
        # speaker clamps: char_c only in core; unknown -> alternate
        speaker = line.get("speaker", "")
        if speaker not in ("char_a", "char_b", "char_c"):
            speaker = "char_a" if (len(all_lines) + i) % 2 == 0 else "char_b"
            line["speaker"] = speaker
        if speaker == "char_c" and line["phase"] != "core":
            line["speaker"] = "char_b"
        if not line.get("on_screen"):
            line["on_screen"] = [line["speaker"]]
        if not str(line.get("zh", "")).strip():
            line["zh"] = ""


# ---------------------------------------------------------------------------
# Phase C: metadata (narration + youtube fields) with target validation
# ---------------------------------------------------------------------------

_META_TARGETS = {
    "hook_intro_en": (70, 110),
    "outro": (80, 110),
}


def _build_metadata_prompt(topic: str, cefr: str, outline: dict, dialogue: list[dict]) -> str:
    import json
    from style_manager import get_active_style_prompt, get_active_thumbnail_hint
    style_prompt = get_active_style_prompt()
    thumb_hint = get_active_thumbnail_hint()
    dialogue_text = json.dumps(
        [{"s": d.get("speaker",""), "t": d.get("text",""), "p": d.get("phase","")}
         for d in dialogue],
        ensure_ascii=False, separators=(",",":"))
    return f"""You are a YouTube content strategist. Generate narration and metadata for this ESL video.

Topic: {topic} ({cefr})
Listening question: {outline.get("listening_question_en","")}
Answer: {outline.get("answer_en","")}
Scene: {outline.get("scene","")}
Characters: {outline.get("char_a_description","")} / {outline.get("char_b_description","")} / {outline.get("char_c_description","")}
Host: {outline.get("host_description","")}

Full dialogue:
{dialogue_text}

Generate JSON with these fields:
{{
  "title": "ENGLISH TITLE (e.g. AT THE COFFEE SHOP)",
  "title_zh": "繁中短标题 (max 6 chars)",
  "scene_zh": "繁中場景描述",
  "intro_zh": "繁中 story hook translation",
  "welcome_en": "short channel welcome (max 8 words)",
  "welcome_zh": "繁中",
  "hook_intro_en": "narrator opening 70-110 words. Must include: greeting, topic intro, the listening question, and a CTA. Vary the structure — don't always start with greeting.",
  "hook_intro_zh": "繁中",
  "outro": "narrator closing 80-110 words: how was it→repeat question→comment CTA→channel description→subscribe→bye",
  "outro_zh": "繁中",
  "host_bg_prompt": "TV studio background prompt (no people)",
  "youtube_title": "高CTR繁中标题 with 【】and ｜ format. Pattern D: 【英文聽力挑戰】{{emoji}}{{topic 繁中}}｜❓你能聽出答案嗎？｜{{CEFR}}慢速英文｜不用背多聽就會用｜英文聽力訓練｜{{English topic}}",
  "youtube_title_en": "high-CTR PURE ENGLISH title (no Chinese). Include topic + key skill + compelling hook. Max 100 chars. Example: Can You Guess Why It's Called Bubble Tea? ☕ Slow English Listening",
  "youtube_description": "full 繁中 description with chapters + key_words",
  "youtube_description_en": "full PURE ENGLISH description. Include chapters section, key_words, hashtags, and subscribe CTA.",
  "youtube_tags": ["tag1","tag2",... 15-20 tags],
  "thumbnail_prompt": "thumbnail image prompt ({thumb_hint} style)",
  "thumbnail_expression": "main character expression",
  "thumbnail_action": "main character action",
  "thumbnail_subtitle": "繁中 short subtitle",
  "thumbnail_icons": [{{"en":"word","zh":"繁中"}}, ... 4-5 icons],
  "scene_images": [{{"prompt":"specific scene description with details (counter, menu board, equipment, decor), 16:9, no people","label":"short English label"}}]
}}

VISUAL STYLE (CRITICAL): The video's art style is: "{style_prompt}". EVERY host_bg_prompt, thumbnail_prompt, and scene_images prompt MUST include this EXACT style descriptor phrase. Do NOT mix other art styles.

IMPORTANT: Generate at least 8 scene_images covering different angles and details of the scene (e.g. exterior, counter, menu board, equipment, seating area, product close-up, kitchen, decoration).

JSON ONLY, no markdown."""


def _generate_metadata_validated(topic: str, cefr: str, outline: dict,
                                 dialogue: list[dict]) -> dict:
    """Round 3 with word-count/tag-count validation + one targeted retry."""
    meta = _chat_and_parse(
        _build_metadata_prompt(topic, cefr, outline, dialogue),
        temperature=0.6, max_tokens=8192, reasoning_effort="low",
        label="metadata",
        system="You are a YouTube content strategist for ESL learning videos. Output valid JSON only.")

    # Validate targets, retry once for failing fields
    failing = _meta_failing_fields(meta)
    if failing:
        print(f"  [LLM] Metadata fields below target, retrying: {failing}")
        try:
            fixes = _chat_and_parse(
                _build_metadata_fix_prompt(meta, failing, topic, cefr),
                temperature=0.6, max_tokens=4096, reasoning_effort="low",
                label="metadata_fix",
                system="You are a YouTube content strategist. Output valid JSON only.")
            if isinstance(fixes, dict):
                for k, v in fixes.items():
                    if v:
                        meta[k] = v
        except RuntimeError as e:
            print(f"  [LLM] Metadata fix failed: {e}")

    _pad_metadata(meta, outline, topic, cefr)
    return meta


def _meta_failing_fields(meta: dict) -> list[str]:
    failing = []
    for field, (lo, hi) in _META_TARGETS.items():
        n = len(str(meta.get(field, "")).split())
        if not (lo <= n <= hi):
            failing.append(field)
    tags = meta.get("youtube_tags", [])
    if isinstance(tags, list) and not (15 <= len(tags) <= 20):
        failing.append("youtube_tags")
    icons = meta.get("thumbnail_icons", [])
    if isinstance(icons, list) and not (4 <= len(icons) <= 5):
        failing.append("thumbnail_icons")
    scenes = meta.get("scene_images", [])
    if not isinstance(scenes, list) or len(scenes) < 8:
        failing.append("scene_images")
    return failing


def _build_metadata_fix_prompt(meta: dict, failing: list[str],
                               topic: str, cefr: str) -> str:
    import json
    hints = {
        "hook_intro_en": "narrator opening, 70-110 words",
        "outro": "narrator closing, 80-110 words",
        "youtube_tags": "15-20 youtube tags (mix 繁中/English)",
        "thumbnail_icons": "4-5 icons",
        "scene_images": "8-12 scene images",
    }
    spec = "\n".join(f'- "{f}": {hints.get(f, "")} (current value: {json.dumps(meta.get(f), ensure_ascii=False)[:200]})'
                     for f in failing)
    return f"""These metadata fields are below spec. Regenerate ONLY these fields.

Topic: {topic} ({cefr})
Fields to fix:
{spec}

Output JSON with ONLY the failing field names as keys, corrected values.
JSON ONLY, no markdown."""


def _pad_metadata(meta: dict, outline: dict, topic: str, cefr: str):
    """Programmatic padding for tags/icons when LLM retries still fall short."""
    tags = meta.get("youtube_tags")
    if isinstance(tags, list):
        extras = []
        kw_ens = [w.get("en", "") for w in outline.get("key_words", []) if w.get("en")]
        for kw in kw_ens:
            extras.append(f"{kw} english")
        extras += [f"{cefr} english listening", "slow english", "esl listening",
                   "english for beginners", f"{topic.lower()}"]
        for e in extras:
            if len(tags) >= 20:
                break
            if e.lower() not in [t.lower() for t in tags]:
                tags.append(e)
    icons = meta.get("thumbnail_icons")
    if isinstance(icons, list):
        for w in outline.get("key_words", []):
            if len(icons) >= 5:
                break
            en, zh = w.get("en", ""), w.get("zh", "")
            if en and zh and en not in [i.get("en") for i in icons if isinstance(i, dict)]:
                icons.append({"en": en, "zh": zh})


# ---------------------------------------------------------------------------
# Programmatic fixes (no LLM)
# ---------------------------------------------------------------------------

def _apply_programmatic_fixes(script: dict, num_lines: int):
    """Deterministic repairs before each gate run: on_screen normalize,
    zh simplified->traditional, speaker clamps, exact line counts."""
    dialogue = script.get("dialogue", [])
    for line in dialogue:
        speaker = line.get("speaker", "")
        if speaker not in ("char_a", "char_b", "char_c"):
            line["speaker"] = "char_a"
        elif speaker == "char_c" and line.get("phase") != "core":
            line["speaker"] = "char_b"
        os_ = line.get("on_screen")
        if isinstance(os_, list) and os_:
            seen = []
            for k in ("char_a", "char_b", "char_c"):
                if k in os_ and k not in seen:
                    seen.append(k)
            line["on_screen"] = seen or [line["speaker"]]
        elif not os_:
            line["on_screen"] = [] if os_ == [] else [line["speaker"]]

    _fix_zh_traditional(script)
    _ensure_env_shots(script, num_lines)
    _enforce_line_counts(script, num_lines)


def _ensure_env_shots(script: dict, num_lines: int):
    """LLM 很少输出空 on_screen；不足下限时从各阶段中段程序化补齐。

    纯视觉决策（画面拍环境而非人物），不影响台词/字幕/音频，避开每阶段
    首尾 3 行（故事衔接处）以降低风险。
    """
    dialogue = script.get("dialogue", [])
    env = sum(1 for l in dialogue if l.get("on_screen") == [])
    env_min = max(2, round(num_lines / 50)) if num_lines else 2
    if env >= env_min or len(dialogue) < 10:
        return
    need = env_min - env
    phases: dict[str, list[int]] = {}
    for i, l in enumerate(dialogue):
        phases.setdefault(l.get("phase", ""), []).append(i)
    candidates = []
    for idxs in phases.values():
        if len(idxs) > 8:
            candidates.extend(idxs[3:-3])
    if not candidates:
        return
    candidates.sort()
    step = max(1, len(candidates) // need)
    chosen = candidates[step // 2::step][:need]
    for i in chosen:
        dialogue[i]["on_screen"] = []
    print(f"  [Fix] padded {len(chosen)} env shots (on_screen=[])")


def _fix_zh_traditional(script: dict):
    """Convert simplified chars to Traditional Chinese via opencc (if available)."""
    try:
        from opencc import OpenCC
    except ImportError:
        return
    cc = OpenCC("s2t")

    def conv(s):
        return cc.convert(s) if isinstance(s, str) and s else s

    for field in ("title_zh", "scene_zh", "intro_zh", "welcome_zh",
                  "hook_intro_zh", "outro_zh", "youtube_title",
                  "youtube_description", "thumbnail_subtitle"):
        script[field] = conv(script.get(field, ""))
    for line in script.get("dialogue", []):
        line["zh"] = conv(line.get("zh", ""))
    for kw in script.get("key_words", []):
        if isinstance(kw, dict):
            kw["zh"] = conv(kw.get("zh", ""))
    for ic in script.get("thumbnail_icons", []):
        if isinstance(ic, dict):
            ic["zh"] = conv(ic.get("zh", ""))


def _enforce_line_counts(script: dict, num_lines: int):
    """Trim/expand dialogue to exactly num_lines with phase ratios intact.

    Trim: env shots first, then from tail of over-budget phases (keeping each
    phase's first 2 / last 2 lines so story joints stay intact).
    Expand: one LLM call inserting lines into the longest phase's middle.
    """
    dialogue = script.get("dialogue", [])
    if len(dialogue) == num_lines or num_lines <= 0:
        return
    ref = dict(zip(("buildup", "core", "reveal", "review"),
                   split_phase_lines(num_lines)))

    if len(dialogue) > num_lines:
        # count current phases
        counts = {"buildup": 0, "core": 0, "reveal": 0, "review": 0}
        for l in dialogue:
            if l.get("phase") in counts:
                counts[l["phase"]] += 1
        # candidate drops per over-budget phase
        drops: set[int] = set()
        for ph, target in ref.items():
            excess = counts[ph] - target
            if excess <= 0:
                continue
            idxs = [i for i, l in enumerate(dialogue) if l.get("phase") == ph]
            # protect first 2 and last 2 lines of the phase
            protected = set(idxs[:2] + idxs[-2:])
            # prefer env shots, then latest lines
            pool = [i for i in idxs if i not in protected]
            env_first = [i for i in pool if dialogue[i].get("on_screen") == []]
            rest = [i for i in pool if dialogue[i].get("on_screen") != []]
            for i in (env_first + rest[::-1])[:excess]:
                drops.add(i)
        if drops:
            script["dialogue"] = [l for i, l in enumerate(dialogue)
                                  if i not in drops]
            print(f"  [Fix] trimmed {len(drops)} lines to meet budget "
                  f"({len(script['dialogue'])}/{num_lines})")
    else:
        try:
            _expand_dialogue(script, num_lines, ref)
        except RuntimeError as e:
            print(f"  [Fix] line expansion failed: {e}")


def _expand_dialogue(script: dict, num_lines: int, ref: dict):
    """Insert missing lines into the middle of the most under-budget phase."""
    dialogue = script["dialogue"]
    counts = {"buildup": 0, "core": 0, "reveal": 0, "review": 0}
    for l in dialogue:
        if l.get("phase") in counts:
            counts[l["phase"]] += 1
    need = num_lines - len(dialogue)
    phase = max(ref, key=lambda p: ref[p] - counts[p])
    idxs = [i for i, l in enumerate(dialogue) if l.get("phase") == phase]
    at = idxs[len(idxs) // 2] if idxs else len(dialogue) // 2
    before = dialogue[max(0, at - 3):at]
    after = dialogue[at:at + 3]

    def fmt(ls):
        return "\n".join(f"  {l.get('speaker','?')}: {l.get('text','')}" for l in ls)

    phase_rule = _PHASE_RULES.get(phase, "")
    prompt = f"""Insert {need} more dialogue lines into the "{phase}" phase of this ongoing ESL conversation.

Lines BEFORE the insertion point:
{fmt(before)}

Lines AFTER the insertion point:
{fmt(after)}

Requirements:
- The new lines fit between them seamlessly (same scene, same topic flow).
- Phase rule: {phase_rule}
- Each line: {{"speaker","text","phase":"{phase}","zh":"繁體中文","on_screen":["char_a","char_b"]}}
- Exactly {need} lines. JSON array ONLY."""

    result = _chat_and_parse(
        prompt, temperature=0.75, max_tokens=4096, reasoning_effort="low",
        label=f"expand_{phase}", system=_WRITER_SYSTEM)
    lines = _as_line_list(result) or []
    lines = lines[:need]
    for l in lines:
        l["phase"] = phase
        if not l.get("on_screen"):
            l["on_screen"] = [l.get("speaker", "char_a")]
    dialogue[at:at] = lines
    print(f"  [Fix] expanded {len(lines)} lines in '{phase}' "
          f"({len(dialogue)}/{num_lines})")


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
        result = _chat_and_parse(
            prompt, temperature=0.3, max_tokens=4096, reasoning_effort="low",
            label="fallback_zh",
            system="You are a professional translator. Translate English to Traditional Chinese (繁體中文). Output valid JSON array only.")
    except RuntimeError as e:
        print(f"  [LLM] Fallback zh translation failed: {e}")
        return {}

    zh_map = {}
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


# ---------------------------------------------------------------------------
# Phase E: LLM critique + targeted patch repair
# ---------------------------------------------------------------------------

_MAX_PATCHES = 8
_PATCH_MERGE_GAP = 5


def _judge_dialogue_compact(script: dict, with_zh: bool) -> str:
    items = []
    for i, l in enumerate(script.get("dialogue", [])):
        item = {"i": i, "s": l.get("speaker", ""), "t": l.get("text", ""),
                "p": l.get("phase", "")}
        if with_zh:
            item["z"] = l.get("zh", "")
        items.append(item)
    return json.dumps(items, ensure_ascii=False, separators=(",", ":"))


def _gate_issues_digest(report: dict) -> str:
    lines = []
    for i in report.get("issues", []):
        tag = "E" if i["severity"] == "error" else "W"
        lines.append(f"[{tag}] {i['check']}: {i['detail']}")
    return "\n".join(lines[:60]) or "(none)"


def _build_judge_prompt(kind: str, script: dict, report: dict) -> str:
    with_zh = kind == "language"
    dialogue_json = _judge_dialogue_compact(script, with_zh)
    outline_block = f"""Story concept: {script.get('story_concept', script.get('story_hook', ''))}
Listening question: {script.get('listening_question_en', '')}
True answer (revealed in reveal phase): {script.get('answer_en', '')}
CEFR level: {script.get('cefr', 'A2')}
"""
    machine = f"""Machine checks already found (do NOT repeat these):
{_gate_issues_digest(report)}
"""
    if kind == "story":
        task = """You are a STORY COHERENCE judge for an ESL slow-listening video script.
Find ONLY story-level problems the machine checks above cannot detect:
1. Topic/story regression: a later line re-introduces a topic or question already settled earlier (as if new).
2. The listening question is semantically ANSWERED before the reveal phase (beyond the keyword leaks already listed).
3. Character personality drift (char_a curious/energetic, char_b calm/knowledgeable, char_c staff).
4. Abrupt/unnatural transitions between sections; reveal phase doesn't clearly explain the answer; ending feels cut off.
5. Facts contradict each other across the story (prices, names, times, places).
Report each problem as a line range [start, end] (0-indexed, inclusive)."""
    else:
        task = """You are a LANGUAGE QUALITY judge for an ESL slow-listening video script (target: overseas Chinese beginners).
Find ONLY language-level problems the machine checks above cannot detect:
1. Robotic stretches: 3+ consecutive lines with flat Q&A rhythm or same sentence structure.
2. Unnatural English: textbook phrasing no native speaker would say; wrong register for the CEFR level (too hard words/grammar).
3. zh translation errors: mistranslation, mistranscribed meaning, overly stiff machine-like 繁體中文, inconsistent terminology for the same English term.
4. Repetitive openers or verbal tics the machine didn't flag.
Report each problem as a line range [start, end] (0-indexed, inclusive)."""

    return f"""{outline_block}{machine}
{task}

Dialogue (compact JSON, i = line index):
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


def _critique_script(script: dict, report: dict) -> list[dict]:
    """Run story judge + language judge, return combined validated issues."""
    combined = []
    for kind in ("story", "language"):
        system = ("You are a strict script quality judge for ESL videos. "
                  "Output valid JSON only.")
        try:
            result = _chat_and_parse(
                _build_judge_prompt(kind, script, report),
                temperature=0.3, max_tokens=8192, reasoning_effort="low",
                label=f"judge_{kind}", system=system)
        except RuntimeError as e:
            print(f"  [QA] {kind} judge failed: {e}")
            continue
        raw = (result.get("issues", []) if isinstance(result, dict)
               else (result if isinstance(result, list) else []))
        found = _parse_judge_issues(raw, len(script.get("dialogue", [])))
        print(f"    {kind} judge: {len(found)} issues")
        combined.extend(found)
    return combined


def _merge_patches(report: dict, judge_issues: list[dict],
                   script: dict) -> list[tuple[int, int, list[str]]]:
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
                          "different wording."))
        elif "泄露" in issue["detail"]:
            for l in ls:
                spans.append((max(0, l - 1), min(n - 1, l + 1),
                              "This line leaks the answer before the reveal "
                              "phase — rephrase to tease/deflect instead."))
        elif "未明确揭晓答案" in issue["detail"]:
            idxs = [i for i, l in enumerate(dialogue)
                    if l.get("phase") == "reveal"]
            if idxs:
                spans.append((idxs[0], idxs[-1],
                              "Reveal phase must clearly state the true answer."))
        elif "speaker 非法" in issue["detail"]:
            for l in ls:
                spans.append((l, l, "Fix the speaker (phase speaker rules)."))

    for issue in judge_issues:
        s, e = issue["lines"]
        hint = f"[{issue['type']}] {issue['problem']} → {issue['fix_hint']}"
        spans.append((s, e, hint))

    if not spans:
        return []

    # Sort and merge overlapping / nearby spans (capped so a merged patch
    # never grows into a whole-script rewrite)
    spans.sort(key=lambda t: (t[0], t[1]))
    merged: list[list] = []
    _MAX_SPAN = 30
    for s, e, hint in spans:
        if (merged and s - merged[-1][1] <= _PATCH_MERGE_GAP
                and max(merged[-1][1], e) - merged[-1][0] + 1 <= _MAX_SPAN):
            merged[-1][1] = max(merged[-1][1], e)
            if hint not in merged[-1][2]:
                merged[-1][2].append(hint)
        else:
            merged.append([s, e, [hint]])
    return [(s, e, hints) for s, e, hints in merged[:_MAX_PATCHES]]


def _repair_patches(script: dict, patches: list[tuple[int, int, list[str]]],
                    outline: dict, cefr: str):
    """Rewrite each patch in place, preserving the exact line count."""
    dialogue = script["dialogue"]
    question = script.get("listening_question_en", "")

    for s, e, hints in sorted(patches, key=lambda p: -p[0]):
        k = e - s + 1
        orig = dialogue[s:e + 1]
        ctx_before = dialogue[max(0, s - 3):s]
        ctx_after = dialogue[e + 1:e + 4]
        phases = sorted({l.get("phase", "") for l in orig})
        rules = "\n".join(_PHASE_RULES.get(p, "") for p in phases if p)

        answer_block = ""
        if any(p in ("reveal", "review") for p in phases):
            answer_block = (f"\nThe true answer to convey: {script.get('answer_en','')}\n"
                            f"Misconception to correct: {script.get('common_misconception_en','')}\n")
        else:
            answer_block = "\nDo NOT reveal the answer here (it is revealed later).\n"

        def fmt(ls):
            return "\n".join(f"  {l.get('speaker','?')}: {l.get('text','')}"
                             for l in ls)

        orig_json = json.dumps(
            [{"speaker": l.get("speaker"), "text": l.get("text"),
              "phase": l.get("phase"), "zh": l.get("zh"),
              "on_screen": l.get("on_screen")} for l in orig],
            ensure_ascii=False, separators=(",", ":"))

        prompt = f"""Rewrite EXACTLY {k} dialogue lines (indices {s}-{e}) of an ongoing ESL conversation.

Story concept: {script.get('story_concept', '')}
Listening question: {question}
{answer_block}
Lines BEFORE (context, do not rewrite):
{fmt(ctx_before)}

Lines AFTER (context, do not rewrite):
{fmt(ctx_after)}

Current lines to rewrite:
{orig_json}

Problems to fix:
{chr(10).join(f"- {h}" for h in hints)}

Requirements:
- Keep EXACTLY {k} lines. Same phase values: {", ".join(phases)}.
- Speaker rules: buildup/reveal/review only char_a/char_b; core may use char_c.
- CEFR {cefr}. Natural conversational English with fillers and varied length.
- Every line keeps fields: speaker/text/phase/zh(繁體中文)/on_screen.
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
        except RuntimeError as ex:
            print(f"    patch L{s}-{e} failed: {ex}")
            continue

        if lines and len(lines) == k:
            for new, old in zip(lines, orig):
                new["phase"] = old.get("phase", new.get("phase"))
                sp = new.get("speaker", "")
                if sp not in ("char_a", "char_b", "char_c"):
                    new["speaker"] = old.get("speaker", "char_a")
                if sp == "char_c" and new["phase"] != "core":
                    new["speaker"] = old.get("speaker", "char_b")
                if not new.get("on_screen"):
                    new["on_screen"] = [new["speaker"]]
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
# Phase A: outline prompt (kept from v3)
# ---------------------------------------------------------------------------

def _build_outline_prompt(topic: str, cefr: str, used_dialogues: list[str] = None) -> str:
    used_hint = ""
    if used_dialogues:
        used_hint = f"""
AVOID DUPLICATES — these scenarios already exist:
{chr(10).join(f"  - {d}" for d in used_dialogues[:10])}
"""
    return f"""You are an expert ESL video director designing a SLOW LISTENING video story for overseas Chinese beginners.

Topic: {topic}
CEFR: {cefr}
CEFR Vocabulary Guide:
- A1: basic everyday words, present tense, short sentences (5-8 words)
- A2: common daily phrases, present/past tense, sentences 5-12 words
- B1: moderate vocabulary, mixed tenses, sentences 8-15 words, some idioms
- B2: advanced vocabulary, complex sentences, natural idioms and phrasal verbs

Design a story where the LISTENING QUESTION has a non-obvious answer — a fun fact or common misconception revealed INSIDE the dialogue (e.g. "Why is it called bubble tea?" not "What flavor did he order?").

CRITICAL — GENDER CONSISTENCY:
- Each character's gender (char_a_gender, char_b_gender, char_c_gender, host_gender) MUST be "male" or "female" (never "..." or empty).
- The gender MUST match the physical description. If host_description says "a young man", host_gender MUST be "male".
- This is essential for correct voice assignment in TTS.

{_build_character_override_prompt(quest=True)}{used_hint}
Output JSON ONLY:
{{
  "story_concept": "one-sentence story concept",
  "listening_question_en": "the listening question (max 12 words, answer is a fun fact)",
  "listening_question_zh": "繁體中文 translation",
  "answer_en": "the true answer revealed in reveal phase",
  "common_misconception_en": "what people commonly but wrongly think",
  "scene": "English scene name (e.g. coffee shop)",
  "char_a_description": "detailed physical description (gender, hair, clothing)",
  "char_a_gender": "male or female",
  "char_a_role": "role in story",
  "char_a_personality": "personality in 1-2 sentences (e.g. curious and energetic, asks lots of questions)",
  "char_b_description": "...",
  "char_b_gender": "...",
  "char_b_role": "...",
  "char_b_personality": "personality in 1-2 sentences (e.g. calm and knowledgeable, likes to explain things)",
  "char_c_description": "staff member description (uniform, etc.)",
  "char_c_gender": "...",
  "char_c_role": "...",
  "host_description": "TV host appearance (separate from dialogue characters)",
  "host_gender": "male or female",
  "key_words": [{{"en": "word", "zh": "繁中"}}],
  "buildup_summary": "3-5 sentence story summary for buildup phase",
  "core_summary": "3-5 sentence story summary for core phase (include teaching + transaction)",
  "reveal_summary": "3-5 sentence story summary for reveal phase",
  "review_summary": "3-5 sentence story summary for review phase"
}}

Topic: {topic}"""




