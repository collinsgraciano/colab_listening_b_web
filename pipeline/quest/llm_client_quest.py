"""Quest LLM client — multi-round generation for high-quality quest scripts.

3-round pipeline:
  Round 1: Story outline (characters, question, key_words, phase summaries)
  Round 2: Per-phase dialogue (buildup→core→reveal→review, with on_screen + zh)
  Round 3: Narration + metadata (hook_intro, outro, youtube_*)

Each round produces small JSON output (2-8K tokens) for higher quality.
Previous rounds' output feeds into subsequent rounds for continuity.
on_screen and zh are generated directly in Round 2 (no separate enhancement round).

Reuses _chat, _extract_json, _load_used_listening_summaries from parent llm_client.
"""
import sys
import os
from pathlib import Path

# Add parent dir to path so we can import llm_client
_PARENT = str(Path(__file__).parent.parent.resolve())
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from llm_client import _chat, _extract_json, _load_used_listening_summaries, _build_character_override_prompt, _get_character_overrides


def split_phase_lines(num_lines: int) -> tuple[int, int, int, int]:
    """Split total lines into (buildup, core, reveal, review) by reference ratio ~30/48/10/12.

    Mirrors the bubble-tea reference video (~18min, 213 lines): long buildup
    where the listening question arises naturally with character development,
    longest core with mixed speakers (education + transactions), a short reveal
    where the answer is exposed, and a review where the answer is confirmed.

    48 -> (14, 23, 5, 6).  250 -> (75, 120, 25, 30).
    Each phase is at least 1; excess/deficit is absorbed by core.
    """
    n_buildup = round(num_lines * 0.30)
    n_core = round(num_lines * 0.48)
    n_reveal = round(num_lines * 0.10)
    n_review = num_lines - n_buildup - n_core - n_reveal

    n = [n_buildup, n_core, n_reveal, n_review]

    # Ensure each phase >= 1 (take from largest)
    for i in range(4):
        if n[i] < 1:
            max_idx = n.index(max(n))
            if max_idx != i and n[max_idx] > 1:
                n[max_idx] -= 1
                n[i] = 1

    # Ensure sum == num_lines (adjust core)
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


def generate_quest_script(topic: str, cefr: str = "A1",
                          lessons_dir: str = None,
                          num_lines: int = 48) -> dict:
    """Generate a quest lesson script via multi-round LLM calls.

    Round 1: Story outline (characters, question, key_words, phase summaries)
    Round 2: Per-phase dialogue generation (buildup→core→reveal→review, with on_screen + zh)
    Round 3: Narration + metadata (hook_intro, outro, youtube_*)

    Returns assembled script dict with all fields.
    """
    import json
    import time

    n_buildup, n_core, n_reveal, n_review = split_phase_lines(num_lines)
    used_summaries = _load_used_listening_summaries(lessons_dir)

    # ── Round 1: Story outline ──────────────────────────────────────────
    print("  [LLM] Round 1: Story outline...")
    outline_prompt = _build_outline_prompt(topic, cefr, used_summaries)
    outline = _chat_and_parse(
        outline_prompt, temperature=0.9, max_tokens=4096, reasoning_effort="medium",
        label="outline",
        system="You are an expert ESL video director designing slow-listening stories for overseas Chinese beginners. Output valid JSON only.")

    # ── Round 2: Per-phase dialogue ─────────────────────────────────────
    print("  [LLM] Round 2: Dialogue generation (4 phases)...")
    all_dialogue = []

    # 2a: Buildup
    buildup_lines = _generate_phase_dialogue(
        "buildup", outline, n_buildup, prev_lines=None, cefr=cefr, temperature=0.85)
    all_dialogue.extend(buildup_lines)
    print(f"    buildup: {len(buildup_lines)} lines")

    # 2b: Core (split into batches of 60 to keep response manageable)
    core_batch_size = 60
    core_generated = []
    prev = buildup_lines[-3:] if len(buildup_lines) >= 3 else buildup_lines
    remaining = n_core
    while remaining > 0:
        batch = min(remaining, core_batch_size)
        lines = _generate_phase_dialogue(
            "core", outline, batch, prev_lines=prev, cefr=cefr, temperature=0.7)
        core_generated.extend(lines)
        prev = lines[-3:] if len(lines) >= 3 else lines
        remaining -= batch
        if remaining > 0:
            time.sleep(2)  # avoid rate limit
    all_dialogue.extend(core_generated)
    print(f"    core: {len(core_generated)} lines")

    # 2c: Reveal
    reveal_lines = _generate_phase_dialogue(
        "reveal", outline, n_reveal, prev_lines=prev, cefr=cefr, temperature=0.8)
    all_dialogue.extend(reveal_lines)
    print(f"    reveal: {len(reveal_lines)} lines")

    # 2d: Review
    review_lines = _generate_phase_dialogue(
        "review", outline, n_review, prev_lines=reveal_lines[-3:] if len(reveal_lines) >= 3 else reveal_lines,
        cefr=cefr, temperature=0.75)
    all_dialogue.extend(review_lines)
    print(f"    review: {len(review_lines)} lines")

    # ── Quick quality check ─────────────────────────────────────────────
    _quality_check(all_dialogue, outline)

    # ── Ensure on_screen defaults (Round 2 generates them, but fallback) ─
    for line in all_dialogue:
        if not line.get("on_screen"):
            line["on_screen"] = [line.get("speaker", "char_a")]

    # ── Round 3: Narration + metadata ───────────────────────────────────
    print("  [LLM] Round 3: Narration + metadata...")
    meta_prompt = _build_metadata_prompt(topic, cefr, outline, all_dialogue)
    meta = _chat_and_parse(
        meta_prompt, temperature=0.6, max_tokens=8192, reasoning_effort="low",
        label="metadata",
        system="You are a YouTube content strategist for ESL learning videos. Output valid JSON only.")

    # ── Fallback: fill any still-empty zh via a small targeted call ─────
    empty_zh = [(i, d.get("text", "")) for i, d in enumerate(all_dialogue)
                if not d.get("zh", "").strip()]
    if empty_zh:
        print(f"  [LLM] {len(empty_zh)} lines have empty zh, generating fallback translations...")
        zh_map = _batch_translate_zh(empty_zh)
        for i, zh in zh_map.items():
            all_dialogue[i]["zh"] = zh

    # ── Assemble final script ──────────────────────────────────────────
    script = {
        "lesson_type": "listening",
        "title": meta.get("title", topic.upper()),
        "cefr": cefr,
        "title_zh": meta.get("title_zh", ""),
        "scene_zh": meta.get("scene_zh", ""),
        "story_hook": outline.get("story_concept", ""),
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
        "host_bg_prompt": meta.get("host_bg_prompt", "a bright modern TV studio set, 3D cartoon style, no people"),
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
    }

    # Ensure per-line defaults
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
    # LLM may not reliably follow the override prompt, so we enforce it here.
    overrides = _get_character_overrides()
    for key in ("char_a", "char_b", "char_c", "host"):
        if key in overrides:
            gender = overrides[key].get("gender", "")
            if gender:
                script[f"{key}_gender"] = gender
                print(f"  [LLM] Override {key}_gender = {gender}")

    return script


# ---------------------------------------------------------------------------
# Helper: chat + extract JSON with retry
# ---------------------------------------------------------------------------

def _batch_translate_zh(lines: list[tuple[int, str]]) -> dict[int, str]:
    """Translate English dialogue lines to Traditional Chinese in one batch call.

    Args:
        lines: list of (index, english_text) for lines missing zh translation.
    Returns:
        dict mapping index -> Traditional Chinese translation.
    """
    if not lines:
        return {}
    import json as _json
    items = [{"i": i, "text": text} for i, text in lines]
    prompt = f"""Translate each English sentence to Traditional Chinese (繁體中文).
Return a JSON array, same length and order, each item with "i" and "zh":
{_json.dumps(items, ensure_ascii=False)}

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
                zh = item.get("zh", "").strip()
                if idx >= 0 and zh:
                    zh_map[idx] = zh
    elif isinstance(result, dict):
        # Single dict with index->zh mapping
        for idx_str, zh in result.items():
            try:
                idx = int(idx_str)
                if zh.strip():
                    zh_map[idx] = zh.strip()
            except (ValueError, TypeError):
                continue

    # For any still-missing translations, use a simple English->placeholder
    for i, _ in lines:
        if i not in zh_map:
            zh_map[i] = "（翻譯待補）"
    return zh_map


def _chat_and_parse(prompt: str, temperature: float = 0.8, max_tokens: int = 8192,
                    reasoning_effort: str = "low", label: str = "",
                    system: str = "") -> dict:
    """Call _chat, extract JSON, retry on failure.

    Retry count is configurable via LLM_RETRIES env var (default 10).
    """
    import json
    import time
    last_error = None
    max_retries = int(os.environ.get("LLM_RETRIES", "10"))
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
        except (json.JSONDecodeError, RuntimeError) as e:
            last_error = e
            print(f"  [LLM {label} retry {attempt+1}/{max_retries}] {type(e).__name__}: {str(e)[:200]}")
            if attempt < max_retries - 1:
                time.sleep(5)
    raise RuntimeError(f"LLM {label} failed after {max_retries} retries: {last_error}")


# ---------------------------------------------------------------------------
# Round 1: Story outline prompt
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


# ---------------------------------------------------------------------------
# Round 2: Per-phase dialogue generation
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

4. SAME SPEAKER CAN SPEAK CONSECUTIVELY (2-3 lines): Sometimes a character explains something across 2 lines without interruption. Do this naturally ~15-20 times across the whole dialogue.

5. EMOTIONAL VARIETY: Express curiosity, excitement, nervousness, surprise, satisfaction, humor. Characters are NOT flat.

6. NATURAL FLOW: Avoid the Q&A pattern (ask→answer→ask→answer). Characters can change topic, remember something, express doubt, make a joke, share personal experience.

7. CHARACTER PERSONALITY: char_a is curious/energetic (asks follow-ups, gets excited), char_b is calm/knowledgeable (explains patiently, teases playfully), char_c is helpful/friendly.

8. AVOID REPETITIVE PATTERNS: Don't start consecutive lines with "Yes," or "No,". Don't repeat the same sentence structure.

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

def _generate_phase_dialogue(phase: str, outline: dict, n_lines: int,
                              prev_lines: list[dict] | None, cefr: str,
                              temperature: float) -> list[dict]:
    """Generate dialogue for one phase with natural dialogue quality rules.

    Output includes on_screen field so Round 4 enhancement is not needed.
    """
    import json

    prev_hint = ""
    if prev_lines:
        prev_texts = [f"  {l.get('speaker','?')}: {l.get('text','')}" for l in prev_lines]
        prev_hint = f"Previous lines (for continuity):\n{chr(10).join(prev_texts)}\n"

    rules = _PHASE_RULES.get(phase, "")
    question = outline.get("listening_question_en", "")
    answer = outline.get("answer_en", "")
    misconception = outline.get("common_misconception_en", "")
    phase_summary = outline.get(f"{phase}_summary", "")
    key_words = ", ".join(w.get("en", "") for w in outline.get("key_words", []))
    char_a_personality = outline.get("char_a_personality", "curious and energetic")
    char_b_personality = outline.get("char_b_personality", "calm and knowledgeable")
    char_a_desc = outline.get("char_a_description", "")
    char_b_desc = outline.get("char_b_description", "")
    char_c_desc = outline.get("char_c_description", "")
    scene = outline.get("scene", "")

    if phase == "reveal":
        extra = f"\nThe ANSWER to reveal: {answer}\nCommon misconception: {misconception}"
    else:
        extra = ""

    # on_screen rules vary by phase
    if phase == "core":
        on_screen_rules = (
            '- buildup/review: mostly ["char_a","char_b"]\n'
            '- core: mix of ["char_a","char_c"], ["char_b","char_c"], ["char_a","char_b"]\n'
            '- Use [] (empty) for 1-2 lines per batch (environment/menu/object shots)'
        )
    else:
        on_screen_rules = (
            '- mostly ["char_a","char_b"]\n'
            '- Use [] (empty) for at most 1 line (environment shot)'
        )

    prompt = f"""You are an ESL dialogue writer. Write {n_lines} lines of natural English dialogue for the "{phase}" phase.

Story: {outline.get("story_concept", "")}
Listening question: {question}
Key words to use: {key_words}
{extra}
Phase summary: {phase_summary}

{prev_hint}
{_DIALOGUE_QUALITY_RULES}

PHASE-SPECIFIC RULES:
- CEFR {cefr} level. Each line 5-15 words, vary the length naturally.
- Characters have personality — char_a is {char_a_personality}, char_b is {char_b_personality}.
- {rules}

CHARACTERS (for on_screen assignment):
- char_a={char_a_desc}
- char_b={char_b_desc}
- char_c={char_c_desc}
Scene: {scene}

on_screen rules:
{on_screen_rules}

Example output (for reference format only):
[{{"speaker":"char_a","text":"Mina, do you want to try something fun after class today?","phase":"{phase}","zh":"Mina，你今天下課後想做點有趣的事嗎？","on_screen":["char_a","char_b"]}}]

Output: JSON array of exactly {n_lines} objects:
[{{"speaker":"char_a","text":"English sentence","phase":"{phase}","zh":"繁體中文翻譯","on_screen":["char_a","char_b"]}}]

Every line MUST include:
- "zh": Traditional Chinese (繁體中文) translation
- "on_screen": array of character keys visible on screen (e.g. ["char_a","char_b"], or [] for environment shots)
NO markdown, NO explanation. JSON array ONLY."""

    result = _chat_and_parse(
        prompt, temperature=temperature, max_tokens=8192,
        reasoning_effort="medium", label=f"dialogue_{phase}",
        system="You are an ESL dialogue writer. Write natural conversational English. Output valid JSON array only.")
    if isinstance(result, list):
        return result
    if isinstance(result, dict) and "dialogue" in result:
        return result["dialogue"]
    return [result]


# ---------------------------------------------------------------------------
# Round 3: Narration + metadata prompt
# ---------------------------------------------------------------------------

def _build_metadata_prompt(topic: str, cefr: str, outline: dict, dialogue: list[dict]) -> str:
    import json
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
  "host_bg_prompt": "TV studio background prompt (3D cartoon, no people)",
  "youtube_title": "高CTR繁中标题 with 【】and ｜ format. Pattern D: 【英文聽力挑戰】{{emoji}}{{topic 繁中}}｜❓你能聽出答案嗎？｜{{CEFR}}慢速英文｜不用背多聽就會用｜英文聽力訓練｜{{English topic}}",
  "youtube_title_en": "high-CTR PURE ENGLISH title (no Chinese). Include topic + key skill + compelling hook. Max 100 chars. Example: Can You Guess Why It's Called Bubble Tea? ☕ Slow English Listening",
  "youtube_description": "full 繁中 description with chapters + key_words",
  "youtube_description_en": "full PURE ENGLISH description (no Chinese). Include chapters section, key_words, hashtags, and subscribe CTA.",
  "youtube_tags": ["tag1","tag2",...],
  "thumbnail_prompt": "thumbnail image prompt",
  "thumbnail_expression": "main character expression",
  "thumbnail_action": "main character action",
  "thumbnail_subtitle": "繁中 short subtitle",
  "thumbnail_icons": [{{"en":"word","zh":"繁中"}}],
  "scene_images": [{{"prompt":"specific 3D cartoon scene description with details (counter, menu board, equipment, decor), 16:9, no people","label":"short English label"}}]
}}

IMPORTANT: Generate at least 8 scene_images covering different angles and details of the scene (e.g. exterior, counter, menu board, equipment, seating area, product close-up, kitchen, decoration).

JSON ONLY, no markdown."""


# ---------------------------------------------------------------------------
# Round 4: Per-line enhancement prompt (batched)
# ---------------------------------------------------------------------------

def _build_enhance_prompt_from_reindexed(batch_lines: list[dict], outline: dict) -> str:
    """Build enhancement prompt for a batch of lines (0-indexed within batch)."""
    import json
    lines_json = json.dumps(batch_lines, ensure_ascii=False, separators=(",",":"))
    char_a = outline.get("char_a_description", "")
    char_b = outline.get("char_b_description", "")
    char_c = outline.get("char_c_description", "")
    scene = outline.get("scene", "")
    return f"""For each dialogue line, decide which characters are visible on screen.

Characters: char_a={char_a} / char_b={char_b} / char_c={char_c}
Scene: {scene}

Lines:
{lines_json}

Rules for "on_screen":
- buildup/review: mostly ["char_a","char_b"]
- core: mix of ["char_a","char_c"], ["char_b","char_c"], ["char_a","char_b"]
- Use [] (empty) for 2-4 lines total (environment/menu/object shots)

Output: JSON array, same length and order:
[{{"i":0,"on_screen":["char_a","char_b"]}}]
JSON array ONLY."""


# ---------------------------------------------------------------------------
# Quality check
# ---------------------------------------------------------------------------

def _quality_check(dialogue: list[dict], outline: dict):
    """Quick checks on generated dialogue."""
    issues = []
    # Check speakers per phase
    for i, line in enumerate(dialogue):
        speaker = line.get("speaker", "")
        phase = line.get("phase", "")
        if phase in ("buildup", "reveal", "review") and speaker == "char_c":
            issues.append(f"  Line {i}: char_c should not appear in {phase}")
        if not line.get("text"):
            issues.append(f"  Line {i}: empty text")
        # Check for char_a/char_b leakage in text
        text = line.get("text", "")
        for leak in ("char_a", "char_b", "char_c"):
            if leak in text.lower():
                issues.append(f"  Line {i}: field name '{leak}' leaked into text")
    # Check key_words usage
    key_words = [w.get("en", "") for w in outline.get("key_words", [])]
    all_text = " ".join(d.get("text", "") for d in dialogue).lower()
    for kw in key_words:
        count = all_text.count(kw.lower())
        if count < 2:
            issues.append(f"  Key word '{kw}' appears only {count} times (need 3+)")
    # Check answer in reveal
    answer = outline.get("answer_en", "").lower()
    reveal_text = " ".join(d.get("text", "") for d in dialogue if d.get("phase") == "reveal").lower()
    if answer and len(answer) > 5 and answer[:10] not in reveal_text:
        # Check if at least some answer keywords appear
        answer_words = [w for w in answer.split() if len(w) > 3]
        found = sum(1 for w in answer_words if w in reveal_text)
        if found < len(answer_words) // 2:
            issues.append(f"  Answer not clearly mentioned in reveal phase")
    if issues:
        print(f"  [Quality] {len(issues)} issues found:")
        for issue in issues:
            print(issue)
    else:
        print("  [Quality] All checks passed")
