"""Standalone LLM client for listening video script generation.

Uses SenseNova DeepSeek V4 Flash (OpenAI-compatible API).
No external project imports.
"""
import json
import re
import os
import sys
import urllib.request
import urllib.error
from pathlib import Path

# Rate limiting: enforce minimum interval between LLM API calls to avoid HTTP 429.
# glm-5.2 is especially aggressive about "request rate increased too quickly".
_LAST_CALL_TIME = 0.0


def _get_min_call_interval() -> float:
    """Read LLM_MIN_INTERVAL from env per-call (not frozen at import time)."""
    return float(os.environ.get("LLM_MIN_INTERVAL", "5.0"))


def _enforce_rate_limit():
    """Sleep if the previous LLM call completed too recently.

    _LAST_CALL_TIME is set AFTER a successful response (not before the request),
    so the interval measures the gap from the end of the previous call to the
    start of the next one.
    """
    import time as _time
    global _LAST_CALL_TIME
    min_interval = _get_min_call_interval()
    elapsed = _time.time() - _LAST_CALL_TIME
    if elapsed < min_interval:
        wait = min_interval - elapsed
        _time.sleep(wait)


def _dump_raw_debug(raw: str, model: str, kind: str) -> str | None:
    """Write a full raw LLM response to LLM_DEBUG_DIR for offline inspection.

    Returns the file path on success, or None when LLM_DEBUG_DIR is unset or
    the write fails — debug dumping must never break the pipeline.
    """
    debug_dir = os.environ.get("LLM_DEBUG_DIR", "").strip()
    if not debug_dir:
        return None
    try:
        from datetime import datetime as _dt
        d = Path(debug_dir)
        d.mkdir(parents=True, exist_ok=True)
        safe_model = re.sub(r"[^\w.-]", "_", model) or "model"
        path = d / f"{kind}_{safe_model}_{_dt.now().strftime('%Y%m%d_%H%M%S_%f')}.json"
        path.write_text(raw, encoding="utf-8")
        return str(path)
    except OSError:
        return None


def _diagnose_response(result: dict) -> str:
    """Build a compact human-readable diagnosis of an LLM response dict.

    Surfaces the fields that explain empty/failed completions (finish_reason,
    message keys, reasoning_content length, usage counts, provider error).
    """
    parts: list[str] = []
    try:
        choices = result.get("choices") or []
        if choices:
            finish = choices[0].get("finish_reason")
            if finish:
                parts.append(f"finish_reason={finish!r}")
            msg = choices[0].get("message") or {}
            keys = list(msg.keys())
            if keys:
                parts.append(f"message keys={keys}")
            rc = msg.get("reasoning_content") or msg.get("reasoning")
            if rc:
                parts.append(f"reasoning_content={len(str(rc))} chars")
    except (AttributeError, IndexError, TypeError):
        pass
    usage = result.get("usage")
    if isinstance(usage, dict):
        u = {k: usage[k] for k in
             ("prompt_tokens", "completion_tokens", "reasoning_tokens", "total_tokens")
             if k in usage}
        if u:
            parts.append(f"usage={u}")
    if result.get("error"):
        parts.append(f"error={result['error']}")
    return "; ".join(parts) if parts else "no diagnostic fields found"


def _chat(messages: list[dict], temperature: float = 0.8, timeout: int = 180,
          max_tokens: int = 8192, reasoning_effort: str = "low") -> str:
    """Call LLM chat completion (SenseNova or OpenAI-compatible), return content string.

    Dispatches based on LLM_PROVIDER env var:
    - "sensenova" (default): SenseNova DeepSeek V4 Flash / glm-5.2
    - "openai": any OpenAI-compatible endpoint (x666.me, etc.)

    Retries on HTTP 429 (rate limit) with exponential backoff.
    Enforces a minimum interval between calls to avoid triggering rate limits.
    """
    import time as _time

    provider = os.environ.get("LLM_PROVIDER", "sensenova")

    if provider == "openai":
        model = os.environ.get("OPENAI_MODEL", "grok-4.6")
        api_key = os.environ.get("OPENAI_API_KEY", "")
        base_url = os.environ.get("OPENAI_BASE_URL", "https://x666.me/v1")
    else:
        model = os.environ.get("SENSENOVA_MODEL", "deepseek-v4-flash")
        api_key = os.environ.get("SENSENOVA_API_KEY", "")
        base_url = os.environ.get("SENSENOVA_BASE", "https://token.sensenova.cn/v1")

    # Retry on 429 (rate limit) and 524 (Cloudflare gateway timeout)
    _RETRY_CODES = [429, 524]
    _RETRY_BACKOFFS = [15, 30, 60, 90, 120]
    # One-shot rescue for finish_reason=length empty content (reasoning burn)
    _length_retried = False

    for _retry_attempt in range(len(_RETRY_BACKOFFS) + 1):
        _enforce_rate_limit()
        body = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        # reasoning_effort is SenseNova-specific; OpenAI-compatible APIs don't support it
        if provider != "openai":
            body["reasoning_effort"] = reasoning_effort
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            f"{base_url}/chat/completions",
            data=data,
            method="POST",
        )
        req.add_header("Authorization", f"Bearer {api_key}")
        req.add_header("Content-Type", "application/json")
        # Cloudflare-protected endpoints (e.g. x666.me) block default Python User-Agent with 403
        req.add_header("User-Agent", "CodelyLLM/1.0")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8")
                if not raw.strip():
                    raise RuntimeError("LLM returned empty response body (HTTP 200, 0 bytes)")
                try:
                    result = json.loads(raw)
                except json.JSONDecodeError:
                    dbg = _dump_raw_debug(raw, model, "non_json")
                    print(f"  [LLM] Non-JSON response (model={model}). "
                          f"Full raw response ({len(raw)} chars)"
                          f"{' saved to ' + dbg if dbg else ''}:")
                    print(raw)
                    print("  [LLM] End raw response")
                    raise RuntimeError(
                        f"LLM returned non-JSON response (model={model}). "
                        f"Full raw ({len(raw)} chars)"
                        f"{' saved to ' + dbg if dbg else ''} shown above: {raw}"
                    ) from None
                if "choices" not in result or not result["choices"]:
                    diag = _diagnose_response(result)
                    dbg = _dump_raw_debug(raw, model, "no_choices")
                    print(f"  [LLM] Response has no 'choices' (model={model}). {diag}")
                    print(f"  [LLM] Full raw response ({len(raw)} chars)"
                          f"{' saved to ' + dbg if dbg else ''}:")
                    print(raw)
                    print("  [LLM] End raw response")
                    raise RuntimeError(
                        f"LLM response has no 'choices' field (model={model}). {diag}. "
                        f"Full raw ({len(raw)} chars)"
                        f"{' saved to ' + dbg if dbg else ''} shown above: {raw}"
                    )
                content = result["choices"][0]["message"]["content"]
                if not content or not content.strip():
                    # Reasoning models can burn the whole token budget on
                    # reasoning (finish_reason=length, content empty). Retry
                    # once with a doubled budget before surfacing the error.
                    choice0 = result["choices"][0]
                    finish = choice0.get("finish_reason") or ""
                    has_reasoning = bool(
                        (choice0.get("message") or {}).get("reasoning_content"))
                    if (finish == "length" and has_reasoning
                            and not _length_retried and max_tokens < 16384):
                        new_max = min(max_tokens * 2, 16384)
                        print(f"  [LLM] Empty content with finish_reason=length "
                              f"(reasoning consumed the token budget); "
                              f"retrying with max_tokens={new_max} (was {max_tokens})...")
                        max_tokens = new_max
                        _length_retried = True
                        continue
                    diag = _diagnose_response(result)
                    dbg = _dump_raw_debug(raw, model, "empty_content")
                    print(f"  [LLM] Empty content (HTTP 200, model={model}). {diag}")
                    print(f"  [LLM] Full raw response ({len(raw)} chars)"
                          f"{' saved to ' + dbg if dbg else ''}:")
                    print(raw)
                    print("  [LLM] End raw response")
                    raise RuntimeError(
                        f"LLM returned empty content (HTTP 200, model={model}). {diag}. "
                        f"Full raw ({len(raw)} chars)"
                        f"{' saved to ' + dbg if dbg else ''} shown above: {raw}"
                    )
                # Record successful completion time for rate limiting.
                # This ensures the next call waits at least min_interval seconds
                # AFTER this response, not from when this request started.
                import time as _time
                _LAST_CALL_TIME = _time.time()
                return content
        except urllib.error.HTTPError as e:
            err = e.read().decode("utf-8", errors="replace")
            if e.code in _RETRY_CODES and _retry_attempt < len(_RETRY_BACKOFFS):
                wait = _RETRY_BACKOFFS[_retry_attempt]
                print(f"  [LLM] HTTP {e.code} ({'rate limited' if e.code == 429 else 'gateway timeout'}), "
                      f"waiting {wait}s before retry "
                      f"({_retry_attempt+1}/{len(_RETRY_BACKOFFS)})... "
                      f"Model: {model}")
                _time.sleep(wait)
                continue
            raise RuntimeError(f"LLM HTTP {e.code}: {err}") from e


def _repair_truncated_json(text: str) -> str:
    """Attempt to repair truncated JSON by closing open strings, arrays, and objects."""
    # Count unmatched braces/brackets
    in_string = False
    escape = False
    stack = []
    i = 0
    while i < len(text):
        c = text[i]
        if escape:
            escape = False
            i += 1
            continue
        if c == '\\' and in_string:
            escape = True
            i += 1
            continue
        if c == '"' and not escape:
            in_string = not in_string
        elif not in_string:
            if c == '{':
                stack.append('}')
            elif c == '[':
                stack.append(']')
            elif c in ('}', ']'):
                if stack and stack[-1] == c:
                    stack.pop()
        i += 1
    # If we're in an unterminated string, close it
    if in_string:
        text += '"'
    # Remove trailing comma if present
    text = re.sub(r',\s*$', '', text.strip())
    # Close all open structures
    while stack:
        text += stack.pop()
    return text


def _extract_json(text: str) -> dict:
    """Extract JSON from LLM response (handles markdown fences + truncated JSON)."""
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try repairing truncated JSON
        repaired = _repair_truncated_json(text)
        return json.loads(repaired)


def _get_character_overrides() -> dict:
    """Read character overrides from env var (set by pipeline_service before step0)."""
    raw = os.environ.get("CHARACTER_OVERRIDES", "")
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}


def _build_character_override_prompt(quest: bool = False) -> str:
    """Build prompt section for pre-defined characters."""
    overrides = _get_character_overrides()
    if not overrides:
        return ""

    _LABELS = {
        "char_a": "Speaker 1 (char_a)",
        "char_b": "Speaker 2 (char_b)",
        "char_c": "Staff character (char_c)",
        "host": "TV Host (host)",
    }
    lines = ["PRE-DEFINED CHARACTERS (MANDATORY — these are FIXED, do NOT change):"]
    for key, info in overrides.items():
        desc = info.get("description", "")
        gender = info.get("gender", "")
        role = info.get("role", "")
        label = _LABELS.get(key, key)
        parts = []
        if desc:
            parts.append(desc)
        meta = []
        if gender:
            meta.append(f"gender: {gender}")
        if role:
            meta.append(f"role: {role}")
        if meta:
            parts.append(f"({', '.join(meta)})")
        lines.append(f"- {label}: {' '.join(parts)}")

    lines.append("")
    lines.append("You MUST follow these rules for pre-defined characters:")
    lines.append("1. Use the EXACT description text as the char_{key}_description value.")
    lines.append("2. Set char_{key}_gender to the specified gender. If gender is not given, infer it from the description.")
    lines.append("3. Set char_{key}_role to the specified role. If role is not given, generate one that fits the description.")
    lines.append("4. Write ALL dialogue for this character to match their role — the story scenario MUST fit these characters' roles (e.g. if role is 'dentist', the conversation should be about a dental visit).")
    lines.append("5. Use the EXACT same description text in ALL image_prompt, video_prompt, and poses entries for this character.")
    lines.append("6. If host is pre-defined, host_description and host_gender MUST match the provided values.")
    return "\n".join(lines) + "\n\n"


def _build_listening_prompt(topic: str, cefr: str, used_dialogues: list[str] = None,
                            num_lines: int = 18) -> str:
    """Build prompt for listening-practice lesson (num_lines + IPA + 繁中)."""
    used_hint = ""
    if used_dialogues:
        used_hint = f"""
IMPORTANT — AVOID DUPLICATES: The following dialogue scenarios have already been generated.
Do NOT create dialogue that is too similar to these. Use a DIFFERENT situation, different speakers, different story:
{chr(10).join(f"  - {d}" for d in used_dialogues[:20])}
"""
    return f"""You are an expert ESL teacher creating ENGLISH LISTENING PRACTICE content for overseas Chinese learners.

CORE MISSION: 帮助海外华人用最地道最日常的英语，搞定真实生活中的每一个场景.
{used_hint}
Topic: {topic}
CEFR Level: {cefr}
CEFR Vocabulary Guide:
- A1: basic everyday words, present tense, short sentences (5-8 words)
- A2: common daily phrases, present/past tense, sentences 5-12 words
- B1: moderate vocabulary, mixed tenses, sentences 8-15 words, some idioms
- B2: advanced vocabulary, complex sentences, natural idioms and phrasal verbs
Output: a JSON object ONLY (no markdown, no explanation).

{_build_character_override_prompt(quest=False)}CONTENT REQUIREMENTS — This is for a LISTENING PRACTICE video targeting overseas Chinese:
- The dialogue must be about REAL-LIFE situations that overseas Chinese people actually face in English-speaking countries — practical, relatable, and immediately useful.
- Topics should be things people encounter in daily life: ordering food, asking for directions, making small talk, dealing with a problem at a store, calling customer service, visiting a doctor, renting an apartment, banking, school registration, etc.
- The conversation must feel 100% NATURAL and REALISTIC — like something you'd overhear in real life, NOT a textbook. Use filler words (like "um", "well", "so"), natural pauses, back-channeling ("oh really?", "that makes sense"), and conversational flow.
- Characters should speak the way REAL Americans do in everyday life: contractions (don't, I'll, can't), casual phrasal verbs (pick up, figure out, run out of), common idioms and slang appropriate for the CEFR level, and natural sentence fragments.
- The dialogue must tell a COMPLETE story with a clear beginning, problem/development, and resolution — but keep it grounded in reality, not exaggerated or melodramatic.
- Include realistic communication patterns: clarifying questions, polite hedging ("I was wondering if...", "Would it be possible to..."), thanking, apologizing, expressing mild frustration or satisfaction naturally.
- Every line should teach something useful — a phrase, expression, or communication strategy that the viewer can immediately apply in their own life.
{used_hint}
TECHNICAL REQUIREMENTS:
- Exactly 2 speakers with natural American English
- Each speaker MUST have a clearly defined ROLE in the story (e.g. "customer" vs "waiter", "passenger" vs "check-in agent"). The role must be appropriate for the topic.
- Exactly {num_lines} dialogue lines (each under 15 words)
- The dialogue must flow as a continuous, coherent story (not disconnected Q&A)
- Every dialogue line MUST include:
  - "text": the English sentence
  - "phonetic": IPA phonetic transcription in /slashes/ (use proper IPA symbols)
  - "zh": Traditional Chinese (繁體中文) translation
  - "image_prompt": a detailed prompt describing what this character looks like AND what they are doing. MUST include: (1) the character's EXACT physical description (same every time for same speaker), (2) their role (e.g. "a waitress", "a customer"), (3) the scene location, (4) the action matching the dialogue text.
  - "video_prompt": a detailed prompt for AI video generation. MUST include the SAME character description and action as image_prompt. MUST also include the dialogue text so the character appears to be speaking those words naturally (e.g. "The character says: 'Hi, I'd like a latte, please.' while gesturing toward the menu"). CRITICAL: the video MUST closely reference the uploaded reference image — the character's appearance, clothing, and the scene must match the reference image exactly.
  - "poses": array of exactly 2 pose descriptions for stop-motion animation. Pose 0 = speaking (mouth open, expressive gesture). Pose 1 = listening (slight smile, relaxed). Each MUST include the character's full physical description (identical every time). NO props, NO objects, NO scene. Example: ["a young woman with brown hair in a green apron, speaking with mouth open, raising right hand", "a young woman with brown hair in a green apron, listening with a slight smile, relaxed posture"].
- "char_a_description": a detailed physical description of speaker 1 (gender, hair color, hairstyle, clothing). This MUST be used identically in ALL of speaker 1's image_prompt and video_prompt entries.
- "char_b_description": a detailed physical description of speaker 2 (gender, hair color, hairstyle, clothing). This MUST be used identically in ALL of speaker 2's image_prompt and video_prompt entries.
- "char_a_gender": "male" or "female" — the gender of speaker 1
- "char_b_gender": "male" or "female" — the gender of speaker 2
- "char_a_role": the role of speaker 1 in the story (e.g. "waitress", "customer")
- "char_b_role": the role of speaker 2 in the story (e.g. "customer", "waitress")
- "youtube_title": a high-CTR YouTube title for overseas Chinese learners. ALL Chinese text in Traditional Chinese (繁體中文). Start with 【】bracket tag, use ｜ as separator, include 3-8 emoji and catchy power phrases (e.g. "不用背多聽就會用", "聽完就能說"). End with ｜{{English topic}}. Length 80-150 chars. Example: "【沉浸式英文動畫】出國怕開口？✈️ 超實用機場英文：訂票、報到、托運行李一次搞定，聽完就能說！｜Airport English"
- "youtube_title_en": a high-CTR YouTube title in PURE ENGLISH (no Chinese). Include the topic, key skill, and a compelling hook. Max 100 chars. Example: "Can You Order Coffee in English? ☕ Real Conversation at a Coffee Shop"
- "youtube_description": a full YouTube video description (max 3000 chars). First line must be a hook with the main keyword. Include a "⏱️ Chapters:" section with timestamps for: 00:00 Title, 00:05 Dialogue, 00:xx Shadowing Practice, 00:xx Outro. End with 3 hashtags (#EnglishListening #ESL #LearnEnglish) and a subscribe CTA. ALL Chinese text in Traditional Chinese (繁體中文).
- "youtube_description_en": a full YouTube video description in PURE ENGLISH (no Chinese). Max 3000 chars. First line = hook with main keyword. Include "⏱️ Chapters:" section with timestamps. End with hashtags and subscribe CTA.
- "youtube_tags": an array of 15-20 SEO tags (mix of short and long-tail keywords, include both English and Traditional Chinese tags)
- "scene": the English name of the scene/location (e.g. "pharmacy", "coffee shop", "hotel lobby"). Used for thumbnail and prompts.
- "thumbnail_expression": the facial expression of the main character on the thumbnail (e.g. "surprised and excited", "confused and thinking", "cheerful and smiling", "friendly and confident")
- "thumbnail_action": a short description of what the main character is doing on the thumbnail (e.g. "pointing to a menu", "holding a shopping bag", "waving hello", "gesturing toward the counter")
- "thumbnail_subtitle": a short Traditional Chinese subtitle shown below the title on the thumbnail (e.g. "18句聽力練習", "每天50句", "實用日常英語")
- "thumbnail_icons": an array of 4-5 objects with "en" and "zh" string keys, describing scene-related keywords shown as circular icons at the bottom of the thumbnail. Each has an English label and a Traditional Chinese label. Example for pharmacy: [{{"en": "Prescription", "zh": "處方"}}, {{"en": "Refill", "zh": "補充"}}, {{"en": "Cough Syrup", "zh": "止咳糖漿"}}, {{"en": "Side Effects", "zh": "副作用"}}]
- "thumbnail_prompt": a detailed prompt for generating a YouTube thumbnail background image. Must describe: a 3D Pixar-style character with an expressive face, the scene location, bright colors, reference-style layout.
- "title": English title (e.g. "AT THE AIRPORT")
- "cefr": the CEFR level of this lesson, exactly "{cefr}" (used for thumbnail level badge)
- "title_zh": Traditional Chinese short title (max 6 characters, e.g. "在機場")
- "scene_zh": Traditional Chinese scene description (e.g. "餐廳 · 點餐")
- "story_hook": a compelling 1-sentence intro that sets the scene
- "intro_zh": Traditional Chinese translation of the intro
- "outro": a short closing line
- "outro_zh": Traditional Chinese translation of the outro
- "practice_intro_en": English instruction before the 跟讀 section
- "practice_intro_zh": Traditional Chinese translation of the practice intro
- ALL Chinese text MUST be in Traditional Chinese (繁體中文)

CONSISTENCY RULES (CRITICAL):
- Gender: char_a_gender/char_b_gender MUST match the description text. If female, description MUST say "a young woman" and ALL her prompts MUST say so. NEVER mix genders.
- Appearance: each speaker's description (hair, clothing, etc.) MUST be IDENTICAL across ALL their image_prompt/video_prompt/poses entries.
- Scene: image_prompt and video_prompt MUST match the dialogue context (if at a restaurant, say "restaurant", NOT "airport"). Scene MUST be consistent throughout ALL lines.
- Speaker field: MUST use "char_a" or "char_b" (not actual names).

JSON schema:
{{
  "title": string,
  "cefr": string,
  "title_zh": string,
  "scene_zh": string,
  "lesson_type": "listening",
  "story_hook": string,
  "intro_zh": string,
  "outro": string,
  "outro_zh": string,
  "practice_intro_en": string,
  "practice_intro_zh": string,
  "char_a_description": string,
  "char_b_description": string,
  "char_a_gender": string,
  "char_b_gender": string,
  "char_a_role": string,
  "char_b_role": string,
  "youtube_title": string,
  "youtube_title_en": string,
  "youtube_description": string,
  "youtube_description_en": string,
  "youtube_tags": [string],
  "thumbnail_prompt": string,
  "scene": string,
  "thumbnail_expression": string,
  "thumbnail_action": string,
  "thumbnail_subtitle": string,
  "thumbnail_icons": [{{"en": string, "zh": string}}],
  "dialogue": [{{"speaker": string, "text": string, "phonetic": string, "zh": string, "image_prompt": string, "video_prompt": string, "poses": [string, string]}}]
}}

Topic: {topic}"""


def _load_used_listening_summaries(lessons_dir: str = None) -> list[str]:
    """Load summaries of previously generated listening dialogues for anti-duplicate.

    Scans a lessons/ directory for JSON files with lesson_type="listening".
    If lessons_dir is None or doesn't exist, returns empty list.
    """
    if not lessons_dir:
        return []
    lessons_path = Path(lessons_dir)
    if not lessons_path.exists():
        return []
    summaries = []
    for f in lessons_path.glob("*.json"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            script = data.get("script", data)
            if script.get("lesson_type") != "listening":
                continue
            title = script.get("title", "")
            story = script.get("story_hook", "")
            first_line = script.get("dialogue", [{}])[0].get("text", "")
            summaries.append(f"{title}: {story} (starts: {first_line[:60]})")
        except (json.JSONDecodeError, KeyError, IndexError):
            continue
    return summaries


def generate_listening_script(topic: str, cefr: str = "A2",
                              lessons_dir: str = None,
                              num_lines: int = 18) -> dict:
    """Generate a listening-practice lesson script via SenseNova DeepSeek V4 Flash.

    Args:
        topic: e.g. "At the Pharmacy"
        cefr: CEFR level (A1, A2, B1, B2, C1, C2)
        lessons_dir: optional path to lessons/ directory for anti-duplicate check
        num_lines: number of dialogue lines to generate (default 18)

    Returns:
        Script dict with dialogue[], char descriptions, title, etc.
    """
    used_summaries = _load_used_listening_summaries(lessons_dir)
    prompt = _build_listening_prompt(topic, cefr, used_dialogues=used_summaries,
                                     num_lines=num_lines)

    # Retry up to 3 times on JSON parse errors (LLM may truncate or produce invalid JSON)
    last_error = None
    for attempt in range(3):
        try:
            content = _chat(
                [
                    {"role": "system", "content": "You are an expert ESL teacher creating English listening practice content for overseas Chinese learners. Output valid JSON only — no markdown, no explanations."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.8 if attempt == 0 else 0.7,
                max_tokens=8192,
            )
            script = _extract_json(content)
            break
        except (json.JSONDecodeError, RuntimeError) as e:
            last_error = e
            err_str = str(e)
            # Log raw content on JSON parse errors for debugging
            if isinstance(e, json.JSONDecodeError):
                print(f"  [LLM retry {attempt+1}/3] JSONDecodeError: {err_str[:200]}")
                print(f"  [LLM] Raw content (first 300 chars): {content[:300] if 'content' in dir() else 'N/A'}")
            else:
                print(f"  [LLM retry {attempt+1}/3] {type(e).__name__}: {err_str[:200]}")
            if attempt < 2:
                import time
                time.sleep(5)
    else:
        raise RuntimeError(f"LLM script generation failed after 3 retries: {last_error}")

    # Ensure lesson_type marker
    script["lesson_type"] = "listening"

    # Ensure all required fields exist
    script.setdefault("story_hook", "")
    script.setdefault("intro_zh", "")
    script.setdefault("outro", "That's all for today. Keep practicing!")
    script.setdefault("outro_zh", "")
    script.setdefault("title", "")
    script["cefr"] = script.get("cefr") or cefr  # used by thumbnail level badge
    script.setdefault("title_zh", script.get("intro_zh", ""))
    script.setdefault("practice_intro_en", "Now let's practice. Listen and repeat each sentence.")
    script.setdefault("practice_intro_zh", "現在來練習。請跟著朗讀每一句。")
    script.setdefault("char_a_description", "")
    script.setdefault("char_b_description", "")
    script.setdefault("char_a_gender", "male")
    script.setdefault("char_b_gender", "female")
    script.setdefault("char_a_role", "")
    script.setdefault("char_b_role", "")
    script.setdefault("youtube_title", "")
    script.setdefault("youtube_title_en", "")
    script.setdefault("youtube_description", "")
    script.setdefault("youtube_description_en", "")
    script.setdefault("youtube_tags", [])
    script.setdefault("thumbnail_prompt", "")
    script.setdefault("scene", "")
    script.setdefault("thumbnail_expression", "surprised and excited")
    script.setdefault("thumbnail_action", "looking toward the camera and gesturing naturally")
    script.setdefault("thumbnail_subtitle", "18句聽力練習")
    script.setdefault("thumbnail_icons", [])

    # Ensure dialogue has all required fields
    for line in script.get("dialogue", []):
        line.setdefault("phonetic", "")
        line.setdefault("zh", "")
        line.setdefault("image_prompt", "")
        line.setdefault("video_prompt", "")
        line.setdefault("poses", [])

    return script


def design_voxcpm_voices(script: dict) -> dict:
    """Design VoxCPM voice descriptions for each character via LLM.

    The LLM analyses character descriptions / roles / genders from the script
    and produces natural-language voice descriptions suitable for the VoxCPM
    ``control`` parameter.

    Top priority: suitability for English teaching — clear articulation,
    moderate pace, warm and engaging tone.

    Returns:
        dict mapping character keys to voice-description strings::

            {"char_a": "...", "char_b": "...", "narrator": "..."}

        Includes ``char_c`` and ``host`` when the script has them (quest mode).
    """
    # Collect characters that need voices
    chars = []
    for key in ("char_a", "char_b", "char_c", "host"):
        desc = script.get(f"{key}_description", "")
        gender = script.get(f"{key}_gender", "")
        role = script.get(f"{key}_role", "")
        if desc or key in ("char_a", "char_b"):
            chars.append(f"- {key}: {desc} (gender: {gender}, role: {role})")
    char_block = "\n".join(chars)

    # Build output schema dynamically
    schema_keys = [c for c in ("char_a", "char_b", "char_c", "host")
                   if script.get(f"{c}_description") or c in ("char_a", "char_b")]
    schema_pairs = ", ".join(f'"{c}": string' for c in schema_keys)
    schema_pairs += ', "narrator": string'

    prompt = f"""You are an expert voice director for an English learning YouTube channel.
Design VoxCPM voice descriptions for each character.

CRITICAL PRIORITIES (in order):
1. SUITABLE FOR ENGLISH TEACHING — crystal-clear articulation, every word distinguishable, moderate conversational pace (not too fast, not too slow). ESL learners must be able to follow along effortlessly.
2. NATURAL AND ENGAGING — warm, friendly, encouraging tone that keeps learners interested and motivated.
3. CHARACTER-APPROPRIATE — match the character's gender, approximate age, and personality from the description below.
4. DISTINCT VOICES — each character should sound noticeably different so learners can easily tell speakers apart.

Characters:
{char_block}

Rules for each voice description:
- Include: gender, age range, voice quality (pitch/tone), speaking style, pace.
- Keep each description under 25 words. Be specific and concrete.
- Use "standard American English" pronunciation. Do NOT use celebrity names.
- Avoid extreme or unusual voice qualities that might reduce clarity.
- The "narrator" voice is for intro/outro segments — it should be a warm, professional, radio-host quality voice, very clear and easy to understand.

Output JSON ONLY (no markdown):
{{{schema_pairs}}}"""

    try:
        content = _chat(
            [{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=2048,
            reasoning_effort="low",
        )
        voices = _extract_json(content)
    except Exception as e:
        print(f"  [VoxCPM] Voice design failed ({e}), using fallback descriptions")
        # Fallback: generic descriptions based on gender
        voices = {}
        for key in schema_keys:
            gender = script.get(f"{key}_gender", "male").lower()
            if gender == "female":
                voices[key] = "Warm female voice, clear articulation, friendly and encouraging, moderate pace."
            else:
                voices[key] = "Warm male voice, clear articulation, friendly and encouraging, moderate pace."
        voices["narrator"] = "Professional female narrator voice, warm and clear, radio-host quality, moderate pace."

    # Ensure narrator exists — respect host_gender in quest mode
    if "narrator" not in voices or not voices["narrator"]:
        host_gender = script.get("host_gender", "").lower()
        if host_gender == "male":
            voices["narrator"] = "Professional male narrator voice, warm and clear, radio-host quality, moderate pace."
        elif host_gender == "female":
            voices["narrator"] = "Professional female narrator voice, warm and clear, radio-host quality, moderate pace."
        else:
            voices["narrator"] = "Professional narrator voice, warm and clear, radio-host quality, moderate pace."

    # Ensure all expected keys exist
    for key in ("char_a", "char_b"):
        if key not in voices or not voices[key]:
            gender = script.get(f"{key}_gender", "male").lower()
            if gender == "female":
                voices[key] = "Warm female voice, clear articulation, friendly, moderate pace."
            else:
                voices[key] = "Warm male voice, clear articulation, friendly, moderate pace."

    return voices


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate listening script")
    parser.add_argument("--topic", required=True, help="Topic")
    parser.add_argument("--cefr", default="A2", choices=["A1", "A2", "B1", "B2", "C1", "C2"], help="CEFR level (default A2)")
    parser.add_argument("--output", default="script.json", help="Output JSON path")
    args = parser.parse_args()

    script = generate_listening_script(args.topic, args.cefr)
    Path(args.output).write_text(json.dumps(script, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Script saved: {args.output}")
    print(f"Title: {script.get('title', '')}")
    print(f"Dialogue lines: {len(script.get('dialogue', []))}")
