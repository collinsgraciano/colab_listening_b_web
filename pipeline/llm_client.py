"""Standalone LLM client for listening video script generation.

Uses SenseNova DeepSeek V4 Flash (OpenAI-compatible API).
No external project imports.
"""
import json
import re
import os
import sys
import threading
import urllib.request
import urllib.error
from pathlib import Path

from style_manager import get_active_style_prompt, get_active_thumbnail_hint

# 线程局部 LLM 环境覆盖：Web 批量生成等后台线程自带 provider 配置，
# 与运行中 pipeline 线程读的 os.environ 互不污染（并发运行互不干扰）。
_LLM_ENV_TLS = threading.local()


def set_llm_env_override(cfg: dict | None) -> None:
    """设置当前线程的 LLM 环境覆盖（传 None 清除）。

    键名与 env var 同名（LLM_PROVIDER / OPENAI_API_KEY / ...）；
    未覆盖的键回退 os.environ。仅影响调用线程内的 LLM 调用。
    """
    _LLM_ENV_TLS.cfg = cfg


def _env_get(name: str, default: str = "") -> str:
    """os.environ.get 的线程局部覆盖版：当前线程的 override 优先。"""
    cfg = getattr(_LLM_ENV_TLS, "cfg", None)
    if cfg is not None and name in cfg:
        return cfg[name]
    return os.environ.get(name, default)


# Rate limiting: enforce minimum interval between LLM API calls to avoid HTTP 429.
# glm-5.2 is especially aggressive about "request rate increased too quickly".
_LAST_CALL_TIME = 0.0  # 最近一次调用的开始时刻（预约槽位）
_RATE_LOCK = threading.Lock()


def _get_min_call_interval() -> float:
    """Read LLM_MIN_INTERVAL per-call (thread-local override first)."""
    return float(_env_get("LLM_MIN_INTERVAL", "5.0"))


def _enforce_rate_limit(min_interval: float | None = None):
    """预约式限速：相邻两次 LLM 调用的开始时刻间隔 ≥ min_interval。

    min_interval 缺省读 LLM_MIN_INTERVAL（线程局部 override 优先）；
    显式传入时用调用方的值（topics_ai / script_library 网关按各自配置），
    但共享同一 _LAST_CALL_TIME，使 Web 批量任务与 pipeline 并发时限速互认，
    避免两路同时打同一 provider 触发 429。
    """
    import time as _time
    global _LAST_CALL_TIME
    with _RATE_LOCK:
        interval = _get_min_call_interval() if min_interval is None else float(min_interval)
        now = _time.time()
        start_at = max(now, _LAST_CALL_TIME + interval)
        _LAST_CALL_TIME = start_at  # 预约本次调用的开始槽位
    if start_at > now:
        _time.sleep(start_at - now)


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

    provider = _env_get("LLM_PROVIDER", "sensenova")

    if provider == "openai":
        model = _env_get("OPENAI_MODEL", "grok-4.6")
        api_key = _env_get("OPENAI_API_KEY", "")
        base_url = _env_get("OPENAI_BASE_URL", "https://x666.me/v1")
    else:
        model = _env_get("SENSENOVA_MODEL", "deepseek-v4-flash")
        api_key = _env_get("SENSENOVA_API_KEY", "")
        base_url = _env_get("SENSENOVA_BASE", "https://token.sensenova.cn/v1")

    # Retry on 429 (rate limit), 502/503/504 (gateway), 524 (Cloudflare timeout)
    _RETRY_CODES = [429, 502, 503, 504, 524]
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
                # 限速槽位已在 _enforce_rate_limit 中预约（start-to-start 间隔），
                # 无需在响应后再次记录时间。
                return content
        except urllib.error.HTTPError as e:
            err = e.read().decode("utf-8", errors="replace")
            if e.code in _RETRY_CODES and _retry_attempt < len(_RETRY_BACKOFFS):
                wait = _RETRY_BACKOFFS[_retry_attempt]
                reason = ("rate limited" if e.code == 429
                          else "gateway error" if e.code in (502, 503, 504)
                          else "gateway timeout")
                print(f"  [LLM] HTTP {e.code} ({reason}), "
                      f"waiting {wait}s before retry "
                      f"({_retry_attempt+1}/{len(_RETRY_BACKOFFS)})... "
                      f"Model: {model}")
                _time.sleep(wait)
                continue
            raise RuntimeError(f"LLM HTTP {e.code}: {err}") from e
        except OSError as e:
            # 网络层瞬断（连接超时/拒绝/DNS/读超时/服务器断连）— HTTPError 的
            # 父类也是 OSError，故放在其后；与 429 同样走 backoff 重试，
            # 避免一次网络抖动报废整个 quest 会话（20+ 次调用）。
            if _retry_attempt < len(_RETRY_BACKOFFS):
                wait = _RETRY_BACKOFFS[_retry_attempt]
                print(f"  [LLM] Network error ({type(e).__name__}: {e}), "
                      f"waiting {wait}s before retry "
                      f"({_retry_attempt+1}/{len(_RETRY_BACKOFFS)})... "
                      f"Model: {model}")
                _time.sleep(wait)
                continue
            raise RuntimeError(
                f"LLM network error after retries: {type(e).__name__}: {e}") from e


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
    """Read character overrides (env var set by pipeline_service before step0)."""
    raw = _env_get("CHARACTER_OVERRIDES", "")
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
        if key == "narration":
            # narration 仅供 TTS 音色覆盖（由 _reuse_characters 写入 script.json），
            # 与 LLM 无关，注入反而诱导模型输出多余字段
            continue
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
        if not desc and gender:
            # voice 模式：外观不预定，LLM 按主题新创作（仅性别/音色固定）
            meta.append("appearance: NOT predefined — create a fresh one")
        if meta:
            parts.append(f"({', '.join(meta)})")
        lines.append(f"- {label}: {' '.join(parts)}")

    lines.append("")
    lines.append("You MUST follow these rules for pre-defined characters:")
    lines.append("1. If a description is given for a character, use the EXACT description text as the char_{key}_description value. If the description is marked \"appearance: NOT predefined\", CREATE a brand-new appearance description for this character that fits the topic naturally — only the gender is fixed.")
    lines.append("2. Set char_{key}_gender to the specified gender. If gender is not given, infer it from the description.")
    lines.append("3. Set char_{key}_role to the specified role. If role is not given, generate one that fits the description.")
    lines.append("4. Write ALL dialogue for this character to match their role — the story scenario MUST fit these characters' roles (e.g. if role is 'dentist', the conversation should be about a dental visit).")
    lines.append("5. If a description is given, use the EXACT same description text in ALL visual prompt entries for this character. If you created a new description, use YOUR new description consistently in ALL entries.")
    lines.append("6. If host is pre-defined, host_description and host_gender MUST match the provided values.")
    return "\n".join(lines) + "\n\n"


def _per_line_prompt_reqs(pf: str, style_prompt: str) -> str:
    """按结构生成行级视觉 prompt 字段要求文本（cutout 无 → 只要求三件套）。"""
    if pf == "original":
        return f"""
  - "video_prompt": a detailed prompt for AI video generation. MUST include: (1) the character's EXACT physical description (same every time for same speaker), (2) their role (e.g. "a waitress", "a customer"), (3) the scene location, (4) the action matching the dialogue text, AND the dialogue text itself so the character appears to be speaking those words naturally (e.g. "The character says: 'Hi, I'd like a latte, please.' while gesturing toward the menu"). CRITICAL: the video MUST closely reference the uploaded reference image — the character's appearance, clothing, and the scene must match the reference image exactly. Style phrase (copy verbatim): "{style_prompt}"."""
    if pf == "original_static":
        return f"""
  - "image_prompt": a detailed prompt describing what this character looks like AND what they are doing. MUST include: (1) the character's EXACT physical description (same every time for same speaker), (2) their role (e.g. "a waitress", "a customer"), (3) the scene location, (4) the action matching the dialogue text. Style phrase (copy verbatim): "{style_prompt}"."""
    return ""


def _desc_consistency_hint(pf: str, speaker_num: int) -> str:
    """char 描述字段的一致性提示（无 prompt 字段的结构不追加）。"""
    if pf == "original":
        return (f" This MUST be used identically in ALL of speaker "
                f"{speaker_num}'s video_prompt entries.")
    if pf == "original_static":
        return (f" This MUST be used identically in ALL of speaker "
                f"{speaker_num}'s image_prompt entries.")
    return ""


def _build_listening_prompt(topic: str, cefr: str, used_dialogues: list[str] = None,
                            num_lines: int = 18,
                            structure: str = "original") -> str:
    """Build prompt for listening-practice lesson (num_lines + IPA + 繁中).

    structure 决定行级视觉 prompt 字段（其余模式不需要的字段不再要求生成）：
    original → video_prompt；original_static → image_prompt；original_cutout → 无。
    """
    pf = structure if structure in ("original", "original_static",
                                    "original_cutout") else "original"
    prompt_fields = {"original": ("video_prompt",),
                     "original_static": ("image_prompt",),
                     "original_cutout": ()}[pf]
    used_hint = ""
    if used_dialogues:
        used_hint = f"""
IMPORTANT — AVOID DUPLICATES: The following dialogue scenarios have already been generated.
Do NOT create dialogue that is too similar to these. Use a DIFFERENT situation, different speakers, different story:
{chr(10).join(f"  - {d}" for d in used_dialogues[:20])}
"""
    style_prompt = get_active_style_prompt()
    thumb_hint = get_active_thumbnail_hint()
    style_section = ""
    if prompt_fields:
        style_section = f"""
VISUAL STYLE (CRITICAL): The video's art style is: "{style_prompt}".
- EVERY {' and '.join(prompt_fields)} entry MUST include this EXACT style descriptor phrase (copy it verbatim).
- Do NOT use any other art style, do NOT mix styles, do NOT add contradicting style words (e.g. photorealistic, sketch).
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
  - "zh": Traditional Chinese (繁體中文) translation{_per_line_prompt_reqs(pf, style_prompt)}
- "char_a_description": a detailed physical description of speaker 1 (gender, hair color, hairstyle, clothing).{_desc_consistency_hint(pf, 1)}
- "char_b_description": a detailed physical description of speaker 2 (gender, hair color, hairstyle, clothing).{_desc_consistency_hint(pf, 2)}
- "char_a_gender": "male" or "female" — the gender of speaker 1
- "char_b_gender": "male" or "female" — the gender of speaker 2
- "char_a_role": the role of speaker 1 in the story (e.g. "waitress", "customer")
- "char_b_role": the role of speaker 2 in the story (e.g. "customer", "waitress")
- "youtube_title": a high-CTR YouTube title for overseas Chinese learners. ALL Chinese text in Traditional Chinese (繁體中文). Start with 【】bracket tag, use ｜ as separator, include 3-8 emoji and catchy power phrases (e.g. "不用背多聽就會用", "聽完就能說"). Optionally open with the Traditional Chinese translation of title_quote in「」quotes as a hook (e.g. 「三號加油機，麻煩了！」). End with ｜{{English topic}}. Length 80-150 chars. Example: "【沉浸式英文動畫】出國怕開口？✈️ 超實用機場英文：訂票、報到、托運行李一次搞定，聽完就能說！｜Airport English"
- "youtube_title_en": a high-CTR YouTube title in PURE ENGLISH (no Chinese). STRONGLY PREFER the "quote hook" pattern: open with title_quote in quotes, then the scene context. Example: '"Pump Number 3, Please" — Paying Inside at an American Gas Station'. Second-best pattern: a curiosity question, e.g. "Can You Order Coffee in English? ☕ Real Conversation at a Coffee Shop". Include the topic, keep it under 100 chars, and make viewers want to click.
- "youtube_description": a full YouTube video description (max 3000 chars). First line must be a hook with the main keyword. Include a "⏱️ Chapters:" section with timestamps for: 00:00 Title, 00:05 Dialogue, 00:xx Shadowing Practice, 00:xx Outro. End with 3 hashtags (#EnglishListening #ESL #LearnEnglish) and a subscribe CTA. ALL Chinese text in Traditional Chinese (繁體中文).
- "youtube_description_en": a full YouTube video description in PURE ENGLISH (no Chinese). Max 3000 chars. First line = hook with main keyword. Include "⏱️ Chapters:" section with timestamps. End with hashtags and subscribe CTA.
- "youtube_tags": an array of 15-20 SEO tags (mix of short and long-tail keywords, include both English and Traditional Chinese tags)
- "scene": the English name of the scene/location (e.g. "pharmacy", "coffee shop", "hotel lobby"). Used for thumbnail and prompts.
- "thumbnail_expression": the facial expression of the main character on the thumbnail (e.g. "surprised and excited", "confused and thinking", "cheerful and smiling", "friendly and confident")
- "thumbnail_action": a short description of what the main character is doing on the thumbnail (e.g. "pointing to a menu", "holding a shopping bag", "waving hello", "gesturing toward the counter")
- "thumbnail_subtitle": a short Traditional Chinese subtitle shown below the title on the thumbnail (e.g. "18句聽力練習", "每天50句", "實用日常英語")
- "thumbnail_icons": an array of 4-5 objects with "en" and "zh" string keys, describing scene-related keywords shown as circular icons at the bottom of the thumbnail. Each has an English label and a Traditional Chinese label. Example for pharmacy: [{{"en": "Prescription", "zh": "處方"}}, {{"en": "Refill", "zh": "補充"}}, {{"en": "Cough Syrup", "zh": "止咳糖漿"}}, {{"en": "Side Effects", "zh": "副作用"}}]
- "thumbnail_prompt": a detailed prompt for generating a YouTube thumbnail background image. Must describe: a {thumb_hint} character with an expressive face, the scene location, bright colors, reference-style layout.
- "title": English title (e.g. "AT THE AIRPORT")
- "title_quote": the single most catchy, memorable dialogue line from THIS dialogue, copied VERBATIM (under 10 words). It will be used as the YouTube title hook. Pick a line that instantly shows what the video teaches (e.g. "Pump number three, please.", "Do you have this in a medium?").
- "cefr": the CEFR level of this lesson, exactly "{cefr}" (used for thumbnail level badge)
- "title_zh": Traditional Chinese short title (max 6 characters, e.g. "在機場")
- "scene_zh": Traditional Chinese scene description (e.g. "餐廳 · 點餐")
- "story_hook": a compelling 1-sentence intro that sets the scene
- "intro_zh": Traditional Chinese translation of the intro
- "welcome_en": a warm YouTube-host greeting opening the video (1-2 sentences), welcoming viewers and hinting at today's topic (e.g. "Hi friends! Welcome back! Today we're checking out at a pharmacy.")
- "welcome_zh": Traditional Chinese translation of the welcome greeting
- "outro": a short closing line
- "outro_zh": Traditional Chinese translation of the outro
- "practice_intro_en": English instruction before the 跟讀 section
- "practice_intro_zh": Traditional Chinese translation of the practice intro
- ALL Chinese text MUST be in Traditional Chinese (繁體中文)

{style_section}CONSISTENCY RULES (CRITICAL):
- Gender: char_a_gender/char_b_gender MUST match the description text. If female, description MUST say "a young woman" and ALL her prompts MUST say so. NEVER mix genders.{_consistency_prompt_rules(pf)}
- Speaker field: MUST use "char_a" or "char_b" (not actual names).

JSON schema:
{{
  "title": string,
  "title_quote": string,
  "cefr": string,
  "title_zh": string,
  "scene_zh": string,
  "lesson_type": "listening",
  "story_hook": string,
  "intro_zh": string,
  "welcome_en": string,
  "welcome_zh": string,
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
  "dialogue": [{{"speaker": string, "text": string, "phonetic": string, "zh": string{_schema_prompt_fields(pf)}}}]
}}

Topic: {topic}"""


def _consistency_prompt_rules(pf: str) -> str:
    """CONSISTENCY RULES 中与视觉 prompt 相关的规则行（无 prompt 字段的结构省略）。"""
    if pf == "original":
        return ("\n- Appearance: each speaker's description (hair, clothing, etc.) "
                "MUST be IDENTICAL across ALL their video_prompt entries."
                "\n- Scene: video_prompt MUST match the dialogue context (if at a "
                "restaurant, say \"restaurant\", NOT \"airport\"). Scene MUST be "
                "consistent throughout ALL lines.")
    if pf == "original_static":
        return ("\n- Appearance: each speaker's description (hair, clothing, etc.) "
                "MUST be IDENTICAL across ALL their image_prompt entries."
                "\n- Scene: image_prompt MUST match the dialogue context (if at a "
                "restaurant, say \"restaurant\", NOT \"airport\"). Scene MUST be "
                "consistent throughout ALL lines.")
    return ""


def _schema_prompt_fields(pf: str) -> str:
    """JSON schema 中行级 prompt 字段片段（cutout 为空串）。"""
    if pf == "original":
        return ', "video_prompt": string'
    if pf == "original_static":
        return ', "image_prompt": string'
    return ""


def _load_used_listening_summaries(lessons_dir: str = None,
                                   lesson_type: str = "listening") -> list[str]:
    """Load summaries of previously generated listening lessons for anti-duplicate.

    Scans a lessons/ directory for JSON files with lesson_type="listening".
    If lessons_dir is None or doesn't exist,
    returns empty list.
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
            if script.get("lesson_type") != lesson_type:
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
                              num_lines: int = 18,
                              structure: str = "original") -> dict:
    """Generate a listening-practice lesson script via SenseNova DeepSeek V4 Flash.

    Args:
        topic: e.g. "At the Pharmacy"
        cefr: CEFR level (A1, A2, B1, B2, C1, C2)
        lessons_dir: optional path to lessons/ directory for anti-duplicate check
        num_lines: number of dialogue lines to generate (default 18)
        structure: original / original_static / original_cutout — 决定行级
            视觉 prompt 字段的裁剪（original→video_prompt, static→image_prompt,
            cutout→无）；未知值回退 original。

    Returns:
        Script dict with dialogue[], char descriptions, title, etc.
    """
    if structure not in ("original", "original_static", "original_cutout"):
        structure = "original"
    used_summaries = _load_used_listening_summaries(lessons_dir)
    prompt = _build_listening_prompt(topic, cefr, used_dialogues=used_summaries,
                                     num_lines=num_lines, structure=structure)

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
    script.setdefault("welcome_en", "")
    script.setdefault("welcome_zh", "")
    script.setdefault("outro", "That's all for today. Keep practicing!")
    script.setdefault("outro_zh", "")
    script.setdefault("title", "")
    script.setdefault("title_quote", "")
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

    # Ensure dialogue has all required fields（行级 prompt 字段按结构裁剪）
    for line in script.get("dialogue", []):
        line.setdefault("phonetic", "")
        line.setdefault("zh", "")
        if structure == "original":
            line.setdefault("video_prompt", "")
        elif structure == "original_static":
            line.setdefault("image_prompt", "")

    # QA: programmatic gate + LLM critique/repair loop (mirrors quest Phase D+E)
    from llm_review import run_listening_qa
    run_listening_qa(script, num_lines, structure=structure)

    return script


def generate_random_voice_designs(count: int = 10,
                                  avoid_names: list[str] | None = None,
                                  language: str = "english",
                                  gender: str = "any") -> list[dict]:
    """Generate diverse random VoiceDesign voice specs via LLM.

    用于 Web 端「自定义音色 → LLM 随机生成」：LLM 批量产出音色设计候选，
    用户试听后挑选喜欢的保存为设计音色。
    gender: "female"/"male" 只生成该性别，"any" 混合。

    Returns:
        list[dict]: 每项 {name, gender, language("en"/"zh"), description, instruct}
    """
    avoid = ", ".join(sorted(n for n in (avoid_names or []) if n)) or "(none)"
    want_gender = gender if gender in ("female", "male") else ""

    lang_req = {
        "english": 'All voices speak ENGLISH. Set every "language" field to "en".',
        "chinese": 'All voices speak CHINESE (Mandarin). Set every "language" field to "zh".',
        "mixed": 'Mix languages: roughly half English ("en") and half Chinese ("zh"). '
                 'Set each "language" field individually.',
    }.get(language, 'All voices speak ENGLISH. Set every "language" field to "en".')

    gender_req = {
        "female": 'All voices MUST be FEMALE. Set every "gender" field to "female".',
        "male": 'All voices MUST be MALE. Set every "gender" field to "male".',
    }.get(want_gender, 'Mix genders: roughly half female ("female") and half male ("male"). '
                        'Set each "gender" field individually.')

    prompt = f"""You are a creative voice director designing voices for Qwen3-TTS VoiceDesign.
Generate {count} DIVERSE, DISTINCT voice designs for language learning videos.

{lang_req}
{gender_req}

Each voice design MUST include these fields:
- "name": a short English given name (3-10 letters, capitalized, e.g. "Luna", "Jasper"). Unique within the list.
- "gender": "female" or "male"
- "language": "en" or "zh" (as required above)
- "description": a short Chinese summary of the voice character, ≤14 chars (e.g. "温柔知性美式女声")
- "instruct": an English description for the VoiceDesign model (2-4 sentences). Must cover:
  (1) timbre: age feel, pitch (high/mid/low), texture (bright/warm/husky/crisp/deep/soft)
  (2) accent (e.g. General American, British RP)
  (3) style: pace, energy, intonation, personality
  (4) learner-friendly delivery: clear articulation, natural pauses

Example instructs (follow this style):
- "Speak in a warm, friendly young American female voice with a mid-range pitch and a relaxed, moderate pace. Sound conversational and expressive, like a native speaker talking with a student."
- "Speak in a mature, confident American female voice with clear articulation, suitable for narration. Keep a steady, engaging pace with gentle emphasis on key phrases."

DIVERSITY requirements — cover a wide spectrum:
- Mix age feels (youthful / young adult / middle-aged / senior)
- Mix pitches: high, mid, low
- Mix energies: calm, warm, lively, energetic, playful, gentle, husky, crisp, confident
- Vary use cases: conversation partner, narrator, cheerful host, storyteller
- For English voices, mostly General American; optionally 1-2 British RP

CONSTRAINTS:
- Every voice must be CLEAR and easy for ESL learners to understand — no mumbling, no extreme speed, no heavy regional accents
- Names MUST NOT duplicate any of these existing voice names: {avoid}
- Do not duplicate names within the list

Output JSON ONLY (no markdown fences):
{{"voices": [{{"name": "...", "gender": "...", "language": "...", "description": "...", "instruct": "..."}}, ...]}}"""

    content = _chat(
        [{"role": "user", "content": prompt}],
        temperature=1.0,
        max_tokens=8192,
        reasoning_effort="low",
    )
    data = _extract_json(content)
    raw = data.get("voices", []) if isinstance(data, dict) else (
        data if isinstance(data, list) else [])

    # Normalize + validate: drop empty entries and name collisions
    seen = set(avoid_names or [])
    result: list[dict] = []
    for v in raw:
        if not isinstance(v, dict):
            continue
        name = str(v.get("name", "")).strip()
        v_gender = str(v.get("gender", "")).strip().lower()
        lang = str(v.get("language", "")).strip().lower()
        instruct = str(v.get("instruct", "")).strip()
        desc = str(v.get("description", "")).strip()
        if not name or not instruct or name in seen:
            continue
        # 指定性别时丢弃不符项（instruct 描述的音色性别与标签一致，改标签会造成错配）
        if want_gender and v_gender != want_gender:
            continue
        if v_gender not in ("female", "male"):
            v_gender = "female"
        if lang not in ("en", "zh"):
            lang = "en"
        seen.add(name)
        result.append({
            "name": name,
            "gender": v_gender,
            "language": lang,
            "description": desc or f"随机设计音色 ({'女声' if v_gender == 'female' else '男声'})",
            "instruct": instruct,
        })
        if len(result) >= count:
            break
    if not result:
        raise RuntimeError("LLM 未返回有效的音色设计")
    return result


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
