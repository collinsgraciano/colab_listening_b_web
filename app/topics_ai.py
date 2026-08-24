"""AI-powered topic management — generation, review, and suggestion application.

All LLM calls read provider config via config_manager.resolve_provider()
(NO os.environ mutation — avoids interfering with a running pipeline thread).
Blocking work should be executed by callers via asyncio.to_thread.
"""
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

from .config_manager import load_config, resolve_provider

# Reuse pipeline's robust JSON extraction (markdown fences + truncation repair)
_PIPELINE_DIR = Path(__file__).parent.parent.resolve() / "pipeline"
if str(_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_DIR))
from llm_client import _extract_json  # noqa: E402

# Rate limiting shared by all calls in this module (per-process, thread-safe enough
# for sequential batch use; the review loop is the only multi-call path)
_LAST_CALL_TIME = 0.0

_REVIEW_BATCH_SIZE = 50
_ISSUE_TYPES = {"duplicate", "similar", "grammar", "too_vague", "unsuitable", "other"}
_SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}

_AUDIENCE = (
    "Overseas Chinese living in English-speaking countries (US/UK/AU/CA) learning "
    "practical, real-life English for daily situations (ESL listening videos, CEFR A2-B1)."
)


class _RetryableError(Exception):
    """Internal: transient LLM failure worth retrying (empty/invalid output)."""


def _chat_json(messages: list[dict], temperature: float = 0.7,
               max_tokens: int = 4096) -> Any:
    """Non-streaming LLM call → parsed JSON (dict or list).

    Reads provider from web config each call. Retries on HTTP 429/524
    (15/30/60s backoff) and on empty/invalid JSON output. Raises RuntimeError
    with a Chinese message on final failure or missing API key.
    """
    global _LAST_CALL_TIME
    config = load_config()
    p_type, base_url, api_key, model = resolve_provider(config)
    if not api_key:
        raise RuntimeError(
            "未配置 LLM API Key — 请先在「参数配置」页面填写 SenseNova 或 OpenAI 兼容提供商的 API Key")
    min_interval = float(config.get("llm_min_interval") or 3)

    backoffs = [15, 30, 60]
    attempt = 0
    while True:
        elapsed = time.time() - _LAST_CALL_TIME
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        body = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if p_type != "openai":
            body["reasoning_effort"] = "low"
        req = urllib.request.Request(
            f"{base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            method="POST")
        req.add_header("Authorization", f"Bearer {api_key}")
        req.add_header("Content-Type", "application/json")
        req.add_header("User-Agent", "CodelyLLM/1.0")
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                raw = resp.read().decode("utf-8")
            _LAST_CALL_TIME = time.time()
            result = json.loads(raw)
            choices = result.get("choices") or [{}]
            content = (choices[0].get("message") or {}).get("content", "")
            if not content or not content.strip():
                raise _RetryableError("LLM 返回了空内容")
            try:
                return _extract_json(content)
            except json.JSONDecodeError:
                raise _RetryableError(
                    f"LLM 输出不是有效 JSON（前 200 字符）: {content[:200]}")
        except urllib.error.HTTPError as e:
            err = e.read().decode("utf-8", errors="replace")[:300]
            if e.code in (429, 524) and attempt < len(backoffs):
                wait = backoffs[attempt]
                attempt += 1
                print(f"  [TopicsAI] HTTP {e.code}, 等待 {wait}s 后重试 ({attempt}/{len(backoffs)})...")
                time.sleep(wait)
                continue
            raise RuntimeError(f"LLM HTTP {e.code}: {err}") from e
        except _RetryableError as e:
            if attempt < len(backoffs):
                wait = min(backoffs[attempt], 10)
                attempt += 1
                print(f"  [TopicsAI] {e}，{wait}s 后重试 ({attempt}/{len(backoffs)})...")
                time.sleep(wait)
                continue
            raise RuntimeError(f"LLM 调用失败（重试 {attempt} 次后）: {e}") from e


def _norm(topic: str) -> str:
    """Normalize a topic for duplicate comparison."""
    return re.sub(r"\s+", " ", topic.strip().lower()).strip(" .!?\"'“”")


def _flatten(topics_data: dict[str, list[str]]) -> list[tuple[str, str]]:
    """Flatten {category: [topics]} → [(category, topic), ...]."""
    return [(cat, t) for cat, ts in topics_data.items() for t in ts]


# ===========================================================================
# Local (free, no LLM) duplicate check
# ===========================================================================

def local_duplicate_check(topics_data: dict[str, list[str]]) -> list[dict]:
    """Find exact / normalized duplicates across the whole library.

    Returns [{"topic": str, "occurrences": [{"category", "index"}...]}] for every
    normalized topic string appearing 2+ times (index = position within category).
    """
    seen: dict[str, list[dict]] = {}
    for cat, ts in topics_data.items():
        for i, t in enumerate(ts):
            seen.setdefault(_norm(t), []).append({"category": cat, "index": i, "topic": t})
    dups = []
    for occurrences in seen.values():
        if len(occurrences) > 1:
            dups.append({"topic": occurrences[0]["topic"], "occurrences": occurrences})
    return dups


# ===========================================================================
# AI generation
# ===========================================================================

def generate_topics(category: str, count: int, topics_data: dict[str, list[str]],
                    used_topics: list[str], hint: str = "") -> dict:
    """Generate `count` new topics for an existing category.

    Returns {"topics": [...], "note": "..."} — already filtered against
    existing/used topics (normalized) and internally deduped.
    """
    existing_all = [t for _, t in _flatten(topics_data)]
    cat_topics = topics_data.get(category, [])
    cat_ref = "\n".join(f"  - {t}" for t in cat_topics[:15]) or "  (empty category)"
    existing_ref = "\n".join(f"  - {t}" for t in existing_all) or "  (none)"
    used_ref = "\n".join(f"  - {t}" for t in used_topics[:100]) or "  (none)"
    hint_block = (f"EXTRA REQUIREMENTS from the channel owner (follow strictly):\n{hint}\n"
                  if hint.strip() else "")

    prompt = f"""TASK: Generate exactly {count} NEW listening-practice video topics for the category "{category}".

AUDIENCE: {_AUDIENCE}

Style reference — existing topics in this category:
{cat_ref}

TOPIC FORMAT RULES:
- English only, 2-6 words, Title Case with short prepositions/articles lowercase (e.g. "At the Pharmacy", "Returning Shoes Without a Receipt")
- Each topic must be a SPECIFIC scenario concrete enough to carry one 18-line two-person dialogue
- Real-life situations that overseas Chinese people actually face in English-speaking countries
- Avoid abstract themes ("Friendship") or textbook categories ("Food") — use concrete scenes ("Ordering Bubble Tea", "Complaining About Cold Food")
- Varied: mix everyday routine, problem-solving, and emotionally engaging situations; suitable for CEFR A2-B1

MATRIX MINING (vary the ANGLE, not just the scene — a proven channel strategy):
Mine each scene from MULTIPLE angles instead of producing N variations of "At the X":
- Customer how-to: "Paying Inside at the Gas Station", "Using Self-Checkout for the First Time"
- Employee POV: "Opening the Coffee Shop at 5 AM", "The Last Customers of the Night", "A Delivery Driver's Last Stop"
- Brand-specific: "Ordering at the McDonald's Drive-Thru", "Shopping at Costco" (brand name = built-in search traffic)
- First-time experience: "First Time at an American Steakhouse", "Trying Southern Food for the First Time"
- Mishap/trouble: "Wrong Food Delivery Address", "The Parking Ticket Argument"
- Specific moment/detail with numbers or quotes: "Working the Lunch Rush at Noon", "Do You Have a Rewards Card?"
Each batch should cover at least 3 DIFFERENT angles from this list.

AVOID DUPLICATES — these topics already exist in the library (do NOT reuse or trivially rephrase any of them):
{existing_ref}

Already-used topics (videos already made — also avoid):
{used_ref}

{hint_block}OUTPUT JSON ONLY:
{{"topics": ["Topic One", "Topic Two"], "note": "一句话中文说明这批话题的思路"}}"""

    data = _chat_json(
        [{"role": "system", "content": "You are an expert ESL content strategist for a YouTube "
                                       "channel serving overseas Chinese learners. Output valid JSON only."},
         {"role": "user", "content": prompt}],
        temperature=0.9, max_tokens=2048)

    raw = data.get("topics", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
    existing_norm = {_norm(t) for t in existing_all}
    seen = set()
    topics = []
    for t in raw:
        if not isinstance(t, str):
            continue
        t = t.strip()
        n = _norm(t)
        if not t or len(t) > 80 or n in existing_norm or n in seen:
            continue
        seen.add(n)
        topics.append(t)
    return {"topics": topics, "note": (data.get("note", "") if isinstance(data, dict) else "")}


def suggest_category(count: int, topics_data: dict[str, list[str]], hint: str = "") -> dict:
    """Let the AI suggest ONE new complementary category + starter topics.

    Returns {"category": "中文 English", "reason": "...", "topics": [...]}.
    """
    cats = "\n".join(f"  - {c}" for c in topics_data.keys()) or "  (none)"
    hint_block = (f"EXTRA REQUIREMENTS from the channel owner (follow strictly):\n{hint}\n"
                  if hint.strip() else "")

    prompt = f"""TASK: Suggest ONE new topic category for an ESL listening-video topic library, plus {count} starter topics.

EXISTING CATEGORIES (the new one must be complementary, NOT overlapping with any of them):
{cats}

AUDIENCE: {_AUDIENCE}

CATEGORY RULES:
- Category key format: "中文 English" matching the existing style (e.g. "旅行 Travel", "银行邮局 Bank & Post")
- Must be a coherent real-life domain that is genuinely useful for the audience and clearly missing above
- Starter topics follow the same rules: English only, 2-6 words, Title Case, specific concrete scenarios

{hint_block}OUTPUT JSON ONLY:
{{"category": "中文 English", "reason": "1-2 句中文理由", "topics": ["Topic One", "Topic Two"]}}"""

    data = _chat_json(
        [{"role": "system", "content": "You are an expert ESL content strategist for a YouTube "
                                       "channel serving overseas Chinese learners. Output valid JSON only."},
         {"role": "user", "content": prompt}],
        temperature=0.9, max_tokens=2048)

    category = str(data.get("category", "")).strip()
    raw = data.get("topics", [])
    topics = [t.strip() for t in raw
              if isinstance(t, str) and t.strip() and len(t.strip()) <= 80]
    return {"category": category, "reason": str(data.get("reason", "")), "topics": topics}


# ===========================================================================
# AI review
# ===========================================================================

def review_topics(topics_data: dict[str, list[str]],
                  progress_cb: Callable[[int, int], None] | None = None) -> dict:
    """Full-library audit: local duplicates + AI batched semantic review.

    progress_cb(batch_index, total_batches) is invoked before each LLM call.
    Returns:
    {
      "local_duplicates": [...],
      "issues": [{"topic", "category", "type", "severity", "reason", "action", "suggestion"}],
      "summary": {"total", "batches", "local_dup_groups", "issues", "high", "medium", "low"}
    }
    """
    local_dups = local_duplicate_check(topics_data)
    flat = _flatten(topics_data)
    if not flat:
        return {"local_duplicates": [], "issues": [],
                "summary": {"total": 0, "batches": 0, "local_dup_groups": 0,
                            "issues": 0, "high": 0, "medium": 0, "low": 0}}

    full_list = "\n".join(f"{i}. [{cat}] {t}" for i, (cat, t) in enumerate(flat, 1))
    batches = [flat[i:i + _REVIEW_BATCH_SIZE]
               for i in range(0, len(flat), _REVIEW_BATCH_SIZE)]

    issues_by_topic: dict[str, dict] = {}
    for bi, batch in enumerate(batches):
        if progress_cb:
            progress_cb(bi + 1, len(batches))
        start = bi * _REVIEW_BATCH_SIZE
        batch_list = "\n".join(f"{start + j + 1}. [{cat}] {t}"
                               for j, (cat, t) in enumerate(batch))
        prompt = f"""TASK: Audit topic items {start + 1}-{start + len(batch)} of the numbered library below.

FULL TOPIC LIBRARY (numbered, for cross-reference — a duplicate/similar pair may involve ANY item in this list, not just the batch):
{full_list}

AUDIT BATCH (review THESE items in detail):
{batch_list}

AUDIENCE: {_AUDIENCE}
Each topic must sustain ONE focused 18-line two-person dialogue video.

FLAG AN ITEM ONLY IF IT CLEARLY HAS ONE OF THESE PROBLEMS:
- "duplicate": essentially the same scenario as another item anywhere in the library (flag the WEAKER / less specific one of the pair, action "remove")
- "similar": meaningfully overlaps another item but could be differentiated — action "rename" with a more specific new title, or "remove" if truly redundant
- "grammar": spelling / grammar / capitalization error — action "rename" with the corrected title
- "too_vague": too broad or abstract to carry one focused dialogue (e.g. "Shopping", "Making Friends") — action "rename" with a concrete scenario
- "unsuitable": not a real-life English situation for this audience, unsafe/sensitive, or impossible to stage as a 2-person dialogue — action "remove"

RULES:
- Be conservative: if unsure, do NOT flag. At most 10 flags per batch.
- "index" MUST be the number of the item you are flagging, from the numbered lists above
- "reason": short Chinese explanation of the problem
- "suggestion": for "rename" → the corrected English topic TITLE ITSELF (2-6 words, e.g. "Exchanging Shoes Without a Receipt") — NEVER an instruction like "specify a scenario"; for "remove" → ""
- "severity": "high" (clear duplicate / unsuitable), "medium" (similar / too_vague), "low" (minor grammar)

OUTPUT JSON ONLY (empty "issues" array if everything is clean):
{{"issues": [{{"index": 12, "type": "similar", "severity": "medium", "reason": "与第 34 条场景高度重叠", "action": "rename", "suggestion": "Exchanging a Damaged Item"}}]}}"""

        data = _chat_json(
            [{"role": "system", "content": "You are a strict but fair content auditor for an ESL "
                                           "listening-practice video library. Output valid JSON only."},
             {"role": "user", "content": prompt}],
            temperature=0.2, max_tokens=4096)

        raw_issues = data.get("issues", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
        for item in raw_issues:
            if not isinstance(item, dict):
                continue
            try:
                idx = int(item.get("index"))
            except (TypeError, ValueError):
                continue
            if not (start + 1 <= idx <= start + len(batch)):
                continue
            cat, topic = flat[idx - 1]
            itype = item.get("type") if item.get("type") in _ISSUE_TYPES else "other"
            severity = item.get("severity") if item.get("severity") in _SEVERITY_ORDER else "low"
            action = item.get("action")
            suggestion = str(item.get("suggestion", "")).strip()
            if action == "rename" and (not suggestion or _norm(suggestion) == _norm(topic)):
                action = "remove"
                suggestion = ""
            if action not in ("remove", "rename"):
                action = "remove"
                suggestion = ""
            issue = {"topic": topic, "category": cat, "type": itype, "severity": severity,
                     "reason": str(item.get("reason", "")).strip(),
                     "action": action, "suggestion": suggestion}
            # Same topic flagged twice (across batches) → keep the more severe
            prev = issues_by_topic.get(topic)
            if prev is None or _SEVERITY_ORDER[severity] < _SEVERITY_ORDER[prev["severity"]]:
                issues_by_topic[topic] = issue

    issues = sorted(issues_by_topic.values(),
                    key=lambda x: (_SEVERITY_ORDER.get(x["severity"], 9), x["category"], x["topic"]))
    summary = {
        "total": len(flat),
        "batches": len(batches),
        "local_dup_groups": len(local_dups),
        "issues": len(issues),
        "high": sum(1 for i in issues if i["severity"] == "high"),
        "medium": sum(1 for i in issues if i["severity"] == "medium"),
        "low": sum(1 for i in issues if i["severity"] == "low"),
    }
    return {"local_duplicates": local_dups, "issues": issues, "summary": summary}


# ===========================================================================
# Apply suggestions
# ===========================================================================

def apply_suggestions(topics_file: str, actions: list[dict]) -> dict:
    """Apply remove/rename actions to topics.json, matching by topic STRING.

    actions: [{"action": "remove"|"rename", "category", "topic", "new_topic"?}]
    String matching (not index) keeps this safe when several actions land at once.
    Returns {"applied": n, "skipped": [reasons], "topics": updated_data}.
    """
    p = Path(topics_file)
    topics_data: dict[str, list[str]] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    applied = 0
    skipped: list[str] = []
    for a in actions:
        act = a.get("action")
        cat = a.get("category", "")
        topic = a.get("topic", "")
        lst = topics_data.get(cat)
        if lst is None or topic not in lst:
            skipped.append(f"未找到 [{cat}] {topic}")
            continue
        if act == "remove":
            lst.remove(topic)
            applied += 1
        elif act == "rename":
            new_topic = str(a.get("new_topic", "")).strip()
            if not new_topic:
                skipped.append(f"缺少新名称 [{cat}] {topic}")
                continue
            if any(new_topic in lst2 for lst2 in topics_data.values()):
                skipped.append(f"已存在，跳过重命名 → {new_topic}")
                continue
            lst[lst.index(topic)] = new_topic
            applied += 1
        else:
            skipped.append(f"未知操作 {act}: [{cat}] {topic}")
    if p.exists():
        p.write_text(json.dumps(topics_data, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"applied": applied, "skipped": skipped, "topics": topics_data}
