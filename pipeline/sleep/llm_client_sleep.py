"""sleep 模式脚本生成：LLM 分批产出 A/B 短对话组 + 元数据。

- 每批默认 50 组（100 行 dialogue），批间落盘 cache_dir 防中断丢失；
- 批 1 同时产出标题/YouTube 元数据（繁中），后续批只产 pairs；
- 批级校验（数量/字段/词长），不合格整批重试（≤2 次），词长软失败降级警告；
- script.json 行格式与 listening 完全一致（speaker char_a/char_b 交替 +
  text/phonetic/zh），组 = 相邻 A/B 两行；char_a 恒男声、char_b 恒女声
  （朗读声部，非剧情角色）。
"""
import json
import hashlib
from pathlib import Path

from llm_client import (_chat, _extract_json, resolve_max_line_words,
                        _env_int_clamped)


def _sleep_max_line_words() -> int:
    """单句最大词数（参考视频 ≤7 词）：SLEEP_MAX_LINE_WORDS → 默认 7，clamp [4,10]。"""
    return _env_int_clamped("SLEEP_MAX_LINE_WORDS", 7, 4, 10)


def _pairs_to_rows(pairs: list[dict]) -> list[dict]:
    """pairs [{a:{text,phonetic,zh}, b:{...}}] → listening 风格 dialogue 行。"""
    rows = []
    for p in pairs:
        a, b = p.get("a") or {}, p.get("b") or {}
        rows.append({"speaker": "char_a", "text": a.get("text", ""),
                     "phonetic": a.get("phonetic", ""), "zh": a.get("zh", "")})
        rows.append({"speaker": "char_b", "text": b.get("text", ""),
                     "phonetic": b.get("phonetic", ""), "zh": b.get("zh", "")})
    return rows


def _cache_key(topic: str, cefr: str, num_pairs: int) -> str:
    raw = f"sleep|{topic}|{cefr}|{num_pairs}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]


def _validate_pairs(pairs: list[dict], expected: int, max_words: int) -> tuple[bool, str, int]:
    """批级校验。返回 (ok, message, soft_violations)。"""
    if len(pairs) != expected:
        return False, f"expected {expected} pairs, got {len(pairs)}", 0
    soft = 0
    for i, p in enumerate(pairs):
        for side in ("a", "b"):
            obj = p.get(side)
            if not isinstance(obj, dict):
                return False, f"pair {i+1} '{side}' missing", 0
            text = (obj.get("text") or "").strip()
            phon = (obj.get("phonetic") or "").strip()
            zh = (obj.get("zh") or "").strip()
            if not text or not phon or not zh:
                return False, f"pair {i+1} '{side}' has empty text/phonetic/zh", 0
            if len(text.split()) > max_words:
                soft += 1
    return True, "", soft


def _batch_prompt(topic: str, cefr: str, count: int, start_idx: int,
                  total_pairs: int, max_words: int, with_meta: bool) -> str:
    cefr_guide = {
        "A1": "very basic everyday words, present tense only",
        "A2": "common daily phrases, present/past tense",
        "B1": "moderate vocabulary, some idioms",
        "B2": "advanced vocabulary, natural idioms and phrasal verbs",
    }.get(cefr, "common daily phrases")
    meta_schema = ""
    meta_reqs = ""
    if with_meta:
        meta_schema = '''
  "title": string,
  "title_zh": string,
  "scene_zh": string,
  "title_quote": string,
  "cefr": string,
  "youtube_title": string,
  "youtube_title_en": string,
  "youtube_description": string,
  "youtube_description_en": string,
  "youtube_tags": [string],'''
        meta_reqs = f'''
- "title": short English title of the phrase collection topic (e.g. "Everyday Phrases — Cleaning the House")
- "title_zh": Traditional Chinese short title (max 8 characters, e.g. "打掃房間")
- "scene_zh": Traditional Chinese scene label (e.g. "居家打掃 · 短句")
- "title_quote": the single most useful/catchy sentence from THIS batch, copied VERBATIM (under 10 words)
- "cefr": exactly "{cefr}"
- "youtube_title": high-CTR YouTube title for overseas Chinese learners, ALL Chinese in Traditional Chinese (繁體中文). Format: 【睡前英文聽力】+ emoji + 主題短語英文 ｜ 痛點句（如"不用背！睡覺聽就會"） ｜ 🌱{cefr} ｜ 💡{total_pairs*2}句循環聽. Length 55-95 chars (NEVER exceed 95). Use ｜ separators, include 3-6 emoji.
- "youtube_title_en": high-CTR YouTube title in PURE ENGLISH (no Chinese), under 100 chars. Mention the topic and the "listen while you sleep / no memorization" angle (e.g. "400 Everyday English Phrases While You Sleep — Cleaning the House").
- "youtube_description": full YouTube description (max 3000 chars), ALL Chinese in Traditional Chinese (繁體中文). First line = hook with 睡前聽/不用背 keyword. Explain the AB-loop listening method (男聲常速 → 女聲慢速 → 男女連貫). End with call-to-action (點讚/訂閱/留言想學的場景) + 3 hashtags (#英文聽力 #睡前英文 #LearnEnglish). Do NOT write timestamps.
- "youtube_description_en": same in PURE ENGLISH (no Chinese), max 3000 chars, same structure. Do NOT write timestamps.
- "youtube_tags": 15-20 SEO tags mixing English and Traditional Chinese.'''
    return f"""You are an expert ESL teacher creating a "listen while you sleep" English phrase-drill video for overseas Chinese learners (zero-basics friendly, background/ASMR style).

Topic: {topic}
CEFR Level: {cefr} ({cefr_guide})

This is batch {start_idx // count + 1}: generate pairs {start_idx + 1} to {start_idx + count} (of {total_pairs} total).

Each pair = a short A sentence (opening/trigger) + a short B sentence (natural reply/continuation) about the topic. These are NOT a connected story — each pair stands alone (different mini-situations welcome: doing the task, asking about it, small talk about it).

CONTENT REQUIREMENTS:
- High-frequency American English only — words people use every single day. NO rare/literary words.
- Each sentence AT MOST {max_words} words (HARD LIMIT).
- Natural, conversational, immediately useful (contractions like don't/I'm/let's are good).
- Pairs should feel like a mini exchange: question→answer, statement→reply, problem→solution.
- Across ALL batches: cover MANY different angles of the topic (actions, questions, polite requests, small complaints, encouragement) — avoid near-duplicate pairs within this batch.

Every sentence MUST include:
- "text": the English sentence
- "phonetic": IPA transcription in /slashes/ (proper IPA symbols)
- "zh": Traditional Chinese (繁體中文) translation — ALL Chinese text MUST be Traditional Chinese{meta_reqs}

Output a JSON object ONLY (no markdown, no explanation):
{{
  "pairs": [
    {{"a": {{"text": string, "phonetic": string, "zh": string}},
      "b": {{"text": string, "phonetic": string, "zh": string}}}}
  ]{meta_schema}
}}

Exactly {count} pairs in "pairs"."""


def _meta_from_batch(batch: dict, topic: str, cefr: str, num_pairs: int) -> None:
    """批 1 产出的元数据键抽出到 meta dict（就地写入 batch 弹出）。"""
    meta = {}
    for k in ("title", "title_zh", "scene_zh", "title_quote", "cefr",
              "youtube_title", "youtube_title_en", "youtube_description",
              "youtube_description_en", "youtube_tags"):
        if k in batch:
            meta[k] = batch.pop(k)
    meta.setdefault("title", topic.upper())
    meta.setdefault("title_zh", "")
    meta.setdefault("scene_zh", "")
    meta.setdefault("title_quote", "")
    meta["cefr"] = cefr
    meta.setdefault("youtube_title", "")
    meta.setdefault("youtube_title_en", "")
    meta.setdefault("youtube_description", "")
    meta.setdefault("youtube_description_en", "")
    meta.setdefault("youtube_tags", [])
    return meta


def _generate_batch(topic, cefr, count, start_idx, total_pairs, max_words,
                    with_meta, temperature):
    prompt = _batch_prompt(topic, cefr, count, start_idx, total_pairs,
                           max_words, with_meta)
    last_err = None
    for attempt in range(3):
        try:
            content = _chat(
                [{"role": "system",
                  "content": "You are an expert ESL teacher creating sleep-listening English phrase drills. Output valid JSON only — no markdown, no explanations."},
                 {"role": "user", "content": prompt}],
                temperature=round(max(0.3, temperature - 0.1 * attempt), 2),
                max_tokens=16384)
            batch = _extract_json(content)
            pairs = batch.get("pairs")
            if not isinstance(pairs, list):
                raise ValueError("JSON has no 'pairs' array")
            ok, msg, soft = _validate_pairs(pairs, count, max_words)
            if not ok:
                raise ValueError(f"batch invalid: {msg}")
            if soft:
                print(f"  [Sleep] WARNING: {soft} sentence(s) exceed {max_words} words — accepted (cards auto-shrink text).")
            return batch, pairs
        except Exception as e:
            last_err = e
            print(f"  [Sleep][Batch retry {attempt + 1}/3] {type(e).__name__}: {str(e)[:200]}")
    raise RuntimeError(f"Sleep batch generation failed after 3 retries: {last_err}")


def generate_sleep_script(topic: str, cefr: str = "A2", num_pairs: int = 200,
                          batch_pairs: int = 50, lessons_dir: str = None,
                          cache_dir: str = None) -> dict:
    """分批生成 sleep 脚本，返回 listening 兼容 script dict。

    批间落盘 cache_dir（默认 lessons_dir 或系统临时目录）——中断后重跑同
    topic/cefr/num_pairs 自动复用已成功批次。
    """
    num_pairs = max(10, min(400, int(num_pairs)))
    batch_pairs = max(10, min(80, int(batch_pairs)))
    max_words = _sleep_max_line_words()

    cdir = Path(cache_dir) if cache_dir else (
        Path(lessons_dir) if lessons_dir else Path.home() / ".sleep_cache")
    cdir.mkdir(parents=True, exist_ok=True)
    ck = _cache_key(topic, cefr, num_pairs)

    all_pairs: list[dict] = []
    meta: dict = {}
    start = 0
    while start < num_pairs:
        count = min(batch_pairs, num_pairs - start)
        batch_file = cdir / f"sleep_{ck}_{start:04d}.json"
        if batch_file.exists():
            print(f"  [Sleep] Batch cache hit: {batch_file.name}")
            data = json.loads(batch_file.read_text(encoding="utf-8"))
            batch, pairs = data.get("batch", {}), data.get("pairs", [])
        else:
            batch, pairs = _generate_batch(topic, cefr, count, start, num_pairs,
                                           max_words, with_meta=(start == 0),
                                           temperature=0.85)
            batch_file.write_text(json.dumps({"batch": batch, "pairs": pairs},
                                             ensure_ascii=False, indent=1),
                                  encoding="utf-8")
        if start == 0:
            meta = _meta_from_batch(dict(batch), topic, cefr, num_pairs)
        all_pairs.extend(pairs)
        start += count
        print(f"  [Sleep] Pairs {start}/{num_pairs} done")

    script = dict(meta)
    script["lesson_type"] = "listening"
    script["structure"] = "sleep"
    # 朗读声部（非剧情角色）：char_a 恒男声、char_b 恒女声，voice_map 按性别映射
    script["char_a_description"] = "male narrator voice"
    script["char_b_description"] = "female narrator voice"
    script["char_a_gender"] = "male"
    script["char_b_gender"] = "female"
    script["char_a_role"] = "narrator (male voice)"
    script["char_b_role"] = "narrator (female voice)"
    # 冷开场：无旁白字幕卡；intro/outro 文案来自配置（sleep_channel_name 等）
    script["welcome_en"] = ""
    script["welcome_zh"] = ""
    script["story_hook"] = ""
    script["intro_zh"] = ""
    script["outro"] = ""
    script["outro_zh"] = ""
    script["practice_intro_en"] = ""
    script["practice_intro_zh"] = ""
    script.setdefault("scene", "")
    script.setdefault("thumbnail_expression", "")
    script.setdefault("thumbnail_action", "")
    script.setdefault("thumbnail_subtitle", f"{num_pairs * 2}句睡前英文")
    script.setdefault("thumbnail_icons", [])
    script["dialogue"] = _pairs_to_rows(all_pairs)
    return script
