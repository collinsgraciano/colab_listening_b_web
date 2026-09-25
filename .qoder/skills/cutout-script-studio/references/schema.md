# Original Cutout Script — Field Contract

A library doc is one JSON file, UTF-8, filename `<id>.json`. The web app
`_seed_from_script_library()` consumes it directly (no LLM).

## Doc wrapper (top level)

```json
{
  "id": "script_cut_<slug>",              // == filename stem; prefix must be "script_"
  "topic": "Ordering Coffee at a Busy Café", // display topic, unique across library
  "cefr": "A2",
  "structure": "original_cutout",
  "llm_provider": "", "llm_model": "",
  "num_lines": 18,
  "created": 1756000000.0,                 // epoch float (any plausible value)
  "status": "draft",
  "review": null,
  "used_by": "", "used_at": 0,
  "script": { ...see below... }
}
```

`slug` = lowercase hyphenated, ≤40 chars, unique.

## script object — required fields (all non-empty unless noted)

### Story & narration (host segments; cutout = quest-style host opening/ending)
| field | rule |
|---|---|
| `lesson_type` | exactly `"listening"` |
| `structure` | exactly `"original_cutout"` (also inside script) |
| `title` | short ALL-CAPS English, e.g. `"AT THE PHARMACY"` |
| `title_zh` | 繁體中文, ≤6 chars, e.g. `"在藥局"` |
| `scene_zh` | 繁中 `"場景 · 動作"`, e.g. `"藥局 · 取藥"` (use `·` with spaces) |
| `story_hook` | 1 English sentence ≤10 words, sets the scene (spoken by host) |
| `intro_zh` | 繁中 translation of story_hook |
| `welcome_en` | 1–2 host sentences, each ≤10 words, warm greeting + hints today's topic |
| `welcome_zh` | 繁中 translation |
| `outro` | closing line(s), each sentence ≤10 words |
| `outro_zh` | 繁中 |
| `practice_intro_en` | instruction before Ch3 跟讀, ≤10 words |
| `practice_intro_zh` | 繁中 |

### Characters (drives pose-sheet image generation; consistency is critical)
| field | rule |
|---|---|
| `char_a_description` | detailed physical English description: gender phrase ("a young woman"/"a middle-aged man"), hair color+style, clothing, build. 25–45 words |
| `char_b_description` | same spec; MUST look different from char_a across the batch of scripts (vary age/hair/outfit) |
| `char_a_gender` / `char_b_gender` | `"male"` or `"female"`, must match the description words |
| `char_a_role` / `char_b_role` | story role, e.g. `"customer"`, `"barista"` |

### Dialogue — `dialogue`: array of exactly 18 objects
```json
{"speaker": "char_a", "text": "...", "phonetic": "/.../", "zh": "..."}
```
- `speaker` strictly alternates: index 0,2,4… = `char_a`; 1,3,5… = `char_b`.
- `text`: English sentence, **≤10 words (HARD LIMIT** — subtitles max 2 lines).
  Natural spoken American English: contractions, fillers, back-channeling.
- `phonetic`: IPA transcription of `text` in `/slashes/`, real IPA symbols
  (ˈ ˌ ə ɜ ʃ ʒ θ ð ŋ ɹ ɪ ɛ ʌ ɔ etc.), linked-sound connected speech.
- `zh`: 繁體中文 translation, natural phrasing, NOT simplified characters.
- **No** `image_prompt` / `video_prompt` / `poses` fields (cutout renders
  stop-motion from pose sheets; those fields belong to other modes).
- Story arc across 18 lines: greeting/setup → request → small problem or
  development → resolution + goodbye. Coherent, single continuous exchange.

### YouTube metadata
| field | rule |
|---|---|
| `youtube_title` | 繁中 high-CTR title, **55–95 chars**. Starts `【…】` tag, `｜` separators, 3–8 emoji, power phrases (e.g. 聽完就能說、不用背多聽就會用). May open with `「title_quote繁中翻譯」` hook. Ends `｜English Topic`. |
| `youtube_title_en` | pure English ≤95 chars, prefer quote-hook: `"Pump Number 3, Please" — Paying Inside at a Gas Station` pattern (use YOUR title_quote) |
| `youtube_description` | 繁中, ≤3000 chars. Line 1 = hook with keyword. NO timestamps. End: 3 hashtags `#EnglishListening #ESL #LearnEnglish` + subscribe CTA |
| `youtube_description_en` | English same rules |
| `youtube_tags` | 15–20 strings, mix EN + 繁中 long-tail SEO |
| `youtube_title_ref` | reference-channel format, **55–95 chars**: `【🎬沉浸式英文動畫】🎧主題英文｜點1・點2・點3｜吸引句｜🌱A2 初級英文｜💡不用背，多聽就會用｜🎧聽力口說練習｜後段逐句跟讀` — 3 content points name what THIS dialogue teaches (繁中, joined by `・`) |
| `youtube_description_ref` | 全繁中, ≤3000: `📌 影片簡介：` + 2–3行鉤子問句 → `這一集帶大家走進…`（敘述場景） → `你會聽到：` 2–4 個 emoji 條目（繁中標題+1–2行說明，引用本對話真實英文短句） → 後半段逐句聽力＋跟讀練習 🎧 段落 → `🌱 適合 A2 初級英文學習者` 段落（不用背，多聽就會用）→ 一句溫暖結尾。無時間軸、無連結、無 hashtag |

### Thumbnail
| field | rule |
|---|---|
| `scene` | English location noun, e.g. `"coffee shop"` |
| `thumbnail_expression` | facial expression phrase, e.g. `"cheerful and smiling"` |
| `thumbnail_action` | what main character does, e.g. `"pointing to a menu"` |
| `thumbnail_subtitle` | 繁中 short, e.g. `"18句聽力練習"` |
| `thumbnail_icons` | 4–5 objects `{"en": "...", "zh": "..."}`, scene keywords, e.g. `{"en":"Oat Milk","zh":"燕麥奶"}` |

## Global language rules
- ALL Chinese text 繁體中文 (never simplified: 说/话/边/门/问… forbidden).
- 繁中 translations faithful + idiomatic.
- English level A2: 5–10 word sentences, high-frequency vocab, present/past
  simple, at most one light idiom per script.

## Direct-use check (what the pipeline reads at runtime)
- Step2 TTS: `dialogue[].text` per speaker voice; narration = `welcome_en`,
  `story_hook` (as hook), `outro`, `practice_intro_en`.
- Step2 images: `char_a/b_description` (+gender) for pose sheets, `scene` for bg.
- Step4 timeline/SRT: `dialogue` + zh.
- Step4.5: `youtube_*`, `thumbnail_*`, `title*`, `scene_zh`, `cefr`.
- Run name comes from `youtube_title` (sanitized) — keep it descriptive.
