"""Story LLM client — 同款三类型（plot/chat/solo）脚本生成，QA 循环与 quest 同构。

结构（~20-30 LLM calls for 150 lines）：
  A. Outline（家庭角色 + 故事概念 + 埋题/回收 + CTA 类型 + NPC 嘉宾）+ Beat sheet
  B. True multi-turn session（组 ≤30 行，复用 quest._session_turn 机制）
  C. Metadata（youtube 字段 + 场景图 + 缩略图字段，数量校验重试）
  D. 程序化门禁（quality_gate_story）
  E. LLM critique（story/language[/engagement] 评审）→ 行数保持的原地修复

env: STORY_KIND / STORY_QA_MAX_ROUNDS（缺省回退 QUEST_QA_MAX_ROUNDS）/ 
SCRIPT_CANDIDATES / SCRIPT_STYLE_BOOST / SCRIPT_ENGAGEMENT_QA /
QUEST_BEAT_LINES / QUEST_MAX_LINE_WORDS（行宽与 quest 共用一套约束链）。

家庭角色设定：STORY_FAMILY_JSON（Web 注入，ensure_ascii）> --family-file >
内置默认（Watson 一家）。char_a~char_d 逐字写入脚本；NPC 嘉宾 char_e 由
LLM 每集设计。复用 quest 的 _chat_and_parse/_session_turn/_pack_groups/
_batch_translate_zh/_fix_zh_traditional 机制。
"""
import json
import sys
import time
from pathlib import Path

_PARENT = str(Path(__file__).parent.parent.resolve())
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from llm_client import (
    _env_get,
    _load_used_listening_summaries,
    _build_character_override_prompt,
    _get_character_overrides,
    _seo_hint_line,
    resolve_max_line_words,
)
from quest.llm_client_quest import (
    _chat_and_parse,
    _session_turn,
    _pack_groups,
    _allocate_beats,
    _batch_translate_zh,
    _fix_zh_traditional,
    _judge_dialogue_compact,
    _gate_issues_digest,
    _env_int,
)
from script_style import (
    build_cliche_block,
    build_device_block,
    build_fewshot_block,
    pick_plot_devices,
)

try:
    from story.quality_gate_story import (
        run_quality_gate, format_report, split_phase_lines_story,
        STORY_KINDS, PHASES, PHASE_RATIOS, SPEAKERS, CTA_TYPES,
    )
except ImportError:
    from quality_gate_story import (
        run_quality_gate, format_report, split_phase_lines_story,
        STORY_KINDS, PHASES, PHASE_RATIOS, SPEAKERS, CTA_TYPES,
    )

# ---------------------------------------------------------------------------
# 家庭角色设定（用户可配置；默认 = Watson 一家）
# ---------------------------------------------------------------------------

DEFAULT_FAMILY = {
    "char_a": {
        "name": "Emma", "gender": "female", "role": "mom",
        "description": ("Emma, a warm and friendly woman in her mid-30s with "
                        "shoulder-length dark brown hair, wearing a casual cream "
                        "sweater and blue jeans. She is patient, kind, and often "
                        "makes gentle jokes."),
    },
    "char_b": {
        "name": "David", "gender": "male", "role": "dad",
        "description": ("David, a cheerful man in his late 30s with short black "
                        "hair and round glasses, wearing a navy polo shirt. He "
                        "loves silly jokes and is always ready to help."),
    },
    "char_c": {
        "name": "Anna", "gender": "female", "role": "daughter",
        "description": ("Anna, a 12-year-old girl with long dark hair in a "
                        "ponytail, wearing a light blue hoodie. She is quiet, "
                        "thoughtful, and loves drawing."),
    },
    "char_d": {
        "name": "Peter", "gender": "male", "role": "son",
        "description": ("Peter, an 8-year-old boy with short messy brown hair, "
                        "wearing a bright orange T-shirt. He is playful, curious, "
                        "and asks lots of questions."),
    },
}

_FAMILY_KEYS = ("char_a", "char_b", "char_c", "char_d")


def load_story_family(family_file: str | None = None) -> dict:
    """家庭角色设定：STORY_FAMILY_JSON env > family_file >
    共享文件 configs/story_family.json（Web 编辑卡写入）> 内置默认（逐槽合并）。

    每槽 {name, gender, role, description}；gender 非法回退内置；description
    为空回退内置。返回 {char_a..char_d: {...}}。
    """
    raw_data = None
    env_json = _env_get("STORY_FAMILY_JSON", "").strip()
    if env_json:
        try:
            raw_data = json.loads(env_json)
            if not isinstance(raw_data, dict):
                raw_data = None
        except json.JSONDecodeError as e:
            print(f"  [Story] WARNING: STORY_FAMILY_JSON 解析失败，回退默认: {e}")
    if raw_data is None and family_file:
        p = Path(family_file)
        if p.exists():
            try:
                raw_data = json.loads(p.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as e:
                print(f"  [Story] WARNING: 家庭设定文件读取失败 {p.name}: {e}")
    if raw_data is None:
        # 与 Web「家庭角色设定」编辑卡共享的配置文件（tts_engine 读
        # kokoro_voice_config.json 同款惯例：仓库 configs/ 目录直读）
        shared = Path(__file__).parent.parent.parent / "configs" / "story_family.json"
        if shared.exists():
            try:
                raw_data = json.loads(shared.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as e:
                print(f"  [Story] WARNING: 共享家庭设定读取失败: {e}")
    family = {}
    for key in _FAMILY_KEYS:
        base = dict(DEFAULT_FAMILY[key])
        slot = (raw_data or {}).get(key)
        if isinstance(slot, dict):
            for k in ("name", "gender", "role", "description"):
                v = str(slot.get(k, "") or "").strip()
                if v:
                    base[k] = v
        if base.get("gender") not in ("male", "female"):
            base["gender"] = DEFAULT_FAMILY[key]["gender"]
        if not str(base.get("description", "")).strip():
            base["description"] = DEFAULT_FAMILY[key]["description"]
        family[key] = base
    return family


def _family_block(family: dict, kind: str) -> str:
    """把家庭设定格式化为 prompt 段落（逐字固定，LLM 不可改写）。"""
    lines = ["THE FAMILY CAST (FIXED — use these EXACT names, genders and looks, "
             "do NOT change or rename them):"]
    for key in _FAMILY_KEYS:
        c = family[key]
        lines.append(f"- {key}: {c['name']} — {c['role']}, {c['description']}")
    if kind == "solo":
        lines.append("ONLY char_a appears in this episode (solo vlog style). "
                     "The other family members may be MENTIONED but never speak "
                     "or appear on screen.")
    elif kind == "chat":
        lines.append("ONLY char_a and the guest (char_e) appear in this episode "
                     "(two-person conversation). The other family members do not "
                     "speak or appear.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Phase A: outline + beat sheet
# ---------------------------------------------------------------------------

_KIND_BRIEF = {
    "plot": (
        "Design a FAMILY STORY episode in the style of a popular ESL listening "
        "channel: an ordinary family day turns into a problem/adventure "
        "(conflict escalates) then resolves warmly. A small concrete DETAIL "
        "must be shown naturally early in the story (the PLANT), and the video "
        "ends by asking the AUDIENCE to recall that detail and write the "
        "answer in the comments (the RECALL question)."),    "chat": (
        "Design a TWO-PERSON DAILY CONVERSATION episode (char_a + one guest): "
        "a slice-of-life activity done together (shopping, cooking, planning a "
        "weekend, visiting a place). Natural teaching moments about everyday "
        "vocabulary. The video ends with a simple open question to the "
        "audience + comment CTA."),
    "solo": (
        "Design a SOLO VLOG episode told by char_a alone to the camera "
        "(morning routine, my family, my hobbies...). Short clear sentences, "
        "lots of simple repetition, friendly teacher tone. The video ends "
        "with a question to the audience + comment CTA."),
}


def _build_outline_prompt(topic: str, cefr: str, kind: str,
                          used_dialogues: list[str] | None = None,
                          devices: list[str] | None = None) -> str:
    device_block = build_device_block(devices) if devices else ""
    used_hint = ""
    if used_dialogues:
        used_hint = ("\nAVOID DUPLICATES — these scenarios already exist:\n"
                     + "\n".join(f"  - {d}" for d in used_dialogues[:10]) + "\n")
    phases_desc = " → ".join(PHASES[kind])
    needs_recall_answer = (kind == "plot")
    recall_field = ('  "recall_answer_en": "the 1-2 word answer to the recall '
                    'question (e.g. \\"a golden cobra\\", \\"chocolate\\")",\n'
                    if needs_recall_answer
                    else '  "recall_answer_en": "",\n')
    guest_block = (
        '  "has_guest": true,\n'
        '  "char_e_description": "guest character physical description (gender, '
        'age, hair, clothing — NOT a family member)",\n'
        '  "char_e_gender": "male or female",\n'
        '  "char_e_role": "guest role in this episode (e.g. shop assistant, '
        'teacher, friend Tom, guard Lewis)",\n'
        if kind != "solo" else
        '  "has_guest": false,\n'
        '  "char_e_description": "",\n'
        '  "char_e_gender": "",\n'
        '  "char_e_role": "",\n'
    )
    cta_hint = {
        "plot": ('"cta_type": pick ONE of ["recall_question", "cliffhanger", '
                 '"golden_words"] — recall_question is the classic choice: the '
                 'video ends by asking viewers to recall the planted detail and '
                 'write it in the comments'),
        "chat": '"cta_type": pick ONE of ["open_question", "simple_cta", "golden_words"]',
        "solo": '"cta_type": pick ONE of ["open_question", "simple_cta"]',
    }[kind]
    if kind == "plot":
        ps_hint = ('"phase_summaries": 3-4 sentence story summary PER phase '
                   'key, each progressive, conflict escalating in the middle '
                   'phases')
    else:
        ps_hint = ('"phase_summaries": 2-3 sentence summary PER phase key')
    return f"""You are an expert ESL video director designing a STORY-DRIVEN listening video for overseas Chinese beginners (A2 level), in the style of popular family-story English channels.

Topic: {topic}
CEFR: {cefr}
Episode type: {kind}
{_KIND_BRIEF[kind]}
Phase structure: {phases_desc}
CEFR Vocabulary Guide:
- A1: basic everyday words, present tense, short sentences (5-8 words)
- A2: common daily phrases, present/past tense, sentences 5-12 words
- B1: moderate vocabulary, mixed tenses, sentences 8-15 words, some idioms
- B2: advanced vocabulary, complex sentences, natural idioms and phrasal verbs

{device_block}CRITICAL — GENDER CONSISTENCY:
- Every character's gender MUST be "male" or "female" and MUST match the physical description.
- The guest (char_e) gender MUST match char_e_description.

{_build_character_override_prompt(quest=True)}{used_hint}
Output JSON ONLY:
{{
  "story_concept": "one-sentence story concept",
  "scene": "English scene name (e.g. museum, kitchen, countryside farmhouse)",
  "recall_question_en": "the question asked to the AUDIENCE at the end (max 14 words){' — about the planted detail' if needs_recall_answer else ''}. NAMING RULE: refer to the planted DETAIL itself (e.g. 'What was hidden inside the vase?') or use ONLY names from the cast above / the guest — NEVER invent a new name",
  "recall_question_zh": "繁體中文 translation",
{recall_field}  {cta_hint},
  "cta_hint_en": "one short English line describing how the finale CTA sounds (e.g. 'Write your answer in the comments. See you next time!')",
  {guest_block}"key_words": [{{"en": "word", "zh": "繁中"}}, ... 4-6 words, taught naturally inside the dialogue],
  {ps_hint}
}}

"phase_summaries" must be a JSON object with EXACTLY these keys: {list(PHASES[kind])}.

Topic: {topic}"""


def _generate_beats(outline: dict, kind: str, phase_counts: dict[str, int],
                    beat_lines: int) -> dict[str, list[dict]]:
    """Per-phase beat plans（链式逐阶段调用；失败回退程序化节拍）。"""
    alloc = {ph: _allocate_beats(phase_counts[ph], beat_lines)
             for ph in PHASES[kind]}
    beats: dict[str, list[dict]] = {}
    prev_events: list[str] = []
    for ph in PHASES[kind]:
        prompt = _build_beat_prompt(outline, kind, ph, alloc, prev_events)
        raw: list = []
        try:
            result = _chat_and_parse(
                prompt, temperature=0.7, max_tokens=4096, reasoning_effort="low",
                label=f"beat_{ph}", retries=4,
                system="You are a story architect planning ESL story-listening video scripts. Output valid JSON only.")
            if isinstance(result, dict):
                raw = result.get(ph, [])
            elif isinstance(result, list):
                raw = result
            raw = [b for b in raw if isinstance(b, dict)]
        except RuntimeError as e:
            print(f"  [LLM] Beat sheet '{ph}' failed ({str(e)[:400]}); "
                  f"using programmatic fallback")
        budgets = alloc[ph]
        while len(raw) > len(budgets):
            raw.pop()
        while len(raw) < len(budgets):
            raw.append({"event": f"{ph} beat {len(raw)+1}: "
                        + (outline.get("phase_summaries", {}) or {}).get(ph, ph)[:120]})
        for b, lines_n in zip(raw, budgets):
            b["lines"] = lines_n
            b["phase"] = ph
            b.setdefault("event", ph)
            b.setdefault("location", outline.get("scene", ""))
            b.setdefault("emotion", "warm and natural")
            b.setdefault("key_words", [])
            spk = b.get("speakers", [])
            allowed = list(SPEAKERS[kind])
            if not isinstance(spk, list):
                spk = []
            spk = [s for s in spk if s in allowed]
            b["speakers"] = spk or allowed[:]
        beats[ph] = raw
        prev_events = [f"{ph[0]}{i+1}: {b.get('event', '')}"
                       for i, b in enumerate(raw)]
    return beats


def _build_beat_prompt(outline: dict, kind: str, phase: str, alloc: dict,
                       prev_events: list[str]) -> str:
    budgets = alloc[phase]
    budget_desc = ", ".join(f"beat{i+1}={n} lines" for i, n in enumerate(budgets))
    prev_hint = ""
    if prev_events:
        prev_hint = ("\nBeats already planned for EARLIER phases (continue this "
                     "story thread, never repeat):\n  "
                     + "\n  ".join(prev_events) + "\n")
    phase_rules = {
        "plot": {
            "opening": ('Cold open: drop the viewer straight into the scene. '
                        'One beat shows the PLANT detail clearly and naturally '
                        '(a character notices/touches/reads it — no meta talk '
                        'about "remembering"). Hook energy in beat 1.'),
            "setup": ('Everyday family warmth, light humor, natural teaching '
                      'moments for key_words. Small hints that something may go '
                      'wrong.'),
            "conflict": ('The problem appears and ESCALATES. Emotions rise, '
                         'stakes get concrete. Family members react in '
                         'character. End on a tense moment.'),
            "twist": ('A turning point: a discovery, a surprise, or a clever '
                      'idea that changes the situation.'),
            "resolution": ('The problem gets solved. Warm family moment, '
                           'gentle lesson learned naturally (no preaching).'),
            "finale": ('Characters talk to the AUDIENCE (breaking the fourth '
                       'wall lightly): ask the recall question, restate the '
                       'answer clearly, invite viewers to write it in the '
                       'comments, warm goodbye.'),
        },
        "chat": {
            "opening": ('Warm start, greet, set up the activity for today.'),
            "activity": ('The main activity in natural steps; teach everyday '
                         'vocabulary in context; small surprises and humor.'),
            "finale": ('Wrap up, ask the audience a simple open question, '
                       'comment CTA, goodbye.'),
        },
        "solo": {
            "opening": ('Friendly intro: "Hello everyone, my name is ..." '
                        'style, introduce today\'s topic.'),
            "body": ('The main content in clear steps; repeat key structures '
                     'naturally; personal little details.'),
            "finale": ('Summarize, ask the audience a question, comment CTA, '
                       'thank viewers and goodbye.'),
        },
    }[kind][phase]
    return f"""Plan the beats for the "{phase}" phase of an ESL story-listening video (type: {kind}).

Story concept: {outline.get("story_concept", "")}
Recall question (asked to the audience at the end): {outline.get("recall_question_en", "")}
Planted detail answer: {outline.get("recall_answer_en", "(n/a)")}
CTA style: {outline.get("cta_type", "")} — {outline.get("cta_hint_en", "")}
Scene: {outline.get("scene", "")}
Key words: {", ".join(w.get("en", "") for w in outline.get("key_words", []))}
Phase summary: {(outline.get("phase_summaries", {}) or {}).get(phase, "")}
{prev_hint}
EXACT line budgets (do NOT change): {budget_desc}

RULES:
1. Each beat is ONE story event at ONE location. Events must be progressive and distinct.
2. {phase_rules}
3. Distribute key_words across beats.
4. Each beat: 1-2 sentence "event" + emotion + which characters are involved ("speakers" hint).

Output JSON ONLY (line counts will be enforced by us):
{{"{phase}": [{{"beat_id": "{phase[0]}1", "location": "...", "event": "...", "emotion": "...", "key_words": ["..."], "speakers": ["char_a","char_b"]}}]}}"""


# ---------------------------------------------------------------------------
# Phase B: multi-turn dialogue session
# ---------------------------------------------------------------------------

_PHASE_RULES = {
    "plot": {
        "opening": (
            'COLD OPEN — no host, no greeting to the camera. The story starts '
            'mid-scene. One character naturally shows/notices the PLANT detail '
            '(the answer to the recall question) as part of the scene. Keep it '
            'playful, never say "remember this".\n'
            'Use at least 1 key_word. Include fillers ("well", "you know", "oh").'),
        "setup": (
            'Everyday family warmth and light humor. Characters talk in '
            'character (mom caring, dad joking, daughter quiet/thoughtful, son '
            'playful). Small hints that something might go wrong.\n'
            'Use at least 1 key_word.'),
        "conflict": (
            'The problem appears and ESCALATES. Reactions stay in character. '
            'Short anxious sentences mixed with longer explanations. End of '
            'this phase should feel tense.\n'
            'Use at least 2 key_words.'),
        "twist": (
            'A turning point: discovery / surprise / clever idea. Surprise '
            'reactions ("Wait—", "No way."). Do NOT fully solve the problem '
            'yet.\n'
            'Use at least 1 key_word.'),
        "resolution": (
            'The problem gets solved step by step. Warm family moment at the '
            'end; a gentle lesson said naturally by a character (no preaching '
            'monologue).\n'
            'Use at least 1 key_word.'),
        "finale": (
            'BREAK THE FOURTH WALL lightly: the family turns to the audience. '
            'Ask the RECALL QUESTION, then restate the ANSWER clearly, then '
            'invite viewers: "write your answer in the comments". Warm goodbye '
            '("See you next time"). This is the LAST phase of the video.\n'
            'The recall question and its answer MUST both be said here.'),
    },
    "chat": {
        "opening": (
            'Warm natural start: greet each other, set up today\'s activity. '
            'No host voice — two real people talking.'),
        "activity": (
            'The main activity in natural steps. Teach everyday vocabulary in '
            'context (one character can briefly explain a word like a real '
            'person would). Include small surprises, preferences, light '
            'negotiation. The guest (char_e) speaks naturally as a friend or '
            'a staff member.'),
        "finale": (
            'Wrap up the activity, express satisfaction, ask the audience ONE '
            'simple open question, invite comments ("tell me in the '
            'comments"), warm goodbye.'),
    },
    "solo": {
        "opening": (
            'Friendly intro to the camera: "Hello everyone, my name is {A}..." '
            'style. Introduce today\'s topic in 2-3 lines. Simple and clear.'),
        "body": (
            'The main content in clear small steps. char_a speaks alone to the '
            'camera; short clear sentences with natural repetition ("I usually '
            'wake up at seven. Then I brush my teeth."). Mentioning family '
            'members is fine but they never speak.'),
        "finale": (
            'Summarize what was shared, ask the audience ONE simple question '
            'about THEIR life, invite comments, thank viewers and say goodbye.'),
    },
}

_DIALOGUE_QUALITY_RULES = """\
NATURAL DIALOGUE QUALITY RULES (MANDATORY — the #1 priority):

1. SENTENCE LENGTH VARIETY: Mix short reactions (3-5 words) with medium (6-10) and longer (11-16). NEVER write 3 consecutive lines that are all 4-6 words.
2. FILLER WORDS (at least 1 per 3-4 lines): "well", "you know", "hmm", "actually", "oh", "I mean", "let me think", "wow"
3. BACK-CHANNELING (every 5-8 lines): "Oh really?", "That makes sense", "Hmm, interesting", "I see", "Right", "Exactly"
4. SAME SPEAKER CAN SPEAK CONSECUTIVELY (2-3 lines) when explaining something.
5. EMOTIONAL VARIETY: curiosity, excitement, nervousness, surprise, relief, humor.
6. FAMILY DYNAMICS: mom (char_a) warm and caring, dad (char_b) loves silly jokes, daughter (char_c) quiet/observant, son (char_d) playful and curious — when they appear.
7. AVOID REPETITIVE PATTERNS: don't start consecutive lines with "Yes," or "No,".
8. STORY CONTINUITY: never restart a settled topic. The plant detail, once shown, must not be re-explained.
9. A2 LANGUAGE: simple words, present/past tense, concrete everyday actions. New key_words get a natural one-line explanation inside the dialogue (e.g. "Bumpy. It means not smooth.").
"""

_WRITER_SYSTEM = (
    "You are an expert ESL dialogue writer for family-story listening videos. "
    "Write natural conversational English. Output valid JSON only.")


def _phase_temperature(phase: str) -> float:
    return {"opening": 0.85, "conflict": 0.8, "twist": 0.85,
            "finale": 0.7}.get(phase, 0.75)


def _generate_dialogue_session(outline: dict, beats: dict, kind: str,
                               cefr: str, family: dict) -> list[dict]:
    """True multi-turn session: full history persists in messages."""
    groups: list[list[dict]] = []
    for ph in PHASES[kind]:
        groups.extend(_pack_groups(beats.get(ph, [])))

    style_section = ""
    if _env_get("SCRIPT_STYLE_BOOST", "").strip().lower() in (
            "1", "true", "yes", "on"):
        mw = resolve_max_line_words("QUEST_MAX_LINE_WORDS")
        style_section = (build_cliche_block() + "\n\n"
                         + build_fewshot_block(mw, cefr) + "\n")

    messages: list[dict] = [{"role": "system", "content": _WRITER_SYSTEM}]
    all_lines: list[dict] = []
    for gi, group in enumerate(groups):
        phases = sorted({b["phase"] for b in group},
                        key=lambda p: PHASES[kind].index(p))
        budget = sum(b["lines"] for b in group)
        prompt = _build_group_prompt(outline, kind, family, group, cefr,
                                     all_lines, first=(gi == 0),
                                     style_section=style_section)
        messages.append({"role": "user", "content": prompt})
        lines = _session_turn(messages, budget,
                              temperature=_phase_temperature(phases[0]))
        _post_validate_lines(lines, group, kind, all_lines)
        all_lines.extend(lines)
        print(f"    group {gi+1}/{len(groups)} [{'+'.join(phases)}]: "
              f"{len(lines)}/{budget} lines")
    return all_lines


def _build_group_prompt(outline: dict, kind: str, family: dict,
                        group: list[dict], cefr: str, all_lines: list[dict],
                        first: bool, style_section: str = "") -> str:
    budget = sum(b["lines"] for b in group)
    mw = resolve_max_line_words("QUEST_MAX_LINE_WORDS")
    phases = sorted({b["phase"] for b in group},
                    key=lambda p: PHASES[kind].index(p))

    head = ""
    if first:
        head = f"""Story: {outline.get("story_concept", "")}
Scene: {outline.get("scene", "")}
Recall question (audience, finale): {outline.get("recall_question_en", "")}
Planted detail: {outline.get("recall_answer_en", "(n/a)")}
CTA style: {outline.get("cta_type", "")}
{_family_block(family, kind)}
Guest: {outline.get("char_e_description", "(no guest this episode)")} — role: {outline.get("char_e_role", "")}
Key words: {", ".join(w.get("en", "") for w in outline.get("key_words", []))}

{_DIALOGUE_QUALITY_RULES}
{style_section}"""
    else:
        head = "Continue the same story/conversation.\n"

    prev_hint = ""
    if all_lines:
        tail = all_lines[-5:]
        prev_hint = ("Last lines written (continue seamlessly from here):\n"
                     + "\n".join(f"  {l.get('speaker','?')}: {l.get('text','')}"
                                 for l in tail) + "\n")

    beats_desc = ""
    for b in group:
        beats_desc += (f"- {b.get('beat_id','?')} ({b['lines']} lines) @ "
                       f"{b.get('location','')}: {b.get('event','')} "
                       f"[emotion: {b.get('emotion','')}"
                       + (f"; characters: {', '.join(b.get('speakers', []))}"
                          if b.get("speakers") else "")
                       + (f"; key_words: {', '.join(b.get('key_words', []))}"
                          if b.get("key_words") else "") + "]\n")

    rules = "\n".join(f"PHASE {ph} RULES:\n{_PHASE_RULES[kind][ph]}"
                      for ph in phases)

    allowed = list(SPEAKERS[kind])
    speaker_rule = (f"Speakers allowed: {', '.join(allowed)}. "
                    if kind == "plot" else
                    f"Speakers allowed: {', '.join(allowed)} ONLY. ")
    on_screen_rule = (
        'on_screen rules: list the characters VISIBLE in that line (2-3 is '
        'typical; solo close-ups like ["char_a"] are fine). The speaker is '
        'always included. NEVER use an empty list. Allowed keys: '
        + ", ".join(allowed) + ".")

    return f"""{head}{prev_hint}
Write the next beats of the story ({budget} lines total):
{beats_desc}
Recall question (asked to audience in finale): {outline.get("recall_question_en", "")}
{("The planted detail answer: " + outline.get("recall_answer_en", "")) if kind == "plot" and outline.get("recall_answer_en") else "Do NOT reveal any planted mystery detail early unless a beat says so."}
CEFR {cefr} level. Each line 4-{mw} words (HARD LIMIT: never exceed {mw} words — subtitles must fit 2 lines), vary length naturally.
{speaker_rule}{on_screen_rule}

{rules}

Output: JSON array of EXACTLY {budget} objects:
[{{"speaker":"char_a","text":"English sentence","phase":"{phases[0]}","zh":"繁體中文翻譯","on_screen":["char_a","char_b"]}}]
Every line MUST include "zh" (Traditional Chinese 繁體中文) and "on_screen".
NO markdown, NO explanation. JSON array ONLY."""


def _post_validate_lines(lines: list[dict], group: list[dict], kind: str,
                         all_lines: list[dict]):
    """Force beat-allocated phase, clamp speakers to kind pool, fill on_screen."""
    allowed = list(SPEAKERS[kind])
    for i, line in enumerate(lines):
        acc = 0
        for b in group:
            acc += b["lines"]
            if i < acc:
                line["phase"] = b["phase"]
                break
        else:
            line["phase"] = group[-1]["phase"] if group else PHASES[kind][0]
        speaker = line.get("speaker", "")
        if speaker not in allowed:
            line["speaker"] = allowed[(len(all_lines) + i) % len(allowed)]
        os_ = line.get("on_screen")
        if isinstance(os_, list):
            os_ = [k for k in os_ if k in allowed]
        if not os_:
            os_ = [line["speaker"]]
        line["on_screen"] = os_
        if not str(line.get("zh", "")).strip():
            line["zh"] = ""


# ---------------------------------------------------------------------------
# Phase C: metadata (youtube fields + scenes + thumbnail)
# ---------------------------------------------------------------------------

def _build_metadata_prompt(topic: str, cefr: str, kind: str,
                           outline: dict, dialogue: list[dict]) -> str:
    import json as _json
    from style_manager import get_active_style_prompt, get_active_thumbnail_hint
    style_prompt = get_active_style_prompt()
    thumb_hint = get_active_thumbnail_hint()
    mw = resolve_max_line_words("QUEST_MAX_LINE_WORDS")
    dialogue_text = _json.dumps(
        [{"s": d.get("speaker", ""), "t": d.get("text", "")} for d in dialogue],
        ensure_ascii=False, separators=(",", ":"))
    kind_title_note = {
        "plot": "It is a FAMILY STORY episode (an everyday adventure with a "
                "problem that gets resolved).",
        "chat": "It is a TWO-PERSON DAILY CONVERSATION episode.",
        "solo": "It is a SOLO VLOG episode told by one person to the camera.",
    }[kind]
    return f"""You are a YouTube content strategist. Generate metadata for this ESL story-listening video.

Topic: {topic} ({cefr})
Episode type: {kind}. {kind_title_note}
Story: {outline.get("story_concept", "")}
Recall question (audience CTA): {outline.get("recall_question_en", "")}
Scene: {outline.get("scene", "")}

Full dialogue:
{dialogue_text}

Generate JSON with these fields:
{{
  "title": "ENGLISH TITLE (e.g. THE MYSTERY OF THE ANCIENT TEMPLE)",
  "title_quote": "the single most catchy dialogue line from the dialogue above, copied VERBATIM (under 10 words), to be used as the title hook",
  "title_zh": "繁中短标题 (max 8 chars)",
  "scene_zh": "繁中場景描述",
  "youtube_title": "高CTR繁中标题 with 【】and ｜ format, e.g. 【英文聽力故事】{{emoji}}{{主題繁中}}｜{{懸念或亮點}}｜{{CEFR}}慢速英文. Total length MUST be ≤95 chars.",
  "youtube_title_en": "high-CTR PURE ENGLISH title in the style: 'Catchy Hook Line' ｜ Short Context ｜ Easy English Listening Story (A2 Level). STRONGLY PREFER opening with title_quote in quotes. Max 100 chars.",
  "youtube_description": "full 繁中 description: story tease, what viewers will practice, key_words, comment CTA mentioning the recall question. Do NOT write chapter timestamps.",
  "youtube_description_en": "full PURE ENGLISH description: story tease, key_words, comment CTA, subscribe note. Do NOT write chapter timestamps.",
  "youtube_tags": ["tag1","tag2",... 15-20 tags]{_seo_hint_line(topic)},
  "thumbnail_expression": "main character expression",
  "thumbnail_action": "main character action",
  "thumbnail_subtitle": "繁中 short subtitle",
  "thumbnail_icons": [{{"en":"word","zh":"繁中"}}, ... 4-5 icons],
  "scene_images": [{{"prompt":"specific scene description with details, 16:9, no people","label":"short English label"}}, ... 8-12 items covering different angles/details of the main location(s)]
}}

VISUAL STYLE (CRITICAL): The video's art style is: "{style_prompt}". EVERY scene_images prompt MUST include this EXACT style descriptor phrase.
THUMBNAIL HINT: {thumb_hint}

JSON ONLY, no markdown."""


def _generate_metadata_validated(topic: str, cefr: str, kind: str,
                                 outline: dict, dialogue: list[dict]) -> dict:
    """数量校验（tags/icons/scene_images）+ 一次定向重试。"""
    meta = _chat_and_parse(
        _build_metadata_prompt(topic, cefr, kind, outline, dialogue),
        temperature=0.6, max_tokens=8192, reasoning_effort="low",
        label="metadata",
        system="You are a YouTube content strategist for ESL story videos. Output valid JSON only.")
    failing = _meta_failing_fields(meta)
    if failing:
        print(f"  [LLM] Metadata fields below target, retrying: {failing}")
        try:
            fixes = _chat_and_parse(
                _build_metadata_fix_prompt(meta, failing),
                temperature=0.6, max_tokens=4096, reasoning_effort="low",
                label="metadata_fix",
                system="You are a YouTube content strategist. Output valid JSON only.")
            if isinstance(fixes, dict):
                for k, v in fixes.items():
                    if v:
                        meta[k] = v
        except RuntimeError as e:
            print(f"  [LLM] Metadata fix failed: {e}")
    return meta


def _meta_failing_fields(meta: dict) -> list[str]:
    failing = []
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


def _build_metadata_fix_prompt(meta: dict, failing: list[str]) -> str:
    import json as _json
    hints = {
        "youtube_tags": "15-20 youtube tags (mix 繁中/English)",
        "thumbnail_icons": "4-5 icons",
        "scene_images": "8-12 scene images (16:9, no people)",
    }
    spec = "\n".join(
        f'- "{f}": {hints.get(f, "")} (current: '
        f"{_json.dumps(meta.get(f), ensure_ascii=False)[:200]})" for f in failing)
    return f"""These metadata fields are below spec. Regenerate ONLY these fields.

Fields to fix:
{spec}

Output JSON with ONLY the failing field names as keys, corrected values.
JSON ONLY, no markdown."""


# ---------------------------------------------------------------------------
# Assembly + programmatic fixes
# ---------------------------------------------------------------------------

def _assemble_script(outline: dict, meta: dict, all_lines: list[dict],
                     kind: str, cefr: str, num_lines: int,
                     family: dict, topic: str) -> dict:
    has_guest = kind != "solo" and bool(outline.get("has_guest", True))
    guest_desc = str(outline.get("char_e_description", "") or "").strip()
    if kind == "chat":
        has_guest = True
        guest_desc = guest_desc or ("a friendly companion who joins char_a "
                                    "for the activity")
    elif kind == "solo":
        has_guest = False
    script = {
        "lesson_type": "listening",
        "structure": "story",
        "story_kind": kind,
        "title": meta.get("title", topic.upper()),
        "title_quote": meta.get("title_quote", ""),
        "cefr": cefr,
        "title_zh": meta.get("title_zh", ""),
        "scene_zh": meta.get("scene_zh", ""),
        "story_concept": outline.get("story_concept", ""),
        "story_hook": outline.get("story_concept", ""),
        "listening_question_en": outline.get("recall_question_en", ""),
        "listening_question_zh": outline.get("recall_question_zh", ""),
        "answer_en": outline.get("recall_answer_en", "") if kind == "plot" else "",
        "cta_type": outline.get("cta_type", "open_question"),
        "cta_hint_en": outline.get("cta_hint_en", ""),
        # 冷开场约束：welcome/hook/outro 一律为空串（时间轴纯对话、TTS 跳过旁白）
        "welcome_en": "",
        "welcome_zh": "",
        "hook_intro_en": "",
        "hook_intro_zh": "",
        "outro": "",
        "outro_zh": "",
        "has_guest": has_guest,
        "key_words": outline.get("key_words", []),
        "scene": outline.get("scene", topic.lower()),
        "scene_images": meta.get("scene_images", []),
        "thumbnail_expression": meta.get("thumbnail_expression", "warm natural smile"),
        "thumbnail_action": meta.get("thumbnail_action", "gesturing naturally"),
        "thumbnail_subtitle": meta.get("thumbnail_subtitle", "英文聽力故事"),
        "thumbnail_icons": meta.get("thumbnail_icons", []),
        "youtube_title": meta.get("youtube_title", ""),
        "youtube_title_en": meta.get("youtube_title_en", ""),
        "youtube_description": meta.get("youtube_description", ""),
        "youtube_description_en": meta.get("youtube_description_en", ""),
        "youtube_tags": meta.get("youtube_tags", []),
        "dialogue": all_lines,
        "_requested_num_lines": num_lines,
    }
    for key in _FAMILY_KEYS:
        c = family[key]
        script[f"{key}_description"] = c["description"]
        script[f"{key}_gender"] = c["gender"]
        script[f"{key}_role"] = c["role"]
    script["char_e_description"] = guest_desc if has_guest else ""
    script["char_e_gender"] = str(outline.get("char_e_gender", "") or "").strip().lower()
    if script["char_e_gender"] not in ("male", "female"):
        script["char_e_gender"] = ""
    script["char_e_role"] = str(outline.get("char_e_role", "") or "").strip()
    return script


def _apply_story_fixes(script: dict, num_lines: int):
    """确定性修复：speaker/on_screen 归一、冷开场字段强空、繁中转换、行数矫正。"""
    kind = str(script.get("story_kind", "plot"))
    if kind not in STORY_KINDS:
        kind = "plot"
    allowed = list(SPEAKERS[kind])
    dialogue = script.get("dialogue", [])
    for line in dialogue:
        speaker = line.get("speaker", "")
        if speaker not in allowed:
            line["speaker"] = allowed[0]
        os_ = line.get("on_screen")
        if isinstance(os_, list) and os_:
            seen = []
            for k in allowed:
                if k in os_ and k not in seen:
                    seen.append(k)
            line["on_screen"] = seen or [line["speaker"]]
        else:
            line["on_screen"] = [line["speaker"]]
    # 冷开场约束：story 模式旁白恒空（防旧脚本混入时生成无用旁白音频/时间轴段）
    for field in ("welcome_en", "welcome_zh", "hook_intro_en", "hook_intro_zh",
                  "outro", "outro_zh"):
        script[field] = ""
    _fix_zh_traditional(script)
    _enforce_line_counts_story(script, num_lines)


def _enforce_line_counts_story(script: dict, num_lines: int):
    """行数矫正（quest 同机制，story 阶段配比）：超→保护首尾各2行后裁尾；
    缺→向最缺行阶段中部插入（一次 LLM 调用）。"""
    kind = str(script.get("story_kind", "plot"))
    if kind not in STORY_KINDS:
        kind = "plot"
    dialogue = script.get("dialogue", [])
    if len(dialogue) == num_lines or num_lines <= 0:
        return
    ref = split_phase_lines_story(num_lines, kind)

    if len(dialogue) > num_lines:
        counts = {ph: 0 for ph in PHASES[kind]}
        for l in dialogue:
            if l.get("phase") in counts:
                counts[l["phase"]] += 1
        drops: set[int] = set()
        for ph, target in ref.items():
            excess = counts[ph] - target
            if excess <= 0:
                continue
            idxs = [i for i, l in enumerate(dialogue)
                    if l.get("phase") == ph]
            protected = set(idxs[:2] + idxs[-2:])
            pool = [i for i in idxs if i not in protected]
            for i in pool[::-1][:excess]:
                drops.add(i)
        if drops:
            script["dialogue"] = [l for i, l in enumerate(dialogue)
                                  if i not in drops]
            print(f"  [Fix] trimmed {len(drops)} lines to meet budget "
                  f"({len(script['dialogue'])}/{num_lines})")
    else:
        try:
            _expand_dialogue_story(script, num_lines, ref, kind)
        except RuntimeError as e:
            print(f"  [Fix] line expansion failed: {e}")


def _expand_dialogue_story(script: dict, num_lines: int, ref: dict, kind: str):
    dialogue = script["dialogue"]
    counts = {ph: 0 for ph in PHASES[kind]}
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
        return "\n".join(f"  {l.get('speaker','?')}: {l.get('text','')}"
                         for l in ls)

    phase_rule = _PHASE_RULES[kind].get(phase, "")
    prompt = f"""Insert {need} more dialogue lines into the "{phase}" phase of this ongoing ESL story.

Lines BEFORE the insertion point:
{fmt(before)}

Lines AFTER the insertion point:
{fmt(after)}

Requirements:
- The new lines fit between them seamlessly (same scene, same story flow).
- Phase rule: {phase_rule}
- Each line: {{"speaker","text","phase":"{phase}","zh":"繁體中文","on_screen":["char_a"]}}
- Exactly {need} lines. JSON array ONLY."""

    result = _chat_and_parse(
        prompt, temperature=0.75, max_tokens=4096, reasoning_effort="low",
        label=f"expand_{phase}", system=_WRITER_SYSTEM)
    lines = result if isinstance(result, list) else []
    lines = [l for l in lines if isinstance(l, dict)][:need]
    for l in lines:
        l["phase"] = phase
        if not l.get("on_screen"):
            l["on_screen"] = [l.get("speaker", "char_a")]
    dialogue[at:at] = lines
    print(f"  [Fix] expanded {len(lines)} lines in '{phase}' "
          f"({len(dialogue)}/{num_lines})")


# ---------------------------------------------------------------------------
# Phase E: LLM critique + targeted patch repair（quest 同构，story 化）
# ---------------------------------------------------------------------------

_MAX_PATCHES = 8
_PATCH_MERGE_GAP = 5
_QA_HARD_CAP = 10


def _build_story_judge_prompt(kind_judge: str, script: dict,
                              report: dict) -> str:
    dialogue_json = _judge_dialogue_compact(script, with_zh=kind_judge == "language")
    outline_block = f"""Story: {script.get('story_concept', '')}
Story kind: {script.get('story_kind', 'plot')}
Recall question (finale, to audience): {script.get('listening_question_en', '')}
Planted detail answer: {script.get('answer_en', '')} (by design it MAY appear early — the finale must recall it)
CEFR level: {script.get('cefr', 'A2')}
Family roles: char_a=mom, char_b=dad, char_c=daughter, char_d=son, char_e=guest.
"""
    machine = f"""Machine checks already found (do NOT repeat these):
{_gate_issues_digest(report)}
"""
    if kind_judge == "story":
        task = """You are a STORY COHERENCE judge for an ESL story-listening video script.
Find ONLY story-level problems the machine checks cannot detect:
1. Story regression: a later line re-introduces a settled topic or re-shows the plant detail as if new.
2. The finale does not naturally turn to the audience / does not ask the recall question clearly.
3. INVENTED NAMES: the recall question (or any line) mentions a character name that is not in the cast
   (family members' names from the script char_*_description, the guest's name, or generic roles) — flag it.
4. Character personality drift (mom warm, dad joking, daughter quiet, son playful, guest consistent).
5. Abrupt transitions; the conflict escalation feels flat; the resolution feels cut off.
6. Facts contradict each other across the story (names, places, times, objects).
Report each problem as a line range [start, end] (0-indexed, inclusive)."""
    elif kind_judge == "engagement":
        task = """You are an ENGAGEMENT judge for an ESL story-listening video script (audience: overseas Chinese learners who want vivid family stories).
The script may be technically correct but DULL. Find the DULLEST stretches (2-6 consecutive lines) and say how to make them vivid:
- add a genuine human reaction (amusement, surprise, relief, mild annoyance)
- add a light joke, a personal remark, or a back-channel
- replace textbook phrasing with the way people actually talk (contractions, fragments)
- vary line length (very short reactions vs fuller sentences)
Rules: do NOT break the story arc or the finale recall, do NOT exceed the per-line word limit. Report each dull stretch as a line range [start, end] (0-indexed, inclusive)."""
    else:
        task = """You are a LANGUAGE QUALITY judge for an ESL story-listening video script (target: overseas Chinese beginners).
Find ONLY language-level problems the machine checks cannot detect:
1. Robotic stretches: 3+ consecutive lines with flat rhythm or same sentence structure.
2. Unnatural English: textbook phrasing; wrong register for the CEFR level.
3. zh translation errors: mistranslation, stiff machine-like 繁體中文, inconsistent terminology.
4. Repetitive openers or verbal tics the machine didn't flag.
Report each problem as a line range [start, end] (0-indexed, inclusive)."""
    return f"""{outline_block}{machine}
{task}

Dialogue (compact JSON, i = line index):
{dialogue_json}

Output JSON ONLY:
{{"issues": [{{"type": "...", "lines": [start, end], "problem": "...", "fix_hint": "how to fix"}}]}}

If everything is fine, output {{"issues": []}}. Max 10 issues, most important first."""


def _parse_story_judge_issues(raw, n_lines: int) -> list[dict]:
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
        if s > e or not str(item.get("problem", "")).strip():
            continue
        issues.append({
            "type": str(item.get("type", "other")),
            "lines": [s, e],
            "problem": str(item["problem"])[:300],
            "fix_hint": str(item.get("fix_hint", ""))[:300],
        })
    return issues[:10]


def _story_critique(script: dict, report: dict) -> list[dict]:
    kinds = ["story", "language"]
    if _env_get("SCRIPT_ENGAGEMENT_QA", "").strip().lower() in (
            "1", "true", "yes", "on"):
        kinds.append("engagement")
    combined = []
    for kj in kinds:
        try:
            result = _chat_and_parse(
                _build_story_judge_prompt(kj, script, report),
                temperature=0.3, max_tokens=8192, reasoning_effort="low",
                label=f"judge_{kj}",
                system="You are a strict script quality judge for ESL videos. Output valid JSON only.")
        except RuntimeError as e:
            print(f"  [QA] {kj} judge failed: {e}")
            continue
        raw = (result.get("issues", []) if isinstance(result, dict)
               else (result if isinstance(result, list) else []))
        found = _parse_story_judge_issues(raw, len(script.get("dialogue", [])))
        if kj == "engagement":
            found = found[:3]
        print(f"    {kj} judge: {len(found)} issues")
        combined.extend(found)
    return combined


def _merge_story_patches(report: dict, judge_issues: list[dict],
                         script: dict) -> list[tuple[int, int, list[str]]]:
    dialogue = script.get("dialogue", [])
    n = len(dialogue)
    spans: list[tuple[int, int, str]] = []
    for issue in report.get("issues", []):
        if issue["severity"] != "error" or not issue.get("lines"):
            continue
        ls = issue["lines"]
        detail = issue.get("detail", "")
        if "完全重复行" in detail:
            later = max(ls)
            spans.append((max(0, later - 1), min(n - 1, later + 1),
                          "Exact duplicate of an earlier line — rewrite with "
                          "different wording."))
        elif "超长" in detail:
            for l in ls:
                spans.append((max(0, l - 1), min(n - 1, l + 1),
                              "Line too long — split or shorten it (hard "
                              "word limit)."))
        elif "speaker 非法" in detail:
            for l in ls:
                spans.append((l, l, "Fix the speaker (must be an allowed "
                                    "character for this story type)."))
        elif "finale 缺少回收问题" in detail:
            fl = [i for i, x in enumerate(dialogue)
                  if x.get("phase") == "finale"]
            if fl:
                spans.append((fl[0], fl[-1],
                              "The finale must ask the audience the recall "
                              f"question: {script.get('listening_question_en', '')} "
                              "and invite viewers to write the answer in the "
                              "comments."))
        elif "行数" in detail and "!= 目标" in detail:
            continue  # 行数由 _enforce_line_counts 处理，不进 patch
        else:
            s, e = ls[0], ls[-1]
            spans.append((max(0, min(s, e)), min(n - 1, max(s, e)), detail))
    for issue in judge_issues:
        s, e = issue["lines"]
        hint = f"[{issue['type']}] {issue['problem']} → {issue['fix_hint']}"
        spans.append((s, e, hint))
    if not spans:
        return []
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


def _repair_story_patches(script: dict,
                          patches: list[tuple[int, int, list[str]]],
                          cefr: str, family: dict):
    """原位重写 patch 行段，严格保持行数。"""
    kind = str(script.get("story_kind", "plot"))
    dialogue = script["dialogue"]
    question = script.get("listening_question_en", "")

    for s, e, hints in sorted(patches, key=lambda p: -p[0]):
        k = e - s + 1
        orig = dialogue[s:e + 1]
        ctx_before = dialogue[max(0, s - 3):s]
        ctx_after = dialogue[e + 1:e + 4]
        phases = sorted({l.get("phase", "") for l in orig},
                        key=lambda p: PHASES.get(kind, PHASES["plot"]).index(p)
                        if p in PHASES.get(kind, PHASES["plot"]) else 99)
        rules = "\n".join(_PHASE_RULES.get(kind, {}).get(p, "") for p in phases)
        recall_block = ""
        if "finale" in phases:
            recall_block = (f"\nFINALE requirement — ask the audience: "
                            f"{question} | The answer to restate: "
                            f"{script.get('answer_en', '')} | CTA: write it in "
                            f"the comments.\n")

        def fmt(ls):
            return "\n".join(f"  {l.get('speaker','?')}: {l.get('text','')}"
                             for l in ls)

        orig_json = json.dumps(
            [{"speaker": l.get("speaker"), "text": l.get("text"),
              "phase": l.get("phase"), "zh": l.get("zh"),
              "on_screen": l.get("on_screen")} for l in orig],
            ensure_ascii=False, separators=(",", ":"))
        mw = resolve_max_line_words("QUEST_MAX_LINE_WORDS")

        prompt = f"""Rewrite EXACTLY {k} dialogue lines (indices {s}-{e}) of an ongoing ESL story.

Story: {script.get('story_concept', '')}
Story type: {kind}
{_family_block(family, kind)}
{recall_block}
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
- Allowed speakers: {", ".join(SPEAKERS[kind])}.
- CEFR {cefr}. Natural conversational English with fillers and varied length.
- Each line AT MOST {mw} words (HARD LIMIT).
- Every line keeps fields: speaker/text/phase/zh(繁體中文)/on_screen.
- Seamless continuity with the before/after context lines.

Output: JSON array of exactly {k} objects. JSON ONLY."""

        messages = [{"role": "system", "content": _WRITER_SYSTEM},
                    {"role": "user", "content": prompt}]
        lines = None
        try:
            content = _chat_and_parse_raw(messages)
            lines = content if isinstance(content, list) else []
            lines = [l for l in lines if isinstance(l, dict)]
            if len(lines) != k:
                got = len(lines)
                messages.append({"role": "assistant", "content": json.dumps(lines, ensure_ascii=False)})
                messages.append({"role": "user", "content":
                                 f"You returned {got} lines; exactly {k} are "
                                 f"required. Return the corrected JSON array of "
                                 f"exactly {k} line objects. JSON ONLY."})
                content = _chat_and_parse_raw(messages)
                lines = content if isinstance(content, list) else []
                lines = [l for l in lines if isinstance(l, dict)]
        except RuntimeError as ex:
            print(f"    patch L{s}-{e} failed: {ex}")
            continue

        if lines and len(lines) == k:
            allowed = list(SPEAKERS[kind])
            for new, old in zip(lines, orig):
                new["phase"] = old.get("phase", new.get("phase"))
                sp = new.get("speaker", "")
                if sp not in allowed:
                    new["speaker"] = old.get("speaker", allowed[0])
                os_ = new.get("on_screen")
                if isinstance(os_, list):
                    os_ = [c for c in os_ if c in allowed]
                new["on_screen"] = os_ or [new["speaker"]]
            dialogue[s:e + 1] = lines
            print(f"    patch L{s}-{e}: rewritten ({len(hints)} hints)")
        else:
            print(f"    patch L{s}-{e}: kept original "
                  f"(got {len(lines) if lines else 0}/{k})")

    empty = [(i, l.get("text", "")) for i, l in enumerate(dialogue)
             if not str(l.get("zh", "")).strip()]
    if empty:
        zh_map = _batch_translate_zh(empty)
        for i, zh in zh_map.items():
            dialogue[i]["zh"] = zh


def _chat_and_parse_raw(messages: list[dict]):
    """chat + parse（重试 2 次），供 patch 修复使用（quest._chat_and_parse 面向单 prompt）。"""
    content = None
    for attempt in range(2):
        content = _quest_chat(messages, temperature=0.6, max_tokens=4096,
                              reasoning_effort="low")
        try:
            from llm_client import _extract_json
            return _extract_json(content)
        except Exception as e:
            print(f"    [LLM patch parse retry {attempt+1}/2] {e}")
            time.sleep(3)
    return None


def _quest_chat(messages, **kwargs):
    from llm_client import _chat
    return _chat(messages, **kwargs)


# ---------------------------------------------------------------------------
# Public entry — orchestrates phases A-E
# ---------------------------------------------------------------------------

def _generate_story_raw(topic: str, cefr: str, kind: str, lessons_dir: str,
                        num_lines: int, family: dict) -> dict:
    """Phase A-D 单次生成（大纲→节拍→对话→元数据 + 程序化修复）。"""
    used_summaries = _load_used_listening_summaries(lessons_dir)
    beat_lines = _env_int("QUEST_BEAT_LINES", 10)

    style_boost = _env_get("SCRIPT_STYLE_BOOST", "").strip().lower() in (
        "1", "true", "yes", "on")
    devices = pick_plot_devices(2) if style_boost else []
    if style_boost:
        print(f"  [LLM] Style boost ON, plot devices: {devices}")
    print("  [LLM] Phase A: story outline...")
    outline = _chat_and_parse(
        _build_outline_prompt(topic, cefr, kind, used_summaries, devices=devices),
        temperature=0.9, max_tokens=4096, reasoning_effort="medium",
        label="outline",
        system="You are an expert ESL video director designing story-listening videos for overseas Chinese beginners. Output valid JSON only.")
    # outline 兜底
    if not str(outline.get("recall_question_en", "")).strip():
        outline["recall_question_en"] = "What is the story about?"
    if kind == "plot" and not str(outline.get("recall_answer_en", "")).strip():
        outline["recall_answer_en"] = ""
    phase_counts = split_phase_lines_story(num_lines, kind)
    print(f"    phases: " + ", ".join(f"{p}={n}" for p, n in phase_counts.items()))

    print(f"  [LLM] Phase A: beat sheet (beat_lines={beat_lines})...")
    beats = _generate_beats(outline, kind, phase_counts, beat_lines)

    print("  [LLM] Phase B: multi-turn dialogue session...")
    all_lines = _generate_dialogue_session(outline, beats, kind, cefr, family)
    print(f"    generated {len(all_lines)} dialogue lines")

    empty_zh = [(i, d.get("text", "")) for i, d in enumerate(all_lines)
                if not str(d.get("zh", "")).strip()]
    if empty_zh:
        print(f"  [LLM] {len(empty_zh)} lines have empty zh, batch translating...")
        zh_map = _batch_translate_zh(empty_zh)
        for i, zh in zh_map.items():
            all_lines[i]["zh"] = zh

    print("  [LLM] Phase C: metadata...")
    meta = _generate_metadata_validated(topic, cefr, kind, outline, all_lines)

    script = _assemble_script(outline, meta, all_lines, kind, cefr,
                              num_lines, family, topic)

    # 行级 phase 兜底 + speaker/on_screen 归一 + 冷开场字段强空
    seq = PHASES[kind]
    acc = 0
    bounds = {}
    for ph in seq:
        bounds[ph] = (acc, acc + phase_counts[ph])
        acc += phase_counts[ph]
    for i, line in enumerate(script["dialogue"]):
        if line.get("phase") not in seq:
            for ph, (lo, hi) in bounds.items():
                if lo <= i < hi:
                    line["phase"] = ph
                    break
            else:
                line["phase"] = seq[-1]

    # Force-override character genders from CHARACTER_OVERRIDES（绑定/素材库复用）
    overrides = _get_character_overrides()
    for key in _FAMILY_KEYS + ("char_e",):
        if key in overrides:
            gender = str(overrides[key].get("gender", "") or "").strip().lower()
            if gender:
                script[f"{key}_gender"] = gender
                print(f"  [LLM] Override {key}_gender = {gender}")

    _apply_story_fixes(script, num_lines)
    return script


def generate_story_script(topic: str, cefr: str = "A2",
                          lessons_dir: str = None,
                          num_lines: int = 150,
                          family: dict | None = None,
                          family_file: str | None = None,
                          story_kind: str = "") -> dict:
    """Generate a story-mode script via multi-round LLM calls (Phase A-E).

    Args:
        family: 家庭角色设定 dict（pipeline 传入）；None 时读
            STORY_FAMILY_JSON env → family_file → 共享文件 → 内置默认。
        family_file: 家庭设定 JSON 文件路径（CLI --family-file）。
        story_kind: plot|chat|solo；空串读 STORY_KIND env，再回退 plot。
    """
    kind = (story_kind or _env_get("STORY_KIND", "") or "plot").strip().lower()
    if kind not in STORY_KINDS:
        print(f"  [LLM] WARNING: unknown story_kind '{kind}', fallback to plot")
        kind = "plot"
    if family is None:
        family = load_story_family(family_file)
    if num_lines < len(PHASES[kind]) * 2:
        num_lines = len(PHASES[kind]) * 2

    candidates = max(1, min(3, _env_int("SCRIPT_CANDIDATES", 1)))
    qa_rounds = _env_int("STORY_QA_MAX_ROUNDS",
                         _env_int("QUEST_QA_MAX_ROUNDS", 3))

    if candidates > 1:
        cap0 = resolve_max_line_words("QUEST_MAX_LINE_WORDS")
        best, best_key = None, None
        for ci in range(candidates):
            try:
                cand = _generate_story_raw(topic, cefr, kind, lessons_dir,
                                           num_lines, family)
            except RuntimeError as e:
                print(f"  [LLM] Candidate {ci + 1}/{candidates} failed: {e}")
                continue
            report = run_quality_gate(cand, num_lines, max_line_words=cap0)
            key = (report["n_errors"], report["n_warnings"])
            print(f"  [LLM] Candidate {ci + 1}/{candidates}: "
                  f"errors={report['n_errors']} warnings={report['n_warnings']}")
            if best_key is None or key < best_key:
                best, best_key = cand, key
        if best is None:
            raise RuntimeError("Story script generation failed: all candidates failed")
        script = best
    else:
        script = _generate_story_raw(topic, cefr, kind, lessons_dir,
                                     num_lines, family)

    # ── Phase D+E: quality gate + critique/repair loop ──────────────────
    _apply_story_fixes(script, num_lines)
    cap = resolve_max_line_words("QUEST_MAX_LINE_WORDS")
    rounds_log = []
    if qa_rounds > 0:
        hard_cap = max(_QA_HARD_CAP, qa_rounds)
        print(f"  [QA] quality loop: min {qa_rounds} rounds, "
              f"error-repair up to {hard_cap} rounds")
        for rnd in range(1, hard_cap + 1):
            report = run_quality_gate(script, num_lines, max_line_words=cap)
            print(f"  [QA] Round {rnd}: gate errors={report['n_errors']} "
                  f"warnings={report['n_warnings']}, running LLM judges...")
            judge_issues = _story_critique(script, report)
            rounds_log.append({"round": rnd, "gate": report,
                               "judge_issues": judge_issues})
            patches = _merge_story_patches(report, judge_issues, script)
            if patches:
                print(f"  [QA] Round {rnd}: repairing {len(patches)} patches: "
                      + ", ".join(f"L{p[0]}-{p[1]}" for p in patches))
                _repair_story_patches(script, patches, cefr, family)
                _apply_story_fixes(script, num_lines)
            if rnd >= qa_rounds and report["n_errors"] == 0:
                print(f"  [QA] Round {rnd}: done (0 errors, {rnd} rounds run)")
                break
            if report["n_errors"] > 0 and not patches:
                print(f"  [QA] Round {rnd}: errors remain but no actionable "
                      f"patches, accepting best effort")
                break

    final = run_quality_gate(script, num_lines, max_line_words=cap)
    script["_qa"] = {"rounds": rounds_log, "final": final}
    print(format_report(final))
    if not final["passed"]:
        print(f"  [QA] WARNING: {final['n_errors']} errors remain after "
              f"{len(rounds_log)} QA rounds (accepted best effort)")
    return script
