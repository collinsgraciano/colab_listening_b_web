"""脚本风格强化工具包 — few-shot 示例 / 情节装置 / 禁用套路 / 张力要求。

供 llm_client.py（original* 三模式）与 quest/llm_client_quest.py 共用。
由配置 script_style_boost（env SCRIPT_STYLE_BOOST）开关，默认关闭；
关闭时生成链路完全不使用本模块（prompt 与原版逐字一致）。
"""
import random

# ---------------------------------------------------------------------------
# 手写 few-shot 示例（仅风格锚定 — 展示地道口语特征，禁止模型照抄内容）
# ---------------------------------------------------------------------------

_EXAMPLE_1 = [
    ("A", "Hey, I'm here for a mobile order — should be under Dana."),
    ("B", "Let me check... Dana... oh, got two here, actually."),
    ("A", "Oh, one of those is probably mine."),
    ("B", "Large iced latte with oat milk?"),
    ("A", "That's... not what I ordered. I got a hot chocolate."),
    ("B", "Hmm, that's weird. This one says Dana too."),
    ("A", "Well, guess we're both named Dana today."),
    ("B", "You know what, take it — she might've ordered both."),
    ("A", "Seriously? That's so nice of you."),
    ("B", "No worries. Mistakes happen, right?"),
    ("A", "Well, thanks! You just made my morning."),
    ("B", "Enjoy! And hey, if she comes back, we'll fix hers too."),
]

_EXAMPLE_2 = [
    ("A", "Hi, I'd like to return these boots. They didn't fit."),
    ("B", "Sure, no problem. Do you have the receipt?"),
    ("A", "I do, but... it's kind of crumpled. My dog got to it."),
    ("B", "Okay, as long as I can read the barcode."),
    ("A", "Here. Hopefully it still scans."),
    ("B", "Yep, got it. Refund to the same card?"),
    ("A", "Actually, store credit might be better. I'll probably exchange them."),
    ("B", "Smart — we just got new winter stock in."),
    ("A", "Oh perfect, that's exactly why I came back."),
    ("B", "Alright, you're all set. The winter aisle's right over there."),
    ("A", "Great, thanks for making that easy."),
]

# ---------------------------------------------------------------------------
# 情节装置（每次生成随机抽取，打散「打招呼→询问→办完→道谢」套路）
# ---------------------------------------------------------------------------

PLOT_DEVICES = [
    "a small misunderstanding that gets cleared up",
    "an unexpected surprise (a discount, a free upgrade, an extra item)",
    "something is sold out, wrong, or missing",
    "an awkward or funny moment (a misheard name, a silly mistake)",
    "a time-pressure moment (a bus is leaving, a store is closing)",
    "a kind gesture from a stranger or a staff member",
    "a price that is much higher or lower than expected",
    "a confused first-timer learning how something works",
    "a tiny mistake that needs fixing (wrong order, wrong size, typo)",
    "a technical hiccup (system down, card declined, app not working)",
    "an unexpected twist in the middle (a rule changed, a plan changed)",
    "a moment of mild frustration that gets resolved politely",
    "a lucky coincidence (running into someone, the last item left)",
    "a beginner's question that reveals something surprising",
]

# ---------------------------------------------------------------------------
# 禁用套路开场白（教科书腔的最大来源）
# ---------------------------------------------------------------------------

CLICHE_OPENERS = [
    '"Hi, how can I help you?"',
    '"How can I help you today?"',
    '"What can I do for you?"',
    '"Can I take your order?"',
    '"Excuse me, sir"',
    '"Welcome to ..."',
]


def pick_plot_devices(n: int = 2) -> list[str]:
    """随机抽取 n 个情节装置（每次生成不同 → 打散套路）。"""
    n = max(1, min(n, len(PLOT_DEVICES)))
    return random.sample(PLOT_DEVICES, n)


def build_device_block(devices: list[str], topic: str = "") -> str:
    """仅情节装置的精简块（大纲 prompt / 完整风格块共用）。

    topic 非空时在标题行锚定主题（单次生成模式）；为空用通用标题（大纲
    prompt 中主题已单独出现，无需重复锚定）。
    """
    devices_block = "\n".join(f"  - {d}" for d in devices)
    if topic:
        return (f"TODAY'S PLOT DEVICES (MANDATORY): Build the story of this "
                f"\"{topic}\" dialogue around these device(s) — weave them in "
                f"naturally:\n{devices_block}\n")
    return (f"TODAY'S PLOT DEVICES (weave them into the story naturally):\n"
            f"{devices_block}\n")


def build_cliche_block() -> str:
    """禁用套路开场白块（对话 prompt 用）。"""
    ban_lines = "\n".join(f'  - {c}' for c in CLICHE_OPENERS)
    return f"""FORBIDDEN CLICHÉS (hard rules):
- NEVER open the dialogue with a generic greeting or a stock service phrase, e.g.:
{ban_lines}
- Start IN THE MIDDLE of the action, or with something specific to this moment.
- Avoid robotic classroom phrasing ("How do you do?", "It is a pleasure to meet you.")"""


def build_fewshot_block(max_words: int, cefr: str = "") -> str:
    """few-shot 示例块（对话 prompt 用）。"""
    examples = []
    for idx, ex in enumerate((_EXAMPLE_1, _EXAMPLE_2), start=1):
        lines = "\n".join(f'  {sp}: "{t}"' for sp, t in ex)
        examples.append(f"--- Style Example {idx} ---\n{lines}")
    examples_text = "\n\n".join(examples)
    level_note = (f" your lines must still obey the {max_words}-word hard limit "
                  f"and {cefr} level" if cefr else
                  f" your lines must still obey the {max_words}-word hard limit")
    return f"""STYLE EXAMPLES (for TONE only — do NOT copy their topic, content, or any line into your script;{level_note}):
{examples_text}

What makes these examples work: contractions everywhere ("I'm", "that's", "we'll"), filler words ("Well", "You know what", "right?"), sentence fragments and short reactions, back-channeling, a small surprise mid-scene, light humor, and a complete resolution. The first line jumps straight into the action."""


def build_style_boost_section(topic: str, cefr: str, max_words: int,
                              devices: list[str] | None = None) -> str:
    """拼装风格强化 prompt 块（STYLE & STORY UPGRADE）。

    devices: 情节装置列表；None 时内部随机抽取。大纲先行模式下由调用方
    生成一次并传给大纲 prompt 与本函数，保证两阶段使用同一装置。
    """
    if devices is None:
        devices = pick_plot_devices(2)
    devices_block = build_device_block(devices, topic=topic)
    cliche_block = build_cliche_block()
    fewshot_block = build_fewshot_block(max_words, cefr)
    return f"""STYLE & STORY UPGRADE (CRITICAL — this section overrides anything that conflicts with it):

{devices_block}
TENSION & STORY REQUIREMENTS:
- Small but REAL stakes: someone in the dialogue genuinely needs or wants something today (not just polite small talk).
- The story must contain AT LEAST ONE unexpected moment: a surprise, a twist, a funny misunderstanding, or a small problem that needs fixing.
- Characters react like real humans: brief emotional responses (amusement, relief, mild annoyance, surprise) BEFORE moving on.
- The ending must feel complete: the situation resolves, and a character reacts to how it turned out.
- Stay grounded in everyday reality — no melodrama, no stacked coincidences.

{cliche_block}

{fewshot_block}
"""
