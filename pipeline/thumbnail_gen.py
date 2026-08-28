"""YouTube thumbnail generator — generates thumbnail with text baked into the AI prompt.

Instead of generating a background then overlaying text via Pillow, this module
puts the title text, level badge, and Chinese subtitle directly into the generate_image
prompt so the AI renders everything in one step.

Fallback: if AI image gen fails, falls back to Pillow text overlay on scene image.
"""
import os
import re
import sys
import json
import subprocess
from pathlib import Path

_PARENT = str(Path(__file__).parent.resolve())
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from media_utils import FONT_EN, FONT_ZH
import sensenova_image

# YouTube thumbnail specs
THUMB_W = 1280
THUMB_H = 720


def _build_thumbnail_prompt(script: dict, structure: str) -> str:
    """Build a prompt that generates a YouTube thumbnail in the reference style.

    Reference style elements:
    - Top-left orange banner: "A1-A2 LEVEL"
    - Top-right green icon: "中英對照"
    - Top center: "沉浸式聽力動畫" (listening) or "沉浸式英文動畫" (original)
    - Main scene: characters in the active visual style + background + props
    - Large text below characters: Traditional Chinese title (e.g. "在藥房買藥英文")
    - Smaller text below: English title (e.g. "AT THE PHARMACY") + subtitle
    - Bottom row of circular icons with bilingual text (scene keywords)

    Key design: the LARGE title is in Traditional Chinese (the audience's native
    language) for maximum CTR, with the English title as a smaller subtitle below.
    """
    from style_manager import get_active_style_prompt, get_active_thumbnail_hint
    style_prompt = get_active_style_prompt()
    thumb_hint = get_active_thumbnail_hint()
    title_en = script.get("title", "ENGLISH LISTENING")
    title_zh = script.get("title_zh", script.get("intro_zh", ""))
    # Build a descriptive Chinese title: topic + "英文" suffix (e.g. "在藥房買藥英文")
    if title_zh and not title_zh.endswith("英文"):
        title_zh_large = f"{title_zh}英文"
    else:
        title_zh_large = title_zh or "日常英語"

    cefr = script.get("cefr", "A2")
    char_a_desc = script.get("char_a_description", "friendly young person")
    char_b_desc = script.get("char_b_description", "friendly young person")
    scene_zh = script.get("scene_zh", script.get("title", "everyday life"))
    scene_en = script.get("scene", script.get("title", "everyday life"))

    expression = script.get("thumbnail_expression", "surprised and excited")
    action = script.get("thumbnail_action", "looking toward the camera and gesturing naturally")
    subtitle = script.get("thumbnail_subtitle", "18句聽力練習")

    icons = script.get("thumbnail_icons", [
        {"en": "Dialogue", "zh": "會話"},
        {"en": "Listening", "zh": "聽力"},
        {"en": "Shadowing", "zh": "跟讀"},
        {"en": "Practice", "zh": "練習"},
    ])
    icon_lines = "  ".join(f"{i['zh']} {i['en']}" for i in icons[:5])

    if structure == "quest":
        top_center = "慢速英文聽力"
    else:
        top_center = "沉浸式英文動畫"

    return f"""A highly complex {thumb_hint} YouTube thumbnail for {scene_en} English listening practice, complete with an orange banner at the top left reading "{cefr} LEVEL" and a green icon at the top right with the text "中英對照". At the top center, the text "{top_center}" is integrated.

The main scene features a detailed view of {scene_en} with {char_a_desc} and {char_b_desc}, both with a {expression} expression, {action}. The background shows a detailed {scene_en} setting with relevant props and environment.

The LARGE bold text below the characters reads "{title_zh_large}" in bright yellow font with thick black outline — this is the main title and must be the most prominent text on the thumbnail. Below this large Chinese title, smaller text reads "{title_en}" in white. Below that, even smaller text reads "{subtitle}".

At the very bottom, a precise row of circular icons is rendered with legible text associated: {icon_lines}.

Clean legible text, bright studio lighting, vibrant colors, highly detailed, professional composition, {style_prompt}, soft shadows, cinematic lighting.

CRITICAL: The largest and most prominent text on the thumbnail must be the Traditional Chinese title "{title_zh_large}". The English title "{title_en}" must be noticeably smaller, serving as a subtitle below the Chinese title. The Chinese audience sees the Chinese title first — it must grab attention."""


def generate_thumbnail(script: dict, scene_img: str, output_path: str,
                        mcp_call_tool=None, mcp_parse_task_id=None,
                        mcp_poll_task=None, mcp_download_file=None,
                        structure: str = "original",
                        char_scene_url: str = None) -> str:
    """Generate a YouTube thumbnail with text baked into the AI prompt (one step).

    If char_scene_url is provided, uses it as a reference image so the thumbnail
    characters match the video's character designs.

    Falls back to Pillow text overlay on scene_img if AI generation fails.
    """
    prompt = _build_thumbnail_prompt(script, structure)

    # If we have a char_scene reference, add instruction to match it
    if char_scene_url:
        prompt += "\n\nIMPORTANT: The characters' appearance, clothing, and hair MUST closely match the uploaded reference image. Use the reference image as the character design guide."

    # SenseNova U1.5 Lite 路径（image_provider=sensenova）：
    # 有 char_scene 参考图走 edits，否则纯文生图；失败直接落 Pillow fallback
    # （不再落 MCP 生成，避免用户选择 sensenova 省积分时被意外消耗）
    if sensenova_image.get_image_provider() == "sensenova":
        print("  [Thumbnail] Generating thumbnail via SenseNova U1.5 Lite...")
        try:
            if char_scene_url:
                url = sensenova_image.edit_image(
                    char_scene_url, prompt,
                    size=sensenova_image.SIZE_MAP["landscape_16_9"],
                    output_format="jpeg")
            else:
                url = sensenova_image.text_to_image(
                    prompt, size=sensenova_image.SIZE_MAP["landscape_16_9"],
                    output_format="jpeg")
            if url and sensenova_image.download_image(url, output_path):
                size_kb = os.path.getsize(output_path) // 1024
                if size_kb > 10:
                    print(f"  [Thumbnail] Saved (U1.5): {output_path} ({size_kb}KB)")
                    return output_path
                print(f"  [Thumbnail] U1.5 image too small ({size_kb}KB), falling back")
            else:
                print("  [Thumbnail] U1.5 generation returned no URL, falling back to Pillow")
        except Exception as e:
            print(f"  [Thumbnail] U1.5 generation failed: {e}, falling back to Pillow")
        print("  [Thumbnail] Using Pillow fallback (text overlay on scene image)...")
        return _pillow_fallback(script, scene_img, output_path, structure)

    # Try AI generation with text baked in
    if mcp_call_tool and mcp_parse_task_id and mcp_poll_task and mcp_download_file:
        print("  [Thumbnail] Generating thumbnail with baked-in text via MCP...")
        try:
            gen_args = {
                "prompt": prompt,
                "provider": "frontier",
                "quality": "high",
                "image_size": '{"width": 1280, "height": 720}',
                "output_format": "jpeg",
            }
            if char_scene_url:
                gen_args["image_urls"] = char_scene_url
                print(f"  [Thumbnail] Using char_scene reference: {char_scene_url[:60]}...")
            result = mcp_call_tool("generate_image", gen_args)
            task_id = mcp_parse_task_id(result)
            if task_id:
                data = mcp_poll_task(task_id, interval=10, max_wait=300)
                url = data.get("url", "")
                if url and mcp_download_file(url, output_path):
                    size_kb = os.path.getsize(output_path) // 1024
                    if size_kb > 10:
                        print(f"  [Thumbnail] Saved (AI baked-in): {output_path} ({size_kb}KB)")
                        return output_path
                    else:
                        print(f"  [Thumbnail] AI image too small ({size_kb}KB), falling back")
                else:
                    print("  [Thumbnail] AI generation returned no URL, falling back to Pillow")
        except Exception as e:
            print(f"  [Thumbnail] AI generation failed: {e}, falling back to Pillow")

    # Fallback: Pillow text overlay on scene image
    print("  [Thumbnail] Using Pillow fallback (text overlay on scene image)...")
    return _pillow_fallback(script, scene_img, output_path, structure)


def _pillow_fallback(script: dict, scene_img: str, output_path: str,
                      structure: str) -> str:
    """Pillow fallback: overlay text on scene image."""
    from PIL import Image, ImageDraw, ImageFont

    if not os.path.exists(scene_img):
        print(f"  [Thumbnail] ERROR: No scene image at {scene_img}")
        return None

    bg = Image.open(scene_img).convert("RGBA").resize((THUMB_W, THUMB_H))

    # Dark gradient on right side
    overlay = Image.new("RGBA", (THUMB_W, THUMB_H), (0, 0, 0, 0))
    ov_draw = ImageDraw.Draw(overlay)
    for x in range(THUMB_W // 2, THUMB_W):
        alpha = int((x - THUMB_W // 2) / (THUMB_W // 2) * 160)
        ov_draw.line([(x, 0), (x, THUMB_H)], fill=(0, 0, 0, alpha))
    ov_draw.rectangle([0, THUMB_H - 80, THUMB_W, THUMB_H], fill=(0, 0, 0, 200))
    bg = Image.alpha_composite(bg, overlay)
    draw = ImageDraw.Draw(bg)

    title_en = script.get("title", "").upper() or "ENGLISH LISTENING"
    title_zh = script.get("title_zh", script.get("intro_zh", ""))
    # Large Chinese title: topic + "英文"
    if title_zh and not title_zh.endswith("英文"):
        title_zh_large = f"{title_zh}英文"
    else:
        title_zh_large = title_zh or "日常英語"
    cefr = script.get("cefr", "A2")
    subtitle = script.get("thumbnail_subtitle", "18句聽力練習")

    STROKE = 8
    MARGIN = 40

    # English title (smaller, white, below Chinese title)
    en_size = 56
    en_font = ImageFont.truetype(FONT_EN, en_size)
    while en_size > 24:
        bbox = draw.textbbox((0, 0), title_en, font=en_font)
        if (bbox[2] - bbox[0]) + STROKE * 2 <= THUMB_W // 2 - MARGIN:
            break
        en_size -= 2
        en_font = ImageFont.truetype(FONT_EN, en_size)
    bbox = draw.textbbox((0, 0), title_en, font=en_font)
    en_w, en_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    en_y = int(THUMB_H * 0.22)
    draw.text((THUMB_W // 2 + (THUMB_W // 2 - en_w) // 2, en_y), title_en,
              font=en_font, fill=(255, 255, 255, 255),
              stroke_width=3, stroke_fill=(0, 0, 0, 255))

    # Large Chinese title (biggest text on thumbnail, yellow + black stroke)
    zh_stroke = 6
    zh_size = 80
    zh_font = ImageFont.truetype(FONT_ZH, zh_size)
    while zh_size > 30:
        bbox = draw.textbbox((0, 0), title_zh_large, font=zh_font)
        if (bbox[2] - bbox[0]) + zh_stroke * 2 <= THUMB_W // 2 - MARGIN:
            break
        zh_size -= 2
        zh_font = ImageFont.truetype(FONT_ZH, zh_size)
    bbox = draw.textbbox((0, 0), title_zh_large, font=zh_font)
    zh_w, zh_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    zh_y = en_y + en_h + 10
    draw.text((THUMB_W // 2 + (THUMB_W // 2 - zh_w) // 2, zh_y), title_zh_large,
              font=zh_font, fill=(255, 220, 0, 255),
              stroke_width=zh_stroke, stroke_fill=(0, 0, 0, 255))

    # Subtitle (smallest, gold)
    if subtitle:
        sub_size = 36
        sub_font = ImageFont.truetype(FONT_ZH, sub_size)
        while sub_size > 18:
            bbox = draw.textbbox((0, 0), subtitle, font=sub_font)
            if (bbox[2] - bbox[0]) + 3 * 2 <= THUMB_W // 2 - MARGIN:
                break
            sub_size -= 2
            sub_font = ImageFont.truetype(FONT_ZH, sub_size)
        bbox = draw.textbbox((0, 0), subtitle, font=sub_font)
        sub_w = bbox[2] - bbox[0]
        draw.text((THUMB_W // 2 + (THUMB_W // 2 - sub_w) // 2, zh_y + zh_h + 10),
                  subtitle, font=sub_font, fill=(255, 200, 80, 255),
                  stroke_width=2, stroke_fill=(0, 0, 0, 255))

    # CEFR badge
    badge_r = 50
    cx, cy = THUMB_W - badge_r - 30, badge_r + 30
    draw.ellipse([cx - badge_r, cy - badge_r, cx + badge_r, cy + badge_r],
                 fill=(220, 50, 50, 255), outline=(255, 255, 255, 255), width=3)
    bfont = ImageFont.truetype(FONT_EN, 42)
    bb = draw.textbbox((0, 0), cefr, font=bfont)
    draw.text((cx - (bb[2] - bb[0]) // 2, cy - (bb[3] - bb[1]) // 2 - 2),
              cefr, font=bfont, fill=(255, 255, 255, 255))

    # Bottom bar
    if structure == "quest":
        label = "Slow Listening + Answer Task"
    else:
        label = "Listen + Repeat + Shadowing"
    lfont = ImageFont.truetype(FONT_EN, 28)
    lb = draw.textbbox((0, 0), label, font=lfont)
    draw.text(((THUMB_W - (lb[2] - lb[0])) // 2, THUMB_H - 55),
              label, font=lfont, fill=(255, 255, 255, 255))

    bg.convert("RGB").save(output_path, "JPEG", quality=90)
    print(f"  [Thumbnail] Saved (Pillow fallback): {output_path}")
    return output_path


# "⏱️ Chapters:" marker line + consecutive chapter timestamp lines (00:00 / 00:xx style)
_CHAPTERS_SECTION = re.compile(
    r"⏱️?\s*Chapters:[^\n]*\n"
    r"(?:[ \t]*[-•*]?[ \t]*(?:[\dx]{1,2}:)?[\dx]{1,2}:[\dx]{2}[^\n]*\n?)+"
)


def _inject_chapters(desc: str, chapters: list[str]) -> str:
    """Inject real chapter timestamps into a description.

    The LLM often leaves placeholder timestamps (00:xx) in its own Chapters
    section — replace that section with the real ones computed from the
    timeline. Append a new section only if none exists.
    """
    if not chapters:
        return desc
    block = "⏱️ Chapters:\n" + "\n".join(chapters)
    if _CHAPTERS_SECTION.search(desc):
        return _CHAPTERS_SECTION.sub(block + "\n", desc, count=1)
    if not desc.strip():
        return block + "\n"
    return desc.rstrip() + "\n\n" + block + "\n"


def save_youtube_metadata(script: dict, timeline: list[dict],
                           output_path: str, structure: str = "original") -> str:
    """Save YouTube metadata (title, description, tags) as JSON.

    Also post-processes the description to insert real timestamps from the timeline.
    """
    # Collect ordered chapter candidates (seconds, label) from the timeline.
    # YouTube chapter rules: the first chapter MUST start at 00:00 and every
    # chapter must span >= 10 seconds — candidates starting less than 10s
    # after the previously kept one are dropped (e.g. quest's short hook_intro).
    if structure == "quest":
        seg_labels = {
            "welcome": "Welcome",
            "hook_intro": "Intro · Listening Task",
            "dialogue": "Slow Dialogue",
            "outro": "Outro · Answer & CTA",
        }
    else:
        seg_labels = {
            "welcome": "Welcome & Hook",
            "dialogue": "Immersive Dialogue",
            "practice_intro": "Shadowing Practice",
            "outro": "Outro",
        }

    marks: list[tuple[float, str]] = []
    seen_types: set[str] = set()
    t_cursor = 0.0
    for seg in timeline:
        seg_type = seg.get("type", "")
        if seg_type in seg_labels and seg_type not in seen_types:
            seen_types.add(seg_type)
            marks.append((t_cursor, seg_labels[seg_type]))
        t_cursor += seg.get("duration", 0)

    def _fmt_ts(seconds):
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m:02d}:{s:02d}"

    chapters = []
    kept_t = 0.0
    for i, (t, label) in enumerate(marks):
        if i == 0:
            kept_t = 0.0  # 首章节必须 00:00，否则 YouTube 禁用全部章节
        elif t - kept_t < 10.0:
            continue
        else:
            kept_t = t
        chapters.append(f"{_fmt_ts(kept_t)} {label}")

    description = script.get("youtube_description", "")
    description = _inject_chapters(description, chapters)

    description_en = script.get("youtube_description_en", "")
    description_en = _inject_chapters(description_en, chapters)

    tags = script.get("youtube_tags", [])
    # Append tags as #hashtag string at the end of description
    hashtags = " ".join(f"#{t.replace(' ', '')}" for t in tags) if tags else ""
    if hashtags:
        description = description.rstrip() + "\n" + hashtags
        if description_en:
            description_en = description_en.rstrip() + "\n" + hashtags

    title = script.get("youtube_title", script.get("title", ""))
    title_en = script.get("youtube_title_en", "")
    # YouTube 标题硬上限 100 字符，超长会被截断 —— 最后防线
    if len(title) > 100:
        print(f"  [YouTube] WARNING: title {len(title)} chars > 100, truncated")
        title = title[:100]
    if len(title_en) > 100:
        print(f"  [YouTube] WARNING: title_en {len(title_en)} chars > 100, truncated")
        title_en = title_en[:100]

    metadata = {
        "title": title,
        "title_en": title_en,
        "description": description,
        "description_en": description_en,
        "tags": tags,
        "chapters": chapters,
    }

    # 多样式选项（title_options）：yt_meta_styles/ 下每个注册样式生成一套备选
    # 标题+简介（如"参考频道同款"），与默认选项并存供上传时多选一。
    try:
        from yt_meta_styles import collect_options
        options = collect_options(script, chapters=chapters, marks=marks, structure=structure)
    except Exception as e:
        print(f"  [YouTube] WARNING: 样式选项收集失败: {e}")
        options = []
    for opt in options:
        if len(opt["title"]) > 100:
            print(f"  [YouTube] WARNING: {opt['style_id']} title {len(opt['title'])} chars > 100, truncated")
            opt["title"] = opt["title"][:100]
        if hashtags:
            opt["description"] = opt["description"].rstrip() + "\n" + hashtags
    if options:
        metadata["title_options"] = options

    Path(output_path).write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  [YouTube] Metadata saved: {output_path}")
    return output_path
