"""sleep 模式画面卡片：纯 Pillow 渲染（截图 1:1 布局，配色/文案全部配置化）。

布局（1280x720，参照参考视频截图）：
浅绿渐变背景 + 角落叶片装饰 → 白色圆角描边卡片 → 左上手写体频道名 →
右上圆角「EN」徽标 → A 句块（深棕 EN / 橄榄绿 IPA / 深灰繁中）→
B 句块（橙 EN / IPA / 繁中）→ 左下粉色发光序号。
IPA 用 cambria（msyh 渲染 IPA 会变豆腐块），EN 句子用 Nunito Bold（回退 msyhbd）。
"""
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from media_utils import FONT_EN, FONT_ZH, FONT_PH, TARGET_W, TARGET_H

_FONTS_DIR = Path(__file__).resolve().parent.parent / "fonts"
_NUNITO = str(_FONTS_DIR / "Nunito-Bold.ttf")
_FONT_EN_CARD = _NUNITO if os.path.exists(_NUNITO) else FONT_EN
_FONT_HANDWRITE_CANDIDATES = [
    r"C:\Windows\Fonts\Inkfree.ttf",
    r"C:\Windows\Fonts\segoepr.ttf",
]

# 默认主题（与 configs sleep_color_* 配置键一一对应；hex 取自参考截图采样）
DEFAULT_THEME = {
    "bg_top": "#eaf4e2",
    "bg_bottom": "#d9edcf",
    "card": "#ffffff",
    "card_border": "#8fb0c9",
    "en_a": "#422006",
    "en_b": "#e05a12",
    "phonetic": "#7d8c1e",
    "zh_text": "#3a3a3a",
    "num": "#ff6fa5",
    "badge_bg": "#f2b8c6",
    "badge_text": "#ffffff",
    "channel_text": "#5a6b52",
    "leaf_a": "#c9e2f2",
    "leaf_b": "#bfe0c8",
}


def _hex_rgb(value, fallback: tuple) -> tuple:
    try:
        v = str(value).strip().lstrip("#")
        if len(v) == 3:
            v = "".join(c * 2 for c in v)
        return tuple(int(v[i:i + 2], 16) for i in (0, 2, 4))
    except Exception:
        return fallback


def build_theme(cfg: dict) -> dict:
    """配置 dict（含 sleep_color_* / sleep_show_leaves 键）→ 渲染主题 dict。"""
    cfg = cfg or {}
    theme = dict(DEFAULT_THEME)
    mapping = {
        "sleep_color_bg_top": "bg_top", "sleep_color_bg_bottom": "bg_bottom",
        "sleep_color_card": "card", "sleep_color_card_border": "card_border",
        "sleep_color_en_a": "en_a", "sleep_color_en_b": "en_b",
        "sleep_color_phonetic": "phonetic", "sleep_color_zh": "zh_text",
        "sleep_color_num": "num", "sleep_color_badge_bg": "badge_bg",
        "sleep_color_badge_text": "badge_text", "sleep_color_channel": "channel_text",
        "sleep_color_leaf": "leaf_a",
    }
    for ck, tk in mapping.items():
        v = str(cfg.get(ck, "") or "").strip()
        if v:
            theme[tk] = v
    theme["show_leaves"] = bool(cfg.get("sleep_show_leaves", True))
    theme["handwrite_font"] = str(cfg.get("sleep_handwrite_font", "") or "").strip()
    return theme


def _handwrite_path(theme: dict) -> str:
    custom = str(theme.get("handwrite_font", "") or "").strip()
    if custom and os.path.exists(custom):
        return custom
    for p in _FONT_HANDWRITE_CANDIDATES:
        if os.path.exists(p):
            return p
    return FONT_EN


def _fit_font(draw: ImageDraw.ImageDraw, text: str, font_path: str,
              start_size: int, min_size: int, max_w: int) -> ImageFont.FreeTypeFont:
    size = start_size
    while size > min_size:
        font = ImageFont.truetype(font_path, size)
        box = draw.textbbox((0, 0), text, font=font)
        if box[2] - box[0] <= max_w:
            return font
        size -= 4
    return ImageFont.truetype(font_path, min_size)


def _wrap_words(draw: ImageDraw.ImageDraw, text: str, font, max_w: int) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    lines, cur = [], words[0]
    for wd in words[1:]:
        trial = f"{cur} {wd}"
        box = draw.textbbox((0, 0), trial, font=font)
        if box[2] - box[0] <= max_w:
            cur = trial
        else:
            lines.append(cur)
            cur = wd
    lines.append(cur)
    return lines


def _draw_text_block(img: Image.Image, draw: ImageDraw.ImageDraw, en: str,
                     phonetic: str, zh: str, cy: int, en_color, theme: dict,
                     en_start: int = 72) -> None:
    """居中一块：EN 大字 → IPA（cambria）→ 繁中。"""
    w = img.width
    max_w = w - 240
    en_font = _fit_font(draw, en, _FONT_EN_CARD, en_start, 30, max_w)
    en_lines = _wrap_words(draw, en, en_font, max_w)
    line_h = en_font.size + 14
    ph_font = ImageFont.truetype(FONT_PH, 34)
    zh_font = ImageFont.truetype(FONT_ZH, 42)
    ph_h = (ph_font.size + 10) if phonetic else 0
    zh_h = (zh_font.size + 12) if zh else 0
    y = cy - (len(en_lines) * line_h + ph_h + zh_h) / 2
    for ln in en_lines:
        box = draw.textbbox((0, 0), ln, font=en_font)
        draw.text(((w - (box[2] - box[0])) / 2, y), ln, font=en_font, fill=en_color)
        y += line_h
    if phonetic:
        box = draw.textbbox((0, 0), phonetic, font=ph_font)
        draw.text(((w - (box[2] - box[0])) / 2, y), phonetic, font=ph_font,
                  fill=_hex_rgb(theme["phonetic"], (125, 140, 30)))
        y += ph_h
    if zh:
        box = draw.textbbox((0, 0), zh, font=zh_font)
        draw.text(((w - (box[2] - box[0])) / 2, y), zh, font=zh_font,
                  fill=_hex_rgb(theme["zh_text"], (58, 58, 58)))


def _draw_leaf(base: Image.Image, cx: int, cy: int, lw: int, lh: int,
               angle: float, color: tuple) -> None:
    leaf = Image.new("RGBA", (lw * 3, lh * 3), (0, 0, 0, 0))
    ld = ImageDraw.Draw(leaf)
    cx0, cy0 = lw, lh
    ld.ellipse([cx0, cy0, cx0 + lw, cy0 + lh], fill=color,
               outline=(255, 255, 255, 210), width=2)
    ld.line([cx0 + lw // 2, cy0 + 2, cx0 + lw // 2, cy0 + lh - 2],
            fill=(255, 255, 255, 170), width=2)
    leaf = leaf.rotate(angle, expand=True, resample=Image.BICUBIC)
    base.paste(leaf, (int(cx - leaf.width / 2), int(cy - leaf.height / 2)), leaf)


def _draw_leaves(base: Image.Image, theme: dict) -> None:
    if not theme.get("show_leaves", True):
        return
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    ld = ImageDraw.Draw(layer)
    ca = (*_hex_rgb(theme["leaf_a"], (201, 226, 242)), 215)
    cb = (*_hex_rgb(theme["leaf_b"], (191, 224, 200)), 215)
    _draw_leaf(layer, 70, 645, 95, 46, -35, ca)
    _draw_leaf(layer, 28, 600, 82, 42, -10, ca)
    _draw_leaf(layer, 115, 690, 88, 44, -60, ca)
    _draw_leaf(layer, 1212, 655, 98, 48, 30, cb)
    _draw_leaf(layer, 1256, 608, 82, 40, 60, cb)
    _draw_leaf(layer, 1168, 698, 88, 42, 10, cb)
    base.paste(layer, (0, 0), layer)


def _draw_background(w: int, h: int, theme: dict) -> Image.Image:
    top = _hex_rgb(theme["bg_top"], (234, 244, 226))
    bottom = _hex_rgb(theme["bg_bottom"], (217, 237, 207))
    base = Image.new("RGB", (w, h))
    for y in range(h):
        t = y / max(1, h - 1)
        base.paste(tuple(int(top[c] + (bottom[c] - top[c]) * t) for c in range(3)),
                   [0, y, w, y + 1])
    _draw_leaves(base, theme)
    return base


def _draw_card_base(theme: dict, w: int, h: int) -> tuple[Image.Image, ImageDraw.ImageDraw, dict]:
    img = _draw_background(w, h, theme).convert("RGBA")
    draw = ImageDraw.Draw(img)
    card = {"x0": int(w * 0.028), "y0": int(h * 0.042),
            "x1": int(w * 0.972), "y1": int(h * 0.972)}
    draw.rounded_rectangle([card["x0"], card["y0"], card["x1"], card["y1"]],
                           radius=36, fill=_hex_rgb(theme["card"], (255, 255, 255)),
                           outline=_hex_rgb(theme["card_border"], (143, 176, 201)),
                           width=2)
    return img, draw, card


def _draw_channel(draw: ImageDraw.ImageDraw, card: dict, theme: dict,
                  channel_name: str) -> None:
    if not channel_name:
        return
    font = ImageFont.truetype(_handwrite_path(theme), 36)
    draw.text((card["x0"] + 40, card["y0"] + 26), channel_name, font=font,
              fill=_hex_rgb(theme["channel_text"], (90, 107, 82)))


def _draw_badge(draw: ImageDraw.ImageDraw, card: dict, theme: dict,
                badge_text: str) -> None:
    if not badge_text:
        return
    bw, bh = 96, 66
    bx0, by0 = card["x1"] - bw, card["y0"]
    draw.rounded_rectangle([bx0, by0, bx0 + bw, by0 + bh], radius=14,
                           fill=_hex_rgb(theme["badge_bg"], (242, 184, 198)))
    font = ImageFont.truetype(FONT_EN, 40)
    box = draw.textbbox((0, 0), badge_text, font=font)
    draw.text((bx0 + (bw - (box[2] - box[0])) / 2,
               by0 + (bh - (box[3] - box[1])) / 2 - box[1]),
              badge_text, font=font,
              fill=_hex_rgb(theme["badge_text"], (255, 255, 255)))


def _draw_number(img: Image.Image, num_text: str, theme: dict, x: int, y: int) -> None:
    """左下发光序号：模糊光晕层 + 锐利文字层。"""
    font = ImageFont.truetype(FONT_EN, 46)
    color = _hex_rgb(theme["num"], (255, 111, 165))
    glow = Image.new("RGBA", img.size, (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    gd.text((x, y), num_text, font=font, fill=(*color, 200))
    glow = glow.filter(ImageFilter.GaussianBlur(7))
    img.paste(glow, (0, 0), glow)
    ImageDraw.Draw(img).text((x, y), num_text, font=font, fill=color)


def render_pair_card(a_line: dict, b_line: dict, idx: int, theme: dict,
                     out_path: str, channel_name: str = "English with me",
                     badge_text: str = "EN", w: int = TARGET_W,
                     h: int = TARGET_H) -> str:
    """渲染一组（A+B）静态卡片 PNG。idx 为 1-based 序号。"""
    img, draw, card = _draw_card_base(theme, w, h)
    _draw_channel(draw, card, theme, channel_name)
    _draw_badge(draw, card, theme, badge_text)
    card_top = card["y0"] + 96
    card_h = card["y1"] - card["y0"] - 140
    _draw_text_block(img, draw, a_line.get("text", ""), a_line.get("phonetic", ""),
                     a_line.get("zh", ""), int(card_top + card_h * 0.35),
                     _hex_rgb(theme["en_a"], (66, 32, 6)), theme, en_start=74)
    _draw_text_block(img, draw, b_line.get("text", ""), b_line.get("phonetic", ""),
                     b_line.get("zh", ""), int(card_top + card_h * 0.80),
                     _hex_rgb(theme["en_b"], (224, 90, 18)), theme, en_start=68)
    _draw_number(img, f"{idx:02d}", theme, card["x0"] + 40, card["y1"] - 86)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.convert("RGB").save(out_path, "PNG")
    return out_path


def render_intro_card(theme: dict, out_path: str,
                      channel_name: str = "English with me",
                      badge_text: str = "EN", w: int = TARGET_W,
                      h: int = TARGET_H) -> str:
    """片头卡：频道名居中（手写体大字）+ 副题 + 徽标 + 叶片。"""
    img, draw, card = _draw_card_base(theme, w, h)
    _draw_badge(draw, card, theme, badge_text)
    hw_font = _fit_font(draw, channel_name, _handwrite_path(theme), 100, 40, w - 320)
    box = draw.textbbox((0, 0), channel_name, font=hw_font)
    draw.text(((w - (box[2] - box[0])) / 2, h / 2 - (box[3] - box[1]) / 2 - 40),
              channel_name, font=hw_font,
              fill=_hex_rgb(theme["channel_text"], (90, 107, 82)))
    sub_font = ImageFont.truetype(FONT_ZH, 34)
    sub = "閉上眼睛 · 輕鬆聽"
    box = draw.textbbox((0, 0), sub, font=sub_font)
    draw.text(((w - (box[2] - box[0])) / 2, h / 2 + 60), sub, font=sub_font,
              fill=_hex_rgb(theme["zh_text"], (58, 58, 58)))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.convert("RGB").save(out_path, "PNG")
    return out_path


def render_outro_card(theme: dict, out_path: str, outro_text: str,
                      channel_name: str = "English with me",
                      badge_text: str = "EN", w: int = TARGET_W,
                      h: int = TARGET_H) -> str:
    """片尾卡：结束语居中 + 频道名 + 徽标。"""
    img, draw, card = _draw_card_base(theme, w, h)
    _draw_channel(draw, card, theme, channel_name)
    _draw_badge(draw, card, theme, badge_text)
    text = (outro_text or "Thanks for listening. See you next time!").strip()
    en_font = _fit_font(draw, text, _FONT_EN_CARD, 64, 30, w - 300)
    lines = _wrap_words(draw, text, en_font, w - 300)
    y = h / 2 - len(lines) * (en_font.size + 14) / 2
    for ln in lines:
        box = draw.textbbox((0, 0), ln, font=en_font)
        draw.text(((w - (box[2] - box[0])) / 2, y), ln, font=en_font,
                  fill=_hex_rgb(theme["en_a"], (66, 32, 6)))
        y += en_font.size + 14
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.convert("RGB").save(out_path, "PNG")
    return out_path


def render_sleep_thumbnail(script: dict, theme: dict, out_path: str,
                           badge_text: str = "EN",
                           channel_name: str = "English with me",
                           w: int = TARGET_W, h: int = TARGET_H) -> str:
    """缩略图：首组 A/B 卡片 + 顶部句数横条（thumbnail_subtitle）。"""
    dialogue = script.get("dialogue", [])
    a_line = dialogue[0] if dialogue else {}
    b_line = dialogue[1] if len(dialogue) > 1 else {}
    img, draw, card = _draw_card_base(theme, w, h)
    _draw_channel(draw, card, theme, channel_name)
    _draw_badge(draw, card, theme, badge_text)
    _draw_text_block(img, draw, a_line.get("text", ""), a_line.get("phonetic", ""),
                     a_line.get("zh", ""), int(card["y0"] + 185),
                     _hex_rgb(theme["en_a"], (66, 32, 6)), theme, en_start=62)
    _draw_text_block(img, draw, b_line.get("text", ""), b_line.get("phonetic", ""),
                     b_line.get("zh", ""), int(card["y0"] + 430),
                     _hex_rgb(theme["en_b"], (224, 90, 18)), theme, en_start=56)
    strip = str(script.get("thumbnail_subtitle", "") or "").strip()
    if strip:
        st_font = ImageFont.truetype(FONT_ZH, 44)
        box = draw.textbbox((0, 0), strip, font=st_font)
        pad_x = 26
        sw = box[2] - box[0] + pad_x * 2
        sx = (w - sw) / 2
        draw.rounded_rectangle([sx, 14, sx + sw, 76], radius=14,
                               fill=(*_hex_rgb(theme["num"], (255, 111, 165)), 235))
        draw.text((sx + pad_x, 14 + (62 - (box[3] - box[1])) / 2 - box[1]), strip,
                  font=st_font, fill=(255, 255, 255))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.convert("RGB").save(out_path, "PNG")
    return out_path
