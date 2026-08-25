"""字幕样式库：内置样式 + 自定义样式 CRUD + 预览渲染.

字幕样式控制对话字幕的字体/字号/颜色/描边/间距/背景条，
渲染核心复用 media_utils.render_subtitle_text_overlay ——
Web 预览与最终视频烧录用同一套代码，所见即所得。
自定义样式持久化在 web 项目的 configs/subtitle_styles_custom.json。
"""
import io
import json
import re
from pathlib import Path
from typing import Any

import media_utils

_WEB_ROOT = Path(__file__).parent.parent
CUSTOM_STYLES_PATH = _WEB_ROOT / "configs" / "subtitle_styles_custom.json"
BG_PREVIEW_DIR = _WEB_ROOT / "configs" / "style_previews"

# 样式可调参数的范围限制（保存时钳制，防呆）
_INT_RANGES = {
    "en_size": (28, 120),
    "zh_size": (24, 100),
    "en_stroke": (0, 12),
    "zh_stroke": (0, 12),
    "bottom_margin": (0, 200),
    "line_gap": (0, 30),
    "en_zh_gap": (0, 60),
    "box_opacity": (20, 90),
}
_COLOR_KEYS = ("en_color", "zh_color", "stroke_color")
_HEX_RE = re.compile(r"^#[0-9a-fA-F]{6}$")
_ID_RE = re.compile(r"^[a-z0-9_]{1,40}$")

# 内置样式（builtin=true，不可编辑/删除）。default 与 burn_subtitles 历史行为
# 逐像素一致（60/51 白字金译、描边 5/4、微软雅黑、无背景条）。
BUILTIN_STYLES: list[dict[str, Any]] = [
    {
        "id": "default",
        "name": "默认（白字金译）",
        "description": "当前项目一直使用的经典字幕：白色英文 + 金色繁中，黑色描边，微软雅黑。与历史成片完全一致。",
        "en_size": 60, "zh_size": 51,
        "en_color": "#FFFFFF", "zh_color": "#FFD700", "stroke_color": "#000000",
        "en_stroke": 5, "zh_stroke": 4,
        "bottom_margin": 36, "line_gap": 6, "en_zh_gap": 15,
        "font_en": "msyhbd", "font_zh": "msyh",
        "box": False, "box_opacity": 55,
        "builtin": True,
    },
    {
        "id": "english_only",
        "name": "纯英文字幕",
        "description": "只显示英文字幕、隐藏中文翻译 —— 沉浸式听力训练用，逼观众靠耳朵；需要中文时可看 YouTube CC 字幕文件。",
        "en_size": 60, "zh_size": 51,
        "en_color": "#FFFFFF", "zh_color": "#FFD700", "stroke_color": "#000000",
        "en_stroke": 5, "zh_stroke": 4,
        "bottom_margin": 36, "line_gap": 6, "en_zh_gap": 15,
        "font_en": "msyhbd", "font_zh": "msyh",
        "box": False, "box_opacity": 55,
        "show_zh": False,
        "builtin": True,
    },
    {
        "id": "big_bold",
        "name": "大字幕（粗描边）",
        "description": "ESL 学习者友好：字号加大、描边加粗，画面繁忙时依然清晰易读，适合 YouTube 手机端观看。",
        "en_size": 72, "zh_size": 61,
        "en_color": "#FFFFFF", "zh_color": "#FFD700", "stroke_color": "#000000",
        "en_stroke": 7, "zh_stroke": 6,
        "bottom_margin": 36, "line_gap": 8, "en_zh_gap": 16,
        "font_en": "msyhbd", "font_zh": "msyh",
        "box": False, "box_opacity": 55,
        "builtin": True,
    },
    {
        "id": "black_bar",
        "name": "黑底字幕条",
        "description": "电视字幕风：文字块后加半透明圆角黑条，浅色/高亮画面下对比度最强，描边可减细更清爽。",
        "en_size": 60, "zh_size": 51,
        "en_color": "#FFFFFF", "zh_color": "#FFD700", "stroke_color": "#000000",
        "en_stroke": 3, "zh_stroke": 3,
        "bottom_margin": 36, "line_gap": 6, "en_zh_gap": 15,
        "font_en": "msyhbd", "font_zh": "msyh",
        "box": True, "box_opacity": 55,
        "builtin": True,
    },
    {
        "id": "minimal",
        "name": "极简细描边",
        "description": "细描边轻量风：更小的描边、浅灰色繁中翻译，不抢画面，适合画面本身是主角的题材。",
        "en_size": 56, "zh_size": 48,
        "en_color": "#FFFFFF", "zh_color": "#E5E7EB", "stroke_color": "#000000",
        "en_stroke": 2, "zh_stroke": 2,
        "bottom_margin": 36, "line_gap": 6, "en_zh_gap": 14,
        "font_en": "msyh", "font_zh": "msyh",
        "box": False, "box_opacity": 55,
        "builtin": True,
    },
    {
        "id": "compact",
        "name": "紧凑小字",
        "description": "小字号紧凑布局：字幕占地最小，露出更多画面，适合画面信息量大的场景（Quest/教学演示）。",
        "en_size": 48, "zh_size": 41,
        "en_color": "#FFFFFF", "zh_color": "#FFD700", "stroke_color": "#000000",
        "en_stroke": 4, "zh_stroke": 3,
        "bottom_margin": 30, "line_gap": 5, "en_zh_gap": 12,
        "font_en": "msyhbd", "font_zh": "msyh",
        "box": False, "box_opacity": 55,
        "builtin": True,
    },
]

_BUILTIN_IDS = {s["id"] for s in BUILTIN_STYLES}

# 预览画布
CANVASES = {
    "16:9": (1280, 720),
    "9:16": (1080, 1920),
}

# 预览默认样例文本（繁体中文，面向海外华人观众）
DEFAULT_SAMPLE_EN = "Could I get a large iced latte, please?"
DEFAULT_SAMPLE_ZH = "我可以要一杯大杯冰拿鐵嗎？"


# ---------------------------------------------------------------------------
# 自定义样式持久化
# ---------------------------------------------------------------------------

def _read_custom() -> list[dict[str, Any]]:
    if not CUSTOM_STYLES_PATH.exists():
        return []
    try:
        data = json.loads(CUSTOM_STYLES_PATH.read_text(encoding="utf-8"))
        return data.get("styles", [])
    except (json.JSONDecodeError, OSError):
        return []


def _write_custom(styles: list[dict[str, Any]]) -> None:
    CUSTOM_STYLES_PATH.parent.mkdir(parents=True, exist_ok=True)
    CUSTOM_STYLES_PATH.write_text(
        json.dumps({"styles": styles}, ensure_ascii=False, indent=2),
        encoding="utf-8")


def _normalize_params(data: dict[str, Any]) -> dict[str, Any]:
    """提取并钳制样式设计参数；非法值抛 ValueError。"""
    params: dict[str, Any] = {}
    for key, (lo, hi) in _INT_RANGES.items():
        if key in data:
            try:
                val = int(data[key])
            except (ValueError, TypeError):
                raise ValueError(f"{key} 必须是整数")
            params[key] = max(lo, min(hi, val))
    for key in _COLOR_KEYS:
        if key in data:
            val = str(data.get(key, "")).strip()
            if not _HEX_RE.match(val):
                raise ValueError(f"{key} 必须是 #RRGGBB 格式的颜色值")
            params[key] = val.upper()
    for key in ("font_en", "font_zh"):
        if key in data and data[key] is not None:
            val = str(data[key]).strip()
            if val and val not in media_utils.SUBTITLE_FONT_CHOICES:
                raise ValueError(f"未知字体: {val}")
            params[key] = val or ("msyhbd" if key == "font_en" else "msyh")
    if "box" in data:
        params["box"] = bool(data["box"])
    if "show_zh" in data and data["show_zh"] is not None:
        params["show_zh"] = bool(data["show_zh"])
    return params


def list_styles() -> list[dict[str, Any]]:
    """内置 + 自定义样式（每项附 builtin 标记；缺省参数补齐）。"""
    result = []
    for s in BUILTIN_STYLES:
        item = dict(s, builtin=True)
        # 补齐缺失参数（新增参数的老内置条目兼容）
        for k, v in media_utils.SUBTITLE_STYLE_LEGACY_DEFAULTS.items():
            item.setdefault(k, v)
        result.append(item)
    for s in _read_custom():
        item = dict(s)
        item["builtin"] = False
        # 补齐缺失参数（老数据兼容）
        for k, v in media_utils.SUBTITLE_STYLE_LEGACY_DEFAULTS.items():
            item.setdefault(k, v)
        result.append(item)
    return result


def get_style(style_id: str) -> dict[str, Any] | None:
    for s in list_styles():
        if s["id"] == style_id:
            return s
    return None


def save_custom_style(data: dict[str, Any]) -> dict[str, Any]:
    """新建或更新自定义样式（按 id 匹配）。返回保存后的样式。"""
    style_id = str(data.get("id", "")).strip()
    if not _ID_RE.match(style_id):
        raise ValueError("样式 ID 只能包含小写字母/数字/下划线，长度 1-40")
    if style_id in _BUILTIN_IDS:
        raise ValueError(f"ID '{style_id}' 与内置样式冲突")
    if not str(data.get("name", "")).strip():
        raise ValueError("样式名称不能为空")

    style: dict[str, Any] = {
        "id": style_id,
        "name": str(data["name"]).strip(),
        "description": str(data.get("description", "")).strip(),
        "builtin": False,
    }
    style.update(_normalize_params(data))

    styles = _read_custom()
    for i, s in enumerate(styles):
        if s["id"] == style_id:
            styles[i] = style
            _write_custom(styles)
            return style
    styles.append(style)
    _write_custom(styles)
    return style


def delete_custom_style(style_id: str) -> bool:
    styles = _read_custom()
    remaining = [s for s in styles if s["id"] != style_id]
    if len(remaining) == len(styles):
        return False
    _write_custom(remaining)
    return True


def get_style_options() -> dict[str, str]:
    """id → 名称（配置页下拉选项）；"" = 跟随参数配置（历史行为）。"""
    opts = {"": "跟随参数配置（默认）"}
    for s in list_styles():
        opts[s["id"]] = s.get("name", s["id"])
    return opts


def get_font_options() -> list[dict[str, Any]]:
    """字体注册表 → [{key, label, available}]（available=文件存在）。"""
    import os
    return [
        {"key": key, "label": label, "available": os.path.exists(path)}
        for key, (label, path) in media_utils.SUBTITLE_FONT_CHOICES.items()
    ]


# ---------------------------------------------------------------------------
# 预览渲染（与成片同一渲染核心 → 所见即所得）
# ---------------------------------------------------------------------------

def list_backgrounds() -> list[dict[str, str]]:
    """预览背景：内置渐变 + 画面风格预览图（真实场景，测可读性）。"""
    bgs = [{"id": "gradient", "name": "测试渐变（内置）"}]
    if BG_PREVIEW_DIR.exists():
        for p in sorted(BG_PREVIEW_DIR.glob("*.png")):
            bgs.append({"id": p.stem, "name": f"场景图 · {p.stem}"})
    return bgs


def _load_background(bg_id: str, w: int, h: int) -> "Image.Image":
    """加载预览背景并居中裁剪到目标画布；无图时生成测试渐变。"""
    from PIL import Image, ImageDraw

    path = BG_PREVIEW_DIR / f"{bg_id}.png"
    if bg_id != "gradient" and path.exists():
        img = Image.open(path).convert("RGB")
        # cover 式缩放：短边撑满 + 居中裁剪
        scale = max(w / img.width, h / img.height)
        nw, nh = int(img.width * scale + 0.5), int(img.height * scale + 0.5)
        img = img.resize((nw, nh), Image.LANCZOS)
        left, top = (nw - w) // 2, (nh - h) // 2
        return img.crop((left, top, left + w, top + h))

    # 内置渐变：右上亮左下暗 + 竖直色带，模拟复杂画面测可读性
    img = Image.new("RGB", (w, h))
    draw = ImageDraw.Draw(img)
    for y in range(h):
        t = y / max(h - 1, 1)
        r = int(40 + 150 * t)
        g = int(55 + 90 * t)
        b = int(75 + 40 * (1 - t))
        draw.line([(0, y), (w, y)], fill=(r, g, b))
    for i in range(1, 5):
        x = int(w * i / 5)
        draw.rectangle([x - 6, int(h * 0.2), x + 6, int(h * 0.45)],
                       fill=(230, 200, 140))
    return img


def render_preview_png(style: dict[str, Any] | None, bg_id: str = "gradient",
                       canvas: str = "16:9",
                       sample_en: str = "", sample_zh: str = "") -> bytes:
    """渲染字幕样式预览图（背景 + 字幕 overlay 合成）→ PNG bytes。

    canvas: "16:9" (1280x720) 或 "9:16" (1080x1920, shorts)。
    """
    from PIL import Image

    w, h = CANVASES.get(canvas, CANVASES["16:9"])
    bg = _load_background(bg_id, w, h).convert("RGBA")
    overlay = media_utils.render_subtitle_text_overlay(
        sample_en or DEFAULT_SAMPLE_EN,
        sample_zh or DEFAULT_SAMPLE_ZH,
        w, h, style)
    out = Image.alpha_composite(bg, overlay).convert("RGB")
    buf = io.BytesIO()
    out.save(buf, "PNG")
    return buf.getvalue()
