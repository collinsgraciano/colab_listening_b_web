"""Subtitle Styles API — 字幕样式设计与预览（页面路由在 pages.py）."""
import asyncio

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse

from ..config_manager import (
    MODES, get_active_mode, load_config, load_mode_config, save_mode_config,
)
import subtitle_style_manager as subtitle_style_lib

router = APIRouter()


def _current_subtitle_style_ctx() -> dict:
    """当前模式的字幕样式上下文（current_id/current_style/legacy 字号）。"""
    config = load_config()
    current_id = config.get("subtitle_style", "")
    current_style = subtitle_style_lib.get_style(current_id) if current_id else None
    try:
        font_size = int(config.get("subtitle_font_size", 60))
    except (ValueError, TypeError):
        font_size = 60
    return {"current_id": current_id, "current_style": current_style,
            "config_font_size": font_size}


@router.get("/api/subtitle-styles")
async def api_subtitle_styles_list():
    ctx = _current_subtitle_style_ctx()
    return {"styles": subtitle_style_lib.list_styles(), **ctx}


@router.post("/api/subtitle-styles/create")
async def api_subtitle_styles_create(request: Request):
    data = await request.json()
    try:
        style = await asyncio.to_thread(subtitle_style_lib.save_custom_style, data)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    return {"ok": True, "style": style}


@router.post("/api/subtitle-styles/update")
async def api_subtitle_styles_update(request: Request):
    data = await request.json()
    style_id = str(data.get("id", ""))
    existing = subtitle_style_lib.get_style(style_id)
    if not existing:
        return JSONResponse({"error": f"样式 '{style_id}' 不存在"}, status_code=404)
    if existing.get("builtin"):
        return JSONResponse({"error": "内置样式不可编辑"}, status_code=400)
    try:
        style = await asyncio.to_thread(subtitle_style_lib.save_custom_style, data)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    return {"ok": True, "style": style}


@router.delete("/api/subtitle-styles/{style_id}")
async def api_subtitle_styles_delete(style_id: str):
    existing = subtitle_style_lib.get_style(style_id)
    if not existing:
        return JSONResponse({"error": f"样式 '{style_id}' 不存在"}, status_code=404)
    if existing.get("builtin"):
        return JSONResponse({"error": "内置样式不可删除"}, status_code=400)
    ok = await asyncio.to_thread(subtitle_style_lib.delete_custom_style, style_id)
    # 删除后：所有正在使用该样式的模式配置回落「跟随参数配置」
    reset_modes = []
    if ok:
        for mode in MODES:
            mc = load_mode_config(mode)
            if mc.get("subtitle_style") == style_id:
                mc["subtitle_style"] = ""
                save_mode_config(mode, mc)
                reset_modes.append(mode)
    return {"ok": ok, "reset_modes": reset_modes}


@router.post("/api/subtitle-styles/select")
async def api_subtitle_styles_select(request: Request):
    """为当前 active mode 设置字幕样式（"" = 跟随参数配置）。"""
    data = await request.json()
    style_id = str(data.get("style_id", "")).strip()
    if style_id and not subtitle_style_lib.get_style(style_id):
        return JSONResponse({"error": f"样式 '{style_id}' 不存在"}, status_code=404)
    mode = get_active_mode()
    mc = load_mode_config(mode)
    mc["subtitle_style"] = style_id
    save_mode_config(mode, mc)
    return {"ok": True, "mode": mode, "subtitle_style": style_id}


@router.post("/api/subtitle-styles/preview/render")
async def api_subtitle_styles_preview_adhoc_post(request: Request):
    """编辑器实时预览：body = {style, mode, bg, sample_en, sample_zh} → PNG。

    按模式渲染：画布/ZH 可见性/字号规则与该模式成片一致。
    """
    data = await request.json()
    style = data.get("style") or None
    if isinstance(style, dict):
        # 允许未保存的草稿参数；宽松处理无效值（渲染层有兜底）
        style = {k: v for k, v in style.items()
                 if k in subtitle_style_lib.media_utils.SUBTITLE_STYLE_LEGACY_DEFAULTS}
    mode = str(data.get("mode", "original"))
    if mode not in subtitle_style_lib.MODE_INFO:
        mode = "original"
    png, mime = await asyncio.to_thread(
        subtitle_style_lib.render_mode_preview, style, mode,
        str(data.get("bg", "gradient")),
        str(data.get("sample_en", "")), str(data.get("sample_zh", "")))
    return Response(content=png, media_type=mime)


@router.get("/api/subtitle-styles/preview/current")
async def api_subtitle_styles_preview_current(mode: str = "original",
                                              bg: str = "gradient",
                                              sample_en: str = "", sample_zh: str = "",
                                              thumb: bool = False):
    """当前模式字幕样式预览（未选样式 → 该模式「跟随参数配置」行为）。"""
    if mode not in subtitle_style_lib.MODE_INFO:
        mode = "original"
    ctx = _current_subtitle_style_ctx()
    data, mime = await asyncio.to_thread(
        subtitle_style_lib.render_mode_preview, ctx["current_style"],
        mode, bg, sample_en, sample_zh, thumb)
    return Response(content=data, media_type=mime)


@router.get("/api/subtitle-styles/preview/{style_id}")
async def api_subtitle_styles_preview(style_id: str, mode: str = "original",
                                      bg: str = "gradient",
                                      sample_en: str = "", sample_zh: str = "",
                                      thumb: bool = False):
    """已保存样式的预览图（按模式渲染，thumb=缩略 JPEG）。"""
    style = subtitle_style_lib.get_style(style_id)
    if not style:
        return JSONResponse({"error": f"样式 '{style_id}' 不存在"}, status_code=404)
    if mode not in subtitle_style_lib.MODE_INFO:
        mode = "original"
    data, mime = await asyncio.to_thread(
        subtitle_style_lib.render_mode_preview, style, mode, bg,
        sample_en, sample_zh, thumb)
    return Response(content=data, media_type=mime)


@router.get("/api/subtitle-styles/modes")
async def api_subtitle_styles_modes():
    """模式预览信息（画布/样例文本/ZH 可见性）。"""
    return {"modes": subtitle_style_lib.list_mode_infos()}


@router.get("/api/subtitle-styles/fonts")
async def api_subtitle_styles_fonts():
    return {"fonts": subtitle_style_lib.get_font_options()}


@router.get("/api/subtitle-styles/backgrounds")
async def api_subtitle_styles_backgrounds():
    return {"backgrounds": subtitle_style_lib.list_backgrounds()}
