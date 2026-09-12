"""Config API — 参数配置 / 预设 / 模式切换."""
import asyncio
import io
import tempfile
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response

from ..config_manager import (
    MODES, MODE_LABELS,
    DEFAULT_QUICK_FIELDS, load_quick_fields, save_quick_fields,
    load_config, load_mode_config, save_mode_config, load_all_mode_configs,
    get_active_mode, set_active_mode,
    get_default_config,
    list_presets, save_preset, load_preset, delete_preset,
)

router = APIRouter()


@router.post("/api/config/save")
async def api_save_config(request: Request):
    data = await request.json()
    mode = data.pop("_mode", "") or get_active_mode()
    if mode not in MODES:
        return JSONResponse({"ok": False, "error": f"未知模式: {mode}"}, status_code=400)
    config = load_mode_config(mode)
    config.update(data)
    config["structure"] = mode
    save_mode_config(mode, config)
    return {"ok": True, "mode": mode}


@router.post("/api/config/save_all")
async def api_save_all_config(request: Request):
    data = await request.json()
    mode = data.pop("_mode", "") or get_active_mode()
    if mode not in MODES:
        return JSONResponse({"ok": False, "error": f"未知模式: {mode}"}, status_code=400)
    data["structure"] = mode
    save_mode_config(mode, data)
    return {"ok": True, "mode": mode}


@router.get("/api/config")
async def api_get_config(mode: str = ""):
    return load_mode_config(mode) if mode in MODES else load_config()


@router.get("/api/config/all")
async def api_get_all_configs():
    """一次性返回 3 个模式的完整配置 + 当前激活模式（控制台预载用）。"""
    return {
        "active_mode": get_active_mode(),
        "modes": load_all_mode_configs(),
        "mode_labels": MODE_LABELS,
    }


@router.post("/api/config/active")
async def api_set_active_mode(request: Request):
    data = await request.json()
    mode = data.get("mode", "")
    if mode not in MODES:
        return JSONResponse({"ok": False, "error": f"未知模式: {mode}"}, status_code=400)
    set_active_mode(mode)
    return {"ok": True, "active_mode": mode}


@router.get("/api/config/defaults")
async def api_get_defaults():
    return get_default_config()


@router.post("/api/config/preset/save")
async def api_save_preset(request: Request):
    data = await request.json()
    name = data.get("name", "")
    config = data.get("config", {})
    if not name:
        return JSONResponse({"ok": False, "error": "名称不能为空"}, status_code=400)
    save_preset(name, config)
    return {"ok": True, "name": name}


@router.get("/api/config/preset/load/{name}")
async def api_load_preset(name: str):
    try:
        return load_preset(name)
    except FileNotFoundError:
        return JSONResponse({"error": "Preset not found"}, status_code=404)


@router.delete("/api/config/preset/{name}")
async def api_delete_preset(name: str):
    delete_preset(name)
    return {"ok": True}


@router.get("/api/config/presets")
async def api_list_presets():
    return {"presets": list_presets()}


# --- 控制台「常用配置」面板字段清单（每模式独立） ---

@router.get("/api/quick_config/fields")
async def api_get_quick_fields(mode: str = ""):
    m = mode if mode in MODES else get_active_mode()
    return {"mode": m, "fields": load_quick_fields(m),
            "defaults": DEFAULT_QUICK_FIELDS}


@router.post("/api/quick_config/fields")
async def api_save_quick_fields(request: Request):
    data = await request.json()
    mode = data.get("mode", "") or get_active_mode()
    if mode not in MODES:
        return JSONResponse({"ok": False, "error": f"未知模式: {mode}"}, status_code=400)
    fields = save_quick_fields(mode, data.get("fields", []))
    return {"ok": True, "mode": mode, "fields": fields}


# --- Sleep 卡片实时预览（配置页 😴 Sleep 组边改边看，真实 Pillow 渲染）---

_SLEEP_SAMPLE_A = {"text": "The house is quiet now",
                   "phonetic": "/ðə haʊs ɪz ˈkwaɪət naʊ/",
                   "zh": "房子現在安靜下來了"}
_SLEEP_SAMPLE_B = {"text": "Time to close your eyes",
                   "phonetic": "/taɪm tə kloʊz jɔːr aɪz/",
                   "zh": "該閉上眼睛了"}


def _render_sleep_preview_png(cfg: dict) -> bytes:
    """intro/pair/outro 三卡纵向合成 → PNG bytes（同步渲染，跑线程池）。"""
    from PIL import Image

    from sleep.sleep_cards import (build_theme, render_intro_card,
                                   render_outro_card, render_pair_card)

    cfg = cfg or {}
    theme = build_theme(cfg)
    channel = str(cfg.get("sleep_channel_name", "") or "")
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        intro = render_intro_card(theme, str(base / "intro.png"), channel_name=channel)
        pair = render_pair_card(_SLEEP_SAMPLE_A, _SLEEP_SAMPLE_B, 1, theme,
                                str(base / "pair.png"), channel_name=channel)
        outro = render_outro_card(theme, str(base / "outro.png"),
                                  str(cfg.get("sleep_outro_text", "") or ""),
                                  channel_name=channel)
        imgs = [Image.open(p) for p in (intro, pair, outro)]
        canvas = Image.new("RGB", (max(im.width for im in imgs),
                                   sum(im.height for im in imgs)), (255, 255, 255))
        y = 0
        for im in imgs:
            canvas.paste(im, (0, y))
            y += im.height
        buf = io.BytesIO()
        canvas.save(buf, "PNG")
        return buf.getvalue()


@router.get("/api/config/sleep_preview")
async def api_sleep_preview_get():
    """用已保存的 sleep 模式配置渲染预览（页面首图）。"""
    png = await asyncio.to_thread(_render_sleep_preview_png, load_mode_config("sleep"))
    return Response(content=png, media_type="image/png")


@router.post("/api/config/sleep_preview")
async def api_sleep_preview_post(request: Request):
    """配置页实时预览：body = 表单收集的 sleep_* 键值（未保存草稿值亦可）。"""
    data = await request.json()
    cfg = {k: v for k, v in data.items()
           if isinstance(k, str) and k.startswith("sleep_")}
    png = await asyncio.to_thread(_render_sleep_preview_png, cfg)
    return Response(content=png, media_type="image/png")
