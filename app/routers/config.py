"""Config API — 参数配置 / 预设 / 模式切换."""
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from ..config_manager import (
    MODES, MODE_LABELS,
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
