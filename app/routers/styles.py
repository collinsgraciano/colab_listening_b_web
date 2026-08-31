"""Visual Styles API — 画面风格管理（页面路由在 pages.py）."""
import asyncio
import threading

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, JSONResponse

from ..config_manager import (
    MODES, get_active_mode, load_config, load_mode_config, save_mode_config,
)
from ..page_mcp import (
    SOURCE_LABELS as MCP_SOURCE_LABELS,
    PageMcpSession, mask_token, resolve_page_tokens,
)
import style_manager as style_lib

router = APIRouter()

# 预览图生成任务状态: {style_id: "running" | "done" | "错误信息"}
_PREVIEW_JOBS: dict[str, str] = {}


@router.get("/api/styles")
async def api_styles_list():
    config = load_config()
    current_id = config.get("visual_style", "pixar3d")
    styles = style_lib.list_styles()
    for s in styles:
        s["has_preview"] = style_lib.preview_path(s["id"]).exists()
    return {
        "styles": styles,
        "current_id": current_id,
        "active_mode": get_active_mode(),
        "preview_jobs": _PREVIEW_JOBS,
    }


@router.post("/api/styles/create")
async def api_styles_create(request: Request):
    data = await request.json()
    try:
        style = await asyncio.to_thread(style_lib.save_custom_style, data)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    return {"ok": True, "style": style}


@router.post("/api/styles/update")
async def api_styles_update(request: Request):
    data = await request.json()
    style_id = str(data.get("id", ""))
    existing = style_lib.get_style(style_id)
    if not existing:
        return JSONResponse({"error": f"风格 '{style_id}' 不存在"}, status_code=404)
    if existing.get("builtin"):
        return JSONResponse({"error": "内置风格不可编辑"}, status_code=400)
    try:
        style = await asyncio.to_thread(style_lib.save_custom_style, data)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    return {"ok": True, "style": style}


@router.delete("/api/styles/{style_id}")
async def api_styles_delete(style_id: str):
    existing = style_lib.get_style(style_id)
    if not existing:
        return JSONResponse({"error": f"风格 '{style_id}' 不存在"}, status_code=404)
    if existing.get("builtin"):
        return JSONResponse({"error": "内置风格不可删除"}, status_code=400)
    ok = await asyncio.to_thread(style_lib.delete_custom_style, style_id)
    # 删除后：所有正在使用该风格的模式配置回落默认
    reset_modes = []
    if ok:
        for mode in MODES:
            mc = load_mode_config(mode)
            if mc.get("visual_style") == style_id:
                mc["visual_style"] = "pixar3d"
                save_mode_config(mode, mc)
                reset_modes.append(mode)
    return {"ok": ok, "reset_modes": reset_modes}


@router.post("/api/styles/select")
async def api_styles_select(request: Request):
    """为当前 active mode 设置画面风格。"""
    data = await request.json()
    style_id = str(data.get("style_id", "")).strip()
    if not style_lib.get_style(style_id):
        return JSONResponse({"error": f"风格 '{style_id}' 不存在"}, status_code=404)
    mode = get_active_mode()
    mc = load_mode_config(mode)
    mc["visual_style"] = style_id
    save_mode_config(mode, mc)
    return {"ok": True, "mode": mode, "visual_style": style_id}


@router.get("/api/styles/preview/{style_id}")
async def api_styles_preview_get(style_id: str):
    p = style_lib.preview_path(style_id)
    if not p.exists():
        return JSONResponse({"error": "预览图不存在"}, status_code=404)
    return FileResponse(str(p), media_type="image/png")


def _generate_preview_worker(style_id: str, style_prompt: str):
    """后台线程：用页面级独立 MCP 会话生成标准测试图（咖啡店双人场景）作为风格预览。

    独立于 pipeline 的全局 MCP 会话——pipeline 运行中也可安全生成。
    Token 来源：本页专属 → 激活模式 mcp_tokens → 本地检测。
    """
    try:
        resolved = resolve_page_tokens("styles")
        if not resolved["tokens"]:
            raise RuntimeError("未配置 MCP Token（本页专属 / 模式配置 / 本地检测均为空）")
        print(f"  [StylePreview] MCP token: {MCP_SOURCE_LABELS[resolved['source']]} "
              f"×{len(resolved['tokens'])} ({mask_token(resolved['tokens'][0])})")
        mcp = PageMcpSession(resolved["tokens"]).initialize()
        prompt = (f"a friendly young woman barista in a green apron and a young man "
                  f"customer talking at a coffee shop counter, warm daylight, "
                  f"{style_prompt}, 16:9")
        result = mcp.call_tool("generate_image", {
            "prompt": prompt,
            "provider": "frontier",
            "quality": "medium",
            "image_size": {"width": 1280, "height": 720},
            "output_format": "png",
        })
        task_id = mcp.parse_task_id(result)
        if not task_id:
            raise RuntimeError("MCP 返回无 task_id")
        data = mcp.poll_task(task_id, interval=10, max_wait=600)
        url = data.get("url", "")
        if not url:
            raise RuntimeError("未获取到图片 URL")
        out = style_lib.preview_path(style_id)
        out.parent.mkdir(parents=True, exist_ok=True)
        if not mcp.download_file(url, str(out)) or out.stat().st_size < 10000:
            out.unlink(missing_ok=True)
            raise RuntimeError("下载文件过小，疑似失败")
        _PREVIEW_JOBS[style_id] = "done"
    except Exception as e:
        _PREVIEW_JOBS[style_id] = f"failed: {e}"


@router.post("/api/styles/preview/{style_id}")
async def api_styles_preview_generate(style_id: str):
    """生成风格预览图（消耗 MCP 积分，后台执行）。"""
    style = style_lib.get_style(style_id)
    if not style:
        return JSONResponse({"error": f"风格 '{style_id}' 不存在"}, status_code=404)
    status = _PREVIEW_JOBS.get(style_id, "")
    if status == "running":
        return {"ok": True, "status": "running", "note": "已有任务进行中"}
    _PREVIEW_JOBS[style_id] = "running"
    threading.Thread(
        target=_generate_preview_worker,
        args=(style_id, style.get("style_prompt", style_lib.DEFAULT_STYLE_PROMPT)),
        daemon=True,
    ).start()
    return {"ok": True, "status": "running"}


@router.get("/api/styles/preview/{style_id}/status")
async def api_styles_preview_status(style_id: str):
    status = _PREVIEW_JOBS.get(style_id, "")
    has = style_lib.preview_path(style_id).exists()
    return {"status": status, "has_preview": has}
