"""MCP Token detection + 页面专属 Token 管理."""
import asyncio

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from ..config_manager import detect_local_mcp_token
from ..page_mcp import (
    PAGES, SOURCE_LABELS,
    PageMcpSession, resolve_page_tokens, load_page_tokens, save_page_tokens,
    mask_token,
)

router = APIRouter()


@router.get("/api/mcp/detect_token")
async def api_detect_token():
    token = detect_local_mcp_token()
    if token:
        # Mask for display but return full for use
        return {"ok": True, "token": token}
    return {"ok": False, "error": "未检测到本地 MCP Token"}


# --- 页面专属 MCP Tokens（🎨画面风格 / 👥人物素材库 独立设置）---

@router.get("/api/mcp/page_tokens")
async def api_page_tokens_get():
    """页面专属 MCP Token 设置 + 当前生效来源。

    tokens_text 为本页存储的原文（供编辑回填）；masked 仅掩码（供生效来源展示）。
    """
    saved = load_page_tokens()
    pages: dict[str, dict] = {}
    for page in PAGES:
        resolved = resolve_page_tokens(page)
        saved_tokens = saved.get(page, [])
        pages[page] = {
            "tokens_text": "\n".join(saved_tokens),
            "saved_count": len(saved_tokens),
            "source": resolved["source"],
            "source_label": SOURCE_LABELS[resolved["source"]],
            "mode": resolved["mode"],
            "effective_count": len(resolved["tokens"]),
            "masked": [mask_token(t) for t in resolved["tokens"][:5]],
        }
    return {"pages": pages}


@router.post("/api/mcp/page_tokens")
async def api_page_tokens_save(request: Request):
    """保存页面专属 MCP Tokens（多行文本，每行一个）。留空 = 回落模式配置。"""
    data = await request.json()
    page = str(data.get("page", "")).strip()
    if page not in PAGES:
        return JSONResponse({"ok": False, "error": f"未知页面: {page}"}, status_code=400)
    tokens_text = str(data.get("tokens", ""))
    try:
        tokens = await asyncio.to_thread(save_page_tokens, page, tokens_text)
    except (OSError, ValueError) as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
    return {"ok": True, "page": page, "count": len(tokens)}


@router.post("/api/mcp/page_tokens/test")
async def api_page_tokens_test(request: Request):
    """用解析后的 token 建独立会话 + tools/list（0 积分）验证可用性。"""
    data = await request.json()
    page = str(data.get("page", "")).strip()
    if page not in PAGES:
        return JSONResponse({"ok": False, "error": f"未知页面: {page}"}, status_code=400)

    def _do_test() -> dict:
        resolved = resolve_page_tokens(page)
        if not resolved["tokens"]:
            return {"ok": False, "source": resolved["source"],
                    "error": "未配置 MCP Token（本页专属 / 模式配置 / 本地检测均为空）"}
        mcp = PageMcpSession(resolved["tokens"]).initialize()
        tools = mcp.list_tools()
        return {"ok": True, "source": resolved["source"],
                "source_label": SOURCE_LABELS[resolved["source"]],
                "token_count": len(resolved["tokens"]), "tools": len(tools)}

    try:
        return await asyncio.to_thread(_do_test)
    except Exception as e:
        return {"ok": False, "error": str(e)[:200]}
