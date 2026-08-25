"""页面级独立 MCP 客户端 — 🎨画面风格预览 / 👥人物素材库图片生成专用。

与 pipeline/mcp_client.py 的全局会话完全隔离：页面后台线程持有自己的
token 列表、轮换指针和 session id，绝不动运行中 pipeline 的全局状态，
两者可并行。支持页面专属 token：

    解析链（resolve_page_tokens）:
        本页专属 tokens (configs/page_mcp_tokens.json)
        → 当前激活模式 mcp_tokens (configs/mode_{mode}.json)
        → 本地 TJGenerators token (~/.codely-cli/mcp-oauth-tokens.json)
        → 均为空则报错
"""
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

WEB_ROOT = Path(__file__).parent.parent.resolve()
PIPELINE_DIR = WEB_ROOT / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

# 复用 pipeline/mcp_client.py 的纯函数与常量（不含全局会话状态）
from mcp_client import MCP_URL, parse_task_id, download_file, _is_credit_error  # noqa: E402

PAGE_TOKENS_PATH = WEB_ROOT / "configs" / "page_mcp_tokens.json"
PAGES = ("styles", "characters")

SOURCE_LABELS = {
    "page": "本页专属",
    "mode": "模式配置",
    "local": "本地检测",
    "none": "未配置",
}


# ---------------------------------------------------------------------------
# 存储：configs/page_mcp_tokens.json
# ---------------------------------------------------------------------------

def _read_token_file() -> dict[str, Any]:
    if not PAGE_TOKENS_PATH.exists():
        return {}
    try:
        data = json.loads(PAGE_TOKENS_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def load_page_tokens() -> dict[str, list[str]]:
    """读取全部页面专属 tokens: {"styles": [...], "characters": [...]}"""
    raw = _read_token_file()
    out: dict[str, list[str]] = {}
    for page in PAGES:
        val = raw.get(page, [])
        if isinstance(val, str):  # 兼容多行文本格式
            val = val.splitlines()
        out[page] = [t.strip() for t in val if isinstance(t, str) and t.strip()]
    return out


def save_page_tokens(page: str, tokens_text: str) -> list[str]:
    """保存某页专属 tokens（多行文本，每行一个）。返回解析后的列表。"""
    if page not in PAGES:
        raise ValueError(f"未知页面: {page}")
    tokens = [t.strip() for t in (tokens_text or "").splitlines() if t.strip()]
    raw = _read_token_file()
    raw[page] = tokens
    PAGE_TOKENS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PAGE_TOKENS_PATH.write_text(
        json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
    return tokens


# ---------------------------------------------------------------------------
# 解析链：本页专属 → 激活模式 mcp_tokens → 本地 token
# ---------------------------------------------------------------------------

def resolve_page_tokens(page: str) -> dict[str, Any]:
    """解析页面生效的 token 集。

    Returns:
        {"tokens": [...], "source": "page"|"mode"|"local"|"none",
         "mode": 当前激活模式名}
    """
    from .config_manager import detect_local_mcp_token, get_active_mode
    mode = get_active_mode()

    page_tokens = load_page_tokens().get(page, [])
    if page_tokens:
        return {"tokens": page_tokens, "source": "page", "mode": mode}

    from .config_manager import load_config
    mode_tokens = [t.strip() for t in
                   (load_config().get("mcp_tokens") or "").splitlines() if t.strip()]
    if mode_tokens:
        return {"tokens": mode_tokens, "source": "mode", "mode": mode}

    local = detect_local_mcp_token()
    if local:
        return {"tokens": [local], "source": "local", "mode": mode}

    return {"tokens": [], "source": "none", "mode": mode}


def mask_token(tok: str) -> str:
    """掩码显示：TJGe…ab3f（绝不回传完整 token）。"""
    if not tok:
        return ""
    if len(tok) <= 10:
        return tok[:2] + "…"
    return f"{tok[:4]}…{tok[-4:]}"


# ---------------------------------------------------------------------------
# 独立会话客户端
# ---------------------------------------------------------------------------

class PageMcpSession:
    """独立于 pipeline 全局会话的 MCP 客户端。

    自持 token 列表（积分错误自动轮换）、session id、消息 id；
    与 mcp_client 全局状态互不干扰，可与运行中的 pipeline 并行。
    """

    def __init__(self, tokens: list[str]):
        cleaned = [t.strip() for t in (tokens or []) if t.strip()]
        if not cleaned:
            raise RuntimeError(
                "未配置 MCP Token（本页专属 / 模式配置 / 本地检测均为空），"
                "请在「MCP Token 设置」面板或配置页填写。")
        self.tokens = cleaned
        self._idx = 0
        self.token = self.tokens[0]
        self._session_id: str | None = None
        self._msg_id = 0

    # --- 底层 JSON-RPC ---

    def _next_id(self) -> int:
        self._msg_id += 1
        return self._msg_id

    def _rotate(self):
        self._idx += 1
        if self._idx >= len(self.tokens):
            raise RuntimeError("ALL_MCP_TOKENS_EXHAUSTED: 本页 MCP Token 全部积分不足。")
        self.token = self.tokens[self._idx]
        self._session_id = None
        print(f"  [PageMCP] 积分不足, switching to token #{self._idx + 1}/{len(self.tokens)}...")
        self._handshake()

    def _call(self, method: str, params: dict | None = None) -> dict:
        payload: dict[str, Any] = {"jsonrpc": "2.0", "method": method, "id": self._next_id()}
        if params is not None:
            payload["params"] = params
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json",
                   "Authorization": f"Bearer {self.token}"}
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        req = urllib.request.Request(MCP_URL, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                sid = resp.headers.get("Mcp-Session-Id") or resp.headers.get("mcp-session-id")
                if sid:
                    self._session_id = sid
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            sid = e.headers.get("Mcp-Session-Id") or e.headers.get("mcp-session-id")
            if sid:
                self._session_id = sid
            body = e.read().decode("utf-8", errors="replace")
            if _is_credit_error(body):
                print(f"  [PageMCP] 积分不足! HTTP {e.code}: {body[:300]}")
                self._rotate()
                return self._call(method, params)
            raise

    def _notify(self, method: str, params: dict | None = None):
        payload: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            payload["params"] = params
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json",
                   "Authorization": f"Bearer {self.token}"}
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        req = urllib.request.Request(MCP_URL, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=30):
                pass
        except Exception:
            pass

    def _handshake(self):
        self._call("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "codely-web-pages", "version": "1.0"},
        })
        self._notify("notifications/initialized")

    # --- 对外接口（语义与 mcp_client 对齐）---

    def initialize(self) -> "PageMcpSession":
        self._handshake()
        print(f"  [PageMCP] Session initialized with {len(self.tokens)} token(s).")
        return self

    def call_tool(self, name: str, arguments: dict) -> dict:
        result = self._call("tools/call", {"name": name, "arguments": arguments})
        if "result" in result:
            for item in result["result"].get("content", []):
                if item.get("type") == "text" and _is_credit_error(item.get("text", "")):
                    print(f"  [PageMCP] 积分不足! 工具响应: {item['text'][:300]}")
                    self._rotate()
                    return self.call_tool(name, arguments)
        return result

    def parse_task_id(self, result: dict) -> str:
        return parse_task_id(result)

    def list_tools(self) -> list[dict]:
        """tools/list（0 积分），用于验证 token 可用性。"""
        result = self._call("tools/list", {})
        return (result.get("result") or {}).get("tools", [])

    def poll_task(self, task_id: str, interval: int = 10, max_wait: int = 600) -> dict:
        """轮询 check_task 至完成/失败/超时。返回 {status, url?, error?}。"""
        elapsed = 0
        while elapsed < max_wait:
            try:
                result = self.call_tool("check_task", {"task_id": task_id})
            except Exception as e:
                if "ALL_MCP_TOKENS_EXHAUSTED" in str(e):
                    raise
                print(f"  [PageMCP] [{elapsed}s] Task {task_id[:16]}...: "
                      f"HTTP error ({type(e).__name__}: {e}), retrying in {interval}s...")
                elapsed += interval
                time.sleep(interval)
                continue

            task_data = self._parse_task_result(result)
            status = task_data.get("status", "")
            if status:
                print(f"  [PageMCP] [{elapsed}s] Task {task_id[:16]}...: {status}")
            if status == "completed":
                return task_data
            if status == "failed":
                print(f"  [PageMCP] FAILED: {json.dumps(task_data, ensure_ascii=False)[:500]}")
                return task_data
            time.sleep(interval)
            elapsed += interval

        print(f"  [PageMCP] TIMEOUT after {max_wait}s for task {task_id}")
        return {}

    @staticmethod
    def _parse_task_result(result: dict) -> dict:
        """解析 check_task 响应（resource 块 JSON 优先，文本块 Markdown 兜底）。"""
        task_data: dict[str, Any] = {"raw_response": json.dumps(result, ensure_ascii=False)}
        content = (result.get("result") or {}).get("content", [])

        # 1. resource 块（含 output URLs + 错误信息）
        for item in content:
            if item.get("type") != "resource":
                continue
            try:
                res_json = json.loads(item.get("resource", {}).get("text", ""))
            except (json.JSONDecodeError, AttributeError):
                continue
            task_data["status"] = res_json.get("status", "")
            out = res_json.get("output") or {}
            data = out.get("data") or {}
            if res_json.get("error"):
                task_data["error"] = res_json["error"]
            elif data.get("error"):
                task_data["error"] = data["error"]
            elif data.get("message") and res_json.get("status") == "failed":
                task_data["error"] = data["message"]
            result_obj = data.get("result") or {}
            url = (result_obj.get("video_url") or result_obj.get("image_url")
                   or result_obj.get("url") or "")
            if not url:
                img_urls = (result_obj.get("image_urls") or data.get("imageUrls")
                            or data.get("image_urls") or [])
                if isinstance(img_urls, list) and img_urls:
                    url = img_urls[0]
            if not url:
                url = data.get("url") or data.get("imageUrl") or ""
            if url:
                task_data["url"] = url
            break

        # 2. 文本块兜底（状态关键词 + Markdown 图片 URL）
        if "url" not in task_data or not task_data.get("status"):
            for item in content:
                if item.get("type") != "text":
                    continue
                text = item.get("text", "")
                task_data["raw_text"] = text
                if not task_data.get("status"):
                    low = text.lower()
                    if "completed" in low or "已完成" in text:
                        task_data["status"] = "completed"
                    elif "failed" in low or "失败" in text:
                        task_data["status"] = "failed"
                    elif "running" in low or "queued" in low or "进行中" in text:
                        task_data["status"] = "running"
                if "url" not in task_data:
                    m = re.search(r"!\[.*?\]\((https?://[^\s)]+)\)", text)
                    if m:
                        task_data["url"] = m.group(1)
                    else:
                        m = re.search(r"(https?://[^\s`'\")]+)", text)
                        if m:
                            task_data["url"] = m.group(1)
                if "url" in task_data and task_data.get("status"):
                    break
        return task_data

    def download_file(self, url: str, dest: str) -> bool:
        return download_file(url, dest)
