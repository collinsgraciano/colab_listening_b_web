"""TJGenerators MCP client — calls MCP server directly via HTTP JSON-RPC.

Supports multiple tokens with automatic rotation on "积分" (credits) errors.

Usage:
    from mcp_client import initialize, call_tool, poll_task
    initialize(tokens=["token1", "token2", "token3"])
    result = call_tool("generate_image", {"prompt": "...", "provider": "frontier"})
    task_id = parse_task_id(result)
    data = poll_task(task_id, interval=10)
    url = data.get("url")
"""
import sys
import os
import json
import time
import re
import threading
import urllib.request
import urllib.error

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
except Exception:
    pass  # stdout may be redirected/captured (pytest) — reconfigure unavailable

MCP_URL = "https://ai-generator.tuanjie.cn/mcp"

# Multi-token support
_TOKENS = []
_token_idx = 0
TOKEN = ""
_session_id = None
_msg_id = 0
# 并发保护：_INIT_LOCK 只保护 ensure_initialized。轮换与会话切换必须走
# _ROTATE_LOCK（RLock——轮换内部的 initialize 若再触发积分轮换需可重入），
# 否则多线程同时收到"积分不足"会把 _token_idx 连加多次，越过可用 token
# 误抛 ALL_MCP_TOKENS_EXHAUSTED（运行中断的主因）。
_ROTATE_LOCK = threading.RLock()
_MSG_ID_LOCK = threading.Lock()

# Windows local: auto-detect token from ~/.codely-cli/mcp-oauth-tokens.json
_TOKEN_FILE = os.path.join(os.environ.get("USERPROFILE", ""), ".codely-cli", "mcp-oauth-tokens.json")
try:
    with open(_TOKEN_FILE, encoding="utf-8") as f:
        _tokens_data = json.load(f)
    _local_token = next(t["token"]["accessToken"] for t in _tokens_data if t["serverName"] == "TJGenerators")
    _TOKENS = [_local_token]
    TOKEN = _local_token
except Exception:
    pass  # User must pass tokens via initialize()


def _next_id():
    global _msg_id
    with _MSG_ID_LOCK:
        _msg_id += 1
        return _msg_id


def _is_credit_error(text: str) -> bool:
    """Check if an error message indicates insufficient credits.
    Only matches actual credit/quota error messages, not the word 'credit' in prompts.
    """
    # Must contain error keywords AND credit/quota keywords
    is_error = any(k in text.lower() for k in ["error", "failed", "denied", "insufficient", "不足", "耗尽", "exhausted"])
    has_credit = any(k in text.lower() for k in ["积分", "credit balance", "credits", "余额不足", "quota exceeded", "billing"])
    # Also match explicit Chinese credit error patterns
    explicit = any(k in text for k in ["积分不足", "积分已耗尽", "余额不足", "额度不足"])
    return (is_error and has_credit) or explicit


def _rotate_token(failed_token=None):
    """Switch to the next available token. Raises if all tokens exhausted.

    failed_token: the token that hit the error. Double-check under the lock —
    if another thread already rotated past it, do nothing and let the caller
    retry with the current token. Without this check, N concurrent credit
    errors each bump _token_idx and falsely raise ALL_MCP_TOKENS_EXHAUSTED
    while later tokens were never tried.
    """
    global _token_idx, TOKEN, _session_id
    with _ROTATE_LOCK:
        if failed_token is not None and TOKEN != failed_token:
            return  # someone else already rotated past the failed token
        while True:
            _token_idx += 1
            if _token_idx >= len(_TOKENS):
                raise RuntimeError("ALL_MCP_TOKENS_EXHAUSTED: All tokens have insufficient credits.")
            TOKEN = _TOKENS[_token_idx]
            _session_id = None  # force re-initialize with new token
            print(f"  [MCP] 积分不足, switching to token #{_token_idx+1}/{len(_TOKENS)}...")
            # Re-initialize MCP session with new token. If this token is itself
            # unusable (expired/invalid → 401/403), skip to the next one
            # instead of crashing the whole run.
            try:
                mcp_call("initialize", {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "codely-cli", "version": "1.0"},
                })
                mcp_notify("notifications/initialized")
                return
            except urllib.error.HTTPError as e:
                if e.code in (401, 403):
                    print(f"  [MCP] token #{_token_idx+1} 无效 (HTTP {e.code}), trying next...")
                    continue
                raise


def _is_session_error(code: int, body: str) -> bool:
    """Stale/unknown MCP session (e.g. a concurrent token rotation reset it)."""
    if code not in (400, 404):
        return False
    return "session" in body.lower()


def mcp_call(method, params=None, _allow_session_retry=True, _drop_session=False):
    """Call MCP method. Auto-rotates token on credit errors.

    TOKEN/_session_id are snapshotted per call: rotation may switch the
    globals mid-flight from another thread, so each request must pin the
    auth it was built with.
    """
    global _session_id
    used_token = TOKEN
    sid = None if _drop_session else _session_id
    msg_id = _next_id()
    payload = {"jsonrpc": "2.0", "method": method, "id": msg_id}
    if params is not None:
        payload["params"] = params
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {used_token}"}
    if sid:
        headers["Mcp-Session-Id"] = sid
    req = urllib.request.Request(MCP_URL, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            new_sid = resp.headers.get("Mcp-Session-Id") or resp.headers.get("mcp-session-id")
            # Only adopt a returned session id while the token is unchanged —
            # a session minted for the old token must not clobber the new one.
            if new_sid and TOKEN == used_token:
                _session_id = new_sid
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        # Stale session: drop the session id and retry once with a fresh one.
        if _allow_session_retry and _is_session_error(e.code, body):
            print(f"  [MCP] HTTP {e.code} session invalid, retrying without session id...")
            return mcp_call(method, params, _allow_session_retry=False, _drop_session=True)
        # Check for credit errors and rotate token
        # (_rotate_token raises ALL_MCP_TOKENS_EXHAUSTED when no tokens left —
        #  even for single-token setups, so credits errors never pass silently)
        if _is_credit_error(body):
            print(f"  [MCP] 积分不足! HTTP {e.code} 响应: {body[:500]}")
            _rotate_token(used_token)
            return mcp_call(method, params)  # retry with new token
        print(f"HTTP {e.code}: {body[:500]}")
        raise


def mcp_notify(method, params=None):
    global _session_id
    payload = {"jsonrpc": "2.0", "method": method}
    if params is not None:
        payload["params"] = params
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {TOKEN}"}
    if _session_id:
        headers["Mcp-Session-Id"] = _session_id
    req = urllib.request.Request(MCP_URL, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30):
            pass
    except Exception:
        pass


def initialize(token=None, tokens=None):
    """Initialize MCP session.

    Args:
        token: Single token string (backward compatible).
        tokens: List of token strings (multi-token with auto-rotation on credit errors).
    On Colab, pass tokens=[...] for multi-token support.
    On Windows local, token auto-detected from ~/.codely-cli/mcp-oauth-tokens.json.
    """
    global _TOKENS, _token_idx, TOKEN
    if tokens:
        _TOKENS = tokens
        _token_idx = 0
        TOKEN = tokens[0]
    elif token:
        _TOKENS = [token]
        _token_idx = 0
        TOKEN = token
    if not TOKEN:
        raise RuntimeError("No MCP token. Pass tokens=['...'] or token='...' to initialize().")
    result = mcp_call("initialize", {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {"name": "codely-cli", "version": "1.0"},
    })
    mcp_notify("notifications/initialized")
    print(f"  [MCP] Initialized with {len(_TOKENS)} token(s).")
    return result


def reinitialize(tokens=None):
    """Reset MCP global state and re-init with new tokens.

    Used by web app to ensure clean state before each pipeline run.
    Without this, stale _session_id from a previous run causes errors.
    """
    global _TOKENS, _token_idx, TOKEN, _session_id, _msg_id
    _TOKENS = []
    _token_idx = 0
    TOKEN = ""
    _session_id = None
    _msg_id = 0
    return initialize(tokens=tokens) if tokens else initialize()


_INIT_LOCK = threading.Lock()


def ensure_initialized(tokens=None):
    """幂等初始化：已用相同 token 集初始化过则直接复用现有会话。

    与 reinitialize 的区别：绝不重置其他线程（如运行中 pipeline 步骤 2/3
    的图片/视频生成轮询）正在使用的全局会话状态。Web 后台线程
    （角色库图片生成、风格预览等）应优先用它。
    """
    with _INIT_LOCK:
        want = [t.strip() for t in (tokens or []) if t.strip()]
        if TOKEN and _TOKENS:
            # 已初始化且 token 集一致（或调用方未指定）→ 复用现有会话
            if not want or set(want) == set(_TOKENS):
                return
        return initialize(tokens=want or None)


def call_tool(name, arguments):
    """Call an MCP tool. Auto-rotates token on credit errors in response."""
    used_token = TOKEN  # snapshot — mcp_call may rotate before returning
    result = mcp_call("tools/call", {"name": name, "arguments": arguments})
    # Check response content for credit errors
    if "result" in result:
        content = result["result"].get("content", [])
        for item in content:
            if item.get("type") == "text":
                text = item.get("text", "")
                if _is_credit_error(text):
                    print(f"  [MCP] 积分不足! 工具响应: {text[:500]}")
                    _rotate_token(used_token)  # raises ALL_MCP_TOKENS_EXHAUSTED if no tokens left
                    return call_tool(name, arguments)  # retry with new token
    return result


def parse_task_id(result):
    """Extract task_id from MCP tool call response (Markdown or JSON)."""
    if "result" not in result:
        return ""
    content = result["result"].get("content", [])
    for item in content:
        if item.get("type") == "text":
            text = item["text"]
            try:
                return json.loads(text).get("task_id", "")
            except json.JSONDecodeError:
                pass
            m = re.search(r"Task\s+ID[\*\s'\"]*:\s*[`'\"]*([a-f0-9]+)", text)
            if m:
                return m.group(1)
    return ""


def poll_task(task_id, interval=40, max_wait=600, stop_check=None):
    """Poll check_task until completed or failed. Returns dict containing:
    - status, url (success)
    - status, raw (full raw check_task response + error message on failure)
    - status="stopped" if stop_check() returns True
    """
    elapsed = 0
    while elapsed < max_wait:
        if stop_check and stop_check():
            print(f"  [STOP] Polling interrupted for task {task_id[:16]}...")
            return {"status": "stopped"}
        try:
            result = call_tool("check_task", {"task_id": task_id})
        except Exception as e:
            err_str = str(e)
            if "ALL_MCP_TOKENS_EXHAUSTED" in err_str:
                raise  # propagate — all tokens dead
            print(f"  [{elapsed}s] Task {task_id[:16]}...: HTTP error ({type(e).__name__}: {e}), retrying in {interval}s...")
            elapsed += interval
            time.sleep(interval)
            continue
        if "result" not in result:
            elapsed += interval
            time.sleep(interval)
            continue

        content = result["result"].get("content", [])
        task_data = {}
        task_data["raw_response"] = json.dumps(result, ensure_ascii=False)

        # 1. Try resource block (contains JSON with output URLs + error info)
        for item in content:
            if item.get("type") == "resource":
                res_text = item.get("resource", {}).get("text", "")
                try:
                    res_json = json.loads(res_text)
                    task_data["status"] = res_json.get("status", "")
                    task_data["raw_json"] = json.dumps(res_json, ensure_ascii=False)
                    out = res_json.get("output") or {}
                    data = out.get("data") or {}
                    if res_json.get("error"):
                        task_data["error"] = res_json["error"]
                    elif data.get("error"):
                        task_data["error"] = data["error"]
                    # Only treat "message" as error if status is failed
                    elif data.get("message") and res_json.get("status") == "failed":
                        task_data["error"] = data["message"]
                    result_obj = data.get("result") or {}
                    url = (result_obj.get("video_url") or result_obj.get("image_url") or
                           result_obj.get("url") or "")
                    if url:
                        task_data["url"] = url
                    if "url" not in task_data:
                        # seedream returns image_urls (plural) inside result_obj
                        img_urls = (result_obj.get("image_urls") or
                                    data.get("imageUrls") or data.get("image_urls") or [])
                        if img_urls and isinstance(img_urls, list) and img_urls[0]:
                            task_data["url"] = img_urls[0]
                    if "url" not in task_data:
                        # Fallback: check data directly for url field
                        url = data.get("url") or data.get("imageUrl") or ""
                        if url:
                            task_data["url"] = url
                    break
                except json.JSONDecodeError:
                    pass

        # 2. Also check text block for URL (Markdown ![](URL))
        if "url" not in task_data:
            for item in content:
                if item.get("type") == "text":
                    text = item.get("text", "")
                    task_data["raw_text"] = text
                    if not task_data.get("status"):
                        if "completed" in text.lower() or "已完成" in text:
                            task_data["status"] = "completed"
                        elif "running" in text.lower() or "queued" in text.lower() or "进行中" in text:
                            task_data["status"] = "running"
                        elif "failed" in text.lower() or "失败" in text:
                            task_data["status"] = "failed"
                    m = re.search(r"!\[.*?\]\((https?://[^\s)]+)\)", text)
                    if m:
                        task_data["url"] = m.group(1)
                        break
                    if "url" not in task_data:
                        m = re.search(r"(https?://[^\s`'\")]+)", text)
                        if m:
                            task_data["url"] = m.group(1)
                            break

        status = task_data.get("status", "")
        if status:
            print(f"  [{elapsed}s] Task {task_id[:16]}...: {status}")
        if status == "completed":
            return task_data
        if status == "failed":
            print(f"  FAILED: {json.dumps(task_data, ensure_ascii=False)[:1000]}")
            return task_data

        # Sleep in small increments so stop_check responds quickly
        remaining = interval
        while remaining > 0:
            if stop_check and stop_check():
                print(f"  [STOP] Polling interrupted for task {task_id[:16]}...")
                return {"status": "stopped"}
            sleep_step = min(remaining, 5)
            time.sleep(sleep_step)
            remaining -= sleep_step
        elapsed += interval

    print(f"  TIMEOUT after {max_wait}s for task {task_id}")
    return {}


def download_file(url, dest):
    """Download a file from URL to local path."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            content_type = resp.headers.get("Content-Type", "")
            if "text/html" in content_type:
                print(f"  Download failed: got HTML page (not a file) from {url[:80]}")
                return False
            with open(dest, "wb") as f:
                f.write(resp.read())
        return os.path.getsize(dest) > 1000
    except Exception as e:
        print(f"  Download failed: {e}")
        return False


if __name__ == "__main__":
    initialize()
    print(f"Session: {_session_id}")
    tools = call_tool("tools/list", {})
    for t in tools.get("result", {}).get("tools", []):
        print(f"  - {t['name']}")
