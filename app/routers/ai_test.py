"""AI Test — LLM Playground API + 自定义 Provider CRUD（页面路由在 pages.py）."""
import asyncio
import json
import threading
import time

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..config_manager import (
    load_config, load_llm_providers, resolve_provider, save_llm_providers,
)
from ..paths import AI_TEST_CONFIG_PATH

router = APIRouter()


def _load_ai_test_config() -> dict:
    defaults = {"system_prompt": ""}
    if AI_TEST_CONFIG_PATH.exists():
        try:
            saved = json.loads(AI_TEST_CONFIG_PATH.read_text(encoding="utf-8"))
            for k in defaults:
                if k in saved:
                    defaults[k] = saved[k]
        except (json.JSONDecodeError, OSError):
            pass
    return defaults


def _save_ai_test_config(cfg: dict) -> None:
    AI_TEST_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    AI_TEST_CONFIG_PATH.write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")


@router.get("/api/ai_test/config")
async def api_ai_test_config_get():
    config = load_config()
    ai_cfg = _load_ai_test_config()
    return {
        "llm_provider": config.get("llm_provider", "sensenova"),
        "sensenova_model": config.get("sensenova_model", "deepseek-v4-flash"),
        "openai_model": config.get("openai_model", "grok-4.6"),
        "openai_base_url": config.get("openai_base_url", ""),
        "system_prompt": ai_cfg.get("system_prompt", ""),
        "custom_providers": load_llm_providers(),
    }


@router.put("/api/ai_test/config")
async def api_ai_test_config_put(request: Request):
    data = await request.json()
    ai_cfg = _load_ai_test_config()
    if "system_prompt" in data:
        ai_cfg["system_prompt"] = data["system_prompt"]
    _save_ai_test_config(ai_cfg)
    return {"ok": True}


# --- Custom LLM Provider CRUD ---

@router.get("/api/ai_test/providers")
async def api_providers_list():
    return {"providers": load_llm_providers()}


@router.post("/api/ai_test/providers")
async def api_providers_create(request: Request):
    data = await request.json()
    name = (data.get("name") or "").strip()
    base_url = (data.get("base_url") or "").strip()
    if not name or not base_url:
        return JSONResponse({"ok": False, "error": "名称和 Base URL 不能为空"}, status_code=400)
    providers = load_llm_providers()
    # Check duplicate name
    if any(p["name"] == name for p in providers):
        return JSONResponse({"ok": False, "error": "名称已存在"}, status_code=400)
    provider = {
        "id": f"custom_{int(time.time()*1000)}",
        "name": name,
        "base_url": base_url,
        "api_key": (data.get("api_key") or "").strip(),
        "models": [m.strip() for m in (data.get("models") or "").split(",") if m.strip()],
    }
    providers.append(provider)
    save_llm_providers(providers)
    return {"ok": True, "provider": provider}


@router.put("/api/ai_test/providers/{provider_id}")
async def api_providers_update(provider_id: str, request: Request):
    data = await request.json()
    providers = load_llm_providers()
    target = None
    for p in providers:
        if p["id"] == provider_id:
            target = p
            break
    if not target:
        return JSONResponse({"ok": False, "error": "未找到"}, status_code=404)
    if "name" in data:
        new_name = data["name"].strip()
        if new_name and new_name != target["name"]:
            if any(p["name"] == new_name for p in providers if p["id"] != provider_id):
                return JSONResponse({"ok": False, "error": "名称已存在"}, status_code=400)
            target["name"] = new_name
    if "base_url" in data:
        target["base_url"] = data["base_url"].strip()
    if "api_key" in data:
        target["api_key"] = data["api_key"].strip()
    if "models" in data:
        if isinstance(data["models"], str):
            target["models"] = [m.strip() for m in data["models"].split(",") if m.strip()]
        else:
            target["models"] = data["models"]
    save_llm_providers(providers)
    return {"ok": True, "provider": target}


@router.delete("/api/ai_test/providers/{provider_id}")
async def api_providers_delete(provider_id: str):
    providers = load_llm_providers()
    providers = [p for p in providers if p["id"] != provider_id]
    save_llm_providers(providers)
    return {"ok": True}


@router.post("/api/ai_test/chat")
async def api_ai_test_chat(request: Request):
    """SSE streaming chat completion endpoint.

    Reads config for API keys, supports SenseNova and OpenAI-compatible providers.
    Returns text/event-stream with token-level streaming.
    """
    import urllib.request
    import urllib.error

    data = await request.json()
    messages = [m for m in data.get("messages", [])
                if isinstance(m, dict) and m.get("role")]
    system_prompt = data.get("system_prompt", "")
    provider = data.get("provider", "sensenova")
    model = data.get("model", "")
    try:
        temperature = float(data.get("temperature", 0.8))
    except (TypeError, ValueError):
        temperature = 0.8
    try:
        max_tokens = int(data.get("max_tokens", 8192))
    except (TypeError, ValueError):
        max_tokens = 8192
    reasoning_effort = data.get("reasoning_effort", "low")
    api_key_override = data.get("api_key", "").strip()

    config = load_config()

    # Override config provider with the one selected in AI test page
    config["llm_provider"] = provider

    p_type, base_url, api_key, resolved_model = resolve_provider(config)
    if api_key_override:
        api_key = api_key_override
    if model:
        resolved_model = model
    model = resolved_model
    # p_type is "sensenova" or "openai" (custom → openai)
    provider = p_type

    if not api_key:
        return JSONResponse(
            {"error": f"未配置 {provider} 的 API Key，请在参数配置页面或左侧设置中填写"},
            status_code=400)
    if not model:
        return JSONResponse(
            {"error": "未指定模型（该 Provider 未配置模型列表，请在自定义 Provider 中添加）"},
            status_code=400)

    # Build messages with system prompt
    full_messages = []
    if system_prompt:
        full_messages.append({"role": "system", "content": system_prompt})
    full_messages.extend(messages)

    body = {
        "model": model,
        "messages": full_messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if provider != "openai":
        body["reasoning_effort"] = reasoning_effort

    body_bytes = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=body_bytes,
        method="POST",
    )
    req.add_header("Authorization", f"Bearer {api_key}")
    req.add_header("Content-Type", "application/json")
    req.add_header("User-Agent", "CodelyLLM/1.0")

    import queue as _queue

    def _produce(q: "_queue.Queue") -> None:
        """工作线程：阻塞读取上游 SSE 流。

        urllib 是同步 I/O，直接在 async generator（事件循环）里读会阻塞
        整个 Web 服务（包括运行中 pipeline 的日志 SSE），故移到线程中转。
        """
        import time as _time
        t0 = _time.time()
        usage_data = None
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                for raw_line in resp:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line or not line.startswith("data: "):
                        continue
                    payload = line[6:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    # Extract token
                    choices = chunk.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            q.put(("token", content))
                    # Extract usage (some APIs send it in the last chunk)
                    if chunk.get("usage"):
                        usage_data = chunk["usage"]
            q.put(("done", {"elapsed": round(_time.time() - t0, 2),
                            "usage": usage_data}))
        except urllib.error.HTTPError as e:
            err = e.read().decode("utf-8", errors="replace")[:500]
            q.put(("error", f"HTTP {e.code}: {err}"))
        except Exception as e:  # noqa: BLE001 — 错误信息原样转发给前端
            q.put(("error", str(e)[:300]))
        finally:
            q.put(None)

    async def event_stream():
        q: _queue.Queue = _queue.Queue()
        threading.Thread(target=_produce, args=(q,), daemon=True).start()
        while True:
            item = await asyncio.to_thread(q.get, True, None)
            if item is None:
                break
            kind, payload = item
            if kind == "token":
                yield f"data: {json.dumps({'type': 'token', 'content': payload})}\n\n"
            elif kind == "done":
                yield f"data: {json.dumps({'type': 'done', 'elapsed': payload['elapsed'], 'usage': payload['usage']})}\n\n"
            elif kind == "error":
                yield f"data: {json.dumps({'type': 'error', 'error': payload})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


