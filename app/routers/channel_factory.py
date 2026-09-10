"""Channel Factory API — 📺频道工坊：LLM 批量生成 YouTube 频道信息 + 收藏 + Logo/Banner 生成.

数据流：
1. POST /generate → 后台线程调 LLM（resolve_provider 同步 urllib，复用 llm_client._extract_json）
   一次产 5 套完整频道信息 → 落盘 configs/channel_drafts.json（防刷新丢失），
   前端 2s 轮询 generate_status（单槽 409 并发守卫，同 characters AI 生成模式）。
2. 用户挑选 → POST /favorite 把候选从 drafts 移入 configs/channel_favorites.json。
3. 收藏后的下一步：POST /generate_assets 为该频道生成 Logo（1024x1024 圆形头像构图）
   与 Banner（YouTube 横幅，文字收在中央安全区），生图通道跟随当前配置 image_provider：
   - sensenova: pipeline/sensenova_image.text_to_image（worker 内注入 SENSENOVA_API_KEY）
   - mcp: app/page_mcp.PageMcpSession 独立会话（默认 seedream 通道，无需 confirm_cost）
   产物存 configs/channel_assets/{profile_id}/，经 /favorites/{pid}/asset/{kind} 预览
   （no-cache + 前端 ?v=mtime_ns 破缓存，同缩略图缓存策略）。
"""
import json
import os
import re
import threading
import time
import urllib.request
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, JSONResponse

from ..config_manager import detect_local_mcp_token, load_config, resolve_provider
from ..page_mcp import PageMcpSession
from ..paths import (
    CHANNEL_ASSETS_DIR, CHANNEL_DRAFTS_PATH, CHANNEL_FAVORITES_PATH,
)

router = APIRouter()

_ASSET_KINDS = ("logo", "banner")
_ID_RE = re.compile(r"^ch_[A-Za-z0-9_]+$")
_HEX_RE = re.compile(r"^#?[0-9a-fA-F]{6}$")


# ===========================================================================
# 存储层：configs/channel_drafts.json + configs/channel_favorites.json
# ===========================================================================

def _load_json(path: Path, default: dict) -> dict:
    if not path.exists():
        return default
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else default
    except (json.JSONDecodeError, OSError):
        return default


def _save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_drafts() -> list:
    return _load_json(CHANNEL_DRAFTS_PATH, {}).get("profiles", [])


def _save_drafts(profiles: list) -> None:
    _save_json(CHANNEL_DRAFTS_PATH, {"updated": time.time(), "profiles": profiles})


def _load_favorites() -> list:
    return _load_json(CHANNEL_FAVORITES_PATH, {}).get("profiles", [])


def _save_favorites(profiles: list) -> None:
    _save_json(CHANNEL_FAVORITES_PATH, {"updated": time.time(), "profiles": profiles})


# ===========================================================================
# LLM 生成：一次 5 套频道信息
# ===========================================================================

_gen_status: dict = {"status": "idle", "error": "", "count": 0}


def _llm_chat(base_url: str, api_key: str, model: str, p_type: str, prompt: str) -> str:
    """同步调 LLM chat/completions（独立小函数，便于测试 monkeypatch）。"""
    body = {
        "model": model,
        "messages": [
            {"role": "system",
             "content": "You are an expert YouTube channel strategist and brand "
                        "designer. Output valid JSON only — no markdown, no explanations."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.9,
        "max_tokens": 8192,
    }
    if p_type != "openai":
        body["reasoning_effort"] = "low"
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        method="POST",
    )
    req.add_header("Authorization", f"Bearer {api_key}")
    req.add_header("Content-Type", "application/json")
    req.add_header("User-Agent", "CodelyLLM/1.0")
    with urllib.request.urlopen(req, timeout=300) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    return result["choices"][0]["message"]["content"]


def _build_prompt(direction: str, avoid_names: list[str]) -> str:
    direction_block = (
        f'\n\nUSER DIRECTION (highest priority — all 5 concepts must fit this '
        f'direction, vary strongly WITHIN it): "{direction}"'
        if direction else
        "\n\nPick 5 clearly different sub-niches across the English-learning "
        "content ecosystem (no two concepts may be similar)."
    )
    avoid_block = "\n".join(f"- {n}" for n in avoid_names) or "(none)"
    return f"""You are a YouTube channel strategist and brand designer for a content studio producing English-learning videos (audience: overseas Chinese ESL learners).

Design exactly 5 COMPLETELY DIFFERENT YouTube channel concepts. Each concept is a full channel identity package the owner will use to create a brand-new YouTube channel.{direction_block}

Requirements:
- The 5 concepts must span clearly different sub-niches / tones / target segments — never two similar ones
- "name_en": catchy, brandable, 2-4 words, easy to spell and pronounce; not generic (avoid names like "English Learning Channel")
- "name_zh": Traditional Chinese (繁體中文) channel name matching the EN brand
- "handle": YouTube handle suggestion starting with @ (lowercase letters/numbers, no spaces, <=20 chars)
- "slogan": one short memorable English tagline (<=8 words)
- "description_en": channel "About" text in English, 60-110 words — what viewers get, tone, and implied upload rhythm
- "description_zh": the same description in Traditional Chinese (繁體中文), 80-140 characters
- "niche": the precise sub-niche in one English phrase
- "audience": target audience in one short English phrase
- "content_series": 3-5 recurring video series ideas, each "Series Name — one-line description" in English
- "tags": 10-16 SEO search keywords, a mix of English and Chinese search phrases viewers actually type
- "brand_colors": exactly 3 hex colors ["#RRGGBB", "#RRGGBB", "#RRGGBB"] (primary / accent / background)
- "brand_style": 3-6 English words describing the visual brand style (e.g. "flat pastel, rounded, friendly")

Do NOT reuse or closely imitate these existing channel names:
{avoid_block}

Output valid JSON only (no markdown, no explanations):
{{"channels": [{{"name_en": "...", "name_zh": "...", "handle": "@...", "slogan": "...", "description_en": "...", "description_zh": "...", "niche": "...", "audience": "...", "content_series": ["..."], "tags": ["..."], "brand_colors": ["#RRGGBB", "#RRGGBB", "#RRGGBB"], "brand_style": "..."}} x5]}}"""


def _normalize_profile(raw: dict, idx: int) -> dict | None:
    """字段规范化 + 服务端生成 id（ms+序号防撞）。name_en 缺失视为无效丢弃。"""
    if not isinstance(raw, dict):
        return None
    name_en = str(raw.get("name_en", "")).strip()
    if not name_en:
        return None
    handle = str(raw.get("handle", "")).strip().replace(" ", "").lower()
    if handle and not handle.startswith("@"):
        handle = "@" + handle
    colors = []
    for c in (raw.get("brand_colors") or [])[:3]:
        c = str(c).strip()
        if _HEX_RE.match(c):
            colors.append(c if c.startswith("#") else f"#{c}")
    while len(colors) < 3:
        colors.append(["#4F46E5", "#F59E0B", "#F8FAFC"][len(colors)])
    ms = int(time.time() * 1000)
    return {
        "id": f"ch_{ms}_{idx}",
        "created": time.time(),
        "name_en": name_en,
        "name_zh": str(raw.get("name_zh", "")).strip(),
        "handle": handle,
        "slogan": str(raw.get("slogan", "")).strip(),
        "description_en": str(raw.get("description_en", "")).strip(),
        "description_zh": str(raw.get("description_zh", "")).strip(),
        "niche": str(raw.get("niche", "")).strip(),
        "audience": str(raw.get("audience", "")).strip(),
        "content_series": [str(s).strip() for s in (raw.get("content_series") or [])
                           if s is not None and str(s).strip()],
        "tags": [str(s).strip() for s in (raw.get("tags") or [])
                 if s is not None and str(s).strip()],
        "brand_colors": colors,
        "brand_style": str(raw.get("brand_style", "")).strip(),
    }


def _generate_batch_worker(direction: str) -> None:
    """后台线程：LLM 生成 5 套频道信息 → 落盘 drafts。"""
    _gen_status.update({"status": "generating", "error": "", "count": 0})
    try:
        p_type, base_url, api_key, model = resolve_provider(load_config())
        if not api_key:
            raise RuntimeError(f"未配置 {p_type} 的 API Key，请在参数配置页面填写")
        if not model:
            raise RuntimeError("未指定模型（该 Provider 未配置模型列表）")
        avoid = [p.get("name_en", "") for p in _load_favorites() if p.get("name_en")]
        prompt = _build_prompt(direction, avoid)

        print(f"  [ChannelFactory] Requesting 5 channel concepts from {model} ({p_type})...")
        content = _llm_chat(base_url, api_key, model, p_type, prompt)

        from llm_client import _extract_json  # pipeline/ 已在 sys.path
        raw_list = _extract_json(content).get("channels") or []
        profiles = []
        for i, raw in enumerate(raw_list):
            p = _normalize_profile(raw, i)
            if p:
                profiles.append(p)
        if not profiles:
            raise RuntimeError("LLM 返回内容中没有有效频道方案（字段缺失或解析失败）")
        _save_drafts(profiles)
        print(f"  [ChannelFactory] Saved {len(profiles)} channel concepts to drafts")
        _gen_status.update({"status": "done", "count": len(profiles), "error": ""})
    except Exception as e:  # noqa: BLE001 — 错误信息原样落状态供前端展示
        print(f"  [ChannelFactory] ERROR: {e}")
        _gen_status.update({"status": "error", "count": 0, "error": str(e)[:300]})


@router.post("/api/channel_factory/generate")
async def api_generate(request: Request):
    """LLM 一次生成 5 套频道信息（静态路径，无通配遮蔽风险）。"""
    if _gen_status.get("status") == "generating":
        return JSONResponse({"ok": False, "error": "已有生成任务进行中，请稍候"}, status_code=409)
    try:
        data = await request.json()
    except Exception:
        data = {}
    direction = str(data.get("direction", "") or "").strip().replace("\n", " ")[:200]

    threading.Thread(target=_generate_batch_worker, args=(direction,),
                     daemon=True).start()
    return {"ok": True, "message": "LLM 生成中（约 1-2 分钟）..."}


@router.get("/api/channel_factory/generate_status")
async def api_generate_status():
    return _gen_status


@router.get("/api/channel_factory/drafts")
async def api_drafts():
    return {"profiles": _load_drafts()}


@router.post("/api/channel_factory/drafts/discard")
async def api_drafts_discard(request: Request):
    """丢弃候选：body {"id": "..."} 丢弃单个；body {} 清空整批。"""
    try:
        data = await request.json()
    except Exception:
        data = {}
    pid = str(data.get("id", "") or "")
    drafts = _load_drafts()
    if pid:
        drafts = [p for p in drafts if p.get("id") != pid]
    else:
        drafts = []
    _save_drafts(drafts)
    return {"ok": True, "count": len(drafts)}


@router.post("/api/channel_factory/favorite")
async def api_favorite(request: Request):
    """收藏候选 → favorites（同一 id 不可重复收藏；drafts 中同 id 移除）。"""
    try:
        data = await request.json()
    except Exception:
        return JSONResponse({"ok": False, "error": "请求体不是 JSON"}, status_code=400)
    profile = _normalize_profile(data if isinstance(data, dict) else {}, 0)
    if profile is None:
        return JSONResponse({"ok": False, "error": "缺少 name_en，无法收藏"}, status_code=400)
    # 保留客户端传来的原 id（素材目录按 id 归档，重收藏同 id 可复用已生成素材）
    orig_id = str(data.get("id", "") or "")
    if _ID_RE.match(orig_id):
        profile["id"] = orig_id

    favorites = _load_favorites()
    if any(p.get("id") == profile["id"] for p in favorites):
        return JSONResponse({"ok": False, "error": "该方案已在收藏夹中"}, status_code=409)
    favorites.insert(0, profile)
    _save_favorites(favorites)
    _save_drafts([p for p in _load_drafts() if p.get("id") != profile["id"]])
    return {"ok": True, "profile": profile}


@router.get("/api/channel_factory/favorites")
async def api_favorites():
    return {"profiles": _load_favorites()}


@router.delete("/api/channel_factory/favorites/{pid}")
async def api_favorite_delete(pid: str):
    """删除收藏条目（已生成的素材文件保留，重收藏同 id 可复用）。"""
    favorites = _load_favorites()
    remaining = [p for p in favorites if p.get("id") != pid]
    if len(remaining) == len(favorites):
        return JSONResponse({"ok": False, "error": "未找到该收藏"}, status_code=404)
    _save_favorites(remaining)
    return {"ok": True}


# ===========================================================================
# Logo / Banner 生成（跟随当前配置 image_provider）
# ===========================================================================

_asset_status: dict = {"status": "idle", "error": "", "profile_id": "",
                       "profile_name": "", "kinds": [], "results": {}, "logs": []}

# YouTube 头像是圆形裁切：构图必须圆形安全；横幅外圈在 TV/桌面端被裁切
_LOGO_PROMPT_TMPL = """Flat vector logo / emblem for a YouTube channel named "{name_en}" ({name_zh}).
Channel niche: {niche}. Slogan: "{slogan}".
Visual style: {brand_style}. Brand colors: primary {c0}, accent {c1}, background {c2}.

Design rules:
- A simple, bold, memorable central ICON that instantly evokes the niche — icon-dominant, minimal detail, flat vector, no photorealism
- Composition centered inside a CIRCLE and stays fully visible when cropped to a circular profile picture (YouTube avatar); generous padding around the icon
- Solid light/white background; use the brand colors for the icon
- At most one very short text element (a monogram or the single most distinctive word) — otherwise icon only; text must be large, clean and perfectly spelled
- No watermark, no clutter, no gradients mesh, no 3D render look"""


def _build_logo_prompt(p: dict) -> str:
    c = p.get("brand_colors") or ["#4F46E5", "#F59E0B", "#F8FAFC"]
    return _LOGO_PROMPT_TMPL.format(
        name_en=p.get("name_en", ""), name_zh=p.get("name_zh", ""),
        niche=p.get("niche", ""), slogan=p.get("slogan", ""),
        brand_style=p.get("brand_style", "modern flat, friendly"),
        c0=c[0], c1=c[1], c2=c[2])


_BANNER_PROMPT_TMPL = """YouTube channel banner artwork, wide 16:9 composition, for a channel named "{name_en}" ({name_zh}).
Channel niche: {niche}. Slogan: "{slogan}".
Visual style: {brand_style}. Brand colors: primary {c0}, accent {c1}, background {c2}.

Design rules:
- Modern flat illustration / clean graphic composition with subtle decorative elements evoking the niche
- CRITICAL: the channel name "{name_en}", the Chinese name "{name_zh}", the slogan and all key visual elements MUST stay inside the CENTRAL SAFE AREA (a centered horizontal band roughly 1546x423 in the 2048x1152 canvas) — everything outside it is cropped on TV and desktop
- Text large, clean, perfectly spelled; balanced hierarchy (EN name dominant, ZH name secondary, slogan small)
- No watermark, no clutter"""


def _build_banner_prompt(p: dict) -> str:
    c = p.get("brand_colors") or ["#4F46E5", "#F59E0B", "#F8FAFC"]
    return _BANNER_PROMPT_TMPL.format(
        name_en=p.get("name_en", ""), name_zh=p.get("name_zh", ""),
        niche=p.get("niche", ""), slogan=p.get("slogan", ""),
        brand_style=p.get("brand_style", "modern flat, friendly"),
        c0=c[0], c1=c[1], c2=c[2])


def _gen_asset_sensenova(prompt: str, dest: Path, size: str, log) -> bool:
    """SenseNova U1.5 Lite 文生图（worker 内已注入 SENSENOVA_API_KEY）。"""
    import sensenova_image  # pipeline/ 已在 sys.path
    url = sensenova_image.text_to_image(prompt, size=size, output_format="png")
    if not url:
        log("SenseNova 未返回图片 URL")
        return False
    if not sensenova_image.download_image(url, str(dest)):
        log("SenseNova 图片下载落盘失败")
        return False
    return True


def _gen_asset_mcp(prompt: str, dest: Path, width: int, height: int,
                   session: PageMcpSession, log) -> bool:
    """MCP generate_image（默认 seedream 通道，无需 confirm_cost）。"""
    gen_args = {
        "prompt": prompt,
        "image_size": json.dumps({"width": width, "height": height}),
        "output_format": "png",
    }
    result = session.call_tool("generate_image", gen_args)
    task_id = session.parse_task_id(result)
    if not task_id:
        raw = ""
        for item in result.get("result", {}).get("content", []):
            if item.get("type") == "text":
                raw = str(item.get("text", ""))[:300].replace("\n", " ")
                break
        log(f"MCP 未返回任务 ID（响应: {raw}）")
        return False
    data = session.poll_task(task_id, interval=10, max_wait=600)
    url = data.get("url", "")
    if data.get("status") != "completed" or not url:
        log(f"MCP 任务未完成: {data.get('status') or 'no status'}"
            + (f" error={data.get('error')}" if data.get("error") else ""))
        return False
    if not session.download_file(url, str(dest)):
        log("MCP 图片下载落盘失败")
        return False
    return True


def _generate_assets_worker(profile_id: str, kinds: list[str]) -> None:
    """后台线程：为收藏的频道生成 Logo/Banner（单槽，顺序逐个）。"""
    _asset_status.update({"status": "running", "error": "", "profile_id": profile_id,
                          "profile_name": "", "kinds": kinds, "results": {}, "logs": []})

    def log(msg: str) -> None:
        _asset_status["logs"].append(f"[{time.strftime('%H:%M:%S')}] {msg}")
        print(f"  [ChannelFactory] {msg}")

    try:
        favorites = _load_favorites()
        profile = next((p for p in favorites if p.get("id") == profile_id), None)
        if profile is None or not _ID_RE.match(profile_id):
            raise RuntimeError("收藏不存在（可能已被删除）")
        _asset_status["profile_name"] = profile.get("name_en", "")
        out_dir = CHANNEL_ASSETS_DIR / profile_id
        out_dir.mkdir(parents=True, exist_ok=True)

        config = load_config()
        provider = str(config.get("image_provider", "mcp"))
        log(f"生图通道: {provider}")

        mcp_session = None
        if provider == "mcp":
            tokens = [t.strip() for t in str(config.get("mcp_tokens", "") or "").splitlines()
                      if t.strip()]
            if not tokens:
                local = detect_local_mcp_token()
                if local:
                    tokens = [local]
            if not tokens:
                raise RuntimeError("未配置 MCP Token（模式配置 / 本地检测均为空）")
            mcp_session = PageMcpSession(tokens).initialize()
        else:
            key = str(config.get("sensenova_api_key", "") or "").strip()
            if not key:
                raise RuntimeError("image_provider=sensenova 但未配置 SenseNova API Key")
            # 与 pipeline_service._set_env 同源同值；运行中的 pipeline 每次启动会重设，无污染
            os.environ["SENSENOVA_API_KEY"] = key

        sizes = {"logo": ("1024x1024", 1024, 1024),
                 "banner": ("2720x1536", 2048, 1152)}
        results = {}
        for kind in kinds:
            if kind not in _ASSET_KINDS:
                continue
            dest = out_dir / f"{kind}.png"
            prompt = _build_logo_prompt(profile) if kind == "logo" else _build_banner_prompt(profile)
            log(f"开始生成 {kind} ...")
            if provider == "mcp":
                _, w, h = sizes[kind]
                ok = _gen_asset_mcp(prompt, dest, w, h, mcp_session, log)
            else:
                sn_size, _, _ = sizes[kind]
                ok = _gen_asset_sensenova(prompt, dest, sn_size, log)
            results[kind] = {"ok": ok, "file": f"{kind}.png" if ok else ""}
            log(f"{kind}: {'OK' if ok else 'FAIL'}")

        # 结束前重读 favorites 再回写素材元数据（缩小与 UI 删除操作的竞态窗口）
        favorites = _load_favorites()
        for p in favorites:
            if p.get("id") != profile_id:
                continue
            for kind, r in results.items():
                if r["ok"]:
                    p[kind] = r["file"]
                    p[f"{kind}_at"] = time.time()
        _save_favorites(favorites)

        if any(r["ok"] for r in results.values()):
            _asset_status.update({"status": "done", "results": results, "error": ""})
        else:
            _asset_status.update({"status": "error", "results": results,
                                  "error": "Logo 与 Banner 均生成失败，见日志"})
    except Exception as e:  # noqa: BLE001 — 错误信息原样落状态供前端展示
        print(f"  [ChannelFactory] ASSET ERROR: {e}")
        _asset_status.update({"status": "error", "error": str(e)[:300]})


@router.post("/api/channel_factory/generate_assets")
async def api_generate_assets(request: Request):
    """为收藏的频道生成 Logo+Banner（静态路径，须在 favorites/{pid} 之前注册防遮蔽）。"""
    if _asset_status.get("status") == "running":
        return JSONResponse({"ok": False, "error": "已有素材生成任务进行中，请稍候"}, status_code=409)
    try:
        data = await request.json()
    except Exception:
        data = {}
    pid = str(data.get("id", "") or "")
    kinds = [k for k in (data.get("kinds") or ["logo", "banner"]) if k in _ASSET_KINDS]
    if not pid or not _ID_RE.match(pid):
        return JSONResponse({"ok": False, "error": "无效的收藏 id"}, status_code=400)
    if not kinds:
        kinds = ["logo", "banner"]
    if not any(p.get("id") == pid for p in _load_favorites()):
        return JSONResponse({"ok": False, "error": "收藏不存在"}, status_code=404)

    threading.Thread(target=_generate_assets_worker, args=(pid, kinds),
                     daemon=True).start()
    return {"ok": True, "message": "素材生成中..."}


@router.get("/api/channel_factory/assets_status")
async def api_assets_status():
    return _asset_status


@router.get("/api/channel_factory/favorites/{pid}/asset/{kind}")
async def api_favorite_asset(pid: str, kind: str):
    """返回已生成的 Logo/Banner 图片（no-cache，前端带 ?v=mtime_ns 破缓存）。"""
    if kind not in _ASSET_KINDS or not _ID_RE.match(pid):
        return JSONResponse({"ok": False, "error": "无效参数"}, status_code=400)
    path = CHANNEL_ASSETS_DIR / pid / f"{kind}.png"
    if not path.exists():
        return JSONResponse({"ok": False, "error": "素材尚未生成"}, status_code=404)
    return FileResponse(path, media_type="image/png",
                        headers={"Cache-Control": "no-cache"})


@router.post("/api/channel_factory/favorites/{pid}/open_folder")
async def api_favorite_open_folder(pid: str):
    """在资源管理器中打开该频道的素材目录（explorer 启动慢，前端 3 秒防抖）。"""
    if not _ID_RE.match(pid):
        return JSONResponse({"ok": False, "error": "无效的收藏 id"}, status_code=400)
    out_dir = CHANNEL_ASSETS_DIR / pid
    if not any(p.get("id") == pid for p in _load_favorites()):
        return JSONResponse({"ok": False, "error": "收藏不存在"}, status_code=404)
    out_dir.mkdir(parents=True, exist_ok=True)
    os.startfile(str(out_dir))  # noqa: S606 — Windows 资源管理器打开目录
    return {"ok": True}
