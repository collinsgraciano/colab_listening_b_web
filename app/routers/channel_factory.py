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
import urllib.error
import urllib.request
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, JSONResponse

from ..config_manager import detect_local_mcp_token, load_config, resolve_provider
from ..page_mcp import PageMcpSession
from ..paths import (
    CHANNEL_ASSETS_DIR, CHANNEL_DRAFTS_PATH, CHANNEL_FAVORITES_PATH,
    CHANNEL_REFERENCES_PATH,
)

router = APIRouter()

_ASSET_KINDS = ("logo", "banner")
_ID_RE = re.compile(r"^ch_[A-Za-z0-9_]+$")
_REF_ID_RE = re.compile(r"^ref_[A-Za-z0-9_]+$")
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
# 参考频道：用户收藏的标杆频道（名称+简介），生成时勾选作为风格参考
# ===========================================================================

# 首次访问播种：用户提供的 3 个同赛道参考频道（可删除/可自行新增）
_SEED_REFERENCES = [
    {
        "id": "ref_seed_1",
        "name": "天天聽英文 Everyday English Learning",
        "description": (
            "歡迎來到「天天聽英文」YouTube 頻道，這裡是你學習英文的最佳夥伴。我們精心製作的影片幫助你利用零碎時間反覆聆聽，"
            "輕鬆提升生活中的英語聽說能力。我們的內容基於日常生活中的真實場景，提供連貫性的對話練習，幫助你在實際情境中流暢地進行英文交流。\n"
            "Welcome to the \"Everyday English Learning\" YouTube channel! We are your best partner for learning English. "
            "Our carefully crafted videos help you make the most of your spare time with repeated listening, making it easy "
            "to enhance your English listening and speaking skills in everyday life. Our content is based on real-life scenarios, "
            "offering continuous dialogue practice to help you communicate fluently in practical situations."),
    },
    {
        "id": "ref_seed_2",
        "name": "強效英文 Powerful English",
        "description": (
            "Welcome to Powerful English.\n"
            "Here you can learn English in a fun and efficient way.\n"
            "Learning English doesn't have to be boring and difficult.\n"
            "My videos will help you improve your listening and speaking skills.\n"
            "What are you waiting for?\n"
            "Subscribe and learn English with us!\n"
            "這裡是「強效英文Powerful English」唯一的正式頻道！\n"
            "提供大家免費且實用的學習影片！\n"
            "讓您用最短的時間、學到最多的內容！\n"
            "快速提升您的聽力與口說能力！\n"
            "感謝大家的訂閱跟分享！"),
    },
    {
        "id": "ref_seed_3",
        "name": "學學英文吧 English with me",
        "description": (
            "您是否常常找不到話題，瞬間冷場呢??\n"
            "包含簡短問題與回答的英文對話，讓你從早到晚都能聊，有滿滿的話題來源可以暢所欲言！\n"
            "在這個頻道中，您可以獲得有效的英語聽力訓練，幫助您提高聽力、累積詞彙量。不管您是初學者還是想更流利地使用英語，我們都有適合您的內容。\n"
            "💡 學習重點：\n"
            "    提升聽力：藉由聽標準英語母語者的發音，快速提高聽力，辨認不同口音和語調。\n"
            "    流利口語：持之以恆反覆聆聽、經過模仿發音和大聲朗讀，逐步達到流利的英語口語。\n"
            "    中文配音：我們的影片包含了中文解說，讓您更輕鬆理解，不需一直盯著螢幕，即使是邊做家事也能聆聽，讓學習效果倍增。\n"
            "    增加詞彙：提供初級詞彙和短語，擴充英語詞彙庫，在面對各種不同的情境時也能從容自信。\n"
            "👍 請按讚、訂閱並分享我們的影片！訂閱後，點擊通知鈴鐺，就能接收最新影片通知！\n"
            "❤️ 如果您喜歡我們的內容，請訂閱、分享和點贊。您的支持對我們非常重要🙏🏻。ENGLISH WITH ME!! 讓我們一起學英語!!"),
    },
]


def _load_references() -> list:
    if not CHANNEL_REFERENCES_PATH.exists():
        now = time.time()
        seeded = [{**r, "created": now} for r in _SEED_REFERENCES]
        _save_references(seeded)
        return seeded
    return _load_json(CHANNEL_REFERENCES_PATH, {}).get("references", [])


def _save_references(references: list) -> None:
    _save_json(CHANNEL_REFERENCES_PATH, {"updated": time.time(), "references": references})


# ===========================================================================
# LLM 生成：一次 5 套频道信息
# ===========================================================================

_gen_status: dict = {"status": "idle", "error": "", "count": 0}


def _llm_chat(base_url: str, api_key: str, model: str, p_type: str,
              prompt: str, temperature: float = 0.9) -> tuple[str, str]:
    """同步调 LLM chat/completions，返回 (content, finish_reason)。

    max_tokens 优先 16384（5 套长简介易顶到 8192 被截断）；Provider 拒绝该上限
    （HTTP 400 报文提到 max_tokens）时回退 8192 重试一次。
    独立小函数，便于测试 monkeypatch。
    """
    body = {
        "model": model,
        "messages": [
            {"role": "system",
             "content": "You are an expert YouTube channel strategist and brand "
                        "designer. Output valid JSON only — no markdown, no explanations."},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": 16384,
    }
    if p_type != "openai":
        body["reasoning_effort"] = "low"
    last_err: Exception | None = None
    for max_tokens in (16384, 8192):
        body["max_tokens"] = max_tokens
        req = urllib.request.Request(
            f"{base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            method="POST",
        )
        req.add_header("Authorization", f"Bearer {api_key}")
        req.add_header("Content-Type", "application/json")
        req.add_header("User-Agent", "CodelyLLM/1.0")
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                result = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8", errors="replace")[:300]
            except Exception:  # noqa: BLE001
                pass
            last_err = RuntimeError(f"LLM API HTTP {e.code}: {detail}")
            if max_tokens == 16384 and e.code == 400 and "max_tokens" in detail.lower():
                print("  [ChannelFactory] Provider 拒绝 max_tokens=16384，回退 8192 重试")
                continue
            raise last_err from e
        choice = (result.get("choices") or [{}])[0]
        return (str(choice.get("message", {}).get("content", "")),
                str(choice.get("finish_reason", "")))
    raise last_err or RuntimeError("LLM 请求失败")


# ---------------------------------------------------------------------------
# 字段级兜底提取：LLM 输出含未转义内层引号 / 结构损坏导致整体 JSON 解析失败时，
# 按 name_en 键切段、逐字段正则提取（损坏值在引号处截断，只损失该字段不整体失败）
# ---------------------------------------------------------------------------
_SALVAGE_STR = r'"((?:[^"\\]|\\.)*)"'
_SALVAGE_STR_FIELDS = ("name_zh", "handle", "slogan", "description_en",
                       "description_zh", "niche", "audience", "brand_style")


def _json_unescape(s: str) -> str:
    try:
        return json.loads(f'"{s}"')
    except Exception:  # noqa: BLE001 — 非法转义序列原样返回
        return s.replace("\\n", "\n").replace('\\"', '"').replace("\\\\", "\\")


def _salvage_profiles_regex(text: str) -> list[dict]:
    out = []
    for seg in re.split(r'"name_en"\s*:\s*"', text)[1:]:
        raw = {"name_en": _json_unescape(seg.split('"', 1)[0])}
        for key in _SALVAGE_STR_FIELDS:
            m = re.search(rf'"{key}"\s*:\s*{_SALVAGE_STR}', seg)
            if m:
                raw[key] = _json_unescape(m.group(1))
        for key in ("content_series", "tags"):
            m = re.search(rf'"{key}"\s*:\s*\[(.*?)\]', seg, re.DOTALL)
            if m:
                raw[key] = [_json_unescape(x) for x in
                            re.findall(_SALVAGE_STR, m.group(1))]
        m = re.search(r'"brand_colors"\s*:\s*\[(.*?)\]', seg, re.DOTALL)
        if m:
            raw["brand_colors"] = [
                x for x in (_json_unescape(s) for s in re.findall(_SALVAGE_STR, m.group(1)))
                if re.fullmatch(r"#?[0-9a-fA-F]{6}", x)][:3]
        out.append(raw)
    return out


_SIMILARITY_MODES = {
    "light": (
        "Lightly inspired — take only the general positioning, audience and tone "
        "from the references; the concepts, names and descriptions must be clearly "
        "distinct and original."),
    "medium": (
        "Moderately modeled — follow the reference channels' description structure, "
        "section organization and bilingual (EN + Traditional Chinese) presentation "
        "style; keep all names, topics and wording original."),
    "high": (
        "Closely modeled (HIGHEST PRIORITY — this overrides any 'be different/original' "
        "wording elsewhere in this prompt; that wording applies only to topics and names "
        "BETWEEN concepts). Mirror the reference channels' description structure "
        "sentence-pattern by sentence-pattern — opening welcome line, what viewers get, "
        "learning-point bullets (with emoji if the references use them), like/subscribe "
        "call-to-action — plus their bilingual layout and overall tone. Change only the "
        "channel name, handle, specific topic details and example wording; the result "
        "must feel like the same family of channels, not merely 'inspired by' them."),
}


def _build_prompt(direction: str, avoid_names: list[str],
                  references: list[dict] | None = None,
                  count: int = 5, similarity: str = "medium") -> str:
    has_refs = bool(references)
    if has_refs:
        task_line = (
            f"Design exactly {count} YouTube channel concepts that ALL belong to the "
            "SAME style family as the reference channels below — they differ from each "
            "other only in specific topic, angle and name, NOT in style."
        )
    else:
        task_line = (
            f"Design exactly {count} COMPLETELY DIFFERENT YouTube channel concepts. "
            "Each concept is a full channel identity package the owner will use to "
            "create a brand-new YouTube channel."
        )
    if direction:
        direction_block = (
            f'\n\nUSER DIRECTION (highest priority — all {count} concepts must fit this '
            f'direction, vary strongly WITHIN it): "{direction}"'
        )
    elif has_refs:
        direction_block = (
            f"\n\nVary only the specific topics/angles across the {count} concepts — "
            "the style stays unified per the SIMILARITY LEVEL below."
        )
    else:
        direction_block = (
            f"\n\nPick {count} clearly different sub-niches across the English-learning "
            "content ecosystem (no two concepts may be similar)."
        )
    references_block = ""
    if references:
        parts = [f'=== Reference {i}: {r.get("name", "")} ===\n{r.get("description", "")}'
                 for i, r in enumerate(references, 1)]
        references_block = (
            "\n\nREFERENCE CHANNELS the user admires (learn from their positioning, "
            "description structure, tone and content focus — e.g. bilingual description "
            "style, clear learning-point bullets, warm encouraging call-to-action. "
            "Never reuse their channel names):\n\n" + "\n\n".join(parts)
            + "\n\nSIMILARITY LEVEL (HIGHEST-PRIORITY style instruction — it overrides "
              "any 'be different/original' wording elsewhere in this prompt; that "
              "wording applies only to topics and names BETWEEN concepts): "
            + _SIMILARITY_MODES.get(similarity, _SIMILARITY_MODES["medium"])
        )
    avoid_block = "\n".join(f"- {n}" for n in avoid_names) or "(none)"
    diversity_bullet = (
        f"- Each of the {count} concepts targets a clearly different specific topic/angle "
        "and has its own distinct name — but ALL of them must stay within the reference "
        "channels' style family at the SIMILARITY LEVEL specified below"
        if has_refs else
        f"- The {count} concepts must span clearly different sub-niches / tones / target "
        "segments — never two similar ones"
    )
    return f"""You are a YouTube channel strategist and brand designer for a content studio producing English-learning videos (audience: overseas Chinese ESL learners).

{task_line}{direction_block}{references_block}

Requirements:
{diversity_bullet}
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
{{"channels": [{{"name_en": "...", "name_zh": "...", "handle": "@...", "slogan": "...", "description_en": "...", "description_zh": "...", "niche": "...", "audience": "...", "content_series": ["..."], "tags": ["..."], "brand_colors": ["#RRGGBB", "#RRGGBB", "#RRGGBB"], "brand_style": "..."}} x{count}]}}"""


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


def _generate_batch_worker(direction: str, reference_ids: list | None = None,
                           count: int = 5, similarity: str = "medium") -> None:
    """后台线程：LLM 生成频道信息 → 落盘 drafts。

    reference_ids：勾选的参考频道 id（worker 内重读文件，最多取 3 个）。
    count：本批套数（1-10）；similarity：与参考的相似程度（light/medium/high）。
    """
    _gen_status.update({"status": "generating", "error": "", "count": 0})
    try:
        p_type, base_url, api_key, model = resolve_provider(load_config())
        if not api_key:
            raise RuntimeError(f"未配置 {p_type} 的 API Key，请在参数配置页面填写")
        if not model:
            raise RuntimeError("未指定模型（该 Provider 未配置模型列表）")
        ref_ids = {r for r in (reference_ids or []) if isinstance(r, str)}
        references = [r for r in _load_references() if r.get("id") in ref_ids][:3]
        avoid = [p.get("name_en", "") for p in _load_favorites() if p.get("name_en")]
        prompt = _build_prompt(direction, avoid, references, count, similarity)
        if references:
            print(f"  [ChannelFactory] Using {len(references)} reference channel(s): "
                  + ", ".join(r.get("name", "") for r in references)
                  + f" (similarity={similarity})")

        print(f"  [ChannelFactory] Requesting 5 channel concepts from {model} ({p_type})...")
        # 相似度越高温度越低：high 紧贴参考需要稳定的模仿输出
        temperature = {"light": 0.9, "medium": 0.8, "high": 0.6}.get(similarity, 0.9)
        content, finish_reason = _llm_chat(base_url, api_key, model, p_type, prompt,
                                           temperature)
        if finish_reason == "length":
            print("  [ChannelFactory] WARNING: LLM 输出被 max_tokens 截断"
                  "（finish_reason=length），将尝试修复/兜底提取")

        from llm_client import _extract_json  # pipeline/ 已在 sys.path
        raw_list: list = []
        try:
            data = _extract_json(content)
            if isinstance(data, dict):
                raw_list = data.get("channels") or []
        except Exception as parse_err:  # noqa: BLE001 — 落 salvage 兜底
            print(f"  [ChannelFactory] 整体 JSON 解析失败，尝试字段级兜底提取: {parse_err}")
        if not raw_list:
            raw_list = _salvage_profiles_regex(content)
        if not raw_list:
            raise RuntimeError(
                "LLM 返回内容无法解析（已尝试截断修复与字段级兜底提取），请重试或更换模型")
        profiles = []
        for i, raw in enumerate(raw_list):
            p = _normalize_profile(raw, i)
            if p:
                profiles.append(p)
        profiles = profiles[:count]  # LLM 偶尔无视数量/兜底提取多收：按请求套数截断
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
    reference_ids = data.get("reference_ids")
    if not isinstance(reference_ids, list):
        reference_ids = []
    reference_ids = [str(r) for r in reference_ids
                     if isinstance(r, str) and _REF_ID_RE.match(r)][:3]
    try:
        count = max(1, min(10, int(data.get("count", 5))))
    except (TypeError, ValueError):
        count = 5
    similarity = data.get("similarity")
    if similarity not in _SIMILARITY_MODES:
        similarity = "medium"

    threading.Thread(target=_generate_batch_worker,
                     args=(direction, reference_ids, count, similarity),
                     daemon=True).start()
    return {"ok": True, "message": f"LLM 生成中（{count} 套，约 1-2 分钟）..."}


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


# ===========================================================================
# 参考频道 API（首次 GET 播种 3 个内置参考；生成时勾选作为风格参考）
# ===========================================================================

@router.get("/api/channel_factory/references")
async def api_references():
    return {"references": _load_references()}


@router.post("/api/channel_factory/references")
async def api_references_add(request: Request):
    """新增参考频道卡片：{name, description}。"""
    try:
        data = await request.json()
    except Exception:
        return JSONResponse({"ok": False, "error": "请求体不是 JSON"}, status_code=400)
    name = str(data.get("name", "") or "").strip()[:120]
    description = str(data.get("description", "") or "").strip()[:4000]
    if not name or not description:
        return JSONResponse({"ok": False, "error": "名称和简介不能为空"}, status_code=400)
    references = _load_references()
    ref = {"id": f"ref_{int(time.time() * 1000)}", "name": name,
           "description": description, "created": time.time()}
    references.insert(0, ref)
    _save_references(references)
    return {"ok": True, "reference": ref}


@router.delete("/api/channel_factory/references/{rid}")
async def api_references_delete(rid: str):
    if not _REF_ID_RE.match(rid):
        return JSONResponse({"ok": False, "error": "无效的参考 id"}, status_code=400)
    references = _load_references()
    remaining = [r for r in references if r.get("id") != rid]
    if len(remaining) == len(references):
        return JSONResponse({"ok": False, "error": "未找到该参考"}, status_code=404)
    _save_references(remaining)
    return {"ok": True}


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
    """收藏列表（附带已生成素材的本地绝对路径 logo_path/banner_path，供一键复制）。"""
    profiles = _load_favorites()
    for p in profiles:
        pid = str(p.get("id", ""))
        for kind in _ASSET_KINDS:
            path = ""
            if _ID_RE.match(pid):
                f = CHANNEL_ASSETS_DIR / pid / f"{kind}.png"
                if f.exists():
                    path = str(f)
            p[f"{kind}_path"] = path
    return {"profiles": profiles}


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
