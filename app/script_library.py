"""Script library — batch script generation, storage, and quality review.

Storage: configs/script_library/{id}.json — one doc per script:
  {id, topic, cefr, structure, llm_provider, llm_model, num_lines, created,
   status: draft|reviewed|used, review, used_by, used_at, script: {...}}

Generation reuses pipeline's LLM clients (generate_listening_script /
generate_quest_script) + _validate_script, executed serially in a background
thread. The batch-selected provider is applied via a thread-local LLM override
(set_llm_env_override) — it never touches os.environ, so concurrent pipeline
runs are unaffected. AI review follows topics_ai's chat pattern but with an
explicit provider override.
"""
import json
import random
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .config_manager import load_mode_config, resolve_provider
from style_manager import resolve_style_prompt

WEB_ROOT = Path(__file__).parent.parent.resolve()
SCRIPTS_DIR = WEB_ROOT / "configs" / "script_library"

# 各模式独立记录的「已用主题」——同一主题在每个模式各可用一次
_USED_BY_MODE_PATH = SCRIPTS_DIR / "used_topics_by_mode.json"
_store_lock = threading.Lock()
# 脚本文档读-改-写互斥（编辑保存 vs 审查/修复后台线程并发写同一 doc）
_doc_lock = threading.Lock()

_PIPELINE_DIR = WEB_ROOT / "pipeline"
if str(_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_DIR))

from llm_client import (  # noqa: E402
    _enforce_rate_limit, _extract_json, resolve_max_line_words,
    set_llm_env_override)

DEFAULT_LINES = {"original": 18, "original_static": 18, "original_cutout": 18, "quest": 48}

# 简体独有字（繁体无此字形）— 检测中文文案误用简体
_SIMP_ONLY_CHARS = set(
    "听说读发买卖开关门问间东车贝见长张网风飞电话号亿层让认语词试请谁来对办还这"
    "个为从会被过儿女严举义乐乡书经济观现检标样员损规则亲课业归块总换热当")

# 性别关键词（前后留空格做词边界匹配）
_GENDER_WORDS = {
    "male": (" man ", " boy ", " he ", " his ", " male "),
    "female": (" woman ", " girl ", " she ", " her ", " female "),
}

# 批量生成/审查共用的运行状态（单任务互斥 + 停止信号 + 进度快照供断线恢复查询）
_batch_state: dict[str, Any] = {"running": False, "stop": threading.Event(),
                                "snapshot": {}}


def _set_snapshot(**kw: Any) -> None:
    _batch_state["snapshot"] = {**(_batch_state.get("snapshot") or {}), **kw}


def batch_status() -> dict:
    """后台批量任务状态（页面刷新后恢复进度显示用）。"""
    return {"running": bool(_batch_state["running"]),
            "snapshot": dict(_batch_state.get("snapshot") or {})}


# ===========================================================================
# Storage
# ===========================================================================

def _doc_path(sid: str) -> Path:
    return SCRIPTS_DIR / f"{sid}.json"


def _script_title(script: dict, topic: str = "") -> str:
    return script.get("youtube_title") or script.get("title") or topic


def _doc_meta(doc: dict) -> dict:
    """Full doc → list-view metadata (no script payload)."""
    script = doc.get("script", {}) or {}
    dialogue = script.get("dialogue", []) or []
    preview = dialogue[0].get("text", "") if dialogue else ""
    review = doc.get("review") or {}
    return {
        "id": doc.get("id", ""),
        "topic": doc.get("topic", ""),
        "title": _script_title(script, doc.get("topic", "")),
        "cefr": doc.get("cefr", ""),
        "structure": doc.get("structure", ""),
        "llm_provider": doc.get("llm_provider", ""),
        "llm_model": doc.get("llm_model", ""),
        "num_lines": doc.get("num_lines", len(dialogue)),
        "created": doc.get("created", 0),
        "status": doc.get("status", "draft"),
        "score": review.get("score"),
        "verdict": review.get("verdict", ""),
        "reviewed": review.get("score") is not None,
        "stale": bool(review.get("stale")),
        "used_by": doc.get("used_by", ""),
        "used_at": doc.get("used_at", 0),
        "lines": len(dialogue),
        "preview": (preview or "")[:80],
        "local_issues": len(review.get("local_issues", []) or []),
        # 角色性别（绑定时对照，防止绑定角色与脚本人物性别相反）
        "genders": {
            "char_a": (script.get("char_a_gender") or ""),
            "char_b": (script.get("char_b_gender") or ""),
        },
    }


def doc_meta(doc: dict) -> dict:
    """Public alias of _doc_meta for API responses."""
    return _doc_meta(doc)


def list_scripts(structure: str = "", status: str = "", q: str = "") -> list[dict]:
    """List script metadata. status: draft|reviewed|used|unused|''(all).

    只匹配 script_*.json 脚本文档 — 排除 used_topics_by_mode.json 等状态文件，
    避免其被渲染成"0 行"幽灵卡片。
    """
    SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    items: list[dict] = []
    for f in sorted(SCRIPTS_DIR.glob("script_*.json")):
        try:
            doc = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not doc.get("id"):
            doc["id"] = f.stem  # 兜底：文件名即权威 id
        meta = _doc_meta(doc)
        if structure and meta["structure"] != structure:
            continue
        if status == "unused":
            if meta["status"] == "used":
                continue
        elif status and meta["status"] != status:
            continue
        if q:
            needle = q.lower()
            hay = f"{meta['topic']} {meta['title']} {meta['preview']}".lower()
            if needle not in hay:
                continue
        items.append(meta)
    # 排序：审查分数降序（未审查视为 -1 排后），同分按创建时间降序
    items.sort(key=lambda m: (m["score"] if m["score"] is not None else -1,
                              m["created"]), reverse=True)
    return items


def get_script_doc(sid: str) -> dict | None:
    # 只接受脚本文档 id（排除 used_topics_by_mode 等状态文件名）
    if not sid.startswith("script_"):
        return None
    p = _doc_path(sid)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _write_doc(doc: dict) -> None:
    SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    _doc_path(doc["id"]).write_text(
        json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")


def save_new_script(script: dict, meta: dict) -> dict:
    sid = f"script_{int(time.time() * 1000):x}_{random.randint(0, 0xffff):04x}"
    doc = {
        "id": sid,
        "topic": meta.get("topic", ""),
        "cefr": meta.get("cefr", ""),
        "structure": meta.get("structure", "original"),
        "llm_provider": meta.get("llm_provider", ""),
        "llm_model": meta.get("llm_model", ""),
        "num_lines": meta.get("num_lines", 0),
        "created": time.time(),
        "status": "draft",
        "review": meta.get("review"),
        "used_by": "",
        "used_at": 0,
        "script": script,
    }
    _write_doc(doc)
    return doc


def update_script(sid: str, patch: dict) -> dict | None:
    """Merge-edit a script doc (script payload and/or topic/cefr).

    内容变更时重算 local_checks；已有 AI 审查结论则标记 stale（已编辑，过期）。
    """
    with _doc_lock:
        doc = get_script_doc(sid)
        if not doc:
            return None
        content_changed = isinstance(patch.get("script"), dict)
        if content_changed:
            doc["script"] = patch["script"]
        if str(patch.get("topic", "")).strip():
            doc["topic"] = str(patch["topic"]).strip()
        if patch.get("cefr"):
            doc["cefr"] = patch["cefr"]
        if content_changed:
            review = doc.get("review") or {}
            review["local_issues"] = local_checks(
                doc["script"], doc.get("structure", "original"),
                len(doc["script"].get("dialogue") or []))
            if review.get("score") is not None:
                review["stale"] = True
            doc["review"] = review
        _write_doc(doc)
    return doc


def delete_script(sid: str) -> bool:
    if not sid.startswith("script_"):
        return False
    p = _doc_path(sid)
    if p.exists():
        p.unlink()
        return True
    return False


def mark_used(sid: str, run_name: str) -> dict | None:
    """Mark a script as used by a run (idempotent)."""
    with _doc_lock:
        doc = get_script_doc(sid)
        if not doc:
            return None
        if doc.get("status") != "used":
            doc["status"] = "used"
            doc["used_by"] = run_name
            doc["used_at"] = time.time()
            _write_doc(doc)
    return doc


def reset_used(sid: str) -> dict | None:
    """Reset the used mark → back to reviewed (if AI-reviewed) or draft."""
    with _doc_lock:
        doc = get_script_doc(sid)
        if not doc:
            return None
        review = doc.get("review") or {}
        doc["status"] = "reviewed" if review.get("score") is not None else "draft"
        doc["used_by"] = ""
        doc["used_at"] = 0
        _write_doc(doc)
    # 同步移除该模式已用主题记录（若由该脚本写入）
    if doc["status"] != "used":
        remove_topic_used_mode(doc.get("structure", ""), doc.get("topic", ""), sid)
    return doc


# ===========================================================================
# Per-mode used topics (各模式独立记录已用主题)
# ===========================================================================

def _load_used_by_mode() -> dict[str, dict]:
    if _USED_BY_MODE_PATH.exists():
        try:
            data = json.loads(_USED_BY_MODE_PATH.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_used_by_mode(data: dict[str, dict]) -> None:
    SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    _USED_BY_MODE_PATH.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def mark_topic_used_mode(mode: str, topic: str, script_id: str = "",
                         run_name: str = "") -> None:
    """Record topic as used IN ONE MODE only (idempotent per mode+topic)."""
    if mode not in DEFAULT_LINES or not topic:
        return
    with _store_lock:
        data = _load_used_by_mode()
        entry = data.setdefault(mode, {})
        if topic not in entry:
            entry[topic] = {"script_id": script_id, "run": run_name,
                            "used_at": time.time()}
            _save_used_by_mode(data)


def remove_topic_used_mode(mode: str, topic: str, script_id: str = "") -> None:
    """Remove a mode-used record (when resetting a script's used mark)."""
    if mode not in DEFAULT_LINES or not topic:
        return
    with _store_lock:
        data = _load_used_by_mode()
        entry = data.get(mode, {})
        rec = entry.get(topic)
        if rec is None:
            return
        # 若记录由其他脚本/全新生成写入，且未指定要删的 script_id，则不删
        if script_id and rec.get("script_id") and rec.get("script_id") != script_id:
            return
        entry.pop(topic, None)
        _save_used_by_mode(data)


def used_topics_for_mode(mode: str) -> list[str]:
    return list(_load_used_by_mode().get(mode, {}).keys())


def used_by_mode_all() -> dict[str, list[str]]:
    return {m: list(v.keys()) for m, v in _load_used_by_mode().items()}


def library_topics_by_mode() -> dict[str, list[str]]:
    """各模式下「库中已有脚本」的主题集合（任意状态）— 前端防重复生成提示用。"""
    out: dict[str, set[str]] = {}
    for f in SCRIPTS_DIR.glob("script_*.json"):
        try:
            doc = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        mode = doc.get("structure", "")
        topic = doc.get("topic", "")
        if mode and topic:
            out.setdefault(mode, set()).add(topic)
    return {m: sorted(v) for m, v in out.items()}


# ===========================================================================
# Batch generation (runs in a background thread; events via queue)
# ===========================================================================

def is_batch_running() -> bool:
    return bool(_batch_state["running"])


def request_stop_batch() -> None:
    _batch_state["stop"].set()


def _resolve_batch_provider(provider_id: str, model: str, structure: str):
    """Resolve batch-selected provider against the structure's mode config.

    Returns ((p_type, base_url, api_key, model), mode_cfg).
    """
    cfg = dict(load_mode_config(structure))
    cfg["llm_provider"] = provider_id or cfg.get("llm_provider", "sensenova")
    if model:
        p_type0, _, _, _ = resolve_provider(cfg)
        if p_type0 == "sensenova":
            cfg["sensenova_model"] = model
        else:
            cfg["openai_model"] = model
    return resolve_provider(cfg), cfg


def _build_llm_override(provider_id: str, model: str, structure: str) -> dict:
    """构建批量生成专用的线程局部 LLM 配置（不修改 os.environ）。

    返回 dict 的键名与 env var 同名，供 set_llm_env_override 使用——
    批量生成线程读自己的 override，运行中的 pipeline 线程读 os.environ，
    两者并发互不污染。
    """
    (p_type, base_url, api_key, resolved_model), cfg = _resolve_batch_provider(
        provider_id, model, structure)
    if not api_key:
        raise RuntimeError(
            f"未配置所选大模型的 API Key（provider={provider_id}）— "
            f"请先在「参数配置」页面填写，或在「AI 对话测试」页管理自定义 Provider")

    ov: dict[str, str] = {
        "LLM_PROVIDER": p_type,
        "LLM_RETRIES": str(cfg.get("llm_retries", 10)),
        "CHARACTER_OVERRIDES": "",  # 批量脚本是通用脚本：不注入角色覆盖
        # 画面风格线程隔离：脚本 prompt 里 get_active_style_prompt() 走
        # _env_get 读这两个键，避免读到运行中 pipeline 写入 os.environ 的风格
        "VISUAL_STYLE_ID": str(cfg.get("visual_style", "pixar3d")),
        "VISUAL_STYLE_PROMPT": resolve_style_prompt(str(cfg.get("visual_style", "pixar3d"))),
    }
    if p_type == "sensenova":
        ov["SENSENOVA_API_KEY"] = api_key
        ov["SENSENOVA_MODEL"] = resolved_model or "deepseek-v4-flash"
    else:
        ov["OPENAI_BASE_URL"] = base_url
        ov["OPENAI_API_KEY"] = api_key
        ov["OPENAI_MODEL"] = resolved_model or "grok-4.6"
    if cfg.get("llm_min_interval"):
        ov["LLM_MIN_INTERVAL"] = str(cfg["llm_min_interval"])
    # QA 轮数全模式生效（quest 读 QUEST_QA_MAX_ROUNDS，original* 读 LISTENING_QA_MAX_ROUNDS）
    if cfg.get("quest_qa_rounds") is not None and cfg.get("quest_qa_rounds") != "":
        ov["LISTENING_QA_MAX_ROUNDS"] = str(cfg["quest_qa_rounds"])
    # 每行最大词数（字幕两行约束）：prompt + 门禁共用
    if cfg.get("max_line_words"):
        ov["LISTENING_MAX_LINE_WORDS"] = str(cfg["max_line_words"])
        ov["QUEST_MAX_LINE_WORDS"] = str(cfg["max_line_words"])
    if structure == "quest":
        if cfg.get("quest_beat_lines"):
            ov["QUEST_BEAT_LINES"] = str(cfg["quest_beat_lines"])
        ov["QUEST_QA_MAX_ROUNDS"] = str(cfg["quest_qa_rounds"])
    return ov


def _generate_one(topic: str, cefr: str, structure: str, num_lines: int,
                  lessons_dir: str | None, max_attempts: int = 3):
    """Generate + validate a single script with retries. Returns (script, attempts)."""
    from pipeline import _validate_script

    quest = (structure == "quest")
    last_err: Exception | None = None
    for attempt in range(max_attempts):
        try:
            if quest:
                from quest.llm_client_quest import generate_quest_script
                script = generate_quest_script(
                    topic, cefr, lessons_dir=lessons_dir, num_lines=num_lines)
            else:
                from llm_client import generate_listening_script
                script = generate_listening_script(
                    topic, cefr, lessons_dir=lessons_dir, num_lines=num_lines,
                    structure=structure)
            valid, msg = _validate_script(script, num_lines, quest=quest)
            if valid:
                return script, attempt + 1
            last_err = RuntimeError(f"校验未通过: {msg}")
        except Exception as e:  # noqa: BLE001 — 记录后重试
            last_err = e
        if attempt < max_attempts - 1:
            time.sleep(3)
    raise last_err or RuntimeError("生成失败")


def generate_batch(params: dict, q, stop_event: threading.Event) -> None:
    """Generate one script per topic, serially. Emits queue events:
    ("progress", msg) / ("script", meta) / ("error_item", {topic, error})
    / ("fatal", msg) / ("done", summary) / None (terminator).
    """
    structure = params.get("structure", "original")
    cefr = params.get("cefr", "A2")
    topics = [str(t).strip() for t in params.get("topics", []) if str(t).strip()]
    provider = params.get("provider", "")
    model = params.get("model", "")
    try:
        num_lines = int(params.get("num_lines") or DEFAULT_LINES.get(structure, 18))
    except (TypeError, ValueError):
        num_lines = DEFAULT_LINES.get(structure, 18)

    summary = {"total": len(topics), "generated": 0, "failed": 0, "stopped": False}
    if not topics:
        q.put(("fatal", "未选择任何主题"))
        q.put(None)
        return

    mode_cfg = load_mode_config(structure)
    lessons_dir = mode_cfg.get("lessons_dir", "") or None
    # quest 单次生成 20+ 次 LLM 调用，减少重试次数避免过长等待
    max_attempts = 2 if structure == "quest" else 3

    try:
        override = _build_llm_override(provider, model, structure)
    except RuntimeError as e:
        q.put(("fatal", str(e)))
        q.put(None)
        return

    # 同模式同主题查重：库中已有该主题脚本时仅提示不阻断（用户可能刻意重生成）
    existing = {m["topic"]: m["id"] for m in list_scripts(structure=structure)}

    # 线程局部覆盖：批量线程用自己的 provider 配置，与运行中的 pipeline 互不干扰
    set_llm_env_override(override)
    try:
        for i, topic in enumerate(topics):
            _set_snapshot(kind="generate", current=topic, done=i,
                          total=len(topics), generated=summary["generated"],
                          failed=summary["failed"])
            if stop_event.is_set():
                summary["stopped"] = True
                q.put(("progress",
                       f"⏹ 已停止（完成 {summary['generated']}/{summary['total']}）"))
                break
            if topic in existing:
                q.put(("progress",
                       f"⚠ 「{topic}」库中已有同名主题脚本（{existing[topic]}），仍继续生成"))
            q.put(("progress",
                   f"[{i + 1}/{len(topics)}] 「{topic}」生成中（{structure}/{cefr}）..."))
            try:
                script, attempts = _generate_one(
                    topic, cefr, structure, num_lines, lessons_dir,
                    max_attempts=max_attempts)
                qa_report = script.pop("_qa", None)
                issues = local_checks(script, structure, num_lines)
                review = {"local_issues": issues}
                if qa_report:
                    review["qa"] = qa_report
                doc = save_new_script(script, {
                    "topic": topic, "cefr": cefr, "structure": structure,
                    "llm_provider": provider, "llm_model": model,
                    "num_lines": num_lines, "review": review,
                })
                extra = "，QA 通过" if qa_report else ""
                q.put(("progress",
                       f"✅ 「{topic}」完成（尝试 {attempts} 次，"
                       f"{len(script.get('dialogue', []))} 行{extra}）"))
                q.put(("script", _doc_meta(doc)))
                summary["generated"] += 1
                existing[topic] = doc["id"]
                _set_snapshot(kind="generate", current=topic, done=i + 1,
                              total=len(topics), generated=summary["generated"],
                              failed=summary["failed"])
            except Exception as e:  # noqa: BLE001
                summary["failed"] += 1
                q.put(("error_item", {"topic": topic,
                                      "error": f"{type(e).__name__}: {e}"}))
                q.put(("progress",
                       f"❌ 「{topic}」失败: {type(e).__name__}: {str(e)[:120]}"))
        q.put(("done", summary))
    finally:
        set_llm_env_override(None)
        q.put(None)


def start_generate_thread(params: dict, q) -> threading.Thread:
    """Start batch generation in a daemon thread (guards re-entry)."""
    if _batch_state["running"]:
        raise RuntimeError("已有批量任务进行中，请等待完成或先停止")
    _batch_state["running"] = True
    _batch_state["stop"].clear()
    _set_snapshot(kind="generate", current="", done=0,
                  total=len(params.get("topics") or []),
                  generated=0, failed=0)

    def _wrap():
        try:
            generate_batch(params, q, _batch_state["stop"])
        finally:
            _batch_state["running"] = False
            _batch_state["snapshot"] = {}

    t = threading.Thread(target=_wrap, daemon=True)
    t.start()
    return t


# ===========================================================================
# Local (free) checks
# ===========================================================================

def local_checks(script: dict, structure: str, num_lines: int) -> list[dict]:
    """Free heuristic checks (no LLM). Returns list of issue dicts."""
    from pipeline import _validate_script

    issues: list[dict] = []
    quest = (structure == "quest")
    valid, msg = _validate_script(script, num_lines, quest=quest)
    if not valid:
        issues.append({"type": "structure", "severity": "high",
                       "line": None, "comment": f"结构校验未通过: {msg}",
                       "suggestion": ""})

    # 简体检测（中文文案必须繁體中文）
    zh_texts = [script.get(k, "") or "" for k in
                ("intro_zh", "welcome_zh", "outro_zh", "practice_intro_zh",
                 "title_zh", "scene_zh")]
    for ln in script.get("dialogue", []) or []:
        zh_texts.append(ln.get("zh", "") or "")
    simp_hits = sorted({c for t in zh_texts for c in _SIMP_ONLY_CHARS if c in t})
    if simp_hits:
        issues.append({"type": "simplified_chinese", "severity": "medium",
                       "line": None,
                       "comment": f"检测到简体字（应为繁體中文）: {''.join(simp_hits[:10])}",
                       "suggestion": "改为对应繁体字"})

    # 行长度（与 QA 门禁同源 max_line_words：线程局部 override → os.environ → 默认 10）
    env_name = ("QUEST_MAX_LINE_WORDS" if structure == "quest"
                else "LISTENING_MAX_LINE_WORDS")
    cap = resolve_max_line_words(env_name)
    for i, ln in enumerate(script.get("dialogue", []) or []):
        words = len((ln.get("text") or "").split())
        if words > cap:
            issues.append({"type": "line_too_long", "severity": "medium",
                           "line": i + 1,
                           "comment": f"第 {i + 1} 行 {words} 词，超过上限 {cap} 词"
                                      f"（运行时 QA 门禁将按此拦截）",
                           "suggestion": "拆分为两行或精简"})

    # 性别 vs 描述一致性
    for key in ("char_a", "char_b", "char_c", "host"):
        gender = (script.get(f"{key}_gender") or "").lower()
        desc = f" {(script.get(f'{key}_description') or '').lower()} "
        if not gender or desc.strip() == "":
            continue
        female_hit = any(w in desc for w in _GENDER_WORDS["female"])
        male_hit = any(w in desc for w in _GENDER_WORDS["male"])
        label = {"char_a": "角色A", "char_b": "角色B", "char_c": "角色C",
                 "host": "主持人"}.get(key, key)
        if gender == "female" and male_hit and not female_hit:
            issues.append({"type": "gender_mismatch", "severity": "high",
                           "line": None,
                           "comment": f"{label} 性别为 female，但描述像是男性",
                           "suggestion": "统一性别或修改描述"})
        elif gender == "male" and female_hit and not male_hit:
            issues.append({"type": "gender_mismatch", "severity": "high",
                           "line": None,
                           "comment": f"{label} 性别为 male，但描述像是女性",
                           "suggestion": "统一性别或修改描述"})
    return issues


# ===========================================================================
# AI review
# ===========================================================================

class _RetryableError(Exception):
    """Internal: transient LLM failure worth retrying (empty/invalid output)."""


def _chat_json_provider(provider_id: str, model: str, structure: str,
                        messages: list[dict], temperature: float = 0.3,
                        max_tokens: int = 2048,
                        resolved: tuple | None = None) -> Any:
    """Non-streaming LLM call with explicit provider → parsed JSON.

    Reads rate-limit config from the structure's mode config. Retries on
    HTTP 429/gateway and network errors (15/30/60s backoff) and on invalid
    JSON output (independent retry budget). `resolved` 可传入预先解析好的
    _resolve_batch_provider 结果（避免重复解析 + 复取 resolved_model）。
    Rate limiting 走 llm_client 的共享限速器（与 pipeline 运行互认）。
    """
    (p_type, base_url, api_key, resolved_model), cfg = (
        resolved or _resolve_batch_provider(provider_id, model, structure))
    if not api_key:
        raise RuntimeError(
            f"未配置所选大模型的 API Key（provider={provider_id or p_type}）— "
            f"请先在「参数配置」页面填写")
    min_interval = float(cfg.get("llm_min_interval") or 3)

    backoffs = [15, 30, 60]
    http_attempt = 0      # HTTP 429/网关/网络错误重试额度
    content_attempt = 0   # 空内容/坏 JSON 重试额度（独立计数，互不挤占）
    while True:
        _enforce_rate_limit(min_interval)
        body = {
            "model": resolved_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if p_type != "openai":
            body["reasoning_effort"] = "low"
        req = urllib.request.Request(
            f"{base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"), method="POST")
        req.add_header("Authorization", f"Bearer {api_key}")
        req.add_header("Content-Type", "application/json")
        req.add_header("User-Agent", "CodelyLLM/1.0")
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                raw = resp.read().decode("utf-8")
            try:
                result = json.loads(raw)
            except json.JSONDecodeError as e:
                # 网关可能返回 HTML 错误页等非 JSON 响应体 — 归入可重试
                raise _RetryableError(
                    f"LLM 返回非 JSON 响应（前 200 字符）: {raw[:200]}") from e
            if not isinstance(result, dict):
                raise _RetryableError(f"LLM 响应不是 JSON 对象: {str(result)[:200]}")
            choices = result.get("choices") or [{}]
            content = (choices[0].get("message") or {}).get("content", "")
            if not content or not content.strip():
                raise _RetryableError("LLM 返回了空内容")
            try:
                return _extract_json(content)
            except json.JSONDecodeError:
                raise _RetryableError(
                    f"LLM 输出不是有效 JSON（前 200 字符）: {content[:200]}")
        except urllib.error.HTTPError as e:
            err = e.read().decode("utf-8", errors="replace")[:300]
            if e.code in (429, 502, 503, 504, 524) and http_attempt < len(backoffs):
                wait = backoffs[http_attempt]
                http_attempt += 1
                print(f"  [ScriptsAI] HTTP {e.code}, 等待 {wait}s 后重试 "
                      f"({http_attempt}/{len(backoffs)})...")
                time.sleep(wait)
                continue
            raise RuntimeError(f"LLM HTTP {e.code}: {err}") from e
        except OSError as e:
            # 网络层瞬断（连接超时/拒绝/DNS/断连）— HTTPError 已在上面单独捕获
            if http_attempt < len(backoffs):
                wait = backoffs[http_attempt]
                http_attempt += 1
                print(f"  [ScriptsAI] 网络错误 {type(e).__name__}，{wait}s 后重试 "
                      f"({http_attempt}/{len(backoffs)})...")
                time.sleep(wait)
                continue
            raise RuntimeError(
                f"LLM 网络错误（重试后仍失败）: {type(e).__name__}: {e}") from e
        except _RetryableError as e:
            if content_attempt < len(backoffs):
                wait = min(backoffs[content_attempt], 10)
                content_attempt += 1
                print(f"  [ScriptsAI] {e}，{wait}s 后重试 "
                      f"({content_attempt}/{len(backoffs)})...")
                time.sleep(wait)
                continue
            raise RuntimeError(f"LLM 调用失败（重试 {content_attempt} 次后）: {e}") from e


def ai_review_script(sid: str, provider_id: str, model: str) -> dict | None:
    """Review one library script via LLM. Persists doc.review. Returns meta."""
    doc = get_script_doc(sid)
    if not doc:
        return None
    script = doc.get("script", {}) or {}
    structure = doc.get("structure", "original")
    cefr = doc.get("cefr", script.get("cefr", "A2"))

    prompt_field = _line_prompt_field(structure)
    lines_ref = []
    for i, ln in enumerate(script.get("dialogue", []) or []):
        entry = (f"{i + 1}. [{ln.get('speaker', '?')}] {ln.get('text', '')}\n"
                 f"   zh: {ln.get('zh', '')}")
        if prompt_field:
            entry += f"\n   prompt: {(ln.get(prompt_field) or '')[:250]}"
        lines_ref.append(entry)
    chars = []
    for key in ("char_a", "char_b", "char_c", "host"):
        if script.get(f"{key}_description"):
            chars.append(f"- {key}: {script[f'{key}_description']} "
                         f"(gender={script.get(f'{key}_gender', '')}, "
                         f"role={script.get(f'{key}_role', '')})")

    prompt = f"""You are a strict ESL content auditor for a YouTube channel serving overseas Chinese learners. Audit the following listening-video script.

Topic: {doc.get('topic', '')}
Structure: {structure}
Target CEFR: {cefr}

Characters:
{chr(10).join(chars) or '  (none)'}

Dialogue:
{chr(10).join(lines_ref)}

YouTube title: {script.get('youtube_title', '')}

CHECK DIMENSIONS:
1. naturalness — 对话是否自然地道（缩略、口语填充词、真实交流模式），有无教科书味
2. cefr — 难度是否匹配 {cefr}（句子长度、词汇）
3. translation — 繁體中文翻译是否准确自然
4. consistency — 角色外观/性别描述在各行视觉 prompt（上方 prompt: 字段，若列出）中是否一致；场景是否一致
5. usefulness — 每行是否有教学价值（可立即套用的表达）
6. metadata — YouTube 标题质量（含繁中、结构完整）

RULES:
- Be conservative: only flag REAL problems. At most 8 issues.
- "line": 行号 (1-based) 或 null
- "comment" / "suggestion": 简体中文
- score 0-100; verdict: "pass" (>=80) / "needs_fix" (60-79) / "fail" (<60)

OUTPUT JSON ONLY:
{{"score": 78, "verdict": "needs_fix", "issues": [{{"dimension": "naturalness", "severity": "high", "line": 3, "comment": "...", "suggestion": "..."}}], "summary_zh": "总评 1-3 句"}}"""

    resolved = _resolve_batch_provider(provider_id, model, structure)
    data = _chat_json_provider(
        provider_id, model, structure,
        [{"role": "system",
          "content": "You are a strict but fair ESL script auditor. Output valid JSON only."},
         {"role": "user", "content": prompt}],
        temperature=0.2, max_tokens=2048, resolved=resolved)

    try:
        score = max(0, min(100, int(data.get("score", 0) or 0)))
    except (TypeError, ValueError):
        score = 0
    verdict = data.get("verdict")
    if verdict not in ("pass", "needs_fix", "fail"):
        verdict = "pass" if score >= 80 else "needs_fix" if score >= 60 else "fail"
    issues = []
    for it in (data.get("issues") or [])[:12]:
        if not isinstance(it, dict):
            continue
        issues.append({
            "dimension": str(it.get("dimension", "other")),
            "severity": it.get("severity") if it.get("severity") in ("high", "medium", "low") else "low",
            "line": it.get("line") if isinstance(it.get("line"), int) else None,
            "comment": str(it.get("comment", "")).strip(),
            "suggestion": str(it.get("suggestion", "")).strip(),
        })

    # 锁内重读最新 doc 再写：LLM 调用期间用户可能已编辑保存，避免整档覆盖
    with _doc_lock:
        fresh = get_script_doc(sid)
        if fresh:
            doc = fresh
        review = doc.get("review") or {}
        review.update({
            "score": score, "verdict": verdict, "issues": issues,
            "summary_zh": str(data.get("summary_zh", "")).strip(),
            "reviewed_at": time.time(),
            "model": resolved[0][3] or provider_id,
            "stale": False,
        })
        doc["review"] = review
        if doc.get("status") != "used":
            doc["status"] = "reviewed"
        _write_doc(doc)
    return _doc_meta(doc)


def review_batch(ids: list[str], provider_id: str, model: str, q,
                 stop_event: threading.Event) -> None:
    """Sequentially review scripts; emits ("progress"/"reviewed"/"error_item"/
    "done") + None terminator (guaranteed via finally)."""
    summary = {"total": len(ids), "reviewed": 0, "failed": 0, "stopped": False}
    try:
        for i, sid in enumerate(ids):
            _set_snapshot(kind="review", current=sid, done=i, total=len(ids),
                          reviewed=summary["reviewed"], failed=summary["failed"])
            if stop_event.is_set():
                summary["stopped"] = True
                break
            q.put(("progress", f"[{i + 1}/{len(ids)}] AI 审查中: {sid}"))
            try:
                meta = ai_review_script(sid, provider_id, model)
                if meta:
                    q.put(("progress",
                           f"✅ 审查完成「{meta['topic']}」: {meta['score']} 分 ({meta['verdict']})"))
                    q.put(("reviewed", meta))
                    summary["reviewed"] += 1
                else:
                    summary["failed"] += 1
                    q.put(("error_item", {"topic": sid, "error": "脚本不存在"}))
            except Exception as e:  # noqa: BLE001
                summary["failed"] += 1
                q.put(("error_item", {"topic": sid,
                                      "error": f"{type(e).__name__}: {e}"}))
        q.put(("done", summary))
    finally:
        # 终止符必达：SSE 端 q.get 阻塞依赖它收尾
        q.put(None)


def start_review_thread(ids: list[str], provider_id: str, model: str, q) -> threading.Thread:
    """Start batch review in a daemon thread (shares the batch busy-guard)."""
    if _batch_state["running"]:
        raise RuntimeError("已有批量任务进行中，请等待完成或先停止")
    _batch_state["running"] = True
    _batch_state["stop"].clear()
    _set_snapshot(kind="review", current="", done=0, total=len(ids),
                  reviewed=0, failed=0)

    def _wrap():
        try:
            review_batch(ids, provider_id, model, q, _batch_state["stop"])
        finally:
            _batch_state["running"] = False
            _batch_state["snapshot"] = {}

    t = threading.Thread(target=_wrap, daemon=True)
    t.start()
    return t


# ===========================================================================
# Selected-issue AI fix (patch-based, line-count preserving)
# ===========================================================================

def _line_prompt_field(structure: str) -> str:
    """该模式行级视觉 prompt 字段（对齐 quality_gate_listening.PROMPT_FIELDS）：
    original→video_prompt, original_static→image_prompt, 其余（cutout/quest）→''。"""
    return {"original": "video_prompt",
            "original_static": "image_prompt"}.get(structure, "")


def _patchable_line_keys(structure: str) -> tuple[str, ...]:
    """按结构允许 AI patch 的行级字段。poses 已废弃（管线零消费方）彻底移除；
    quest 行无 phonetic（_validate_script 对 quest 不查 phonetic）。"""
    keys = ["text", "zh", "speaker"]
    if structure != "quest":
        keys.append("phonetic")
    prompt_field = _line_prompt_field(structure)
    if prompt_field:
        keys.append(prompt_field)
    return tuple(keys)


def _apply_patch(script: dict, patch: dict,
                 structure: str = "original") -> dict:
    """Apply an LLM JSON patch to a deep copy of the script. Returns the copy."""
    import copy
    patched = copy.deepcopy(script)
    dialogue = patched.get("dialogue") or []
    patchable = _patchable_line_keys(structure)
    for p in (patch.get("dialogue") or []):
        if not isinstance(p, dict):
            continue
        idx = p.get("index")
        if not isinstance(idx, int) or not (0 <= idx < len(dialogue)):
            continue
        for k in patchable:
            if k in p and p[k] is not None:
                dialogue[idx][k] = p[k]
    for k, v in (patch.get("fields") or {}).items():
        if isinstance(v, (str, list)) and k not in ("dialogue", "lesson_type"):
            patched[k] = v
    return patched


def fix_script(sid: str, issues: list[dict], provider_id: str, model: str,
               re_review: bool, q, stop_event: threading.Event) -> None:
    """Wrapper: 兜底保证 fatal 事件与 None 终止符必达（SSE 端 q.get 阻塞依赖它收尾）。"""
    try:
        _fix_script_inner(sid, issues, provider_id, model, re_review, q, stop_event)
    except Exception as e:  # noqa: BLE001 — 任何未捕获异常都要转成 fatal 事件
        q.put(("fatal", f"修复失败: {type(e).__name__}: {e}"))
    finally:
        q.put(None)


def _fix_script_inner(sid: str, issues: list[dict], provider_id: str, model: str,
                      re_review: bool, q, stop_event: threading.Event) -> None:
    """Fix SELECTED issues on one script via LLM patch, then optionally re-review.

    Emits ("progress"/"fixed"/"reviewed"/"done"/"fatal") + None terminator.
    Original script is preserved unless the patched version passes validation.
    """
    from pipeline import _validate_script

    doc = get_script_doc(sid)
    if not doc:
        q.put(("fatal", "脚本不存在"))
        q.put(None)
        return
    script = doc.get("script") or {}
    dialogue = script.get("dialogue") or []
    n = len(dialogue)
    structure = doc.get("structure", "original")
    cefr = doc.get("cefr", script.get("cefr", "A2"))
    if not n:
        q.put(("fatal", "脚本无对话内容"))
        q.put(None)
        return

    prompt_field = _line_prompt_field(structure)
    fields_note = "/".join(_patchable_line_keys(structure))
    if prompt_field:
        prompt_rule = (f'- "prompt:" in the dialogue listing shows the current '
                       f'{prompt_field} — use it as context when fixing '
                       f'consistency issues')
        link_rule = (f"- If you change a character description, also patch that "
                     f"character's affected dialogue lines' {prompt_field} "
                     f"to keep the description consistent")
    else:
        prompt_rule = ("- This structure has NO line-level visual prompt fields — "
                       "do NOT patch or invent image_prompt/video_prompt/poses")
        link_rule = ("- If you change a character description, no dialogue-line "
                     "visual prompt updates are needed in this structure")
    lines_ref = "\n".join(
        f"{i}. [{ln.get('speaker', '?')}] {ln.get('text', '')}\n"
        f"   zh: {ln.get('zh', '')}"
        + (f"\n   phonetic: {ln.get('phonetic', '')}" if structure != "quest" else "")
        + (f"\n   prompt: {(ln.get(prompt_field) or '')[:200]}" if prompt_field else "")
        for i, ln in enumerate(dialogue))
    chars = "\n".join(
        f"- {k}: {script.get(k + '_description', '')} "
        f"(gender={script.get(k + '_gender', '')}, role={script.get(k + '_role', '')})"
        for k in ("char_a", "char_b", "char_c", "host")
        if script.get(k + "_description"))
    issues_ref = "\n".join(
        f"- [{it.get('dimension') or it.get('type') or '?'}/"
        f"{it.get('severity', '?')}"
        f"{'/line ' + str(it['line']) if it.get('line') else ''}] "
        f"{it.get('comment', '')} → {it.get('suggestion', '')}"
        for it in issues[:12])

    q.put(("progress", "AI 正在修复选中的问题（补丁式，行数保持不变）..."))
    prompt = f"""You are an expert ESL script editor. Apply ONLY the selected fixes below to a listening-video script. Change nothing else.

Topic: {doc.get('topic', '')}
Structure: {structure}
Target CEFR: {cefr}

SELECTED ISSUES TO FIX:
{issues_ref}

CURRENT SCRIPT:
Characters:
{chars or '  (none)'}

Dialogue (index is 0-based, {n} lines total):
{lines_ref}

YouTube title: {script.get('youtube_title', '')}

OUTPUT a JSON PATCH ONLY (no markdown, no explanation):
{{"dialogue": [{{"index": 0, "text": "corrected text", "zh": "繁體中文", "phonetic": "/ipa/"}}, ...], "fields": {{"youtube_title": "..."}}}}

RULES:
- The dialogue MUST keep exactly {n} lines — never add or remove lines
- Include ONLY dialogue lines that change; within a line include ONLY the fields that change ({fields_note})
{prompt_rule}
- Do NOT change "speaker" unless an issue explicitly requires it
- "fields" may contain top-level script fields (char_a_description, char_b_description, youtube_title, title, title_zh, intro_zh, outro_zh, ...)
{link_rule}
- All Chinese output MUST be Traditional Chinese (繁體中文)
- If an issue cannot be fixed without changing the line count, skip it (it will be handled manually)"""

    try:
        data = _chat_json_provider(
            provider_id, model, structure,
            [{"role": "system",
              "content": "You are a precise ESL script editor. Output valid JSON patches only."},
             {"role": "user", "content": prompt}],
            temperature=0.3, max_tokens=4096)
    except Exception as e:  # noqa: BLE001
        q.put(("fatal", f"修复调用失败: {e}"))
        q.put(None)
        return

    if not isinstance(data, dict):
        q.put(("fatal", "AI 返回的补丁格式无效（非 JSON 对象）"))
        q.put(None)
        return

    patched = _apply_patch(script, data, structure)
    valid, msg = _validate_script(patched, n, quest=(structure == "quest"))
    if not valid:
        q.put(("fatal", f"修复后校验未通过（原稿已保留）: {msg}"))
        q.put(None)
        return

    # 保存：清掉过时的 AI 审查结论，重算本地校验
    doc["script"] = patched
    review = doc.get("review") or {}
    for k in ("score", "verdict", "issues", "summary_zh", "reviewed_at", "model"):
        review.pop(k, None)
    review["local_issues"] = local_checks(patched, structure, n)
    doc["review"] = review
    if doc.get("status") != "used":
        doc["status"] = "draft"
    _write_doc(doc)
    q.put(("progress", "✅ 修复已保存"))
    q.put(("fixed", _doc_meta(doc)))

    if re_review and not stop_event.is_set():
        try:
            meta = ai_review_script(sid, provider_id, model)
            q.put(("reviewed", meta or _doc_meta(doc)))
        except Exception as e:  # noqa: BLE001
            q.put(("progress", f"⚠ 自动复审失败: {str(e)[:120]}"))
    q.put(("done", {"fixed": 1, "re_reviewed": bool(re_review)}))
    q.put(None)


def start_fix_thread(sid: str, issues: list[dict], provider_id: str, model: str,
                     re_review: bool, q) -> threading.Thread:
    """Start a single-script fix in a daemon thread (shares busy-guard)."""
    if _batch_state["running"]:
        raise RuntimeError("已有批量任务进行中，请等待完成或先停止")
    _batch_state["running"] = True
    _batch_state["stop"].clear()
    _set_snapshot(kind="fix", current=sid, done=0, total=1, reviewed=0, failed=0)

    def _wrap():
        try:
            fix_script(sid, issues, provider_id, model, re_review, q,
                       _batch_state["stop"])
        finally:
            _batch_state["running"] = False
            _batch_state["snapshot"] = {}

    t = threading.Thread(target=_wrap, daemon=True)
    t.start()
    return t
