"""SEO 词池：把高频搜索短语并入 YouTube tags + 为 LLM prompt 提供关键词提示。

数据文件 pipeline/seo_keywords.json：
    {"core": ["english speaking practice", ...],          # 每支影片都并入
     "by_topic": [{"match": ["work", "職場", ...],        # match 子串命中主题文本时并入
                   "tags": ["workplace english conversation", ...]}, ...]}

两个入口：
    merge_tags(tags, match_text)  — thumbnail_gen.save_youtube_metadata 并入 tags 字段
    keyword_hint(match_text)      — llm_client / quest 生成 prompt 注入的搜索短语提示

词池文件缺失/损坏 → 空池，merge_tags 原样返回、keyword_hint 返回空串（零行为变化）。
YouTube tags 字段硬限 500 字符（上传时按逗号拼接计长），merge_tags 按 tag 粒度丢弃
超长项，不截半个词。
"""
import json
from pathlib import Path

_POOL_FILE = Path(__file__).resolve().parent / "seo_keywords.json"
MAX_TAGS_CHARS = 500


def load_pool() -> dict:
    """读词池并清洗；缺失/损坏回退空池。"""
    try:
        data = json.loads(_POOL_FILE.read_text(encoding="utf-8"))
        core = [str(t).strip() for t in data.get("core", []) if str(t).strip()]
        groups = []
        for g in data.get("by_topic", []):
            match = [str(m).strip().lower() for m in g.get("match", []) if str(m).strip()]
            tags = [str(t).strip() for t in g.get("tags", []) if str(t).strip()]
            if match and tags:
                groups.append({"match": match, "tags": tags})
        return {"core": core, "by_topic": groups}
    except (OSError, ValueError, TypeError):
        return {"core": [], "by_topic": []}


def _match_tags(match_text: str, pool: dict) -> list[str]:
    """按子串命中返回 by_topic 词组（保持词池文件顺序）。"""
    text = (match_text or "").lower()
    if not text:
        return []
    out: list[str] = []
    for g in pool["by_topic"]:
        if any(m in text for m in g["match"]):
            out.extend(g["tags"])
    return out


def merge_tags(tags: list[str], match_text: str = "") -> list[str]:
    """合并词池：脚本 LLM tags 优先 → by_topic 命中 → core。

    大小写去重；总长（按逗号拼接计）截到 MAX_TAGS_CHARS，保序丢弃放不下的 tag。
    """
    pool = load_pool()
    if not pool["core"] and not pool["by_topic"]:
        return list(tags or [])
    result: list[str] = []
    seen: set[str] = set()
    total = 0
    for t in list(tags or []) + _match_tags(match_text, pool) + pool["core"]:
        t = str(t).strip()
        key = t.lower()
        if not t or key in seen:
            continue
        extra = len(t) + (1 if result else 0)
        if total + extra > MAX_TAGS_CHARS:
            continue
        seen.add(key)
        result.append(t)
        total += extra
    return result


def keyword_hint(match_text: str = "") -> str:
    """供生成 prompt 注入的搜索短语串（主题命中 8 个 + core 4 个，≤12 个）。

    空池返回空串 —— prompt 保持与无词池时逐字节一致。
    """
    pool = load_pool()
    if not pool["core"] and not pool["by_topic"]:
        return ""
    picked: list[str] = []
    seen: set[str] = set()
    for t in _match_tags(match_text, pool)[:8] + pool["core"][:4]:
        key = t.lower()
        if key not in seen:
            seen.add(key)
            picked.append(t)
    return ", ".join(picked)
