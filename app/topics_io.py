"""Topic library file I/O shared by topics and scripts endpoints."""
import json
import os
import tempfile
from pathlib import Path


def load_topics_data(config: dict) -> dict:
    """Load topics.json → {category: [topics]} (empty dict on missing/broken)."""
    topics_file = config.get("topics_file", "")
    if topics_file and Path(topics_file).exists():
        try:
            return json.loads(Path(topics_file).read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def load_used_topic_names(config: dict) -> list[str]:
    """Load used_topics.json → list of topic names."""
    used_file = config.get("used_topics_file", "")
    if not used_file:
        output_dir = config.get("output_dir", "./output")
        used_file = str(Path(output_dir) / "used_topics.json")
    if Path(used_file).exists():
        try:
            return list(json.loads(Path(used_file).read_text(encoding="utf-8")).keys())
        except (json.JSONDecodeError, OSError):
            return []
    return []


def save_topics_data(topics_file: str, data: dict) -> None:
    """原子写入 topics.json（tempfile + os.replace，防并发写丢失）。"""
    p = Path(topics_file)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, str(p))
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def save_used_topics(used_file: str, data: dict) -> None:
    """原子写入 used_topics.json（tempfile + os.replace，防并发写丢失）。"""
    p = Path(used_file)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, str(p))
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
