"""Topic library file I/O shared by topics and scripts endpoints."""
import json
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
