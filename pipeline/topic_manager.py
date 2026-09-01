"""Topic manager for listening video pipeline — random selection + anti-duplicate.

Reads topics.json (categorized topic pool), randomly picks an unused topic,
and records it in used_topics.json to prevent repeats.

topics.json format:
    {"日常Life": ["Making Breakfast", "Doing Laundry", ...], "旅行": [...]}

used_topics.json format:
    {"Making Breakfast": {"used_at": "2026-08-10 18:00:00"}, ...}
"""
import json
import os
import random
import sys
import tempfile
from datetime import datetime
from pathlib import Path


def load_topics(topics_file: str) -> dict:
    """Load topics.json — returns {category: [topic, ...]} (empty dict on missing/broken)."""
    p = Path(topics_file)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def load_used_topics(used_file: str) -> dict:
    """Load used_topics.json — returns {topic: {used_at: ...}} (empty dict on missing/broken)."""
    p = Path(used_file)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def get_all_topics(topics: dict) -> list[str]:
    """Flatten all topics from all categories into a single list."""
    all_topics = []
    for cat_topics in topics.values():
        all_topics.extend(cat_topics)
    return all_topics


def _mark_used(used_file: str, topic: str) -> None:
    """Record a topic as used in used_topics.json (idempotent, atomic write)."""
    used = load_used_topics(used_file)
    used[topic] = {"used_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    p = Path(used_file)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(used, f, ensure_ascii=False, indent=2)
        os.replace(tmp, str(p))
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def mark_topic_used(used_file: str, topic: str) -> None:
    """Public API: mark a topic as used — call AFTER its script is safely saved,
    so failed runs and resume runs don't burn through the topic pool."""
    _mark_used(used_file, topic)


def pick_random_topic(topics_file: str, used_file: str, mark: bool = True) -> str | None:
    """Pick a random topic that hasn't been used yet.

    If all topics are used, resets used_topics.json and picks from the full pool.
    Set mark=False to defer marking (caller marks via mark_topic_used after success).
    Returns the topic string, or None if topics_file is empty/missing.
    """
    topics = load_topics(topics_file)
    if not topics:
        return None

    all_topics = get_all_topics(topics)
    if not all_topics:
        return None

    used = load_used_topics(used_file)
    available = [t for t in all_topics if t not in used]

    if not available:
        # All topics used — raise instead of silently wiping history
        raise ValueError(
            f"所有 {len(all_topics)} 个主题均已使用完毕。"
            "请在主题管理页面重置已用主题或添加新主题。")

    chosen = random.choice(available)
    print(f"  [Topic] Randomly selected: '{chosen}' (from {len(available)} available)")

    if mark:
        _mark_used(used_file, chosen)

    return chosen


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Pick a random topic")
    parser.add_argument("--topics-file", default="topics.json", help="Path to topics.json")
    parser.add_argument("--used-file", default="used_topics.json", help="Path to used_topics.json")
    args = parser.parse_args()

    topic = pick_random_topic(args.topics_file, args.used_file)
    if topic:
        print(f"Selected: {topic}")
    else:
        print("No topics found.")
