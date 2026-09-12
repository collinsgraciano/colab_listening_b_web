"""片头库索引 IO（configs/intro_library.json）— 路由与 pipeline_service 共用。

索引条目：{id, name, source(local/ai), duration, created, subtitle, bgm,
announce}；视频文件固定为 configs/intro_videos/{id}/intro.mp4（统一规格
1280x720 / 25fps / aac 44100 立体声 / 10s，见 pipeline/sleep/intro_video.py）。
"""
import json
import re
import time
from pathlib import Path

from .paths import INTRO_LIBRARY_PATH, INTRO_VIDEOS_DIR

INTRO_ID_RE = re.compile(r"^intro_[A-Za-z0-9_]+$")


def load_library() -> list[dict]:
    if not INTRO_LIBRARY_PATH.exists():
        return []
    try:
        data = json.loads(INTRO_LIBRARY_PATH.read_text(encoding="utf-8"))
        return data.get("intros", []) if isinstance(data, dict) else []
    except (json.JSONDecodeError, OSError):
        return []


def save_library(intros: list[dict]) -> None:
    INTRO_LIBRARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    INTRO_LIBRARY_PATH.write_text(
        json.dumps({"updated": time.time(), "intros": intros},
                   ensure_ascii=False, indent=2), encoding="utf-8")


def intro_file(intro_id: str) -> Path:
    return INTRO_VIDEOS_DIR / intro_id / "intro.mp4"


def resolve_video_path(intro_sel: str) -> str:
    """sleep_intro_video 配置值 → mp4 绝对路径。支持 intro_id 或绝对路径。

    无效（空 / 库中不存在 / 文件缺失）返回空串，调用方回退默认片头。
    """
    sel = str(intro_sel or "").strip()
    if not sel:
        return ""
    direct = Path(sel)
    if direct.is_absolute() and direct.suffix.lower() == ".mp4" and direct.exists():
        return str(direct)
    if INTRO_ID_RE.match(sel):
        f = intro_file(sel)
        if f.exists():
            return str(f)
    return ""
