"""Character library listing/meta helpers (dedup across pages & API)."""
import json

from .paths import LIBRARY_DIR


def list_library_chars(include_pose_count: bool = False) -> list[dict]:
    """List library characters (meta + image_url), newest first.

    include_pose_count=True 时附加 pose_count（人物素材库管理页徽标）。
    """
    chars = []
    if not LIBRARY_DIR.exists():
        return chars
    for d in sorted(LIBRARY_DIR.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        meta_path = d / "meta.json"
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        thumb = d / "thumb.png"
        meta["image_url"] = f"/api/character_library/{d.name}/image" if thumb.exists() else ""
        if include_pose_count:
            if meta.get("structure") == "quest":
                meta["pose_count"] = sum(1 for p in d.glob("pose_char_a_*.png")
                                         if "_c" not in p.stem)
            else:
                meta["pose_count"] = 1 if (d / "char_scene.png").exists() else 0
        chars.append(meta)
    return chars


def write_library_meta(lib_id: str, meta: dict) -> None:
    """Persist meta.json of a library character."""
    meta_path = LIBRARY_DIR / lib_id / "meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
