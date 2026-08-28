"""Checkpoint save/load helpers for pipeline resume support."""
import json
import time
from pathlib import Path


def save_checkpoint(work_dir: Path, step: str, **extra):
    """Save progress to checkpoint.json for resume support."""
    cp_path = work_dir / "checkpoint.json"
    cp = {}
    if cp_path.exists():
        try:
            cp = json.loads(cp_path.read_text(encoding="utf-8"))
        except Exception:
            cp = {}
    if step not in cp.get("completed_steps", []):
        cp.setdefault("completed_steps", []).append(step)
    cp.update(extra)
    cp["_run_dir"] = str(work_dir)
    cp["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    cp_path.write_text(json.dumps(cp, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  [Checkpoint] Saved: {step}")


def load_checkpoint(work_dir: Path) -> dict:
    """Load checkpoint from work_dir, or scan subdirectories for incomplete runs.

    A run is considered complete when step6_4k is done (so a crash during 4K
    upscale can still be resumed). Among multiple incomplete runs, the most
    recently updated one wins.
    """
    candidates = []

    def _check(cp_path: Path):
        try:
            cp = json.loads(cp_path.read_text(encoding="utf-8"))
            if "step6_4k" not in cp.get("completed_steps", []):
                candidates.append((cp_path.stat().st_mtime, cp_path.parent, cp))
            else:
                cp_path.unlink()
        except Exception:
            pass

    _check(work_dir / "checkpoint.json")
    if not candidates and work_dir.exists():
        # 回收站文件夹（Web 移入的已删除运行）不参与续传扫描
        recycle_names = ("_recycle_bin", ".recycle_bin")
        for sub in work_dir.iterdir():
            if sub.is_dir() and sub.name not in recycle_names:
                _check(sub / "checkpoint.json")

    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        _, run_dir, cp = candidates[0]
        print(f"  [Resume] Found incomplete run in: {run_dir.name}")
        return cp
    return {}


def step_done(checkpoint: dict, step: str) -> bool:
    return step in checkpoint.get("completed_steps", [])
