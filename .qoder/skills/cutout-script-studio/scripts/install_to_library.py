# -*- coding: utf-8 -*-
"""Install validated cutout docs into configs/script_library/ (file copy only).

Usage: python install_to_library.py [--force]
Runs the validator first; refuses on any error unless --force.
"""
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SRC = ROOT / "cutout_script_studio" / "scripts"
DST = ROOT / "configs" / "script_library"


def main() -> int:
    force = "--force" in sys.argv
    if not force:
        r = subprocess.run([sys.executable, str(HERE / "validate_scripts.py")],
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace")
        tail = "\n".join((r.stdout or "").splitlines()[-8:])
        if r.returncode != 0:
            print("VALIDATION FAILED — not installed. Fix errors first:")
            print(tail)
            return 1
    DST.mkdir(parents=True, exist_ok=True)
    copied = skipped = 0
    for f in sorted(SRC.glob("*.json")):
        if not f.stem.startswith("script_cut_"):
            print(f"skip (bad id): {f.name}")
            continue
        target = DST / f.name
        if target.exists() and not force:
            skipped += 1
            continue
        shutil.copyfile(f, target)
        copied += 1
    print(f"installed {copied} script(s) into configs/script_library/ "
          f"({skipped} already present, skipped)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
