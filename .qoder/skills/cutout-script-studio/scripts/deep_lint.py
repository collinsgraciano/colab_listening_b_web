# -*- coding: utf-8 -*-
"""Deep quality lint for cutout docs — the deterministic half of the review loop.

Usage: python deep_lint.py [dir_with_json_files | single-file.json]
Default dir: <repo_root>/cutout_script_studio/scripts
Checks dialogue CRAFT (repetition, rhythm, hooks, IPA coverage), not schema
(see validate_scripts.py for that). Exit 0 iff no ERRORS (warnings allowed,
but the SKILL.md review loop requires warnings be fixed or justified).
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[4]

FILLERS = ("um", "uh", "well", "so ", "oh ", "wow", "really", "huh", "yeah",
           "hey", "okay", "hmm", "like,", "I mean", "you know", "no way",
           "serious", "right?", "got it", "sure")

STOPWORDS = set("""
a an the and or but if then than that this these those i you he she it we they
me him her us them my your his its our their am is are was were be been being
do does did done to of in on at for with from by about as not no yes just
""".split())


def _norm(s: str) -> str:
    s = s.lower().replace("\u2019", "'")
    s = re.sub(r"[^a-z0-9' ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _wcount(s: str) -> int:
    return len(_norm(s).split())


def _ngrams(norm_text: str, n: int):
    w = norm_text.split()
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def check_file(path: Path):
    errs, warns = [], []
    doc = json.loads(path.read_text(encoding="utf-8"))
    s = doc.get("script") or {}
    dl = s.get("dialogue") or []
    if len(dl) < 10:
        return ["dialogue too short to lint"], []

    norm_lines = [_norm(l.get("text", "")) for l in dl]

    # 1. title_quote must occur verbatim (case/punct-insensitive) in dialogue
    tq = _norm(str(s.get("title_quote", "")))
    if tq and not any(tq in t for t in norm_lines if t):
        errs.append("title_quote 不是任何一句台词的子串（须逐字摘自对话）")

    # 2. exact duplicated lines
    dup = [t for t, c in Counter(norm_lines).items() if c > 1 and t]
    if dup:
        errs.append(f"重复台词行: {[d[:30] for d in dup[:3]]}")

    # 3. shared 6-word phrases across different lines (patchwork feel)
    seen6 = {}
    for idx, t in enumerate(norm_lines):
        for g in _ngrams(t, 6):
            if g in seen6:
                errs.append(f"6词长短语在第 {seen6[g]+1} 与第 {idx+1} 行重复")
                break
            seen6[g] = idx

    # 4. line openings (first 2 words) over-reused
    opens = Counter(" ".join(t.split()[:2]) for t in norm_lines if t)
    for o, c in opens.items():
        if c >= 5:
            errs.append(f"行首 '{o}' 重复 {c} 次（≥5，节奏单调）")
        elif c >= 3:
            warns.append(f"行首 '{o}' 重复 {c} 次")

    # 5. conversational texture: fillers / back-channeling present
    joined = " ".join(norm_lines)
    fill_n = sum(joined.count(f) for f in FILLERS)
    if fill_n < 3:
        warns.append(f"填充语/附和语过少（{fill_n} 处）— 对白偏教科书腔")

    # 6. tiny lines pile-up
    tiny = sum(1 for t in norm_lines if 0 < _wcount(t) <= 2)
    if tiny > 6:
        warns.append(f"≤2 词短句 {tiny} 句（>6）— 信息密度不足")

    # 7. content-word monotony
    cw = Counter(w for t in norm_lines for w in t.split()
                 if len(w) >= 4 and w not in STOPWORDS)
    for w, c in cw.most_common(3):
        if c >= max(6, int(len(dl) * 0.28)):
            warns.append(f"实词 '{w}' 出现在 {c}/{len(dl)} 行 — 词汇单一")

    # 8. speaker balance
    wa = sum(_wcount(l["text"]) for i, l in enumerate(dl) if i % 2 == 0)
    wb = sum(_wcount(l["text"]) for i, l in enumerate(dl) if i % 2 == 1)
    if wb > 0 and (wa / wb > 2 or wb / wa > 2):
        warns.append(f"两人词数失衡（a={wa} b={wb}）— 像独角戏")

    # 9. IPA present on every line and roughly tracks the text
    for i, l in enumerate(dl):
        ph = str(l.get("phonetic", ""))
        if not (ph.startswith("/") and ph.endswith("/") and len(ph) > 4):
            errs.append(f"第 {i+1} 行缺 IPA")
            continue
        pw = len(ph.strip("/").replace("ˈ", " ").replace("ˌ", " ").split())
        tw = _wcount(str(l.get("text", "")))
        if tw and not (tw * 0.6 - 1 <= pw <= tw * 1.6 + 2):
            warns.append(f"第 {i+1} 行 IPA 词数 {pw} 与文本词数 {tw} 差距过大")

    # 10. narration sentences echo dialogue openings (host ≠ guest voice)
    outro = _norm(str(s.get("outro", "")))
    if outro and any(outro == t for t in norm_lines):
        warns.append("outro 与某句台词完全相同")
    return errs, warns


def main() -> int:
    arg = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "cutout_script_studio" / "scripts"
    if arg.is_file():
        files = [arg]
    else:
        files = sorted(arg.glob("script_cut_*.json"))
    n_err = n_warn = 0
    for f in files:
        try:
            errs, warns = check_file(f)
        except Exception as e:  # noqa: BLE001
            errs, warns = [f"lint crash: {type(e).__name__}: {e}"], []
        for e in errs:
            print(f"ERROR {f.name}: {e}")
        for w in warns:
            print(f"WARN  {f.name}: {w}")
        n_err += len(errs)
        n_warn += len(warns)
    print(f"deep-lint: checked {len(files)} files, "
          f"{n_err} errors, {n_warn} warnings")
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main())
