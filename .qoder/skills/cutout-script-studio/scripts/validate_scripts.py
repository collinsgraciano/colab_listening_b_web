# -*- coding: utf-8 -*-
"""Validate cutout library docs (schema contract + quality gates).

Usage: python validate_scripts.py [dir_with_json_files]
Default dir: <repo_root>/cutout_script_studio/scripts
Exit 0 if no ERRORS (warnings allowed). Prints one line per issue.
"""
import json
import re
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[4]

SIMPLIFIED_ONLY = set(
    "听说读发买卖开关门问间东车贝见长张网风飞电话号亿层让认语词试请谁来对办还这"
    "个为从会被过儿女严举义乐乡书经济观现样员损规则亲课业归块总换热当")
# 补充常用简体独有字（均为繁简字形不同的字；宁误报勿漏报，误报由改写规避）
SIMPLIFIED_ONLY |= set("运营轮转换谢调询预验简单迟达际难吗尽")

GENDER_WORDS = {
    "male": (" man ", " boy ", " he ", " his ", " male ", " gentleman "),
    "female": (" woman ", " girl ", " she ", " her ", " female ", " lady "),
}

STR_FIELDS = [
    "title", "title_quote", "cefr", "title_zh", "scene_zh", "story_hook",
    "intro_zh", "welcome_en", "welcome_zh", "outro", "outro_zh",
    "practice_intro_en", "practice_intro_zh",
    "char_a_description", "char_b_description",
    "char_a_gender", "char_b_gender", "char_a_role", "char_b_role",
    "youtube_title", "youtube_title_en", "youtube_description",
    "youtube_description_en", "youtube_title_ref", "youtube_description_ref",
    "scene", "thumbnail_expression", "thumbnail_action", "thumbnail_subtitle",
]

CAP = 10  # max words per English sentence (LISTENING_MAX_LINE_WORDS)


def _words(s):
    return len(s.split())


def _sentences(s):
    return [p.strip() for p in re.split(r"[.!?]+", s) if p.strip()]


def _has_simplified(s):
    return [c for c in SIMPLIFIED_ONLY if c in (s or "")]


def check_doc(path: Path, seen: dict) -> tuple[list, list]:
    errs, warns = [], []
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        return [f"JSON parse error: {e}"], []
    if not isinstance(doc, dict):
        return ["doc is not a JSON object"], []

    sid = str(doc.get("id", ""))
    if not sid.startswith("script_cut_"):
        errs.append("id must start with 'script_cut_'")
    if path.stem != sid:
        errs.append(f"filename {path.name} != id {sid}")
    slug = sid[len("script_cut_"):] if sid.startswith("script_cut_") else sid
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{2,40}", slug):
        errs.append(f"bad slug '{slug}' (lowercase/digits/hyphens, 3-41)")
    topic = str(doc.get("topic", "")).strip()
    if not topic:
        errs.append("empty topic")
    key = topic.lower()
    if key in seen["topics"]:
        errs.append(f"duplicate topic (also in {seen['topics'][key]})")
    seen["topics"][key] = path.name
    if slug in seen["slugs"]:
        errs.append(f"duplicate slug (also in {seen['slugs'][slug]})")
    seen["slugs"][slug] = path.name
    if doc.get("cefr") != "A2":
        errs.append("cefr must be 'A2'")
    if doc.get("structure") != "original_cutout":
        errs.append("doc structure must be 'original_cutout'")
    if doc.get("num_lines") != 18:
        errs.append("doc num_lines must be 18")

    s = doc.get("script")
    if not isinstance(s, dict):
        return errs + ["missing script object"], warns
    if s.get("lesson_type") != "listening":
        errs.append("script.lesson_type must be 'listening'")
    if s.get("structure") != "original_cutout":
        errs.append("script.structure must be 'original_cutout'")
    if s.get("cefr") != "A2":
        errs.append("script.cefr must be 'A2'")
    for f in STR_FIELDS:
        if not str(s.get(f, "")).strip():
            errs.append(f"missing/empty script.{f}")

    zh_fields = [f for f in STR_FIELDS if f.endswith("_zh") or f in
                 ("youtube_title", "youtube_description", "youtube_title_ref",
                  "youtube_description_ref", "thumbnail_subtitle")]
    for f in zh_fields:
        hits = _has_simplified(str(s.get(f, "")))
        if hits:
            errs.append(f"简化字 in script.{f}: {''.join(sorted(set(hits))[:12])}")

    # narration sentence caps
    for f in ("story_hook", "welcome_en", "outro", "practice_intro_en"):
        for sent in _sentences(str(s.get(f, ""))):
            if _words(sent) > CAP:
                errs.append(f"script.{f} sentence >{CAP} words: '{sent[:50]}'")
    tq = str(s.get("title_quote", "")).strip().strip('"\u201c\u201d「」').lower()
    if tq and _words(tq) >= 10:
        errs.append("title_quote must be under 10 words")

    # title lengths
    yt = str(s.get("youtube_title", ""))
    if not (40 <= len(yt) <= 95):
        errs.append(f"youtube_title length {len(yt)} outside 40-95")
    if "【" not in yt or "｜" not in yt:
        errs.append("youtube_title needs 【】 tag and ｜ separators")
    ytr = str(s.get("youtube_title_ref", ""))
    if not (40 <= len(ytr) <= 95):
        errs.append(f"youtube_title_ref length {len(ytr)} outside 40-95")
    if "沉浸式英文動畫" not in ytr:
        errs.append("youtube_title_ref must use 【…沉浸式英文動畫】 format")
    yte = str(s.get("youtube_title_en", ""))
    if not (25 <= len(yte) <= 99):
        errs.append(f"youtube_title_en length {len(yte)} outside 25-99")
    for f in ("youtube_description", "youtube_description_en",
              "youtube_description_ref"):
        if len(str(s.get(f, ""))) > 3000:
            errs.append(f"script.{f} over 3000 chars")
    tags = s.get("youtube_tags")
    if not isinstance(tags, list) or not (15 <= len(tags) <= 20):
        errs.append("youtube_tags must be a 15-20 item array")
    icons = s.get("thumbnail_icons")
    if not isinstance(icons, list) or not (4 <= len(icons) <= 5):
        errs.append("thumbnail_icons must be 4-5 items")
    else:
        for i, ic in enumerate(icons):
            if not (isinstance(ic, dict) and str(ic.get("en", "")).strip()
                    and str(ic.get("zh", "")).strip()):
                errs.append(f"thumbnail_icons[{i}] needs non-empty en+zh")
            elif _has_simplified(str(ic.get("zh", ""))):
                errs.append(f"简化字 in thumbnail_icons[{i}].zh")
    tz = str(s.get("title_zh", ""))
    if len(tz) > 6:
        errs.append(f"title_zh too long ({len(tz)}>6)")
    if "·" not in str(s.get("scene_zh", "")):
        warns.append("scene_zh should look like '場景 · 動作'")

    # gender vs description
    for key in ("char_a", "char_b"):
        g = str(s.get(f"{key}_gender", "")).lower()
        d = f" {str(s.get(f'{key}_description', '')).lower()} "
        if g not in ("male", "female"):
            errs.append(f"{key}_gender must be male/female")
            continue
        fem = any(w in d for w in GENDER_WORDS["female"])
        mal = any(w in d for w in GENDER_WORDS["male"])
        if g == "female" and mal and not fem:
            errs.append(f"{key}: gender female but description reads male")
        if g == "male" and fem and not mal:
            errs.append(f"{key}: gender male but description reads female")

    # dialogue
    dl = s.get("dialogue")
    if not isinstance(dl, list) or len(dl) != 18:
        errs.append(f"dialogue must have exactly 18 lines (got {len(dl) if isinstance(dl, list) else 'N/A'})")
        return errs, warns
    fp_parts = []
    for i, ln in enumerate(dl):
        if not isinstance(ln, dict):
            errs.append(f"line {i} not object")
            continue
        extra = set(ln.keys()) - {"speaker", "text", "phonetic", "zh"}
        if extra:
            errs.append(f"line {i} has forbidden keys {sorted(extra)} (cutout: text/phonetic/zh/speaker only)")
        want = "char_a" if i % 2 == 0 else "char_b"
        if ln.get("speaker") != want:
            errs.append(f"line {i} speaker must be {want}")
        text = str(ln.get("text", "")).strip()
        if not text:
            errs.append(f"line {i} empty text")
        elif _words(text) > CAP:
            errs.append(f"line {i} text {_words(text)} words > {CAP}: '{text[:48]}'")
        ph = str(ln.get("phonetic", "")).strip()
        if not (len(ph) > 2 and ph.startswith("/") and ph.endswith("/")):
            errs.append(f"line {i} phonetic must be /IPA/")
        zh = str(ln.get("zh", "")).strip()
        if not zh:
            errs.append(f"line {i} empty zh")
        elif _has_simplified(zh):
            errs.append(f"line {i} zh has 简化字: {''.join(sorted(set(_has_simplified(zh))))[:12]}")
        fp_parts.append(text.lower())
    if tq and not any(tq in p for p in fp_parts):
        errs.append("title_quote not found verbatim in any dialogue line")
    fp = "|".join(fp_parts)
    if fp in seen["fps"]:
        errs.append(f"duplicate dialogue (same as {seen['fps'][fp]})")
    seen["fps"][fp] = path.name
    return errs, warns


def main() -> int:
    d = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "cutout_script_studio" / "scripts"
    single_file = None
    if d.is_file():
        single_file, d = d, d.parent
    elif not d.is_dir():
        print(f"ERROR: dir not found: {d}")
        return 2
    # existing library topics also count for dedupe
    seen = {"topics": {}, "slugs": {}, "fps": {}}
    lib = ROOT / "configs" / "script_library"
    if lib.is_dir():
        for f in lib.glob("script_*.json"):
            try:
                t = str(json.loads(f.read_text(encoding="utf-8")).get("topic", "")).strip()
            except Exception:  # noqa: BLE001
                continue
            if t:
                seen["topics"][t.lower()] = f"(library {f.name})"
    files = sorted(d.glob("*.json"))
    if single_file is not None:
        files = [single_file]
    n_err = n_warn = 0
    for f in files:
        errs, warns = check_doc(f, seen)
        for e in errs:
            print(f"ERROR {f.name}: {e}")
        for w in warns:
            print(f"warn  {f.name}: {w}")
        n_err += len(errs)
        n_warn += len(warns)
    print(f"checked {len(files)} files, {n_err} errors, {n_warn} warnings")
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main())
