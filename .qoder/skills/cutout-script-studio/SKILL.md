---
name: cutout-script-studio
description: Generate ready-to-run English listening-practice dialogue scripts for the colab_listening_b_web pipeline in "original_cutout" (Original Cutout 4章+人物抠图) video structure mode, with a mandatory multi-round review loop (deterministic lint + AI rubric review) for grammar-perfect, engaging dialogue. Use when asked to create/batch-generate ESL cutout scripts, topic packs, review/fix existing cutout scripts, or one-click runnable script JSON files for this project. Every script must be authored fresh by AI per topic — never by filling Python templates.
---

# Cutout Script Studio

## Overview

Produce `original_cutout` listening-practice scripts as JSON library docs that the
web dashboard can run directly (脚本库 → 一键生成视频, zero LLM Step 0). Each script =
one topic: an N-line two-speaker American-English dialogue (N = doc `num_lines`,
default 18; current batch uses 60) + IPA + 繁體中文 + host
narration (welcome/hook/outro/practice_intro) + full YouTube metadata.

## Hard rules

1. **AI-authored, always.** Write every script's content yourself from the topic.
   NEVER write or run a Python/JS generator that fills a template with variables;
   the only allowed scripts are the bundled validator, deep-lint and installer.
2. **Schema contract.** Exact field list, formats and limits in
   `references/schema.md`. Non-negotiables: dialogue length == doc `num_lines`
   (batch size chosen before generating; every line still ≤10 words),
   every English
   sentence ≤10 words, 繁體中文 everywhere Chinese, IPA in /slashes/,
   `structure: "original_cutout"` in both doc and script, doc `id` ==
   `script_cut_<slug>` == filename stem.
3. **No repeats.** Topics, storylines, character looks, and `title_quote` must not
   duplicate each other or existing library topics
   (`configs/script_library/`, check `topic` fields first).
4. **Validate before done.** After writing files, run BOTH deterministic gates and
   fix every reported issue by rewriting the offending content (still AI-authored):
   `python .qoder/skills/cutout-script-studio/scripts/validate_scripts.py cutout_script_studio/scripts`
   and `python .qoder/skills/cutout-script-studio/scripts/deep_lint.py cutout_script_studio/scripts`.
5. **Review loop, every script, repeatedly.** No script is done after one pass.
   Run the AI review loop from `references/review_rubric.md`: independent reviewer
   pass(es) scoring 6 dimensions, fix by rewriting, re-run both gates, until 2
   consecutive fully-clean rounds (max 4 rounds, leftover warnings justified).
6. **Install last.**
   `python .qoder/skills/cutout-script-studio/scripts/install_to_library.py`
   enforces both deterministic gates, then copies reviewed files into
   `configs/script_library/` so they appear in the
   dashboard 脚本库 filtered by mode `original_cutout`.

## Workflow

1. Read `references/schema.md` fully (field-by-field spec + quality bar).
2. Get the topic list — from `cutout_script_studio/topics_100.json` or the user's
   own topics. Work in batches (≈10 topics per batch keeps quality).
3. For each topic write the complete story first in your head: two characters with
   concrete roles for the scene, beginning → small problem/development → natural
   resolution, overseas-Chinese daily-life practicality. Then write the JSON file to
   `cutout_script_studio/scripts/<slug>.json`.
   - Characters: free per script (age/gender/outfit vary with the scene).
   - Dialogue must sound like overheard real American speech: contractions,
     fillers (um/well/so), back-channeling, polite hedging.
   - `title_quote`: pick verbatim from your own dialogue (<10 words, the hookiest).
   - YouTube titles: follow the two patterns in schema.md (繁中【】style and
     English quote-hook style) plus the `*_ref` reference-channel style.
4. Deterministic gates: run `validate_scripts.py` and `deep_lint.py` over all
   written files; fix failures by rewriting content; re-run until 0 errors
   (deep_lint warnings fixed or justified).
5. AI review loop per `references/review_rubric.md` — every script, repeatedly:
   - Batch mode: writer agents finish ALL files first; then spawn SEPARATE
     reviewer agents (fresh eyes, ≈5 files each) that read each full script and
     score the 6 rubric dimensions, emitting one issue line per finding.
   - A writer (or the reviewer itself for one-offs) fixes every error/warning
     by rewriting the offending lines, then re-runs both gates.
   - Loop until each script has 2 consecutive fully-clean rounds; hard stop at
     4 rounds with written justifications for any surviving warning.
6. Install into the library; report per-script review summary (rubric format).

## Quality bar (what makes a script good)

- Every line teaches something immediately usable (a phrase, pattern, or strategy).
- CEFR A2: short sentences, high-frequency vocabulary, present/past tense, minimal idioms.
- Coherent single conversation, not disconnected Q&A; roles drive who asks what.
- 繁體中文 translations natural (not machine-literal); descriptions and gender
  fields mutually consistent; thumbnails fields concrete and scene-specific.

## Resources

- `references/schema.md` — authoritative field spec for doc + script + dialogue.
- `references/review_rubric.md` — 6-dimension review loop: severities, loop
  policy (≥2 rounds, 2 consecutive clean, max 4), issue + report formats.
- `scripts/validate_scripts.py` — deterministic checks (schema, 10-word cap, IPA,
  繁中, gender consistency, title lengths, cross-file duplicates). Read-only.
- `scripts/deep_lint.py` — deterministic craft lint (verbatim title_quote,
  duplicated lines/phrases, reused openings, filler density, tiny-line count,
  vocabulary monotony, speaker balance, IPA-vs-text coverage). Read-only.
- `scripts/install_to_library.py` — runs BOTH gates then copies validated docs into
  `configs/script_library/`. Refuses files that fail validation.
