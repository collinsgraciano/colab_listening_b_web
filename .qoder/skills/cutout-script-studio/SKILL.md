---
name: cutout-script-studio
description: Generate ready-to-run English listening-practice dialogue scripts for the colab_listening_b_web pipeline in "original_cutout" (Original Cutout 4章+人物抠图) video structure mode. Use when asked to create/batch-generate ESL cutout scripts, topic packs, or one-click runnable script JSON files for this project. Every script must be authored fresh by AI per topic — never by filling Python templates.
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
   the only allowed scripts are the bundled validator and installer.
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
4. **Validate before done.** After writing files, run
   `python .qoder/skills/cutout-script-studio/scripts/validate_scripts.py cutout_script_studio/scripts`
   and fix every reported issue by rewriting the offending content (still AI-authored).
5. **Install last.**
   `python .qoder/skills/cutout-script-studio/scripts/install_to_library.py`
   copies validated files into `configs/script_library/` so they appear in the
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
4. Run the validator over all written files; fix failures; re-run until clean.
5. Install into the library; report counts.

## Quality bar (what makes a script good)

- Every line teaches something immediately usable (a phrase, pattern, or strategy).
- CEFR A2: short sentences, high-frequency vocabulary, present/past tense, minimal idioms.
- Coherent single conversation, not disconnected Q&A; roles drive who asks what.
- 繁體中文 translations natural (not machine-literal); descriptions and gender
  fields mutually consistent; thumbnails fields concrete and scene-specific.

## Resources

- `references/schema.md` — authoritative field spec for doc + script + dialogue.
- `scripts/validate_scripts.py` — deterministic checks (schema, 10-word cap, IPA,
  繁中, gender consistency, title lengths, cross-file duplicates). Read-only.
- `scripts/install_to_library.py` — copies validated docs into
  `configs/script_library/`. Refuses files that fail validation.
