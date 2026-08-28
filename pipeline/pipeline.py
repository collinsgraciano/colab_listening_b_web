#!/usr/bin/env python3
"""Standalone listening video generation pipeline — Colab version.

Usage:
    python pipeline.py --topic "At the Pharmacy" --cefr A2 --output ./output --mcp-tokens TOK1,TOK2

Steps (each an independently resumable function):
  step0  LLM script generation (SenseNova: deepseek-v4-flash or glm-5.2)
  step1  MCP init (multi-token rotation)
  step2  Concurrent: image generation + TTS audio; clip_0 launched in parallel
  step3  Group consecutive dialogue lines, one Seedance2 clip per group
         (skipped entirely in original_static/quest/original_cutout mode)
  step4  Timeline + SRT building
  step4.5 YouTube metadata + thumbnail
  step5  Final video composition (FFmpeg + Pillow)
  step6  Optional 4K upscale

main() is a thin orchestrator; all per-step logic lives in _stepN_* functions
so each can be read, tested, or re-run in isolation.

Domain modules:
  checkpoint.py      — save/load resume state
  clip_gen.py        — video clip creation, polling, retry, task builders
  tts_pipeline.py    — batch TTS generation (narration + dialogue + vocab/quiz)
  image_gen.py       — character/scene/dialogue image generation + resume check
  timeline_enrich.py — fill audio_dur/duration on timeline segments
  group_audio.py     — concat per-line TTS into per-group audio files
  media_utils.py     — shared FFmpeg/Pillow helpers (concat, subtitle burn, loudnorm)
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
import threading
from pathlib import Path

# Add scripts directory to path
SCRIPTS_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPTS_DIR))

from mcp_client import initialize, call_tool, parse_task_id, poll_task, download_file
from llm_client import generate_listening_script
from tts_engine import TTSEngine
from timeline import build_listening_timeline, build_srt_from_timeline, \
    rewrite_title_card_as_host_segments
from video_compose import compose_listening
from grouping_b import build_dialogue_groups
from topic_manager import pick_random_topic, mark_topic_used
from media_utils import get_duration as _get_audio_duration, safe_filename as _safe_dirname
from checkpoint import save_checkpoint as _save_checkpoint, load_checkpoint as _load_checkpoint, step_done as _step_done
from clip_gen import (
    file_ok as _file_ok,
    build_scene_clip_task as _build_scene_clip_task,
    build_group_clip_tasks as _build_group_clip_tasks,
    generate_video_clips as _generate_video_clips,
)
from tts_pipeline import generate_tts as _generate_tts
from image_gen import (
    check_step2_resume as _check_step2_resume,
    generate_images as _generate_images,
    generate_dialogue_images as _generate_dialogue_images,
    generate_quest_atlases as _generate_quest_atlases,
    generate_scene_atlas as _generate_scene_atlas,
    reupload_for_cdn as _reupload_for_cdn,
)
from timeline_enrich import enrich_timeline as _enrich_timeline
from group_audio import build_group_info as _build_group_info
from style_manager import resolve_style_prompt as _resolve_style_prompt


def _get_style_prompt(args) -> str:
    """当前画面风格 prompt 片段（env 注入优先，CLI --visual-style 兜底）。

    pipeline_service 直接调用 _stepN_* 时不经过 main()，因此通过
    VISUAL_STYLE_PROMPT 环境变量传递风格（与 CHARACTER_OVERRIDES 同模式）。
    """
    return os.environ.get("VISUAL_STYLE_PROMPT") or _resolve_style_prompt(
        getattr(args, "visual_style", None))


# ---------------------------------------------------------------------------
# Script validation + generation
# ---------------------------------------------------------------------------

def _validate_script(script: dict, num_lines: int,
                     quest: bool = False) -> tuple[bool, str]:
    """Validate a generated script. Returns (is_valid, error_message)."""
    dialogue = script.get("dialogue", [])
    if len(dialogue) < num_lines:
        return False, f"Dialogue count {len(dialogue)} < required {num_lines}"
    for i, line in enumerate(dialogue):
        text = line.get("text", "").strip()
        if not text:
            return False, f"Dialogue line {i} has empty 'text'"
        if not line.get("zh", "").strip():
            return False, f"Dialogue line {i} has empty 'zh'"
        if not quest and not line.get("phonetic", "").strip():
            return False, f"Dialogue line {i} has empty 'phonetic'"
        if not line.get("speaker", ""):
            return False, f"Dialogue line {i} has empty 'speaker'"
    if quest:
        if not script.get("listening_question_en", "").strip():
            return False, "Quest script has empty 'listening_question_en'"
        if not script.get("hook_intro_en", "").strip():
            return False, "Quest script has empty 'hook_intro_en'"
        if not script.get("welcome_en", "").strip():
            return False, "Quest script has empty 'welcome_en'"
        if not script.get("char_c_description", "").strip():
            return False, "Quest script has empty 'char_c_description'"
        if not script.get("host_description", "").strip():
            return False, "Quest script has empty 'host_description'"
        phases = [line.get("phase", "") for line in dialogue]
        for ph in ("buildup", "core", "reveal", "review"):
            if ph not in phases:
                return False, f"Quest dialogue missing phase '{ph}'"
        order = {"buildup": 0, "core": 1, "reveal": 2, "review": 3}
        seq = [order.get(p, -1) for p in phases]
        if any(b < a for a, b in zip(seq, seq[1:])):
            return False, "Quest dialogue phases are out of order (must be buildup -> core -> reveal -> review)"
        for i, line in enumerate(dialogue):
            speaker = line.get("speaker", "")
            phase = line.get("phase", "")
            if phase in ("buildup", "reveal", "review") and speaker not in ("char_a", "char_b"):
                return False, f"Dialogue line {i} ({phase}) speaker must be char_a/char_b, got '{speaker}'"
            if phase == "core" and speaker not in ("char_a", "char_b", "char_c"):
                return False, f"Dialogue line {i} (core) speaker must be char_a/char_b/char_c, got '{speaker}'"
        # Ensure char_c appears at least once in core
        core_speakers = [line.get("speaker", "") for line in dialogue if line.get("phase") == "core"]
        if "char_c" not in core_speakers:
            return False, "Quest core phase must include at least one char_c (staff) line"
    return True, ""


def _generate_script_with_retry(topic, cefr, lessons_dir, num_lines,
                                quest=False, structure="original",
                                max_attempts=5) -> dict:
    """Generate and validate script, retrying on failure."""
    for attempt in range(max_attempts):
        try:
            print(f"  [Script] Attempt {attempt+1}/{max_attempts}...")
            if quest:
                from quest.llm_client_quest import generate_quest_script
                script = generate_quest_script(topic, cefr, lessons_dir=lessons_dir,
                                               num_lines=num_lines)
            else:
                script = generate_listening_script(topic, cefr, lessons_dir=lessons_dir,
                                                   num_lines=num_lines,
                                                   structure=structure)
            valid, msg = _validate_script(script, num_lines, quest=quest)
            if valid:
                print(f"  [Script] Valid: {len(script['dialogue'])} lines")
                return script
            print(f"  [Script] Invalid: {msg}")
        except Exception as e:
            print(f"  [Script] Error: {e}")
        if attempt < max_attempts - 1:
            time.sleep(3)
    raise RuntimeError(f"Script generation failed after {max_attempts} attempts")


# ---------------------------------------------------------------------------
# CLI + orchestration helpers
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate English listening practice video")
    parser.add_argument("--topic", default=None, help="Topic (e.g. 'At the Pharmacy'). If not specified, picks randomly from topics.json")
    parser.add_argument("--cefr", default="A2", choices=["A1", "A2", "B1", "B2", "C1", "C2"], help="CEFR level (default A2)")
    parser.add_argument("--output", default="./output", help="Output directory")
    parser.add_argument("--clip-duration", type=int, default=15, help="Video clip duration in seconds")
    parser.add_argument("--image-concurrency", type=int, default=4, help="Max concurrent image generation tasks (1-4, default 4)")
    parser.add_argument("--clip-concurrency", type=int, default=4, help="Max concurrent video clip tasks (1-5, default 4)")
    parser.add_argument("--practice-duration", type=float, default=3.0, help="Silence duration in Ch3")
    parser.add_argument("--ch3-en-repeats", type=int, default=3,
                        help="Ch3 practice: English repetitions per line (0-10, 0=skip EN)")
    parser.add_argument("--ch3-zh-repeats", type=int, default=1,
                        help="Ch3 practice: Chinese repetition count per line (0-10, 0=skip ZH)")
    parser.add_argument("--ch3-zh-always", action=argparse.BooleanOptionalAction, default=True,
                        help="Ch3 practice: always show Chinese text on English frames (default: on)")
    parser.add_argument("--ch3-practice-intro-show", action=argparse.BooleanOptionalAction, default=True,
                        help="Ch3 practice intro text card before practice section (default: on; off = skip the whole segment)")
    parser.add_argument("--pad", type=float, default=None, help="Audio pad between segments (default 0.4; quest mode 5.0 — long thinking pauses for beginners)")
    parser.add_argument("--render-fps", type=int, default=8, help="Quest stop-motion render framerate (default 8; lower=faster but choppier)")
    parser.add_argument("--workers", type=int, default=1, help="Quest render threads (1=single, 2+=multi, 0=auto=cpu_count)")
    parser.add_argument("--lessons-dir", default=None, help="Lessons dir for anti-duplicate check")
    parser.add_argument("--topics-file", default=str(Path(__file__).parent / "topics.json"), help="Path to topics.json")
    parser.add_argument("--used-topics-file", default=None, help="Path to used_topics.json (default: <output>/used_topics.json — persists on Drive across Colab sessions)")
    parser.add_argument("--num-lines", type=int, default=None, help="Number of dialogue lines (default: 18; quest mode: 48)")
    parser.add_argument("--mcp-tokens", default=None, help="TJGenerators MCP OAuth tokens, comma-separated for multi-token rotation")
    parser.add_argument("--mcp-token", default=None, help="(Deprecated) Single MCP token. Use --mcp-tokens instead.")
    parser.add_argument("--image-provider", default="mcp", choices=["mcp", "sensenova"],
                        help="Image generation provider: 'mcp' (default, TJGenerators credits) or 'sensenova' (U1.5 Lite, API billing with SENSENOVA_API_KEY)")
    parser.add_argument("--api-key", default=None, help="SenseNova API key (or set SENSENOVA_API_KEY env var)")
    parser.add_argument("--model", default=None,
                        help="LLM model name. SenseNova: 'deepseek-v4-flash' (default) or 'glm-5.2'. OpenAI-compatible: 'grok-4.6' (default), 'gemini-3.1-pro-preview', 'claude-sonnet-5', etc.")
    parser.add_argument("--llm-provider", default="sensenova", choices=["sensenova", "openai"],
                        help="LLM provider: 'sensenova' (default) or 'openai' (OpenAI-compatible endpoint)")
    parser.add_argument("--openai-base-url", default=None, help="OpenAI-compatible API base URL (default: https://x666.me/v1)")
    parser.add_argument("--openai-api-key", default=None, help="OpenAI-compatible API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--openai-model", default=None, help="OpenAI-compatible model name (default: grok-4.6). Available: grok-4.6, grok-4.5, gemini-3.1-pro-preview, gemini-3.7-flash, claude-sonnet-5, gemini-2.5-pro-1m")
    parser.add_argument("--llm-retries", type=int, default=10, help="Max retries per LLM round (default 10). Set higher for unreliable endpoints.")
    parser.add_argument("--structure", default="original", choices=["original", "original_static", "quest", "original_cutout"],
                        help="Video structure: 'original' (4-chapter, video clips), 'original_static' (4-chapter, static images, no video clips), 'quest' (task-hook listening), 'original_cutout' (original 4-chapter + quest-style character cutout animation)")
    parser.add_argument("--host-character", default="", choices=["", "char_a", "char_b"],
                        help="Original Cutout only: bind the host appearance/voice to a dialogue character for intro/outro segments (''= generate a separate host)")
    parser.add_argument("--visual-style", default="pixar3d",
                        help="Visual art style id from style_manager.py (default pixar3d = 3D cartoon Pixar-like). Affects all image/video/thumbnail prompts + LLM script prompts")
    parser.add_argument("--animation", default="landing", choices=["none", "landing", "stop_motion"],
                        help="Dialogue animation: 'none' (static), 'landing' (landing transform), 'stop_motion' (multi-pose + optical flow). Default: landing")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint in output dir")
    parser.add_argument("--no-4k", dest="no_4k", action="store_true", help="Skip the final 4K upscaling step")
    parser.add_argument("--no-zh-subtitle", dest="no_zh_subtitle", action="store_true", help="Hide Chinese subtitles (default: show ZH subtitles)")
    parser.add_argument("--subtitle-font-size", type=int, default=60, help="English subtitle font size in pixels (default 60). ZH subtitle is auto-scaled to 85%% of EN size.")
    parser.add_argument("--subtitle-style", default="", help="Subtitle style id from subtitle_style_manager (web 字幕样式页). Empty = legacy behavior driven by --subtitle-font-size.")
    parser.add_argument("--tts-rate", default=None, help="Override dialogue English TTS rate (e.g. '-15%%', '0%%'). Default: mode-dependent (quest '0%%', others '-15%%')")
    parser.add_argument("--tts-engine", default="kokoro", choices=["kokoro", "qwen", "moss"],
                        help="TTS engine: 'kokoro' (default, local) or 'qwen' (Qwen3-TTS local GPU) or 'moss' (MOSS-TTS-Nano local CPU)")
    parser.add_argument("--qwen-model-path", default=r"H:\models\Qwen3-TTS-12Hz-0.6B-CustomVoice",
                        help="Qwen3-TTS CustomVoice model path (preset speakers)")
    parser.add_argument("--qwen-base-model-path", default=r"H:\models\Qwen3-TTS-12Hz-1.7B-Base",
                        help="Qwen3-TTS Base model path (for voice clone)")
    parser.add_argument("--qwen-voicedesign-model-path", default=r"H:\models\Qwen3-TTS-12Hz-1.7B-VoiceDesign",
                        help="Qwen3-TTS VoiceDesign model path (for designed voices, e.g. English female)")
    parser.add_argument("--qwen-device", default="cuda:0",
                        help="Device for Qwen3-TTS (e.g. cuda:0, cpu)")
    parser.add_argument("--moss-model-path", default=r"H:\models\MOSS-TTS-Nano-Model",
                        help="MOSS-TTS-Nano model checkpoint path")
    parser.add_argument("--moss-tokenizer-path", default=r"H:\models\MOSS-Audio-Tokenizer-Nano",
                        help="MOSS Audio Tokenizer path")
    parser.add_argument("--moss-device", default="cpu",
                        help="Device for MOSS-TTS (e.g. cpu, cuda:0, default: cpu)")
    parser.add_argument("--moss-repo-dir", default=r"H:\models\MOSS-TTS-Nano",
                        help="MOSS-TTS-Nano repo dir (containing moss_tts_nano_runtime.py)")
    parser.add_argument("--moss-tts-temperature", type=float, default=0.8,
                        help="MOSS-TTS audio sampling temperature, lower = more stable (default 0.8)")
    parser.add_argument("--moss-tts-retry", type=int, default=3,
                        help="MOSS-TTS per-sentence retry attempts with re-seed (default 3)")
    parser.add_argument("--upscale-timeout", type=int, default=3600, help="Timeout in seconds for 4K upscale (default 3600)")
    return parser.parse_args()


def _resolve_topic(args, checkpoint: dict) -> str:
    """Resolve topic (priority): checkpoint topic > --topic > random pick."""
    if checkpoint.get("topic"):
        topic = checkpoint["topic"]
        print(f"  [Topic] Resuming topic from checkpoint: '{topic}'")
        return topic
    if args.topic:
        print(f"  [Topic] Using specified topic: '{args.topic}'")
        return args.topic
    print("  [Topic] No topic specified, picking randomly from topics.json...")
    return None


def _resolve_run_dir(parent_dir: Path, checkpoint: dict) -> Path:
    """On resume: use the checkpoint's recorded run dir (or newest subfolder
    containing a checkpoint). On fresh start: parent_dir (temporary)."""
    if checkpoint and checkpoint.get("_run_dir") and Path(checkpoint["_run_dir"]).exists():
        return Path(checkpoint["_run_dir"])
    if checkpoint:
        cands = [(p.stat().st_mtime, p) for p in parent_dir.iterdir()
                 if p.is_dir() and (p / "checkpoint.json").exists()]
        if cands:
            cands.sort(reverse=True)
            return cands[0][1]
    return parent_dir


# ---------------------------------------------------------------------------
# Step functions
# ---------------------------------------------------------------------------

def _step0_script(args, checkpoint: dict, topic: str, parent_dir: Path,
                  used_topics_file: str) -> tuple[dict, Path, dict]:
    """Step 0: generate (or resume) the script and create the run directory."""
    print("=" * 60)
    _llm_provider = os.environ.get("LLM_PROVIDER", "sensenova")
    if _llm_provider == "openai":
        _llm_model = os.environ.get("OPENAI_MODEL", "grok-4.6")
        print(f"Step 0: Generating script via LLM (OpenAI-compatible: {_llm_model})...")
    else:
        _llm_model = os.environ.get("SENSENOVA_MODEL", "deepseek-v4-flash")
        print(f"Step 0: Generating script via LLM (SenseNova {_llm_model})...")

    work_dir = _resolve_run_dir(parent_dir, checkpoint)

    def _dirs(base: Path) -> dict:
        return {
            "images": base / "images",
            "clips": base / "clips",
            "audio": base / "audio",
            "subtitles": base / "subtitles",
            "videos": base / "videos",
        }

    dirs = _dirs(work_dir)
    script_path = work_dir / "script.json"
    cp_structure = checkpoint.get("structure")
    cp_animation = checkpoint.get("animation", "")
    if not cp_animation:
        cp_animation = args.animation
    structure_match = (cp_structure == args.structure)
    # 结构一致的 resume：其它产物相关参数若与 checkpoint 不一致，提示但不阻断
    # （旧行为是静默沿用旧产物，参数改动容易被忽略）
    if (_step_done(checkpoint, "step0_script") and script_path.exists()
            and structure_match):
        print("  [Resume] Loading existing script...")
        for param, cp_key in (("animation", "animation"),
                              ("visual_style", "visual_style"),
                              ("host_character", "host_character")):
            cp_val = checkpoint.get(cp_key)
            cur_val = getattr(args, param, "")
            if cp_val is not None and str(cp_val) != str(cur_val):
                print(f"  [Resume] WARNING: {param} changed "
                      f"({cp_val or '(empty)'} → {cur_val or '(empty)'}) — "
                      f"existing artifacts were built with the old value.")
        script = json.loads(script_path.read_text(encoding="utf-8"))
        mark_topic_used(used_topics_file, topic)
    else:
        script = _generate_script_with_retry(
            topic, args.cefr, args.lessons_dir, args.num_lines,
            quest=(args.structure == "quest"),
            structure=(args.structure if args.structure != "quest" else "original"))
        yt_title = script.get("youtube_title", script.get("title", topic))
        safe_title = _safe_dirname(yt_title, topic)
        work_dir = parent_dir / safe_title
        work_dir.mkdir(parents=True, exist_ok=True)
        dirs = _dirs(work_dir)
        script_path = work_dir / "script.json"
        for d in dirs.values():
            d.mkdir(parents=True, exist_ok=True)
        qa_report = script.pop("_qa", None)
        script["structure"] = args.structure
        script_path.write_text(json.dumps(script, ensure_ascii=False, indent=2), encoding="utf-8")
        if qa_report:
            qa_path = work_dir / "qa_report.json"
            qa_path.write_text(json.dumps(qa_report, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"  QA report saved: {qa_path}")
        _save_checkpoint(work_dir, "step0_script", topic=topic, cefr=args.cefr,
                         structure=args.structure, animation=args.animation,
                         visual_style=getattr(args, "visual_style", ""),
                         host_character=getattr(args, "host_character", ""))
        mark_topic_used(used_topics_file, topic)
    print(f"  Script saved: {script_path}")
    print(f"  Title: {script.get('title', '')}")
    print(f"  Dialogue lines: {len(script.get('dialogue', []))}")
    if args.structure == "original_cutout" and not (script.get("welcome_en") or "").strip():
        # 主持人开场 (commit 302fdb2) 之前的旧脚本没有 welcome_en/welcome_zh 字段
        print("  [Resume] WARNING: script has no 'welcome_en' — written by an older "
              "schema; the host welcome segment will be skipped. A fresh run "
              "(new script) is recommended.")
    return script, work_dir, dirs


def _step1_mcp(args):
    """Step 1: initialize the TJGenerators MCP session."""
    print("\n" + "=" * 60)
    print("Step 1: Initializing TJGenerators MCP...")
    raw_tokens = args.mcp_tokens or args.mcp_token or ""
    tokens = [t.strip() for t in raw_tokens.split(",") if t.strip()]
    if not tokens:
        initialize()
    else:
        initialize(tokens=tokens)
    print("  MCP connected.")


def _step2_images_tts(args, checkpoint: dict, script: dict, work_dir: Path, dirs: dict,
                      stop_check=None) -> dict:
    """Step 2: concurrent image generation + TTS audio, then launch clip_0."""
    print("\n" + "=" * 60)
    print("Step 2: Concurrent generation — images + TTS audio...")

    dialogue = script.get("dialogue", [])
    n = len(dialogue)
    is_original_static = (args.structure == "original_static")
    is_quest = (args.structure == "quest")
    is_original_cutout = (args.structure == "original_cutout")
    # ch3_zh_repeats=0 时时间轴无 listen_zh 段，中文音频无需生成
    include_zh = (not is_quest) and getattr(args, "ch3_zh_repeats", 1) > 0
    img_dir, audio_dir, clips_dir = dirs["images"], dirs["audio"], dirs["clips"]

    char_a_desc = script.get("char_a_description", "friendly young man")
    char_b_desc = script.get("char_b_description", "friendly young woman")
    scene = script.get("scene") or args.topic

    style_prompt = _get_style_prompt(args)
    if not os.environ.get("VISUAL_STYLE_PROMPT"):
        os.environ["VISUAL_STYLE_PROMPT"] = style_prompt
    print(f"  [Style] Using style_prompt: {style_prompt[:80]}...")

    if is_original_cutout:
        # 主持人形象绑定写入脚本，供 TTS 音色联动读取（tts_pipeline）
        script["host_character"] = getattr(args, "host_character", "") or ""

    image_prompts = []
    if not is_quest and not is_original_cutout:
        image_prompts.append(
            (f"Character design sheet, {char_a_desc} on the left, {char_b_desc} on the right, plain white background, full body, front view, {style_prompt}, no text, no background, 16:9", "char_scene.png"))
        image_prompts.append(
            (f"Scene background, {scene}, wide shot, showing all key elements of the scene, {style_prompt}, no characters, no text, 16:9", "scene.png"))
    elif is_original_cutout:
        image_prompts.append(
            (f"Scene background, {scene}, wide shot, showing all key elements of the scene, {style_prompt}, no characters, no text, 16:9", "scene.png"))
    if is_quest:
        char_c_desc = script.get("char_c_description", "friendly staff member")
        # Host background (TV studio)
        host_bg_prompt = script.get("host_bg_prompt", "a bright modern TV studio set with a large screen behind, warm lighting")
        image_prompts.append((f"{host_bg_prompt}, {style_prompt}, no people, 16:9", "host_bg.png"))
        # Scene backgrounds generated as 2×2 atlas in _generate_scene_atlas below
    elif is_original_cutout:
        # 主持人开场/结尾的演播室背景（quest 同款，listening 脚本无 host_bg_prompt 字段）
        image_prompts.append((f"a bright modern TV studio set with a large screen behind, warm lighting, "
                              f"{style_prompt}, no people, no text, 16:9", "host_bg.png"))

    # --- Resume check ---
    resume_result = _check_step2_resume(checkpoint, script, dirs, n, is_quest,
                                        include_zh=include_zh)
    tts_thread = None
    if resume_result is not None:
        tts_results, image_urls = resume_result
    else:
        tts_results = {}

        def _tts_worker():
            try:
                _generate_tts(script, dialogue, audio_dir, tts_results,
                              quest=is_quest,
                              host_narration=is_original_cutout,
                              tts_rate=args.tts_rate,
                              tts_engine=args.tts_engine,
                              stop_check=stop_check,
                              include_zh=include_zh)
            except Exception as e:
                import traceback
                traceback.print_exc()
                tts_results["fatal_error"] = f"{type(e).__name__}: {e}"

        tts_thread = threading.Thread(target=_tts_worker, daemon=True)
        tts_thread.start()
        if not include_zh:
            print("  [TTS] ch3_zh_repeats=0 → 跳过中文音频生成")
        print("  [TTS] Started TTS generation in background thread.")

        image_urls = _generate_images(image_prompts, img_dir, tts_thread,
                                      max_workers=args.image_concurrency,
                                      image_size="landscape_16_9")

        if is_original_static:
            char_scene_cdn = image_urls.get("char_scene.png", "")
            _generate_dialogue_images(
                dialogue, img_dir, char_a_desc, char_b_desc, scene,
                is_quest, char_scene_cdn, "", tts_thread,
                max_workers=args.image_concurrency,
                style_prompt=style_prompt)
        elif is_quest:
            _generate_quest_atlases(script, img_dir, tts_thread,
                                    max_workers=args.image_concurrency,
                                    style_prompt=style_prompt)
            _scene_images = script.get("scene_images", [])
            _generate_scene_atlas(_scene_images, scene, img_dir, tts_thread,
                                   style_prompt=style_prompt)
        elif is_original_cutout:
            # original_cutout: per-character pose atlas (char_a + char_b, 8 poses each)
            # 主持人未绑定角色时额外生成独立主持人图集
            _cutout_keys = ["char_a", "char_b"]
            if not getattr(args, "host_character", ""):
                _cutout_keys.append("host")
            _generate_quest_atlases(script, img_dir, tts_thread,
                                    max_workers=args.image_concurrency,
                                    style_prompt=style_prompt,
                                    char_keys=_cutout_keys)

        if is_quest or is_original_cutout:
            # 全新生成路径：姿势图集只落本地。此处把 char_a 参考图补传 CDN 写入
            # image_urls，供 Step 4.5 缩略图做角色参考（与 --resume 路径行为一致）。
            _ref_path = img_dir / "pose_char_a_0.png"
            if _ref_path.exists():
                _ref_url = _reupload_for_cdn(str(_ref_path), "pose_char_a_0.png")
                if _ref_url:
                    image_urls["pose_char_a_0.png"] = _ref_url

        print("  [Image] All images done. Waiting for TTS...")

    # --- Launch clip_0 in parallel ---
    scene_url = image_urls.get("scene.png", "")
    char_scene_url = image_urls.get("char_scene.png", "")
    clips_dir.mkdir(parents=True, exist_ok=True)
    scene_clip_task = None
    scene_clip_thread = None

    if is_original_static or is_quest or is_original_cutout:
        clip_paths = []
        print(f"  [{'Static' if is_original_static else 'Quest' if is_quest else 'Cutout'}] Skipping clip_0 generation (no video clips)")
    else:
        scene_clip_task = _build_scene_clip_task(scene, scene_url, style_prompt=style_prompt)
        clip0_path = str(clips_dir / "clip_0.mp4")
        clip_paths = [clip0_path if _file_ok(clip0_path, 500000) else None]
        if clip_paths[0] is None:
            scene_clip_thread = threading.Thread(
                target=_generate_video_clips,
                args=([scene_clip_task], clips_dir, clip_paths),
                kwargs={"stop_check": stop_check, "max_concurrency": 1},
                daemon=True,
            )
            scene_clip_thread.start()
            print("  [Video] Scene clip (clip_0) generation started in parallel with TTS.")
        else:
            print("  [Resume] clip_0 already exists, reusing.")

    if tts_thread is not None:
        # 使用超时 join，让 stop_check 有机会生效
        while tts_thread.is_alive():
            tts_thread.join(timeout=0.5)
            if stop_check and stop_check():
                print("  [TTS] Stop requested during TTS join, breaking...", flush=True)
                break

    _tts_err = tts_results.get("fatal_error")
    if _tts_err:
        if _tts_err == "stopped":
            print("  [TTS] Generation stopped by user.")
        else:
            print("  [TTS] FATAL: TTS generation thread crashed:")
            print(f"    {_tts_err}")
            raise RuntimeError(f"TTS generation failed: {_tts_err}. Fix the issue and re-run with --resume.")
    if stop_check and stop_check():
        return {
            "scene": scene,
            "image_urls": image_urls,
            "scene_url": scene_url,
            "char_scene_url": char_scene_url,
            "tts_results": tts_results,
            "scene_clip_task": scene_clip_task,
            "scene_clip_thread": scene_clip_thread,
            "clip_paths": clip_paths,
        }
    _got_en = len(tts_results.get("normal_paths", []))
    if _got_en < n:
        raise RuntimeError(
            f"TTS incomplete: got {_got_en}/{n} English dialogue audio files. "
            f"Re-run with --resume to continue.")

    _save_checkpoint(work_dir, "step2_images_tts")

    return {
        "scene": scene,
        "image_urls": image_urls,
        "scene_url": scene_url,
        "char_scene_url": char_scene_url,
        "tts_results": tts_results,
        "scene_clip_task": scene_clip_task,
        "scene_clip_thread": scene_clip_thread,
        "clip_paths": clip_paths,
    }


def _step3_clips(args, checkpoint: dict, work_dir: Path, dirs: dict, script: dict,
                 ctx: dict, stop_check=None) -> tuple[list[str], list[dict], dict]:
    """Step 3: generate group video clips (skipped entirely in image/quest mode)."""
    print("\n" + "=" * 60)
    dialogue = script.get("dialogue", [])
    tts_results = ctx["tts_results"]
    normal_paths = tts_results.get("normal_paths", [])
    zh_paths = tts_results.get("zh_paths", [])
    dialogue_durations = tts_results.get("dialogue_durations", [])
    audio_dir, clips_dir = dirs["audio"], dirs["clips"]

    if args.structure in ("original_static", "quest", "original_cutout"):
        print(f"Step 3: Skipped ({args.structure} mode — no video generation)")
        _save_checkpoint(work_dir, "step3_video")
        print(f"  TTS: {len(normal_paths)} EN + {sum(1 for p in zh_paths if p)} ZH")
        return [], [], {}

    print(f"Step 3: Generating video clips (Seedance2, up to {args.clip_duration}s per group)...")

    if ctx["scene_clip_thread"] is not None:
        ctx["scene_clip_thread"].join()

    groups = build_dialogue_groups(dialogue, dialogue_durations, args.clip_duration)
    n_groups = len(groups)
    print(f"  [Group] {len(dialogue)} dialogue lines -> {n_groups} video groups:")
    for gi, g in enumerate(groups):
        print(f"    Group {gi}: lines={g['lines']} speakers={g['speakers']} total_audio={g['total_audio']:.1f}s")

    group_tasks = _build_group_clip_tasks(ctx["scene"], ctx["char_scene_url"],
                                          groups, dialogue, args.pad,
                                          style_prompt=_get_style_prompt(args))
    n_total_clips = 1 + len(group_tasks)

    clip_paths = list(ctx["clip_paths"])
    for gi in range(len(group_tasks)):
        p = str(clips_dir / f"clip_{gi+1}.mp4")
        clip_paths.append(p if _file_ok(p, 500000) else None)
    reused = sum(1 for p in clip_paths if p is not None)
    if reused:
        print(f"  [Resume] Reusing {reused}/{n_total_clips} existing clips.")

    if all(p is not None for p in clip_paths):
        print("  [Resume] Step 3 already done — all clips present.")
    else:
        video_thread = threading.Thread(
            target=_generate_video_clips,
            args=(group_tasks, clips_dir, clip_paths, 1),
            kwargs={"stop_check": stop_check, "max_concurrency": args.clip_concurrency},
            daemon=True,
        )
        video_thread.start()
        print("  Waiting for group video clips to complete...")
        video_thread.join()
        print("  >> Video clips done.")
        if not (stop_check and stop_check()):
            _save_checkpoint(work_dir, "step3_video")

    ok_count = sum(1 for p in clip_paths if p is not None)
    print(f"  Clips: {ok_count}/{n_total_clips}, TTS: {len(normal_paths)} EN + {sum(1 for p in zh_paths if p)} ZH")

    if stop_check and stop_check():
        return clip_paths, [], {}

    group_info, line_to_group = _build_group_info(
        groups, normal_paths, dialogue_durations, audio_dir, clip_paths, args.pad,
        fps=24)  # original 模式 24fps：组音频总长与 timeline 帧格精确同源
    return clip_paths, group_info, line_to_group


def _step4_timeline(args, checkpoint: dict, script: dict, work_dir: Path,
                    dirs: dict, tts_results: dict) -> tuple[list[dict], dict, list[str], list[str]]:
    """Step 4: build timeline + SRT (or resume from meta.json)."""
    print("\n" + "=" * 60)
    print("Step 4: Building timeline + SRT...")
    sub_dir = dirs["subtitles"]
    srt_path = sub_dir / "output.srt"
    meta_path = sub_dir / "meta.json"

    if _step_done(checkpoint, "step4_timeline") and srt_path.exists() and meta_path.exists():
        print("  [Resume] Loading existing timeline + SRT...")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return (meta["timeline"], meta.get("narration", {}),
                meta.get("normal_paths", []), meta.get("zh_paths", []))

    tts = TTSEngine()
    dialogue_durations = tts_results.get("dialogue_durations", [])
    narration = tts_results.get("narration", {})
    normal_paths = tts_results.get("normal_paths", [])
    zh_paths = tts_results.get("zh_paths", [])

    if args.structure == "quest":
        from quest.timeline_quest import build_quest_timeline, build_srt_from_timeline_quest
        timeline = build_quest_timeline(script, dialogue_durations, pad=args.pad)
        _enrich_timeline(timeline, tts, args.pad, dialogue_durations, zh_paths, narration, output_fps=25)
        srt = build_srt_from_timeline_quest(timeline, gap=0.0)
    else:
        timeline = build_listening_timeline(
            script, dialogue_durations,
            pad=args.pad, practice_duration=args.practice_duration,
            en_repeats=getattr(args, "ch3_en_repeats", 3),
            zh_repeats=getattr(args, "ch3_zh_repeats", 1),
            practice_intro_show=bool(getattr(args, "ch3_practice_intro_show", True)),
        )
        if args.structure == "original_cutout":
            # 开头/结尾对齐 quest 主持人形式：移除标题卡，插入 welcome + hook_intro
            rewrite_title_card_as_host_segments(timeline, script)
        # original/original_static 按 24fps 编码输出，quantize 消除逐段累计漂移
        # （与 quest/cutout 的 25fps 量化同理）
        _enrich_timeline(timeline, tts, args.pad, dialogue_durations, zh_paths, narration,
                         output_fps=25 if args.structure == "original_cutout" else 24)
        srt = build_srt_from_timeline(timeline, gap=0.0)

    srt_path.parent.mkdir(parents=True, exist_ok=True)
    srt_path.write_text(srt, encoding="utf-8")
    print(f"  SRT saved: {srt_path}")

    meta = {
        "timeline": timeline,
        "script": script,
        "pad": args.pad,
        "narration": narration,
        "normal_paths": normal_paths,
        "zh_paths": zh_paths,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  Meta saved: {meta_path}")
    _save_checkpoint(work_dir, "step4_timeline")
    return timeline, narration, normal_paths, zh_paths


def _step45_thumbnail(args, checkpoint: dict, script: dict, work_dir: Path,
                      dirs: dict, timeline: list[dict], ctx: dict) -> None:
    """Step 4.5: generate YouTube thumbnail + metadata."""
    print("\n" + "-" * 60)
    print("Step 4.5: Generating YouTube metadata + thumbnail...")
    from thumbnail_gen import generate_thumbnail, save_youtube_metadata

    thumb_path = str(work_dir / "thumbnail.jpg")
    yt_meta_path = str(work_dir / "youtube_metadata.json")
    if _step_done(checkpoint, "step4.5_thumbnail") and os.path.exists(yt_meta_path) and os.path.exists(thumb_path):
        print("  [Resume] Thumbnail + YouTube metadata already exist, skipping...")
        return

    if args.structure in ("quest",):
        _s0 = dirs["images"] / "scene_0.png"
        scene_img_full = str(_s0 if _s0.exists() else dirs["images"] / "scene.png")
    else:
        scene_img_full = str(dirs["images"] / "scene.png")
    if args.structure in ("quest", "original_cutout"):
        char_scene_cdn = ctx["image_urls"].get("pose_char_a_0.png", "")
    else:
        char_scene_cdn = ctx["image_urls"].get("char_scene.png", "")
    generate_thumbnail(
        script=script,
        scene_img=scene_img_full,
        output_path=thumb_path,
        mcp_call_tool=call_tool,
        mcp_parse_task_id=parse_task_id,
        mcp_poll_task=poll_task,
        mcp_download_file=download_file,
        structure=args.structure,
        char_scene_url=char_scene_cdn,
    )
    save_youtube_metadata(
        script=script,
        timeline=timeline,
        output_path=yt_meta_path,
        structure=args.structure,
    )
    _save_checkpoint(work_dir, "step4.5_thumbnail")


def _step5_compose(args, checkpoint: dict, script: dict, work_dir: Path, dirs: dict,
                   clip_paths: list[str], timeline: list[dict], narration: dict,
                   normal_paths: list[str], zh_paths: list[str], tts_results: dict,
                   group_info: list[dict], line_to_group: dict,
                   stop_check=None) -> tuple[str, str]:
    """Step 5: compose the final video."""
    print("\n" + "=" * 60)
    print("Step 5: Composing final video...")
    scene_img = str(dirs["images"] / "scene.png")
    if args.structure == "quest":
        _s0 = dirs["images"] / "scene_0.png"
        if _s0.exists():
            scene_img = str(_s0)
    sub_dir = dirs["subtitles"]

    # 字幕样式：style_id → 样式 dict（"" → None，走 --subtitle-font-size 历史行为）
    sub_style: dict | None = None
    if getattr(args, "subtitle_style", ""):
        try:
            from subtitle_style_manager import get_style as _get_sub_style
            sub_style = _get_sub_style(args.subtitle_style)
        except ImportError:
            sub_style = None
        if sub_style is None:
            print(f"  [SubtitleStyle] 未找到样式 '{args.subtitle_style}'，回退默认字幕参数")

    def progress_cb(pct, msg):
        print(f"  [{pct}%] {msg}")

    yt_title = script.get("youtube_title", script.get("title", "final"))
    safe_vid_name = _safe_dirname(yt_title, "final_video")
    final_video_path = work_dir / f"{safe_vid_name}.mp4"
    if _step_done(checkpoint, "step5_compose") and final_video_path.exists():
        print("  [Resume] Final video already exists, skipping compose...")
        return str(final_video_path), safe_vid_name

    if args.structure == "quest":
        from quest.video_compose_quest import compose_quest
        # Build per-character pose map (all chars: 8 poses each)
        char_pose_map = {}
        for ck in ("char_a", "char_b", "char_c"):
            poses = [str(dirs["images"] / f"pose_{ck}_{j}.png") for j in range(8)]
            if all(os.path.exists(p) for p in poses):
                char_pose_map[ck] = poses
        # Host: 8 poses
        host_poses = [str(dirs["images"] / f"pose_host_{j}.png") for j in range(8)]
        if not all(os.path.exists(p) for p in host_poses):
            host_poses = [str(dirs["images"] / f"pose_host_{j}.png") for j in range(4)]
        char_pose_map["host"] = host_poses

        # Fallback pose_images (legacy)
        dialogue = script.get("dialogue", [])
        pose_images = []
        for line in dialogue:
            speaker = line.get("speaker", "char_a")
            poses = char_pose_map.get(speaker, [])
            if not poses:
                poses = char_pose_map.get("char_a", [scene_img])
            pose_images.append(poses)

        # Host background (TV studio)
        host_bg = str(dirs["images"] / "host_bg.png")
        if not os.path.exists(host_bg):
            host_bg = scene_img

        # Multiple scene backgrounds for dialogue variety
        scene_bg_list = [scene_img]
        si_list = script.get("scene_images", [])
        for si in range(len(si_list)):
            p = str(dirs["images"] / f"scene_{si}.png")
            if os.path.exists(p):
                scene_bg_list.append(p)
            else:
                scene_bg_list.append(scene_img)

        final_path = compose_quest(
            work_dir=str(work_dir),
            pose_images=pose_images,
            char_pose_map=char_pose_map,
            host_poses=host_poses,
            host_bg=host_bg,
            scene_bg_list=scene_bg_list,
            render_fps=getattr(args, "render_fps", 12),
            workers=getattr(args, "workers", 1),
            timeline=timeline,
            script=script,
            narration=narration,
            normal_paths=normal_paths,
            scene_img=scene_img,
            srt_dir=str(sub_dir),
            pad=args.pad,
            show_zh=not getattr(args, "no_zh_subtitle", False),
            subtitle_font_size=args.subtitle_font_size,
            subtitle_style=sub_style,
            progress_cb=progress_cb,
            stop_check=stop_check,
        )
    elif args.structure == "original_static":
        from video_compose import compose_image
        n = len(script.get("dialogue", []))
        dialogue_images = [str(dirs["images"] / f"dialogue_img_{i}.png") for i in range(n)]
        final_path = compose_image(
            work_dir=str(work_dir),
            dialogue_images=dialogue_images,
            pose_images=[],
            background_img=scene_img,
            timeline=timeline,
            script=script,
            narration=narration,
            normal_paths=normal_paths,
            zh_paths=zh_paths,
            scene_img=scene_img,
            srt_dir=str(sub_dir),
            pad=args.pad,
            progress_cb=progress_cb,
            animation="none",
            subtitle_font_size=args.subtitle_font_size,
            subtitle_style=sub_style,
            show_zh=not getattr(args, "no_zh_subtitle", False),
            ch3_zh_always=bool(getattr(args, "ch3_zh_always", True)),
        )
    elif args.structure == "original_cutout":
        from original_cutout_compose import compose_original_cutout
        # Build per-character pose map (char_a/char_b only, 8 poses each)
        char_pose_map = {}
        for ck in ("char_a", "char_b"):
            poses = [str(dirs["images"] / f"pose_{ck}_{j}.png") for j in range(8)]
            if all(os.path.exists(p) for p in poses):
                char_pose_map[ck] = poses
        if not char_pose_map:
            print("  [Cutout] WARNING: no pose images found, compose will use fallback statics")
        # 主持人姿势：绑定角色时复用其图集，否则用独立主持人图集（8 姿势，缺则回退 4）
        _host_bound = getattr(args, "host_character", "")
        if _host_bound and _host_bound in char_pose_map:
            host_poses = char_pose_map[_host_bound]
        else:
            host_poses = [str(dirs["images"] / f"pose_host_{j}.png") for j in range(8)]
            if not all(os.path.exists(p) for p in host_poses):
                host_poses = [str(dirs["images"] / f"pose_host_{j}.png") for j in range(4)]
        # 演播室背景（quest 同款）；缺失时回退场景图
        host_bg = str(dirs["images"] / "host_bg.png")
        if not os.path.exists(host_bg):
            host_bg = scene_img
        scene_bg_list = [scene_img]
        final_path = compose_original_cutout(
            work_dir=str(work_dir),
            char_pose_map=char_pose_map,
            scene_bg_list=scene_bg_list,
            host_poses=host_poses,
            host_bg=host_bg,
            timeline=timeline,
            script=script,
            narration=narration,
            normal_paths=normal_paths,
            zh_paths=zh_paths,
            scene_img=scene_img,
            srt_dir=str(sub_dir),
            pad=args.pad,
            render_fps=getattr(args, "render_fps", 12),
            workers=getattr(args, "workers", 1),
            show_zh=not getattr(args, "no_zh_subtitle", False),
            subtitle_font_size=args.subtitle_font_size,
            subtitle_style=sub_style,
            ch3_zh_always=bool(getattr(args, "ch3_zh_always", True)),
            progress_cb=progress_cb,
            stop_check=stop_check,
        )
    else:
        final_path = compose_listening(
            work_dir=str(work_dir),
            clip_paths=clip_paths,
            timeline=timeline,
            script=script,
            narration=narration,
            normal_paths=normal_paths,
            zh_paths=zh_paths,
            scene_img=scene_img,
            srt_dir=str(sub_dir),
            pad=args.pad,
            progress_cb=progress_cb,
            group_info=group_info,
            line_to_group=line_to_group,
            subtitle_font_size=args.subtitle_font_size,
            subtitle_style=sub_style,
            show_zh=not getattr(args, "no_zh_subtitle", False),
            ch3_zh_always=bool(getattr(args, "ch3_zh_always", True)),
        )
    if not final_path:
        print("  [Compose] Interrupted or no output, skipping checkpoint save.")
        return "", safe_vid_name
    _save_checkpoint(work_dir, "step5_compose")
    return final_path, safe_vid_name


def _step6_4k(args, checkpoint: dict, work_dir: Path, final_path: str,
              safe_vid_name: str) -> Path | None:
    """Step 6: upscale the final video to 4K. Returns the 4K path or None."""
    print("\n" + "=" * 60)
    print("Step 6: Upscaling to 4K...")
    final_4k_path = work_dir / f"{safe_vid_name}_4K.mp4"
    if args.no_4k:
        print("  [4K] Skipped (--no-4k).")
        return None
    if _step_done(checkpoint, "step6_4k") and final_4k_path.exists():
        print("  [Resume] 4K video already exists, skipping...")
        return final_4k_path
    try:
        r = subprocess.run(
            ["ffmpeg", "-i", final_path,
             "-vf", "scale=3840:2160:flags=lanczos",
             "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-threads", "0",
             "-c:a", "copy",
             str(final_4k_path), "-y"],
            capture_output=True,
            timeout=max(60, int(getattr(args, "upscale_timeout", 3600) or 3600)))
    except subprocess.TimeoutExpired:
        print(f"  [4K] Upscaling timed out (>{getattr(args, 'upscale_timeout', 3600)}s) — 720p version is still available.")
        r = None
        final_4k_path.unlink(missing_ok=True)
        raise RuntimeError("STEP6_4K_FAILED") from None
    except Exception as e:
        print(f"  [4K] Upscaling error: {e}")
        r = None
        final_4k_path.unlink(missing_ok=True)
        raise RuntimeError("STEP6_4K_FAILED") from None
    if r is not None and r.returncode == 0 and final_4k_path.exists():
        size_4k = os.path.getsize(final_4k_path) / (1024 * 1024)
        print(f"  4K video saved: {final_4k_path} ({size_4k:.1f}MB)")
        _save_checkpoint(work_dir, "step6_4k")
        return final_4k_path
    if r is not None:
        print(f"  [4K] Upscaling failed, 720p version is still available.")
        stderr = r.stderr.decode("utf-8", errors="replace")[-500:] if r.stderr else ""
        if stderr:
            print(f"  [4K] FFmpeg stderr: {stderr}")
        # 失败必须抛出：否则调用方会误以为全部完成而清除 checkpoint，
        # 导致无法用 --resume 直接重试 4K 步骤
        raise RuntimeError("STEP6_4K_FAILED")
    return None


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def main():
    args = _parse_args()

    # Per-structure defaults (explicit CLI values win)
    if args.num_lines is None:
        args.num_lines = 48 if args.structure == "quest" else 18
    if args.pad is None:
        args.pad = 0.4

    # Ch3 跟读次数：clamp 到 0-10，防止误填导致视频长度失控
    args.ch3_en_repeats = max(0, min(10, int(args.ch3_en_repeats)))
    args.ch3_zh_repeats = max(0, min(10, int(args.ch3_zh_repeats)))

    # original_cutout: quest-style TTS rate (0% = normal speed)
    if args.tts_rate is None and args.structure == "original_cutout":
        args.tts_rate = "0%"

    # original_static: always use static images (no landing/stop_motion animation)
    if args.structure == "original_static":
        args.animation = "none"

    os.environ["LLM_RETRIES"] = str(args.llm_retries)
    # 画面风格：注入 env 供 llm_client / thumbnail_gen / 各 step 读取
    os.environ["VISUAL_STYLE_ID"] = str(getattr(args, "visual_style", "pixar3d"))
    os.environ["VISUAL_STYLE_PROMPT"] = _resolve_style_prompt(args.visual_style)
    style_name = ""
    try:
        from style_manager import get_style
        _s = get_style(args.visual_style)
        style_name = f" ({_s['name']})" if _s else ""
    except Exception:
        pass
    print(f"  [Style] Visual style: {args.visual_style}{style_name}")
    # Full raw LLM responses are dumped here when _chat hits errors
    os.environ.setdefault("LLM_DEBUG_DIR", str(Path(args.output).resolve() / "llm_debug"))

    # TTS engine config
    if args.qwen_model_path:
        os.environ["QWEN_MODEL_PATH"] = args.qwen_model_path
    if args.qwen_base_model_path:
        os.environ["QWEN_BASE_MODEL_PATH"] = args.qwen_base_model_path
    if args.qwen_voicedesign_model_path:
        os.environ["QWEN_VOICEDSIGN_MODEL_PATH"] = args.qwen_voicedesign_model_path
    if args.qwen_device:
        os.environ["QWEN_DEVICE"] = args.qwen_device

    # MOSS-TTS env
    if args.moss_model_path:
        os.environ["MOSS_MODEL_PATH"] = args.moss_model_path
    if args.moss_tokenizer_path:
        os.environ["MOSS_TOKENIZER_PATH"] = args.moss_tokenizer_path
    if args.moss_device:
        os.environ["MOSS_DEVICE"] = args.moss_device
    if args.moss_repo_dir:
        os.environ["MOSS_REPO_DIR"] = args.moss_repo_dir
    if args.moss_tts_temperature is not None:
        os.environ["MOSS_TTS_TEMPERATURE"] = str(args.moss_tts_temperature)
    if args.moss_tts_retry:
        os.environ["MOSS_TTS_RETRY"] = str(args.moss_tts_retry)

    if args.llm_provider == "openai":
        os.environ["LLM_PROVIDER"] = "openai"
        if args.openai_base_url:
            os.environ["OPENAI_BASE_URL"] = args.openai_base_url
        if args.openai_api_key:
            os.environ["OPENAI_API_KEY"] = args.openai_api_key
        if args.openai_model:
            os.environ["OPENAI_MODEL"] = args.openai_model
        elif args.model:
            os.environ["OPENAI_MODEL"] = args.model
        os.environ.setdefault("OPENAI_BASE_URL", "https://x666.me/v1")
        os.environ.setdefault("OPENAI_MODEL", "grok-4.6")
        if not os.environ.get("OPENAI_API_KEY"):
            print("ERROR: OPENAI_API_KEY not set. Pass --openai-api-key or set env var.")
            sys.exit(1)
    else:
        os.environ["LLM_PROVIDER"] = "sensenova"
        if args.api_key:
            os.environ["SENSENOVA_API_KEY"] = args.api_key
        if args.model:
            os.environ["SENSENOVA_MODEL"] = args.model
        if not os.environ.get("SENSENOVA_API_KEY"):
            print("ERROR: SENSENOVA_API_KEY not set. Pass --api-key or set env var.")
            sys.exit(1)

    # 生图 Provider：mcp（默认）或 sensenova（U1.5 Lite，读 IMAGE_PROVIDER env）
    os.environ["IMAGE_PROVIDER"] = args.image_provider
    if args.image_provider == "sensenova" and not os.environ.get("SENSENOVA_API_KEY"):
        print("ERROR: image_provider=sensenova requires SENSENOVA_API_KEY.")
        sys.exit(1)

    parent_dir = Path(args.output).resolve()
    parent_dir.mkdir(parents=True, exist_ok=True)
    # 新布局：每种模式一个独立文件夹 output/{mode}/{run_name}/
    mode_dir = parent_dir / args.structure
    mode_dir.mkdir(parents=True, exist_ok=True)

    used_topics_file = args.used_topics_file or str(parent_dir / "used_topics.json")

    if args.resume:
        checkpoint = _load_checkpoint(mode_dir)
        if checkpoint:
            cp_struct = checkpoint.get("structure")
            cp_anim = checkpoint.get("animation", args.animation)
            struct_changed = (cp_struct != args.structure)
            if cp_struct and struct_changed:
                print(f"  [Resume] Structure changed ({cp_struct}/{cp_anim} → {args.structure}/{args.animation}), starting fresh.")
                checkpoint = {}
            else:
                print(f"  [Resume] Found checkpoint: {checkpoint.get('completed_steps', [])}")
        else:
            print(f"  [Resume] No incomplete checkpoint, starting fresh.")
            checkpoint = {}
    else:
        checkpoint = {}

    topic = _resolve_topic(args, checkpoint)
    if topic is None:
        topic = pick_random_topic(args.topics_file, used_topics_file, mark=False)
        if not topic:
            print("  [Topic] ERROR: No topics found. Please specify --topic or provide topics.json.")
            sys.exit(1)
    args.topic = topic

    script, work_dir, dirs = _step0_script(args, checkpoint, topic, mode_dir, used_topics_file)
    _step1_mcp(args)
    ctx = _step2_images_tts(args, checkpoint, script, work_dir, dirs)
    clip_paths, group_info, line_to_group = _step3_clips(args, checkpoint, work_dir, dirs, script, ctx)
    timeline, narration, normal_paths, zh_paths = _step4_timeline(
        args, checkpoint, script, work_dir, dirs, ctx["tts_results"])
    _step45_thumbnail(args, checkpoint, script, work_dir, dirs, timeline, ctx)
    final_path, safe_vid_name = _step5_compose(
        args, checkpoint, script, work_dir, dirs, clip_paths, timeline,
        narration, normal_paths, zh_paths, ctx["tts_results"], group_info, line_to_group)
    final_4k_path = _step6_4k(args, checkpoint, work_dir, final_path, safe_vid_name)

    cp_path = work_dir / "checkpoint.json"
    if cp_path.exists():
        cp_path.unlink()
        print("  [Checkpoint] Cleared — next run will start fresh.")

    print("\n" + "=" * 60)
    print(f"DONE! Final video: {final_path}")
    print(f"Size: {os.path.getsize(final_path) / (1024*1024):.1f}MB")
    if final_4k_path is not None and final_4k_path.exists():
        print(f"4K video: {final_4k_path}")
        print(f"4K Size: {os.path.getsize(final_4k_path) / (1024*1024):.1f}MB")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        if "ALL_MCP_TOKENS_EXHAUSTED" in str(e):
            print("\n" + "=" * 60)
            print("FATAL: 所有 MCP Token 积分已耗尽！")
            print("请充值积分后重新运行（加 --resume 参数）继续上次未完成的视频。")
            print("=" * 60)
            sys.exit(1)
        if "STEP6_4K_FAILED" in str(e):
            print("\n[4K] 上采样失败 — 1080p 成片已完成且可用。")
            print("checkpoint 已保留：加 --resume 可直接重试 4K 步骤；"
                  "或加 --no-4k --resume 直接完成本次运行。")
            sys.exit(1)
        raise
    except KeyboardInterrupt:
        print("\n\n用户中断。下次运行加 --resume 可继续。")
        sys.exit(0)
