"""Pipeline service: directly imports and calls pipeline.py step functions.

Replaces the subprocess approach with direct Python function calls,
enabling real-time progress tracking and intermediate result access.
"""
import io
import json
import os
import re
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

# Add pipeline source to path
PIPELINE_DIR = Path(__file__).parent.parent.parent / "colab_listening_b"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

# Import pipeline modules (all lazy-heavy, import is cheap)
from mcp_client import initialize as mcp_initialize, reinitialize as mcp_reinit
from topic_manager import pick_random_topic, mark_topic_used
from media_utils import safe_filename as _safe_dirname
from checkpoint import (
    save_checkpoint as _save_checkpoint,
    load_checkpoint as _load_checkpoint,
    step_done as _step_done,
)

# Step detection patterns
STEP_PATTERNS = [
    (r"Step 0[:\s]", "step0_script", "LLM 脚本生成"),
    (r"Step 1[:\s]", "step1_mcp", "MCP 初始化"),
    (r"Step 2[:\s]", "step2_images_tts", "图片 + TTS 生成"),
    (r"Step 3[:\s]", "step3_video", "视频片段生成"),
    (r"Step 4\.5[:\s]", "step45_thumbnail", "缩略图 + 元数据"),
    (r"Step 4[:\s]", "step4_timeline", "时间轴 + SRT"),
    (r"Step 5[:\s]", "step5_compose", "视频合成"),
    (r"Step 6[:\s]", "step6_4k", "4K 超分辨率"),
]

STEP_ORDER = [
    "step0_script", "step1_mcp", "step2_images_tts",
    "step3_video", "step4_timeline", "step45_thumbnail",
    "step5_compose", "step6_4k",
]


class _LineBuffer(io.StringIO):
    """Capture stdout line-by-line, forwarding complete lines to a callback."""

    def __init__(self, on_line):
        super().__init__()
        self._on_line = on_line
        self._buf = ""

    def write(self, text):
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._on_line(line.rstrip("\r"))
        return len(text)

    def flush(self):
        if self._buf:
            self._on_line(self._buf.rstrip("\r"))
            self._buf = ""


class PipelineService:
    """Manages pipeline execution in a background thread with direct imports."""

    def __init__(self):
        self._thread: threading.Thread | None = None
        self._stop_flag = threading.Event()
        self._lock = threading.Lock()
        self.log_lines: list[str] = []
        self.status: str = "idle"
        self.current_step: str = ""
        self.current_step_label: str = ""
        self.started_at: float = 0
        self.finished_at: float = 0
        self.config: dict = {}
        self.error: str = ""
        self.work_dir: str = ""
        self.final_path: str = ""
        self._step_mode: bool = False
        self._paused_after_step: str = ""

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def is_paused(self) -> bool:
        return self._step_mode and self.status == "paused"

    def start(self, config: dict[str, Any], resume: bool = False, step_mode: bool = False) -> bool:
        if self.is_running:
            return False

        self.config = config
        self._stop_flag.clear()
        self._step_mode = step_mode
        self._paused_after_step = ""
        with self._lock:
            self.log_lines = []
            self.status = "running"
            self.current_step = ""
            self.current_step_label = ""
            self.started_at = time.time()
            self.finished_at = 0
            self.error = ""
            self.work_dir = ""
            self.final_path = ""

        self._thread = threading.Thread(
            target=self._run, args=(config, resume), daemon=True)
        self._thread.start()
        return True

    def _wait_for_step_approval(self, step_name: str):
        """In step mode, pause after each step and wait for user to continue."""
        if not self._step_mode:
            return
        self._paused_after_step = step_name
        with self._lock:
            self.status = "paused"
        self._on_log_line(f"\n⏸ [Step Mode] Paused after {step_name}. Review the output, then click 'Continue' to proceed.")

        # Wait until continue or stop
        while self._paused_after_step and not self._stop_flag.is_set():
            time.sleep(0.5)

        if self._stop_flag.is_set():
            return
        with self._lock:
            self.status = "running"
        self._on_log_line(f"▶ [Step Mode] Continuing after {step_name}...")

    def continue_step(self):
        """Resume from a step-mode pause."""
        self._paused_after_step = ""

    def get_progress(self) -> dict:
        with self._lock:
            if self.current_step:
                try:
                    idx = STEP_ORDER.index(self.current_step)
                    progress = int((idx + 1) / len(STEP_ORDER) * 100)
                except ValueError:
                    progress = 0
            else:
                progress = 0

            elapsed = 0
            if self.started_at:
                end = self.finished_at if self.finished_at else time.time()
                elapsed = int(end - self.started_at)

            return {
                "status": self.status,
                "current_step": self.current_step,
                "current_step_label": self.current_step_label,
                "progress": progress,
                "elapsed": elapsed,
                "log_count": len(self.log_lines),
                "is_running": self.is_running,
                "error": self.error,
                "work_dir": self.work_dir,
                "final_path": self.final_path,
                "step_mode": self._step_mode,
                "paused_after_step": self._paused_after_step,
            }

    def _set_env(self, config: dict) -> None:
        """Set env vars from config dict before importing pipeline modules."""
        os.environ["LLM_RETRIES"] = str(config.get("llm_retries", 10))
        if config.get("llm_min_interval"):
            os.environ["LLM_MIN_INTERVAL"] = str(config["llm_min_interval"])

        provider = config.get("llm_provider", "sensenova")
        os.environ["LLM_PROVIDER"] = provider
        if provider == "sensenova":
            if config.get("sensenova_api_key"):
                os.environ["SENSENOVA_API_KEY"] = config["sensenova_api_key"]
            if config.get("sensenova_model"):
                os.environ["SENSENOVA_MODEL"] = config["sensenova_model"]
        else:
            if config.get("openai_base_url"):
                os.environ["OPENAI_BASE_URL"] = config["openai_base_url"]
            if config.get("openai_api_key"):
                os.environ["OPENAI_API_KEY"] = config["openai_api_key"]
            if config.get("openai_model"):
                os.environ["OPENAI_MODEL"] = config["openai_model"]

        if config.get("voxcpm_worker_url"):
            os.environ["VOXCPM_WORKER_URL"] = config["voxcpm_worker_url"]
        if config.get("voxcpm_api_key"):
            os.environ["VOXCPM_API_KEY"] = config["voxcpm_api_key"]

        if sys.platform == "win32" and "HF_ENDPOINT" not in os.environ:
            os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    def _build_args(self, config: dict) -> SimpleNamespace:
        """Convert config dict → SimpleNamespace matching pipeline.py args."""
        num_lines = config.get("num_lines", "")
        try:
            num_lines = int(num_lines) if num_lines else None
        except (ValueError, TypeError):
            num_lines = None

        pad = config.get("pad", "")
        try:
            pad = float(pad) if pad else None
        except (ValueError, TypeError):
            pad = None

        structure = config.get("structure", "original")
        if num_lines is None:
            num_lines = 48 if structure == "quest" else 18
        if pad is None:
            pad = 0.4

        tokens_raw = config.get("mcp_tokens", "").strip()
        mcp_tokens = ",".join(
            t.strip() for t in tokens_raw.split("\n") if t.strip()
        ) if tokens_raw else ""

        return SimpleNamespace(
            topic=config.get("topic", "") or None,
            cefr=config.get("cefr", "A2"),
            num_lines=num_lines,
            structure=structure,
            animation=config.get("animation", "landing"),
            llm_provider=config.get("llm_provider", "sensenova"),
            sensenova_api_key=config.get("sensenova_api_key", ""),
            sensenova_model=config.get("sensenova_model", "deepseek-v4-flash"),
            model=config.get("sensenova_model", "deepseek-v4-flash"),
            api_key=config.get("sensenova_api_key", ""),
            openai_base_url=config.get("openai_base_url", ""),
            openai_api_key=config.get("openai_api_key", ""),
            openai_model=config.get("openai_model", "grok-4.6"),
            llm_retries=int(config.get("llm_retries", 10)),
            mcp_tokens=mcp_tokens or None,
            mcp_token=None,
            clip_duration=int(config.get("clip_duration", 15)),
            output=config.get("output_dir", "./output"),
            topics_file=config.get("topics_file", str(PIPELINE_DIR / "topics.json")),
            used_topics_file=config.get("used_topics_file", "") or None,
            lessons_dir=config.get("lessons_dir", "") or None,
            practice_duration=float(config.get("practice_duration", 3.0)),
            pad=pad,
            render_fps=int(config.get("render_fps", 8)),
            workers=int(config.get("workers", 1)),
            subtitle_font_size=int(config.get("subtitle_font_size", 60)),
            no_zh_subtitle=bool(config.get("no_zh_subtitle", False)),
            no_4k=bool(config.get("no_4k", False)),
            tts_engine=config.get("tts_engine", "kokoro"),
            tts_rate=config.get("tts_rate", "") or None,
            voxcpm_worker_url=config.get("voxcpm_worker_url", ""),
            voxcpm_api_key=config.get("voxcpm_api_key", ""),
            upscale_timeout=int(config.get("upscale_timeout", 3600)),
            resume=False,
        )

    def _on_log_line(self, line: str):
        """Called for each stdout line from pipeline."""
        with self._lock:
            self.log_lines.append(line)
            if len(self.log_lines) > 5000:
                self.log_lines = self.log_lines[-3000:]
            for pattern, step_id, step_label in STEP_PATTERNS:
                if re.search(pattern, line):
                    self.current_step = step_id
                    self.current_step_label = step_label
                    break

    def _run(self, config: dict, resume: bool):
        """Main pipeline execution in background thread."""
        # Import here so heavy imports happen in the worker thread
        from pipeline import (
            _step0_script, _step1_mcp, _step2_images_tts,
            _step3_clips, _step4_timeline, _step45_thumbnail,
            _step5_compose, _step6_4k,
            _generate_script_with_retry, _resolve_topic, _resolve_run_dir,
        )

        # Set env vars
        self._set_env(config)

        # Build args namespace
        args = self._build_args(config)
        args.resume = resume

        # Redirect stdout to capture print() output
        old_stdout = sys.stdout
        buf = _LineBuffer(self._on_log_line)
        sys.stdout = buf

        try:
            parent_dir = Path(args.output).resolve()
            parent_dir.mkdir(parents=True, exist_ok=True)
            used_topics_file = args.used_topics_file or str(parent_dir / "used_topics.json")

            # Load checkpoint for resume
            if resume:
                checkpoint = _load_checkpoint(parent_dir)
                if not checkpoint:
                    checkpoint = {}
            else:
                checkpoint = {}

            # Resolve topic
            topic = _resolve_topic(args, checkpoint)
            if topic is None:
                topic = pick_random_topic(args.topics_file, used_topics_file, mark=False)
                if not topic:
                    self._fail("No topics found. Please specify a topic or provide topics.json.")
                    return
            args.topic = topic

            # Step 0: Script generation
            script, work_dir, dirs = _step0_script(
                args, checkpoint, topic, parent_dir, used_topics_file)
            self.work_dir = str(work_dir)

            if self._stop_flag.is_set():
                self._set_stopped()
                return
            self._wait_for_step_approval("step0_script")
            if self._stop_flag.is_set():
                self._set_stopped()
                return

            # Step 1: MCP init
            raw_tokens = args.mcp_tokens or ""
            tokens = [t.strip() for t in raw_tokens.split(",") if t.strip()] if raw_tokens else []
            if tokens:
                mcp_reinit(tokens=tokens)
            else:
                mcp_reinit()

            if self._stop_flag.is_set():
                self._set_stopped()
                return
            self._wait_for_step_approval("step1_mcp")
            if self._stop_flag.is_set():
                self._set_stopped()
                return

            # Step 2: Images + TTS
            ctx = _step2_images_tts(args, checkpoint, script, work_dir, dirs)

            if self._stop_flag.is_set():
                self._set_stopped()
                return
            self._wait_for_step_approval("step2_images_tts")
            if self._stop_flag.is_set():
                self._set_stopped()
                return

            # Step 3: Video clips
            clip_paths, group_info, line_to_group = _step3_clips(
                args, checkpoint, work_dir, dirs, script, ctx)

            if self._stop_flag.is_set():
                self._set_stopped()
                return
            self._wait_for_step_approval("step3_video")
            if self._stop_flag.is_set():
                self._set_stopped()
                return

            # Step 4: Timeline
            timeline, narration, normal_paths, zh_paths = _step4_timeline(
                args, checkpoint, script, work_dir, dirs, ctx["tts_results"])

            if self._stop_flag.is_set():
                self._set_stopped()
                return
            self._wait_for_step_approval("step4_timeline")
            if self._stop_flag.is_set():
                self._set_stopped()
                return

            # Step 4.5: Thumbnail
            _step45_thumbnail(args, checkpoint, script, work_dir, dirs, timeline, ctx)

            if self._stop_flag.is_set():
                self._set_stopped()
                return
            self._wait_for_step_approval("step45_thumbnail")
            if self._stop_flag.is_set():
                self._set_stopped()
                return

            # Step 5: Compose
            final_path, safe_vid_name = _step5_compose(
                args, checkpoint, script, work_dir, dirs, clip_paths, timeline,
                narration, normal_paths, zh_paths, ctx["tts_results"],
                group_info, line_to_group)
            self.final_path = final_path

            if self._stop_flag.is_set():
                self._set_stopped()
                return
            self._wait_for_step_approval("step5_compose")
            if self._stop_flag.is_set():
                self._set_stopped()
                return

            # Step 6: 4K
            final_4k_path = _step6_4k(args, checkpoint, work_dir, final_path, safe_vid_name)

            # Clear checkpoint on completion
            cp_path = work_dir / "checkpoint.json"
            if cp_path.exists():
                cp_path.unlink()

            with self._lock:
                self.status = "done"
                self.finished_at = time.time()
            self._on_log_line("")
            self._on_log_line("=" * 60)
            self._on_log_line(f"DONE! Final video: {final_path}")
            fsize = os.path.getsize(final_path) / (1024 * 1024)
            self._on_log_line(f"Size: {fsize:.1f}MB")
            if final_4k_path and Path(final_4k_path).exists():
                self._on_log_line(f"4K video: {final_4k_path}")

        except Exception as e:
            self._fail(f"{type(e).__name__}: {e}")
            import traceback
            tb = traceback.format_exc()
            for line in tb.split("\n"):
                self._on_log_line(line)
        finally:
            sys.stdout = old_stdout
            buf.flush()
            with self._lock:
                if self.status == "running":
                    self.status = "done"
                self.finished_at = time.time()

    def _fail(self, msg: str):
        with self._lock:
            self.status = "error"
            self.error = msg
            self.finished_at = time.time()
        self._on_log_line(f"\nFATAL: {msg}")

    def _set_stopped(self):
        with self._lock:
            self.status = "stopped"
            self.finished_at = time.time()
        self._on_log_line("\n⏹ Pipeline stopped by user")

    def stop(self):
        self._stop_flag.set()

    def get_logs_since(self, since: int = 0) -> list[str]:
        with self._lock:
            if since < len(self.log_lines):
                return self.log_lines[since:]
        return []


# Singleton
_service: PipelineService | None = None


def get_service() -> PipelineService:
    global _service
    if _service is None:
        _service = PipelineService()
    return _service
