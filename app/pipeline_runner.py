"""Pipeline runner: executes pipeline.py as subprocess,
streams logs via SSE, manages process lifecycle."""
import asyncio
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import AsyncGenerator

from .config_manager import build_cli_args, load_config

# Step detection patterns (match pipeline.py output)
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

STEP_LABELS = {
    "step0_script": "LLM 脚本生成",
    "step1_mcp": "MCP 初始化",
    "step2_images_tts": "图片 + TTS 生成",
    "step3_video": "视频片段生成",
    "step4_timeline": "时间轴 + SRT",
    "step45_thumbnail": "缩略图 + 元数据",
    "step5_compose": "视频合成",
    "step6_4k": "4K 超分辨率",
}


class PipelineRunner:
    """Manages a single pipeline.py subprocess execution."""

    def __init__(self):
        self.process: subprocess.Popen | None = None
        self.log_lines: list[str] = []
        self.log_lock = threading.Lock()
        self.status: str = "idle"  # idle, running, done, error, stopped
        self.current_step: str = ""
        self.current_step_label: str = ""
        self.started_at: float = 0
        self.finished_at: float = 0
        self.config: dict = {}
        self._new_log_event = threading.Event()

    @property
    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def start(self, config: dict, resume: bool = False) -> bool:
        if self.is_running:
            return False

        self.config = config
        self.log_lines = []
        self.status = "running"
        self.current_step = ""
        self.current_step_label = ""
        self.started_at = time.time()
        self.finished_at = 0

        try:
            cli_args = build_cli_args(config, resume=resume)
        except FileNotFoundError as e:
            self._append_log(f"FATAL: {e}")
            self.status = "error"
            return False

        # Set env vars that pipeline.py reads
        env = os.environ.copy()
        provider = config.get("llm_provider", "sensenova")
        env["LLM_PROVIDER"] = provider
        if provider == "sensenova" and config.get("sensenova_api_key"):
            env["SENSENOVA_API_KEY"] = config["sensenova_api_key"]
        if config.get("sensenova_model"):
            env["SENSENOVA_MODEL"] = config["sensenova_model"]
        if provider == "openai":
            if config.get("openai_base_url"):
                env["OPENAI_BASE_URL"] = config["openai_base_url"]
            if config.get("openai_api_key"):
                env["OPENAI_API_KEY"] = config["openai_api_key"]
            if config.get("openai_model"):
                env["OPENAI_MODEL"] = config["openai_model"]
        if config.get("voxcpm_worker_url"):
            env["VOXCPM_WORKER_URL"] = config["voxcpm_worker_url"]
        if config.get("voxcpm_api_key"):
            env["VOXCPM_API_KEY"] = config["voxcpm_api_key"]

        # Set HF_ENDPOINT for Windows (China mirror)
        if sys.platform == "win32" and "HF_ENDPOINT" not in env:
            env["HF_ENDPOINT"] = "https://hf-mirror.com"

        self._append_log(f"$ {' '.join(cli_args)}\n")

        try:
            self.process = subprocess.Popen(
                cli_args,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=str(Path(cli_args[1]).parent),
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                text=True,
            )
        except Exception as e:
            self._append_log(f"FATAL: Failed to start pipeline: {e}")
            self.status = "error"
            return False

        # Start reader thread
        t = threading.Thread(target=self._read_output, daemon=True)
        t.start()

        return True

    def _read_output(self):
        """Read subprocess output line by line."""
        assert self.process is not None
        try:
            for line in self.process.stdout:
                self._append_log(line.rstrip("\n\r"))
        except Exception as e:
            self._append_log(f"[Reader error] {e}")
        finally:
            retcode = self.process.wait()
            self.finished_at = time.time()
            if self.status == "stopped":
                pass  # Already set by stop()
            elif retcode == 0:
                self.status = "done"
                self._append_log("\n✅ Pipeline 完成!")
            else:
                self.status = "error"
                self._append_log(f"\n❌ Pipeline 失败 (exit code {retcode})")

    def _append_log(self, line: str):
        with self.log_lock:
            self.log_lines.append(line)
            # Keep last 5000 lines to avoid memory issues
            if len(self.log_lines) > 5000:
                self.log_lines = self.log_lines[-3000:]
            # Detect current step
            for pattern, step_id, step_label in STEP_PATTERNS:
                if re.search(pattern, line):
                    self.current_step = step_id
                    self.current_step_label = step_label
                    break
        self._new_log_event.set()

    def stop(self):
        if not self.is_running:
            return
        self.status = "stopped"
        self._append_log("\n⏹ 用户停止了 Pipeline")
        try:
            if sys.platform == "win32":
                self.process.terminate()
            else:
                self.process.send_signal(signal.SIGTERM)
            # Wait 5s then kill
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
        except Exception:
            pass

    def get_new_logs(self, since: int = 0) -> list[str]:
        with self.log_lock:
            if since < len(self.log_lines):
                return self.log_lines[since:]
        return []

    def get_progress(self) -> dict:
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
        }


# Singleton instance
_runner: PipelineRunner | None = None


def get_runner() -> PipelineRunner:
    global _runner
    if _runner is None:
        _runner = PipelineRunner()
    return _runner
