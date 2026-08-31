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

from .config_manager import MODES, resolve_provider, load_config, find_run_dir
from . import run_mutex

# Add pipeline source to path (local copy — fully independent)
PIPELINE_DIR = Path(__file__).parent.parent / "pipeline"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

# Import pipeline modules (all lazy-heavy, import is cheap)
from mcp_client import initialize as mcp_initialize, reinitialize as mcp_reinit
from topic_manager import pick_random_topic
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

    def reconfigure(self, *args, **kwargs):
        # Pipeline 模块在 win32 下 import 时会调用 sys.stdout.reconfigure()；
        # Web 模式下 stdout 是本对象，接受并忽略该调用避免 AttributeError。
        pass


def _cfg_int(config: dict, key: str, default: int) -> int:
    """配置值安全转 int（空串/非法值回退默认），并 clamp 到 0-10。"""
    try:
        return max(0, min(10, int(config.get(key, default))))
    except (TypeError, ValueError):
        return default


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
        # 与模式测试互斥（MCP/TTS/FFmpeg 抢资源）；失败时在日志区提示占用方
        if not run_mutex.try_acquire("pipeline"):
            self._on_log_line(
                f"⏳ 启动被拒绝：当前被「{run_mutex.current_owner()}」占用，"
                f"请等待其完成后再启动主 pipeline。")
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

            run_name = ""
            if self.work_dir:
                run_name = Path(self.work_dir).name

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
                "run_name": run_name,
                "final_path": self.final_path,
                "step_mode": self._step_mode,
                "paused_after_step": self._paused_after_step,
            }

    def _set_env(self, config: dict) -> None:
        """Set env vars from config dict before importing pipeline modules."""
        os.environ["LLM_RETRIES"] = str(config.get("llm_retries", 10))
        if config.get("llm_min_interval"):
            os.environ["LLM_MIN_INTERVAL"] = str(config["llm_min_interval"])

        # 画面风格：注入 env 供 llm_client / thumbnail_gen / 各 step 读取
        from style_manager import resolve_style_prompt as _rsp
        style_id = str(config.get("visual_style", "pixar3d"))
        os.environ["VISUAL_STYLE_ID"] = style_id
        os.environ["VISUAL_STYLE_PROMPT"] = _rsp(style_id)

        # 生图 Provider：mcp（默认，TJGenerators 积分）或 sensenova（U1.5 Lite API 计费）
        os.environ["IMAGE_PROVIDER"] = str(config.get("image_provider", "mcp"))

        # 抠图引擎：auto（默认，有权重用 MODNet）/ modnet / white_threshold（原方法）
        os.environ["MATTING_ENGINE"] = str(config.get("matting_engine", "auto"))

        # SenseNova key：除 LLM 外，生图 Provider=sensenova 也依赖（LLM 可走自定义 OpenAI 通道，二者解耦）
        if str(config.get("sensenova_api_key") or "").strip():
            os.environ["SENSENOVA_API_KEY"] = str(config["sensenova_api_key"]).strip()

        provider = config.get("llm_provider", "sensenova")
        p_type, p_base_url, p_api_key, p_model = resolve_provider(config)
        os.environ["LLM_PROVIDER"] = p_type
        if p_type == "sensenova":
            if p_api_key:
                os.environ["SENSENOVA_API_KEY"] = p_api_key
            if p_model:
                os.environ["SENSENOVA_MODEL"] = p_model
        else:
            if p_base_url:
                os.environ["OPENAI_BASE_URL"] = p_base_url
            if p_api_key:
                os.environ["OPENAI_API_KEY"] = p_api_key
            if p_model:
                os.environ["OPENAI_MODEL"] = p_model

        if sys.platform == "win32" and "HF_ENDPOINT" not in os.environ:
            os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

        # 多轮生成 QA 旋钮（quest 与 original* 脚本生成读取；0=关闭需显式写入）
        if config.get("quest_beat_lines"):
            os.environ["QUEST_BEAT_LINES"] = str(config["quest_beat_lines"])
        if config.get("quest_qa_rounds") is not None and config.get("quest_qa_rounds") != "":
            os.environ["QUEST_QA_MAX_ROUNDS"] = str(config["quest_qa_rounds"])
            os.environ["LISTENING_QA_MAX_ROUNDS"] = str(config["quest_qa_rounds"])

        # 每行最大词数（字幕两行约束）：prompt 硬约束 + QA 门禁 + 渲染兜底共用
        if config.get("max_line_words"):
            os.environ["LISTENING_MAX_LINE_WORDS"] = str(config["max_line_words"])
            os.environ["QUEST_MAX_LINE_WORDS"] = str(config["max_line_words"])

        # MOSS-TTS env vars (read by generate_tts -> MossTTSEngine)
        if config.get("moss_model_path"):
            os.environ["MOSS_MODEL_PATH"] = str(config["moss_model_path"])
        if config.get("moss_tokenizer_path"):
            os.environ["MOSS_TOKENIZER_PATH"] = str(config["moss_tokenizer_path"])
        if config.get("moss_device"):
            os.environ["MOSS_DEVICE"] = str(config["moss_device"])
        if config.get("moss_repo_dir"):
            os.environ["MOSS_REPO_DIR"] = str(config["moss_repo_dir"])
        if config.get("moss_tts_temperature"):
            os.environ["MOSS_TTS_TEMPERATURE"] = str(config["moss_tts_temperature"])
        if config.get("moss_tts_retry"):
            os.environ["MOSS_TTS_RETRY"] = str(config["moss_tts_retry"])
        if config.get("moss_tts_top_p"):
            os.environ["MOSS_TTS_TOP_P"] = str(config["moss_tts_top_p"])
        if config.get("moss_tts_top_k"):
            os.environ["MOSS_TTS_TOP_K"] = str(config["moss_tts_top_k"])
        if config.get("moss_tts_rep_penalty"):
            os.environ["MOSS_TTS_REP_PENALTY"] = str(config["moss_tts_rep_penalty"])
        if config.get("moss_tts_text_temperature"):
            os.environ["MOSS_TTS_TEXT_TEMPERATURE"] = str(config["moss_tts_text_temperature"])
        if config.get("moss_tts_greedy"):
            os.environ["MOSS_TTS_GREEDY"] = "1"

        # Qwen-TTS env vars（与 CLI main() 注入保持一致；不设则引擎用内置默认）
        if config.get("qwen_model_path"):
            os.environ["QWEN_MODEL_PATH"] = str(config["qwen_model_path"])
        if config.get("qwen_base_model_path"):
            os.environ["QWEN_BASE_MODEL_PATH"] = str(config["qwen_base_model_path"])
        if config.get("qwen_voicedesign_model_path"):
            os.environ["QWEN_VOICEDSIGN_MODEL_PATH"] = str(config["qwen_voicedesign_model_path"])
        if config.get("qwen_device"):
            os.environ["QWEN_DEVICE"] = str(config["qwen_device"])

    # Character image file patterns per structure
    # （仅 char_scene.png 为 original/original_static 的角色合图；
    #   旧清单中的 char_a_ref/char_b_ref 从未由 image_gen 生成，已移除）
    _CHAR_FILES_ORIGINAL = ["char_scene.png"]

    def _build_character_overrides(self, config: dict) -> dict:
        """Build character overrides dict for LLM prompt injection.

        Sources (priority: library > run reuse > fixes):
          - character_library: {char_a: "lib_id"} → description + gender from library
            (仅音色+性别角色 description 为空 → 只注入 gender + qwen_speaker)
          - character_reuse: {char_a: "image"|"desc"|"voice"} → description + gender from run
            ("voice" → 只取 gender + qwen_speaker，外观每次由 LLM 按主题新创作)
          - character_fixes: {char_a: "custom desc"} → custom description

        role is NEVER injected — LLM always generates role based on topic.
        """
        source = config.get("character_source", "")
        reuse_raw = config.get("character_reuse", "")
        fixes_raw = config.get("character_fixes", "")
        lib_raw = config.get("character_library", "")
        voices_raw = config.get("character_voices", "")
        moss_voices_raw = config.get("character_moss_voices", "")
        kokoro_voices_raw = config.get("character_kokoro_voices", "")

        reuse_map: dict = {}
        fixes_map: dict[str, str] = {}
        lib_map: dict[str, str] = {}

        if reuse_raw:
            try:
                reuse_map = json.loads(reuse_raw) if isinstance(reuse_raw, str) else reuse_raw
            except (json.JSONDecodeError, TypeError):
                pass
        if fixes_raw:
            try:
                fixes_map = json.loads(fixes_raw) if isinstance(fixes_raw, str) else fixes_raw
            except (json.JSONDecodeError, TypeError):
                pass
        if lib_raw:
            try:
                lib_map = json.loads(lib_raw) if isinstance(lib_raw, str) else lib_raw
            except (json.JSONDecodeError, TypeError):
                pass
        voices_map: dict[str, str] = {}
        if voices_raw:
            try:
                voices_map = json.loads(voices_raw) if isinstance(voices_raw, str) else voices_raw
            except (json.JSONDecodeError, TypeError):
                pass
        moss_voices_map: dict[str, str] = {}
        if moss_voices_raw:
            try:
                moss_voices_map = json.loads(moss_voices_raw) if isinstance(moss_voices_raw, str) else moss_voices_raw
            except (json.JSONDecodeError, TypeError):
                pass
        kokoro_voices_map: dict[str, str] = {}
        if kokoro_voices_raw:
            try:
                kokoro_voices_map = json.loads(kokoro_voices_raw) if isinstance(kokoro_voices_raw, str) else kokoro_voices_raw
            except (json.JSONDecodeError, TypeError):
                pass

        structure = config.get("structure", "original")
        if structure == "quest":
            all_keys = ["char_a", "char_b", "char_c", "host"]
        elif structure == "original_cutout":
            # 独立主持人（未绑定角色时）同样参与复用，否则每次新跑随机生成新主持人
            all_keys = ["char_a", "char_b", "host", "narration"]
        else:
            all_keys = ["char_a", "char_b", "narration"]

        for k, v in list(reuse_map.items()):
            if v is True:
                reuse_map[k] = "image"
        if source and not reuse_map:
            reuse_map = {k: "image" for k in all_keys}

        overrides: dict[str, dict] = {}
        source_script: dict = {}

        if source:
            output_dir = Path(config.get("output_dir", "./output"))
            source_dir = find_run_dir(output_dir, source)
            sp = source_dir / "script.json" if source_dir else None
            if sp and sp.exists():
                try:
                    source_script = json.loads(sp.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    pass

        # Library dir
        lib_dir = Path(__file__).parent.parent / "configs" / "character_library"

        for key in all_keys:
            char_info: dict[str, str] = {}

            # Priority 1: library character
            lib_id = lib_map.get(key, "")
            if lib_id:
                lib_meta_path = lib_dir / lib_id / "meta.json"
                if lib_meta_path.exists():
                    try:
                        lib_meta = json.loads(lib_meta_path.read_text(encoding="utf-8"))
                        for suffix in ["description", "gender", "qwen_speaker",
                                       "moss_voice", "kokoro_voice"]:
                            val = lib_meta.get(suffix, "")
                            if val:
                                char_info[suffix] = val
                    except (json.JSONDecodeError, OSError):
                        pass

            # Priority 2: run reuse (only if no library assignment for this key)
            if not char_info:
                mode = reuse_map.get(key, False)
                if mode in ("image", "desc") and source_script:
                    for suffix in ["description", "gender"]:
                        val = source_script.get(f"{key}_{suffix}", "")
                        if val:
                            char_info[suffix] = val
                elif mode == "voice" and source_script:
                    # voice 模式：只固定音色+性别，外观由 LLM 按主题重新生成
                    for suffix in ["gender", "qwen_speaker", "moss_voice",
                                   "kokoro_voice"]:
                        val = source_script.get(f"{key}_{suffix}", "")
                        if val:
                            char_info[suffix] = val

            # Priority 3: custom fix (overrides description only)
            fix_desc = (fixes_map.get(key, "") or "").strip()
            if fix_desc:
                char_info["description"] = fix_desc

            if char_info:
                overrides[key] = char_info

            # Priority 4: explicit voice binding (highest priority)
            voice = voices_map.get(key, "").strip()
            if voice:
                if key not in overrides:
                    overrides[key] = {}
                overrides[key]["qwen_speaker"] = voice

            # Priority 5: explicit MOSS voice binding
            moss_voice = moss_voices_map.get(key, "").strip()
            if moss_voice:
                if key not in overrides:
                    overrides[key] = {}
                overrides[key]["moss_voice"] = moss_voice

            # Priority 6: explicit Kokoro voice binding
            kokoro_voice = kokoro_voices_map.get(key, "").strip()
            if kokoro_voice:
                if key not in overrides:
                    overrides[key] = {}
                overrides[key]["kokoro_voice"] = kokoro_voice

        return overrides

    def _reuse_characters(self, source_run: str, work_dir: Path,
                          script: dict, dirs: dict):
        """Copy character images and/or override descriptions.

        Mode "image": copy images + override description + gender (NOT role)
        Mode "desc":  override description only (NOT gender, NOT role, no images)
        Mode "voice": override gender + qwen_speaker only (no description, no images
                      — appearance is re-created by the LLM for every topic)
        character_fixes: override description (no source needed)

        Library characters with empty description (仅音色+性别) naturally inject
        gender + qwen_speaker only (empty values are skipped below).

        Called after step0 but before step2. Pipeline skips existing images.
        """
        import shutil

        reuse_raw = self.config.get("character_reuse", "")
        fixes_raw = self.config.get("character_fixes", "")
        reuse_map: dict = {}
        fixes_map: dict[str, str] = {}

        if reuse_raw:
            try:
                reuse_map = json.loads(reuse_raw) if isinstance(reuse_raw, str) else reuse_map
            except (json.JSONDecodeError, TypeError):
                pass
        if fixes_raw:
            try:
                fixes_map = json.loads(fixes_raw) if isinstance(fixes_raw, str) else fixes_raw
            except (json.JSONDecodeError, TypeError):
                pass

        structure = self.config.get("structure", "original")
        if structure == "quest":
            all_char_keys = ["char_a", "char_b", "char_c", "host"]
        elif structure == "original_cutout":
            # 独立主持人（未绑定角色时）同样参与复用
            all_char_keys = ["char_a", "char_b", "host", "narration"]
        else:
            all_char_keys = ["char_a", "char_b", "narration"]

        for k, v in list(reuse_map.items()):
            if v is True:
                reuse_map[k] = "image"
        if source_run and not reuse_map:
            reuse_map = {k: "image" for k in all_char_keys}

        overridden = []
        copied = []

        if source_run:
            output_dir = Path(self.config.get("output_dir", "./output"))
            source_dir = find_run_dir(output_dir, source_run)
            if not source_dir:
                self._on_log_line(f"  [Reuse] Source run not found: {source_run}")
            else:
                sp = source_dir / "script.json"
                if not sp.exists():
                    self._on_log_line(f"  [Reuse] No script.json in source run")
                else:
                    source_script = json.loads(sp.read_text(encoding="utf-8"))
                    src_img_dir = source_dir / "images"
                    dst_img_dir = dirs["images"]

                    # --- Mode "image": copy images + override desc + gender (NOT role) ---
                    image_keys = [k for k in all_char_keys if reuse_map.get(k) == "image"]
                    if (structure not in ("quest", "original_cutout")
                            and ("char_a" in image_keys) != ("char_b" in image_keys)):
                        # char_scene.png 是 A+B 合图：只选其一无法只复制单个角色
                        self._on_log_line(
                            "  [Reuse] WARNING: image 复用需角色A、B 同时选择"
                            "（char_scene 为双人合图），当前仅选其一时不会复制图片")
                    for key in image_keys:
                        for suffix in ["description", "gender", "qwen_speaker",
                                       "zh_voice", "moss_voice", "kokoro_voice"]:
                            field = f"{key}_{suffix}"
                            val = source_script.get(field, "")
                            if val:
                                script[field] = val
                                overridden.append(field)
                        if structure in ("quest", "original_cutout"):
                            for j in range(8):
                                src = src_img_dir / f"pose_{key}_{j}.png"
                                if src.exists():
                                    dst = dst_img_dir / src.name
                                    if not dst.exists():
                                        shutil.copy2(str(src), str(dst))
                                        copied.append(src.name)
                            # 闭嘴配对（quest 运行无此文件自动跳过）
                            for j in range(8):
                                src = src_img_dir / f"pose_{key}_{j}_c.png"
                                if src.exists():
                                    dst = dst_img_dir / src.name
                                    if not dst.exists():
                                        shutil.copy2(str(src), str(dst))
                                        copied.append(src.name)
                            atlas = src_img_dir / f"pose_atlas_{key}.png"
                            if atlas.exists():
                                dst = dst_img_dir / atlas.name
                                if not dst.exists():
                                    shutil.copy2(str(atlas), str(dst))
                                    copied.append(atlas.name)
                            if key == "host":
                                hb = src_img_dir / "host_bg.png"
                                if hb.exists():
                                    dst = dst_img_dir / "host_bg.png"
                                    if not dst.exists():
                                        shutil.copy2(str(hb), str(dst))
                                        copied.append("host_bg.png")
                        else:
                            if "char_a" in image_keys and "char_b" in image_keys:
                                for fname in self._CHAR_FILES_ORIGINAL:
                                    src = src_img_dir / fname
                                    if src.exists():
                                        dst = dst_img_dir / fname
                                        if not dst.exists():
                                            shutil.copy2(str(src), str(dst))
                                            copied.append(fname)
                                for f in src_img_dir.glob("pose_atlas_*.png"):
                                    dst = dst_img_dir / f.name
                                    if not dst.exists():
                                        shutil.copy2(str(f), str(dst))
                                        copied.append(f.name)
                                for f in src_img_dir.glob("pose_*_*.png"):
                                    if f.name.startswith("pose_atlas"):
                                        continue
                                    dst = dst_img_dir / f.name
                                    if not dst.exists():
                                        shutil.copy2(str(f), str(dst))
                                        copied.append(f.name)

                    # --- Mode "desc": override description + gender (NOT role, no images) ---
                    for key in all_char_keys:
                        if reuse_map.get(key) != "desc":
                            continue
                        for suffix in ["description", "gender", "qwen_speaker",
                                       "zh_voice", "moss_voice", "kokoro_voice"]:
                            val = source_script.get(f"{key}_{suffix}", "")
                            if val:
                                script[f"{key}_{suffix}"] = val
                                overridden.append(f"{key}_{suffix}")

                    # --- Mode "voice": override gender + qwen_speaker + moss_voice only
                    #     (fresh appearance every run — description stays LLM-generated) ---
                    for key in all_char_keys:
                        if reuse_map.get(key) != "voice":
                            continue
                        for suffix in ["gender", "qwen_speaker", "moss_voice",
                                       "kokoro_voice", "zh_voice"]:
                            val = source_script.get(f"{key}_{suffix}", "")
                            if val:
                                script[f"{key}_{suffix}"] = val
                                overridden.append(f"{key}_{suffix}")

        # --- Library characters: copy images + override desc + gender (NOT role) ---
        lib_raw = self.config.get("character_library", "")
        lib_map: dict[str, str] = {}
        if lib_raw:
            try:
                lib_map = json.loads(lib_raw) if isinstance(lib_raw, str) else lib_raw
            except (json.JSONDecodeError, TypeError):
                pass

        if lib_map:
            lib_base = Path(__file__).parent.parent / "configs" / "character_library"
            for key, lib_id in lib_map.items():
                if not lib_id or key not in all_char_keys:
                    continue
                lib_char_dir = lib_base / lib_id
                lib_meta_path = lib_char_dir / "meta.json"
                if not lib_meta_path.exists():
                    self._on_log_line(f"  [Library] {lib_id} not found")
                    continue
                lib_meta = json.loads(lib_meta_path.read_text(encoding="utf-8"))
                # Override description + gender（仅音色+性别角色 description 为空
                # → 只注入 gender + qwen_speaker，外观每次由 LLM 重新生成）
                for suffix in ["description", "gender", "qwen_speaker",
                               "zh_voice", "moss_voice", "kokoro_voice"]:
                    val = lib_meta.get(suffix, "")
                    if val:
                        script[f"{key}_{suffix}"] = val
                        overridden.append(f"{key}_{suffix}")
                # Copy images: library stores files as pose_{source_key}_*.png
                # Need to rename to target key (e.g. char_a_0.png → pose_char_a_0.png)
                src_key = lib_meta.get("source_key", key)
                lib_structure = lib_meta.get("structure", structure)
                dst_img_dir = dirs["images"]
                if lib_structure in ("quest", "original_cutout"):
                    for j in range(8):
                        src = lib_char_dir / f"pose_{src_key}_{j}.png"
                        if src.exists():
                            dst_name = f"pose_{key}_{j}.png"
                            dst = dst_img_dir / dst_name
                            if not dst.exists():
                                shutil.copy2(str(src), str(dst))
                                copied.append(dst_name)
                    # 闭嘴配对（quest 素材库无此文件自动跳过）
                    for j in range(8):
                        src = lib_char_dir / f"pose_{src_key}_{j}_c.png"
                        if src.exists():
                            dst_name = f"pose_{key}_{j}_c.png"
                            dst = dst_img_dir / dst_name
                            if not dst.exists():
                                shutil.copy2(str(src), str(dst))
                                copied.append(dst_name)
                    atlas = lib_char_dir / f"pose_atlas_{src_key}.png"
                    if atlas.exists():
                        dst_name = f"pose_atlas_{key}.png"
                        dst = dst_img_dir / dst_name
                        if not dst.exists():
                            shutil.copy2(str(atlas), str(dst))
                            copied.append(dst_name)
                    if key == "host":
                        hb = lib_char_dir / "host_bg.png"
                        if hb.exists():
                            dst = dst_img_dir / "host_bg.png"
                            if not dst.exists():
                                shutil.copy2(str(hb), str(dst))
                                copied.append("host_bg.png")
                else:
                    cs = lib_char_dir / "char_scene.png"
                    if cs.exists():
                        dst = dst_img_dir / "char_scene.png"
                        if not dst.exists():
                            shutil.copy2(str(cs), str(dst))
                            copied.append("char_scene.png")
                self._on_log_line(f"  [Library] {key} ← {lib_id} ({lib_meta.get('name', '')})")

        # --- character_fixes: custom description (no source needed) ---
        for key in all_char_keys:
            fix_desc = fixes_map.get(key, "").strip() if fixes_map else ""
            if not fix_desc:
                continue
            old_desc = script.get(f"{key}_description", "")
            script[f"{key}_description"] = fix_desc
            overridden.append(f"{key}_description")
            if reuse_map.get(key) != "image" and old_desc:
                for line in script.get("dialogue", []):
                    if line.get("speaker") != key:
                        continue
                    for pf in ("image_prompt", "video_prompt"):
                        old_val = line.get(pf, "")
                        if old_desc in old_val:
                            line[pf] = old_val.replace(old_desc, fix_desc)
                    poses = line.get("poses", [])
                    if poses:
                        line["poses"] = [
                            p.replace(old_desc, fix_desc) if old_desc in p else p
                            for p in poses
                        ]

        # --- character_voices: override qwen_speaker (highest priority) ---
        voices_raw = self.config.get("character_voices", "")
        voices_map: dict[str, str] = {}
        if voices_raw:
            try:
                voices_map = json.loads(voices_raw) if isinstance(voices_raw, str) else voices_raw
            except (json.JSONDecodeError, TypeError):
                pass
        for key in all_char_keys:
            voice = voices_map.get(key, "").strip()
            if voice:
                script[f"{key}_qwen_speaker"] = voice
                overridden.append(f"{key}_qwen_speaker")

        # --- character_zh_voices: override zh_voice (Chinese TTS voice binding) ---
        zh_voices_raw = self.config.get("character_zh_voices", "")
        zh_voices_map: dict[str, str] = {}
        if zh_voices_raw:
            try:
                zh_voices_map = json.loads(zh_voices_raw) if isinstance(zh_voices_raw, str) else zh_voices_raw
            except (json.JSONDecodeError, TypeError):
                pass
        for key in all_char_keys:
            zh_voice = zh_voices_map.get(key, "").strip()
            if zh_voice:
                script[f"{key}_zh_voice"] = zh_voice
                overridden.append(f"{key}_zh_voice")

        # --- character_moss_voices: override moss_voice (MOSS TTS voice binding) ---
        moss_voices_raw = self.config.get("character_moss_voices", "")
        moss_voices_map: dict[str, str] = {}
        if moss_voices_raw:
            try:
                moss_voices_map = json.loads(moss_voices_raw) if isinstance(moss_voices_raw, str) else moss_voices_raw
            except (json.JSONDecodeError, TypeError):
                pass
        for key in all_char_keys:
            moss_voice = moss_voices_map.get(key, "").strip()
            if moss_voice:
                script[f"{key}_moss_voice"] = moss_voice
                overridden.append(f"{key}_moss_voice")

        # --- character_kokoro_voices: override kokoro_voice (Kokoro TTS voice binding) ---
        kokoro_voices_raw = self.config.get("character_kokoro_voices", "")
        kokoro_voices_map: dict[str, str] = {}
        if kokoro_voices_raw:
            try:
                kokoro_voices_map = json.loads(kokoro_voices_raw) if isinstance(kokoro_voices_raw, str) else kokoro_voices_raw
            except (json.JSONDecodeError, TypeError):
                pass
        for key in all_char_keys:
            kokoro_voice = kokoro_voices_map.get(key, "").strip()
            if kokoro_voice:
                script[f"{key}_kokoro_voice"] = kokoro_voice
                overridden.append(f"{key}_kokoro_voice")

        script_path = work_dir / "script.json"
        script_path.write_text(
            json.dumps(script, ensure_ascii=False, indent=2), encoding="utf-8")

        self._on_log_line(f"  [Reuse] Copied {len(copied)} images from '{source_run or '(none)'}'")
        self._on_log_line(f"  [Reuse] Overridden: {', '.join(overridden)}")
        if copied:
            self._on_log_line(f"  [Reuse] Files: {', '.join(copied[:10])}")

    def _build_args(self, config: dict) -> SimpleNamespace:
        """Convert config dict → SimpleNamespace matching pipeline.py args."""
        # Resolve LLM provider (handles custom: prefix → openai)
        p_type, p_base_url, p_api_key, p_model = resolve_provider(config)

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

        # original_static: always use static images (no landing/stop_motion)
        animation = config.get("animation", "landing")
        if structure == "original_static":
            animation = "none"

        # tts_rate 为旧全局覆盖（兼容）；分项参数优先（tts_pipeline.resolve_tts_rate）
        tts_rate = config.get("tts_rate", "") or None

        tokens_raw = config.get("mcp_tokens", "").strip()
        mcp_tokens = ",".join(
            t.strip() for t in tokens_raw.split("\n") if t.strip()
        ) if tokens_raw else ""

        return SimpleNamespace(
            topic=config.get("topic", "") or None,
            cefr=config.get("cefr", "A2"),
            num_lines=num_lines,
            max_line_words=_cfg_int(config, "max_line_words", 10),
            structure=structure,
            animation=animation,
            visual_style=str(config.get("visual_style", "pixar3d")),
            llm_provider=config.get("llm_provider", "sensenova"),
            sensenova_api_key=p_api_key if p_type == "sensenova" else "",
            sensenova_model=p_model if p_type == "sensenova" else "deepseek-v4-flash",
            model=p_model if p_type == "sensenova" else "",
            api_key=p_api_key if p_type == "sensenova" else "",
            openai_base_url=p_base_url if p_type == "openai" else "",
            openai_api_key=p_api_key if p_type == "openai" else "",
            openai_model=p_model if p_type == "openai" else "grok-4.6",
            llm_retries=int(config.get("llm_retries", 10)),
            mcp_tokens=mcp_tokens or None,
            mcp_token=None,
            clip_duration=int(config.get("clip_duration", 15)),
            image_concurrency=int(config.get("image_concurrency", 4)),
            clip_concurrency=int(config.get("clip_concurrency", 4)),
            output=config.get("output_dir", "./output"),
            topics_file=config.get("topics_file", str(PIPELINE_DIR / "topics.json")),
            used_topics_file=config.get("used_topics_file", "") or None,
            lessons_dir=config.get("lessons_dir", "") or None,
            practice_duration=float(config.get("practice_duration", 3.0)),
            ch3_en_repeats=_cfg_int(config, "ch3_en_repeats", 3),
            ch3_zh_repeats=_cfg_int(config, "ch3_zh_repeats", 1),
            ch3_zh_always=bool(config.get("ch3_zh_always", True)),
            pad=pad,
            render_fps=int(config.get("render_fps", 8)),
            workers=int(config.get("workers", 1)),
            subtitle_font_size=int(config.get("subtitle_font_size", 60)),
            subtitle_style=str(config.get("subtitle_style", "") or ""),
            no_zh_subtitle=bool(config.get("no_zh_subtitle", False)),
            no_4k=bool(config.get("no_4k", False)),
            no_thumbnail=bool(config.get("no_thumbnail", False)),
            tts_engine=config.get("tts_engine", "kokoro"),
            tts_rate=tts_rate,
            tts_rate_en=config.get("tts_rate_en", "") or None,
            tts_rate_zh=config.get("tts_rate_zh", "") or None,
            tts_rate_narration=config.get("tts_rate_narration", "") or None,
            qwen_model_path=config.get("qwen_model_path", ""),
            qwen_base_model_path=config.get("qwen_base_model_path", ""),
            qwen_voicedesign_model_path=config.get("qwen_voicedesign_model_path", ""),
            qwen_device=config.get("qwen_device", ""),
            moss_model_path=config.get("moss_model_path", ""),
            moss_tokenizer_path=config.get("moss_tokenizer_path", ""),
            moss_device=config.get("moss_device", "cpu"),
            moss_repo_dir=config.get("moss_repo_dir", ""),
            moss_tts_temperature=config.get("moss_tts_temperature", 0.8),
            moss_tts_retry=config.get("moss_tts_retry", 3),
            moss_tts_top_p=config.get("moss_tts_top_p", 0.95),
            moss_tts_top_k=config.get("moss_tts_top_k", 25),
            moss_tts_rep_penalty=config.get("moss_tts_rep_penalty", 1.2),
            moss_tts_text_temperature=config.get("moss_tts_text_temperature", 1.0),
            moss_tts_greedy=bool(config.get("moss_tts_greedy", False)),
            upscale_timeout=int(config.get("upscale_timeout", 3600)),
            upscale_engine=str(config.get("upscale_engine", "ffmpeg")),
            matting_engine=str(config.get("matting_engine", "auto")),
            host_character=str(config.get("host_character", "") or ""),
            resume=False,
        )

    def _seed_from_script_library(self, script_id: str, args, topic: str,
                                  parent_dir: Path, used_topics_file: str):
        """Step 0 alternative: seed the run dir from a library script (no LLM).

        Writes script.json + subdirs + checkpoint, marks the script USED.
        Returns (script, work_dir, dirs); (None, None, None) after _fail().
        """
        from . import script_library

        doc = script_library.get_script_doc(script_id)
        if not doc:
            self._fail(f"脚本库中未找到脚本: {script_id}")
            return None, None, None
        script = doc.get("script") or {}
        if not script.get("dialogue"):
            self._fail(f"脚本库脚本无对话内容: {script_id}")
            return None, None, None
        if doc.get("structure") and doc["structure"] != args.structure:
            self._fail(
                f"脚本结构不匹配: 脚本为 {doc['structure']}，当前模式为 {args.structure}")
            return None, None, None

        topic = doc.get("topic") or script.get("title") or topic
        args.topic = topic
        if doc.get("cefr"):
            args.cefr = doc["cefr"]

        self._on_log_line("\n" + "=" * 60)
        self._on_log_line("Step 0: 使用脚本库脚本（跳过 LLM 生成）...")
        yt_title = script.get("youtube_title", script.get("title", topic))
        safe_title = _safe_dirname(yt_title, topic)
        work_dir = parent_dir / args.structure / safe_title
        work_dir.mkdir(parents=True, exist_ok=True)
        dirs = {k: work_dir / k for k in ("images", "clips", "audio", "subtitles", "videos")}
        for d in dirs.values():
            d.mkdir(parents=True, exist_ok=True)
        script_path = work_dir / "script.json"
        script["structure"] = args.structure
        script_path.write_text(
            json.dumps(script, ensure_ascii=False, indent=2), encoding="utf-8")
        _save_checkpoint(work_dir, "step0_script", topic=topic, cefr=args.cefr,
                         structure=args.structure, animation=args.animation,
                         visual_style=str(self.config.get("visual_style", "")),
                         host_character=str(self.config.get("host_character", "") or ""))
        # 各模式独立记录已用主题（不再写全局 used_topics.json，
        # 同一主题仍可在其他模式生成/使用）
        script_library.mark_topic_used_mode(
            args.structure, topic, script_id=script_id, run_name=safe_title)
        script_library.mark_used(script_id, run_name=safe_title)
        review = doc.get("review") or {}
        score_info = (f"，审查 {review.get('score')} 分"
                      if review.get("score") is not None else "")
        self._on_log_line(f"  [ScriptLib] {topic} "
                          f"({len(script.get('dialogue', []))} 行{score_info})")
        self._on_log_line(f"  [ScriptLib] 已标记为「已使用」(run: {safe_title})")
        self._on_log_line(f"  Script saved: {script_path}")
        return script, work_dir, dirs

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
        # 无论 _run 主体在哪一步抛异常（含 try 块之前的 env/args 构建），
        # 都必须释放互斥锁，否则模式测试/下次启动会被永久阻塞
        try:
            self._run_inner(config, resume)
        finally:
            run_mutex.release("pipeline")

    def _run_inner(self, config: dict, resume: bool):
        """Main pipeline execution body (mutex held by _run)."""
        # Import here so heavy imports happen in the worker thread
        from pipeline import (
            _step0_script, _step1_mcp, _step2_images_tts,
            _step3_clips, _step4_timeline, _step45_thumbnail,
            _step5_compose, _step6_4k,
            _generate_script_with_retry, _resolve_topic, _resolve_run_dir,
        )

        # Set env vars
        self._set_env(config)

        # Inject character overrides into env so LLM prompt includes them
        overrides = self._build_character_overrides(config)
        if overrides:
            os.environ["CHARACTER_OVERRIDES"] = json.dumps(overrides, ensure_ascii=False)
            self._on_log_line(f"  [CharOverride] {len(overrides)} characters pre-defined for LLM: "
                              f"{', '.join(overrides.keys())}")
        else:
            os.environ.pop("CHARACTER_OVERRIDES", None)

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
            # 新布局：每种模式一个独立文件夹 output/{mode}/{run_name}/
            mode_dir = parent_dir / args.structure
            mode_dir.mkdir(parents=True, exist_ok=True)
            # Full raw LLM responses are dumped here when _chat hits errors
            os.environ["LLM_DEBUG_DIR"] = str(parent_dir / "llm_debug")
            used_topics_file = args.used_topics_file or str(parent_dir / "used_topics.json")

            # Load checkpoint for resume
            if resume:
                checkpoint = _load_checkpoint(mode_dir)
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

            # Step 0: Script generation — 脚本库脚本直接落盘（跳过 LLM 生成）
            script_id = str(config.get("script_id") or "").strip()
            if script_id and not resume:
                script, work_dir, dirs = self._seed_from_script_library(
                    script_id, args, topic, parent_dir, used_topics_file)
                if script is None:
                    return
            else:
                script, work_dir, dirs = _step0_script(
                    args, checkpoint, topic, mode_dir, used_topics_file)
                # 全新生成也补记「模式已用」（脚本页按模式排除主题用）
                try:
                    from . import script_library as _sl
                    _sl.mark_topic_used_mode(
                        args.structure, topic, run_name=Path(work_dir).name)
                except Exception:
                    pass
            self.work_dir = str(work_dir)

            # --- Character reuse: copy images + override descriptions ---
            char_source = config.get("character_source", "")
            char_fixes = config.get("character_fixes", "")
            char_lib = config.get("character_library", "")
            if (char_source or char_fixes or char_lib) and not resume:
                self._reuse_characters(char_source, work_dir, script, dirs)

            if self._stop_flag.is_set():
                self._set_stopped()
                return
            self._wait_for_step_approval("step0_script")
            if self._stop_flag.is_set():
                self._set_stopped()
                return

            # Reload script in case user edited it during step-mode pause
            _script_path = work_dir / "script.json"
            if _script_path.exists():
                script = json.loads(_script_path.read_text(encoding="utf-8"))
                self._on_log_line("  [Step Mode] Reloaded script.json (edits applied).")

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
            ctx = _step2_images_tts(args, checkpoint, script, work_dir, dirs,
                                    stop_check=self._stop_flag.is_set)

            if self._stop_flag.is_set():
                self._set_stopped()
                return
            self._wait_for_step_approval("step2_images_tts")
            if self._stop_flag.is_set():
                self._set_stopped()
                return

            # Step 3: Video clips
            clip_paths, group_info, line_to_group = _step3_clips(
                args, checkpoint, work_dir, dirs, script, ctx,
                stop_check=self._stop_flag.is_set)

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
                group_info, line_to_group, stop_check=self._stop_flag.is_set)
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

        except SystemExit as e:
            # pipeline 模块用 sys.exit(1) 中止（图片缺失 / MCP token 耗尽 /
            # 质检门禁 exit=2 等）。SystemExit 不是 Exception 子类，
            # except Exception 接不住 — 不显式处理的话 finally 兜底会把
            # running 误标成 done（前端弹"视频生成完成"）。
            self._fail(f"Pipeline aborted (exit code {e.code})")
        except Exception as e:
            self._fail(f"{type(e).__name__}: {e}")
            import traceback
            tb = traceback.format_exc()
            for line in tb.split("\n"):
                self._on_log_line(line)
        finally:
            sys.stdout = old_stdout
            buf.flush()
            os.environ.pop("CHARACTER_OVERRIDES", None)
            with self._lock:
                if self.status == "running":
                    # 正常完成路径在上面已显式置 done；走到这里仍 running
                    # 说明线程异常退出 — 一律标记失败，绝不静默显示为完成
                    self.status = "error"
                    self.error = "Pipeline thread exited without a terminal status"
                self.finished_at = time.time()

    def _fail(self, msg: str):
        with self._lock:
            self.status = "error"
            self.error = msg
            self.finished_at = time.time()
        self._on_log_line(f"\nFATAL: {msg}")

    # ------------------------------------------------------------------
    # Recompose: re-burn subtitles with a new style on a finished run
    # ------------------------------------------------------------------

    @staticmethod
    def _probe_fps(path: str) -> int:
        """ffprobe 视频帧率（quest=25 / 其余=24），失败回退 24。"""
        import subprocess as _sp
        try:
            r = _sp.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=r_frame_rate",
                 "-of", "default=nw=1:nk=1", path],
                capture_output=True, text=True, timeout=30)
            s = (r.stdout or "").strip().splitlines()[0] if r.stdout else ""
            val = 0.0
            if "/" in s:
                num, den = s.split("/", 1)
                val = float(num) / float(den) if float(den) else 0.0
            elif s:
                val = float(s)
            if val > 0:
                return int(round(val))
        except Exception:
            pass
        return 24

    def refresh_youtube_metadata(self, run_name: str, mode: str = "") -> tuple[bool, str]:
        """用 script.json + subtitles/meta.json 的 timeline 重新生成 youtube_metadata.json。

        复用 Step 4.5 的 save_youtube_metadata，零 AI 成本秒级完成，
        用于脚本编辑后刷新标题/简介/章节。返回 (ok, message)。
        """
        if self.is_running:
            return False, "Pipeline 正在运行中，请等待完成后再刷新"

        output_dir = Path(load_config().get("output_dir", "./output"))
        run_dir = find_run_dir(output_dir, run_name, mode)
        if not run_dir:
            return False, f"运行不存在: {run_name}"
        script_path = run_dir / "script.json"
        meta_path = run_dir / "subtitles" / "meta.json"
        if not script_path.exists():
            return False, "缺少 script.json"
        if not meta_path.exists():
            return False, "缺少 subtitles/meta.json（请先跑完 Step 4）"
        try:
            script = json.loads(script_path.read_text(encoding="utf-8"))
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            from thumbnail_gen import save_youtube_metadata
            out = save_youtube_metadata(
                script=script,
                timeline=meta.get("timeline", []),
                output_path=str(run_dir / "youtube_metadata.json"),
                structure=script.get("structure", "original"),
            )
            return True, f"已重新生成: {Path(out).name}"
        except Exception as e:
            return False, f"生成失败: {e}"

    def regenerate_thumbnail(self, run_name: str, mode: str = "") -> tuple[bool, str]:
        """为已有运行再生成一张缩略图（thumbnail_N.jpg 递增，旧图全部保留）。

        复用 thumbnail_gen.generate_thumbnail 与当前配置的生图 Provider
        （mcp / sensenova），角色参考图从本地 images/ 取。照 recompose 模式
        在后台线程执行；期间 is_running=True 与主 pipeline / 模式测试互斥。
        返回 (ok, message)。
        """
        if self.is_running:
            return False, "Pipeline 正在运行中，请等待完成后再生成"

        config = load_config()
        output_dir = Path(config.get("output_dir", "./output"))
        run_dir = find_run_dir(output_dir, run_name, mode)
        if not run_dir:
            return False, f"运行不存在: {run_name}"
        script_path = run_dir / "script.json"
        if not script_path.exists():
            return False, "缺少 script.json"
        try:
            script = json.loads(script_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            return False, f"script.json 读取失败: {e}"

        structure = script.get("structure", "")
        if structure not in MODES:
            structure = run_dir.parent.name if run_dir.parent.name in MODES else "original"

        # env 注入（_set_env 的最小子集）：生图 Provider / SenseNova key / 画面风格
        # （thumbnail_gen._build_thumbnail_prompt 经 style_manager 读后两个 env）
        os.environ["IMAGE_PROVIDER"] = str(config.get("image_provider", "mcp"))
        if str(config.get("sensenova_api_key") or "").strip():
            os.environ["SENSENOVA_API_KEY"] = str(config["sensenova_api_key"]).strip()
        from style_manager import resolve_style_prompt
        style_id = str(config.get("visual_style", "pixar3d"))
        os.environ["VISUAL_STYLE_ID"] = style_id
        os.environ["VISUAL_STYLE_PROMPT"] = resolve_style_prompt(style_id)

        # 角色参考图本地路径（线程内再决定转 CDN URL 还是 base64）
        if structure in ("quest", "original_cutout"):
            ref_img = run_dir / "images" / "pose_char_a_0.png"
        else:
            ref_img = run_dir / "images" / "char_scene.png"

        # 输出文件名：无主图时补 thumbnail.jpg，否则 thumbnail_N.jpg 递增（旧图全保留）
        if not (run_dir / "thumbnail.jpg").exists():
            out_name = "thumbnail.jpg"
        else:
            n = 2
            while (run_dir / f"thumbnail_{n}.jpg").exists():
                n += 1
            out_name = f"thumbnail_{n}.jpg"

        if not run_mutex.try_acquire("thumbnail_regen"):
            return False, (f"资源被占用：{run_mutex.current_owner()}"
                           f"（主 pipeline / 模式测试运行中请等待完成）")

        self._stop_flag.clear()
        with self._lock:
            self.log_lines = []
            self.status = "running"
            self.current_step = ""
            self.current_step_label = "缩略图重生成"
            self.started_at = time.time()
            self.finished_at = 0
            self.error = ""
            self.work_dir = str(run_dir)
            self.final_path = ""

        self._thread = threading.Thread(
            target=self._thumbnail_regen_run,
            args=(run_dir, script, structure, out_name,
                  str(ref_img) if ref_img.exists() else ""),
            daemon=True)
        self._thread.start()
        return True, f"缩略图生成已启动（将保存为 {out_name}）"

    def _thumbnail_regen_run(self, run_dir: Path, script: dict, structure: str,
                             out_name: str, ref_img_path: str):
        """后台线程：复用 Step 4.5 的 generate_thumbnail 生成一张新缩略图。"""
        old_stdout = sys.stdout
        buf = _LineBuffer(self._on_log_line)
        sys.stdout = buf
        try:
            print("=" * 60)
            print(f"ThumbnailRegen: {run_dir.name} -> {out_name}")
            provider = os.environ.get("IMAGE_PROVIDER", "mcp")
            if provider == "mcp":
                mcp_initialize()
            char_scene_url = ""
            if ref_img_path:
                if provider == "sensenova":
                    # sensenova edit_image 内部 _to_image_url 把本地路径转 base64
                    char_scene_url = ref_img_path
                else:
                    from image_gen import reupload_for_cdn
                    try:
                        char_scene_url = reupload_for_cdn(
                            ref_img_path, Path(ref_img_path).name) or ""
                    except Exception as e:
                        print(f"[ThumbnailRegen] 参考图上传失败，退回无参考图生成: {e}")

            from pipeline import call_tool, parse_task_id, poll_task, download_file
            from thumbnail_gen import generate_thumbnail

            # 场景图：仅 Pillow 兜底分支需要；AI 分支纯 prompt 生成不受影响
            if structure == "quest":
                scene_img = run_dir / "images" / "scene_0.png"
                if not scene_img.exists():
                    scene_img = run_dir / "images" / "scene.png"
            else:
                scene_img = run_dir / "images" / "scene.png"

            out_path = generate_thumbnail(
                script=script,
                scene_img=str(scene_img),
                output_path=str(run_dir / out_name),
                mcp_call_tool=call_tool,
                mcp_parse_task_id=parse_task_id,
                mcp_poll_task=poll_task,
                mcp_download_file=download_file,
                structure=structure,
                char_scene_url=char_scene_url,
            )
            if out_path and Path(out_path).exists():
                with self._lock:
                    self.status = "done"
                    self.finished_at = time.time()
                print("=" * 60)
                print(f"ThumbnailRegen DONE! {out_name}")
            else:
                self._fail("缩略图生成失败：AI 生成与 Pillow 兜底均未产出文件"
                           "（检查生图 Provider 配置 / MCP token / images 场景图）")
        except Exception as e:
            self._fail(f"ThumbnailRegen {type(e).__name__}: {e}")
            import traceback
            for line in traceback.format_exc().split("\n"):
                self._on_log_line(line)
        finally:
            sys.stdout = old_stdout
            buf.flush()
            run_mutex.release("thumbnail_regen")

    def recompose(self, run_name: str, subtitle_style: str = "", font_size: int = 60,
                  show_zh: bool = True, regen_4k: bool = False,
                  mode: str = "") -> tuple[bool, str]:
        """对已完成运行重选字幕样式重渲视频（仅字幕烧录 + 音量归一，本地渲染）。

        复用 videos/final_no_sub.mp4 + subtitles/meta.json + script.json，
        不消耗 MCP / LLM / TTS 积分。返回 (ok, message)。
        """
        if self.is_running:
            return False, "Pipeline 正在运行中，请等待完成后再重渲"

        output_dir = Path(load_config().get("output_dir", "./output"))
        run_dir = find_run_dir(output_dir, run_name, mode)
        if not run_dir:
            return False, f"运行不存在: {run_name}"
        no_sub = run_dir / "videos" / "final_no_sub.mp4"
        meta_path = run_dir / "subtitles" / "meta.json"
        script_path = run_dir / "script.json"
        if not no_sub.exists() or no_sub.stat().st_size < 1_000_000:
            return False, "缺少 videos/final_no_sub.mp4（旧运行或已清理，无法仅重渲字幕）"
        if not meta_path.exists():
            return False, "缺少 subtitles/meta.json（时间轴数据缺失）"
        if not script_path.exists():
            return False, "缺少 script.json"

        self._stop_flag.clear()
        self._step_mode = False
        self._paused_after_step = ""
        with self._lock:
            self.log_lines = []
            self.status = "running"
            self.current_step = ""
            self.current_step_label = "字幕样式重渲（仅本地渲染）"
            self.started_at = time.time()
            self.finished_at = 0
            self.error = ""
            self.work_dir = str(run_dir)
            self.final_path = ""

        self._thread = threading.Thread(
            target=self._recompose_run,
            args=(run_dir, subtitle_style, int(font_size), bool(show_zh), bool(regen_4k)),
            daemon=True)
        self._thread.start()
        return True, "重渲已启动"

    def _recompose_run(self, run_dir: Path, subtitle_style_id: str,
                       font_size: int, show_zh: bool, regen_4k: bool):
        """后台线程：从 final_no_sub 重烧字幕 → loudnorm → 处理 4K。"""
        import subprocess as _sp
        from media_utils import burn_subtitles, apply_final_loudnorm

        old_stdout = sys.stdout
        buf = _LineBuffer(self._on_log_line)
        sys.stdout = buf
        try:
            print("=" * 60)
            print(f"Recompose: 字幕样式重渲 — {run_dir.name}")
            style = None
            if subtitle_style_id:
                from subtitle_style_manager import get_style as _get_sub_style
                style = _get_sub_style(subtitle_style_id)
                if style is None:
                    print(f"  [Recompose] 未找到样式 '{subtitle_style_id}'，回退 legacy 字号参数")
                else:
                    print(f"  [Recompose] 字幕样式: {style.get('name', subtitle_style_id)}")
            if style is None:
                print(f"  [Recompose] 跟随参数配置（legacy 字号 {font_size}）")

            script = json.loads((run_dir / "script.json").read_text(encoding="utf-8"))
            meta = json.loads((run_dir / "subtitles" / "meta.json").read_text(encoding="utf-8"))
            timeline = meta["timeline"]
            pad = float(meta.get("pad", 0.4))
            no_sub = str(run_dir / "videos" / "final_no_sub.mp4")
            out_fps = self._probe_fps(no_sub)
            print(f"  [Recompose] timeline {len(timeline)} 段, pad={pad}, fps={out_fps}")

            def progress_cb(pct, msg):
                print(f"  [{pct}%] {msg}")

            final_path = burn_subtitles(
                no_sub, timeline, script, str(run_dir), str(run_dir / "subtitles"),
                pad, progress_cb,
                show_zh=show_zh, en_font_size=font_size,
                zh_font_size=int(font_size * 0.85),
                out_fps=out_fps, style=style)
            self.final_path = final_path

            print("  [Recompose] 音量归一化 (loudnorm)...")
            apply_final_loudnorm(final_path, str(run_dir / "videos"))

            # 旧 4K 的字幕已过期：删除；按需用新片重新生成
            old_4k = sorted(run_dir.glob("*_4K.mp4"))
            for p in old_4k:
                p.unlink(missing_ok=True)
                print(f"  [Recompose] 已删除过期 4K: {p.name}")
            if regen_4k:
                four_k = run_dir / f"{Path(final_path).stem}_4K.mp4"
                print("  [Recompose] 重新生成 4K (本地 ffmpeg scale)...")
                r = _sp.run(
                    ["ffmpeg", "-i", final_path,
                     "-vf", "scale=3840:2160:flags=lanczos",
                     "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-threads", "0",
                     "-c:a", "copy", str(four_k), "-y"],
                    capture_output=True, timeout=3600)
                if r.returncode == 0 and four_k.exists():
                    print(f"  [Recompose] 4K 完成: {four_k.name}")
                else:
                    print("  [Recompose] 4K 生成失败（720p 版本仍可用）")
                    four_k.unlink(missing_ok=True)

            with self._lock:
                self.status = "done"
                self.finished_at = time.time()
            size_mb = Path(final_path).stat().st_size / (1024 * 1024)
            print("=" * 60)
            print(f"Recompose DONE! {final_path} ({size_mb:.1f}MB)")
        except Exception as e:
            self._fail(f"Recompose {type(e).__name__}: {e}")
            import traceback
            for line in traceback.format_exc().split("\n"):
                self._on_log_line(line)
        finally:
            sys.stdout = old_stdout
            buf.flush()
            with self._lock:
                if self.status == "running":
                    self.status = "done"
                self.finished_at = time.time()


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
