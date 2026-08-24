"""Configuration manager: load/save JSON configs, preset management,
and build CLI args for pipeline.py."""
import json
import os
from pathlib import Path
from typing import Any

# Resolve paths
WEB_ROOT = Path(__file__).parent.parent.resolve()
PIPELINE_DIR = Path(__file__).parent.parent.parent / "colab_listening_b"
CONFIGS_DIR = WEB_ROOT / "configs"
DEFAULT_CONFIG_PATH = CONFIGS_DIR / "default.json"

# All configurable parameters with defaults, types, and metadata
PARAM_SPEC = {
    # --- Content ---
    "topic": {"default": "", "type": "text", "group": "content",
              "label": "主题", "help": "留空则从主题库随机选择"},
    "cefr": {"default": "A2", "type": "select", "group": "content",
             "label": "CEFR 等级", "options": ["A1", "A2", "B1", "B2", "C1", "C2"]},
    "num_lines": {"default": "", "type": "number", "group": "content",
                  "label": "对话行数", "help": "留空=自动 (original:18, quest:48)"},
    "structure": {"default": "original", "type": "select", "group": "content",
                  "label": "视频结构", "options": {
                      "original": "Original (4章视频片段)",
                      "image": "Image (纯图片+动画)",
                      "quest": "Quest (任务听力)"}},
    "animation": {"default": "landing", "type": "select", "group": "content",
                  "label": "动画类型 (image模式)", "options": {
                      "none": "None (静态)",
                      "landing": "Landing (降落变换)",
                      "stop_motion": "Stop Motion (定格动画)"}},
    "character_source": {"default": "", "type": "text", "group": "content",
                         "label": "复用角色来源", "help": "留空=全部新生成。选择之前运行可复用其角色"},
    "character_reuse": {"default": "", "type": "text", "group": "content",
                        "label": "复用角色选择", "help": "选择哪些角色从来源复用 (JSON)"},
    "character_fixes": {"default": "", "type": "text", "group": "content",
                        "label": "固定角色描述", "help": "手动指定角色外观描述 (JSON)"},
    "character_library": {"default": "", "type": "text", "group": "content",
                           "label": "素材库分配", "help": "JSON: {char_a: 'lib_id', char_b: 'lib_id'}"},
    "character_voices": {"default": "", "type": "text", "group": "content",
                          "label": "角色音色绑定", "help": "JSON: {char_a: 'Vivian', char_b: 'Ryan'} — 复用角色时绑定 Qwen TTS 音色"},

    # --- LLM ---
    "llm_provider": {"default": "sensenova", "type": "select", "group": "llm",
                     "label": "LLM Provider", "options": {
                         "sensenova": "SenseNova",
                         "openai": "OpenAI Compatible"}},
    "sensenova_api_key": {"default": "", "type": "password", "group": "llm",
                          "label": "SenseNova API Key"},
    "sensenova_model": {"default": "deepseek-v4-flash", "type": "select", "group": "llm",
                        "label": "SenseNova Model", "options": ["deepseek-v4-flash", "glm-5.2"]},
    "openai_base_url": {"default": "https://x666.me/v1", "type": "text", "group": "llm",
                        "label": "OpenAI Base URL"},
    "openai_api_key": {"default": "", "type": "password", "group": "llm",
                      "label": "OpenAI API Key"},
    "openai_model": {"default": "grok-4.6", "type": "select", "group": "llm",
                    "label": "OpenAI Model", "options": [
                        "grok-4.6", "grok-4.5", "gemini-3.1-pro-preview",
                        "gemini-3.7-flash", "claude-sonnet-5", "gemini-2.5-pro-1m"]},
    "llm_retries": {"default": 10, "type": "number", "group": "llm",
                    "label": "LLM 重试次数"},
    "llm_min_interval": {"default": 3, "type": "number", "group": "llm",
                         "label": "LLM 最小间隔(秒)"},
    "quest_beat_lines": {"default": 10, "type": "number", "group": "llm",
                         "label": "Quest 节拍行数", "help": "节拍表每拍的行数预算 (默认10)"},
    "quest_qa_rounds": {"default": 3, "type": "number", "group": "llm",
                        "label": "Quest QA 轮数", "help": "自评-修复循环最大轮数 (默认3, 0=关闭)"},

    # --- TTS ---
    "tts_engine": {"default": "kokoro", "type": "select", "group": "tts",
                   "label": "TTS 引擎", "options": {
                       "kokoro": "Kokoro (本地)",
                       "voxcpm": "VoxCPM (Cloudflare Worker)",
                       "qwen": "Qwen3-TTS (本地 GPU)"}},
    "tts_rate": {"default": "", "type": "text", "group": "tts",
                 "label": "TTS 语速", "help": "如 -15%, 0% (留空=模式默认)"},
    "voxcpm_worker_url": {"default": "https://curly-grass-9b8c.caimeifeng3.workers.dev",
                          "type": "text", "group": "tts", "label": "VoxCPM Worker URL"},
    "voxcpm_api_key": {"default": "", "type": "password", "group": "tts",
                       "label": "VoxCPM API Key"},
    "qwen_model_path": {"default": r"H:\models\Qwen3-TTS-12Hz-0.6B-CustomVoice",
                        "type": "text", "group": "tts",
                        "label": "Qwen3-TTS 模型路径",
                        "help": "CustomVoice 模型 (预设音色)"},
    "qwen_base_model_path": {"default": r"H:\models\Qwen3-TTS-12Hz-1.7B-Base",
                             "type": "text", "group": "tts",
                             "label": "Qwen3-TTS Base 模型路径",
                             "help": "Base 模型 (voice clone 需要)"},
    "qwen_device": {"default": "cuda:0", "type": "text", "group": "tts",
                    "label": "Qwen3-TTS 设备", "help": "如 cuda:0, cpu"},

    # --- MCP / Image ---
    "mcp_tokens": {"default": "", "type": "textarea", "group": "mcp",
                   "label": "MCP Tokens", "help": "每行一个 token, 多 token 自动轮换"},
    "image_concurrency": {"default": 4, "type": "number", "group": "mcp",
                          "label": "图片并发数", "help": "1-4, 默认4"},
    "clip_duration": {"default": 15, "type": "number", "group": "mcp",
                      "label": "视频片段时长(秒)", "help": "4-15"},
    "clip_concurrency": {"default": 4, "type": "number", "group": "mcp",
                          "label": "视频并发数", "help": "1-5, 默认4 (MCP最大并发5)"},
    "output_dir": {"default": str(PIPELINE_DIR / "output"), "type": "text", "group": "mcp",
                   "label": "输出目录"},
    "topics_file": {"default": str(PIPELINE_DIR / "topics.json"), "type": "text", "group": "mcp",
                    "label": "主题库文件"},
    "used_topics_file": {"default": "", "type": "text", "group": "mcp",
                         "label": "已用主题文件", "help": "留空=<output>/used_topics.json"},
    "lessons_dir": {"default": "", "type": "text", "group": "mcp",
                    "label": "防重复目录", "help": "留空=不检查"},

    # --- Video ---
    "practice_duration": {"default": 3.0, "type": "number", "group": "video",
                          "label": "练习间隔(秒)"},
    "pad": {"default": "", "type": "text", "group": "video",
            "label": "音频间隔(秒)", "help": "留空=自动 (0.4)"},
    "render_fps": {"default": 8, "type": "number", "group": "video",
                   "label": "渲染帧率 (stop_motion)"},
    "workers": {"default": 1, "type": "number", "group": "video",
                "label": "渲染线程数", "help": "0=自动(CPU核数)"},
    "subtitle_font_size": {"default": 60, "type": "number", "group": "video",
                          "label": "字幕字体大小"},
    "no_zh_subtitle": {"default": False, "type": "checkbox", "group": "video",
                       "label": "隐藏中文字幕"},
    "no_4k": {"default": False, "type": "checkbox", "group": "video",
              "label": "跳过4K"},
    "upscale_timeout": {"default": 3600, "type": "number", "group": "video",
                       "label": "4K超时(秒)"},
}

# Group display metadata
GROUP_META = {
    "content": {"label": "内容设置", "icon": "📝", "order": 1},
    "llm": {"label": "LLM 设置", "icon": "🤖", "order": 2},
    "tts": {"label": "TTS 语音", "icon": "🎙️", "order": 3},
    "mcp": {"label": "MCP / 图片", "icon": "🎨", "order": 4},
    "video": {"label": "视频合成", "icon": "🎬", "order": 5},
}


# --- Per-mode config storage ---
# 3 种视频结构模式各一份完整独立配置，切换互不覆盖。
# default.json 仅作首次迁移源；active_mode.json 记录当前激活模式。

MODES = ["original", "image", "quest"]
MODE_LABELS = {
    "original": "Original (4章视频片段)",
    "image": "Image (纯图片+动画)",
    "quest": "Quest (任务听力)",
}
ACTIVE_MODE_PATH = CONFIGS_DIR / "active_mode.json"


def _mode_config_path(mode: str) -> Path:
    return CONFIGS_DIR / f"mode_{mode}.json"


def _read_legacy_default() -> dict[str, Any] | None:
    """Read legacy default.json (migration source only)."""
    if DEFAULT_CONFIG_PATH.exists():
        try:
            return json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
    return None


def get_active_mode() -> str:
    if ACTIVE_MODE_PATH.exists():
        try:
            mode = json.loads(ACTIVE_MODE_PATH.read_text(encoding="utf-8")).get("mode", "original")
            if mode in MODES:
                return mode
        except (json.JSONDecodeError, OSError):
            pass
    return "original"


def set_active_mode(mode: str) -> str:
    if mode not in MODES:
        raise ValueError(f"未知模式: {mode}")
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    ACTIVE_MODE_PATH.write_text(
        json.dumps({"mode": mode}, ensure_ascii=False, indent=2), encoding="utf-8")
    return mode


def load_mode_config(mode: str) -> dict[str, Any]:
    """Load full config for a mode; lazily initialize from legacy default.json."""
    if mode not in MODES:
        raise ValueError(f"未知模式: {mode}")
    path = _mode_config_path(mode)
    if path.exists():
        try:
            saved = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            saved = None
    else:
        saved = None
    if saved is None:
        # 首次：继承现有全局配置（或出厂默认）
        base = _read_legacy_default() or get_default_config()
        saved = dict(base)
    merged = get_default_config()
    merged.update(saved)
    merged["structure"] = mode  # 结构由模式文件决定，恒等于文件名
    save_mode_config(mode, merged)
    return merged


def save_mode_config(mode: str, config: dict[str, Any]) -> None:
    if mode not in MODES:
        raise ValueError(f"未知模式: {mode}")
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    config = dict(config)
    config["structure"] = mode  # 防止串模式
    _mode_config_path(mode).write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


def load_all_mode_configs() -> dict[str, dict[str, Any]]:
    return {mode: load_mode_config(mode) for mode in MODES}


def get_default_config() -> dict[str, Any]:
    return {k: v["default"] for k, v in PARAM_SPEC.items()}


def load_config() -> dict[str, Any]:
    """Current active mode's config (all pages follow the active mode)."""
    return load_mode_config(get_active_mode())


def save_config(config: dict[str, Any]) -> None:
    save_mode_config(get_active_mode(), config)


def list_presets() -> list[str]:
    if not CONFIGS_DIR.exists():
        return []
    return sorted([
        f.stem for f in CONFIGS_DIR.glob("*.json")
        if f.name != "default.json"
    ])


def save_preset(name: str, config: dict[str, Any]) -> None:
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(c for c in name if c.isalnum() or c in "-_") or "preset"
    path = CONFIGS_DIR / f"{safe_name}.json"
    path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


def load_preset(name: str) -> dict[str, Any]:
    path = CONFIGS_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Preset '{name}' not found")
    return json.loads(path.read_text(encoding="utf-8"))


def delete_preset(name: str) -> None:
    path = CONFIGS_DIR / f"{name}.json"
    if path.exists():
        path.unlink()


# --- Custom LLM Providers ---

LLM_PROVIDERS_PATH = CONFIGS_DIR / "llm_providers.json"


def load_llm_providers() -> list[dict]:
    """Load custom LLM providers from configs/llm_providers.json."""
    if not LLM_PROVIDERS_PATH.exists():
        return []
    try:
        data = json.loads(LLM_PROVIDERS_PATH.read_text(encoding="utf-8"))
        return data.get("providers", [])
    except (json.JSONDecodeError, OSError):
        return []


def save_llm_providers(providers: list[dict]) -> None:
    LLM_PROVIDERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    LLM_PROVIDERS_PATH.write_text(
        json.dumps({"providers": providers}, ensure_ascii=False, indent=2),
        encoding="utf-8")


def get_provider_options() -> dict[str, str]:
    """Return all LLM provider options (static + custom) as {value: label}."""
    options = {
        "sensenova": "SenseNova",
        "openai": "OpenAI Compatible (内置)",
    }
    custom = load_llm_providers()
    for p in custom:
        options[f"custom:{p['id']}"] = p["name"]
    return options


def resolve_provider(config: dict[str, Any]) -> tuple[str, str, str, str]:
    """Resolve LLM provider config → (provider_type, base_url, api_key, model).

    For custom:* providers, reads from llm_providers.json.
    Returns provider_type as 'sensenova' or 'openai' (custom always → openai).
    """
    provider = config.get("llm_provider", "sensenova")
    if provider == "sensenova":
        return (
            "sensenova",
            "https://token.sensenova.cn/v1",
            config.get("sensenova_api_key", ""),
            config.get("sensenova_model", "deepseek-v4-flash"),
        )
    elif provider.startswith("custom:"):
        custom_id = provider.split(":", 1)[1]
        customs = load_llm_providers()
        cp = next((p for p in customs if p["id"] == custom_id), None)
        if cp:
            model = config.get("openai_model") or (cp["models"][0] if cp.get("models") else "")
            return (
                "openai",
                cp.get("base_url", ""),
                cp.get("api_key", ""),
                model,
            )
    # Default: openai
    return (
        "openai",
        config.get("openai_base_url", "https://x666.me/v1"),
        config.get("openai_api_key", ""),
        config.get("openai_model", "grok-4.6"),
    )


def build_cli_args(config: dict[str, Any], resume: bool = False) -> list[str]:
    """Build pipeline.py CLI arguments from config dict."""
    args: list[str] = ["python"]

    # Find pipeline.py
    pipeline_py = PIPELINE_DIR / "pipeline.py"
    if not pipeline_py.exists():
        # Fallback: check if pipeline.py is in the same parent dir
        alt = WEB_ROOT / "pipeline.py"
        if alt.exists():
            pipeline_py = alt
        else:
            raise FileNotFoundError(f"pipeline.py not found in {PIPELINE_DIR}")
    args.append(str(pipeline_py))

    # Content
    if config.get("topic"):
        args += ["--topic", config["topic"]]
    args += ["--cefr", str(config.get("cefr", "A2"))]
    if config.get("num_lines"):
        args += ["--num-lines", str(config["num_lines"])]
    args += ["--structure", str(config.get("structure", "original"))]
    args += ["--animation", str(config.get("animation", "landing"))]

    # LLM
    provider = config.get("llm_provider", "sensenova")
    p_type, p_base_url, p_api_key, p_model = resolve_provider(config)
    args += ["--llm-provider", p_type]
    if p_type == "sensenova":
        if p_api_key:
            args += ["--api-key", p_api_key]
        args += ["--model", str(p_model or "deepseek-v4-flash")]
    else:
        if p_base_url:
            args += ["--openai-base-url", p_base_url]
        if p_api_key:
            args += ["--openai-api-key", p_api_key]
        args += ["--openai-model", str(p_model or "grok-4.6")]
    args += ["--llm-retries", str(config.get("llm_retries", 10))]
    if config.get("llm_min_interval"):
        os.environ["LLM_MIN_INTERVAL"] = str(config["llm_min_interval"])

    # TTS
    args += ["--tts-engine", str(config.get("tts_engine", "kokoro"))]
    if config.get("tts_rate"):
        args += ["--tts-rate", config["tts_rate"]]
    if config.get("voxcpm_worker_url"):
        args += ["--voxcpm-worker-url", config["voxcpm_worker_url"]]
    if config.get("voxcpm_api_key"):
        args += ["--voxcpm-api-key", config["voxcpm_api_key"]]
    if config.get("qwen_model_path"):
        args += ["--qwen-model-path", config["qwen_model_path"]]
    if config.get("qwen_base_model_path"):
        args += ["--qwen-base-model-path", config["qwen_base_model_path"]]
    if config.get("qwen_device"):
        args += ["--qwen-device", config["qwen_device"]]

    # MCP
    tokens_raw = config.get("mcp_tokens", "").strip()
    if tokens_raw:
        tokens = [t.strip() for t in tokens_raw.split("\n") if t.strip()]
        if tokens:
            args += ["--mcp-tokens", ",".join(tokens)]

    args += ["--image-concurrency", str(config.get("image_concurrency", 4))]
    args += ["--clip-duration", str(config.get("clip_duration", 15))]
    args += ["--clip-concurrency", str(config.get("clip_concurrency", 4))]
    args += ["--output", str(config.get("output_dir", "./output"))]
    if config.get("topics_file"):
        args += ["--topics-file", config["topics_file"]]
    if config.get("used_topics_file"):
        args += ["--used-topics-file", config["used_topics_file"]]
    if config.get("lessons_dir"):
        args += ["--lessons-dir", config["lessons_dir"]]

    # Video
    args += ["--practice-duration", str(config.get("practice_duration", 3.0))]
    if config.get("pad"):
        try:
            pad_val = float(config["pad"])
            args += ["--pad", str(pad_val)]
        except ValueError:
            pass
    args += ["--render-fps", str(config.get("render_fps", 8))]
    args += ["--workers", str(config.get("workers", 1))]
    args += ["--subtitle-font-size", str(config.get("subtitle_font_size", 60))]
    if config.get("no_zh_subtitle"):
        args.append("--no-zh-subtitle")
    if config.get("no_4k"):
        args.append("--no-4k")
    args += ["--upscale-timeout", str(config.get("upscale_timeout", 3600))]

    if resume:
        args.append("--resume")

    return args


def detect_local_mcp_token() -> str | None:
    """Try to read MCP OAuth token from Codely CLI config."""
    home = Path.home()
    token_file = home / ".codely-cli" / "mcp-oauth-tokens.json"
    if not token_file.exists():
        return None
    try:
        data = json.loads(token_file.read_text(encoding="utf-8"))
        # Format: [{"serverName": "...", "token": {"accessToken": "...", ...}, ...}]
        if isinstance(data, list):
            for entry in data:
                if not isinstance(entry, dict):
                    continue
                tok = entry.get("token")
                if isinstance(tok, dict):
                    at = tok.get("accessToken") or tok.get("access_token")
                    if at and isinstance(at, str) and len(at) > 10:
                        return at
                elif isinstance(tok, str) and len(tok) > 10:
                    return tok
                for key in ("access_token", "accessToken"):
                    val = entry.get(key)
                    if val and isinstance(val, str) and len(val) > 10:
                        return val
        # Format: {"serverName": {"token": "..."}, ...} or {"token": "..."}
        if isinstance(data, dict):
            for key in ("access_token", "token", "accessToken"):
                if key in data:
                    val = data[key]
                    if isinstance(val, str) and len(val) > 10:
                        return val
                    if isinstance(val, dict):
                        at = val.get("accessToken") or val.get("access_token")
                        if at and isinstance(at, str):
                            return at
            for v in data.values():
                if isinstance(v, str) and len(v) > 20:
                    return v
                if isinstance(v, dict):
                    at = v.get("accessToken") or v.get("access_token") or v.get("token")
                    if at and isinstance(at, str) and len(at) > 10:
                        return at
        return None
    except (json.JSONDecodeError, OSError):
        return None
