"""Configuration manager: load/save JSON configs, preset management,
and build CLI args for pipeline.py."""
import json
import os
import sys
from pathlib import Path
from typing import Any

# Resolve paths
WEB_ROOT = Path(__file__).parent.parent.resolve()
CONFIGS_DIR = WEB_ROOT / "configs"
DEFAULT_CONFIG_PATH = CONFIGS_DIR / "default.json"

# pipeline 模块目录（自包含：topics.json / pipeline.py 均在本项目内，
# 不再依赖旧的外部 colab_listening_b 目录）
PIPELINE_DIR = WEB_ROOT / "pipeline"

# pipeline 模块目录（style_manager 在其中）
_LOCAL_PIPELINE_DIR = PIPELINE_DIR
if str(_LOCAL_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(_LOCAL_PIPELINE_DIR))

# All configurable parameters with defaults, types, and metadata
PARAM_SPEC = {
    # --- Content ---
    "topic": {"default": "", "type": "text", "group": "content",
              "label": "主题", "help": "留空则从主题库随机选择"},
    "cefr": {"default": "A2", "type": "select", "group": "content",
             "label": "CEFR 等级", "options": ["A1", "A2", "B1", "B2", "C1", "C2"]},
    "num_lines": {"default": "", "type": "number", "group": "content",
                  "label": "对话行数", "help": "留空=自动 (original:18, quest:48)"},
    "max_line_words": {"default": 10, "type": "number", "group": "content",
                       "label": "每行最大词数",
                       "help": "对话每行/旁白每句词数上限（默认 10，建议 5-15），保证字幕最多显示两行；超长行触发 QA 修复"},
    "structure": {"default": "original", "type": "select", "group": "content",
                  "label": "视频结构", "options": {
                      "original": "Original (4章视频片段)",
                      "original_static": "Original Static (4章静态图片)",
                      "original_cutout": "Original Cutout (4章+人物抠图)",
                      "quest": "Quest (任务听力)"}},
    "animation": {"default": "landing", "type": "select", "group": "content",
                  "label": "动画类型", "options": {
                      "none": "None (静态)",
                      "landing": "Landing (降落变换)",
                      "stop_motion": "Stop Motion (定格动画)"}},
    "visual_style": {"default": "pixar3d", "type": "select", "group": "content",
                     "label": "画面风格",
                     "options": {"pixar3d": "3D 卡通（皮克斯）"},
                     "help": "贯穿图片/视频/缩略图的画面风格，可在「画面风格」页管理自定义风格"},
    "character_source": {"default": "", "type": "text", "group": "content",
                         "label": "复用角色来源", "help": "留空=全部新生成。选择之前运行可复用其角色"},
    "character_reuse": {"default": "", "type": "text", "group": "content",
                        "label": "复用角色选择", "help": "JSON: {char_a: 'image'|'desc'|'voice'} — image=图片+描述+性别, desc=描述+性别, voice=仅音色+性别(外观每次重新生成)"},
    "character_fixes": {"default": "", "type": "text", "group": "content",
                        "label": "固定角色描述", "help": "手动指定角色外观描述 (JSON)"},
    "character_library": {"default": "", "type": "text", "group": "content",
                           "label": "素材库分配", "help": "JSON: {char_a: 'lib_id', char_b: 'lib_id'}"},
    "character_voices": {"default": "", "type": "text", "group": "content",
                          "label": "角色音色绑定", "help": "JSON: {char_a: 'Vivian', char_b: 'Ryan'} — 复用角色时绑定 Qwen TTS 音色"},
    "character_zh_voices": {"default": "", "type": "text", "group": "content",
                            "label": "中文声音绑定", "help": "JSON: {char_a: 'zh-CN-YunxiNeural', char_b: 'zh-CN-XiaoxiaoNeural'} — 绑定中文翻译旁白的 edge-tts 声音"},
    "host_character": {"default": "", "type": "select", "group": "content",
                       "label": "主持人形象绑定",
                       "options": {"": "独立主持人（新生成）", "char_a": "角色 A", "char_b": "角色 B"},
                       "help": "Original Cutout 模式生效：开场/结尾主持人复用所选角色的姿势图集与音色，不再单独生成主持人"},

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
                        "label": "QA 轮数（全模式）",
                        "help": "最少审查-修复轮数；有 error 时一直修到 0 error 或 10 轮上限 (默认3, 0=关闭)"},

    # --- TTS ---
    "tts_engine": {"default": "kokoro", "type": "select", "group": "tts",
                   "label": "TTS 引擎", "options": {
                       "kokoro": "Kokoro (本地)",
                       "qwen": "Qwen3-TTS (本地 GPU)",
                       "moss": "MOSS-TTS-Nano (本地 CPU)"}},
    "tts_rate": {"default": "", "type": "text", "group": "tts",
                 "label": "TTS 语速", "help": "如 -15%, 0% (留空=模式默认)"},
    "qwen_model_path": {"default": r"H:\models\Qwen3-TTS-12Hz-0.6B-CustomVoice",
                        "type": "text", "group": "tts",
                        "label": "Qwen3-TTS 模型路径",
                        "help": "CustomVoice 模型 (预设音色)"},
    "qwen_base_model_path": {"default": r"H:\models\Qwen3-TTS-12Hz-1.7B-Base",
                             "type": "text", "group": "tts",
                             "label": "Qwen3-TTS Base 模型路径",
                             "help": "Base 模型 (voice clone 需要)"},
    "qwen_voicedesign_model_path": {"default": r"H:\models\Qwen3-TTS-12Hz-1.7B-VoiceDesign",
                                    "type": "text", "group": "tts",
                                    "label": "Qwen3-TTS VoiceDesign 模型路径",
                                    "help": "VoiceDesign 模型 (设计音色/英语女声需要)"},
    "qwen_device": {"default": "cuda:0", "type": "text", "group": "tts",
                    "label": "Qwen3-TTS 设备", "help": "如 cuda:0, cpu"},

    "moss_model_path": {"default": r"H:\models\MOSS-TTS-Nano-Model", "type": "text", "group": "tts",
                        "label": "MOSS-TTS-Nano 模型路径",
                        "help": "模型 checkpoint 目录"},
    "moss_tokenizer_path": {"default": r"H:\models\MOSS-Audio-Tokenizer-Nano", "type": "text", "group": "tts",
                            "label": "MOSS Audio Tokenizer 路径",
                            "help": "音频 Tokenizer 目录"},
    "moss_device": {"default": "cpu", "type": "text", "group": "tts",
                    "label": "MOSS-TTS 设备", "help": "如 cpu, cuda:0 (默认 CPU)"},
    "moss_repo_dir": {"default": r"H:\models\MOSS-TTS-Nano", "type": "text", "group": "tts",
                      "label": "MOSS-TTS-Nano 仓库目录",
                      "help": "包含 infer.py / moss_tts_nano_runtime.py 的仓库路径"},
    "moss_tts_temperature": {"default": 0.8, "type": "number", "group": "tts",
                             "label": "MOSS-TTS 采样温度",
                             "help": "越低越稳定（推荐 0.6-0.8），越高越有表现力；修改后自动重新生成 TTS 缓存"},
    "moss_tts_retry": {"default": 3, "type": "number", "group": "tts",
                       "label": "MOSS-TTS 重试次数",
                       "help": "每句合成失败/校验不达标时换 seed 重试的次数（1=不重试）"},

    # --- MCP / Image ---
    "mcp_tokens": {"default": "", "type": "textarea", "group": "mcp",
                   "label": "MCP Tokens", "help": "每行一个 token, 多 token 自动轮换"},
    "image_concurrency": {"default": 4, "type": "number", "group": "mcp",
                          "label": "图片并发数", "help": "1-4, 默认4"},
    "image_provider": {"default": "mcp", "type": "select", "group": "mcp",
                       "label": "生图 Provider", "options": ["mcp", "sensenova"],
                       "help": "mcp=TJGenerators(积分)；sensenova=SenseNova U1.5 Lite(API 计费, 复用 SenseNova API Key)"},
    "clip_duration": {"default": 15, "type": "number", "group": "mcp",
                      "label": "视频片段时长(秒)", "help": "4-15"},
    "clip_concurrency": {"default": 4, "type": "number", "group": "mcp",
                          "label": "视频并发数", "help": "1-5, 默认4 (MCP最大并发5)"},
    "output_dir": {"default": str(WEB_ROOT / "output"), "type": "text", "group": "mcp",
                   "label": "输出目录"},
    "topics_file": {"default": str(PIPELINE_DIR / "topics.json"), "type": "text", "group": "mcp",
                    "label": "主题库文件"},
    "used_topics_file": {"default": "", "type": "text", "group": "mcp",
                         "label": "已用主题文件", "help": "留空=<output>/used_topics.json"},
    "lessons_dir": {"default": "", "type": "text", "group": "mcp",
                    "label": "防重复目录", "help": "留空=不检查"},

    # --- Video ---
    "practice_duration": {"default": 3.0, "type": "number", "group": "video",
                          "label": "Ch3 跟读间隔(秒)",
                          "help": "Ch3 每次朗读（英文/中文）后的停顿秒数。Quest 模式不适用"},
    "ch3_en_repeats": {"default": 3, "type": "number", "group": "video",
                       "label": "Ch3 英文重复次数",
                       "help": "跟读练习每句英文朗读遍数 (0-10, 0=不播英文)。Quest 模式不适用"},
    "ch3_zh_repeats": {"default": 1, "type": "number", "group": "video",
                       "label": "Ch3 中文重复次数",
                       "help": "中文翻译朗读遍数 (0-10, 0=不播中文)。Quest 模式不适用"},
    "ch3_zh_always": {"default": True, "type": "checkbox", "group": "video",
                      "label": "Ch3 中文常显",
                      "help": "英文跟读时画面同步显示中文翻译。Quest 模式不适用"},
    "ch3_practice_intro_show": {"default": True, "type": "checkbox", "group": "video",
                                "label": "Ch3 练习引导卡",
                                "help": "跟读练习前的引导文字卡（逐句显示）。关闭后整段跳过。Quest 模式不适用"},
    "pad": {"default": "", "type": "text", "group": "video",
            "label": "音频间隔(秒)", "help": "留空=自动 (0.4)"},
    "render_fps": {"default": 8, "type": "number", "group": "video",
                   "label": "渲染帧率 (stop_motion)"},
    "workers": {"default": 1, "type": "number", "group": "video",
                "label": "渲染线程数", "help": "0=自动(CPU核数)"},
    "subtitle_style": {"default": "", "type": "select", "group": "video",
                       "label": "字幕样式",
                       "options": {"": "跟随参数配置（默认）"},
                       "help": "「字幕样式」页设计并选择；跟随参数配置时使用下方字幕字体大小"},
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
# 每种视频结构模式各一份完整独立配置，切换互不覆盖。
# default.json 仅作首次迁移源；active_mode.json 记录当前激活模式。

MODES = ["original", "original_static", "original_cutout", "quest"]
MODE_LABELS = {
    "original": "Original (4章视频片段)",
    "original_static": "Original Static (4章静态图片)",
    "original_cutout": "Original Cutout (4章+人物抠图)",
    "quest": "Quest (任务听力)",
}
ACTIVE_MODE_PATH = CONFIGS_DIR / "active_mode.json"

# --- 输出目录布局 ---
# 新布局：output/{mode}/{run_name}/，每种模式一个独立文件夹；
# 旧扁平布局 output/{run_name}/ 继续兼容读写。
# 回收站：output/_recycle_bin/（与模式文件夹平级的独立文件夹）。
RECYCLE_DIRNAME = "_recycle_bin"
LEGACY_RECYCLE_DIRNAME = ".recycle_bin"  # 旧版回收站（兼容恢复/清空）

MODE_SHORT_LABELS = {
    "original": "Original",
    "original_static": "Static",
    "original_cutout": "Cutout",
    "quest": "Quest",
}


def _is_safe_run_name(name: str) -> bool:
    return bool(name) and name not in (".", "..") and "/" not in name and "\\" not in name


def find_run_dir(output_dir: Path | str, name: str, mode_hint: str = "") -> Path | None:
    """按运行名查找运行目录（兼容新旧两种布局）。

    查找顺序：mode_hint 指定的模式文件夹 → 旧扁平布局 → 各模式文件夹。
    同名运行跨模式重名时以 hint 消歧；未找到返回 None。
    """
    root = Path(output_dir)
    if not _is_safe_run_name(name):
        return None
    if mode_hint in MODES:
        p = root / mode_hint / name
        if p.is_dir():
            return p
    if name not in MODES and name not in (RECYCLE_DIRNAME, LEGACY_RECYCLE_DIRNAME):
        p = root / name
        if p.is_dir():
            return p
    for mode in MODES:
        p = root / mode / name
        if p.is_dir():
            return p
    return None


def iter_run_dirs(output_dir: Path | str) -> list[Path]:
    """遍历输出目录下所有运行目录（新旧布局，按修改时间倒序）。

    新布局取模式文件夹下一层；旧扁平布局取根目录一层
    （模式文件夹与回收站文件夹除外，不要求 script.json 已生成）。
    """
    root = Path(output_dir)
    runs: list[Path] = []
    if not root.is_dir():
        return runs
    for entry in root.iterdir():
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        if entry.name in (RECYCLE_DIRNAME, LEGACY_RECYCLE_DIRNAME):
            continue
        if entry.name in MODES:
            runs.extend(d for d in entry.iterdir()
                        if d.is_dir() and not d.name.startswith("."))
        else:
            runs.append(entry)  # 旧扁平布局运行目录
    runs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return runs


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
            models = cp.get("models") or []
            model = config.get("openai_model") or ""
            if model not in models:
                # 配置里的 openai_model 不属于该 Provider（如内置 OpenAI 的模型）→ 用第一个
                model = models[0] if models else ""
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
    if config.get("max_line_words"):
        args += ["--max-line-words", str(config["max_line_words"])]
    args += ["--structure", str(config.get("structure", "original"))]
    args += ["--animation", str(config.get("animation", "landing"))]
    if config.get("host_character"):
        args += ["--host-character", str(config["host_character"])]
    args += ["--visual-style", str(config.get("visual_style", "pixar3d"))]

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
    if config.get("qwen_model_path"):
        args += ["--qwen-model-path", config["qwen_model_path"]]
    if config.get("qwen_base_model_path"):
        args += ["--qwen-base-model-path", config["qwen_base_model_path"]]
    if config.get("qwen_voicedesign_model_path"):
        args += ["--qwen-voicedesign-model-path", config["qwen_voicedesign_model_path"]]
    if config.get("qwen_device"):
        args += ["--qwen-device", config["qwen_device"]]
    if config.get("moss_model_path"):
        args += ["--moss-model-path", config["moss_model_path"]]
    if config.get("moss_tokenizer_path"):
        args += ["--moss-tokenizer-path", config["moss_tokenizer_path"]]
    if config.get("moss_device"):
        args += ["--moss-device", config["moss_device"]]
    if config.get("moss_repo_dir"):
        args += ["--moss-repo-dir", config["moss_repo_dir"]]
    if config.get("moss_tts_temperature"):
        args += ["--moss-tts-temperature", str(config["moss_tts_temperature"])]
    if config.get("moss_tts_retry"):
        args += ["--moss-tts-retry", str(config["moss_tts_retry"])]

    # MCP
    tokens_raw = config.get("mcp_tokens", "").strip()
    if tokens_raw:
        tokens = [t.strip() for t in tokens_raw.split("\n") if t.strip()]
        if tokens:
            args += ["--mcp-tokens", ",".join(tokens)]

    args += ["--image-concurrency", str(config.get("image_concurrency", 4))]
    args += ["--image-provider", str(config.get("image_provider", "mcp"))]
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
    args += ["--ch3-en-repeats", str(int(config.get("ch3_en_repeats", 3) or 0))]
    args += ["--ch3-zh-repeats", str(int(config.get("ch3_zh_repeats", 1) or 0))]
    if config.get("ch3_zh_always", True):
        args.append("--ch3-zh-always")
    else:
        args.append("--no-ch3-zh-always")
    if config.get("ch3_practice_intro_show", True):
        args.append("--ch3-practice-intro-show")
    else:
        args.append("--no-ch3-practice-intro-show")
    if config.get("pad"):
        try:
            pad_val = float(config["pad"])
            args += ["--pad", str(pad_val)]
        except ValueError:
            pass
    args += ["--render-fps", str(config.get("render_fps", 8))]
    args += ["--workers", str(config.get("workers", 1))]
    if config.get("subtitle_style"):
        args += ["--subtitle-style", str(config["subtitle_style"])]
    args += ["--subtitle-font-size", str(config.get("subtitle_font_size", 60))]
    if config.get("no_zh_subtitle"):
        args.append("--no-zh-subtitle")
    if config.get("no_4k"):
        args.append("--no-4k")
    args += ["--upscale-timeout", str(config.get("upscale_timeout", 3600))]

    if resume:
        args.append("--resume")

    return args


def _extract_token(entry: dict) -> str | None:
    """从单条 token 文件条目中提取 accessToken（兼容多种字段布局）。"""
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
    return None


def detect_local_mcp_token() -> str | None:
    """Try to read MCP OAuth token from Codely CLI config.

    优先精确匹配 TJGenerators server（本项目唯一的 MCP 图片生成服务），
    避免将来配置其他 MCP server 后误取别家 token；找不到再宽松兜底。
    """
    home = Path.home()
    token_file = home / ".codely-cli" / "mcp-oauth-tokens.json"
    if not token_file.exists():
        return None
    try:
        data = json.loads(token_file.read_text(encoding="utf-8"))
        # Format: [{"serverName": "...", "token": {"accessToken": "...", ...}, ...}]
        if isinstance(data, list):
            # 1. 优先 TJGenerators
            for entry in data:
                if (isinstance(entry, dict)
                        and entry.get("serverName") == "TJGenerators"):
                    at = _extract_token(entry)
                    if at:
                        return at
            # 2. 宽松兜底：任意条目的 token
            for entry in data:
                if not isinstance(entry, dict):
                    continue
                at = _extract_token(entry)
                if at:
                    return at
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
