"""Centralized filesystem paths for the web app."""
import sys
from pathlib import Path

WEB_ROOT = Path(__file__).parent.parent.resolve()
TEMPLATES_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"
PIPELINE_DIR = WEB_ROOT / "pipeline"
CONFIGS_DIR = WEB_ROOT / "configs"
LIBRARY_DIR = CONFIGS_DIR / "character_library"
TRASH_META_FILENAME = ".trash_meta.json"
SCRIPTS_FORM_PATH = CONFIGS_DIR / "scripts_form.json"
CHARACTER_SETS_PATH = CONFIGS_DIR / "character_sets.json"
AI_TEST_CONFIG_PATH = CONFIGS_DIR / "ai_test_config.json"
QWEN_VOICE_CONFIG_PATH = CONFIGS_DIR / "qwen_voice_config.json"
CUSTOM_VOICES_DIR = CONFIGS_DIR / "custom_voices"
VOICE_PREVIEWS_DIR = CONFIGS_DIR / "voice_previews"
KOKORO_VOICE_CONFIG_PATH = CONFIGS_DIR / "kokoro_voice_config.json"
MOSS_VOICE_CONFIG_PATH = CONFIGS_DIR / "moss_voice_config.json"
MOSS_VOICES_DIR = CONFIGS_DIR / "moss_voices"
MOSS_PREVIEWS_DIR = CONFIGS_DIR / "moss_previews"
CHANNEL_DRAFTS_PATH = CONFIGS_DIR / "channel_drafts.json"
CHANNEL_FAVORITES_PATH = CONFIGS_DIR / "channel_favorites.json"
CHANNEL_ASSETS_DIR = CONFIGS_DIR / "channel_assets"
CHANNEL_REFERENCES_PATH = CONFIGS_DIR / "channel_references.json"
# 片头库（sleep 模式 10 秒片头：本地动画 / MCP AI 视频，Web 生成入库）
INTRO_VIDEOS_DIR = CONFIGS_DIR / "intro_videos"
INTRO_LIBRARY_PATH = CONFIGS_DIR / "intro_library.json"


def ensure_pipeline_on_path() -> None:
    """把 pipeline/ 目录加入 sys.path（幂等；须在模块导入期调用一次，
    保证 style_manager/subtitle_style_manager 等顶层 import 可用）。"""
    p = str(PIPELINE_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)
