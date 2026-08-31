"""FastAPI main application — all routes and API endpoints."""
import asyncio
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path

from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import (
    HTMLResponse, JSONResponse, StreamingResponse,
    FileResponse, RedirectResponse, PlainTextResponse, Response,
)
from fastapi.staticfiles import StaticFiles

# Suppress noisy Windows ConnectionResetError on video stream disconnect
logging.getLogger("asyncio").setLevel(logging.CRITICAL)

from .config_manager import (
    PARAM_SPEC, GROUP_META, get_default_config,
    load_config, save_config,
    list_presets, save_preset, load_preset, delete_preset,
    build_cli_args, detect_local_mcp_token,
    load_llm_providers, save_llm_providers,
    get_provider_options, resolve_provider,
    MODES, MODE_LABELS, MODE_SHORT_LABELS, get_active_mode, set_active_mode,
    load_mode_config, save_mode_config, load_all_mode_configs,
    effective_param_spec, param_effective_in_mode,
    RECYCLE_DIRNAME, LEGACY_RECYCLE_DIRNAME,
    find_run_dir, iter_run_dirs,
)
from .pipeline_service import get_service, STEP_ORDER
from .mode_test_service import get_mode_test_service
from .config_manager import detect_local_mcp_token
from .page_mcp import (
    PAGES as MCP_PAGES, SOURCE_LABELS as MCP_SOURCE_LABELS,
    PageMcpSession, resolve_page_tokens, load_page_tokens, save_page_tokens,
    mask_token,
)
from . import topics_ai
from . import script_library
import style_manager as style_lib
import subtitle_style_manager as subtitle_style_lib

# 共享基础设施（路径 / 模板 / SSE / 话题文件 IO / TTS 共享状态）
from .paths import (
    WEB_ROOT, TEMPLATES_DIR, STATIC_DIR, PIPELINE_DIR, LIBRARY_DIR,
    TRASH_META_FILENAME, SCRIPTS_FORM_PATH, CHARACTER_SETS_PATH, AI_TEST_CONFIG_PATH,
    QWEN_VOICE_CONFIG_PATH, CUSTOM_VOICES_DIR, VOICE_PREVIEWS_DIR,
    KOKORO_VOICE_CONFIG_PATH, MOSS_VOICE_CONFIG_PATH, MOSS_VOICES_DIR, MOSS_PREVIEWS_DIR,
)
from .templating import templates
from .sse import sse_line as _sse, SSE_HEADERS as _SSE_HEADERS
from .topics_io import (
    load_topics_data as _load_topics_data,
    load_used_topic_names as _load_used_topic_names,
)
from .tts_state import TTS_SYNTH_LOCK as _TTS_SYNTH_LOCK, PREVIEW_TEXTS as _PREVIEW_TEXTS
from .routers import config as config_routes
from .routers import health as health_routes
from .routers import mcp_tokens as mcp_tokens_routes
from .routers import mode_test as mode_test_routes
from .routers import run as run_routes
from .routers import runs as runs_routes
from .routers import scripts as scripts_routes
from .routers import styles as styles_routes
from .routers import subtitle_styles as subtitle_styles_routes
from .routers import topics as topics_routes
from .routers import characters as characters_routes
from .routers import ai_test as ai_test_routes
from .routers import pages as pages_routes
from .routers.ai_test import _load_ai_test_config
from .routers import voices_kokoro as voices_kokoro_routes
from .routers import voices_moss as voices_moss_routes
from .routers import voices_qwen as voices_qwen_routes
from .routers.voices_qwen import auto_freeze_pending_designed_voices

# FastAPI app
app = FastAPI(title="Listening Video Generator")


class _NoCacheStaticFiles(StaticFiles):
    """静态文件禁用浏览器缓存（本地开发工具，CSS/JS 迭代频繁，防止改版后浏览器继续用旧缓存）。"""

    def file_response(self, *args, **kwargs):
        resp = super().file_response(*args, **kwargs)
        resp.headers["Cache-Control"] = "no-cache"
        return resp


app.mount("/static", _NoCacheStaticFiles(directory=str(STATIC_DIR)), name="static")


app.include_router(pages_routes.router)

# ===========================================================================
# Config API
# ===========================================================================

app.include_router(config_routes.router)

# ===========================================================================
# Pipeline Run API
# ===========================================================================

app.include_router(run_routes.router)

# ===========================================================================
# 模式效果测试（一次性生成迷你素材 + 零消耗本地合成）
# ===========================================================================

app.include_router(mode_test_routes.router)

app.include_router(topics_routes.router)

# ===========================================================================
# Visual Styles — 画面风格管理
# ===========================================================================

app.include_router(styles_routes.router)

# Subtitle Styles — 字幕样式设计与预览
# ===========================================================================

app.include_router(subtitle_styles_routes.router)

# ===========================================================================
# Scripts Library — batch generation & quality review
# ===========================================================================

app.include_router(scripts_routes.router)

app.include_router(runs_routes.router)

# ===========================================================================
# MCP Token detection
# ===========================================================================

app.include_router(mcp_tokens_routes.router)

app.include_router(characters_routes.router)

# ===========================================================================
# Health check
# ===========================================================================

app.include_router(health_routes.router)

# ===========================================================================
# QwenTTS Voice Management
# ===========================================================================


app.include_router(voices_qwen_routes.router)

# Kokoro TTS Voices (本地引擎，固定音色清单 + 试听)
# ===========================================================================

app.include_router(voices_kokoro_routes.router)

# MOSS-TTS Voice Management
# ===========================================================================


app.include_router(voices_moss_routes.router)

# AI Test — LLM Playground
# ===========================================================================


app.include_router(ai_test_routes.router)

# Character Library Management Page
# ===========================================================================

# Startup
# ===========================================================================

@app.on_event("startup")
async def startup():
    # Ensure configs dir exists
    (WEB_ROOT / "configs").mkdir(parents=True, exist_ok=True)
    (WEB_ROOT / "configs" / "character_library").mkdir(parents=True, exist_ok=True)
    # 首次运行：从 legacy default.json 迁移生成 3 个模式配置文件
    load_all_mode_configs()
    # 设计音色自动冻结：内置英文女声 + 残留未冻结的设计音色 → 后台转为克隆音色（一次性）
    try:
        auto_freeze_pending_designed_voices()
    except Exception as e:  # noqa: BLE001 — 冻结失败不阻塞启动，下次启动重试
        print(f"[startup] 设计音色自动冻结入队失败: {e}")
