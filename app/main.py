"""FastAPI main application — 应用装配层：静态资源、路由注册、启动任务。

路由实现按领域拆分在 app/routers/（pages / health / config / run / mode_test /
topics / scripts / styles / subtitle_styles / runs / mcp_tokens / characters /
voices_qwen / voices_kokoro / voices_moss / ai_test），include 顺序与拆分前的
banner 顺序一致，保证路径匹配优先级不变。
"""
import logging

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

# Suppress noisy Windows ConnectionResetError on video stream disconnect
logging.getLogger("asyncio").setLevel(logging.CRITICAL)

from .config_manager import load_all_mode_configs
from .paths import STATIC_DIR, WEB_ROOT
from .routers import (
    ai_test as ai_test_routes,
    characters as characters_routes,
    config as config_routes,
    health as health_routes,
    mcp_tokens as mcp_tokens_routes,
    mode_test as mode_test_routes,
    pages as pages_routes,
    run as run_routes,
    runs as runs_routes,
    scripts as scripts_routes,
    styles as styles_routes,
    subtitle_styles as subtitle_styles_routes,
    topics as topics_routes,
    voices_kokoro as voices_kokoro_routes,
    voices_moss as voices_moss_routes,
    voices_qwen as voices_qwen_routes,
)
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

# 注册顺序 = 拆分前 main.py 的 banner 顺序
app.include_router(pages_routes.router)
app.include_router(config_routes.router)
app.include_router(run_routes.router)
app.include_router(mode_test_routes.router)
app.include_router(topics_routes.router)
app.include_router(styles_routes.router)
app.include_router(subtitle_styles_routes.router)
app.include_router(scripts_routes.router)
app.include_router(runs_routes.router)
app.include_router(mcp_tokens_routes.router)
app.include_router(characters_routes.router)
app.include_router(health_routes.router)
app.include_router(voices_qwen_routes.router)
app.include_router(voices_kokoro_routes.router)
app.include_router(voices_moss_routes.router)
app.include_router(ai_test_routes.router)


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
