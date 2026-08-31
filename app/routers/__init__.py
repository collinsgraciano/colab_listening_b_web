"""API 路由包：导入任一 router 前确保 pipeline/ 已入 sys.path
（style_manager/subtitle_style_manager 等顶层模块依赖此前完成 bootstrap）。"""
from ..paths import ensure_pipeline_on_path

ensure_pipeline_on_path()
