"""YouTube 元数据多样式选项注册包。

每个样式 = 本目录下一个独立 .py 文件（`_` 前缀的除外），需暴露：

    STYLE_ID: str                      # 唯一 id，写入 youtube_metadata.json
    LABEL: str                         # 展示名（gallery 卡片标题）
    STRUCTURES: tuple[str, ...]        # 适用结构，如 ("original", "original_static", "original_cutout")
    build(script, ctx) -> dict | None  # 返回 {"title": str, "description": str}；无法生成时返回 None

ctx 字段：
    chapters:   默认章节串列表（"00:00 Welcome & Hook" 风格，英文 label）
    marks:      [(秒, seg_type), ...] 章节 mark（未做 10s 过滤，样式自行按 YouTube
                硬规则处理：首章必须 00:00、相邻 >=10s）
    structure:  当前结构模式

新增样式：丢一个新 .py 进本目录即可，save_youtube_metadata 会自动收集进
youtube_metadata.json 的 title_options。脚本缺样式所需字段（如旧脚本没有
youtube_title_ref）时 build() 返回 None，该样式整体跳过，不影响其他样式与默认选项。
"""
import importlib
import pkgutil
from pathlib import Path


def iter_styles():
    """自动发现并逐个 yield 样式模块（跳过 _ 前缀；坏模块只告警不中断）。"""
    for mod in pkgutil.iter_modules([str(Path(__file__).parent)]):
        if mod.name.startswith("_"):
            continue
        try:
            yield importlib.import_module(f".{mod.name}", __name__)
        except Exception as e:
            print(f"  [YtStyles] WARNING: 样式模块 {mod.name} 加载失败: {e}")


def collect_options(script: dict, chapters: list[str],
                    marks: list[tuple[float, str]], structure: str) -> list[dict]:
    """收集适用于当前结构的全部样式选项，供 save_youtube_metadata 写入 title_options。"""
    options: list[dict] = []
    ctx = {"chapters": chapters, "marks": marks, "structure": structure}
    for mod in iter_styles():
        try:
            if structure not in getattr(mod, "STRUCTURES", ()):
                continue
            built = mod.build(script, ctx)
        except Exception as e:
            print(f"  [YtStyles] WARNING: 样式 {getattr(mod, 'STYLE_ID', mod.__name__)} 生成失败: {e}")
            continue
        if not built:
            continue
        options.append({
            "style_id": mod.STYLE_ID,
            "label": mod.LABEL,
            "title": str(built.get("title", "")),
            "description": str(built.get("description", "")),
        })
    return options
