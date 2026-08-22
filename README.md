# colab_listening_b Web 管理界面

本地运行的 Web 应用，用于管理和运行英语听力视频生成 Pipeline。

## 快速启动

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 启动 Web 服务
python -m uvicorn app.main:app --host 0.0.0.0 --port 59510

# 或直接运行 run.bat (Windows)
run.bat
```

打开浏览器访问 http://localhost:59510

## 功能

- **控制台**: 一键启动/继续/停止 Pipeline，实时日志推送，进度指示
- **参数配置**: 所有 Pipeline 参数可视化配置（内容/LLM/TTS/MCP/视频合成），支持预设保存加载
- **主题管理**: 浏览/添加/删除主题，查看已用主题，随机选择
- **运行历史**: 查看所有生成的视频，在线播放，查看脚本，删除记录

## 配置说明

所有参数保存在 `configs/default.json`。预设文件也存放在 `configs/` 目录。

## 架构

```
colab_listening_b_web/
├── app/
│   ├── main.py              # FastAPI 路由 + API
│   ├── config_manager.py     # 配置读写 + CLI 参数构建
│   ├── pipeline_runner.py    # subprocess 执行 + SSE 日志流
│   ├── templates/            # Jinja2 HTML 模板
│   └── static/              # CSS + JS
├── configs/                  # 配置文件 + 预设
├── requirements.txt
└── run.bat
```

Web 应用通过 subprocess 调用 `../colab_listening_b/pipeline.py`，不修改任何现有代码。
