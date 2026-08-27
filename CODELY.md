# CODELY.md — colab_listening_b_web

## Project Overview

本地 Web 管理界面，用于管理和运行英语听力视频生成 Pipeline。通过 FastAPI Web UI 一键启动/继续/停止视频生成流程，实时查看日志和进度，可视化管理所有参数、主题和运行历史。

**核心能力：** 将英语对话脚本（LLM 生成）→ TTS 配音 → AI 图片/视频生成 → FFmpeg 合成 → 4K 超分辨率 完整流程封装为可交互的 Web 应用，支持断点续传、单步调试模式、角色复用。

**技术栈：** FastAPI + Jinja2 SSR + Tailwind CSS + HTMX + 原生 JS

## Architecture

```
colab_listening_b_web/
├── app/                          # Web 应用层
│   ├── main.py                   # FastAPI 路由 + API 端点（页面/配置/Pipeline/主题/运行历史/角色复用）
│   ├── config_manager.py         # 参数规格定义(PARAM_SPEC)、配置读写、预设管理、CLI 参数构建、MCP Token 检测
│   ├── pipeline_service.py       # 核心：直接 import pipeline 模块，后台线程执行，步骤进度追踪，单步模式
│   ├── pipeline_runner.py        # 兼容性包装（re-export get_service）
│   ├── templates/                # Jinja2 HTML 模板（base + dashboard + config + topics + runs + gallery + partials/）
│   └── static/                    # CSS (style.css) + JS (app.js)
├── pipeline/                     # 独立 Pipeline 代码（自包含，不依赖外部项目）
│   ├── pipeline.py               # 主编排器：7 步流程 + argparse CLI 入口
│   ├── llm_client.py             # LLM 客户端（SenseNova / OpenAI 兼容 API；listening 脚本生成）
│   ├── mcp_client.py             # TJGenerators MCP HTTP 客户端（多 Token 轮换，积分耗尽自动切换）
│   ├── tts_engine.py             # Kokoro TTS 引擎（英文+中文，音量归一化，语音修复）
│   ├── tts_pipeline.py           # 批量 TTS 生成（对话+旁白+词汇）
│   ├── image_gen.py              # AI 图片生成（角色/场景/对话图/姿势图，含断点续传检查）
│   ├── clip_gen.py               # Seedance2 视频片段生成（任务构建+轮询+下载+重试）
│   ├── grouping_b.py             # 对话行分组（连续行合并为一个视频片段）
│   ├── group_audio.py            # 分组音频拼接
│   ├── timeline.py               # 时间轴构建（listening 4章结构）+ SRT 生成
│   ├── timeline_enrich.py        # 时间轴补全（audio_dur/duration）
│   ├── video_compose.py          # FFmpeg + Pillow 视频合成（original/original_static 模式）
│   ├── original_cutout_compose.py # Original Cutout 合成（人物抠图停格动画）
│   ├── stop_motion.py            # 定格动画渲染（多姿势+光流插帧，quest/cutout 共用）
│   ├── thumbnail_gen.py          # YouTube 缩略图 + 元数据生成
│   ├── media_utils.py            # FFmpeg/Pillow 共享工具（concat/字幕烧录/loudnorm/分辨率探测）
│   ├── topic_manager.py          # 主题随机选择 + 防重复（topics.json + used_topics.json）
│   ├── checkpoint.py             # 断点续传（checkpoint.json 保存/加载/步骤完成检查）
│   ├── topics.json                # 主题库（分类 JSON）
│   └── quest/                     # Quest 结构变体（任务听力模式）
│       ├── llm_client_quest.py   # Quest 脚本生成（48行，4阶段：buildup→core→reveal→review）
│       ├── timeline_quest.py     # Quest 时间轴 + SRT
│       └── video_compose_quest.py # Quest 视频合成（定格动画，角色姿势图）
├── configs/                      # 配置文件 + 预设
│   └── default.json              # 默认配置（含所有参数）
├── requirements.txt              # Python 依赖
└── run.bat                       # Windows 启动脚本
```

### 关键设计

- **直接导入而非 subprocess：** `pipeline_service.py` 直接 `import` pipeline 模块，在后台线程中调用 `_step0_script()` ~ `_step6_4k()` 函数，实现实时进度追踪和中间结果访问。stdout 被 `_LineBuffer` 逐行捕获转发到 SSE 日志流。
- **单步模式：** `_step_mode` 在每步完成后暂停，等待用户点击"继续"，便于调试和检查中间产物。
- **断点续传：** 每步完成后写 `checkpoint.json`，`--resume` 时跳过已完成步骤。完整运行后清除 checkpoint。
- **角色复用：** 支持从之前的运行中复制角色图片和描述，或用自定义描述覆盖（`character_source` + `character_reuse` + `character_fixes`）。
- **角色套装：** 整套角色配置（复用源/复用模式/素材库绑定/Qwen音色/描述覆盖）可命名保存到 `configs/character_sets.json`，控制台「复用角色」区下拉一键应用到当前模式（同名单模式覆盖更新；跨模式应用有确认警告）。
- **多 Token 轮换：** MCP 客户端支持多个 OAuth Token，积分耗尽时自动切换到下一个。

## Pipeline Steps

| 步骤 | 函数 | 说明 |
|------|------|------|
| Step 0 | `_step0_script` | LLM 脚本生成（SenseNova / OpenAI 兼容 API），含验证+重试 |
| Step 1 | `_step1_mcp` | TJGenerators MCP 初始化（多 Token 轮换） |
| Step 2 | `_step2_images_tts` | 并发：AI 图片生成 + TTS 配音（后台线程） |
| Step 3 | `_step3_clips` | Seedance2 视频片段生成（original_static/original_cutout/quest 模式跳过） |
| Step 4 | `_step4_timeline` | 时间轴 + SRT 字幕构建 |
| Step 4.5 | `_step45_thumbnail` | YouTube 缩略图 + 元数据生成 |
| Step 5 | `_step5_compose` | FFmpeg + Pillow 最终视频合成 |
| Step 6 | `_step6_4k` | 可选 4K 超分辨率（`--no-4k` 跳过） |

### 视频结构模式

- **original** — 4 章视频片段模式（对话用 Seedance2 AI 视频，含跟读练习）
- **original_static** — 静态图片模式（4 章、无视频生成、逐行对话配图，支持 none/landing/stop_motion 动画类型）
- **original_cutout** — 人物抠图模式（原版 4 章 + quest 式角色抠图停格动画：char_a/b 各 8 姿势图集）
- **quest** — 任务听力模式（48 行对话，4 阶段结构，定格动画，3+1 角色）

### 标题策略（title_quote 引语钩子）

- LLM 脚本新增 `title_quote` 字段：从对话中逐字挑选最抓耳的一句（<10 词）
- `youtube_title_en` 优先用引语钩子模式：`"Pump Number 3, Please" — Paying Inside at an American Gas Station`
- `youtube_title`（繁中）可用引语的繁中翻译开头（「三號加油機，麻煩了！」）

## Building and Running

### 启动 Web 服务

```bash
# 安装依赖
pip install -r requirements.txt

# 启动（方式一：run.bat，端口 8765，仅本机访问）
run.bat

# 启动（方式二：手动 uvicorn，可自定义端口和访问范围）
python -m uvicorn app.main:app --host 0.0.0.0 --port 59510
```

浏览器访问 http://localhost:8765（run.bat）或 http://localhost:59510（手动）

### 直接运行 Pipeline（CLI）

```bash
cd pipeline
python pipeline.py --topic "At the Pharmacy" --cefr A2 --output ./output --mcp-tokens TOK1,TOK2 --structure original
python pipeline.py --resume  # 断点续传
```

### Pipeline 外部依赖

- **FFmpeg** — 必须在 PATH 中（视频合成、音频处理、4K 超分辨率）
- **TJGenerators MCP** — AI 图片/视频生成（需 OAuth Token，配置在 `mcp_tokens` 或 `~/.codely-cli/mcp-oauth-tokens.json`）
- **Kokoro TTS** — 本地 TTS 引擎（首次使用自动从 HuggingFace 下载模型，Windows 自动用 hf-mirror.com）
- **SenseNova API Key** — LLM 脚本生成（或使用 OpenAI 兼容 API）
- **Pillow (PIL)** — 静态帧渲染、字幕 PNG 叠加、缩略图生成

## Development Conventions

### 代码风格

- Python 3.12+ 类型注解（`dict[str, Any]`, `list[str]`, `str | None`）
- FastAPI 路由用 `async def`；Pipeline 执行在 `threading.Thread` 中同步运行
- 配置参数集中定义在 `PARAM_SPEC` 字典（含 default/type/group/label/help/options）
- Pipeline 模块自包含：不 import 外部项目，所有依赖在 `pipeline/` 目录内
- 中文注释和日志（面向中文用户），代码标识符用英文
- Windows 兼容：`sys.stdout.reconfigure(encoding='utf-8')` 处理控制台编码

### 前端约定

- Jinja2 模板继承 `base.html`（侧边栏 + 内容区）
- Tailwind CSS（CDN）+ HTMX 用于动态交互
- SSE（Server-Sent Events）实时推送日志和状态
- API 路径约定：`/api/{domain}/{action}`（如 `/api/run/start`, `/api/config/save`）

### 配置管理

- 默认配置：`configs/default.json`（所有参数的当前值）
- 预设文件：`configs/{name}.json`（保存的参数组合）
- 配置合并策略：`get_default_config()` → `merged.update(saved)`（新增参数自动补全默认值）
- 敏感文件：`configs/*.local.json` 被 `.gitignore` 忽略

### Git 工作流

- 每次修改完代码后，立即 commit 并 push 到 GitHub

### Pipeline 关键约束

- 时间轴 gap 必须为 0.0（FFmpeg concat 无缝，非零 gap 导致 SRT 漂移）
- TTS 音频必须 `loudnorm` 归一化（跨语音音量一致）
- `setpts` 变速公式：`target_dur/vid_dur`（>1 减速，<1 加速），需配合 `fps=25` 滤镜
- 中文文本在 Windows 上用 Pillow 渲染（FFmpeg drawtext 不支持中文）
- IPA 音标用 `C:\Windows\Fonts\cambria.ttc`（msyh.ttc 渲染为方框）
- Seedance2 视频最大并发 5 个任务，超出需分批创建
- MCP 下载文件需验证大小 > 500KB（防止静默失败的小文件）
- `probe_resolution` 解析 ffprobe csv 输出（`1080,1920` 逗号分隔，非 `x` 分隔）— 字幕叠加按实测画布渲染
- 主题库矩阵策略：每个场景从多角度开采（顾客操作/员工POV/品牌具体/首次体验/小麻烦/文化问答），`pipeline/topics.json` 现 726 条 26 分类；topics_ai.py 生成 prompt 已内置矩阵思维

## Codely Structured Memories

### User

### Feedback
- [2026-08-26 20:02:43] User wants every code change committed and pushed to GitHub immediately after modification. **Why:** user explicitly asked to enforce this as a standing rule in CODELY.md. **How to apply:** after completing any code edit in this project, run git add + commit + push right away without being asked.
- [2026-08-27 01:48:36] CODELY.md 正文无法用 replace/write_file 直接编辑（工具报 "memory file" 错误），须走 Python 脚本字节级替换；本项目及多数仓库文件为 CRLF 换行，search 输出看不出 \r，模式串要先 python repr 探针确认行尾，并把 pattern 中 \n 统一转为 \r\n，且每处替换加 assert count==1 防静默错位。
- [2026-08-27 02:03:12] [2026-08-27 02:02] Windows 验证类命令的执行方式：不要用超长 `python -c "..."; ...` 单行，改用 write_file 写临时 .py（必要时配 .cmd）到系统临时目录（%TEMP%≈C:\Users\Administrator\AppData\Local\Temp）再 run_shell_command 执行，跑完立即 del 删除。**Why:** 用户明确指示（"别用powershell -c了 把刚才那条py_compile换成.cmd临时脚本或.bat来跑 写到系统临时目录，跑完删除"）；另注意 PowerShell 不允许 `&` 串联命令（ParserError），多条命令用 `;` 分隔。**How to apply:** 本项目/Windows 会话中做 py_compile 批量校验、一次性冒烟测试时套用此模式。
- [2026-08-27 19:59:24] [2026-08-27] 用户决策：4K 步骤必须保持在字幕烧录之后执行（scale=3840:2160 不改）——字幕随 4K 等比放大是期望行为，否则字幕在 4K 下过小。**Why:** 审查报告曾建议改为先4K后字幕+等比缩放，用户明确否决。**How to apply:** 后续审查/重构不要再把“4K 放大 Pillow 字幕”当问题报告或改动 _step6_4k 的执行顺序。
- [2026-08-27 19:59:37] [2026-08-27] replace 工具在本仓库（CRLF 文件）使用教训：old_string 以换行结尾而 new_string 没有时，会把下一行合并进上一行造成 SyntaxError（当天踩坑2次：pipeline.py 的 checkpoint print 行、qwen_tts_engine.py 的 def 行）。**Why:** 尾部换行结构不一致被静默合并。**How to apply:** 编辑时让 old_string/new_string 保持相同的首尾换行结构；删大段代码优先用临时 python 脚本 + 唯一锚点正则 + assert 唯一性断言，写盘放最后保证可安全重跑；每次编辑后立即 py_compile 验证。

### Project
- [2026-08-27 00:42:36] MOSS-TTS-Nano 引擎集成在 colab_listening_b_web 项目。模型路径：H:\models\MOSS-TTS-Nano-Model (checkpoint), H:\models\MOSS-Audio-Tokenizer-Nano (tokenizer), H:\models\MOSS-TTS-Nano (repo,含 moss_tts_nano_runtime.py)。API 入口：NanoTTSService.synthesize(text, voice, mode="voice_clone", output_audio_path)。16 个内置预设音色 (Ava/Adam/Bella 等)。**Why:** 用户已安装模型，需集成为第三 TTS 引擎选项。**How to apply:** moss_tts_engine.py 中 _get_service 必须在 import NanoTTSService 前先 import torchaudio_sf_patch（修复 Windows torchcodec DLL 缺失），否则合成报错 "TorchCodec is required"。CPU 生成一句英文约 15s，中文约 9s。2026-08-27 质量增强关键事实：①MOSS 会生成 1-3s 病态停顿（句中/句尾死寂），_clean_sentence_audio 用 20ms 帧级 RMS<0.008 检测压缩（句中>0.6s→0.25s，句首→30ms，句尾→80ms），时长校验必须在压缩之后做；②a.m./Dr./Mr. 等缩写句点不能触发分句（正则 lookbehind），过短碎片送模型会呓语；③中文台词默认中文预设音色 Junhao(男)/Xiaoyu(女)，英文参考音说中文带口音。
- [2026-08-27 01:48:36] [2026-08-27] 视频结构模式仅保留4种：original / original_static / original_cutout / quest。quest_v2 与 shorts 已于 2026-08-27 应用户要求彻底删除（连 structures.py/audio_envelope.py 一起清掉），旧结构名 image/static/checkpoint 迁移垫片也已移除。**Why:** 用户原话"其他的模式删除，相关代码脚本全部清理干净，不要有一点残留"。**How to apply:** 后续会话不要再引用或恢复 quest_v2/shorts/唇同步管线；若见到含这些名字的历史产物/checkpoint 属已废弃数据；新增模式建议需先向用户确认。
- [2026-08-27 02:03:12] [2026-08-27 02:02] original_cutout 开头/结尾已对齐 quest 主持人形式（commit 302fdb2）：timeline.rewrite_title_card_as_host_segments 移除标题卡并插入 welcome+hook_intro（speaker=host），outro 也为主持人出镜定格动画（演播室 host_bg.png，复用 quest._render_sm_segment）；TTS 旁白键变为 welcome/hook/outro + practice_intro（listening 脚本新增 welcome_en/welcome_zh 字段）；新配置「主持人形象绑定」(--host-character, char_a/char_b) 绑定时复用该角色姿势图集与音色、不再生成独立 host 图集。**Why:** 用户要求 ch2/ch3 不动、开头结尾换成 Quest 主持人形式并可绑定角色形象。**How to apply:** 改 cutout 时间轴时勿重建 title_card；resume 校验按 narration_names 结构清单在 image_gen.check_step2_resume。
- [2026-08-27 19:59:31] [2026-08-27] 密钥泄露后续（2026-08-27 三批修复中已将 configs/{default,mode_*,ori,quest,test}.json git rm --cached + gitignore，本地文件保留）：Sens eNova/OpenAI API key 与 MCP token 仍留在 git 历史与 GitHub 远程，用户需自行到对应平台轮换凭据。**Why:** untrack 不能清除历史。**How to apply:** 用户确认已轮换前，涉及这些密钥的操作注意失效风险；轮换后提醒更新本地 default.json。
- [2026-08-27 21:03:17] [2026-08-27] Ch3 跟读参数化与角色复用修复（commit 54187fe）。①timeline.build_listening_timeline 新增 en_repeats/zh_repeats（0-10，可全 0→跳过 practice_intro），段序=EN×N+sil → ZH×M+sil，废弃旧"中文后再播一次英文"尾播；②三个 compose 函数新增 ch3_zh_always kwarg：为 True 时 en_{i}.png 直接渲染含中文（内容同 zh 帧），段内文件选择逻辑零改动；③配置键 ch3_en_repeats/ch3_zh_repeats/ch3_zh_always(默认True) 在 PARAM_SPEC video 组，quest 模式 config.html 隐藏、间隔沿用 practice_duration；④build_cli_args 缺失键 ch3_zh_always 必须 get(k, True)——否则直传 dict 时默认变关闭。**Why:** 用户确认 EN×N→ZH×M 结构与开关默认开。**How to apply:** 后续调整 Ch3 结构只改 timeline.py 双循环；中文常显走渲染层 flag。

### Reference

