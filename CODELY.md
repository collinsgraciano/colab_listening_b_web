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
│   ├── matting.py                # MODNet AI 抠图（本地 ONNX，matting_engine 配置，白度法保留可选）
│   ├── sr_upscale.py             # realesr-animevideov3 AI 4K 超分（torch CUDA fp16 双管道流式，upscale_engine 配置）
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

- **按模式过滤：** 配置页只渲染当前模式实际消费的参数（`PARAM_SPEC` 条目 `modes` 标注，缺省=全模式生效；`effective_param_spec(mode)` 过滤，`param_effective_in_mode` 查询）。生效性以管线消费方为准——original: clip_duration/clip_concurrency；cutout: animation/dh_*/host_character/render_fps/workers/matting_engine；quest: quest_beat_lines/render_fps/workers/matting_engine；quest 无 Ch3/中文（practice_duration/ch3_*/character_zh_voices 仅 original*）；static 无特有项（animation 强制 none）。被过滤键保存后回落默认值（load_mode_config 合并补全），CLI 构建不受影响；新增管线参数须同步维护 modes 标注
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
- 本地 AI 引擎两件套（2026-08-30 上线；同日 digital_human 数字人引擎已应用户要求整体删除，代码/配置/UI 均无残留）：MODNet 抠图 `H:\models\modnet\`、AI 超分 `H:\models\upscaling\`（ModelScope 镜像下载，SHA256 校验过）；全部走「配置选项 + 原方法保留」：matting_engine=auto/modnet/white_threshold、upscale_engine=ffmpeg/ai（默认 ffmpeg）；无 MODNet 时整帧 cover 兜底
- 主题库矩阵策略：每个场景从多角度开采（顾客操作/员工POV/品牌具体/首次体验/小麻烦/文化问答），`pipeline/topics.json` 现 726 条 26 分类；topics_ai.py 生成 prompt 已内置矩阵思维

## Codely Structured Memories

### User

### Feedback
- [2026-08-26 20:02:43] User wants every code change committed and pushed to GitHub immediately after modification. **Why:** user explicitly asked to enforce this as a standing rule in CODELY.md. **How to apply:** after completing any code edit in this project, run git add + commit + push right away without being asked.
- [2026-08-27 01:48:36] CODELY.md 正文无法用 replace/write_file 直接编辑（工具报 "memory file" 错误），须走 Python 脚本字节级替换；本项目及多数仓库文件为 CRLF 换行，search 输出看不出 \r，模式串要先 python repr 探针确认行尾，并把 pattern 中 \n 统一转为 \r\n，且每处替换加 assert count==1 防静默错位。
- [2026-08-27 22:28:22] [2026-08-27 02:02] Windows 验证类命令的执行方式：不要用超长 `python -c "..."; ...` 单行，改用 write_file 写临时 .py（必要时配 .cmd）到系统临时目录（%TEMP%≈C:\Users\Administrator\AppData\Local\Temp）再 run_shell_command 执行，跑完立即 del 删除。**Why:** 用户明确指示（"别用powershell -c了 把刚才那条py_compile换成.cmd临时脚本或.bat来跑 写到系统临时目录，跑完删除"）；另注意 PowerShell 不允许 `&` 串联命令（ParserError），多条命令用 `;` 分隔。**How to apply:** 本项目/Windows 会话中做 py_compile 批量校验、一次性冒烟测试时套用此模式。[2026-08-27 22:28 补充] 带参调用 .cmd 用 `& "C:\...\pycheck.cmd" "arg1" "arg2"`（`cmd /c "path" "arg"` 的引号会被拼成一个路径报 CommandNotFound）；参数化 .cmd 内用 `:loop / if "%~1"=="" goto done / shift / goto loop` 一次校验多文件，避免每次重写临时脚本；curl.exe 输出不要直接管道给 python -c（编码/BOM 解析失败），先 -o 存文件再解析。

- [2026-08-27 19:59:24] [2026-08-27] 用户决策：4K 步骤必须保持在字幕烧录之后执行（scale=3840:2160 不改）——字幕随 4K 等比放大是期望行为，否则字幕在 4K 下过小。**Why:** 审查报告曾建议改为先4K后字幕+等比缩放，用户明确否决。**How to apply:** 后续审查/重构不要再把“4K 放大 Pillow 字幕”当问题报告或改动 _step6_4k 的执行顺序。
- [2026-08-27 19:59:37] [2026-08-27] replace 工具在本仓库（CRLF 文件）使用教训：old_string 以换行结尾而 new_string 没有时，会把下一行合并进上一行造成 SyntaxError（当天踩坑2次：pipeline.py 的 checkpoint print 行、qwen_tts_engine.py 的 def 行）。**Why:** 尾部换行结构不一致被静默合并。**How to apply:** 编辑时让 old_string/new_string 保持相同的首尾换行结构；删大段代码优先用临时 python 脚本 + 唯一锚点正则 + assert 唯一性断言，写盘放最后保证可安全重跑；每次编辑后立即 py_compile 验证。
- [2026-08-30 09:24:18] [2026-08-30] 验证 config.html 渲染字段时勿用 `data-field="{key}" in html` 子串断言——JS 里 querySelector('[data-field="animation"]') 字符串会造成误报。**Why:** 模式过滤测试首轮因此误判泄漏。**How to apply:** 用元素级正则提取真实表单控件：`re.findall(r'<(?:input|select|textarea)\\b[^>]*?data-field="([^"]+)"', html, re.DOTALL)` 再做集合断言（[^>]* 默认跨行，多行 select 标签也能匹配）。
- [2026-08-30 19:18:30] image_provider=sensenova 的 SENSENOVA_API_KEY 注入已与 LLM provider 解耦（commit 4d142bb）：_set_env 在 provider 分支之外统一注入 config.sensenova_api_key，CLI --api-key 同样提前注入；mode_test_service 复用 _set_env 一并生效。**Why:** 用户 llm_provider=custom(x666.me) + image_provider=sensenova 组合下，配置里已填的 key 被忽略导致 Step 2 生图 FATAL，报错文案误导用户重填。**How to apply:** 后续新增依赖 SENSENOVA_API_KEY 的功能直接读 env 即可，不要再把注入条件绑到 llm_provider；Web 运行中改 image_provider 无需重启服务（_set_env 每次运行都会重设 env）。

### Project
- [2026-08-27 00:42:36] MOSS-TTS-Nano 引擎集成在 colab_listening_b_web 项目。模型路径：H:\models\MOSS-TTS-Nano-Model (checkpoint), H:\models\MOSS-Audio-Tokenizer-Nano (tokenizer), H:\models\MOSS-TTS-Nano (repo,含 moss_tts_nano_runtime.py)。API 入口：NanoTTSService.synthesize(text, voice, mode="voice_clone", output_audio_path)。16 个内置预设音色 (Ava/Adam/Bella 等)。**Why:** 用户已安装模型，需集成为第三 TTS 引擎选项。**How to apply:** moss_tts_engine.py 中 _get_service 必须在 import NanoTTSService 前先 import torchaudio_sf_patch（修复 Windows torchcodec DLL 缺失），否则合成报错 "TorchCodec is required"。CPU 生成一句英文约 15s，中文约 9s。2026-08-27 质量增强关键事实：①MOSS 会生成 1-3s 病态停顿（句中/句尾死寂），_clean_sentence_audio 用 20ms 帧级 RMS<0.008 检测压缩（句中>0.6s→0.25s，句首→30ms，句尾→80ms），时长校验必须在压缩之后做；②a.m./Dr./Mr. 等缩写句点不能触发分句（正则 lookbehind），过短碎片送模型会呓语；③中文台词默认中文预设音色 Junhao(男)/Xiaoyu(女)，英文参考音说中文带口音。
- [2026-08-27 01:48:36] [2026-08-27] 视频结构模式仅保留4种：original / original_static / original_cutout / quest。quest_v2 与 shorts 已于 2026-08-27 应用户要求彻底删除（连 structures.py/audio_envelope.py 一起清掉），旧结构名 image/static/checkpoint 迁移垫片也已移除。**Why:** 用户原话"其他的模式删除，相关代码脚本全部清理干净，不要有一点残留"。**How to apply:** 后续会话不要再引用或恢复 quest_v2/shorts/唇同步管线；若见到含这些名字的历史产物/checkpoint 属已废弃数据；新增模式建议需先向用户确认。
- [2026-08-27 02:03:12] [2026-08-27 02:02] original_cutout 开头/结尾已对齐 quest 主持人形式（commit 302fdb2）：timeline.rewrite_title_card_as_host_segments 移除标题卡并插入 welcome+hook_intro（speaker=host），outro 也为主持人出镜定格动画（演播室 host_bg.png，复用 quest._render_sm_segment）；TTS 旁白键变为 welcome/hook/outro + practice_intro（listening 脚本新增 welcome_en/welcome_zh 字段）；新配置「主持人形象绑定」(--host-character, char_a/char_b) 绑定时复用该角色姿势图集与音色、不再生成独立 host 图集。**Why:** 用户要求 ch2/ch3 不动、开头结尾换成 Quest 主持人形式并可绑定角色形象。**How to apply:** 改 cutout 时间轴时勿重建 title_card；resume 校验按 narration_names 结构清单在 image_gen.check_step2_resume。
- [2026-08-27 19:59:31] [2026-08-27] 密钥泄露后续（2026-08-27 三批修复中已将 configs/{default,mode_*,ori,quest,test}.json git rm --cached + gitignore，本地文件保留）：Sens eNova/OpenAI API key 与 MCP token 仍留在 git 历史与 GitHub 远程，用户需自行到对应平台轮换凭据。**Why:** untrack 不能清除历史。**How to apply:** 用户确认已轮换前，涉及这些密钥的操作注意失效风险；轮换后提醒更新本地 default.json。
- [2026-08-27 21:03:17] [2026-08-27] Ch3 跟读参数化与角色复用修复（commit 54187fe）。①timeline.build_listening_timeline 新增 en_repeats/zh_repeats（0-10，可全 0→跳过 practice_intro），段序=EN×N+sil → ZH×M+sil，废弃旧"中文后再播一次英文"尾播；②三个 compose 函数新增 ch3_zh_always kwarg：为 True 时 en_{i}.png 直接渲染含中文（内容同 zh 帧），段内文件选择逻辑零改动；③配置键 ch3_en_repeats/ch3_zh_repeats/ch3_zh_always(默认True) 在 PARAM_SPEC video 组，quest 模式 config.html 隐藏、间隔沿用 practice_duration；④build_cli_args 缺失键 ch3_zh_always 必须 get(k, True)——否则直传 dict 时默认变关闭。**Why:** 用户确认 EN×N→ZH×M 结构与开关默认开。**How to apply:** 后续调整 Ch3 结构只改 timeline.py 双循环；中文常显走渲染层 flag。
- [2026-08-27 22:28:22] [2026-08-27 22:28] FastAPI 路由遮蔽陷阱（main.py /api/scripts）：新增静态子路径 GET（如 /api/scripts/batch_status）必须注册在 `/api/scripts/{sid}` 之前，否则被当作 sid 匹配返回该 handler 的 404 JSON（签名：返回 {"error":"脚本不存在"} 而非预期数据）。**Why:** 实测 batch_status 曾返回 404，移动注册顺序后修复；同文件 form_options 因注册在前而幸免。**How to apply:** 往已有 {param} 通配路由的路径组里加静态端点时，一律插到通配路由定义之前，并在代码处留注释（main.py 已留）。
- [2026-08-27 22:54:45] [2026-08-27 22:54] 行级 LLM prompt 字段已按模式裁剪（commit c2c8b0d）：original 只生成 video_prompt、original_static 只生成 image_prompt、original_cutout 全不生成；poses 字段彻底废弃（管线从未消费，唯一读取者 generate_pose_images 已删除）。quality_gate_listening.PROMPT_FIELDS 为矩阵权威定义，_resolve_structure 显式参数 > script.structure > original 兜底；generate_listening_script/run_listening_qa/run_listening_quality_gate 均新增 structure 参数（未知值回退 original）。**Why:** 用户确认"全部裁剪"方案，省数千 token/降低 JSON 截断风险。**How to apply:** 后续新增模式需同步更新 PROMPT_FIELDS、llm_client 三个 _prompt 辅助函数、script_library 批量生成调用；不要再把 poses/image_prompt(除static)/video_prompt(除original) 当必需字段加回；旧脚本兼容靠 .get() 兜底不受影响。
- [2026-08-28 00:37:18] 本仓库常有多个并行会话同时改代码：2026-08-28 本会话改的 pipeline.py 被 another 会话的 commit a9e829f（00:29:36，做 ch3_zh_repeats include_zh）顺带卷入提交，message 与实际内容部分不符。**Why:** 用户同时开多个 Codely 会话在同一 worktree 工作，git add 整文件会扫走别人的未提交改动。**How to apply:** 提交前用 git status/diff 核对每个改动文件是否真是自己改的，只 add 自己明确改过的文件；发现工作区出现非自己改动的内容不要回退、不要提交，向用户说明即可。
- [2026-08-28 00:45:08] [2026-08-28] YouTube 元数据规则定型（commit dd9c4bb）：①章节硬规则=首章必须 00:00 且相邻 ≥10s（违反则 YouTube 禁用全部章节）——save_youtube_metadata 用候选 marks+10s 过滤实现，5s title_card 永不成为章节、dialogue 钳到 00:00 兜片头、quest hook_intro 被过滤属预期；②用户决策：youtube_title 目标 55-95 字符（gate (40,95)、保存 >100 截断），描述尾部 hashtag 保持全部追加、不去重不限量；③thumbnail_prompt 死字段已彻底移除（同 poses 类别），不要再加回 prompt/流程；④新增 POST /api/runs/{name}/yt_meta_refresh 从 script.json+meta.json 秒级重生成 youtube_metadata.json。**Why:** quest/cutout 原章节缺 00:00 会被 YouTube 全禁；80-150 字符标题上传被截断；用户确认裁剪与"保持全部追加"方案。**How to apply:** 改 timeline 结构/章节 label 时保持两条章节硬规则；hashtag 策略不要再"优化"成限量；章节 label 无代码消费方依赖，可改文案。
- [2026-08-28 12:42:01] [2026-08-28 12:45] youtube_metadata.json 支持多样式选项（commit fa26c3f）：顶层字段不变，新增 title_options 数组（reference_v1 参考频道同款繁中标题/简介，含時間軸繁中章节注入+链接配置区）。用户决策：同款字段 youtube_title_ref/description_ref 由 Step 0 LLM 生成（不用纯模板）；gallery 页要选项卡片+复制按钮；链接区留可编辑配置（reference_v1.py 顶部 PLAYLISTS/RELATED_VIDEOS）。**Why:** 用户要为三个 original* 模式多一套参考频道风格供上传多选一，且以后会增加更多参考样式。**How to apply:** 扩展新样式 = pipeline/yt_meta_styles/ 丢新 .py（STYLE_ID/LABEL/STRUCTURES/build 契约，docstring 有说明）；注意 save_youtube_metadata 的 marks 元素是英文 label 非 seg_type；旧脚本无 ref 字段自动跳过该样式，勿当作 bug；quest 不生成选项。
- [2026-08-28 13:06:06] [2026-08-28 13:10] QA 循环语义定型（commit b8b8481）：配置 quest_qa_rounds（现名「QA 轮数（全模式）」）驱动全部 4 模式——同时写 QUEST_QA_MAX_ROUNDS（quest）与 LISTENING_QA_MAX_ROUNDS（original*；llm_review._env_int 必须走 llm_client._env_get，直接读 os.environ 会让批量生成的线程局部 override 失效）；循环=跑满设置轮数且每轮必跑双评审（story+language），轮满仍有 error 继续修到 0（硬上限 max(10, N)，防死循环），有 error 但无可行动 patch 提前 break（空转保护），0=关闭需 `is not None and != ""` 判断才写入 env。**Why:** 用户明确选择"无 error 也跑满设置轮数（深度审查，不提前结束）"+"硬上限 10 轮"（否决了无限循环和连续无改善即停），不要当 bug"优化"回旧的超限 best-effort 行为。**How to apply:** 改 QA 循环时保持 pipeline/llm_review.py 与 pipeline/quest/llm_client_quest.py Phase E 两处语义一致。
- [2026-08-28 23:06:52] [2026-08-28 23:10] 运行目录布局改为 output/{mode}/{run_name}/（4 模式各一文件夹），回收站为 output/_recycle_bin/（兼容旧 .recycle_bin）。**Why:** 用户要求不同模式分开保存。**How to apply:** 路径解析统一走 config_manager.find_run_dir/iter_run_dirs（兼容旧扁平布局）；运行名跨模式可能重名，runs/gallery 相关 API 支持 ?mode= 查询参数消歧；checkpoint.load_checkpoint 扫描时跳过回收站文件夹；commit a175090 附带提交了并行会话的 run_mutex.py（pipeline_service import 依赖，非本会话功能）。
- [2026-08-28 23:09:22] [2026-08-28 23:20] 「⚡ 模式效果测试」页已上线（/mode-test，commit 0c46d1c）：每模式一次性生成迷你素材（original* 4行/quest 8行）存 output/.mode_test/assets/{mode}/（active.json 指针 + test_assets.json 清单），之后经 _step5_compose 零积分本地合成测试视频；与主 pipeline 经 app/run_mutex.py 全局互斥。**Why:** 用户要随时零消耗对比各模式效果。**How to apply:** .mode_test 是隐藏测试目录——运行列表/iter_run_dirs 靠点前缀自动排除，勿"修复"成可见或当垃圾清理；素材合成结构参数读 manifest 快照（timeline 已按快照构建），勿改读当前配置；素材生成中断靠 checkpoint 续传，成功后服务会删 checkpoint。
- [2026-08-29 01:19:13] [2026-08-29 01:20] 生图 Provider 双通道上线（commit af59fd2）：image_provider 配置（mcp 默认 | sensenova=U1.5 Lite，API 计费复用 SENSENOVA_API_KEY）。用户决策：新增选项不替换 MCP、watermark=false（公测免费去水印）。关键实现事实：①U1.5 两个端点均为同步 JSON 接口——/v1/images/generations 纯文生图、/v1/images/edits 走 images[].image_url（公网 URL 或 data:image/*;base64，不支持 multipart/裸 base64/chat completions）；②返回 URL 仅 24h 有效→sensenova 路径统一"下载落盘→reupload_for_cdn 换 TOS 永久 URL"再进 image_urls（Seedance/resume 安全）；③U1.5 无 is_segmentation，quest/cutout 姿势图集靠 stop_motion 白度抠图 fallback；④图集类 prompt 必须 prompt_extend=False 防 "4x2/2x2 grid" 布局被润色改写；⑤缩略图 U1.5 失败直落 Pillow 不回退 MCP（防意外耗积分）。**Why:** MCP 积分易耗尽，SenseNova key 已有。**How to apply:** 尺寸映射在 pipeline/sensenova_image.py SIZE_MAP（16:9→2720x1536，图集 4096x2720/4096x2304）；改生图相关逻辑时两个 provider 分支都要维护。
- [2026-08-29 23:03:35] [2026-08-29 23:05] 句子限长约束链上线：配置 max_line_words（content 组，默认 10，clamp [4,20]，空/0→10）保证字幕最多两行。env 双通道 LISTENING_MAX_LINE_WORDS（original*）+ QUEST_MAX_LINE_WORDS（quest），注入点 pipeline_service._set_env / script_library 批量 override（线程局部，读端必须经 llm_client.resolve_max_line_words 的 _env_get）/ pipeline.py main()（CLI --max-line-words）。约束三层：①LLM prompt（对话行 HARD LIMIT + 旁白字段 welcome_en/story_hook/outro/practice_intro_en、quest hook_intro_en/outro 每句 ≤cap，CEFR 词数区间 hi=min(hi,cap)）；②质检门禁超长 warning→error 触发 QA 修复（quality_gate_listening._resolve_max_line_words / quest 同构）；③渲染兜底 media_utils._split_overlong_entries（burn_subtitles 内按实际字号测量，EN >2 行字幕条按词边界拆条+ZH 按字符比例+时长按字符占比，覆盖旧脚本/脚本库/QA 残留）。**Why:** 用户实测 15 词句字幕折 3 行影响学习。**How to apply:** 旁白超长是 warning 不是 error（旁白无行号不可 patch，error 会误触 QA 空转保护提前终止循环，勿"修复"成 error）；改字幕字号时兜底自动适配；quest _META_TARGETS 总词数目标（70-110/80-110）与每句 ≤cap 共存。
- [2026-08-30 01:04:31] [2026-08-30] 本机网络：huggingface.co 与 hf-mirror.com 均 TCP 不通，ModelScope（modelscope.cn）可达。**Why:** 下载模型权重必须优先找 ModelScope 镜像（AI-ModelScope/* 社区镜像或官方中文仓库）。**How to apply:** HF 直链失败时改用 https://www.modelscope.cn/models/{ns}/{name}/resolve/{rev}/{file}。
- [2026-08-30 21:57:39] [2026-08-30] 用户清单来源项目=Timeline Studio（github.com/MartinDelophy/ai-video-editor，MIT，浏览器 WebGPU 视频编辑器）；其 ONNX 模型镜像仓 martindelophy/timeline-studio-onnx-models（ModelScope+HF 双托管，revision a201b681）含 JoyVASA+LivePortrait 全部权重，SHA256 清单在仓库 src/config/{joyVasa,livePortrait}.js。**Why:** 本地 AI 能力的模型与推理逻辑移植自此项目（joyvasa.worker.js + liveportrait.worker.js）；其中 digital_human 数字人引擎已于 2026-08-30 应用户要求整体删除（代码/配置/UI 无残留，H:\models\digital_human 权重仍在但已无消费方）。**How to apply:** MODNet 抠图与 AI 超分仍在本仓库使用；LivePortrait 用纯 ONNX landmark 检测（无需 dlib/insightface，避开 Windows 编译坑）的结论仍有效。

- [2026-08-30 21:57:55] [2026-08-30] 本地 AI 能力上线（commit 7a2cf01/fb1565a/ea591ed）：①抠图 matting_engine=auto/modnet/white_threshold（MODNet ONNX H:\models\modnet，stop_motion.remove_bg 三分支，白度法保留）；②4K upscale_engine=ffmpeg/ai 默认 ffmpeg（realesr-animevideov3 torch CUDA fp16 ~110ms/帧，sr_upscale.py rawvideo 双管道流式，权重 H:\models\upscaling）；③digital_human 数字人（animation=digital_human，JoyVASA+LivePortrait）已于 2026-08-30 应用户要求整体删除（pipeline/digital_human.py、dh_quality/dh_neural_fps 参数、portrait 生图、config.html 显隐逻辑全清）。**Why:** 用户要求「保留原来的方法功能，可以在配置里设置用哪个和用不用」，随后又要求删除数字人。**How to apply:** 抠图/超分仍走 PARAM_SPEC→build_cli_args/_build_args→env/args 三层接线；不要把回退路径当 bug 删掉（权重缺失回退是设计行为）；不要再恢复 digital_human 相关代码或参数。

- [2026-08-30 21:58:08] [2026-08-30] animation 参数四模式真实语义（勿按旧文档理解）：original_cutout 下 none/landing/stop_motion 三值渲染结果完全相同（都走双人定格动画，合成器无 animation 分支——原 digital_human 分支已于同日删除）；original_static 管线强制 animation="none"（pipeline.py main + _build_args 双处，landing/stop_motion 实为死值）。**Why:** CODELY.md 架构段旧文档仍写 static「支持 none/landing/stop_motion」与实际不符。**How to apply:** 给 static 恢复动画需先解除两处强制；cutout 动画选项文案已如实标注等效（commit 61efb31）。

- [2026-08-30 13:33:13] [2026-08-30 13:35] moss-tts「视频第一句杂音」根因与修复（commit 848c714）：original_cutout 的 welcome.mp3 恒为 MOSS 引擎冷启动后第一次推理，超短句（如 "Welcome back!" 2词）首采样易产生满能量爆音/含混——历史 5 次 cutout 运行 4 次 welcome 首句呈 [≤12%,>75%,>75%,>75%]（帧RMS占p95%）无起音包络形态，近静音/时长校验均拦不住，成片忠实搬运即杂音。**Why:** 段级 afade 仅 50ms 压不住；下游 concat/loudnorm 链路经包络相关 0.975 验证无污染。**How to apply:** 修复在引擎层：①_get_service 后 _warm_up() 丢弃式预热（~2.7s，真实首句不再冷启动）；②_synthesize_sentence Validation 1b 用 _has_leading_burst（首4帧无起音形态检测，末次接受仅告警）+ 淡入 10ms→25ms；quest/original 首句音频结构不同（quest welcome=qwen 时无此问题；original 首句 dialogue_0 时模型已热）但切 moss 引擎同样受保护。检测器签名可复用：head[0]≤0.15 且 head[1..3]≥0.75（169 文件 0 误报，qwen 样本全负）。
- [2026-08-30 20:54:52] [2026-08-30 21:00] 缩略图「仍走 MCP 耗积分」根因与修复（commit b6cd1e2）：dashboard.html 的 MODE_CONFIGS 是页面加载时的 Jinja 配置快照，startPipeline() 直接把快照 POST /api/run/start，后端 data.config 优先——配置页刚保存的 image_provider=sensenova 被旧快照的 mcp 覆盖，缩略图/生图全走 MCP。**Why:** 用户实测缩略图未跟随「🎨MCP/图片」生图 Provider；thumbnail_gen.py 的 sensenova 分支本身无 bug。**How to apply:** ①startPipeline 现在先 fetch /api/config/all 刷新 MODE_CONFIGS 再合并快捷覆盖项（_doStartPipeline）——后续给 dashboard 加任何"启动前快捷覆盖"都必须走这个刷新流程，勿直接用页面快照；②配置新增 no_thumbnail checkbox（mcp 组，默认 False）：勾选=Step 4.5 只生成 youtube_metadata.json 不生成 thumbnail.jpg，接线=PARAM_SPEC/build_cli_args --no-thumbnail/_build_args.no_thumbnail，_step45_thumbnail 的 resume 判定是 `no_thumbnail or exists(thumb_path)`；③当前 4 个 mode_*.json 的 image_provider 都是 mcp（2026-08-30 20:47 状态），用户要省积分需自行切 sensenova。

### Reference

