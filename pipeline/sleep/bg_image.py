"""sleep 模式背景图生成：按本期主题 AI 生成 1 张（低透明度衬底用）。

- 双通道跟随 sensenova_image.get_image_provider()：sensenova 直接 HTTP 生成
  （本地消费无需 CDN 重传）；mcp 由调用方传入 call_tool 等函数（Step 2 对
  sleep 按需初始化 MCP，照缩略图先例）；
- 文件级续传：images/sleep_bg.png 已存在即跳过（resume 零消耗）；
- 失败非致命：背景图是增强功能，任何失败只告警并返回 ""，卡片回退纯渐变，
  不中断与生图无关的 sleep 运行。
"""
import os
from pathlib import Path


def build_bg_prompt(topic: str, style_prompt: str = "") -> str:
    """主题 → 背景图 prompt（静谧睡前氛围，无文字无人物）。"""
    t = str(topic or "").strip() or "everyday life"
    parts = [
        f"Calm dreamy pastel scene related to: {t}.",
        "Soft muted colors, gentle diffused night lighting, peaceful relaxing "
        "sleep atmosphere, clean simple composition, wide shot.",
        "no text, no words, no letters, no watermark, no people",
    ]
    if style_prompt:
        parts.append(style_prompt)
    return ", ".join(parts)


def ensure_sleep_bg_image(img_dir: str, topic: str, style_prompt: str = "",
                          mcp_call_tool=None, mcp_parse_task_id=None,
                          mcp_poll_task=None, mcp_download_file=None) -> str:
    """确保 sleep 背景图存在。返回图片绝对路径；失败/关闭回退返回 ""。

    调用方（pipeline._step2_images_tts sleep 分支）在 sleep_bg_image 开启且
    未配置固定路径时调用；固定路径分支在调用方处理（无需生成）。
    """
    dest = str(Path(img_dir) / "sleep_bg.png")
    if os.path.exists(dest) and os.path.getsize(dest) > 10_000:
        print("  [SleepBG] sleep_bg.png 已存在，跳过生成（resume 续传）")
        return dest

    import sensenova_image

    prompt = build_bg_prompt(topic, style_prompt)
    try:
        if sensenova_image.get_image_provider() == "sensenova":
            print("  [SleepBG] SenseNova 生成背景图中（2720x1536）...")
            url = sensenova_image.text_to_image(prompt, size="2720x1536",
                                                output_format="png")
            if not url or not sensenova_image.download_image(url, dest):
                raise RuntimeError("下载落盘失败")
        else:
            if not all((mcp_call_tool, mcp_parse_task_id, mcp_poll_task,
                        mcp_download_file)):
                print("  [SleepBG] MCP 通道未初始化 —— 跳过背景图生成（卡片回退纯渐变）")
                return ""
            print("  [SleepBG] MCP 生成背景图中（landscape_16_9）...")
            result = mcp_call_tool("generate_image", {
                "prompt": prompt,
                "provider": "seedream",
                "image_size": "landscape_16_9",
                "output_format": "png",
            })
            task_id = mcp_parse_task_id(result)
            if not task_id:
                raise RuntimeError("MCP 未返回任务 ID")
            data = mcp_poll_task(task_id, interval=10, max_wait=600)
            url = data.get("url", "")
            if not url or not mcp_download_file(url, dest):
                raise RuntimeError(f"生成失败（status={data.get('status')}）")
    except Exception as e:  # noqa: BLE001 — 增强功能失败不中断 sleep 运行
        print(f"  [SleepBG] WARNING: 背景图生成失败（忽略，卡片回退纯渐变）: {e}")
        return ""

    if not os.path.exists(dest) or os.path.getsize(dest) < 10_000:
        print("  [SleepBG] WARNING: 背景图文件无效（忽略）")
        return ""
    print(f"  [SleepBG] 背景图已生成: {dest}")
    return dest
