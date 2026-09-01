"""缩略图重生成 CLI（独立子进程入口，供 Web 并行调用）。

Web 端 app/thumbnail_regen_service.py 通过 subprocess 启动本脚本：
- 参数经 stdin 传一行 UTF-8 JSON（避免 argv 中文/emoji run 目录名乱码）
- 日志直接 print 到 stdout，由 Web 层 PIPE 收集展示
- 退出码 0=成功 / 1=失败

与主 pipeline 完全进程隔离：独立 sys.stdout、独立 mcp_client 会话，
可与正在运行的 pipeline / 模式测试 / 4K 生成并行（不获取 run_mutex）。
"""
import json
import os
import sys
import traceback
from pathlib import Path

# Windows GBK 控制台安全输出
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")
# stdin 显式 UTF-8：Windows 子进程管道默认按 locale（GBK）解码，而父进程
# 写入的是 UTF-8 字节——中文 run 名会变乱码、emoji 变 \udcXX 代理转义，
# 导致 run_dir 路径 FileNotFoundError（双保险：父进程还用 ensure_ascii=True）
sys.stdin.reconfigure(encoding="utf-8", errors="replace")

_PARENT = str(Path(__file__).parent.resolve())
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)


def main() -> int:
    params = json.loads(sys.stdin.readline() or "{}")
    run_dir = Path(params["run_dir"])
    structure = params.get("structure", "original")
    out_name = params["out_name"]
    ref_img = params.get("ref_img", "")
    provider = params.get("provider", "mcp")
    style_id = params.get("style_id", "pixar3d")
    style_prompt = params.get("style_prompt", "")

    print("=" * 60)
    print(f"ThumbnailRegen: {run_dir.name} -> {out_name}")

    # 先读脚本（路径错误快速失败，不白初始化 MCP / 上传参考图）
    script = json.loads((run_dir / "script.json").read_text(encoding="utf-8"))

    # env 注入（与 Web 端 _set_env 的最小子集一致）
    os.environ["IMAGE_PROVIDER"] = provider
    if params.get("sensenova_api_key"):
        os.environ["SENSENOVA_API_KEY"] = params["sensenova_api_key"]
    os.environ["VISUAL_STYLE_ID"] = style_id
    if style_prompt:
        os.environ["VISUAL_STYLE_PROMPT"] = style_prompt

    if provider == "mcp":
        from mcp_client import initialize as mcp_initialize
        tokens = [t.strip() for t in params.get("mcp_tokens", []) if t.strip()]
        mcp_initialize(tokens=tokens or None)
        char_scene_url = ""
        if ref_img:
            from image_gen import reupload_for_cdn
            try:
                char_scene_url = reupload_for_cdn(ref_img, Path(ref_img).name) or ""
            except Exception as e:
                print(f"[ThumbnailRegen] 参考图上传失败，退回无参考图生成: {e}")
    else:
        # sensenova edit_image 内部 _to_image_url 把本地路径转 base64
        char_scene_url = ref_img

    from pipeline import call_tool, parse_task_id, poll_task, download_file
    from thumbnail_gen import generate_thumbnail

    # 场景图：仅 Pillow 兜底分支需要；AI 分支纯 prompt 生成不受影响
    if structure == "quest":
        scene_img = run_dir / "images" / "scene_0.png"
        if not scene_img.exists():
            scene_img = run_dir / "images" / "scene.png"
    else:
        scene_img = run_dir / "images" / "scene.png"

    out_path = generate_thumbnail(
        script=script,
        scene_img=str(scene_img),
        output_path=str(run_dir / out_name),
        mcp_call_tool=call_tool,
        mcp_parse_task_id=parse_task_id,
        mcp_poll_task=poll_task,
        mcp_download_file=download_file,
        structure=structure,
        char_scene_url=char_scene_url,
    )
    if out_path and Path(out_path).exists():
        print("=" * 60)
        print(f"ThumbnailRegen DONE! {out_name}")
        return 0
    print("ThumbnailRegen FAILED: AI 生成与 Pillow 兜底均未产出文件"
          "（检查生图 Provider 配置 / MCP token / images 场景图）")
    return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception:
        print(f"ThumbnailRegen ERROR: {traceback.format_exc()}")
        sys.exit(1)
