"""缩略图重生成服务（独立子进程，可与主 pipeline 并行运行）。

旧实现走 pipeline_service 后台线程，与主 pipeline 共享 PipelineService 单例
状态 / sys.stdout / mcp_client 全局会话，因此必须互斥（run_mutex）。本服务把
生成逻辑放进 pipeline/thumbnail_regen_cli.py 独立子进程：进程级隔离天然消除
上述冲突，故不获取 run_mutex，主 pipeline 运行中也可给旧 run 生成缩略图。

单槽设计：全局同时只允许一个缩略图任务（与 runs.html _thumbRegenBusy 前端
假设一致）。日志经 PIPE 收集，GET /api/thumbnail_regen/status 增量轮询。
"""
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

from .config_manager import MODES, load_config, find_run_dir
from .paths import PIPELINE_DIR

_CLI_PATH = PIPELINE_DIR / "thumbnail_regen_cli.py"


class ThumbnailRegenService:
    """单槽子进程任务管理：start() 启动，get_status() 增量查询。"""

    def __init__(self):
        self._lock = threading.Lock()
        self._task: dict | None = None

    @property
    def _running(self) -> bool:
        t = self._task
        return bool(t) and t["status"] == "running"

    def start(self, run_name: str, mode: str = "") -> tuple[bool, str]:
        """为已有运行再生成一张缩略图（thumbnail_N.jpg 递增，旧图全部保留）。

        与主 pipeline / 模式测试 / 4K 生成并行（不获取 run_mutex）。
        返回 (ok, message)。
        """
        if self._running:
            return False, "已有缩略图生成任务进行中，请等待完成"

        config = load_config()
        output_dir = Path(config.get("output_dir", "./output"))
        run_dir = find_run_dir(output_dir, run_name, mode)
        if not run_dir:
            return False, f"运行不存在: {run_name}"
        script_path = run_dir / "script.json"
        if not script_path.exists():
            return False, "缺少 script.json"
        try:
            script = json.loads(script_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            return False, f"script.json 读取失败: {e}"

        # 同 run 守卫：该运行正在生成中时 images 可能未就绪，且 Step 4.5 会自动出图
        from .pipeline_service import get_service
        main_svc = get_service()
        if main_svc.is_running and main_svc.work_dir:
            try:
                if Path(main_svc.work_dir).resolve() == run_dir.resolve():
                    return False, "该运行正在生成中，Step 4.5 会自动生成缩略图，请等运行完成"
            except OSError:
                pass

        structure = script.get("structure", "")
        if structure not in MODES:
            structure = run_dir.parent.name if run_dir.parent.name in MODES else "original"

        # 角色参考图本地路径（子进程内再决定转 CDN URL 还是 base64）
        if structure in ("quest", "original_cutout"):
            ref_img = run_dir / "images" / "pose_char_a_0.png"
        else:
            ref_img = run_dir / "images" / "char_scene.png"

        # 输出文件名：无主图时补 thumbnail.jpg，否则 thumbnail_N.jpg 递增（旧图全保留）
        if not (run_dir / "thumbnail.jpg").exists():
            out_name = "thumbnail.jpg"
        else:
            n = 2
            while (run_dir / f"thumbnail_{n}.jpg").exists():
                n += 1
            out_name = f"thumbnail_{n}.jpg"

        # 生图 Provider / SenseNova key / 画面风格（style_manager 在 pipeline/ 下）
        from style_manager import resolve_style_prompt
        style_id = str(config.get("visual_style", "pixar3d"))
        tokens_raw = str(config.get("mcp_tokens", "") or "").strip()
        mcp_tokens = [t.strip() for t in tokens_raw.split("\n") if t.strip()]

        payload = {
            "run_dir": str(run_dir),
            "structure": structure,
            "out_name": out_name,
            "ref_img": str(ref_img) if ref_img.exists() else "",
            "provider": str(config.get("image_provider", "mcp")),
            "sensenova_api_key": str(config.get("sensenova_api_key", "") or "").strip(),
            "style_id": style_id,
            "style_prompt": resolve_style_prompt(style_id),
            "mcp_tokens": mcp_tokens,
        }

        try:
            proc = subprocess.Popen(
                [sys.executable, str(_CLI_PATH)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace",
                cwd=str(PIPELINE_DIR))
        except OSError as e:
            return False, f"子进程启动失败: {e}"

        with self._lock:
            self._task = {
                "status": "running",
                "logs": [],
                "error": "",
                "run_name": run_name,
                "mode": mode or "",
                "run_dir": str(run_dir),
                "out_name": out_name,
                "started_at": time.time(),
                "finished_at": 0,
                "_proc": proc,
            }

        threading.Thread(target=self._read_loop, args=(proc,), daemon=True).start()
        # ensure_ascii=True（默认）：payload 全 ASCII，子进程 stdin 无论用什么
        # 编码解码（GBK/locale）都得到同一字符串——run 名含中文/emoji 时
        # 子进程管道默认 GBK 解码 UTF-8 字节会产生乱码路径（FileNotFoundError）
        try:
            proc.stdin.write(json.dumps(payload) + "\n")
            proc.stdin.flush()
        except (OSError, ValueError):
            pass  # 读线程按退出码收尾
        finally:
            try:
                proc.stdin.close()
            except (OSError, ValueError):
                pass
        return True, f"缩略图生成已启动（将保存为 {out_name}）"

    def _read_loop(self, proc: subprocess.Popen):
        """后台读线程：逐行收集子进程日志，按退出码收尾。"""
        assert self._task is not None
        for line in proc.stdout:
            with self._lock:
                self._task["logs"].append(line.rstrip("\r\n"))
        code = proc.wait()
        with self._lock:
            if self._task.get("_proc") is proc:
                self._task["finished_at"] = time.time()
                if code == 0:
                    self._task["status"] = "done"
                else:
                    self._task["status"] = "error"
                    logs = self._task["logs"]
                    self._task["error"] = next(
                        (l for l in reversed(logs)
                         if "FAILED" in l or "ERROR" in l or "失败" in l),
                        f"子进程退出码 {code}")

    def get_status(self, since: int = 0) -> dict:
        """单槽任务状态 + 增量日志（形状对齐 /api/run/logs/since）。"""
        with self._lock:
            t = self._task
            if not t:
                return {"logs": [], "total": 0,
                        "status": {"status": "idle", "error": "",
                                   "run_name": "", "mode": "", "out_name": ""}}
            logs = t["logs"]
            total = len(logs)
            return {
                "logs": logs[max(0, since):],
                "total": total,
                "status": {
                    "status": t["status"],
                    "error": t["error"],
                    "run_name": t["run_name"],
                    "mode": t["mode"],
                    "out_name": t["out_name"],
                    "elapsed": int((t["finished_at"] or time.time()) - t["started_at"]),
                },
            }


# Singleton
_service: ThumbnailRegenService | None = None


def get_thumb_regen_service() -> ThumbnailRegenService:
    global _service
    if _service is None:
        _service = ThumbnailRegenService()
    return _service
