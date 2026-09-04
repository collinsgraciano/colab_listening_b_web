"""运行历史本地批量操作队列 — 生成4K / 混BGM / 混BGM 4K 后端驻留串行执行。

设计要点：
- 后台标签页浏览器会节流 JS 定时器（隐藏约 5 分钟后最快 1 次/分钟），还可能冻结
  标签页，原「前端逐项排队」的批量在页面缩到后台时会变慢甚至停摆。本服务把排队
  调度搬进服务端线程：页面后台/刷新/关闭都不影响批量执行，前端只提交清单+轮询进度。
- 复用 PipelineService 现有任务方法（generate_4k / bgm_mix）：模式配置解析、
  run_mutex 互斥、日志、悬浮面板轮询链路全部不变；worker 逐项启动并轮询
  get_progress 至终态（与 batch_queue_service 轮询主 run 的方式一致）。
- 单槽：同一时刻只有一个本地批量在跑（运行中再次 start 返回失败）。
- stop 语义：完成当前项后停止，剩余项标记 skipped（不打断正在渲染的任务）；
  队列不落盘，服务重启即清空（正在渲染的任务随重启一并终止，属既有行为）。
"""
import threading
import time
from collections import deque
from typing import Any

KIND_LABELS = {"4k": "生成4K", "bgm": "混BGM", "bgm4k": "混BGM 4K"}
MAX_ITEMS = 200
MAX_NAME_LEN = 200
BUSY_RETRY_SECONDS = 3
BUSY_RETRY_MAX = 10  # 单项启动被占用时的重试上限（~30s，覆盖互斥锁收尾窗口与短暂占用）
POLL_SECONDS = 1.0

_batch_log: deque[str] = deque(maxlen=60)


def _push(msg: str) -> None:
    _batch_log.append(f"[{time.strftime('%H:%M:%S')}] {msg}")
    print(f"[LocalBatch] {msg}")


class LocalBatchService:
    """单槽串行批量：worker 线程逐项调用 PipelineService 任务方法并等待终态。"""

    def __init__(self):
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._state = "idle"  # idle | running | stopping | done | stopped
        self._kind = ""
        self._items: list[dict[str, Any]] = []
        self._current = ""
        self._stop_requested = False
        self._started_at = 0.0
        self._finished_at = 0.0

    # ------------------------------------------------------------------
    # Controls
    # ------------------------------------------------------------------

    def start(self, kind: str, raw_items: Any) -> tuple[bool, str]:
        if kind not in KIND_LABELS:
            return False, f"未知批量类型: {kind or '(空)'}"
        if not isinstance(raw_items, list) or not raw_items:
            return False, "items 不能为空"
        items: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for raw in raw_items[:MAX_ITEMS]:
            if not isinstance(raw, dict):
                continue
            name = str(raw.get("name", "")).strip()[:MAX_NAME_LEN]
            mode = str(raw.get("mode", "")).strip()
            if not name or (name, mode) in seen:
                continue
            seen.add((name, mode))
            items.append({"name": name, "mode": mode,
                          "status": "pending", "error": "", "output": ""})
        if not items:
            return False, "没有有效的运行项"
        with self._lock:
            if self._state in ("running", "stopping") and self._thread \
                    and self._thread.is_alive():
                return False, "已有本地批量任务进行中，请等待完成或先停止"
            self._kind = kind
            self._items = items
            self._current = ""
            self._stop_requested = False
            self._state = "running"
            self._started_at = time.time()
            self._finished_at = 0.0
        self._thread = threading.Thread(
            target=self._worker, name="local-batch-worker", daemon=True)
        self._thread.start()
        _push(f"本地批量已启动: {KIND_LABELS[kind]} × {len(items)} 项")
        return True, f"本地批量已启动（{len(items)} 项，串行执行）"

    def stop(self) -> tuple[bool, str]:
        with self._lock:
            if self._state != "running":
                return False, "没有进行中的本地批量"
            self._stop_requested = True
            self._state = "stopping"
        _push("已请求停止批量（当前任务完成后停止，剩余项跳过）")
        return True, "将在当前任务完成后停止"

    def status(self) -> dict:
        with self._lock:
            done_n = sum(1 for it in self._items
                         if it.get("status") in ("done", "error", "skipped"))
            return {
                "state": self._state,
                "running": self._state in ("running", "stopping"),
                "kind": self._kind,
                "total": len(self._items),
                "done": done_n,
                "current": self._current,
                "results": [dict(it) for it in self._items],
                "started_at": self._started_at,
                "finished_at": self._finished_at,
                "logs": list(_batch_log),
            }

    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------

    def _call_item(self, service, it: dict[str, Any]) -> tuple[bool, str]:
        if self._kind == "4k":
            return service.generate_4k(it["name"], mode=it["mode"])
        if self._kind == "bgm":
            return service.bgm_mix(it["name"], mode=it["mode"])
        return service.bgm_mix(it["name"], mode=it["mode"], target="4k")

    def _start_item(self, service, it: dict[str, Any]) -> bool:
        """启动单项；互斥被占时重试（与原前端 409 重试语义一致，上限更宽）。"""
        for attempt in range(1, BUSY_RETRY_MAX + 1):
            ok, msg = self._call_item(service, it)
            if ok:
                return True
            if ("资源被占用" in msg or "正在运行中" in msg) \
                    and attempt < BUSY_RETRY_MAX:
                _push(f"等待资源释放（{it['name']}，第 {attempt} 次）...")
                time.sleep(BUSY_RETRY_SECONDS)
                continue
            it["status"] = "error"
            it["error"] = msg
            _push(f"启动失败（跳过）: {it['name']} — {msg}")
            return False
        return False

    def _wait_item(self, service, it: dict[str, Any]) -> None:
        """轮询共享 PipelineService 状态至终态（与 batch_queue 轮询方式一致）。"""
        while True:
            time.sleep(POLL_SECONDS)
            p = service.get_progress()
            if p.get("status") in ("done", "error", "stopped") \
                    and not p.get("is_running"):
                break
        st = p.get("status")
        it["output"] = p.get("final_path", "")
        if st == "done":
            it["status"] = "done"
            _push(f"完成: {it['name']}")
        elif st == "stopped":
            it["status"] = "error"
            it["error"] = "任务被手动停止"
            _push(f"任务被手动停止: {it['name']}")
        else:
            it["status"] = "error"
            it["error"] = p.get("error", "") or "任务执行失败"
            _push(f"任务失败（跳过）: {it['name']} — {it['error']}")

    def _worker(self) -> None:
        from . import pipeline_service
        service = pipeline_service.get_service()
        try:
            for it in self._items:
                with self._lock:
                    if self._stop_requested:
                        it["status"] = "skipped"
                        it["error"] = "用户停止批量"
                        continue
                    it["status"] = "running"
                    self._current = it["name"]
                try:
                    if self._start_item(service, it):
                        self._wait_item(service, it)
                except Exception as e:  # noqa: BLE001 — 单项异常不杀死 worker
                    it["status"] = "error"
                    it["error"] = f"{type(e).__name__}: {e}"
                    _push(f"任务异常（跳过）: {it['name']} — {it['error']}")
        finally:
            with self._lock:
                self._state = "stopped" if self._stop_requested else "done"
                self._finished_at = time.time()
                self._current = ""
            ok_n = sum(1 for it in self._items if it.get("status") == "done")
            _push(f"本地批量结束（state={self._state}，成功 {ok_n}/{len(self._items)}）")


# Singleton
_service: LocalBatchService | None = None


def get_local_batch() -> LocalBatchService:
    global _service
    if _service is None:
        _service = LocalBatchService()
    return _service
