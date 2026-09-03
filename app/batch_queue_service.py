"""批量生成队列 — 控制台多任务串行自动生成视频。

设计要点：
- 后端驻留队列（非前端排队）：每个视频 30-60+ 分钟，关闭浏览器不能中断队列；
  队列状态原子落盘 configs/batch_queue.json，页面刷新/服务重启不丢任务。
- 复用 PipelineService：worker 逐项调用 pipeline_service.get_service().start(config)
  并轮询至结束，日志/SSE/进度条/断点续传全部走现有链路。
- 配置解析时机 = 任务开始时：load_mode_config(mode) 取该模式当前最新配置，
  叠加内容覆盖项（topic/script_id/cefr/structure）——队列项固定"内容"，
  参数跟随对应模式当前配置（与控制台手动启动行为一致）。
- 串行互斥：同一时刻只有一个 run（复用 run_mutex）；mutex 被占（模式测试/
  4K/BGM/手动 run）时 worker 每 5s 重试等待，不标记失败。
- 暂停语义：当前视频完成后暂停；手动停止当前 run → 该项 stopped + 队列转暂停
  （手动停止=想停下来的意图；与"失败自动跳过继续"区分，仅 error 才跳过）。
"""
import json
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any

from .config_manager import MODES, load_mode_config

WEB_ROOT = Path(__file__).parent.parent.resolve()
QUEUE_PATH = WEB_ROOT / "configs" / "batch_queue.json"

ITEM_STATUSES = ("pending", "running", "done", "error", "stopped", "interrupted")
FINISHED_STATUSES = ("done", "error", "stopped", "interrupted")

MAX_QUEUE_ITEMS = 100
MAX_TOPIC_LEN = 200
MUTEX_WAIT_SECONDS = 5  # run_mutex 被占时的重试间隔

_queue_log = deque(maxlen=60)
_lock = threading.Lock()


def _log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    _queue_log.append(line)
    print(f"[BatchQueue] {msg}")


def _new_item_id() -> str:
    return f"bq_{int(time.time() * 1000)}_{uuid.uuid4().hex[:4]}"


class BatchQueueService:
    """串行批量队列：管理 items/state，worker 线程驱动 PipelineService。"""

    def __init__(self):
        self._items: list[dict[str, Any]] = []
        self._state = "idle"  # idle | running | paused
        self._pause_requested = False
        self._thread: threading.Thread | None = None
        self._load(recover=True)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self, recover: bool = False) -> None:
        if not QUEUE_PATH.exists():
            return
        try:
            data = json.loads(QUEUE_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            _log("队列文件损坏，已忽略（configs/batch_queue.json）")
            return
        items = data.get("items") or []
        if not isinstance(items, list):
            return
        self._items = [it for it in items if isinstance(it, dict) and it.get("id")]
        self._state = data.get("state", "idle")
        if recover and self._state == "running":
            # 上次服务退出时队列还在跑：正在跑的项标记 interrupted（其运行目录
            # 有 checkpoint 可手动续传），队列停在 paused 等用户确认后恢复
            for it in self._items:
                if it.get("status") == "running":
                    it["status"] = "interrupted"
                    it["error"] = "服务重启时中断（运行目录 checkpoint 可续传）"
                    it["finished_at"] = time.time()
            self._state = "paused"
            self._save()
            _log("检测到上次未完成的批量队列，已转为暂停状态")

    def _save(self) -> None:
        data = {"state": self._state, "items": self._items}
        tmp = QUEUE_PATH.with_suffix(".json.tmp")
        try:
            tmp.write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(QUEUE_PATH)
        except OSError as e:
            _log(f"队列落盘失败: {e}")

    # ------------------------------------------------------------------
    # Add items (validation)
    # ------------------------------------------------------------------

    def _pending_keys(self) -> set[tuple[str, str]]:
        """队列中尚未跑完的 (type, 内容键) 集合，用于去重。"""
        keys = set()
        for it in self._items:
            if it.get("status") in ("pending", "running"):
                if it.get("type") == "script":
                    keys.add(("script", str(it.get("script_id", ""))))
                else:
                    keys.add(("topic", f"{it.get('mode')}|{it.get('topic', '')}"))
        return keys

    def _reject_script(self, script_id: str) -> str:
        """脚本项预检：返回拒绝原因，空串=通过。"""
        from . import script_library
        doc = script_library.get_script_doc(script_id)
        if not doc:
            return f"脚本不存在: {script_id}"
        if doc.get("status") == "used":
            return f"脚本已被使用: {doc.get('topic', script_id)}"
        structure = doc.get("structure") or ""
        if structure not in MODES:
            return f"脚本结构无效: {structure}"
        if not (doc.get("script") or {}).get("dialogue"):
            return f"脚本无对话内容: {doc.get('topic', script_id)}"
        return ""

    def add_items(self, raw_items: Any) -> tuple[list[dict], list[dict]]:
        """批量入队。返回 (added_items, rejected[{index, reason}])。

        raw_items 元素: {type: "script"|"topic", mode, script_id?/topic?, cefr?}
        - script 项 mode 以脚本自身 structure 为准（权威）
        - topic 项 mode 必须是合法模式
        """
        from . import script_library

        if not isinstance(raw_items, list) or not raw_items:
            return [], [{"index": -1, "reason": "items 不能为空"}]

        added: list[dict] = []
        rejected: list[dict] = []
        pending_keys = self._pending_keys()
        added_keys: set[tuple[str, str]] = set()

        for i, raw in enumerate(raw_items):
            if not isinstance(raw, dict):
                rejected.append({"index": i, "reason": "格式错误"})
                continue
            itype = str(raw.get("type", "")).strip()
            cefr = str(raw.get("cefr", "") or "").strip()
            if len(self._items) + len(added) >= MAX_QUEUE_ITEMS:
                rejected.append({"index": i, "reason": f"队列已满（上限 {MAX_QUEUE_ITEMS} 项）"})
                continue

            if itype == "script":
                script_id = str(raw.get("script_id", "") or "").strip()
                reason = self._reject_script(script_id)
                if reason:
                    rejected.append({"index": i, "reason": reason})
                    continue
                key = ("script", script_id)
                if key in pending_keys or key in added_keys:
                    rejected.append({"index": i, "reason": "该脚本已在队列中"})
                    continue
                doc = script_library.get_script_doc(script_id) or {}
                script = doc.get("script") or {}
                mode = doc.get("structure") or ""
                item = {
                    "id": _new_item_id(),
                    "type": "script",
                    "mode": mode,
                    "script_id": script_id,
                    "topic": doc.get("topic") or script.get("title") or "",
                    "cefr": cefr or doc.get("cefr", ""),
                    "label": script.get("youtube_title") or script.get("title")
                             or doc.get("topic") or script_id,
                    "status": "pending",
                    "error": "", "run_name": "", "final_path": "",
                    "added_at": time.time(), "started_at": 0, "finished_at": 0,
                }
            elif itype == "topic":
                mode = str(raw.get("mode", "") or "").strip()
                topic = str(raw.get("topic", "") or "").strip()
                if mode not in MODES:
                    rejected.append({"index": i, "reason": f"无效模式: {mode or '(空)'}"})
                    continue
                if not topic:
                    rejected.append({"index": i, "reason": "主题为空"})
                    continue
                topic = topic[:MAX_TOPIC_LEN]
                key = ("topic", f"{mode}|{topic}")
                if key in pending_keys or key in added_keys:
                    rejected.append({"index": i, "reason": f"主题已在队列中: {topic}"})
                    continue
                item = {
                    "id": _new_item_id(),
                    "type": "topic",
                    "mode": mode,
                    "script_id": "",
                    "topic": topic,
                    "cefr": cefr,
                    "label": topic,
                    "status": "pending",
                    "error": "", "run_name": "", "final_path": "",
                    "added_at": time.time(), "started_at": 0, "finished_at": 0,
                }
            else:
                rejected.append({"index": i, "reason": f"未知类型: {itype}"})
                continue

            added.append(item)
            added_keys.add(key)

        if added:
            with _lock:
                self._items.extend(added)
                self._save()
            _log(f"入队 {len(added)} 项"
                 + (f"，拒绝 {len(rejected)} 项" if rejected else ""))
        return added, rejected

    # ------------------------------------------------------------------
    # Queue controls
    # ------------------------------------------------------------------

    def _find(self, item_id: str) -> dict | None:
        return next((it for it in self._items if it.get("id") == item_id), None)

    def start(self) -> tuple[bool, str]:
        """启动队列处理（state=idle → running；paused 用 resume）。"""
        with _lock:
            has_pending = any(it.get("status") == "pending" for it in self._items)
            if not has_pending:
                return False, "队列中没有待处理任务"
            if self._state == "running":
                return False, "队列已在运行中"
            self._state = "running"
            self._pause_requested = False
            self._save()
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(
                target=self._worker, name="batch-queue-worker", daemon=True)
            self._thread.start()
        _log("队列已启动")
        return True, "批量队列已启动"

    def pause(self) -> tuple[bool, str]:
        """请求暂停：当前项跑完后暂停（不打断正在运行的 run）。"""
        with _lock:
            if self._state != "running":
                return False, "队列未在运行中"
            self._pause_requested = True
        _log("已请求暂停（当前视频完成后生效）")
        return True, "将在当前视频完成后暂停"

    def resume(self) -> tuple[bool, str]:
        with _lock:
            if self._state != "paused":
                return False, "队列不处于暂停状态"
            self._state = "running"
            self._pause_requested = False
            self._save()
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(
                target=self._worker, name="batch-queue-worker", daemon=True)
            self._thread.start()
        _log("队列已恢复")
        return True, "批量队列已恢复"

    def remove(self, item_id: str) -> tuple[bool, str]:
        """移除单个未开始任务（仅 pending）。"""
        with _lock:
            it = self._find(item_id)
            if not it:
                return False, "任务不存在"
            if it.get("status") != "pending":
                return False, "只能移除未开始的任务"
            self._items.remove(it)
            self._save()
        _log(f"已移除队列项: {it.get('label', item_id)}")
        return True, "已移除"

    def clear(self, scope: str = "finished") -> tuple[bool, str, int]:
        """清理队列。scope=finished 移除已完成项；scope=all 清空整个队列
        （队列运行中拒绝 all，避免误清正在跑的任务）。返回 (ok, msg, removed)。"""
        with _lock:
            if scope == "all":
                if self._state == "running":
                    return False, "队列运行中，请先暂停或停止当前任务", 0
                removed = len(self._items)
                self._items = []
                self._state = "idle"
                self._pause_requested = False
            else:
                removed = len(self._items)
                self._items = [it for it in self._items
                               if it.get("status") not in FINISHED_STATUSES]
                removed -= len(self._items)
                if not self._items:
                    self._state = "idle"
                    self._pause_requested = False
            self._save()
        _log(f"已清理 {removed} 项（scope={scope}）")
        return True, f"已清理 {removed} 项", removed

    def stop_current(self) -> tuple[bool, str]:
        """停止当前正在跑的 run（worker 检测到 stopped 后将队列转暂停）。"""
        from . import pipeline_service
        service = pipeline_service.get_service()
        if not service.is_running:
            return False, "当前没有正在运行的任务"
        service.stop()
        _log("已请求停止当前任务（队列将转暂停）")
        return True, "正在停止当前任务"

    def is_blocking_manual_start(self) -> bool:
        """批量队列运行中时，拒绝控制台手动单次启动（避免互斥冲突）。"""
        with _lock:
            return self._state == "running"

    def status(self) -> dict:
        with _lock:
            counts: dict[str, int] = {}
            for it in self._items:
                st = it.get("status", "pending")
                counts[st] = counts.get(st, 0) + 1
            return {
                "state": self._state,
                "items": [dict(it) for it in self._items],
                "counts": counts,
                "pending": counts.get("pending", 0),
                "logs": list(_queue_log),
            }

    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------

    def _build_config(self, item: dict) -> dict | None:
        """任务开始时构建运行配置；脚本项二次校验失败返回 None（原因写 item.error）。"""
        from . import script_library

        mode = item.get("mode", "")
        if mode not in MODES:
            item["error"] = f"无效模式: {mode}"
            return None
        config = load_mode_config(mode)
        config["structure"] = mode
        if item.get("cefr"):
            config["cefr"] = item["cefr"]
        if item.get("type") == "script":
            doc = script_library.get_script_doc(str(item.get("script_id", "")))
            if not doc:
                item["error"] = "脚本库中未找到脚本（可能已删除）"
                return None
            if doc.get("status") == "used":
                item["error"] = f"脚本已被使用: {doc.get('topic', '')}"
                return None
            if (doc.get("structure") or "") != mode:
                item["error"] = f"脚本结构与模式不符: {doc.get('structure')} != {mode}"
                return None
            config["script_id"] = item["script_id"]
            if doc.get("topic"):
                config["topic"] = doc["topic"]
            if doc.get("cefr"):
                config["cefr"] = doc["cefr"]
        else:
            config["script_id"] = ""
            config["topic"] = item.get("topic", "")
        return config

    def _wait_for_resource(self, item: dict, config: dict) -> bool:
        """等待 run_mutex / PipelineService 空闲并启动 run。返回 False=等待期间
        用户请求暂停（任务回退 pending，队列转暂停）。"""
        from . import pipeline_service, run_mutex
        service = pipeline_service.get_service()
        while True:
            with _lock:
                if self._pause_requested or self._state != "running":
                    item["status"] = "pending"
                    item["started_at"] = 0
                    self._pause_requested = False
                    self._state = "paused"
                    self._save()
                    _log("等待资源期间队列已暂停，任务退回待处理")
                    return False
            owner = run_mutex.current_owner()
            if owner:
                _log(f"等待资源释放（占用方: {owner}）...")
            elif service.is_running:
                _log("等待手动运行结束...")
            elif service.start(config):
                return True
            # start 失败但资源已空闲 → 短暂竞争窗口，稍后重试
            time.sleep(MUTEX_WAIT_SECONDS)

    def _process_item(self, item: dict) -> None:
        """执行单个队列项：构建配置 → 等资源 → 启动 run → 轮询至结束 → 归档。"""
        from . import pipeline_service
        service = pipeline_service.get_service()

        config = self._build_config(item)
        if config is None:
            item["status"] = "error"
            item["finished_at"] = time.time()
            with _lock:
                self._save()
            _log(f"任务失败（跳过继续）: {item.get('label', '')} — {item.get('error', '')}")
            return
        # config 经实例属性传递给 _wait_for_resource（避免长签名）
        item["_config"] = config
        _log(f"开始生成: {item.get('label', '')}（{item.get('mode', '')}）")
        started = self._wait_for_resource(item, config)
        if not started:
            return

        # 轮询至 run 终态（done/error/stopped）
        while True:
            time.sleep(1)
            p = service.get_progress()
            if p.get("status") in ("done", "error", "stopped") and not p.get("is_running"):
                break

        final_status = p.get("status", "error")
        item["finished_at"] = time.time()
        item["run_name"] = p.get("run_name", "")
        item["final_path"] = p.get("final_path", "")
        if final_status == "done":
            item["status"] = "done"
        elif final_status == "stopped":
            item["status"] = "stopped"
            item["error"] = "用户手动停止"
        else:
            item["status"] = "error"
            item["error"] = p.get("error", "") or "未知错误"

        pause_queue = final_status == "stopped"
        with _lock:
            if pause_queue:
                self._state = "paused"
                self._pause_requested = False
            self._save()
        if final_status == "done":
            _log(f"完成: {item.get('label', '')} → {Path(item['final_path']).name if item['final_path'] else '(无产物路径)'}")
        elif pause_queue:
            _log(f"任务被手动停止: {item.get('label', '')}（队列已暂停）")
        else:
            _log(f"任务失败（跳过继续）: {item.get('label', '')} — {item.get('error', '')}")

    def _worker(self) -> None:
        while True:
            with _lock:
                if self._pause_requested or self._state != "running":
                    if self._state == "running":
                        self._state = "paused"
                    self._pause_requested = False
                    self._save()
                    break
                item = next(
                    (it for it in self._items if it.get("status") == "pending"), None)
                if item is None:
                    self._state = "idle"
                    self._save()
                    _log("队列已全部处理完成")
                    break
                item["status"] = "running"
                item["started_at"] = time.time()
                item["error"] = ""
                self._save()
            try:
                self._process_item(item)
            except Exception as e:  # noqa: BLE001 — 任何异常都不能杀死 worker
                item["status"] = "error"
                item["error"] = f"{type(e).__name__}: {e}"
                item["finished_at"] = time.time()
                with _lock:
                    self._save()
                _log(f"任务异常（跳过继续）: {item.get('label', '')} — {item['error']}")
        _log(f"worker 退出（state={self._state}）")


# Singleton
_service: BatchQueueService | None = None


def get_batch_queue() -> BatchQueueService:
    global _service
    if _service is None:
        _service = BatchQueueService()
    return _service
