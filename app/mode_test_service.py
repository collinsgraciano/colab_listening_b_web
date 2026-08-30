"""模式效果测试服务：一次性生成各模式迷你素材，之后零积分本地合成测试视频。

两阶段流程（均与主 pipeline 通过 run_mutex 互斥）：

  素材生成（消耗 LLM/MCP/TTS，每模式只需一次）：
    step0(LLM mini 脚本) → step2(图片/姿势图集 + TTS) → step3(仅 original 视频片段)
    → step4(timeline + subtitles/meta.json，本地) → 写 test_assets.json 清单 + active.json 指针

  测试合成（纯本地 FFmpeg/Pillow，随时可跑、零消耗）：
    读素材集 script.json + subtitles/meta.json 恢复 timeline/narration/paths
    → 重建 clip_paths/group_info（仅 original）→ 直调 _step5_compose
    （其内部已按 structure 分发 4 种 compose，并重建姿势图/host/场景等）

目录布局（不进运行历史——main.py 各列表已排除 TEST_DIRNAME）：
    output/.mode_test/
    ├── used_topics.json                  # 测试主题防重复（不污染主库）
    ├── llm_debug/
    └── assets/{mode}/
        ├── active.json                   # {"set": "<素材集目录名>"} 当前生效指针
        └── {safe_title}/                 # _step0 原生 work_dir 格式
            ├── script.json  test_assets.json
            ├── images/ clips/ audio/  subtitles/meta.json
            └── *.mp4                     # 测试合成产物（set_dir 根，与主流程一致）
"""
import json
import os
import re
import shutil
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from .config_manager import load_config, load_mode_config
from .pipeline_service import PipelineService, STEP_PATTERNS, _LineBuffer
from . import run_mutex

TEST_DIRNAME = ".mode_test"
MINI_LINES = {"original": 4, "original_static": 4, "original_cutout": 4, "quest": 8}
MODES = list(MINI_LINES)


def test_root() -> Path:
    return Path(load_config().get("output_dir", "./output")).resolve() / TEST_DIRNAME


class ModeTestService:
    """模式测试编排：单线程槽位（素材生成 / 批量合成二选一）+ SSE 日志。"""

    def __init__(self):
        self._thread: threading.Thread | None = None
        self._stop_flag = threading.Event()
        self._lock = threading.Lock()
        self.log_lines: list[str] = []
        self.status: str = "idle"          # idle|running|done|error|stopped
        self._phase: str = "idle"          # idle|generate|compose
        self._mutex_owner: str = ""        # 启动时捕获，finally 按此释放
        self._current_mode: str = ""
        self._current_step_label: str = ""
        self._compose_modes: list[str] = []
        self._compose_done: list[str] = []
        self.started_at: float = 0
        self.finished_at: float = 0
        self.error: str = ""
        self.work_dir: str = ""
        # 复用主 pipeline 服务的纯辅助方法（_set_env/_build_args/_build_character_overrides），
        # 独立实例不触碰 get_service() 单例的运行状态
        self._svc = PipelineService()

    # ------------------------------------------------------------------
    # 基础状态
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _asset_root(self, mode: str) -> Path:
        return test_root() / "assets" / mode

    def _active_set(self, mode: str) -> Path | None:
        """active.json 指向的素材集目录（不存在或残缺返回 None）。"""
        root = self._asset_root(mode)
        active_path = root / "active.json"
        if not active_path.exists():
            return None
        try:
            set_name = json.loads(
                active_path.read_text(encoding="utf-8")).get("set", "")
        except (json.JSONDecodeError, OSError):
            return None
        if not set_name:
            return None
        set_dir = root / set_name
        if not (set_dir / "script.json").exists():
            return None
        return set_dir

    def _on_log_line(self, line: str):
        with self._lock:
            self.log_lines.append(line)
            if len(self.log_lines) > 5000:
                self.log_lines = self.log_lines[-3000:]
        for pattern, _step_id, step_label in STEP_PATTERNS:
            if line and re.search(pattern, line):
                with self._lock:
                    self._current_step_label = step_label
                break

    def get_progress(self) -> dict:
        with self._lock:
            elapsed = 0
            if self.started_at:
                end = self.finished_at if self.finished_at else time.time()
                elapsed = int(end - self.started_at)
            return {
                "status": self.status,
                "phase": self._phase,
                "current_mode": self._current_mode,
                "current_step_label": self._current_step_label,
                "compose_modes": list(self._compose_modes),
                "compose_done": list(self._compose_done),
                "elapsed": elapsed,
                "is_running": self.is_running,
                "error": self.error,
                "work_dir": self.work_dir,
                "mutex_owner": run_mutex.current_owner(),
            }

    def get_logs_since(self, since: int = 0) -> list[str]:
        with self._lock:
            if since < len(self.log_lines):
                return self.log_lines[since:]
        return []

    def stop(self):
        self._stop_flag.set()

    def _fail(self, msg: str):
        with self._lock:
            self.status = "error"
            self.error = msg
            self.finished_at = time.time()
        self._on_log_line(f"\nFATAL: {msg}")

    def _finish(self, status: str):
        with self._lock:
            self.status = status
            self.finished_at = time.time()

    # ------------------------------------------------------------------
    # 启动入口
    # ------------------------------------------------------------------

    def start_generate(self, mode: str, topic: str = "", force: bool = False) -> tuple[bool, str]:
        if mode not in MODES:
            return False, f"未知模式: {mode}"
        if self.is_running:
            return False, "测试任务运行中，请先停止或等待完成"
        if not run_mutex.try_acquire("mode_test:generate"):
            return False, f"资源被占用：{run_mutex.current_owner()}（主 pipeline 运行中请等待完成）"

        self._stop_flag.clear()
        self._mutex_owner = "mode_test:generate"
        with self._lock:
            self.log_lines = []
            self.status = "running"
            self._phase = "generate"
            self._current_mode = mode
            self._compose_modes = []
            self._compose_done = []
            self.started_at = time.time()
            self.finished_at = 0
            self.error = ""
            self.work_dir = ""

        self._thread = threading.Thread(
            target=self._thread_wrapper,
            args=(self._gen_run, mode, topic, bool(force)),
            daemon=True)
        self._thread.start()
        return True, "素材生成已启动"

    def start_compose(self, modes: list[str]) -> tuple[bool, str]:
        modes = [m for m in modes if m in MODES]
        if not modes:
            return False, "未选择任何模式"
        if self.is_running:
            return False, "测试任务运行中，请先停止或等待完成"
        missing = [m for m in modes if self._active_set(m) is None]
        if missing:
            return False, f"以下模式尚未生成素材: {', '.join(missing)}"
        if not run_mutex.try_acquire("mode_test:compose"):
            return False, f"资源被占用：{run_mutex.current_owner()}（主 pipeline 运行中请等待完成）"

        self._stop_flag.clear()
        self._mutex_owner = "mode_test:compose"
        with self._lock:
            self.log_lines = []
            self.status = "running"
            self._phase = "compose"
            self._current_mode = modes[0]
            self._compose_modes = list(modes)
            self._compose_done = []
            self.started_at = time.time()
            self.finished_at = 0
            self.error = ""
            self.work_dir = ""

        self._thread = threading.Thread(
            target=self._thread_wrapper,
            args=(self._compose_run, list(modes)),
            daemon=True)
        self._thread.start()
        return True, "测试合成已启动"

    def _thread_wrapper(self, fn, *args):
        """统一线程外壳：无论 fn 在哪一步抛异常都保证释放互斥锁。"""
        try:
            fn(*args)
        except Exception as e:
            self._fail(f"{type(e).__name__}: {e}")
            import traceback
            for line in traceback.format_exc().split("\n"):
                self._on_log_line(line)
        finally:
            run_mutex.release(self._mutex_owner)
            with self._lock:
                if self.status == "running":
                    self.status = "done"
                self.finished_at = time.time()

    # ------------------------------------------------------------------
    # 阶段一：素材生成（mini 全流程）
    # ------------------------------------------------------------------

    def _gen_run(self, mode: str, topic_input: str, force: bool):
        old_stdout = sys.stdout
        buf = _LineBuffer(self._on_log_line)
        sys.stdout = buf
        try:
            root = self._asset_root(mode)
            print("=" * 60)
            print(f"[ModeTest] 「{mode}」迷你测试素材生成（{MINI_LINES[mode]} 行对话）...")

            if force and root.exists():
                print(f"  [Reset] 删除旧素材目录: {root}")
                shutil.rmtree(root, ignore_errors=True)

            from checkpoint import load_checkpoint as _load_checkpoint
            root.mkdir(parents=True, exist_ok=True)
            checkpoint = _load_checkpoint(root)
            if checkpoint:
                print(f"  [Resume] 检测到未完成的素材生成，继续上次进度...")

            # 配置 + env（复用主 pipeline 的注入逻辑：LLM provider / 画面风格 /
            # QA 旋钮 / TTS 模型路径 / HF_ENDPOINT 等）
            config = dict(load_mode_config(mode))
            config["structure"] = mode
            self._svc._set_env(config)
            overrides = self._svc._build_character_overrides(config)
            if overrides:
                os.environ["CHARACTER_OVERRIDES"] = json.dumps(
                    overrides, ensure_ascii=False)
                print(f"  [CharOverride] {len(overrides)} 个角色预定义: "
                      f"{', '.join(overrides.keys())}")
            else:
                os.environ.pop("CHARACTER_OVERRIDES", None)

            args = self._svc._build_args(config)
            args.structure = mode
            args.num_lines = MINI_LINES[mode]
            t_root = test_root()
            used_topics_file = t_root / "used_topics.json"
            t_root.mkdir(parents=True, exist_ok=True)
            args.used_topics_file = str(used_topics_file)
            args.output = str(root)
            args.no_4k = True
            os.environ["LLM_DEBUG_DIR"] = str(t_root / "llm_debug")

            # 主题：resume 沿用 checkpoint 记录；否则用户指定 or 随机
            # （写入 .mode_test 的 used_topics，与主库隔离）
            topic = ""
            if checkpoint.get("topic"):
                topic = checkpoint["topic"]
                print(f"  [Topic] Resume 沿用 checkpoint 主题: '{topic}'")
            elif (topic_input or "").strip():
                topic = topic_input.strip()
                print(f"  [Topic] 使用指定主题: '{topic}'")
            else:
                from topic_manager import pick_random_topic
                topic = pick_random_topic(
                    args.topics_file, str(used_topics_file), mark=False)
                if not topic:
                    self._fail("没有可用主题（topics.json 为空或该模式已全部用尽）")
                    return
                print(f"  [Topic] 随机主题: '{topic}'")
            args.topic = topic

            # Step 0: LLM mini 脚本（resume 时自动加载已有 script.json）
            from pipeline import (
                _step0_script, _step2_images_tts, _step3_clips, _step4_timeline)
            script, work_dir, dirs = _step0_script(
                args, checkpoint, topic, root, str(used_topics_file))
            self.work_dir = str(work_dir)

            if self._stop_flag.is_set():
                self._finish("stopped")
                self._on_log_line("\n⏹ 素材生成已停止（checkpoint 已保留，可继续生成）")
                return

            # Step 1: MCP init（多 token 轮换）
            from mcp_client import reinitialize as mcp_reinit
            raw_tokens = args.mcp_tokens or ""
            tokens = ([t.strip() for t in raw_tokens.split(",") if t.strip()]
                      if raw_tokens else [])
            print("\n" + "=" * 60)
            print("Step 1: Initializing TJGenerators MCP...")
            if tokens:
                mcp_reinit(tokens=tokens)
            else:
                mcp_reinit()
            print("  MCP connected.")

            if self._stop_flag.is_set():
                self._finish("stopped")
                self._on_log_line("\n⏹ 素材生成已停止（checkpoint 已保留，可继续生成）")
                return

            # Step 2: 图片/姿势图集 + TTS（并发，内部按模式区分）
            ctx = _step2_images_tts(
                args, checkpoint, script, work_dir, dirs,
                stop_check=self._stop_flag.is_set)

            if self._stop_flag.is_set():
                self._finish("stopped")
                self._on_log_line("\n⏹ 素材生成已停止（checkpoint 已保留，可继续生成）")
                return

            # Step 3: 视频片段（仅 original；其余模式内部直接跳过）
            clip_paths, group_info, line_to_group = _step3_clips(
                args, checkpoint, work_dir, dirs, script, ctx,
                stop_check=self._stop_flag.is_set)

            if self._stop_flag.is_set():
                self._finish("stopped")
                self._on_log_line("\n⏹ 素材生成已停止（checkpoint 已保留，可继续生成）")
                return

            # Step 4: timeline + SRT + meta.json（本地，TTSEngine 仅用于 ffprobe 探测时长）
            _step4_timeline(args, checkpoint, script, work_dir, dirs,
                            ctx["tts_results"])

            # 写素材清单 + 激活指针
            manifest = {
                "structure": mode,
                "topic": topic,
                "num_lines": args.num_lines,
                "cefr": args.cefr,
                "set_name": work_dir.name,
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "visual_style": args.visual_style,
                "animation": args.animation,
                "dh_quality": getattr(args, "dh_quality", "preview"),
                "dh_neural_fps": int(getattr(args, "dh_neural_fps", 3) or 3),
                "pad": args.pad,
                "practice_duration": args.practice_duration,
                "clip_duration": args.clip_duration,
                "ch3_en_repeats": args.ch3_en_repeats,
                "ch3_zh_repeats": args.ch3_zh_repeats,
                "ch3_zh_always": args.ch3_zh_always,
                "ch3_practice_intro_show": bool(
                    getattr(args, "ch3_practice_intro_show", True)),
                "render_fps": args.render_fps,
                "workers": args.workers,
                "host_character": args.host_character,
                "tts_engine": args.tts_engine,
                "tts_rate": args.tts_rate,
                "dialogue_durations": ctx["tts_results"].get(
                    "dialogue_durations", []),
                "n_groups": len(group_info),
                "clip_files": [Path(p).name for p in clip_paths if p],
            }
            (work_dir / "test_assets.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2),
                encoding="utf-8")
            (root / "active.json").write_text(
                json.dumps({"set": work_dir.name}, ensure_ascii=False),
                encoding="utf-8")
            # 素材生成完成：清除 checkpoint（测试集不支持 step5 之后的 resume，
            # 残留 checkpoint 会被 _load_checkpoint 误判为「未完成」）
            (work_dir / "checkpoint.json").unlink(missing_ok=True)

            n_clips = len(manifest["clip_files"])
            print("\n" + "=" * 60)
            print(f"[ModeTest] 「{mode}」素材生成完成: {work_dir.name}")
            print(f"  主题: {topic} | {args.num_lines} 行 | 视频片段 {n_clips} 个")
            print(f"  之后可在测试页反复零消耗合成测试视频")
        finally:
            sys.stdout = old_stdout
            buf.flush()
            os.environ.pop("CHARACTER_OVERRIDES", None)

    # ------------------------------------------------------------------
    # 阶段二：测试合成（纯本地，零消耗）
    # ------------------------------------------------------------------

    def _compose_run(self, modes: list[str]):
        old_stdout = sys.stdout
        buf = _LineBuffer(self._on_log_line)
        sys.stdout = buf
        try:
            from pipeline import _step5_compose
            print("=" * 60)
            print(f"[ModeTest] 批量测试合成: {', '.join(modes)}")
            for mode in modes:
                if self._stop_flag.is_set():
                    self._finish("stopped")
                    self._on_log_line("\n⏹ 测试合成已停止")
                    return
                with self._lock:
                    self._current_mode = mode
                print("\n" + "=" * 60)
                print(f"[ModeTest] 合成测试视频: {mode}")
                ok, msg = self._compose_one(mode, _step5_compose)
                if ok:
                    with self._lock:
                        self._compose_done.append(mode)
                    print(f"  ✓ {mode}: {msg}")
                else:
                    print(f"  ✗ {mode}: {msg}")
                    if self._stop_flag.is_set():
                        self._finish("stopped")
                        return
            print("\n" + "=" * 60)
            print("[ModeTest] 批量合成结束")
        finally:
            sys.stdout = old_stdout
            buf.flush()

    def _compose_one(self, mode: str, _step5_compose) -> tuple[bool, str]:
        """单模式本地合成。结构参数一律取素材清单快照（timeline 已按快照构建，
        用户中途改配置不会造成错位）；字幕样式参数取当前配置（方便试新样式）。"""
        set_dir = self._active_set(mode)
        if set_dir is None:
            return False, "素材集不存在"
        meta_path = set_dir / "subtitles" / "meta.json"
        manifest_path = set_dir / "test_assets.json"
        if not meta_path.exists():
            return False, "缺少 subtitles/meta.json（素材生成未完成）"
        if not manifest_path.exists():
            return False, "缺少 test_assets.json（素材生成未完成）"
        try:
            script = json.loads(
                (set_dir / "script.json").read_text(encoding="utf-8"))
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            return False, f"素材文件读取失败: {e}"

        timeline = meta.get("timeline") or []
        narration = meta.get("narration", {})
        normal_paths = meta.get("normal_paths", [])
        zh_paths = meta.get("zh_paths", [])
        if not timeline:
            return False, "meta.json 中 timeline 为空"

        config = dict(load_mode_config(mode))
        config["structure"] = mode
        args = self._svc._build_args(config)
        # ---- 结构参数：素材快照优先 ----
        args.pad = float(manifest.get("pad", 0.4))
        args.practice_duration = float(manifest.get("practice_duration", 3.0))
        args.clip_duration = int(manifest.get("clip_duration", 15))
        args.ch3_en_repeats = int(manifest.get("ch3_en_repeats", 3))
        args.ch3_zh_repeats = int(manifest.get("ch3_zh_repeats", 1))
        args.ch3_zh_always = bool(manifest.get("ch3_zh_always", True))
        args.render_fps = int(manifest.get("render_fps", 8))
        args.workers = int(manifest.get("workers", 1))
        args.host_character = str(manifest.get("host_character", "") or "")
        args.num_lines = int(manifest.get("num_lines", MINI_LINES[mode]))
        # 数字人参数同样按素材快照回放（无值时保持当前配置）
        args.animation = str(manifest.get("animation", getattr(args, "animation", "stop_motion")))
        args.dh_quality = str(manifest.get("dh_quality", getattr(args, "dh_quality", "preview")))
        args.dh_neural_fps = int(manifest.get("dh_neural_fps", getattr(args, "dh_neural_fps", 3)))

        dirs = {
            "images": set_dir / "images",
            "clips": set_dir / "clips",
            "audio": set_dir / "audio",
            "subtitles": set_dir / "subtitles",
            "videos": set_dir / "videos",
        }
        self.work_dir = str(set_dir)

        clip_paths: list[str | None] = []
        group_info: list[dict] = []
        line_to_group: dict = {}
        if mode == "original":
            from grouping_b import build_dialogue_groups
            from group_audio import build_group_info as _build_group_info
            from clip_gen import file_ok as _file_ok
            durations = manifest.get("dialogue_durations") or []
            groups = build_dialogue_groups(
                script.get("dialogue", []), durations, args.clip_duration)
            n_manifest = int(manifest.get("n_groups", -1))
            if n_manifest >= 0 and len(groups) != n_manifest:
                return False, (f"对话分组数与素材不一致"
                               f"（{len(groups)} != {n_manifest}），请重新生成素材")
            clips_dir = dirs["clips"]
            p0 = clips_dir / "clip_0.mp4"
            clip_paths.append(str(p0) if _file_ok(str(p0), 500000) else None)
            for gi in range(len(groups)):
                p = clips_dir / f"clip_{gi+1}.mp4"
                clip_paths.append(str(p) if _file_ok(str(p), 500000) else None)
            missing = [i for i, p in enumerate(clip_paths) if p is None]
            if missing:
                return False, f"视频片段缺失: clip_{','.join(str(i) for i in missing)}.mp4"
            group_info, line_to_group = _build_group_info(
                groups, normal_paths, durations, dirs["audio"], clip_paths,
                args.pad, fps=24)

        # 注：compose_image/compose_listening 不支持 stop_check（quest/cutout 支持），
        # original/original_static 的单次合成无法中途打断，仅模式间可停止
        final_path, _safe_name = _step5_compose(
            args, {}, script, set_dir, dirs, clip_paths, timeline, narration,
            normal_paths, zh_paths, {}, group_info, line_to_group,
            stop_check=self._stop_flag.is_set)
        # _step5_compose 成功会写 checkpoint(step5_compose)，测试集不需要，删掉防误判
        (set_dir / "checkpoint.json").unlink(missing_ok=True)
        if not final_path:
            return False, "合成中断或无输出"
        return True, str(final_path)

    # ------------------------------------------------------------------
    # 状态查询（页面轮询）
    # ------------------------------------------------------------------

    def asset_status(self, mode: str) -> dict:
        info = {
            "mode": mode,
            "has_active": False,
            "manifest": None,
            "interrupted": False,
            "interrupted_set": "",
            "videos": [],
            "size_mb": 0.0,
        }
        set_dir = self._active_set(mode)
        if set_dir is not None:
            info["has_active"] = True
            manifest_path = set_dir / "test_assets.json"
            if manifest_path.exists():
                try:
                    info["manifest"] = json.loads(
                        manifest_path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    pass
            for v in sorted(set_dir.glob("*.mp4")):
                info["videos"].append({
                    "name": v.name,
                    "size_mb": round(v.stat().st_size / (1024 * 1024), 1),
                    "mtime": v.stat().st_mtime,
                    "url": f"/api/mode_test/video/{mode}/{v.name}",
                })
            info["videos"].sort(key=lambda x: x["mtime"], reverse=True)
            try:
                info["size_mb"] = round(
                    sum(f.stat().st_size for f in set_dir.rglob("*")
                        if f.is_file()) / (1024 * 1024), 1)
            except OSError:
                pass
        # 中断检测：素材目录下残留 checkpoint（生成完成后会被删除）
        if not self.is_running:
            root = self._asset_root(mode)
            if root.exists():
                for sub in root.iterdir():
                    if sub.is_dir() and (sub / "checkpoint.json").exists():
                        info["interrupted"] = True
                        info["interrupted_set"] = sub.name
                        break
        return info

    def full_status(self) -> dict:
        return {
            "progress": self.get_progress(),
            "modes": [self.asset_status(m) for m in MODES],
            "mini_lines": dict(MINI_LINES),
        }


# Singleton
_service: ModeTestService | None = None


def get_mode_test_service() -> ModeTestService:
    global _service
    if _service is None:
        _service = ModeTestService()
    return _service
