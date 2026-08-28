"""全局运行互斥锁：主 pipeline 与模式测试共用重资源（MCP / TTS / FFmpeg），
同一时刻只允许一个重任务运行。非阻塞获取，获取失败返回 False 由调用方提示。
"""
import threading

_lock = threading.Lock()
_owner = ""
_owner_lock = threading.Lock()


def try_acquire(owner: str) -> bool:
    """尝试获取运行权。成功返回 True；已被占用（主 pipeline 或模式测试）返回 False。"""
    global _owner
    if not _lock.acquire(blocking=False):
        return False
    with _owner_lock:
        _owner = owner
    return True


def release(owner: str) -> None:
    """释放运行权（仅持有者本人可释放，避免误释放他人锁）。"""
    global _owner
    with _owner_lock:
        if _owner != owner:
            return
        _owner = ""
    _lock.release()


def current_owner() -> str:
    with _owner_lock:
        return _owner
