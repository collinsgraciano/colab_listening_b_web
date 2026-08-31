"""Shared TTS synthesis state across the three voice routers (qwen/kokoro/moss)."""
import threading

# 串行化 GPU 合成：批量试听任务与单次试听共用同一模型实例，防并发冲突
TTS_SYNTH_LOCK = threading.Lock()

# Must match the defaults used in voices.html JS
PREVIEW_TEXTS = {
    "english": "Hello, this is a voice preview test.",
    "chinese": "你好，这是音色试听测试。",
}
