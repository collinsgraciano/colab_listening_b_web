"""SenseNova U1.5 Lite 生图客户端（同步接口，无轮询）。

依赖 SENSENOVA_API_KEY 环境变量（配置键 sensenova_api_key，可与 LLM provider 分开填写）。
IMAGE_PROVIDER=sensenova 时 image_gen / thumbnail_gen 的生图走本模块，
视频生成（Seedance2）仍走 MCP。

接口约束（官方文档）：
- POST /v1/images/generations：文生图，仅 prompt，不支持图像输入
- POST /v1/images/edits：图片编辑，images[].image_url 支持公网 URL 或
  data:image/*;base64 Data-URL（不支持裸 base64），第一张为主编辑图
- 两者均同步返回，n=1；尺寸须为 32 的倍数且在 [512,4096]，宽高比 ≤3:1
- response_format=url 的链接仅 24 小时有效 → 调用方必须立即下载落盘
"""
import base64
import json
import os
import time
import urllib.request
import urllib.error

MODEL_ID = "sensenova-u1.5-lite"
BASE_URL = "https://token.sensenova.cn/v1"

# 项目尺寸规格 → U1.5 合法尺寸
SIZE_MAP = {
    "landscape_16_9": "2720x1536",  # 16:9 2K（角色图/场景图/缩略图）
    "portrait_16_9": "1536x2720",   # 9:16 2K（竖版）
    "atlas_pose": "4096x2720",      # 4x2 姿势图集，3:2 4K（每格 ~1024x1360）
    "atlas_scene": "4096x2304",     # 2x2 场景图集，16:9 4K（每格 2048x1152）
    "atlas_seq": "4096x4096",       # 4x4 序列帧图集，方形 4K（每格 1024x1024）
}

_MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
         ".webp": "image/webp"}

# 可重试的 HTTP 状态码（限流/服务端故障）
_RETRYABLE = {429, 500, 502, 503, 504}


def get_image_provider() -> str:
    """当前生图 provider：'mcp'（默认）或 'sensenova'。"""
    return os.environ.get("IMAGE_PROVIDER", "mcp").strip().lower()


def _api_key() -> str:
    key = os.environ.get("SENSENOVA_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "SENSENOVA_API_KEY 未设置 — 生图 Provider=sensenova 需要在 Web 配置中"
            "填写 SenseNova API Key（配置键 sensenova_api_key，可与 LLM provider 分开填写）")
    return key


def _to_image_url(image_ref: str) -> str:
    """本地文件路径 → base64 Data-URL；http(s)/data URL 原样返回。"""
    if image_ref.startswith(("http://", "https://", "data:")):
        return image_ref
    with open(image_ref, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    mime = _MIME.get(os.path.splitext(image_ref)[1].lower(), "image/png")
    return f"data:{mime};base64,{b64}"


def _post(path: str, payload: dict, timeout: int, retries: int) -> dict:
    """POST JSON 到 SenseNova，带限流/网络错误退避重试。"""
    url = BASE_URL + path
    body = json.dumps(payload).encode("utf-8")
    api_key = _api_key()
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, data=body, method="POST")
            req.add_header("Authorization", f"Bearer {api_key}")
            req.add_header("Content-Type", "application/json")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")[:300]
            if e.code in (401, 403):
                raise RuntimeError(f"SenseNova API 鉴权失败 (HTTP {e.code}): {detail}")
            if e.code in _RETRYABLE and attempt < retries:
                wait = 30 * (attempt + 1)
                print(f"  [SenseNova] HTTP {e.code}，{wait}s 后重试 "
                      f"({attempt + 1}/{retries})：{detail[:120]}")
                time.sleep(wait)
                continue
            raise RuntimeError(f"SenseNova API HTTP {e.code}: {detail}")
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            if attempt < retries:
                wait = 30 * (attempt + 1)
                print(f"  [SenseNova] 网络错误（{e}），{wait}s 后重试 "
                      f"({attempt + 1}/{retries})")
                time.sleep(wait)
                continue
            raise RuntimeError(f"SenseNova API 网络错误: {e}")
    raise RuntimeError("SenseNova API 请求失败（重试耗尽）")


def _extract_url(resp: dict) -> str:
    data = (resp.get("data") or [{}])[0]
    url = data.get("url", "")
    if url:
        return url
    raise RuntimeError("SenseNova 响应中无图片 URL: "
                       + json.dumps(resp, ensure_ascii=False)[:300])


def text_to_image(prompt: str, size: str = "2720x1536", output_format: str = "png",
                  watermark: bool = False, prompt_extend: bool = True,
                  timeout: int = 600, retries: int = 2) -> str:
    """文生图，返回临时 URL（24h 有效，调用方须立即下载）。

    prompt_extend=False 用于网格图集等结构化 prompt，防止润色改写布局描述。
    watermark=false 公测期免费（之后将转为付费功能，显式传参防默认值变更）。
    """
    payload = {
        "model": MODEL_ID,
        "prompt": prompt,
        "n": 1,
        "size": size,
        "output_format": output_format,
        "response_format": "url",
        "watermark": watermark,
        "prompt_extend": prompt_extend,
    }
    return _extract_url(_post("/images/generations", payload, timeout, retries))


def edit_image(image_ref: str, prompt: str, size: str = "auto",
               output_format: str = "png", watermark: bool = False,
               prompt_extend: bool = True, timeout: int = 600,
               retries: int = 2) -> str:
    """图生图/图片编辑：参考图 + 编辑提示词，返回临时 URL（24h 有效）。

    image_ref：本地文件路径（自动转 Data-URL）或公网 http(s) URL。
    size='auto' 自动适配主图（第一张参考图）。
    """
    payload = {
        "model": MODEL_ID,
        "images": [{"image_url": _to_image_url(image_ref)}],
        "prompt": prompt,
        "n": 1,
        "size": size,
        "output_format": output_format,
        "response_format": "url",
        "watermark": watermark,
        "prompt_extend": prompt_extend,
    }
    return _extract_url(_post("/images/edits", payload, timeout, retries))


def download_image(url: str, dest: str) -> bool:
    """下载图片到本地并做基本校验（URL 仅 24h 有效，必须立即落盘）。"""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=180) as resp:
            content_type = resp.headers.get("Content-Type", "")
            if "text/html" in content_type:
                print("  [SenseNova] 下载失败：返回的是 HTML 页面而非图片")
                return False
            with open(dest, "wb") as f:
                f.write(resp.read())
        if os.path.getsize(dest) <= 1000:
            print(f"  [SenseNova] 下载文件过小: {dest}")
            return False
        return True
    except Exception as e:
        print(f"  [SenseNova] 下载失败: {e}")
        return False
