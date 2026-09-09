"""「👨‍👩‍👧‍👦 家庭角色设定」读写 + 家庭预设管理（story 模式专属配置）。

configs/story_family.json 由 Web 编辑卡与 pipeline 共读：
- GET /api/story_family → 4 槽位完整设定（文件缺槽/缺键逐项回退内置默认）
- PUT /api/story_family → 保存 4 槽位（逐槽 {name, gender, role, description}）

家庭预设（像角色套装/素材库一样命名保存整套家庭，随时切换）：
- GET  /api/story_family/presets        → {presets: {name: {family, created}}}
- POST /api/story_family/presets/save   → {name, family} 命名保存当前四槽位
- POST /api/story_family/presets/delete → {name} 删除预设
预设存 configs/story_family_presets.json；预设只填充编辑卡，需「保存设定」
才写入激活家庭（story_family.json）。

pipeline 侧经 STORY_FAMILY_JSON env 注入（pipeline_service._set_env），
生成器 load_story_family 逐槽合并 —— 文件缺槽位/缺键安全。
"""
import json
import time

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from ..paths import CONFIGS_DIR, ensure_pipeline_on_path

ensure_pipeline_on_path()

from story.llm_client_story import load_story_family  # noqa: E402

router = APIRouter()

_FAMILY_KEYS = ("char_a", "char_b", "char_c", "char_d")
_FAMILY_PATH = CONFIGS_DIR / "story_family.json"
_FAMILY_PRESETS_PATH = CONFIGS_DIR / "story_family_presets.json"
_CHAR_LABELS = {
    "char_a": "角色 A（char_a）",
    "char_b": "角色 B（char_b）",
    "char_c": "角色 C（char_c）",
    "char_d": "角色 D（char_d）",
}


def _clean_family(fam) -> tuple[dict | None, str]:
    """校验四槽位（gender 白名单），返回 (cleaned, error)。缺槽跳过=保留默认。"""
    if not isinstance(fam, dict):
        return None, "缺少 family 或格式错误"
    cleaned: dict = {}
    for key in _FAMILY_KEYS:
        slot = fam.get(key)
        if slot is None:
            continue
        if not isinstance(slot, dict):
            return None, f"{key} 格式错误"
        entry = {}
        for k in ("name", "gender", "role", "description"):
            v = str(slot.get(k, "") or "").strip()
            if v:
                entry[k] = v
        if entry.get("gender") and entry["gender"] not in ("male", "female"):
            return None, f"{_CHAR_LABELS[key]} gender 非法（只接受 male/female）"
        if entry:
            cleaned[key] = entry
    return cleaned, ""


def _load_family_presets() -> dict:
    if _FAMILY_PRESETS_PATH.exists():
        try:
            data = json.loads(_FAMILY_PRESETS_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _write_family_presets(presets: dict) -> None:
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    _FAMILY_PRESETS_PATH.write_text(
        json.dumps(presets, ensure_ascii=False, indent=2), encoding="utf-8")


@router.get("/api/story_family")
async def api_get_story_family():
    """返回合并内置默认后的 4 槽位家庭设定（前端编辑卡预填用）。"""
    return {"family": load_story_family(), "labels": _CHAR_LABELS}


@router.put("/api/story_family")
async def api_put_story_family(request: Request):
    """保存激活家庭设定。缺槽 = 保留内置默认；gender 只接受 male/female。"""
    data = await request.json()
    fam = data.get("family") if isinstance(data, dict) else None
    cleaned, err = _clean_family(fam)
    if err:
        return JSONResponse({"ok": False, "error": err}, status_code=400)
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    _FAMILY_PATH.write_text(
        json.dumps(cleaned, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"ok": True, "family": load_story_family()}


@router.get("/api/story_family/presets")
async def api_family_presets():
    """列出全部已存家庭预设 {预设名: {family, created}}。"""
    return {"presets": _load_family_presets()}


@router.post("/api/story_family/presets/save")
async def api_family_preset_save(request: Request):
    """把编辑卡当前四槽位存为命名家庭预设（同名覆盖）。"""
    data = await request.json()
    name = str(data.get("name", "") or "").strip()
    if not name or len(name) > 60:
        return JSONResponse({"ok": False, "error": "预设名无效（1-60 字符）"},
                            status_code=400)
    cleaned, err = _clean_family(data.get("family"))
    if err:
        return JSONResponse({"ok": False, "error": err}, status_code=400)
    presets = _load_family_presets()
    presets[name] = {"family": cleaned, "created": time.time()}
    _write_family_presets(presets)
    return {"ok": True, "presets": presets}


@router.post("/api/story_family/presets/delete")
async def api_family_preset_delete(request: Request):
    """删除命名家庭预设（不影响当前激活家庭 story_family.json）。"""
    data = await request.json()
    name = str(data.get("name", "") or "").strip()
    presets = _load_family_presets()
    if name not in presets:
        return JSONResponse({"ok": False, "error": "预设不存在"}, status_code=404)
    presets.pop(name)
    _write_family_presets(presets)
    return {"ok": True, "presets": presets}
