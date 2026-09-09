"""「👨‍👩‍👧‍👦 家庭角色设定」读写（story 模式专属配置）。

configs/story_family.json 由 Web 编辑卡与 pipeline 共读：
- GET /api/story_family → 4 槽位完整设定（文件缺槽/缺键逐项回退内置默认）
- PUT /api/story_family → 保存 4 槽位（逐槽 {name, gender, role, description}）

pipeline 侧经 STORY_FAMILY_JSON env 注入（pipeline_service._set_env），
生成器 load_story_family 逐槽合并 —— 文件缺槽位/缺键安全。
"""
import json

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from ..paths import CONFIGS_DIR, ensure_pipeline_on_path

ensure_pipeline_on_path()

from story.llm_client_story import load_story_family  # noqa: E402

router = APIRouter()

_FAMILY_KEYS = ("char_a", "char_b", "char_c", "char_d")
_FAMILY_PATH = CONFIGS_DIR / "story_family.json"
_CHAR_LABELS = {
    "char_a": "角色 A（char_a）",
    "char_b": "角色 B（char_b）",
    "char_c": "角色 C（char_c）",
    "char_d": "角色 D（char_d）",
}


@router.get("/api/story_family")
async def api_get_story_family():
    """返回合并内置默认后的 4 槽位家庭设定（前端编辑卡预填用）。"""
    return {"family": load_story_family(), "labels": _CHAR_LABELS}


@router.put("/api/story_family")
async def api_put_story_family(request: Request):
    """保存家庭设定。缺槽 = 保留内置默认；gender 只接受 male/female。"""
    data = await request.json()
    fam = data.get("family") if isinstance(data, dict) else None
    if not isinstance(fam, dict):
        return JSONResponse({"ok": False, "error": "缺少 family 字段"},
                            status_code=400)
    cleaned: dict = {}
    for key in _FAMILY_KEYS:
        slot = fam.get(key)
        if slot is None:
            continue
        if not isinstance(slot, dict):
            return JSONResponse({"ok": False, "error": f"{key} 格式错误"},
                                status_code=400)
        entry = {}
        for k in ("name", "gender", "role", "description"):
            v = str(slot.get(k, "") or "").strip()
            if v:
                entry[k] = v
        if entry.get("gender") and entry["gender"] not in ("male", "female"):
            return JSONResponse(
                {"ok": False,
                 "error": f"{_CHAR_LABELS[key]} gender 非法（只接受 male/female）"},
                status_code=400)
        if entry:
            cleaned[key] = entry
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    _FAMILY_PATH.write_text(
        json.dumps(cleaned, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"ok": True, "family": load_story_family()}
