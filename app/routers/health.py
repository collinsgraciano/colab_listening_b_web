"""Health check endpoint."""
from fastapi import APIRouter

from ..paths import PIPELINE_DIR

router = APIRouter()


@router.get("/api/health")
async def health():
    return {"ok": True, "pipeline_dir": str(PIPELINE_DIR),
            "pipeline_exists": PIPELINE_DIR.exists()}
