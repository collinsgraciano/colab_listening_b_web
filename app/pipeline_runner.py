"""Pipeline runner — thin wrapper around PipelineService.

Kept for backward compatibility with main.py imports.
All logic now lives in pipeline_service.py (direct import, not subprocess).
"""
from .pipeline_service import get_service, PipelineService, STEP_ORDER

# Re-export for main.py compatibility
get_runner = get_service
