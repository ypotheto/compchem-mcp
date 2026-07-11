import traceback
import functools
import logging
from typing import Any, Callable, Dict, List, Optional
from pydantic import BaseModel
from ypotheto_compchem_mcp import __version__

class WarningInfo(BaseModel):
    type: str
    message: str

class ArtifactInfo(BaseModel):
    kind: str  # "structure" | "plot" | "report"
    description: str
    url: str

class ToolResponseEnvelope(BaseModel):
    ok: bool
    results: Dict[str, Any]
    warnings: List[WarningInfo] = []
    interpretation: str
    artifacts: List[ArtifactInfo] = []
    meta: Dict[str, Any] = {}

def make_success_response(
    results: Dict[str, Any],
    interpretation: str,
    warnings: Optional[List[WarningInfo]] = None,
    artifacts: Optional[List[ArtifactInfo]] = None,
    meta: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Helper to construct a successful response envelope."""
    envelope = ToolResponseEnvelope(
        ok=True,
        results=results,
        interpretation=interpretation,
        warnings=warnings or [],
        artifacts=artifacts or [],
        meta={
            "server_version": __version__,
            **(meta or {})
        }
    )
    return envelope.model_dump()

def make_error_response(
    code: str,
    message: str,
    hint: Optional[str] = None
) -> Dict[str, Any]:
    """Helper to construct an error response envelope."""
    return {
        "ok": False,
        "error": {
            "code": code,
            "message": message,
            "hint": hint or ""
        }
    }

def mcp_tool_decorator(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator for MCP tools.
    - Catches exceptions and formats them as a clean error envelope.
    - Resolves the current workspace and validates quotas before executing the tool.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # 1. Quota Check
        from ypotheto_compchem_mcp.workspace import get_workspace_id, workspace_manager
        workspace_id = get_workspace_id()
        is_under_quota, quota_msg = workspace_manager.check_quotas(workspace_id)
        if not is_under_quota:
            return make_error_response("QUOTA_EXCEEDED", quota_msg)
            
        # 2. Execute tool
        try:
            return func(*args, **kwargs)
        except ValueError as ve:
            # Map standard value errors (e.g. invalid SMILES or missing coordinates)
            return make_error_response("INVALID_ARGUMENT", str(ve))
        except FileNotFoundError as fnf:
            return make_error_response("NOT_FOUND", str(fnf))
        except Exception as e:
            # Log unexpected exceptions to stderr and return generic error
            logging.error(f"Unexpected error in tool {func.__name__}: {str(e)}\n{traceback.format_exc()}")
            return make_error_response(
                "INTERNAL_ERROR",
                f"An unexpected internal error occurred: {str(e)}",
                hint="Verify input parameters or check system logs."
            )
    return wrapper
