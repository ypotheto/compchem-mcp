import functools
import inspect
import logging
import time
import traceback
import uuid
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from ypotheto_compchem_mcp import __version__
from ypotheto_compchem_mcp.errors import CompchemError


class WarningInfo(BaseModel):
    type: str
    message: str

class ArtifactInfo(BaseModel):
    kind: str  # "structure" | "plot" | "report"
    description: str
    url: str

class ToolResponseEnvelope(BaseModel):
    ok: bool
    results: dict[str, Any]
    warnings: list[WarningInfo] = []
    interpretation: str
    artifacts: list[ArtifactInfo] = []
    meta: dict[str, Any] = {}

def make_success_response(
    results: dict[str, Any],
    interpretation: str,
    warnings: list[WarningInfo] | None = None,
    artifacts: list[ArtifactInfo] | None = None,
    meta: dict[str, Any] | None = None
) -> dict[str, Any]:
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

def build_provenance(
    software: str,
    method: str | None = None,
    functional: str | None = None,
    basis: str | None = None
) -> dict[str, Any]:
    """
    Build a `meta.provenance` dict recording which backend/version/method/basis
    actually produced a result, so a client can tell exactly how a number was
    computed. Version lookup is best-effort - some backends (e.g. a vendored
    driver script) aren't installed as a package with metadata.
    """
    import importlib.metadata
    try:
        version = importlib.metadata.version(software)
    except Exception:
        version = "unknown"
    provenance: dict[str, Any] = {"software": software, "version": version}
    if method is not None:
        provenance["method"] = method
    if functional is not None:
        provenance["functional"] = functional
    if basis is not None:
        provenance["basis"] = basis
    return provenance

def make_error_response(
    code: str,
    message: str,
    hint: str | None = None
) -> dict[str, Any]:
    """Helper to construct an error response envelope."""
    return {
        "ok": False,
        "error": {
            "code": code,
            "message": message,
            "hint": hint or ""
        }
    }

def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000


def _run_maybe_async(awaitable: Any) -> Any:
    """Resolve an awaitable returned by an (as-yet nonexistent) async tool body.
    Only safe when there is no already-running event loop in this thread."""
    import asyncio
    return asyncio.run(awaitable)


def mcp_tool_decorator(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator for MCP tools.
    - Catches exceptions and formats them as a clean error envelope.
    - Resolves the current workspace and validates quotas before executing the tool.
    - Logs usage (timing + success/failure) for every call.

    NOTE on async tools: this wrapper is intentionally synchronous, not `async def`.
    Every tool in this codebase is a plain sync function, and dozens of tests call
    decorated tools directly expecting a dict back (`res = some_tool(...)`), not a
    coroutine. Making this wrapper `async def` would require awaiting it everywhere
    (FastMCP-side and test-side), which means migrating the whole test suite to
    pytest-asyncio - a large, separate change tracked as a follow-up, not bundled
    silently into this fix. If a tool body is itself `async def`, its coroutine is
    still awaited here via a private event loop (see `_run_maybe_async` below), so
    async tool bodies work when called directly; nested-event-loop safety for a tool
    called from within an already-async server context has not been exercised (no
    such tool exists yet) and should be verified before the first async tool ships.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        from ypotheto_compchem_mcp.usage import log_usage
        from ypotheto_compchem_mcp.utils.plotting import close_all_open_figures
        from ypotheto_compchem_mcp.workspace import get_workspace_id, workspace_manager

        # 1. Quota Check
        workspace_id = get_workspace_id()
        is_under_quota, quota_msg = workspace_manager.check_quotas(workspace_id)
        if not is_under_quota:
            return make_error_response("QUOTA_EXCEEDED", quota_msg)

        # 2. Execute tool
        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            if inspect.isawaitable(result):
                result = _run_maybe_async(result)
        except CompchemError as ce:
            close_all_open_figures()
            log_usage(workspace_id, func.__name__, _elapsed_ms(start), ok=False)
            return make_error_response(ce.code, str(ce), ce.hint)
        except ValueError as ve:
            # Map standard value errors (e.g. invalid SMILES or missing coordinates)
            close_all_open_figures()
            log_usage(workspace_id, func.__name__, _elapsed_ms(start), ok=False)
            return make_error_response("INVALID_ARGUMENT", str(ve))
        except FileNotFoundError as fnf:
            close_all_open_figures()
            log_usage(workspace_id, func.__name__, _elapsed_ms(start), ok=False)
            return make_error_response("NOT_FOUND", str(fnf))
        except Exception as e:
            # Log unexpected exceptions server-side under a correlation id (keeps
            # full tracebacks out of the model's context while staying debuggable),
            # and return a generic but honest error.
            close_all_open_figures()
            correlation_id = uuid.uuid4().hex[:8]
            logging.error(
                f"Unexpected error in tool {func.__name__} (correlation_id={correlation_id}): {str(e)}\n{traceback.format_exc()}"
            )
            log_usage(workspace_id, func.__name__, _elapsed_ms(start), ok=False)
            return make_error_response(
                "INTERNAL_ERROR",
                f"An unexpected internal error occurred: {str(e)}",
                hint=f"Reference id: {correlation_id} in server logs."
            )
        else:
            log_usage(workspace_id, func.__name__, _elapsed_ms(start), ok=True)
            return result
    return wrapper
