import time
import pytest
from ypotheto_compchem_mcp.envelope import mcp_tool_decorator
from ypotheto_compchem_mcp.errors import (
    BackendUnavailableError,
    ValidationError,
    CalculationFailedError,
)
from ypotheto_compchem_mcp.jobs import job_manager, _job_error_from_exception
from ypotheto_compchem_mcp.workspace import get_workspace_id


def test_backend_unavailable_maps_through_decorator():
    @mcp_tool_decorator
    def dummy_tool():
        raise BackendUnavailableError("xtb missing", hint="install xtb")

    res = dummy_tool()
    assert res["ok"] is False
    assert res["error"]["code"] == "BACKEND_UNAVAILABLE"
    assert res["error"]["message"] == "xtb missing"
    assert res["error"]["hint"] == "install xtb"


def test_validation_error_maps_through_decorator():
    @mcp_tool_decorator
    def dummy_tool():
        raise ValidationError("bad input")

    res = dummy_tool()
    assert res["ok"] is False
    assert res["error"]["code"] == "INVALID_ARGUMENT"
    assert res["error"]["message"] == "bad input"


def test_calculation_failed_maps_through_decorator():
    @mcp_tool_decorator
    def dummy_tool():
        raise CalculationFailedError("diverged", hint="try smaller basis")

    res = dummy_tool()
    assert res["ok"] is False
    assert res["error"]["code"] == "CALCULATION_FAILED"
    assert res["error"]["hint"] == "try smaller basis"


def test_plain_value_error_still_maps_to_invalid_argument():
    @mcp_tool_decorator
    def dummy_tool():
        raise ValueError("legacy path")

    res = dummy_tool()
    assert res["ok"] is False
    assert res["error"]["code"] == "INVALID_ARGUMENT"


def test_successful_tool_passes_through():
    @mcp_tool_decorator
    def dummy_tool():
        return {"ok": True, "results": {}, "interpretation": "done", "warnings": [], "artifacts": [], "meta": {}}

    res = dummy_tool()
    assert res["ok"] is True


def test_job_error_from_exception_preserves_compchem_error_code():
    exc = BackendUnavailableError("xtb missing", hint="install xtb")
    err = _job_error_from_exception(exc)
    assert err["code"] == "BACKEND_UNAVAILABLE"
    assert err["message"] == "xtb missing"
    assert err["hint"] == "install xtb"


def test_job_error_from_exception_falls_back_for_plain_exceptions():
    err = _job_error_from_exception(RuntimeError("boom"))
    assert err["code"] == "INTERNAL_JOB_ERROR"
    assert err["message"] == "boom"


def test_async_job_preserves_backend_unavailable_code_through_thread_fallback():
    """
    Background jobs call engine functions directly (bypassing mcp_tool_decorator).
    Confirms a CompchemError raised inside a submitted job surfaces its real code/hint
    through get_job_status, not a generic INTERNAL_JOB_ERROR - this runs on the
    thread-fallback path since no database is configured in the test environment.
    """
    workspace_id = get_workspace_id()

    def failing_job(workspace_id):
        raise BackendUnavailableError("mace-torch is not installed", hint="pip install mace-torch")

    job = job_manager.submit_job(workspace_id, failing_job, 1, workspace_id)

    status = None
    for _ in range(50):
        status = job_manager.get_job(workspace_id, job.job_id)
        if status and status.status in ("completed", "failed"):
            break
        time.sleep(0.1)

    assert status is not None
    assert status.status == "failed"
    assert status.error["code"] == "BACKEND_UNAVAILABLE"
    assert status.error["hint"] == "pip install mace-torch"
