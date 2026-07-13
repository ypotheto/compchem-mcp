import json

import ypotheto_compchem_mcp.usage as usage_module
from ypotheto_compchem_mcp.config import settings
from ypotheto_compchem_mcp.usage import log_usage


def test_log_usage_is_lazy_and_binds_to_current_data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    monkeypatch.setattr(usage_module, "_logger_instance", None)
    monkeypatch.setattr(usage_module, "_bound_log_file", None)

    log_file = tmp_path / "usage.log"
    assert not log_file.exists()

    log_usage("ws1", "some_tool", 12.3, ok=True)

    assert log_file.exists()
    line = log_file.read_text(encoding="utf-8").strip().splitlines()[-1]
    event = json.loads(line)
    assert event["tool"] == "some_tool"
    assert event["ok"] is True
    assert event["workspace_id"] == "ws1"
    assert event["duration_ms"] == 12
    assert event["timestamp"].endswith("+00:00") or "Z" in event["timestamp"]


def test_importing_usage_module_does_not_create_data_dir(tmp_path, monkeypatch):
    """Regression test: usage.py used to create settings.data_dir and open a file
    handler as a side effect of merely being imported. Verifies a fresh data_dir
    stays untouched until log_usage is actually called."""
    fresh_dir = tmp_path / "not_yet_created"
    monkeypatch.setattr(settings, "data_dir", fresh_dir)
    monkeypatch.setattr(usage_module, "_logger_instance", None)
    monkeypatch.setattr(usage_module, "_bound_log_file", None)

    assert not fresh_dir.exists()

    import importlib
    importlib.reload(usage_module)

    assert not fresh_dir.exists()
