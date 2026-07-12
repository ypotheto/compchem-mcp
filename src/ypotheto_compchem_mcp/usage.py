import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from ypotheto_compchem_mcp.config import settings

_logger_instance: Optional[logging.Logger] = None
_bound_log_file = None


def _get_logger() -> logging.Logger:
    """
    Lazily construct the usage logger on first use, bound to the CURRENT
    settings.data_dir at that moment - not at import time. Import-time setup
    would (a) create directories as a side effect of merely importing this
    module, and (b) bake in whatever settings.data_dir happened to be at
    import time, ignoring any later reconfiguration (e.g. test isolation).
    """
    global _logger_instance, _bound_log_file
    log_file = settings.data_dir / "usage.log"
    if _logger_instance is not None and _bound_log_file == log_file:
        return _logger_instance

    logger = logging.getLogger(f"compchem_mcp_usage.{id(log_file)}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    settings.data_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(file_handler)

    _logger_instance = logger
    _bound_log_file = log_file
    return logger


def log_usage(
    workspace_id: str,
    tool_name: str,
    duration_ms: float,
    ok: bool,
    molecule_id: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None
) -> None:
    """Log a tool execution event to the usage.log file in JSON Lines format."""
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "workspace_id": workspace_id,
        "tool": tool_name,
        "duration_ms": int(duration_ms),
        "ok": ok,
        "molecule_id": molecule_id or "",
        "details": details or {}
    }
    _get_logger().info(json.dumps(event))
