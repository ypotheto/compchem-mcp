import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional
from ypotheto_compchem_mcp.config import settings

# Set up logging for usage metrics
usage_logger = logging.getLogger("compchem_mcp_usage")
usage_logger.setLevel(logging.INFO)

# Ensure the log file directory exists
log_file = settings.data_dir / "usage.log"
settings.data_dir.mkdir(parents=True, exist_ok=True)

# Add file handler
file_handler = logging.FileHandler(log_file, encoding="utf-8")
file_handler.setFormatter(logging.Formatter("%(message)s"))
usage_logger.addHandler(file_handler)

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
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "workspace_id": workspace_id,
        "tool": tool_name,
        "duration_ms": int(duration_ms),
        "ok": ok,
        "molecule_id": molecule_id or "",
        "details": details or {}
    }
    usage_logger.info(json.dumps(event))
