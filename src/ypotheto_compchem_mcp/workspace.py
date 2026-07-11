import hashlib
from contextvars import ContextVar
from pathlib import Path
from ypotheto_compchem_mcp.config import settings

# ContextVar to propagate the workspace ID across request contexts/tasks
current_workspace_id: ContextVar[str] = ContextVar("current_workspace_id", default="local")

def get_workspace_id() -> str:
    """Retrieve the current workspace ID from the context."""
    return current_workspace_id.get()

def get_workspace_id_from_token(token: str) -> str:
    """Hash the Bearer token to generate a safe, isolated workspace directory name."""
    if not token:
        return "local"
    return hashlib.sha256(token.encode()).hexdigest()[:16]

class WorkspaceManager:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir

    def get_workspace_dir(self, workspace_id: str) -> Path:
        """Get the base path for a workspace."""
        path = self.data_dir / "workspaces" / workspace_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_artifacts_dir(self, workspace_id: str) -> Path:
        """Get the artifacts folder path for a workspace."""
        path = self.get_workspace_dir(workspace_id) / "artifacts"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def check_quotas(self, workspace_id: str) -> tuple[bool, str]:
        """
        Verify if the workspace is within its size and file count quotas.
        Returns (is_under_quota, warning_or_error_message).
        """
        if workspace_id == "local":
            return True, ""  # No quotas on local runs
            
        workspace_dir = self.get_workspace_dir(workspace_id)
        
        # Max quotas
        MAX_FILES = 200
        MAX_BYTES = 50 * 1024 * 1024  # 50 MB
        
        file_count = 0
        total_bytes = 0
        
        for file_path in workspace_dir.rglob("*"):
            if file_path.is_file():
                file_count += 1
                total_bytes += file_path.stat().st_size
                
        if file_count > MAX_FILES:
            return False, f"Quota Exceeded: Workspace contains {file_count} files (limit {MAX_FILES}). Please delete unused files."
            
        if total_bytes > MAX_BYTES:
            mb_used = total_bytes / (1024 * 1024)
            mb_limit = MAX_BYTES / (1024 * 1024)
            return False, f"Quota Exceeded: Workspace size is {mb_used:.2f} MB (limit {mb_limit:.1f} MB)."
            
        return True, ""

workspace_manager = WorkspaceManager(settings.data_dir)
