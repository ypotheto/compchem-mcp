import abc
from pathlib import Path
import shutil

class StorageBackend(abc.ABC):
    @abc.abstractmethod
    def read_file(self, workspace_id: str, path: str) -> bytes:
        """Read a file's content as bytes."""
        pass
        
    @abc.abstractmethod
    def write_file(self, workspace_id: str, path: str, data: bytes) -> str:
        """Write bytes to a file. Returns the relative path."""
        pass
        
    @abc.abstractmethod
    def delete_file(self, workspace_id: str, path: str) -> None:
        """Delete a file or directory."""
        pass
        
    @abc.abstractmethod
    def file_exists(self, workspace_id: str, path: str) -> bool:
        """Check if a file exists."""
        pass
        
    @abc.abstractmethod
    def list_files(self, workspace_id: str, prefix: str = "") -> list[str]:
        """List files in the workspace matching a prefix."""
        pass

class LocalDirBackend(StorageBackend):
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        
    def _resolve_path(self, workspace_id: str, path: str) -> Path:
        # Directory traversal protection
        clean_path = path.lstrip("/")
        if ".." in clean_path:
            raise ValueError("Access Denied: directory traversal attempt.")
        resolved = self.base_dir / "workspaces" / workspace_id / clean_path
        resolved.parent.mkdir(parents=True, exist_ok=True)
        return resolved

    def read_file(self, workspace_id: str, path: str) -> bytes:
        resolved = self._resolve_path(workspace_id, path)
        if not resolved.exists() or not resolved.is_file():
            raise FileNotFoundError(f"File not found: {path}")
        return resolved.read_bytes()

    def write_file(self, workspace_id: str, path: str, data: bytes) -> str:
        resolved = self._resolve_path(workspace_id, path)
        resolved.write_bytes(data)
        return path

    def delete_file(self, workspace_id: str, path: str) -> None:
        resolved = self._resolve_path(workspace_id, path)
        if resolved.exists():
            if resolved.is_file():
                resolved.unlink()
            elif resolved.is_dir():
                shutil.rmtree(resolved)

    def file_exists(self, workspace_id: str, path: str) -> bool:
        resolved = self._resolve_path(workspace_id, path)
        return resolved.exists() and resolved.is_file()

    def list_files(self, workspace_id: str, prefix: str = "") -> list[str]:
        workspace_dir = self.base_dir / "workspaces" / workspace_id
        if not workspace_dir.exists():
            return []
        prefix_clean = prefix.lstrip("/")
        search_dir = workspace_dir / prefix_clean
        if not search_dir.exists():
            return []
            
        files = []
        if search_dir.is_file():
            files.append(prefix_clean)
        elif search_dir.is_dir():
            for p in search_dir.rglob("*"):
                if p.is_file():
                    rel = p.relative_to(workspace_dir)
                    files.append(str(rel).replace("\\", "/"))
        return files
