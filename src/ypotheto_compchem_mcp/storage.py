import abc
import os
import shutil
from pathlib import Path
import boto3
from botocore.client import Config as BotoConfig
from botocore.exceptions import ClientError
from ypotheto_compchem_mcp.config import settings

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

class SpacesBackend(StorageBackend):
    """
    DigitalOcean Spaces (S3-compatible) storage backend.
    Maintains key namespace segregation via spaces_prefix.
    """
    def __init__(
        self,
        bucket: str,
        endpoint_url: str,
        access_key: str,
        secret_key: str,
        region: str,
        prefix: str = "compchem-mcp",
    ) -> None:
        self._bucket = bucket
        self._prefix = prefix.strip("/")
        self._client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region,
            config=BotoConfig(s3={"addressing_style": "virtual"}),
        )

    def _key(self, workspace_id: str, path: str) -> str:
        clean_path = path.lstrip("/")
        if ".." in clean_path:
            raise ValueError("Access Denied: directory traversal attempt.")
        inner_key = f"workspaces/{workspace_id}/{clean_path}"
        return f"{self._prefix}/{inner_key}" if self._prefix else inner_key

    @staticmethod
    def _is_not_found(exc: ClientError) -> bool:
        code = exc.response.get("Error", {}).get("Code", "")
        status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        return code in ("404", "NoSuchKey", "NotFound") or status == 404

    def read_file(self, workspace_id: str, path: str) -> bytes:
        key = self._key(workspace_id, path)
        try:
            response = self._client.get_object(Bucket=self._bucket, Key=key)
        except ClientError as exc:
            if self._is_not_found(exc):
                raise FileNotFoundError(f"File not found: {path}") from exc
            raise
        return response["Body"].read()

    def write_file(self, workspace_id: str, path: str, data: bytes) -> str:
        key = self._key(workspace_id, path)
        self._client.put_object(Bucket=self._bucket, Key=key, Body=data)
        return path

    def delete_file(self, workspace_id: str, path: str) -> None:
        key = self._key(workspace_id, path)
        self._client.delete_object(Bucket=self._bucket, Key=key)

    def file_exists(self, workspace_id: str, path: str) -> bool:
        key = self._key(workspace_id, path)
        try:
            self._client.head_object(Bucket=self._bucket, Key=key)
        except ClientError as exc:
            if self._is_not_found(exc):
                return False
            raise
        return True

    def list_files(self, workspace_id: str, prefix: str = "") -> list[str]:
        clean_prefix = prefix.lstrip("/")
        s3_prefix = self._key(workspace_id, clean_prefix)
        results: list[str] = []
        paginator = self._client.get_paginator("list_objects_v2")
        base_prefix = f"{self._prefix}/workspaces/{workspace_id}/" if self._prefix else f"workspaces/{workspace_id}/"
        strip_len = len(base_prefix)
        for page in paginator.paginate(Bucket=self._bucket, Prefix=s3_prefix):
            for obj in page.get("Contents", []):
                results.append(obj["Key"][strip_len:])
        return results

import threading

class StorageProxy(StorageBackend):
    def __init__(self):
        self._backend = None
        self._lock = threading.Lock()
        
    @property
    def backend(self) -> StorageBackend:
        if self._backend is None:
            with self._lock:
                if self._backend is None:
                    self._backend = _build_storage_backend()
        return self._backend
        
    def reset(self):
        with self._lock:
            self._backend = None

    def read_file(self, workspace_id: str, path: str) -> bytes:
        return self.backend.read_file(workspace_id, path)

    def write_file(self, workspace_id: str, path: str, data: bytes) -> str:
        return self.backend.write_file(workspace_id, path, data)

    def delete_file(self, workspace_id: str, path: str) -> None:
        return self.backend.delete_file(workspace_id, path)

    def file_exists(self, workspace_id: str, path: str) -> bool:
        return self.backend.file_exists(workspace_id, path)

    def list_files(self, workspace_id: str, prefix: str = "") -> list[str]:
        return self.backend.list_files(workspace_id, prefix)

def _build_storage_backend() -> StorageBackend:
    if settings.spaces_bucket is None:
        return LocalDirBackend(settings.data_dir)
    if not (settings.spaces_endpoint and settings.spaces_key and settings.spaces_secret):
        raise ValueError(
            "COMPCHEM_SPACES_BUCKET is set, so COMPCHEM_SPACES_ENDPOINT, "
            "COMPCHEM_SPACES_KEY, and COMPCHEM_SPACES_SECRET must be set too"
        )
    return SpacesBackend(
        bucket=settings.spaces_bucket,
        endpoint_url=settings.spaces_endpoint,
        access_key=settings.spaces_key,
        secret_key=settings.spaces_secret,
        region=settings.spaces_region,
        prefix=settings.spaces_prefix,
    )

storage = StorageProxy()
