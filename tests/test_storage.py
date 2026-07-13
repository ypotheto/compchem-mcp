import ast
import importlib
import importlib.abc
import importlib.util
import sys
import threading
from pathlib import Path

import pytest

import ypotheto_compchem_mcp.storage as storage_module
from ypotheto_compchem_mcp.storage import LocalDirBackend


@pytest.fixture
def backend(tmp_path) -> LocalDirBackend:
    return LocalDirBackend(tmp_path)


def test_write_then_read_roundtrips(backend: LocalDirBackend) -> None:
    backend.write_file("ws1", "artifacts/a1/result.txt", b"hello world")
    assert backend.read_file("ws1", "artifacts/a1/result.txt") == b"hello world"


def test_write_overwrites_existing_file(backend: LocalDirBackend) -> None:
    backend.write_file("ws1", "a/b.txt", b"first")
    backend.write_file("ws1", "a/b.txt", b"second")
    assert backend.read_file("ws1", "a/b.txt") == b"second"


def test_read_missing_file_raises_file_not_found(backend: LocalDirBackend) -> None:
    with pytest.raises(FileNotFoundError):
        backend.read_file("ws1", "does/not/exist.txt")


def test_exists_true_after_write_false_before(backend: LocalDirBackend) -> None:
    assert backend.file_exists("ws1", "a/b.txt") is False
    backend.write_file("ws1", "a/b.txt", b"data")
    assert backend.file_exists("ws1", "a/b.txt") is True


def test_delete_removes_file(backend: LocalDirBackend) -> None:
    backend.write_file("ws1", "a/b.txt", b"data")
    backend.delete_file("ws1", "a/b.txt")
    assert backend.file_exists("ws1", "a/b.txt") is False


def test_delete_missing_file_is_idempotent(backend: LocalDirBackend) -> None:
    backend.delete_file("ws1", "never/written.txt")  # must not raise


def test_list_files_returns_only_files_under_prefix(backend: LocalDirBackend) -> None:
    backend.write_file("ws1", "datasets/ds_1.txt", b"1")
    backend.write_file("ws1", "datasets/ds_2.txt", b"2")
    backend.write_file("ws1", "other/ds_3.txt", b"3")

    results = backend.list_files("ws1", "datasets")

    assert sorted(results) == ["datasets/ds_1.txt", "datasets/ds_2.txt"]


def test_read_does_not_create_workspace_directory(tmp_path: Path) -> None:
    backend = LocalDirBackend(tmp_path)
    with pytest.raises(FileNotFoundError):
        backend.read_file("ws-never-created", "some/file.txt")
    assert not (tmp_path / "workspaces" / "ws-never-created").exists()


@pytest.mark.parametrize(
    "evil_path",
    [
        "C:\\evil\\file.txt",
        "a\\..\\b",
        "../secret.txt",
        "a/../../secret.txt",
        "/etc/passwd",
        "..",
    ],
)
def test_traversal_attempts_are_rejected(backend: LocalDirBackend, evil_path: str) -> None:
    with pytest.raises(ValueError):
        backend.write_file("ws1", evil_path, b"data")
    with pytest.raises(ValueError):
        backend.read_file("ws1", evil_path)


def test_percent_encoded_segment_is_treated_as_a_literal_filename(backend: LocalDirBackend) -> None:
    # By the time a path reaches the storage layer it has already been routed
    # through Starlette, which does not decode a %2f segment into a literal
    # slash - so "..%2f" here is just an odd (but harmless) filename, not a
    # traversal attempt, and must not be rejected.
    backend.write_file("ws1", "weird..%2fname.txt", b"data")
    assert backend.read_file("ws1", "weird..%2fname.txt") == b"data"


def test_concurrent_write_read_delete_across_threads_raises_no_errors(tmp_path: Path) -> None:
    backend = LocalDirBackend(tmp_path)
    errors: list[BaseException] = []

    def worker(i: int) -> None:
        try:
            for j in range(25):
                path = f"artifacts/worker-{i}/item-{j}.txt"
                backend.write_file("ws1", path, f"{i}-{j}".encode())
                assert backend.read_file("ws1", path) == f"{i}-{j}".encode()
                backend.list_files("ws1")
                backend.delete_file("ws1", path)
        except BaseException as exc:  # want to see genuinely anything
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []


def test_spaces_backend_key_with_empty_prefix_has_no_trailing_dot() -> None:
    # Regression: PurePosixPath("") stringifies to "." - if the path validator
    # returned a normalized PurePosixPath instead of the original string, an
    # empty listing prefix would corrupt to "workspaces/ws1/." and match
    # nothing in S3, silently breaking list_files() for the whole-workspace case.
    from ypotheto_compchem_mcp.storage import SpacesBackend

    backend = SpacesBackend(
        bucket="test-bucket",
        endpoint_url="https://example.invalid",
        access_key="test",
        secret_key="test",
        region="us-east-1",
        prefix="compchem-mcp",
    )
    assert backend._key("ws1", "") == "compchem-mcp/workspaces/ws1/"


def test_no_module_level_boto3_import() -> None:
    tree = ast.parse(Path(storage_module.__file__).read_text(encoding="utf-8"))
    top_level_modules = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level_modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_level_modules.add(node.module.split(".")[0])
    assert "boto3" not in top_level_modules
    assert "botocore" not in top_level_modules


class _BlockedLoader(importlib.abc.Loader):
    def create_module(self, spec):
        return None

    def exec_module(self, module):
        raise ImportError(f"blocked for test: {module.__name__}")


class _BlockBoto3Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == "boto3" or fullname.startswith("boto3.") or fullname == "botocore" or fullname.startswith("botocore."):
            return importlib.util.spec_from_loader(fullname, _BlockedLoader())
        return None


def test_module_imports_and_local_backend_works_without_boto3(monkeypatch) -> None:
    for name in [
        m for m in list(sys.modules)
        if m == "boto3" or m.startswith("boto3.") or m == "botocore" or m.startswith("botocore.")
    ]:
        monkeypatch.delitem(sys.modules, name, raising=False)

    blocker = _BlockBoto3Finder()
    sys.meta_path.insert(0, blocker)
    try:
        importlib.reload(storage_module)  # must succeed even with boto3 blocked
        backend = storage_module.LocalDirBackend(Path.cwd())
        # Check against the freshly-reloaded module's own class, not the one
        # imported at this test file's top level - reload() rebinds the name to
        # a new class object, so the old reference would never isinstance-match.
        assert isinstance(backend, storage_module.StorageBackend)

        with pytest.raises(ImportError):
            storage_module.SpacesBackend(
                bucket="b", endpoint_url="https://x", access_key="a", secret_key="s", region="nyc3",
            )
    finally:
        sys.meta_path.remove(blocker)
        importlib.reload(storage_module)
