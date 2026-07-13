import importlib
import importlib.abc
import importlib.util
import sys

# Packages declared as optional extras in pyproject.toml (Phase 5.1). None of
# these may be required just to import the server module tree - a core-only
# `pip install ypotheto-compchem-mcp` must still start the server and serve
# the RDKit/ASE tool subset.
_OPTIONAL_PACKAGES = [
    "pyscf", "cclib",           # [qm]
    "chgnet", "mace",           # [mlff]
    "cantera", "juliacall",     # [thermo]
    "MDAnalysis",               # [md]
    "sella",                    # [ts]
    "psycopg2",                 # [db]
    "boto3", "botocore",        # [s3]
]


class _BlockedLoader(importlib.abc.Loader):
    def create_module(self, spec):
        return None

    def exec_module(self, module):
        raise ImportError(f"blocked for test: {module.__name__}")


class _BlockOptionalPackagesFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        top_level = fullname.split(".")[0]
        if top_level in _OPTIONAL_PACKAGES:
            return importlib.util.spec_from_loader(fullname, _BlockedLoader())
        return None


def test_server_module_tree_imports_without_any_optional_backend(monkeypatch):
    for name in [
        m for m in list(sys.modules)
        if m.split(".")[0] in _OPTIONAL_PACKAGES
    ]:
        monkeypatch.delitem(sys.modules, name, raising=False)

    # Also drop every already-imported ypotheto_compchem_mcp module so the
    # blocked finder is actually exercised on a fresh import, not served from
    # the existing (already-optional-backends-loaded) module cache.
    for name in [m for m in list(sys.modules) if m.split(".")[0] == "ypotheto_compchem_mcp"]:
        monkeypatch.delitem(sys.modules, name, raising=False)

    blocker = _BlockOptionalPackagesFinder()
    sys.meta_path.insert(0, blocker)
    try:
        # Since Phase 8's server.py factory refactor, the 15 tool modules are
        # only imported lazily inside create_server() (not at
        # `import ypotheto_compchem_mcp.server` time) - actually calling it
        # here is what exercises the "no optional backend required" claim;
        # a bare module import alone would trivially pass without checking
        # anything.
        import ypotheto_compchem_mcp.server
        from ypotheto_compchem_mcp.config import settings
        ypotheto_compchem_mcp.server.create_server(settings)
    finally:
        sys.meta_path.remove(blocker)
        # Restore a clean module cache for subsequent tests in this session.
        for name in [m for m in list(sys.modules) if m.split(".")[0] == "ypotheto_compchem_mcp"]:
            monkeypatch.delitem(sys.modules, name, raising=False)
        importlib.import_module("ypotheto_compchem_mcp.server")
