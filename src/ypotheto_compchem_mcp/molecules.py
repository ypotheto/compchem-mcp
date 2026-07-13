import logging
import threading
import time
from collections import OrderedDict
from typing import Any

from ypotheto_compchem_mcp.errors import MoleculeNotFoundError


class MoleculeStore:
    """Cached facade over the molecule index/file/Postgres tiering already
    implemented in `chemistry.builder_engine` (`load_molecule_index`,
    `save_molecule_coords`, `get_molecule_path`) - reuses that logic rather
    than duplicating it, and adds what it doesn't provide: a short-lived,
    thread-safe cache for repeated `list`/`describe` calls (every
    `load_molecule_index` call re-hits Postgres or remote storage with no
    caching at all otherwise), and molecule deletion (no delete capability
    existed anywhere in this codebase before this).

    Deliberately does NOT replace `load_molecule_from_workspace`/
    `get_molecule_path` as the read path for the 40+ existing tools that load
    a molecule's actual structure to compute something - those are
    unaffected. This store is specifically for the metadata-level
    list/describe/delete operations the new tools below need.
    """

    def __init__(self, max_cache_entries: int = 32, cache_ttl_seconds: float = 5.0) -> None:
        self._cache: OrderedDict[str, tuple[float, dict[str, Any]]] = OrderedDict()
        self._lock = threading.Lock()
        self._max_cache_entries = max_cache_entries
        self._cache_ttl_seconds = cache_ttl_seconds

    def _get_index(self, workspace_id: str) -> dict[str, Any]:
        from ypotheto_compchem_mcp.chemistry.builder_engine import load_molecule_index

        with self._lock:
            cached = self._cache.get(workspace_id)
            if cached is not None:
                fetched_at, index = cached
                if time.time() - fetched_at < self._cache_ttl_seconds:
                    self._cache.move_to_end(workspace_id)
                    return index

        index = load_molecule_index(workspace_id)
        with self._lock:
            self._cache[workspace_id] = (time.time(), index)
            self._cache.move_to_end(workspace_id)
            while len(self._cache) > self._max_cache_entries:
                self._cache.popitem(last=False)
        return index

    def invalidate(self, workspace_id: str) -> None:
        with self._lock:
            self._cache.pop(workspace_id, None)

    @staticmethod
    def _normalize_entry(key: str, entry: dict[str, Any]) -> dict[str, Any]:
        """Backfill `molecule_id` from the index's own dict key if a stored
        entry doesn't already carry it. Found in practice: one call site
        (`ensemble_pipeline.py`'s temporary reference-conformer registration)
        saved an incomplete meta dict missing molecule_id/name/smiles/method
        entirely - fixed at the source, but this stays as a defensive
        normalization since the index is otherwise-untrusted persisted state
        (local disk or Postgres) this store didn't necessarily write itself."""
        if "molecule_id" in entry:
            return entry
        return {"molecule_id": key, **entry}

    def list(self, workspace_id: str) -> list[dict[str, Any]]:
        index = self._get_index(workspace_id)
        entries = [self._normalize_entry(k, v) for k, v in index.items()]
        return sorted(entries, key=lambda m: m["molecule_id"])

    def describe(self, workspace_id: str, molecule_id: str) -> dict[str, Any]:
        index = self._get_index(workspace_id)
        if molecule_id not in index:
            raise MoleculeNotFoundError(f"Molecule '{molecule_id}' not found in this workspace.")
        return self._normalize_entry(molecule_id, index[molecule_id])

    def delete(self, workspace_id: str, molecule_id: str) -> None:
        from ypotheto_compchem_mcp.chemistry.builder_engine import (
            get_molecules_dir,
            save_molecule_index,
        )
        from ypotheto_compchem_mcp.database import get_connection
        from ypotheto_compchem_mcp.storage import storage

        index = self._get_index(workspace_id)
        if molecule_id not in index:
            raise MoleculeNotFoundError(f"Molecule '{molecule_id}' not found in this workspace.")

        mol_dir = get_molecules_dir(workspace_id)
        for ext in ("sdf", "xyz"):
            (mol_dir / f"{molecule_id}.{ext}").unlink(missing_ok=True)
            try:
                storage.delete_file(workspace_id, f"molecules/{molecule_id}.{ext}")
            except FileNotFoundError:
                pass

        conn = get_connection()
        if conn is not None:
            try:
                cur = conn.cursor()
                cur.execute(
                    "DELETE FROM compchem.molecules WHERE molecule_id = %s AND workspace_id = %s",
                    (molecule_id, workspace_id),
                )
                conn.commit()
            except Exception:
                logging.error(f"Failed to delete molecule {molecule_id} from PostgreSQL", exc_info=True)
            finally:
                conn.close()

        remaining = {k: v for k, v in index.items() if k != molecule_id}
        save_molecule_index(workspace_id, remaining)
        self.invalidate(workspace_id)


molecule_store = MoleculeStore()
