from ypotheto_compchem_mcp.chemistry.builder_engine import build_molecule_from_smiles_engine
from ypotheto_compchem_mcp.errors import MoleculeNotFoundError
from ypotheto_compchem_mcp.modules.builder_tools import (
    delete_molecule,
    describe_molecule,
    list_molecules,
)
from ypotheto_compchem_mcp.molecules import MoleculeStore
from ypotheto_compchem_mcp.workspace import get_workspace_id


def test_list_molecules_reflects_stored_molecules():
    before = list_molecules()
    baseline_count = before["results"]["count"]

    ethanol = build_molecule_from_smiles_engine("CCO", "Ethanol")
    acetone = build_molecule_from_smiles_engine("CC(C)=O", "Acetone")

    after = list_molecules()
    assert after["ok"] is True
    assert after["results"]["count"] == baseline_count + 2
    ids = {m["molecule_id"] for m in after["results"]["molecules"]}
    assert ethanol["molecule_id"] in ids
    assert acetone["molecule_id"] in ids


def test_describe_molecule_returns_metadata():
    ethanol = build_molecule_from_smiles_engine("CCO", "Ethanol")

    result = describe_molecule(ethanol["molecule_id"])

    assert result["ok"] is True
    assert result["results"]["name"] == "Ethanol"
    assert result["results"]["formula"]
    assert "Ethanol" in result["interpretation"]


def test_describe_unknown_molecule_returns_not_found_error():
    result = describe_molecule("mol_totally_bogus")

    assert result["ok"] is False
    assert result["error"]["code"] == "MOLECULE_NOT_FOUND"


def test_delete_molecule_removes_it_from_the_index_and_disk():
    ethanol = build_molecule_from_smiles_engine("CCO", "Ethanol")
    molecule_id = ethanol["molecule_id"]

    assert describe_molecule(molecule_id)["ok"] is True

    result = delete_molecule(molecule_id)
    assert result["ok"] is True
    assert result["results"] == {"molecule_id": molecule_id, "deleted": True}

    after = describe_molecule(molecule_id)
    assert after["ok"] is False
    assert after["error"]["code"] == "MOLECULE_NOT_FOUND"

    ids = {m["molecule_id"] for m in list_molecules()["results"]["molecules"]}
    assert molecule_id not in ids


def test_delete_unknown_molecule_raises_not_found():
    import pytest

    store = MoleculeStore()
    with pytest.raises(MoleculeNotFoundError):
        store.delete(get_workspace_id(), "mol_totally_bogus")


def test_store_cache_is_invalidated_after_delete():
    """The MoleculeStore's short TTL cache must not resurrect a deleted
    molecule if list()/describe() is called again immediately afterward."""
    store = MoleculeStore(cache_ttl_seconds=60)
    workspace_id = get_workspace_id()

    built = build_molecule_from_smiles_engine("CCO", "Ethanol")
    molecule_id = built["molecule_id"]

    assert any(m["molecule_id"] == molecule_id for m in store.list(workspace_id))

    store.delete(workspace_id, molecule_id)

    assert not any(m["molecule_id"] == molecule_id for m in store.list(workspace_id))
