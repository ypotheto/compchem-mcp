from unittest.mock import patch

from ypotheto_compchem_mcp.chemistry.builder_engine import build_molecule_from_smiles_engine
from ypotheto_compchem_mcp.modules.builder_tools import (
    _MAX_INLINE_CONTENT_BYTES,
    get_3d_coordinates,
)


def test_get_3d_coordinates_returns_content_inline_when_small():
    mol_res = build_molecule_from_smiles_engine("CCO", "Ethanol")
    molecule_id = mol_res["molecule_id"]

    res = get_3d_coordinates(molecule_id, format="xyz")

    assert res["ok"] is True
    assert res["results"]["content"] is not None
    assert res["warnings"] == []
    assert len(res["artifacts"]) == 1

def test_get_3d_coordinates_omits_content_when_over_size_limit():
    mol_res = build_molecule_from_smiles_engine("CCO", "Ethanol")
    molecule_id = mol_res["molecule_id"]

    oversized_content = "X" * (_MAX_INLINE_CONTENT_BYTES + 1)
    with patch("pathlib.Path.read_text", return_value=oversized_content):
        res = get_3d_coordinates(molecule_id, format="xyz")

    assert res["ok"] is True
    assert res["results"]["content"] is None
    assert len(res["warnings"]) == 1
    assert res["warnings"][0]["type"] == "CONTENT_TOO_LARGE"
    assert len(res["artifacts"]) == 1

def test_get_3d_coordinates_supports_pdb_format():
    # PDB is never persisted alongside a molecule's XYZ/SDF - generated on the
    # fly from the stored SDF's RDKit Mol instead (README previously claimed
    # PDB support that get_3d_coordinates didn't actually have).
    mol_res = build_molecule_from_smiles_engine("CCO", "Ethanol")
    molecule_id = mol_res["molecule_id"]

    res = get_3d_coordinates(molecule_id, format="pdb")

    assert res["ok"] is True
    assert res["results"]["format"] == "pdb"
    assert "HETATM" in res["results"]["content"]
    assert res["artifacts"][0]["url"].endswith(".pdb")

def test_get_3d_coordinates_rejects_unknown_format():
    mol_res = build_molecule_from_smiles_engine("CCO", "Ethanol")
    molecule_id = mol_res["molecule_id"]

    res = get_3d_coordinates(molecule_id, format="mol2")

    assert res["ok"] is False
    assert res["error"]["code"] == "INVALID_ARGUMENT"
