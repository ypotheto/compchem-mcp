from ypotheto_compchem_mcp.chemistry.builder_engine import build_molecule_from_smiles_engine
from ypotheto_compchem_mcp.chemistry.descriptors import calculate_descriptors_engine
from ypotheto_compchem_mcp.modules.builder_tools import build_molecule_from_smiles
from ypotheto_compchem_mcp.modules.cheminformatics_tools import calculate_descriptors
from ypotheto_compchem_mcp.workspace import get_workspace_id

def test_builder_engine():
    workspace_id = get_workspace_id()
    # Build ethanol
    res = build_molecule_from_smiles_engine("CCO", "Ethanol")
    assert res["molecule_id"].startswith("mol_")
    assert res["name"] == "Ethanol"
    assert res["formula"] == "C2H6O"
    assert res["num_atoms"] == 9
    assert len(res["xyz_block"]) > 0
    assert len(res["sdf_block"]) > 0
    assert len(res["svg_data"]) > 0
    assert res["method"] in ("MMFF94", "UFF")

def test_descriptors_engine():
    workspace_id = get_workspace_id()
    # Build water
    mol_res = build_molecule_from_smiles_engine("O", "Water")
    molecule_id = mol_res["molecule_id"]
    
    # Calculate descriptors
    res = calculate_descriptors_engine(workspace_id, molecule_id)
    assert res["molecule_id"] == molecule_id
    assert abs(res["descriptors"]["molecular_weight"] - 18.01) < 0.1
    assert res["descriptors"]["hydrogen_bond_donors"] == 2
    assert res["descriptors"]["hydrogen_bond_acceptors"] == 1
    assert res["descriptors"]["rotatable_bonds"] == 0
    assert res["lipinski_filter"]["passes"] is True
    assert res["lipinski_filter"]["violations_count"] == 0

def test_mcp_tools_integration():
    # Call the tool function directly (under local context)
    envelope = build_molecule_from_smiles("CCO", "Ethanol")
    assert envelope["ok"] is True
    assert "molecule_id" in envelope["results"]
    molecule_id = envelope["results"]["molecule_id"]
    assert len(envelope["artifacts"]) == 3
    assert "Built molecule" in envelope["interpretation"]
    
    # Check descriptors tool
    desc_envelope = calculate_descriptors(molecule_id)
    assert desc_envelope["ok"] is True
    assert desc_envelope["results"]["molecule_id"] == molecule_id
    assert "molecular_weight" in desc_envelope["results"]["descriptors"]
