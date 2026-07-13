import pytest

from ypotheto_compchem_mcp.chemistry.builder_engine import build_molecule_from_smiles_engine
from ypotheto_compchem_mcp.chemistry.solubility_engine import (
    calculate_hsp_distance_engine,
    calculate_hsp_engine,
)
from ypotheto_compchem_mcp.modules.solubility_tools import calculate_hsp, calculate_hsp_distance
from ypotheto_compchem_mcp.workspace import get_workspace_id


def test_acetone_solubility():
    workspace_id = get_workspace_id()
    # Build Acetone (CC(C)=O)
    res_build = build_molecule_from_smiles_engine("CC(C)=O", "Acetone")
    molecule_id = res_build["molecule_id"]
    
    # Calculate HSP
    res = calculate_hsp_engine(workspace_id, molecule_id)
    hsp = res["hansen_parameters"]
    
    # Check volume (33.5 * 2 + 10.8 = 77.8)
    assert res["molar_volume_cm3_mol"] == pytest.approx(77.8, abs=1e-2)
    # Check mapped groups
    assert res["mapped_groups"]["CH3"] == 2
    assert res["mapped_groups"]["carbonyl_CO"] == 1
    
    # Check Hansen parameters
    # delta_d = 1130 / 77.8 = 14.52
    # delta_p = 770 / 77.8 = 9.90
    # delta_h = sqrt(2000 / 77.8) = 5.07
    assert hsp["dispersion_delta_d"] == pytest.approx(14.52, abs=0.1)
    assert hsp["polar_delta_p"] == pytest.approx(9.90, abs=0.1)
    assert hsp["hydrogen_bonding_delta_h"] == pytest.approx(5.07, abs=0.1)

def test_ethanol_solubility():
    workspace_id = get_workspace_id()
    # Build Ethanol (CCO)
    res_build = build_molecule_from_smiles_engine("CCO", "Ethanol")
    molecule_id = res_build["molecule_id"]
    
    res = calculate_hsp_engine(workspace_id, molecule_id)

    # Check mapped groups: 1 CH3, 1 CH2, 1 hydroxyl_OH
    assert res["mapped_groups"]["CH3"] == 1
    assert res["mapped_groups"]["CH2"] == 1
    assert res["mapped_groups"]["hydroxyl_OH"] == 1
    
    # Volume: 33.5 + 16.1 + 10.0 = 59.6
    assert res["molar_volume_cm3_mol"] == pytest.approx(59.6, abs=1e-2)

def test_solubility_distance_acetone_ethanol():
    workspace_id = get_workspace_id()
    
    acetone = build_molecule_from_smiles_engine("CC(C)=O", "Acetone")
    ethanol = build_molecule_from_smiles_engine("CCO", "Ethanol")
    
    res = calculate_hsp_distance_engine(workspace_id, acetone["molecule_id"], ethanol["molecule_id"])
    assert res["ok"] is True
    # Ethanol delta: 15.10, 8.39, 18.32
    # Acetone delta: 14.52, 9.90, 5.07
    # d_d = 15.10 - 14.52 = 0.58 => 4 * d_d^2 = 1.34
    # d_p = 8.39 - 9.90 = -1.51 => d_p^2 = 2.28
    # d_h = 18.32 - 5.07 = 13.25 => d_h^2 = 175.56
    # Ra^2 = 1.34 + 2.28 + 175.56 = 179.18
    # Ra = sqrt(179.18) = 13.38
    assert res["results"]["hansen_distance_ra"] == pytest.approx(13.38, abs=0.5)
    assert res["results"]["miscibility_estimate"] == "Poorly Compatible / Insoluble"

def test_solubility_tools_mcp():
    # Test through the FastMCP tool wrappers
    acetone = build_molecule_from_smiles_engine("CC(C)=O", "Acetone MCP")
    ethanol = build_molecule_from_smiles_engine("CCO", "Ethanol MCP")
    
    hsp_env = calculate_hsp(acetone["molecule_id"])
    assert hsp_env["ok"] is True
    assert "polar_delta_p" in hsp_env["results"]["hansen_parameters"]
    
    dist_env = calculate_hsp_distance(acetone["molecule_id"], ethanol["molecule_id"])
    assert dist_env["ok"] is True
    assert "hansen_distance_ra" in dist_env["results"]
