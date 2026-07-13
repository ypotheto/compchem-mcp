from unittest.mock import patch

from ypotheto_compchem_mcp.chemistry.builder_engine import build_molecule_from_smiles_engine
from ypotheto_compchem_mcp.chemistry.vib_engine import (
    run_vibrations_engine,
    simulate_ir_spectrum_engine,
)
from ypotheto_compchem_mcp.modules.vibrations_tools import (
    calculate_vibrations,
    simulate_ir_spectrum,
)
from ypotheto_compchem_mcp.workspace import get_workspace_id


@patch("ypotheto_compchem_mcp.modules.vibrations_tools._estimate_time_seconds", return_value=1)
def test_vibrations_engine_ff(mock_est):
    workspace_id = get_workspace_id()
    # Build Water
    mol_res = build_molecule_from_smiles_engine("O", "Water")
    molecule_id = mol_res["molecule_id"]
    
    # Run vibrations with forcefield MMFF94 (runs locally on Windows)
    res = run_vibrations_engine(workspace_id, molecule_id, method="MMFF94")
    assert res["ok"] is True
    assert "frequencies_cm1" in res["results"]
    assert len(res["results"]["frequencies_cm1"]) == 9 # 3 * 3 atoms = 9 modes
    assert res["results"]["zero_point_energy_ev"] > 0.0
    assert "thermochemistry" in res["results"]
    assert res["results"]["thermochemistry"]["temperature_k"] == 298.15
    assert res["results"]["thermochemistry"]["gibbs_free_energy_ev"] != 0.0

@patch("ypotheto_compchem_mcp.modules.vibrations_tools._estimate_time_seconds", return_value=1)
def test_ir_spectrum_engine_ff(mock_est):
    workspace_id = get_workspace_id()
    mol_res = build_molecule_from_smiles_engine("O", "Water")
    molecule_id = mol_res["molecule_id"]
    
    # Run IR simulation (runs locally on Windows)
    res = simulate_ir_spectrum_engine(workspace_id, molecule_id, method="MMFF94")
    assert res["ok"] is True
    assert "frequencies_cm1" in res["results"]
    assert "intensities" in res["results"]
    assert len(res["plot_bytes"]) > 0

@patch("ypotheto_compchem_mcp.modules.vibrations_tools._estimate_time_seconds", return_value=1)
def test_vibrations_tools_sync(mock_est):
    mol_res = build_molecule_from_smiles_engine("O", "Water")
    molecule_id = mol_res["molecule_id"]
    
    # Call calculate_vibrations tool synchronously
    envelope = calculate_vibrations(molecule_id, method="MMFF94", run_async=False)
    assert envelope["ok"] is True
    assert "zero_point_energy_ev" in envelope["results"]
    assert len(envelope["artifacts"]) == 1
    assert envelope["meta"]["provenance"]["software"] == "rdkit"
    assert envelope["meta"]["provenance"]["method"] == "MMFF94"
    
    # Call simulate_ir_spectrum tool synchronously
    ir_envelope = simulate_ir_spectrum(molecule_id, method="MMFF94", run_async=False)
    assert ir_envelope["ok"] is True
    assert len(ir_envelope["artifacts"]) == 1
    assert "ir_spectrum.png" in ir_envelope["artifacts"][0]["url"]
