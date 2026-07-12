import pytest
from unittest.mock import patch, MagicMock
from rdkit import Chem
from ypotheto_compchem_mcp.chemistry.builder_engine import build_molecule_from_smiles_engine
from ypotheto_compchem_mcp.workspace import get_workspace_id
from ypotheto_compchem_mcp.chemistry.ensemble_pipeline import run_ensemble_thermochemistry_engine
from ypotheto_compchem_mcp.modules.ensemble_tools import run_ensemble_thermochemistry

@patch("ypotheto_compchem_mcp.chemistry.ensemble_pipeline.CREST_AVAILABLE", True)
@patch("ypotheto_compchem_mcp.chemistry.ensemble_pipeline.XTB_AVAILABLE", True)
@patch("ypotheto_compchem_mcp.modules.ensemble_tools.CREST_AVAILABLE", True)
@patch("ypotheto_compchem_mcp.modules.ensemble_tools.XTB_AVAILABLE", True)
@patch("ypotheto_compchem_mcp.chemistry.ensemble_pipeline.run_conformer_search_engine")
@patch("ypotheto_compchem_mcp.chemistry.ensemble_pipeline.run_xtb_calculation_engine")
def test_ensemble_pipeline_success(mock_xtb, mock_crest):
    workspace_id = get_workspace_id()
    # Build Water
    mol_res = build_molecule_from_smiles_engine("O", "Water")
    molecule_id = mol_res["molecule_id"]
    
    # Mock CREST conformer search output (2 conformers)
    mock_crest.return_value = {
        "ok": True,
        "results": {
            "molecule_id": molecule_id,
            "num_conformers": 2,
            "conformers": [
                {
                    "conformer_id": f"{molecule_id}_conf_0",
                    "energy_hartree": -5.070,
                    "energy_ev": -137.9,
                    "relative_energy_kcal": 0.0,
                    "xyz_block": "3\nconf 0\nO 0.0 0.0 0.1\nH 0.0 0.7 -0.4\nH 0.0 -0.7 -0.4"
                },
                {
                    "conformer_id": f"{molecule_id}_conf_1",
                    "energy_hartree": -5.068,
                    "energy_ev": -137.8,
                    "relative_energy_kcal": 1.25,
                    "xyz_block": "3\nconf 1\nO 0.0 0.0 0.2\nH 0.0 0.7 -0.4\nH 0.0 -0.7 -0.4"
                }
            ]
        }
    }
    
    # Mock xTB calls
    # Call 1: Conf 0 Optimization
    # Call 2: Conf 0 Vibrations
    # Call 3: Conf 1 Optimization
    # Call 4: Conf 1 Vibrations
    def xtb_side_effect(ws, mol_id, task, **kwargs):
        if task == "geometry_optimization":
            # Conformer 0 energy: -138.0 eV, Conformer 1 energy: -137.95 eV
            energy = -138.0 if "ref_0" in mol_id else -137.95
            return {
                "ok": True,
                "results": {
                    "energy_ev": energy,
                    "dipole_moment_debye": [0.0, 0.0, 1.8]
                }
            }
        elif task == "vibrations":
            # Return realistic frequencies for Water (3 normal modes: symmetric stretch, bending, asymmetric stretch)
            # Plus some translation/rotation modes (which IdealGasThermo ignores if they are small or < 10 cm-1)
            return {
                "ok": True,
                "results": {
                    "frequencies_cm1": [1594.0, 3657.0, 3756.0]
                }
            }
        return {"ok": False}
        
    mock_xtb.side_effect = xtb_side_effect
    
    res = run_ensemble_thermochemistry_engine(
        workspace_id,
        molecule_id,
        method="GFN2-xTB",
        max_conformers_to_optimize=2,
        energy_threshold_kcal=3.0
    )
    
    assert res["ok"] is True
    results = res["results"]
    assert results["molecule_id"] == molecule_id
    assert len(results["refined_conformers"]) == 2
    
    # Gibbs energies
    conf0 = results["refined_conformers"][0]
    conf1 = results["refined_conformers"][1]
    
    assert conf0["electronic_energy_ev"] == -138.0
    assert conf1["electronic_energy_ev"] == -137.95
    
    # Both should have valid thermochemistry corrections computed via IdealGasThermo
    assert conf0["gibbs_correction_ev"] != 0.0
    assert conf1["gibbs_correction_ev"] != 0.0
    
    # Boltzmann weighting checks
    # Conformer 0 Gibbs free energy should be lower than Conformer 1 Gibbs free energy
    assert conf0["total_gibbs_energy_ev"] < conf1["total_gibbs_energy_ev"]
    assert conf0["boltzmann_population"] > conf1["boltzmann_population"]
    assert conf0["boltzmann_population"] + conf1["boltzmann_population"] == pytest.approx(1.0)
    
    # Ensemble free energy should lie between the two conformer values
    assert conf0["total_gibbs_energy_ev"] < results["ensemble_gibbs_free_energy_ev"] < conf1["total_gibbs_energy_ev"]

@patch("ypotheto_compchem_mcp.modules.ensemble_tools.CREST_AVAILABLE", False)
def test_ensemble_pipeline_missing_binaries_graceful_fail():
    workspace_id = get_workspace_id()
    # Build Water
    mol_res = build_molecule_from_smiles_engine("O", "Water")
    molecule_id = mol_res["molecule_id"]
    
    res = run_ensemble_thermochemistry(molecule_id, run_async=False)
    assert res["ok"] is False
    assert res["error"]["code"] == "BACKEND_UNAVAILABLE"
    assert "CREST and xTB binaries are required" in res["error"]["message"]
