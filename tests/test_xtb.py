import os
import pytest
from unittest.mock import patch, MagicMock
from rdkit import Chem
from ypotheto_compchem_mcp.chemistry.builder_engine import build_molecule_from_smiles_engine, load_molecule_from_workspace
from ypotheto_compchem_mcp.workspace import get_workspace_id
from ypotheto_compchem_mcp.chemistry.xtb_engine import (
    run_xtb_calculation_engine,
    run_conformer_search_engine
)
from ypotheto_compchem_mcp.modules.xtb_tools import run_xtb_calculation, run_conformer_search

def test_xtb_missing_throws():
    with patch("ypotheto_compchem_mcp.chemistry.xtb_engine.XTB_AVAILABLE", False):
        with pytest.raises(RuntimeError) as exc:
            run_xtb_calculation_engine("ws", "mol", "single_point")
        assert "xtb executable is not available" in str(exc.value)

def test_crest_missing_throws():
    with patch("ypotheto_compchem_mcp.chemistry.xtb_engine.CREST_AVAILABLE", False):
        with pytest.raises(RuntimeError) as exc:
            run_conformer_search_engine("ws", "mol")
        assert "crest or xtb executable is not available" in str(exc.value)

@patch("ypotheto_compchem_mcp.chemistry.xtb_engine.XTB_AVAILABLE", True)
@patch("subprocess.run")
def test_xtb_single_point_parsing(mock_run):
    workspace_id = get_workspace_id()
    # Build Water
    mol_res = build_molecule_from_smiles_engine("O", "Water")
    molecule_id = mol_res["molecule_id"]
    
    # Mock stdout containing realistic xTB total energy and dipole moment outputs
    mock_stdout = """
          -------------------------------------------------
          | xTB version 6.5.1                             |
          -------------------------------------------------
          
          :: total energy               -5.0704818731558 Eh
          
          | TOTAL ENERGY               -5.0704818731558 Eh   |
          
          dipole:
                 x           y           z         tot (Debye)
           0.00000     0.00000    -1.85420       1.85420
    """
    
    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.stdout = mock_stdout
    mock_proc.stderr = ""
    mock_run.return_value = mock_proc
    
    res = run_xtb_calculation_engine(workspace_id, molecule_id, task="single_point")
    assert res["ok"] is True
    assert res["results"]["energy_hartree"] == -5.0704818731558
    assert res["results"]["energy_ev"] == round(-5.0704818731558 * 27.211386, 4)
    assert res["results"]["dipole_moment_debye"] == [0.0, 0.0, -1.85420]

@patch("ypotheto_compchem_mcp.chemistry.xtb_engine.XTB_AVAILABLE", True)
@patch("subprocess.run")
def test_xtb_opt_coordinate_transfer(mock_run):
    workspace_id = get_workspace_id()
    # Build Water
    mol_res = build_molecule_from_smiles_engine("O", "Water")
    molecule_id = mol_res["molecule_id"]
    
    # Mock opt XYZ file generation during optimization
    original_run = subprocess_run_stub = mock_run
    
    def side_effect(args, **kwargs):
        # Write dummy optimized coordinate file to the target cwd directory
        cwd = kwargs.get("cwd")
        if cwd:
            xyz_content = """3
optimized water coordinates
O   0.00000000   0.00000000   0.15000000
H   0.00000000   0.76000000  -0.48000000
H   0.00000000  -0.76000000  -0.48000000
"""
            with open(os.path.join(cwd, "xtbopt.xyz"), "w", encoding="utf-8") as f:
                f.write(xyz_content)
                
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = "TOTAL ENERGY -5.0704818731558 Eh"
        mock_proc.stderr = ""
        return mock_proc
        
    mock_run.side_effect = side_effect
    
    res = run_xtb_calculation_engine(workspace_id, molecule_id, task="geometry_optimization")
    assert res["ok"] is True
    assert res["results"]["energy_hartree"] == -5.0704818731558
    
    # Load optimized molecule from workspace and check coordinates
    opt_mol = load_molecule_from_workspace(workspace_id, molecule_id)
    conf = opt_mol.GetConformer()
    assert conf.GetAtomPosition(0).z == pytest.approx(0.15)

@patch("ypotheto_compchem_mcp.chemistry.xtb_engine.CREST_AVAILABLE", True)
@patch("ypotheto_compchem_mcp.chemistry.xtb_engine.XTB_AVAILABLE", True)
@patch("subprocess.run")
def test_crest_conformer_boltzmann_weights(mock_run):
    workspace_id = get_workspace_id()
    # Build Water (3 atoms)
    mol_res = build_molecule_from_smiles_engine("O", "Water")
    molecule_id = mol_res["molecule_id"]
    
    def side_effect(args, **kwargs):
        cwd = kwargs.get("cwd")
        if cwd:
            # Write 2 dummy conformers to crest_conformers.xyz
            xyz_content = """3
conformer 0 energy: -5.070
O   0.00000000   0.00000000   0.119
H   0.00000000   0.76100000  -0.478
H   0.00000000  -0.76100000  -0.478
3
conformer 1 energy: -5.068
O   0.00000000   0.00000000   0.120
H   0.00000000   0.76200000  -0.479
H   0.00000000  -0.76200000  -0.479
"""
            with open(os.path.join(cwd, "crest_conformers.xyz"), "w", encoding="utf-8") as f:
                f.write(xyz_content)
                
            # Write energies in Eh (Hartree)
            # Energy gap = 0.002 Eh (approx 1.25 kcal/mol)
            energies_content = """-5.070
-5.068
"""
            with open(os.path.join(cwd, "crest.energies"), "w", encoding="utf-8") as f:
                f.write(energies_content)
                
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = "CREST conformer search done."
        mock_proc.stderr = ""
        return mock_proc
        
    mock_run.side_effect = side_effect
    
    res = run_conformer_search_engine(workspace_id, molecule_id)
    assert res["ok"] is True
    assert res["results"]["num_conformers"] == 2
    
    confs = res["results"]["conformers"]
    assert confs[0]["energy_hartree"] == -5.070
    assert confs[1]["energy_hartree"] == -5.068
    
    # Delta E = 0.002 Eh * 627.509 kcal/mol = 1.255 kcal/mol
    assert confs[1]["relative_energy_kcal"] == pytest.approx(1.255, abs=0.01)
    
    # Boltzmann population: e^(-0.0) vs e^(-1.255 / 0.59248) -> e^(-2.118) = 0.120
    # Rel weights: 1.0 vs 0.120 -> populations: 1/1.12 = 89%, 0.12/1.12 = 11%
    assert confs[0]["boltzmann_population"] > 0.85
    assert confs[1]["boltzmann_population"] < 0.15
