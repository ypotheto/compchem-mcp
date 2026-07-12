import pytest
from rdkit import Chem
from ypotheto_compchem_mcp.chemistry.preflight import (
    validate_charge_spin_multiplicity,
    validate_basis_set_coverage,
    estimate_computational_resources
)

def test_charge_spin_parity():
    # Water (H2O) -> 10 electrons (Z=8 for O, Z=1 for each H)
    mol_water = Chem.MolFromSmiles("O")
    mol_water = Chem.AddHs(mol_water)
    
    # Even electrons: singlet (1) is valid
    ok, err = validate_charge_spin_multiplicity(mol_water, charge=0, spin=1)
    assert ok is True
    
    # Even electrons: doublet (2) is invalid
    ok, err = validate_charge_spin_multiplicity(mol_water, charge=0, spin=2)
    assert ok is False
    assert "Spin multiplicity must be ODD" in err
    
    # Even electrons: triplet (3) is valid
    ok, err = validate_charge_spin_multiplicity(mol_water, charge=0, spin=3)
    assert ok is True
    
    # Cation H2O+ -> 9 electrons. Doublet (2) is valid
    ok, err = validate_charge_spin_multiplicity(mol_water, charge=1, spin=2)
    assert ok is True
    
    # Cation H2O+ -> 9 electrons. Singlet (1) is invalid
    ok, err = validate_charge_spin_multiplicity(mol_water, charge=1, spin=1)
    assert ok is False
    assert "Spin multiplicity must be EVEN" in err
    
    # Multiplicity too high (unpaired spins > total electrons)
    ok, err = validate_charge_spin_multiplicity(mol_water, charge=0, spin=15)
    assert ok is False
    assert "exceeds the total number of electrons" in err

def test_basis_set_coverage():
    # Water (H2O) -> Only light atoms (H, O)
    mol_water = Chem.MolFromSmiles("O")
    
    ok, err = validate_basis_set_coverage(mol_water, "6-31g*")
    assert ok is True
    
    # Heavy halogen molecule -> Iodine (Z=53)
    mol_iodine = Chem.MolFromSmiles("I") # Hydroiodic acid / Iodine atom
    
    # 6-31g supports up to Ar (Z=18). Iodine is Z=53, so it should fail.
    ok, err = validate_basis_set_coverage(mol_iodine, "6-31g*")
    assert ok is False
    assert "does not support heavy elements" in err
    
    # def2-svp supports up to Rn (Z=86). Iodine should pass.
    ok, err = validate_basis_set_coverage(mol_iodine, "def2-svp")
    assert ok is True

def test_resource_estimation():
    mol_water = Chem.MolFromSmiles("O")
    mol_water = Chem.AddHs(mol_water)
    
    # Force field
    est_ff = estimate_computational_resources(mol_water, "MMFF94", "sto-3g", "single_point")
    assert est_ff["estimated_wall_time_seconds"] == 1
    assert est_ff["recommended_run_mode"] == "sync"
    
    # DFT on sto-3g
    est_dft = estimate_computational_resources(mol_water, "DFT", "sto-3g", "single_point")
    assert est_dft["estimated_wall_time_seconds"] < 10
    assert est_dft["recommended_run_mode"] == "sync"
    
    # Large molecule (e.g. butane) DFT with larger basis cc-pvtz
    mol_large = Chem.MolFromSmiles("CCCC")
    mol_large = Chem.AddHs(mol_large)
    
    est_opt = estimate_computational_resources(mol_large, "DFT", "cc-pvtz", "geometry_optimization")
    assert est_opt["estimated_wall_time_seconds"] > 10
    assert est_opt["recommended_run_mode"] == "async"

def test_preflight_mcp_tool():
    from ypotheto_compchem_mcp.modules.scientific_preflight_tools import run_scientific_preflight
    from ypotheto_compchem_mcp.chemistry.builder_engine import build_molecule_from_smiles_engine
    
    # Build water molecule in workspace
    res = build_molecule_from_smiles_engine("O")
    molecule_id = res["molecule_id"]
    
    # Run preflight tool (valid singlet water sto-3g)
    tool_res = run_scientific_preflight(molecule_id, method="DFT", basis="sto-3g", charge=0, spin=1, task="single_point")
    assert tool_res["ok"] is True
    assert tool_res["results"]["preflight_passed"] is True
    assert tool_res["results"]["validation"]["total_electrons"] == 10

    # Compute-credits estimate is surfaced as advisory info: present in the
    # machine-readable results AND mentioned in the interpretation string, so
    # the client LLM can warn users before launching an expensive job. This is
    # advisory only - nothing here enforces or bills against it.
    assert "compute_credits_cost" in tool_res["results"]["estimates"]
    assert tool_res["results"]["estimates"]["compute_credits_cost"] >= 0
    assert "credits" in tool_res["interpretation"]
    
    # Run preflight tool (invalid doublet water sto-3g)
    tool_res_fail = run_scientific_preflight(molecule_id, method="DFT", basis="sto-3g", charge=0, spin=2, task="single_point")
    assert tool_res_fail["ok"] is False
    assert tool_res_fail["error"]["code"] == "INVALID_CHARGE_SPIN"
