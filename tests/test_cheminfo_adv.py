import pytest
from ypotheto_compchem_mcp.workspace import get_workspace_id
from ypotheto_compchem_mcp.chemistry.standardizer import standardize_molecule_engine, enumerate_tautomers_engine
from ypotheto_compchem_mcp.chemistry.conformer_engine import search_conformers_engine, save_conformer_as_molecule_engine
from ypotheto_compchem_mcp.chemistry.polymer_engine import register_monomer_engine, build_polymer_chain_engine
from ypotheto_compchem_mcp.modules.cheminformatics_tools import standardize_molecule, enumerate_tautomers, search_conformers, save_conformer_as_molecule
from ypotheto_compchem_mcp.modules.polymer_tools import register_monomer, build_polymer_chain

def test_standardize_molecule_salt_neutralize():
    workspace_id = get_workspace_id()
    # Sodium acetate containing salt Na+ and charged acetate group
    acetate_smiles = "CC(=O)[O-].[Na+]"
    
    res = standardize_molecule_engine(workspace_id, acetate_smiles)
    # Salt should be stripped and acetate neutralized to Acetic Acid (CC(=O)O)
    assert res["standardized_smiles"] == "CC(=O)O"
    assert "Stripped salts/counter-ions" in res["steps_taken"]
    assert "Neutralized formal charges" in res["steps_taken"]

def test_tautomer_enumeration():
    workspace_id = get_workspace_id()
    # Standardize enol form of Acetone to canonical keto form
    res = standardize_molecule_engine(workspace_id, "CC(=C)O")
    assert res["standardized_smiles"] == "CC(C)=O" # canonical form should be Acetone
    molecule_id = res["molecule_id"]
    
    # Enumerate tautomers
    enum_res = enumerate_tautomers_engine(workspace_id, molecule_id)
    assert enum_res["tautomers_count"] >= 2
    smiles_list = [t["smiles"] for t in enum_res["tautomers"]]
    assert "CC(C)=O" in smiles_list
    assert "C=C(C)O" in smiles_list

def test_conformer_search_butane():
    workspace_id = get_workspace_id()
    # Build butane
    from ypotheto_compchem_mcp.chemistry.builder_engine import build_molecule_from_smiles_engine
    mol_res = build_molecule_from_smiles_engine("CCCC", "Butane")
    molecule_id = mol_res["molecule_id"]
    
    # Search conformers
    res = search_conformers_engine(workspace_id, molecule_id, num_conformers=10, rmsd_threshold=0.3)
    assert res["conformers_found"] >= 2 # Anti and Gauche conformers
    assert res["conformers"][0]["relative_energy_kcal_mol"] == 0.0 # Lowest energy conformer
    assert res["conformers"][0]["boltzmann_population"] > 0.0
    
    # Extract conformer
    rdkit_conf_id = res["conformers"][0]["rdkit_conformer_id"]
    ext_res = save_conformer_as_molecule_engine(workspace_id, molecule_id, rdkit_conf_id)
    assert ext_res["molecule_id"].startswith("mol_")
    assert ext_res["num_atoms"] == mol_res["num_atoms"]

def test_polymer_chain_propylene():
    workspace_id = get_workspace_id()
    # Register Propylene monomer with connection points [1*] and [2*]
    # *CC(*)C -> first * is [1*], second * is [2*]
    monomer = register_monomer_engine(workspace_id, "*CC(*)C", name="Propylene")
    assert monomer["monomer_id"].startswith("mon_")
    assert "[1*]" in monomer["smiles"]
    assert "[2*]" in monomer["smiles"]
    
    # Build polypropylene trimer (DP = 3)
    polymer = build_polymer_chain_engine(workspace_id, monomer["monomer_id"], dp=3)
    assert polymer["polymer_molecule_id"].startswith("mol_")
    assert polymer["dp"] == 3
    # Check that dummy atoms are stripped/capped in final SMILES (contains no '*' or 'dummy')
    assert "*" not in polymer["smiles"]
    # Propylene trimer formula is C9H20 (since end groups are capped with H)
    assert polymer["formula"] == "C9H20"

def test_cheminfo_tools_mcp():
    # Test through the FastMCP tool wrappers
    # 1. Standardize Na Acetate
    envelope = standardize_molecule("CC(=O)[O-].[Na+]")
    assert envelope["ok"] is True
    assert envelope["results"]["standardized_smiles"] == "CC(=O)O"
    molecule_id = envelope["results"]["molecule_id"]
    assert len(envelope["artifacts"]) == 1
    
    # 2. Enumerate tautomers
    taut_envelope = enumerate_tautomers(molecule_id)
    assert taut_envelope["ok"] is True
    
    # 3. Conformer search
    conf_envelope = search_conformers(molecule_id, num_conformers=5)
    assert conf_envelope["ok"] is True
    
    # 4. Save conformer
    conf_id = conf_envelope["results"]["conformers"][0]["rdkit_conformer_id"]
    save_envelope = save_conformer_as_molecule(molecule_id, conf_id)
    assert save_envelope["ok"] is True

def test_polymer_tools_mcp():
    # 1. Register monomer
    envelope = register_monomer("*CC(*)C", name="Propylene MCP")
    assert envelope["ok"] is True
    monomer_id = envelope["results"]["monomer_id"]
    
    # 2. Build polypropylene
    chain_envelope = build_polymer_chain(monomer_id, dp=4)
    assert chain_envelope["ok"] is True
    assert chain_envelope["results"]["dp"] == 4
    assert len(chain_envelope["artifacts"]) == 1
