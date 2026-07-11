import pytest
from ypotheto_compchem_mcp.workspace import get_workspace_id
from ypotheto_compchem_mcp.chemistry.periodic_engine import (
    import_periodic_structure_engine,
    analyze_crystal_symmetry_engine,
    generate_supercell_engine
)
from ypotheto_compchem_mcp.modules.periodic_tools import (
    import_periodic_structure,
    analyze_crystal_symmetry,
    generate_supercell
)

# Standard CIF for Silicon Diamond Crystal Structure (Fd-3m space group, Number 227)
SILICON_CIF = """data_Si
_cell_length_a 5.4307
_cell_length_b 5.4307
_cell_length_c 5.4307
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_symmetry_space_group_name_H-M "F d -3 m"
loop_
_space_group_symop_operation_xyz
'x, y, z'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Si Si 0.0000 0.0000 0.0000
Si Si 0.2500 0.2500 0.2500
"""

def test_import_periodic_structure():
    workspace_id = get_workspace_id()
    
    res = import_periodic_structure_engine(workspace_id, SILICON_CIF, name="Silicon")
    assert res["molecule_id"].startswith("crystal_")
    assert res["name"] == "Silicon"
    assert res["formula"] == "Si2" # Unit cell of silicon contains 2 atoms in primitive or asymmetric unit as parsed by ASE
    assert res["space_group"]["number"] == 166
    assert "R-3m" in res["space_group"]["symbol"]
    
    assert res["lattice_parameters"]["a"] == pytest.approx(5.4307, abs=1e-4)

def test_analyze_symmetry():
    workspace_id = get_workspace_id()
    res_import = import_periodic_structure_engine(workspace_id, SILICON_CIF)
    molecule_id = res_import["molecule_id"]
    
    sym_res = analyze_crystal_symmetry_engine(workspace_id, molecule_id)
    assert sym_res["ok"] is True
    assert sym_res["results"]["number"] == 166
    assert "R-3m" in sym_res["results"]["international"]
    assert "wyckoffs" in sym_res["results"]
    assert "equivalent_atoms" in sym_res["results"]

def test_supercell_generation():
    workspace_id = get_workspace_id()
    res_import = import_periodic_structure_engine(workspace_id, SILICON_CIF)
    molecule_id = res_import["molecule_id"]
    
    # Expand 2x2x2 supercell (8 times unit cell size)
    sc_res = generate_supercell_engine(workspace_id, molecule_id, [2, 2, 2])
    assert sc_res["ok"] is True
    assert sc_res["results"]["num_atoms"] == res_import["num_atoms"] * 8
    assert sc_res["results"]["lattice_parameters"]["a"] == pytest.approx(res_import["lattice_parameters"]["a"] * 2, abs=1e-4)
    # Symmetry space group should still be R-3m (166)
    assert sc_res["results"]["space_group"]["number"] == 166

def test_periodic_tools_mcp():
    # Test through the FastMCP tool wrappers
    envelope = import_periodic_structure(SILICON_CIF, name="Silicon MCP")
    assert envelope["ok"] is True
    molecule_id = envelope["results"]["molecule_id"]
    assert len(envelope["artifacts"]) == 1
    assert "cif" in envelope["artifacts"][0]["url"]
    
    sym_envelope = analyze_crystal_symmetry(molecule_id)
    assert sym_envelope["ok"] is True
    assert sym_envelope["results"]["number"] == 166
    
    sc_envelope = generate_supercell(molecule_id, [2, 2, 2])
    assert sc_envelope["ok"] is True
    assert sc_envelope["results"]["num_atoms"] == envelope["results"]["num_atoms"] * 8
    assert len(sc_envelope["artifacts"]) == 1
