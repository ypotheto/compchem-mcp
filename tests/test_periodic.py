import pytest
from ypotheto_compchem_mcp.workspace import get_workspace_id
from ypotheto_compchem_mcp.chemistry.periodic_engine import (
    import_periodic_structure_engine,
    analyze_crystal_symmetry_engine,
    generate_supercell_engine,
    build_surface_slab_engine,
    add_adsorbate_to_surface_engine,
    run_periodic_dft_engine
)
from ypotheto_compchem_mcp.modules.periodic_tools import (
    import_periodic_structure,
    analyze_crystal_symmetry,
    generate_supercell,
    build_surface_slab,
    add_adsorbate_to_surface,
    run_periodic_dft
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


def test_slab_adsorbate_dft():
    workspace_id = get_workspace_id()
    
    # 1. Import bulk silicon crystal
    res_import = import_periodic_structure_engine(workspace_id, SILICON_CIF)
    bulk_id = res_import["molecule_id"]
    
    # 2. Build slab
    slab_res = build_surface_slab_engine(workspace_id, bulk_id, miller_indices=[1, 1, 1], layers=3, vacuum_size=8.0)
    assert slab_res["ok"] is True
    slab_id = slab_res["results"]["slab_molecule_id"]
    assert slab_res["results"]["num_atoms"] > 0
    
    # 3. Create dummy adsorbate
    from ypotheto_compchem_mcp.chemistry.builder_engine import save_molecule_coords
    co_xyz = "2\nCarbon Monoxide\nC 0.0 0.0 0.0\nO 0.0 0.0 1.1"
    save_molecule_coords(workspace_id, "mol_co", "", co_xyz, {"molecule_id": "mol_co", "name": "CO"})
    
    # 4. Add adsorbate to surface slab
    ads_res = add_adsorbate_to_surface_engine(workspace_id, slab_id, "mol_co", height=1.6, position_type="ontop")
    assert ads_res["ok"] is True
    combined_id = ads_res["results"]["combined_molecule_id"]
    
    # 5. Run periodic boundary calculation (sync)
    dft_res = run_periodic_dft_engine(workspace_id, combined_id, kpts=[1, 1, 1], method="xTB")
    assert dft_res["ok"] is True
    assert "energy_ev" in dft_res["results"]
    
    # 6. Test FastMCP tool wrappers
    tool_slab = build_surface_slab(bulk_id, miller_indices=[1, 1, 0], layers=2, vacuum_size=5.0)
    assert tool_slab["ok"] is True
    
    tool_ads = add_adsorbate_to_surface(tool_slab["results"]["slab_molecule_id"], "mol_co", height=1.2, position_type="0.2,0.2")
    assert tool_ads["ok"] is True
    
    tool_dft = run_periodic_dft(tool_ads["results"]["combined_molecule_id"], kpts=[1, 1, 1], method="xTB", run_async=False)
    assert tool_dft["ok"] is True
    assert "energy_ev" in tool_dft["results"]
