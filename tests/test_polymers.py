import os
from unittest.mock import MagicMock, patch

from ypotheto_compchem_mcp.chemistry.polymer_engine import (
    build_polymer_chain_engine,
    pack_amorphous_cell_engine,
    register_monomer_engine,
    run_lammps_simulation_engine,
)
from ypotheto_compchem_mcp.modules.polymer_tools import (
    analyze_md_trajectory,
    build_polymer_chain,
    pack_amorphous_cell,
    register_monomer,
    run_lammps_simulation,
)
from ypotheto_compchem_mcp.workspace import get_workspace_id


def test_polymer_chain_construction():
    workspace_id = get_workspace_id()
    
    # 1. Register monomer
    mon_meta = register_monomer_engine(workspace_id, "*CC*", "Ethylene")
    assert mon_meta["name"] == "Ethylene"
    assert "monomer_id" in mon_meta
    
    # 2. Build polymer chain
    chain_res = build_polymer_chain_engine(workspace_id, mon_meta["monomer_id"], dp=3)
    assert "polymer_molecule_id" in chain_res
    assert chain_res["num_atoms"] > 0
    assert "smiles" in chain_res
    
    # 3. Test MCP Tool wrappers
    res = register_monomer("*CC*", "Ethylene")
    assert res["ok"] is True, f"register_monomer failed: {res}"
    
    res_chain = build_polymer_chain(res["results"]["monomer_id"], dp=2)
    assert res_chain["ok"] is True, f"build_polymer_chain failed: {res_chain}"
    assert "polymer_molecule_id" in res_chain["results"]

def test_pack_amorphous_cell_and_simulate():
    workspace_id = get_workspace_id()
    
    # Create two dummy molecules in workspace
    from ypotheto_compchem_mcp.chemistry.builder_engine import save_molecule_coords
    mol1_xyz = "3\nWater molecule\nO 0.0 0.0 0.0\nH 0.0 0.0 0.9\nH 0.0 0.9 0.0"
    mol2_xyz = "2\nHydrogen gas\nH 0.0 0.0 0.0\nH 0.0 0.0 0.7"
    
    save_molecule_coords(workspace_id, "mol_h2o", "", mol1_xyz, {"molecule_id": "mol_h2o", "name": "water"})
    save_molecule_coords(workspace_id, "mol_h2", "", mol2_xyz, {"molecule_id": "mol_h2", "name": "hydrogen"})
    
    # Pack amorphous cell
    pack_res = pack_amorphous_cell_engine(
        workspace_id=workspace_id,
        molecule_ids=["mol_h2o", "mol_h2"],
        counts=[2, 3],
        density_g_cm3=0.8,
        box_size_angstrom=12.0
    )
    assert pack_res["ok"] is True, f"pack_amorphous_cell_engine failed: {pack_res}"
    assert pack_res["num_atoms"] == 12 # 2*3 + 3*2 = 12 atoms
    assert "cell_" in pack_res["packed_molecule_id"]
    
    # Run simulation - no LAMMPS binary is installed in this environment, so this
    # exercises the ASE-LJ fallback path, which must now disclose itself honestly
    # (engine_used + a fallback warning) rather than silently reporting as LAMMPS.
    sim_res = run_lammps_simulation_engine(
        workspace_id=workspace_id,
        packed_cell_id=pack_res["packed_molecule_id"],
        steps=50,
        timestep_fs=1.0,
        temperature_k=300.0,
        ensemble="nvt"
    )
    assert sim_res["ok"] is True, f"run_lammps_simulation_engine failed: {sim_res}"
    assert "results" in sim_res
    assert sim_res["results"]["final_density_g_cm3"] > 0.0
    assert sim_res["results"]["engine_used"] == "ase-lj-fallback"
    assert len(sim_res["warnings"]) == 1
    assert sim_res["warnings"][0]["type"] == "fallback"

    # Tool wrapper packing test
    tool_pack = pack_amorphous_cell(
        molecule_ids=["mol_h2o", "mol_h2"],
        counts=[2, 3],
        density_g_cm3=0.8,
        box_size_angstrom=12.0,
        run_async=False
    )
    assert tool_pack["ok"] is True, f"pack_amorphous_cell tool failed: {tool_pack}"
    assert "packed_molecule_id" in tool_pack["results"]
    
    # Tool wrapper simulation test
    tool_sim = run_lammps_simulation(
        packed_molecule_id=tool_pack["results"]["packed_molecule_id"],
        steps=50,
        timestep_fs=1.0,
        temperature_k=300.0,
        ensemble="nvt",
        run_async=False
    )
    assert tool_sim["ok"] is True, f"run_lammps_simulation tool failed: {tool_sim}"
    assert "final_density_g_cm3" in tool_sim["results"]
    assert tool_sim["results"]["engine_used"] == "ase-lj-fallback"
    assert len(tool_sim["warnings"]) == 1
    
    # Trajectory analysis
    traj_url = sim_res["results"]["trajectory_file_url"]
    analysis_res = analyze_md_trajectory(traj_url)
    assert analysis_res["ok"] is True, f"analyze_md_trajectory failed: {analysis_res}"
    assert "radius_of_gyration_angstrom" in analysis_res["results"]
    assert "mean_squared_displacement_angstrom2" in analysis_res["results"]


def test_lammps_simulation_parses_real_thermo_output_when_available():
    """
    With LAMMPS available (mocked) and a successful run, the engine must report
    parsed thermo values and engine_used == "lammps" - not the old hardcoded
    placeholders (-150.0 kcal/mol, 0.9 g/cm3).
    """
    workspace_id = get_workspace_id()

    from ypotheto_compchem_mcp.chemistry.builder_engine import save_molecule_coords
    mol_xyz = "2\nHydrogen gas\nH 0.0 0.0 0.0\nH 0.0 0.0 0.7"
    save_molecule_coords(workspace_id, "mol_h2_lmp", "", mol_xyz, {"molecule_id": "mol_h2_lmp", "name": "hydrogen"})

    pack_res = pack_amorphous_cell_engine(
        workspace_id=workspace_id,
        molecule_ids=["mol_h2_lmp"],
        counts=[4],
        density_g_cm3=0.8,
        box_size_angstrom=12.0
    )
    assert pack_res["ok"] is True

    canned_stdout = (
        "Step          Temp          Press          PotEng         KinEng         TotEng        Density\n"
        "         0   300           1.2345        -20.5          0.895         -19.605       0.87654321\n"
        "       100   298.5         1.1998        -22.3          0.891         -21.409       0.87700000\n"
        "Loop time of 0.01 on 1 procs for 100 steps with 8 atoms\n"
    )

    def fake_subprocess_run(cmd, cwd, check, stdout, stderr, text):
        traj_path = os.path.join(cwd, "trajectory.xyz")
        with open(traj_path, "w", encoding="utf-8") as f:
            f.write("8\nstep 0\nH 0.0 0.0 0.0\n" * 1)
        stdout.write(canned_stdout)
        return MagicMock(stdout=None, stderr="", returncode=0)

    with patch("ypotheto_compchem_mcp.chemistry.polymer_engine.LAMMPS_AVAILABLE", True), \
         patch("ypotheto_compchem_mcp.chemistry.polymer_engine.subprocess.run", side_effect=fake_subprocess_run), \
         patch("ypotheto_compchem_mcp.chemistry.polymer_engine.shutil.which", return_value="/usr/bin/lmp"):
        sim_res = run_lammps_simulation_engine(
            workspace_id=workspace_id,
            packed_cell_id=pack_res["packed_molecule_id"],
            steps=100,
            timestep_fs=1.0,
            temperature_k=300.0,
            ensemble="nvt"
        )

    assert sim_res["ok"] is True
    assert sim_res["results"]["engine_used"] == "lammps"
    assert sim_res["results"]["potential_energy_kcal_mol"] == -22.3
    assert sim_res["results"]["final_density_g_cm3"] == 0.877
    assert sim_res["warnings"] == []
