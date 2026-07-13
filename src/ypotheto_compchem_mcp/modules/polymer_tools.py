
from mcp.server.fastmcp import FastMCP

from ypotheto_compchem_mcp.artifacts import register_artifact
from ypotheto_compchem_mcp.chemistry.polymer_engine import (
    analyze_md_trajectory_engine,
    build_polymer_chain_engine,
    pack_amorphous_cell_engine,
    register_monomer_engine,
    run_lammps_simulation_engine,
)
from ypotheto_compchem_mcp.envelope import make_success_response, mcp_tool_decorator
from ypotheto_compchem_mcp.workspace import get_workspace_id, workspace_manager


def _finalize_pack_amorphous_cell(res: dict) -> dict:
    interpretation = (
        f"Amorphous cell packed successfully: {res['packed_molecule_id']} ({res['name']}).\n"
        f"Box Size = {res['box_size_angstrom']:.2f} A, Target Density = {res['density_g_cm3']:.2f} g/cm3, Total Atoms = {res['num_atoms']}."
    )
    return make_success_response(
        results=res,
        interpretation=interpretation,
        meta={"packed_molecule_id": res["packed_molecule_id"]}
    )

def run_pack_amorphous_cell_job(workspace_id, molecule_ids, counts, density_g_cm3, box_size_angstrom):
    res = pack_amorphous_cell_engine(workspace_id, molecule_ids, counts, density_g_cm3, box_size_angstrom)
    return _finalize_pack_amorphous_cell(res)

@mcp_tool_decorator
def register_monomer(
    smiles: str,
    name: str,
    head_idx: int | None = None,
    tail_idx: int | None = None
) -> dict:
    """
    Register a monomer repeat unit, defining attachment connection points for polymer building.
    You can either:
    1. Pass a SMILES string with connection dummy atoms '*' (e.g. '*CC(*)C' for propylene,
       where the first '*' acts as head [1*] and the second '*' acts as tail [2*]).
    2. Pass a standard SMILES string and specify head_idx and tail_idx.
    
    Parameters:
    - smiles: SMILES representation of the repeat unit.
    - name: Human-readable label for the monomer (e.g., 'Propylene').
    - head_idx: The 0-based atom index that connects to the previous unit's tail (optional).
    - tail_idx: The 0-based atom index that connects to the next unit's head (optional).
    """
    workspace_id = get_workspace_id()
    
    res = register_monomer_engine(workspace_id, smiles, name, head_idx, tail_idx)
    
    interpretation = (
        f"Monomer repeat unit '{name}' registered successfully under ID: {res['monomer_id']}. "
        f"Connectivity representation: {res['smiles']}."
    )
    
    return make_success_response(
        results=res,
        interpretation=interpretation,
        meta={
            "monomer_id": res["monomer_id"],
            "type": "monomer_definition"
        }
    )

@mcp_tool_decorator
def build_polymer_chain(
    monomer_id: str,
    dp: int,
    tacticity: str = "isotactic",
    name: str | None = None
) -> dict:
    """
    Assemble repeat units head-to-tail to form a 3D-relaxed polymer chain of specified length.
    
    Parameters:
    - monomer_id: The registered monomer handle (e.g., mon_a1b2c3d4).
    - dp: Degree of Polymerization (total repeat units in chain).
    - tacticity: Stereocontrol, either 'isotactic', 'syndiotactic', or 'atactic' (default 'isotactic').
    - name: Optional label for the resulting polymer molecule.
    """
    workspace_id = get_workspace_id()
    
    res = build_polymer_chain_engine(workspace_id, monomer_id, dp, tacticity, name)
    polymer_id = res["polymer_molecule_id"]
    
    # Register 2D depict SVG
    svg_art = register_artifact(
        f"{polymer_id}.svg",
        res["svg_data"].encode("utf-8"),
        "depiction",
        f"2D depiction of polymer chain {polymer_id}"
    )
    
    interpretation = (
        f"Polymer chain constructed successfully: {polymer_id} ({res['name']}). "
        f"Degree of Polymerization (DP) = {dp}. "
        f"Formula: {res['formula']}, Total Atoms: {res['num_atoms']}."
    )
    
    res_clean = {k: v for k, v in res.items() if k != "svg_data"}
    
    return make_success_response(
        results=res_clean,
        interpretation=interpretation,
        artifacts=[svg_art],
        meta={
            "molecule_id": polymer_id,
            "monomer_id": monomer_id,
            "dp": dp
        }
    )


@mcp_tool_decorator
def pack_amorphous_cell(
    molecule_ids: list[str],
    counts: list[int],
    density_g_cm3: float = 0.9,
    box_size_angstrom: float | None = None,
    run_async: bool = True
) -> dict:
    """
    Pack polymer chains and solvent molecules into a periodic box using Packmol.
    
    Parameters:
    - molecule_ids: List of workspace molecule IDs (e.g. ['mol_a1b2', 'mol_c3d4'])
    - counts: Number of copies of each molecule to pack (e.g. [5, 100])
    - density_g_cm3: Target density of packed cell in g/cm3 (default is 0.9)
    - box_size_angstrom: Optional box size side length. Estimated if not provided.
    - run_async: If true, runs packing in background (recommended, default is True).
    """
    workspace_id = get_workspace_id()
    est_sec = 10
    
    # Lazy import to avoid circular dependency
    from ypotheto_compchem_mcp.jobs import job_manager
    
    if run_async:
        job = job_manager.submit_job(
            workspace_id,
            run_pack_amorphous_cell_job,
            est_sec,
            workspace_id,
            molecule_ids,
            counts,
            density_g_cm3,
            box_size_angstrom
        )
        return make_success_response(
            results={
                "job_id": job.job_id,
                "status": job.status,
                "estimated_time_seconds": job.estimated_time_seconds,
                "message": f"Submitted amorphous packing job. Poll status via get_job_status('{job.job_id}')."
            },
            interpretation=f"Amorphous cell packing job submitted. Job ID: {job.job_id}."
        )

    res = pack_amorphous_cell_engine(workspace_id, molecule_ids, counts, density_g_cm3, box_size_angstrom)
    return _finalize_pack_amorphous_cell(res)


@mcp_tool_decorator
def run_lammps_simulation(
    packed_molecule_id: str,
    steps: int = 1000,
    timestep_fs: float = 1.0,
    temperature_k: float = 300.0,
    pressure_atm: float = 1.0,
    ensemble: str = "npt",
    run_async: bool = True
) -> dict:
    """
    Run classical MD simulation in LAMMPS (or ASE fallback).
    
    Parameters:
    - packed_molecule_id: Workspace ID of the packed amorphous cell.
    - steps: Total MD integration steps (default 1000).
    - timestep_fs: Integration timestep in femtoseconds (default 1.0).
    - temperature_k: Target temperature in Kelvin (default 300.0).
    - pressure_atm: Target pressure in atmospheres (only for NPT, default 1.0).
    - ensemble: Thermodynamic ensemble ('npt', 'nvt', or 'nve').
    - run_async: If true, runs MD in background (default is True).
    """
    workspace_id = get_workspace_id()
    est_sec = 20
    
    from ypotheto_compchem_mcp.jobs import job_manager
    
    if run_async:
        job = job_manager.submit_job(
            workspace_id,
            run_lammps_simulation_engine,
            est_sec,
            workspace_id,
            packed_molecule_id,
            steps,
            timestep_fs,
            temperature_k,
            pressure_atm,
            ensemble
        )
        return make_success_response(
            results={
                "job_id": job.job_id,
                "status": job.status,
                "estimated_time_seconds": job.estimated_time_seconds,
                "message": f"Submitted MD simulation. Poll status via get_job_status('{job.job_id}')."
            },
            interpretation=f"MD simulation job submitted. Job ID: {job.job_id}."
        )
        
    res = run_lammps_simulation_engine(workspace_id, packed_molecule_id, steps, timestep_fs, temperature_k, pressure_atm, ensemble)
    return make_success_response(
        results=res["results"],
        interpretation=res["interpretation"],
        artifacts=res.get("artifacts", []),
        warnings=res.get("warnings", []),
        meta={"packed_molecule_id": packed_molecule_id}
    )


@mcp_tool_decorator
def analyze_md_trajectory(
    trajectory_file_id: str
) -> dict:
    """
    Analyze MD trajectory XYZ file to compute Radius of Gyration, RDF, and MSD.
    
    Parameters:
    - trajectory_file_id: Trajectory artifact or file name in workspace.
    """
    workspace_id = get_workspace_id()
    
    import urllib.parse
    filename = trajectory_file_id
    artifact_id = None
    if "://" in filename or "/artifacts/" in filename:
        parsed_path = urllib.parse.urlparse(filename).path
        parts = [p for p in parsed_path.split("/") if p]
        if len(parts) >= 4:
            artifact_id = parts[-2]
            filename = parts[-1]
            
    if artifact_id:
        file_path = workspace_manager.get_artifacts_dir(workspace_id) / artifact_id / filename
    else:
        file_path = workspace_manager.get_workspace_dir(workspace_id) / filename
        if not file_path.exists():
            file_path = workspace_manager.get_artifacts_dir(workspace_id) / filename
            
    if not file_path.exists():
        raise FileNotFoundError(f"Trajectory file {trajectory_file_id} not found.")
            
    traj_xyz = file_path.read_text(encoding="utf-8")
    
    res = analyze_md_trajectory_engine(workspace_id, traj_xyz)
    return make_success_response(
        results=res["results"],
        interpretation=res["interpretation"]
    )


def register_polymer_tools(mcp: FastMCP) -> None:
    mcp.tool()(register_monomer)
    mcp.tool()(build_polymer_chain)
    mcp.tool()(pack_amorphous_cell)
    mcp.tool()(run_lammps_simulation)
    mcp.tool()(analyze_md_trajectory)
