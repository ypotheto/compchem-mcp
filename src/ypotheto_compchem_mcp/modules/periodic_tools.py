
from mcp.server.fastmcp import FastMCP

from ypotheto_compchem_mcp.artifacts import register_artifact
from ypotheto_compchem_mcp.chemistry.periodic_engine import (
    add_adsorbate_to_surface_engine,
    analyze_crystal_symmetry_engine,
    build_surface_slab_engine,
    generate_supercell_engine,
    import_periodic_structure_engine,
    run_periodic_dft_engine,
)
from ypotheto_compchem_mcp.envelope import make_success_response, mcp_tool_decorator
from ypotheto_compchem_mcp.workspace import get_workspace_id


@mcp_tool_decorator
def import_periodic_structure(
    cif_content: str,
    name: str | None = None
) -> dict:
    """
    Import a periodic crystal structure from a CIF file.
    Calculates lattice parameters and space group symmetry.
    
    Parameters:
    - cif_content: The text content of the CIF (Crystallographic Information File).
    - name: Optional label for the crystal structure.
    """
    workspace_id = get_workspace_id()
    
    res = import_periodic_structure_engine(workspace_id, cif_content, name)
    molecule_id = res["molecule_id"]
    
    # Register CIF content as structure artifact
    cif_art = register_artifact(
        f"{molecule_id}.cif",
        res["cif_block"].encode("utf-8"),
        "structure",
        f"CIF coordinates of {res['name']}"
    )
    
    interpretation = (
        f"Successfully imported periodic structure {molecule_id} ({res['name']}). "
        f"Formula: {res['formula']}, Atoms in unit cell: {res['num_atoms']}. "
        f"Lattice: a={res['lattice_parameters']['a']:.3f} Å, "
        f"b={res['lattice_parameters']['b']:.3f} Å, c={res['lattice_parameters']['c']:.3f} Å. "
        f"Symmetry: Space Group {res['space_group']['symbol']} (Number {res['space_group']['number']})."
    )
    
    # Remove large file blocks from JSON-RPC results to keep payload small
    res_clean = {k: v for k, v in res.items() if k not in ("cif_block", "xyz_block")}
    
    return make_success_response(
        results=res_clean,
        interpretation=interpretation,
        artifacts=[cif_art],
        meta={
            "molecule_id": molecule_id,
            "type": "periodic_structure"
        }
    )

@mcp_tool_decorator
def analyze_crystal_symmetry(
    molecule_id: str
) -> dict:
    """
    Perform deep crystallographic symmetry and space group analysis for a stored structure.
    Returns Space Group symbol, Hall symbol, rotations, translations, and Wyckoff sites.
    
    Parameters:
    - molecule_id: The stored periodic structure handle (e.g., crystal_a1b2c3d4)
    """
    workspace_id = get_workspace_id()
    
    res = analyze_crystal_symmetry_engine(workspace_id, molecule_id)
    
    interpretation = (
        f"Symmetry analysis completed for {molecule_id}. "
        f"Crystallographic space group: {res['results']['international']} "
        f"(Number {res['results']['number']}). Hall symbol: {res['results']['hall']}."
    )
    
    return make_success_response(
        results=res["results"],
        interpretation=interpretation,
        meta={
            "molecule_id": molecule_id
        }
    )

@mcp_tool_decorator
def generate_supercell(
    molecule_id: str,
    sc_matrix: list[int],
    name: str | None = None
) -> dict:
    """
    Expand a unit cell periodic structure into a supercell.
    
    Parameters:
    - molecule_id: The unit cell periodic structure handle.
    - sc_matrix: A scaling list of integers. Specify either:
        * 3 integers [nx, ny, nz] for diagonal scaling along cell vectors.
        * 9 integers for a full 3x3 transformation matrix.
    - name: Optional label for the resulting supercell structure.
    """
    workspace_id = get_workspace_id()
    
    res = generate_supercell_engine(workspace_id, molecule_id, sc_matrix, name)
    super_id = res["results"]["supercell_molecule_id"]
    
    # Save supercell CIF as structure artifact
    cif_art = register_artifact(
        f"{super_id}.cif",
        res["cif_block"].encode("utf-8"),
        "structure",
        f"Supercell CIF coordinates of {super_id}"
    )
    
    interpretation = (
        f"Supercell generated successfully: {super_id}. "
        f"Atoms in supercell: {res['results']['num_atoms']}. "
        f"Symmetry: Space Group {res['results']['space_group']['symbol']} (Number {res['results']['space_group']['number']})."
    )
    
    return make_success_response(
        results=res["results"],
        interpretation=interpretation,
        artifacts=[cif_art],
        meta={
            "parent_molecule_id": molecule_id,
            "supercell_molecule_id": super_id
        }
    )


@mcp_tool_decorator
def build_surface_slab(
    bulk_molecule_id: str,
    miller_indices: list[int],
    layers: int,
    vacuum_size: float = 10.0
) -> dict:
    """
    Generate a surface slab from bulk periodic crystal structure.
    
    Parameters:
    - bulk_molecule_id: Workspace bulk crystal structure ID (e.g. crystal_a1b2).
    - miller_indices: Miller indices [h, k, l] representing target surface plane (e.g. [1, 1, 1]).
    - layers: Number of layers of crystal in the slab (integer).
    - vacuum_size: Height of vacuum region above slab in Angstroms (default 10.0).
    """
    workspace_id = get_workspace_id()
    
    res = build_surface_slab_engine(workspace_id, bulk_molecule_id, miller_indices, layers, vacuum_size)
    slab_id = res["results"]["slab_molecule_id"]
    
    cif_art = register_artifact(
        f"{slab_id}.cif",
        res["cif_block"].encode("utf-8"),
        "structure",
        f"Slab CIF coordinates of {slab_id}"
    )
    
    interpretation = (
        f"Surface slab generated successfully: {slab_id}.\n"
        f"Miller Indices = {tuple(miller_indices)}, Layers = {layers}, Vacuum = {vacuum_size} A.\n"
        f"Chemical Formula = {res['results']['formula']}, Atoms = {res['results']['num_atoms']}."
    )
    
    return make_success_response(
        results=res["results"],
        interpretation=interpretation,
        artifacts=[cif_art],
        meta={
            "parent_bulk_molecule_id": bulk_molecule_id,
            "slab_molecule_id": slab_id
        }
    )


@mcp_tool_decorator
def add_adsorbate_to_surface(
    slab_molecule_id: str,
    adsorbate_molecule_id: str,
    height: float = 1.5,
    position_type: str = "ontop"
) -> dict:
    """
    Place a non-periodic adsorbate molecule onto a periodic surface slab.
    
    Parameters:
    - slab_molecule_id: Target periodic surface slab structure ID.
    - adsorbate_molecule_id: Adsorbate molecule ID (e.g. mol_a1b2).
    - height: Distance above surface plane in Angstroms (default 1.5).
    - position_type: Surface site type, e.g. 'ontop', 'bridge', 'hollow', or coordinates like '0.5,0.5'.
    """
    workspace_id = get_workspace_id()
    
    res = add_adsorbate_to_surface_engine(workspace_id, slab_molecule_id, adsorbate_molecule_id, height, position_type)
    combined_id = res["results"]["combined_molecule_id"]
    
    cif_art = register_artifact(
        f"{combined_id}.cif",
        res["cif_block"].encode("utf-8"),
        "structure",
        f"Adsorbed complex CIF of {combined_id}"
    )
    
    interpretation = (
        f"Adsorbate placed successfully onto surface slab: {combined_id}.\n"
        f"Combined Formula = {res['results']['formula']}, Total Atoms = {res['results']['num_atoms']}."
    )
    
    return make_success_response(
        results=res["results"],
        interpretation=interpretation,
        artifacts=[cif_art],
        meta={
            "slab_molecule_id": slab_molecule_id,
            "combined_molecule_id": combined_id
        }
    )


@mcp_tool_decorator
def run_periodic_dft(
    molecule_id: str,
    kpts: list[int] | None = None,
    method: str = "xTB",
    run_async: bool = True
) -> dict:
    """
    Perform periodic DFT or semi-empirical GFN-xTB PBC energy calculations.
    
    Parameters:
    - molecule_id: Periodic crystal or slab structure ID.
    - kpts: K-point grid dimensions, e.g. [1, 1, 1] or [2, 2, 2].
    - method: Quantum chemical solver ('xTB' or DFT).
    - run_async: If true, runs periodic energy evaluation in background (default is True).
    """
    workspace_id = get_workspace_id()
    est_sec = 25
    
    from ypotheto_compchem_mcp.jobs import job_manager
    
    if run_async:
        job = job_manager.submit_job(
            workspace_id,
            run_periodic_dft_engine,
            est_sec,
            workspace_id,
            molecule_id,
            kpts,
            method
        )
        return make_success_response(
            results={
                "job_id": job.job_id,
                "status": job.status,
                "estimated_time_seconds": job.estimated_time_seconds,
                "message": f"Submitted periodic boundary calculation. Poll status via get_job_status('{job.job_id}')."
            },
            interpretation=f"Periodic boundary calculation job submitted. Job ID: {job.job_id}."
        )
        
    res = run_periodic_dft_engine(workspace_id, molecule_id, kpts, method)
    return make_success_response(
        results=res["results"],
        interpretation=res["interpretation"]
    )


def register_periodic_tools(mcp: FastMCP) -> None:
    mcp.tool()(import_periodic_structure)
    mcp.tool()(analyze_crystal_symmetry)
    mcp.tool()(generate_supercell)
    mcp.tool()(build_surface_slab)
    mcp.tool()(add_adsorbate_to_surface)
    mcp.tool()(run_periodic_dft)
