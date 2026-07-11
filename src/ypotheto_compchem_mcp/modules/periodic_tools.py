from typing import List, Optional
from ypotheto_compchem_mcp.server import mcp
from ypotheto_compchem_mcp.envelope import mcp_tool_decorator, make_success_response
from ypotheto_compchem_mcp.artifacts import register_artifact
from ypotheto_compchem_mcp.workspace import get_workspace_id
from ypotheto_compchem_mcp.chemistry.periodic_engine import (
    import_periodic_structure_engine,
    analyze_crystal_symmetry_engine,
    generate_supercell_engine
)

@mcp.tool()
@mcp_tool_decorator
def import_periodic_structure(
    cif_content: str,
    name: Optional[str] = None
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
        f"Lattice: a={res['lattice_parameters']['a']:.3f} Å, b={res['lattice_parameters']['b']:.3f} Å, c={res['lattice_parameters']['c']:.3f} Å. "
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

@mcp.tool()
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

@mcp.tool()
@mcp_tool_decorator
def generate_supercell(
    molecule_id: str,
    sc_matrix: List[int],
    name: Optional[str] = None
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
