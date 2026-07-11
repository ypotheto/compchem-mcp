from typing import Optional
from ypotheto_compchem_mcp.server import mcp
from ypotheto_compchem_mcp.envelope import mcp_tool_decorator, make_success_response
from ypotheto_compchem_mcp.artifacts import register_artifact
from ypotheto_compchem_mcp.chemistry.builder_engine import build_molecule_from_smiles_engine, get_molecule_path

@mcp.tool()
@mcp_tool_decorator
def build_molecule_from_smiles(smiles: str, name: Optional[str] = None) -> dict:
    """
    Generate optimized 3D coordinates from a SMILES representation.
    Use when building or initializing a new molecule structure.
    
    Parameters:
    - smiles: The SMILES string (e.g. CCO for ethanol, C(=O)O for formic acid)
    - name: Optional label for the molecule (e.g. 'Ethanol')
    """
    res = build_molecule_from_smiles_engine(smiles, name)
    
    # Save SDF, XYZ, and SVG as artifacts
    xyz_bytes = res["xyz_block"].encode("utf-8")
    sdf_bytes = res["sdf_block"].encode("utf-8")
    
    xyz_art = register_artifact(f"{res['molecule_id']}.xyz", xyz_bytes, "structure", "3D Coordinates (XYZ)")
    sdf_art = register_artifact(f"{res['molecule_id']}.sdf", sdf_bytes, "structure", "3D Coordinates (SDF)")
    svg_art = register_artifact(f"{res['molecule_id']}.svg", res["svg_data"], "plot", "2D Layout Depiction (SVG)")
    
    results = {
        "molecule_id": res["molecule_id"],
        "name": res["name"],
        "formula": res["formula"],
        "num_atoms": res["num_atoms"],
        "method": res["method"]
    }
    
    interpretation = (
        f"Built molecule '{res['name']}' ({res['formula']}) with {res['num_atoms']} atoms from SMILES. "
        f"A 3D coordinate model was generated and pre-optimized with the '{res['method']}' force field. "
        f"Assigned molecule handle: {res['molecule_id']}."
    )
    
    return make_success_response(
        results=results,
        interpretation=interpretation,
        artifacts=[xyz_art, sdf_art, svg_art],
        meta={"molecule_id": res["molecule_id"], "smiles": smiles}
    )

@mcp.tool()
@mcp_tool_decorator
def get_3d_coordinates(molecule_id: str, format: str = "xyz") -> dict:
    """
    Retrieve coordinate contents (SDF or XYZ) of a stored molecule.
    Use when needing to view coordinate tables or output structures.
    
    Parameters:
    - molecule_id: The stored molecule handle (e.g., mol_a1b2c3d4)
    - format: Coordinate format, either 'xyz' or 'sdf' (default is 'xyz')
    """
    clean_fmt = format.lower().strip()
    if clean_fmt not in ("xyz", "sdf"):
        raise ValueError("Invalid format. Must be either 'xyz' or 'sdf'.")
        
    from ypotheto_compchem_mcp.workspace import get_workspace_id
    workspace_id = get_workspace_id()
    path = get_molecule_path(workspace_id, molecule_id, clean_fmt)
    content = path.read_text(encoding="utf-8")
    
    # Register download artifact
    content_bytes = content.encode("utf-8")
    artifact = register_artifact(f"{molecule_id}.{clean_fmt}", content_bytes, "structure", f"3D Structure ({clean_fmt.upper()})")
    
    results = {
        "molecule_id": molecule_id,
        "format": clean_fmt,
        "content": content
    }
    
    interpretation = f"Retrieved 3D coordinates for {molecule_id} in {clean_fmt.upper()} format."
    
    return make_success_response(
        results=results,
        interpretation=interpretation,
        artifacts=[artifact],
        meta={"molecule_id": molecule_id}
    )
