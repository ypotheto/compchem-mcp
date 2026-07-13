
from ypotheto_compchem_mcp.artifacts import register_artifact
from ypotheto_compchem_mcp.chemistry.builder_engine import (
    build_molecule_from_smiles_engine,
    get_molecule_path,
)
from ypotheto_compchem_mcp.envelope import WarningInfo, make_success_response, mcp_tool_decorator
from ypotheto_compchem_mcp.molecules import molecule_store
from ypotheto_compchem_mcp.server import mcp

_MAX_INLINE_CONTENT_BYTES = 50 * 1024

@mcp.tool()
@mcp_tool_decorator
def build_molecule_from_smiles(smiles: str, name: str | None = None) -> dict:
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
    Retrieve coordinate contents (SDF, XYZ, or PDB) of a stored molecule.
    Use when needing to view coordinate tables or output structures.

    Parameters:
    - molecule_id: The stored molecule handle (e.g., mol_a1b2c3d4)
    - format: Coordinate format, one of 'xyz', 'sdf', or 'pdb' (default is 'xyz')
    """
    clean_fmt = format.lower().strip()
    if clean_fmt not in ("xyz", "sdf", "pdb"):
        raise ValueError("Invalid format. Must be one of 'xyz', 'sdf', or 'pdb'.")

    from ypotheto_compchem_mcp.workspace import get_workspace_id
    workspace_id = get_workspace_id()
    if clean_fmt == "pdb":
        # PDB is never persisted alongside the XYZ/SDF a molecule is saved as
        # (see save_molecule_coords) - generate it on the fly from the stored
        # SDF's RDKit Mol instead of reading a file that would never exist.
        from rdkit import Chem

        from ypotheto_compchem_mcp.chemistry.builder_engine import load_molecule_from_workspace
        mol = load_molecule_from_workspace(workspace_id, molecule_id)
        content = Chem.MolToPDBBlock(mol)
    else:
        path = get_molecule_path(workspace_id, molecule_id, clean_fmt)
        content = path.read_text(encoding="utf-8")
    
    # Register download artifact
    content_bytes = content.encode("utf-8")
    artifact = register_artifact(f"{molecule_id}.{clean_fmt}", content_bytes, "structure", f"3D Structure ({clean_fmt.upper()})")

    warnings = []
    if len(content_bytes) > _MAX_INLINE_CONTENT_BYTES:
        # Omit inline content rather than let a large structure (many atoms, or
        # SDF conformer blocks) blow up the response size - the artifact still
        # carries the full content.
        results = {
            "molecule_id": molecule_id,
            "format": clean_fmt,
            "content": None
        }
        warnings.append(WarningInfo(
            type="CONTENT_TOO_LARGE",
            message=(
                f"Structure content ({len(content_bytes)} bytes) exceeds the "
                f"{_MAX_INLINE_CONTENT_BYTES}-byte inline limit; omitted from the response. "
                f"Download the full content from the artifact URL instead."
            )
        ))
        interpretation = (
            f"Retrieved 3D coordinates for {molecule_id} in {clean_fmt.upper()} format. "
            f"Content omitted inline (exceeds size limit) - see the attached artifact."
        )
    else:
        results = {
            "molecule_id": molecule_id,
            "format": clean_fmt,
            "content": content
        }
        interpretation = f"Retrieved 3D coordinates for {molecule_id} in {clean_fmt.upper()} format."

    return make_success_response(
        results=results,
        interpretation=interpretation,
        warnings=warnings,
        artifacts=[artifact],
        meta={"molecule_id": molecule_id}
    )

@mcp.tool()
@mcp_tool_decorator
def list_molecules() -> dict:
    """
    List all molecules stored in the current workspace.
    Use to see what structures are already available before building duplicates.
    """
    from ypotheto_compchem_mcp.workspace import get_workspace_id
    workspace_id = get_workspace_id()
    molecules = molecule_store.list(workspace_id)

    interpretation = (
        f"Found {len(molecules)} molecule(s) in this workspace."
        if molecules
        else "No molecules stored in this workspace yet."
    )
    return make_success_response(
        results={"molecules": molecules, "count": len(molecules)},
        interpretation=interpretation
    )

@mcp.tool()
@mcp_tool_decorator
def describe_molecule(molecule_id: str) -> dict:
    """
    Retrieve stored metadata (name, formula, SMILES, atom count, method) for a molecule
    without loading its full 3D coordinates.

    Parameters:
    - molecule_id: The stored molecule handle (e.g. mol_a1b2c3d4)
    """
    from ypotheto_compchem_mcp.workspace import get_workspace_id
    workspace_id = get_workspace_id()
    info = molecule_store.describe(workspace_id, molecule_id)

    interpretation = (
        f"{molecule_id}: '{info.get('name', '')}' ({info.get('formula', '')}), "
        f"{info.get('num_atoms', '?')} atoms, built via {info.get('method', 'unknown')}."
    )
    return make_success_response(
        results=info,
        interpretation=interpretation,
        meta={"molecule_id": molecule_id}
    )

@mcp.tool()
@mcp_tool_decorator
def delete_molecule(molecule_id: str) -> dict:
    """
    Permanently delete a stored molecule's coordinates and metadata from this workspace.
    This cannot be undone - any downstream artifact/job still referencing this
    molecule_id will start failing with a not-found error.

    Parameters:
    - molecule_id: The stored molecule handle to delete (e.g. mol_a1b2c3d4)
    """
    from ypotheto_compchem_mcp.workspace import get_workspace_id
    workspace_id = get_workspace_id()
    molecule_store.delete(workspace_id, molecule_id)

    return make_success_response(
        results={"molecule_id": molecule_id, "deleted": True},
        interpretation=f"Deleted molecule {molecule_id} from this workspace."
    )
