import json

from mcp.server.fastmcp import FastMCP

from ypotheto_compchem_mcp.artifacts import register_artifact
from ypotheto_compchem_mcp.chemistry.descriptors import calculate_descriptors_engine
from ypotheto_compchem_mcp.envelope import make_success_response, mcp_tool_decorator


@mcp_tool_decorator
def calculate_descriptors(molecule_id: str) -> dict:
    """
    Calculate molecular properties (descriptors) and Lipinski's Rule of Five compliance.
    Use when checking molecular properties, polar surface area (TPSA), lipophilicity (LogP), or drug-likeness.
    
    Parameters:
    - molecule_id: The stored molecule handle (e.g. mol_a1b2c3d4)
    """
    from ypotheto_compchem_mcp.workspace import get_workspace_id
    workspace_id = get_workspace_id()
    
    res = calculate_descriptors_engine(workspace_id, molecule_id)
    
    # Save descriptors as a JSON report artifact
    res_bytes = json.dumps(res, indent=2).encode("utf-8")
    report_art = register_artifact(f"{molecule_id}_descriptors.json", res_bytes, "report", "Descriptor Profile Report")
    
    desc = res["descriptors"]
    filt = res["lipinski_filter"]
    
    lipinski_status = "PASSED" if filt["passes"] else f"FAILED ({filt['violations_count']} violations)"
    
    interpretation = (
        f"Calculated descriptors for {molecule_id}: "
        f"MW = {desc['molecular_weight']:.2f} g/mol, "
        f"LogP = {desc['logp']:.2f}, "
        f"TPSA = {desc['tpsa']:.2f} Å², "
        f"Rotatable Bonds = {desc['rotatable_bonds']}. "
        f"Lipinski Filter: {lipinski_status}."
    )
    
    return make_success_response(
        results=res,
        interpretation=interpretation,
        artifacts=[report_art],
        meta={"molecule_id": molecule_id}
    )

@mcp_tool_decorator
def standardize_molecule(
    smiles_or_sdf: str,
    strip_salts: bool = True,
    neutralize: bool = True,
    canonicalize_tautomer: bool = True,
    name: str | None = None
) -> dict:
    """
    Standardize a molecule: parses structure, strips salts, neutralizes formal charge,
    canonicalizes tautomers, and sanitizes/minimized the output.
    
    Parameters:
    - smiles_or_sdf: SMILES string or SDF block of the molecule to standardize.
    - strip_salts: If True, strips counter-ions and salt fragments (default True).
    - neutralize: If True, neutralizes formal charges (default True).
    - canonicalize_tautomer: If True, standardizes to the canonical tautomer (default True).
    - name: Optional name for the standardized molecule.
    """
    from ypotheto_compchem_mcp.chemistry.standardizer import standardize_molecule_engine
    from ypotheto_compchem_mcp.workspace import get_workspace_id
    
    workspace_id = get_workspace_id()
    res = standardize_molecule_engine(
        workspace_id, smiles_or_sdf, strip_salts, neutralize, canonicalize_tautomer, name
    )
    
    molecule_id = res["molecule_id"]
    
    # Save standard 2D layout SVG as artifact
    svg_art = register_artifact(
        f"{molecule_id}.svg",
        res["svg_data"].encode("utf-8"),
        "depiction",
        f"2D Layout of {res['name']}"
    )
    
    interpretation = (
        f"Molecule standardized successfully: {molecule_id}. "
        f"Formula: {res['formula']}. Standardized SMILES: {res['standardized_smiles']}. "
        f"Steps executed: {', '.join(res['steps_taken']) if res['steps_taken'] else 'none'}."
    )
    
    res_clean = {k: v for k, v in res.items() if k != "svg_data"}
    
    return make_success_response(
        results=res_clean,
        interpretation=interpretation,
        artifacts=[svg_art],
        meta={
            "molecule_id": molecule_id,
            "type": "standardized_molecule"
        }
    )

@mcp_tool_decorator
def enumerate_tautomers(
    molecule_id: str
) -> dict:
    """
    Enumerate all tautomeric forms for a stored molecule.
    
    Parameters:
    - molecule_id: The stored molecule handle.
    """
    from ypotheto_compchem_mcp.chemistry.standardizer import enumerate_tautomers_engine
    from ypotheto_compchem_mcp.workspace import get_workspace_id
    
    workspace_id = get_workspace_id()
    res = enumerate_tautomers_engine(workspace_id, molecule_id)
    
    interpretation = (
        f"Tautomer enumeration completed for {molecule_id}. "
        f"Found {res['tautomers_count']} possible tautomers."
    )
    
    return make_success_response(
        results=res,
        interpretation=interpretation,
        meta={"molecule_id": molecule_id}
    )

@mcp_tool_decorator
def search_conformers(
    molecule_id: str,
    num_conformers: int = 50,
    rmsd_threshold: float = 0.5
) -> dict:
    """
    Generate multiple conformers for a molecule, relax them, prune duplicates,
    and rank them by forcefield energy and Boltzmann populations.
    
    Parameters:
    - molecule_id: The stored molecule handle.
    - num_conformers: Maximum number of conformers to embed initially (default 50).
    - rmsd_threshold: RMSD threshold in Angstroms for pruning duplicates (default 0.5 Å).
    """
    from ypotheto_compchem_mcp.chemistry.conformer_engine import search_conformers_engine
    from ypotheto_compchem_mcp.workspace import get_workspace_id
    
    workspace_id = get_workspace_id()
    res = search_conformers_engine(workspace_id, molecule_id, num_conformers, rmsd_threshold)
    
    interpretation = (
        f"Conformer ensemble search completed. "
        f"Found {res['conformers_found']} unique conformers (RMSD threshold {rmsd_threshold} Å). "
        f"Lowest energy conformer: {res['lowest_energy_kcal_mol']:.4f} kcal/mol."
    )
    
    return make_success_response(
        results=res,
        interpretation=interpretation,
        meta={"molecule_id": molecule_id}
    )

@mcp_tool_decorator
def save_conformer_as_molecule(
    parent_molecule_id: str,
    rdkit_conformer_id: int,
    name: str | None = None
) -> dict:
    """
    Extract a single conformer from a search result and save it as a new molecule in the workspace.
    
    Parameters:
    - parent_molecule_id: The molecule handle that underwent the conformer search.
    - rdkit_conformer_id: The internal RDKit conformer ID to extract.
    - name: Optional label for the resulting molecule.
    """
    from ypotheto_compchem_mcp.chemistry.conformer_engine import save_conformer_as_molecule_engine
    from ypotheto_compchem_mcp.workspace import get_workspace_id
    
    workspace_id = get_workspace_id()
    res = save_conformer_as_molecule_engine(workspace_id, parent_molecule_id, rdkit_conformer_id, name)
    
    molecule_id = res["molecule_id"]
    
    svg_art = register_artifact(
        f"{molecule_id}.svg",
        res["svg_data"].encode("utf-8"),
        "depiction",
        f"2D Layout of conformer molecule {molecule_id}"
    )
    
    interpretation = (
        f"Extracted conformer {rdkit_conformer_id} from {parent_molecule_id} and "
        f"saved as new molecule: {molecule_id} ({res['name']})."
    )
    
    res_clean = {k: v for k, v in res.items() if k != "svg_data"}
    
    return make_success_response(
        results=res_clean,
        interpretation=interpretation,
        artifacts=[svg_art],
        meta={
            "molecule_id": molecule_id,
            "parent_id": parent_molecule_id,
            "parent_conformer_id": rdkit_conformer_id
        }
    )


def register_cheminformatics_tools(mcp: FastMCP) -> None:
    mcp.tool()(calculate_descriptors)
    mcp.tool()(standardize_molecule)
    mcp.tool()(enumerate_tautomers)
    mcp.tool()(search_conformers)
    mcp.tool()(save_conformer_as_molecule)
