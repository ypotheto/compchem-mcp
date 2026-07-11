from typing import Optional
from ypotheto_compchem_mcp.server import mcp
from ypotheto_compchem_mcp.envelope import mcp_tool_decorator, make_success_response
from ypotheto_compchem_mcp.artifacts import register_artifact
from ypotheto_compchem_mcp.workspace import get_workspace_id
from ypotheto_compchem_mcp.chemistry.polymer_engine import (
    register_monomer_engine,
    build_polymer_chain_engine
)

@mcp.tool()
@mcp_tool_decorator
def register_monomer(
    smiles: str,
    name: str,
    head_idx: Optional[int] = None,
    tail_idx: Optional[int] = None
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

@mcp.tool()
@mcp_tool_decorator
def build_polymer_chain(
    monomer_id: str,
    dp: int,
    tacticity: str = "isotactic",
    name: Optional[str] = None
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
