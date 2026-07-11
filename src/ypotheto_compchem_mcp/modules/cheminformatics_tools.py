import json
from ypotheto_compchem_mcp.server import mcp
from ypotheto_compchem_mcp.envelope import mcp_tool_decorator, make_success_response
from ypotheto_compchem_mcp.artifacts import register_artifact
from ypotheto_compchem_mcp.chemistry.descriptors import calculate_descriptors_engine

@mcp.tool()
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
