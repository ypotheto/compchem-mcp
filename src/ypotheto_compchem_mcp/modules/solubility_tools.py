from ypotheto_compchem_mcp.chemistry.solubility_engine import (
    calculate_hsp_distance_engine,
    calculate_hsp_engine,
)
from ypotheto_compchem_mcp.envelope import make_success_response, mcp_tool_decorator
from ypotheto_compchem_mcp.server import mcp
from ypotheto_compchem_mcp.workspace import get_workspace_id


@mcp.tool()
@mcp_tool_decorator
def calculate_hsp(
    molecule_id: str
) -> dict:
    """
    Calculate the Hansen Solubility Parameters (HSP) and Cohesive Energy Density (CED)
    for a stored molecule using the Hoftyzer-Van Krevelen (HVK) group contribution method.
    Useful for solvent screening, miscibility estimation, and formulation compatibility.
    
    Parameters:
    - molecule_id: The stored molecule handle.
    """
    workspace_id = get_workspace_id()
    res = calculate_hsp_engine(workspace_id, molecule_id)
    
    hsp = res["hansen_parameters"]
    ced = res["cohesive_energy"]
    
    interpretation = (
        f"Calculated Hansen Solubility Parameters for {molecule_id}: "
        f"Dispersion (delta_d) = {hsp['dispersion_delta_d']:.2f}, "
        f"Polar (delta_p) = {hsp['polar_delta_p']:.2f}, "
        f"H-bonding (delta_h) = {hsp['hydrogen_bonding_delta_h']:.2f} {hsp['unit']}. "
        f"Total Solubility = {hsp['total_solubility_delta']:.2f} {hsp['unit']}. "
        f"Cohesive Energy Density = {ced['cohesive_energy_density_j_cm3']:.2f} J/cm^3."
    )
    
    return make_success_response(
        results=res,
        interpretation=interpretation,
        meta={"molecule_id": molecule_id}
    )

@mcp.tool()
@mcp_tool_decorator
def calculate_hsp_distance(
    molecule_id_1: str,
    molecule_id_2: str
) -> dict:
    """
    Calculate the Hansen Solubility Parameter (HSP) distance (Ra) between two stored molecules.
    A smaller distance Ra suggests higher miscibility, compatibility, or solubility.
    
    Parameters:
    - molecule_id_1: The first stored molecule handle (e.g. the polymer or solute).
    - molecule_id_2: The second stored molecule handle (e.g. the solvent).
    """
    workspace_id = get_workspace_id()
    res = calculate_hsp_distance_engine(workspace_id, molecule_id_1, molecule_id_2)
    
    interpretation = (
        f"Hansen solubility distance Ra between {molecule_id_1} and {molecule_id_2} is "
        f"{res['results']['hansen_distance_ra']:.2f} MPa^0.5. "
        f"Compatibility: {res['results']['miscibility_estimate']}."
    )
    
    return make_success_response(
        results=res["results"],
        interpretation=interpretation,
        meta={
            "molecule_id_1": molecule_id_1,
            "molecule_id_2": molecule_id_2
        }
    )
