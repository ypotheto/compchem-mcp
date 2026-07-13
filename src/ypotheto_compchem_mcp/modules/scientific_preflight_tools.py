from mcp.server.fastmcp import FastMCP

from ypotheto_compchem_mcp.chemistry.builder_engine import load_molecule_from_workspace
from ypotheto_compchem_mcp.chemistry.preflight import (
    estimate_computational_resources,
    validate_basis_set_coverage,
    validate_charge_spin_multiplicity,
)
from ypotheto_compchem_mcp.envelope import (
    make_error_response,
    make_success_response,
    mcp_tool_decorator,
)
from ypotheto_compchem_mcp.workspace import get_workspace_id


@mcp_tool_decorator
def run_scientific_preflight(
    molecule_id: str,
    method: str = "DFT",
    basis: str = "6-31g*",
    charge: int = 0,
    spin: int = 1,
    task: str = "single_point"
) -> dict:
    """
    Validate molecule consistency and estimate calculation resources before submission.
    Use to verify charge/spin multiplicity consistency, basis set compatibility, and compute costs.
    
    Parameters:
    - molecule_id: The stored molecule handle (e.g. mol_a1b2c3d4)
    - method: The target method (HF, DFT, or semi-empirical like AM1/PM6/xTB)
    - basis: The basis set (e.g. sto-3g, 6-31g*, def2-svp)
    - charge: Total molecular charge (default is 0)
    - spin: Spin multiplicity (2S + 1) (default is 1, singlet)
    - task: Target task, either 'single_point', 'geometry_optimization', or 'vibrations'
    """
    workspace_id = get_workspace_id()
    
    try:
        mol = load_molecule_from_workspace(workspace_id, molecule_id)
    except Exception as e:
        return make_error_response("MOLECULE_NOT_FOUND", f"Could not load molecule {molecule_id}: {str(e)}")
        
    # Run preflight validations
    ok, err_msg = validate_charge_spin_multiplicity(mol, charge, spin)
    if not ok:
        return make_error_response("INVALID_CHARGE_SPIN", err_msg)
        
    ok, err_msg = validate_basis_set_coverage(mol, basis)
    if not ok:
        return make_error_response("UNSUPPORTED_BASIS_SET", err_msg)
        
    # Estimate resources
    estimates = estimate_computational_resources(mol, method, basis, task)
    
    results = {
        "molecule_id": molecule_id,
        "method": method,
        "basis": basis,
        "charge": charge,
        "spin": spin,
        "task": task,
        "preflight_passed": True,
        "validation": {
            "total_electrons": sum(atom.GetAtomicNum() for atom in mol.GetAtoms()) - charge,
            "charge_spin_valid": True,
            "basis_set_coverage_valid": True
        },
        "estimates": estimates
    }
    
    interpretation = (
        f"Scientific preflight checks PASSED for molecule {molecule_id}.\n"
        f"The requested calculation ({method}/{basis}, charge={charge}, multiplicity={spin}) is valid.\n"
        f"Estimated Wall Time: {estimates['estimated_wall_time_seconds']} seconds.\n"
        f"Estimated RAM: {estimates['estimated_ram_mb']} MB.\n"
        f"Estimated Billing Cost: {estimates['compute_credits_cost']} credits.\n"
        f"Recommended run mode: {estimates['recommended_run_mode'].upper()}."
    )
    
    return make_success_response(results, interpretation)


def register_scientific_preflight_tools(mcp: FastMCP) -> None:
    mcp.tool()(run_scientific_preflight)
