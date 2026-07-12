from typing import Optional
from ypotheto_compchem_mcp.server import mcp
from ypotheto_compchem_mcp.envelope import mcp_tool_decorator, make_success_response, make_error_response
from ypotheto_compchem_mcp.errors import BackendUnavailableError
from ypotheto_compchem_mcp.workspace import get_workspace_id
from ypotheto_compchem_mcp.jobs import job_manager
from ypotheto_compchem_mcp.chemistry.ensemble_pipeline import run_ensemble_thermochemistry_engine
from ypotheto_compchem_mcp.chemistry.xtb_engine import XTB_AVAILABLE, CREST_AVAILABLE

@mcp.tool()
@mcp_tool_decorator
def run_ensemble_thermochemistry(
    molecule_id: str,
    method: str = "GFN2-xTB",
    solvent: Optional[str] = None,
    energy_window_kcal: float = 6.0,
    max_conformers_to_optimize: int = 5,
    energy_threshold_kcal: float = 3.0,
    charge: int = 0,
    spin: int = 1,
    run_async: bool = True
) -> dict:
    """
    Run the Ensemble Thermochemistry Pipeline (enumerate -> optimize -> frequency-check -> Boltzmann rank).
    Calculates Boltzmann populations and ensemble-averaged free energy G.
    
    Parameters:
    - molecule_id: The stored molecule handle (e.g. mol_a1b2c3d4)
    - method: Underlying xTB parameterization ('GFN1-xTB', 'GFN2-xTB', or 'GFN-FF')
    - solvent: Implicit GBSA/ALPB solvent model name (e.g. water, methanol, benzene)
    - energy_window_kcal: Conformer search energy window in kcal/mol (default is 6.0)
    - max_conformers_to_optimize: Limit refinement calculations to top N conformers (default is 5)
    - energy_threshold_kcal: Limit refinement to conformers within X kcal/mol of minimum (default is 3.0)
    - charge: Net molecular charge (default is 0)
    - spin: Spin multiplicity (2S + 1) (default is 1, singlet)
    - run_async: If true, runs pipeline in background (highly recommended, default is True).
    """
    if not XTB_AVAILABLE or not CREST_AVAILABLE:
        raise BackendUnavailableError(
            "CREST and xTB binaries are required to run the ensemble thermochemistry pipeline.",
            hint="Install the crest and xtb binaries to run ensemble thermochemistry."
        )
        
    workspace_id = get_workspace_id()
    
    # Preflight check on charge/spin multiplicity
    from ypotheto_compchem_mcp.chemistry.builder_engine import load_molecule_from_workspace
    from ypotheto_compchem_mcp.chemistry.preflight import validate_charge_spin_multiplicity
    try:
        mol = load_molecule_from_workspace(workspace_id, molecule_id)
    except Exception as e:
        return make_error_response("MOLECULE_NOT_FOUND", f"Could not load molecule {molecule_id}: {str(e)}")
        
    ok, err = validate_charge_spin_multiplicity(mol, charge, spin)
    if not ok:
        return make_error_response("INVALID_CHARGE_SPIN", err)
        
    natoms = mol.GetNumAtoms()
    # Conformer searches are expensive, plus multiple optimizations/frequencies
    est_sec = max(60, int(3.0 * natoms ** 2))
    
    if run_async or est_sec >= 10:
        job = job_manager.submit_job(
            workspace_id,
            run_ensemble_thermochemistry_engine,
            est_sec,
            workspace_id,
            molecule_id,
            method,
            solvent,
            energy_window_kcal,
            max_conformers_to_optimize,
            energy_threshold_kcal,
            charge,
            spin
        )
        return make_success_response(
            results={
                "job_id": job.job_id,
                "status": job.status,
                "estimated_time_seconds": job.estimated_time_seconds,
                "message": f"Submitted ensemble thermochemistry pipeline. Poll status via get_job_status('{job.job_id}')."
            },
            interpretation=f"Ensemble thermochemistry pipeline submitted to background. Job ID: {job.job_id}. Estimate: {est_sec} seconds."
        )
        
    res = run_ensemble_thermochemistry_engine(
        workspace_id,
        molecule_id,
        method,
        solvent,
        energy_window_kcal,
        max_conformers_to_optimize,
        energy_threshold_kcal,
        charge,
        spin
    )
    
    if not res["ok"]:
        return make_error_response(res["error"]["code"], res["error"]["message"])
        
    return make_success_response(
        results=res["results"],
        interpretation=res["interpretation"],
        meta={
            "molecule_id": molecule_id,
            "method": f"Ensemble/{method}"
        }
    )
