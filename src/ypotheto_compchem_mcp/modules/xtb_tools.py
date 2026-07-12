from typing import Optional
from ypotheto_compchem_mcp.server import mcp
from ypotheto_compchem_mcp.envelope import mcp_tool_decorator, make_success_response, make_error_response
from ypotheto_compchem_mcp.workspace import get_workspace_id
from ypotheto_compchem_mcp.jobs import job_manager
from ypotheto_compchem_mcp.chemistry.xtb_engine import (
    run_xtb_calculation_engine,
    run_conformer_search_engine,
    XTB_AVAILABLE,
    CREST_AVAILABLE
)

@mcp.tool()
@mcp_tool_decorator
def run_xtb_calculation(
    molecule_id: str,
    task: str = "single_point",
    method: str = "GFN2-xTB",
    solvent: Optional[str] = None,
    charge: int = 0,
    spin: int = 1,
    run_async: bool = False
) -> dict:
    """
    Run fast semi-empirical GFN-xTB calculations.
    GFN2-xTB and GFN-FF run in seconds and are ideal for quick structural evaluations, frequency checks, or optimizations before DFT runs.
    
    Parameters:
    - molecule_id: The stored molecule handle (e.g. mol_a1b2c3d4)
    - task: The type of calculation ('single_point', 'geometry_optimization', or 'vibrations')
    - method: GFN parameterization ('GFN1-xTB', 'GFN2-xTB', or 'GFN-FF')
    - solvent: Implicit GBSA/ALPB solvent model name (e.g. water, benzene, methanol, chloroform)
    - charge: Net molecular charge (default is 0)
    - spin: Spin multiplicity (2S + 1) (default is 1, singlet)
    - run_async: If true, runs calculation in background and returns job ID immediately.
    """
    if not XTB_AVAILABLE:
        raise RuntimeError("xtb executable is not available on this system host.")
        
    workspace_id = get_workspace_id()
    
    # Run preflight checks first
    from ypotheto_compchem_mcp.chemistry.builder_engine import load_molecule_from_workspace
    from ypotheto_compchem_mcp.chemistry.preflight import validate_charge_spin_multiplicity
    try:
        mol = load_molecule_from_workspace(workspace_id, molecule_id)
    except Exception as e:
        return make_error_response("MOLECULE_NOT_FOUND", f"Could not load molecule {molecule_id}: {str(e)}")
        
    ok, err = validate_charge_spin_multiplicity(mol, charge, spin)
    if not ok:
        return make_error_response("INVALID_CHARGE_SPIN", err)
        
    # xTB calculations are generally fast, but optimizations/vibrations on larger molecules can take > 10s
    natoms = mol.GetNumAtoms()
    est_sec = 2 if task == "single_point" else int(0.1 * natoms)
    
    if run_async or est_sec >= 10:
        job = job_manager.submit_job(
            workspace_id,
            run_xtb_calculation_engine,
            est_sec,
            workspace_id,
            molecule_id,
            task,
            method,
            solvent,
            charge,
            spin
        )
        return make_success_response(
            results={
                "job_id": job.job_id,
                "status": job.status,
                "estimated_time_seconds": job.estimated_time_seconds,
                "message": f"Submitted xTB calculation. Poll status via get_job_status('{job.job_id}')."
            },
            interpretation=f"xTB calculation submitted to background. Job ID: {job.job_id}. Estimate: {est_sec} seconds."
        )
        
    res = run_xtb_calculation_engine(workspace_id, molecule_id, task, method, solvent, charge, spin)
    if not res["ok"]:
        return make_error_response(res["error"]["code"], res["error"]["message"])
        
    return make_success_response(
        results=res["results"],
        interpretation=res["interpretation"],
        warnings=res.get("warnings", []),
        meta={
            "molecule_id": molecule_id,
            "method": f"{method}/{task}"
        }
    )

@mcp.tool()
@mcp_tool_decorator
def run_conformer_search(
    molecule_id: str,
    method: str = "GFN2-xTB",
    solvent: Optional[str] = None,
    energy_window_kcal: float = 6.0,
    run_async: bool = True
) -> dict:
    """
    Generate conformer ensembles using CREST (Conformer-Rotamer Ensemble Sampling Tool).
    Returns Boltzmann-ranked conformer list with relative energies and coordinates.
    
    Parameters:
    - molecule_id: The stored molecule handle (e.g. mol_a1b2c3d4)
    - method: Underlying xTB method ('GFN1-xTB', 'GFN2-xTB', or 'GFN-FF')
    - solvent: Implicit GBSA/ALPB solvent model name (e.g. water, benzene)
    - energy_window_kcal: Conformer energy threshold in kcal/mol (default is 6.0)
    - run_async: If true, runs CREST in background (strongly recommended, default is True).
    """
    if not CREST_AVAILABLE:
        raise RuntimeError("crest executable is not available on this system host.")
        
    workspace_id = get_workspace_id()
    
    try:
        from ypotheto_compchem_mcp.chemistry.builder_engine import load_molecule_from_workspace
        mol = load_molecule_from_workspace(workspace_id, molecule_id)
        natoms = mol.GetNumAtoms()
    except Exception as e:
        return make_error_response("MOLECULE_NOT_FOUND", f"Could not load molecule {molecule_id}: {str(e)}")
        
    est_sec = max(30, int(1.5 * natoms ** 2))
    
    if run_async or est_sec >= 10:
        job = job_manager.submit_job(
            workspace_id,
            run_conformer_search_engine,
            est_sec,
            workspace_id,
            molecule_id,
            method,
            solvent,
            energy_window_kcal
        )
        return make_success_response(
            results={
                "job_id": job.job_id,
                "status": job.status,
                "estimated_time_seconds": job.estimated_time_seconds,
                "message": f"Submitted CREST conformer search. Poll status via get_job_status('{job.job_id}')."
            },
            interpretation=f"CREST conformer search submitted to background. Job ID: {job.job_id}. Estimate: {est_sec} seconds."
        )
        
    res = run_conformer_search_engine(workspace_id, molecule_id, method, solvent, energy_window_kcal)
    if not res["ok"]:
        return make_error_response(res["error"]["code"], res["error"]["message"])
        
    return make_success_response(
        results=res["results"],
        interpretation=res["interpretation"],
        meta={
            "molecule_id": molecule_id,
            "method": f"CREST/{method}"
        }
    )
