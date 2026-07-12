from typing import Optional
from ypotheto_compchem_mcp.server import mcp
from ypotheto_compchem_mcp.envelope import mcp_tool_decorator, make_success_response
from ypotheto_compchem_mcp.workspace import get_workspace_id
from ypotheto_compchem_mcp.chemistry.kinetics_engine import (
    run_transition_state_search_engine,
    run_neb_calculation_engine
)

@mcp.tool()
@mcp_tool_decorator
def run_transition_state_search(
    molecule_id: str,
    method: str = "xTB",
    functional: str = "B3LYP",
    basis: str = "sto-3g",
    charge: int = 0,
    spin: int = 0,
    run_async: bool = True
) -> dict:
    """
    Perform a transition state search (first-order saddle point) using the Sella optimizer.
    
    Parameters:
    - molecule_id: Workspace ID of the approximate guess TS structure.
    - method: Quantum chemical method ('xTB' or DFT methods like 'B3LYP').
    - functional: DFT functional if method is not xTB (default 'B3LYP').
    - basis: Basis set if method is not xTB (default 'sto-3g').
    - charge: Net charge of the system (default 0).
    - spin: Spin multiplicity (default 0).
    - run_async: If true, runs optimization in the background (default is True).
    """
    workspace_id = get_workspace_id()
    est_sec = 30
    
    from ypotheto_compchem_mcp.jobs import job_manager
    
    if run_async:
        job = job_manager.submit_job(
            workspace_id,
            run_transition_state_search_engine,
            est_sec,
            workspace_id,
            molecule_id,
            method,
            functional,
            basis,
            charge,
            spin
        )
        return make_success_response(
            results={
                "job_id": job.job_id,
                "status": job.status,
                "estimated_time_seconds": job.estimated_time_seconds,
                "message": f"Submitted transition state search. Poll status via get_job_status('{job.job_id}')."
            },
            interpretation=f"Transition state search job submitted. Job ID: {job.job_id}."
        )
        
    res = run_transition_state_search_engine(
        workspace_id, molecule_id, method, functional, basis, charge, spin
    )
    
    interpretation = (
        f"Transition state search completed successfully: {res['ts_molecule_id']} ({res['name']}).\n"
        f"Final Energy = {res['energy_ev']:.4f} eV, Atoms = {res['num_atoms']}."
    )
    
    return make_success_response(
        results=res,
        interpretation=interpretation,
        meta={"ts_molecule_id": res["ts_molecule_id"]}
    )

@mcp.tool()
@mcp_tool_decorator
def run_neb_calculation(
    reactant_molecule_id: str,
    product_molecule_id: str,
    num_images: int = 5,
    method: str = "xTB",
    functional: str = "B3LYP",
    basis: str = "sto-3g",
    charge: int = 0,
    spin: int = 0,
    interpolation: str = "idpp",
    run_async: bool = True
) -> dict:
    """
    Optimize reaction pathway and energy barrier using Nudged Elastic Band (NEB).
    
    Parameters:
    - reactant_molecule_id: Workspace ID of the reactant structure.
    - product_molecule_id: Workspace ID of the product structure.
    - num_images: Number of intermediate replica images (default 5).
    - method: Quantum chemical method ('xTB' or DFT).
    - functional: DFT functional if method is not xTB (default 'B3LYP').
    - basis: Basis set if method is not xTB (default 'sto-3g').
    - charge: Net charge of the system (default 0).
    - spin: Spin multiplicity (default 0).
    - interpolation: Method to create intermediate images ('linear' or 'idpp', default 'idpp').
    - run_async: If true, runs NEB optimization in background (default is True).
    """
    workspace_id = get_workspace_id()
    est_sec = 60
    
    from ypotheto_compchem_mcp.jobs import job_manager
    
    if run_async:
        job = job_manager.submit_job(
            workspace_id,
            run_neb_calculation_engine,
            est_sec,
            workspace_id,
            reactant_molecule_id,
            product_molecule_id,
            num_images,
            method,
            functional,
            basis,
            charge,
            spin,
            interpolation
        )
        return make_success_response(
            results={
                "job_id": job.job_id,
                "status": job.status,
                "estimated_time_seconds": job.estimated_time_seconds,
                "message": f"Submitted NEB pathway calculation. Poll status via get_job_status('{job.job_id}')."
            },
            interpretation=f"NEB pathway job submitted. Job ID: {job.job_id}."
        )
        
    res = run_neb_calculation_engine(
        workspace_id, reactant_molecule_id, product_molecule_id, num_images,
        method, functional, basis, charge, spin, interpolation
    )
    
    return make_success_response(
        results=res["results"],
        interpretation=res["interpretation"],
        artifacts=res.get("artifacts", []),
        meta={"reactant_molecule_id": reactant_molecule_id, "product_molecule_id": product_molecule_id}
    )
