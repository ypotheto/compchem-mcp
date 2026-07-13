from ypotheto_compchem_mcp.chemistry.mlff_engine import (
    run_mlff_molecular_dynamics_engine,
    run_mlff_optimization_engine,
)
from ypotheto_compchem_mcp.envelope import make_success_response, mcp_tool_decorator
from ypotheto_compchem_mcp.server import mcp
from ypotheto_compchem_mcp.workspace import get_workspace_id


@mcp.tool()
@mcp_tool_decorator
def run_mlff_optimization(
    molecule_id: str,
    model_name: str = "CHGNet",
    fmax: float = 0.05,
    run_async: bool = True
) -> dict:
    """
    Optimize molecular or periodic structures using pre-trained Machine Learning Force Fields (MLFFs).
    
    Parameters:
    - molecule_id: Molecule or crystal structure ID in the workspace.
    - model_name: Name of the pre-trained MLFF model (e.g. 'CHGNet' or 'MACE').
    - fmax: Optimization force convergence threshold (default 0.05).
    - run_async: If true, runs optimization in the background (default is True).
    """
    workspace_id = get_workspace_id()
    est_sec = 25
    
    from ypotheto_compchem_mcp.jobs import job_manager
    
    if run_async:
        job = job_manager.submit_job(
            workspace_id,
            run_mlff_optimization_engine,
            est_sec,
            workspace_id,
            molecule_id,
            model_name,
            fmax
        )
        return make_success_response(
            results={
                "job_id": job.job_id,
                "status": job.status,
                "estimated_time_seconds": job.estimated_time_seconds,
                "message": f"Submitted MLFF structure optimization. Poll status via get_job_status('{job.job_id}')."
            },
            interpretation=f"MLFF optimization job submitted. Job ID: {job.job_id}."
        )
        
    res = run_mlff_optimization_engine(workspace_id, molecule_id, model_name, fmax)
    return make_success_response(
        results=res["results"],
        interpretation=res["interpretation"],
        meta={"optimized_molecule_id": res["results"]["optimized_molecule_id"]}
    )

@mcp.tool()
@mcp_tool_decorator
def run_mlff_molecular_dynamics(
    molecule_id: str,
    model_name: str = "CHGNet",
    steps: int = 1000,
    timestep_fs: float = 1.0,
    temperature_k: float = 300.0,
    ensemble: str = "nvt",
    run_async: bool = True
) -> dict:
    """
    Run classical MD simulations driven by MLFF forces.
    
    Parameters:
    - molecule_id: Starting crystal or molecular structure ID in the workspace.
    - model_name: Name of the pre-trained MLFF model (e.g. 'CHGNet' or 'MACE').
    - steps: Total MD integration steps (default 1000).
    - timestep_fs: Integration timestep in femtoseconds (default 1.0).
    - temperature_k: Target temperature in Kelvin (default 300.0).
    - ensemble: Thermodynamic ensemble ('nvt', 'npt', or 'nve', default 'nvt').
    - run_async: If true, runs MD simulation in background (default is True).
    """
    workspace_id = get_workspace_id()
    est_sec = 35
    
    from ypotheto_compchem_mcp.jobs import job_manager
    
    if run_async:
        job = job_manager.submit_job(
            workspace_id,
            run_mlff_molecular_dynamics_engine,
            est_sec,
            workspace_id,
            molecule_id,
            model_name,
            steps,
            timestep_fs,
            temperature_k,
            ensemble
        )
        return make_success_response(
            results={
                "job_id": job.job_id,
                "status": job.status,
                "estimated_time_seconds": job.estimated_time_seconds,
                "message": f"Submitted MLFF MD simulation. Poll status via get_job_status('{job.job_id}')."
            },
            interpretation=f"MLFF MD simulation job submitted. Job ID: {job.job_id}."
        )
        
    res = run_mlff_molecular_dynamics_engine(
        workspace_id, molecule_id, model_name, steps, timestep_fs, temperature_k, ensemble
    )
    return make_success_response(
        results=res["results"],
        interpretation=res["interpretation"],
        artifacts=res.get("artifacts", []),
        meta={"molecule_id": molecule_id}
    )
