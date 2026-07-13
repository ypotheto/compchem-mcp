
from mcp.server.fastmcp import FastMCP

from ypotheto_compchem_mcp.artifacts import register_artifact
from ypotheto_compchem_mcp.chemistry.md_engine import run_molecular_dynamics_engine
from ypotheto_compchem_mcp.chemistry.qm_engine import (
    estimate_time_seconds as _estimate_time_seconds,
)
from ypotheto_compchem_mcp.envelope import make_success_response, mcp_tool_decorator
from ypotheto_compchem_mcp.jobs import job_manager
from ypotheto_compchem_mcp.workspace import get_workspace_id


def _finalize_run_molecular_dynamics(
    res: dict, molecule_id: str, steps: int, temperature_k: float, ensemble: str, calculator_type: str
) -> dict:
    traj_bytes = res["trajectory_xyz"].encode("utf-8")
    traj_art = register_artifact(f"{molecule_id}_trajectory.xyz", traj_bytes, "structure", "MD Trajectory Coordinates (XYZ)")
    plot_art = register_artifact(f"{molecule_id}_md_profile.png", res["plot_bytes"], "plot", "MD Energy/Temperature Profile Plot")

    interpretation = (
        f"Molecular Dynamics simulation completed ({ensemble} ensemble, {steps} steps at {temperature_k} K). "
        f"Final temperature = {res['results']['final_temperature_k']:.1f} K. "
        f"Saved coordinate trajectory and diagnostic energy/temperature plot."
    )

    return make_success_response(
        results=res["results"],
        interpretation=interpretation,
        warnings=res["warnings"],
        artifacts=[traj_art, plot_art],
        meta={
            "molecule_id": molecule_id,
            "ensemble": ensemble,
            "calculator": calculator_type
        }
    )

def run_molecular_dynamics_job(
    workspace_id, molecule_id, steps, time_step_fs, temperature_k, ensemble,
    calculator_type, functional, basis, charge, spin, progress_callback=None
):
    res = run_molecular_dynamics_engine(
        workspace_id, molecule_id, steps, time_step_fs, temperature_k, ensemble,
        calculator_type, functional, basis, charge, spin, progress_callback
    )
    return _finalize_run_molecular_dynamics(res, molecule_id, steps, temperature_k, ensemble, calculator_type)

@mcp_tool_decorator
def run_molecular_dynamics(
    molecule_id: str,
    steps: int = 200,
    time_step_fs: float = 0.5,
    temperature_k: float = 300.0,
    ensemble: str = "NVT",
    calculator_type: str = "MMFF94",
    functional: str | None = "B3LYP",
    basis: str | None = "sto-3g",
    charge: int = 0,
    spin: int = 0,
    run_async: bool = True
) -> dict:
    """
    Run molecular dynamics (MD) simulations to study motion and thermal relaxation.
    
    Parameters:
    - molecule_id: The stored molecule handle (e.g. mol_a1b2c3d4)
    - steps: Total MD simulation steps (default 200)
    - time_step_fs: Integrator timestep in femtoseconds (default 0.5 fs)
    - temperature_k: Target simulation temperature in Kelvin (default 300 K)
    - ensemble: Thermodynamic ensemble, either 'NVT' (Langevin) or 'NVE' (VelocityVerlet) (default NVT)
    - calculator_type: Energy calculator, either 'MMFF94', 'UFF', 'DFT', or 'HF' (default MMFF94)
    - functional: XC functional (only used for DFT, e.g. B3LYP)
    - basis: Orbital basis set (only used for DFT/HF, e.g. sto-3g)
    - charge: Net molecular charge (default is 0)
    - spin: Spin state 2S (number of unpaired electrons, default is 0)
    - run_async: If true, runs in background thread (default is True).
    """
    workspace_id = get_workspace_id()
    
    # Calculate execution time estimate
    # Forcefield is very fast (~0.01 seconds total), DFT scales O(N^3)
    est_sec = 3
    if calculator_type.upper() not in ("MMFF94", "UFF"):
        single_point = _estimate_time_seconds(workspace_id, molecule_id, calculator_type, basis or "sto-3g")
        est_sec = int(single_point * steps * 0.4)
        
    if run_async or est_sec >= 10:
        job = job_manager.submit_job(
            workspace_id,
            run_molecular_dynamics_job,
            est_sec,
            workspace_id,
            molecule_id,
            steps,
            time_step_fs,
            temperature_k,
            ensemble,
            calculator_type,
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
                "message": f"Submitted Molecular Dynamics simulation. Poll status via get_job_status('{job.job_id}')."
            },
            interpretation=f"MD calculation submitted in background. Job ID: {job.job_id}. Estimate: {est_sec} seconds."
        )

    res = run_molecular_dynamics_engine(
        workspace_id,
        molecule_id,
        steps,
        time_step_fs,
        temperature_k,
        ensemble,
        calculator_type,
        functional,
        basis,
        charge,
        spin
    )
    return _finalize_run_molecular_dynamics(res, molecule_id, steps, temperature_k, ensemble, calculator_type)


def register_dynamics_tools(mcp: FastMCP) -> None:
    mcp.tool()(run_molecular_dynamics)
