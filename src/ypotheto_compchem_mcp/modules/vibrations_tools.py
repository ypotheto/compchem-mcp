import json
from ypotheto_compchem_mcp.server import mcp
from ypotheto_compchem_mcp.envelope import mcp_tool_decorator, make_success_response
from ypotheto_compchem_mcp.artifacts import register_artifact
from ypotheto_compchem_mcp.workspace import get_workspace_id
from ypotheto_compchem_mcp.jobs import job_manager
from ypotheto_compchem_mcp.chemistry.vib_engine import run_vibrations_engine, simulate_ir_spectrum_engine
from ypotheto_compchem_mcp.modules.quantum_tools import _estimate_time_seconds

@mcp.tool()
@mcp_tool_decorator
def calculate_vibrations(
    molecule_id: str,
    method: str = "DFT",
    functional: str = "B3LYP",
    basis: str = "sto-3g",
    charge: int = 0,
    spin: int = 0,
    run_async: bool = True
) -> dict:
    """
    Run vibrational frequency analysis and calculate thermochemistry corrections.
    
    Parameters:
    - molecule_id: The stored molecule handle (e.g. mol_a1b2c3d4)
    - method: Method type, either 'DFT', 'HF', 'MMFF94', or 'UFF'
    - functional: XC functional (only used for DFT, e.g. B3LYP)
    - basis: Orbital basis set (only used for DFT/HF, e.g. sto-3g)
    - charge: Net molecular charge (default is 0)
    - spin: Spin state 2S (number of unpaired electrons, default is 0)
    - run_async: If true, runs in the background (default is True).
    """
    workspace_id = get_workspace_id()
    # Vibrations take roughly 6 * single point times
    est_sec = _estimate_time_seconds(workspace_id, molecule_id, method, basis) * 6
    
    if run_async or est_sec >= 10:
        job = job_manager.submit_job(
            workspace_id,
            run_vibrations_engine,
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
                "message": f"Submitted vibrational analysis. Poll status via get_job_status('{job.job_id}')."
            },
            interpretation=f"Vibrational analysis submitted to background. Job ID: {job.job_id}. Estimate: {est_sec} seconds."
        )
        
    res = run_vibrations_engine(workspace_id, molecule_id, method, functional, basis, charge, spin)
    
    # Save as JSON report artifact
    res_bytes = json.dumps(res, indent=2).encode("utf-8")
    report_art = register_artifact(f"{molecule_id}_vibrations.json", res_bytes, "report", "Vibrations Analysis Report")
    
    interpretation = (
        f"Vibrational analysis completed for {molecule_id}. "
        f"Zero-Point Energy (ZPE) = {res['results']['zero_point_energy_ev']:.4f} eV ({res['results']['zero_point_energy_kcal']:.2f} kcal/mol). "
        f"Found {res['results']['imaginary_modes_count']} imaginary frequencies. "
        f"Gibbs Free Energy = {res['results']['thermochemistry']['gibbs_free_energy_ev']:.4f} eV."
    )
    
    return make_success_response(
        results=res["results"],
        interpretation=interpretation,
        warnings=res["warnings"],
        artifacts=[report_art],
        meta={
            "molecule_id": molecule_id,
            "method": f"{method}/{basis}"
        }
    )

@mcp.tool()
@mcp_tool_decorator
def simulate_ir_spectrum(
    molecule_id: str,
    method: str = "DFT",
    functional: str = "B3LYP",
    basis: str = "sto-3g",
    charge: int = 0,
    spin: int = 0,
    run_async: bool = True
) -> dict:
    """
    Simulate IR intensities and generate a Lorentzian IR spectrum plot.
    
    Parameters:
    - molecule_id: The stored molecule handle (e.g. mol_a1b2c3d4)
    - method: Method type, either 'DFT', 'HF', 'MMFF94', or 'UFF'
    - functional: XC functional (only used for DFT, e.g. B3LYP)
    - basis: Orbital basis set (only used for DFT/HF, e.g. sto-3g)
    - charge: Net molecular charge (default is 0)
    - spin: Spin state 2S (number of unpaired electrons, default is 0)
    - run_async: If true, runs in the background (default is True).
    """
    workspace_id = get_workspace_id()
    est_sec = _estimate_time_seconds(workspace_id, molecule_id, method, basis) * 6
    
    if run_async or est_sec >= 10:
        job = job_manager.submit_job(
            workspace_id,
            simulate_ir_spectrum_engine,
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
                "message": f"Submitted IR spectrum simulation. Poll status via get_job_status('{job.job_id}')."
            },
            interpretation=f"IR simulation submitted to background. Job ID: {job.job_id}. Estimate: {est_sec} seconds."
        )
        
    res = simulate_ir_spectrum_engine(workspace_id, molecule_id, method, functional, basis, charge, spin)
    
    # Save plot PNG as artifact
    plot_art = register_artifact(f"{molecule_id}_ir_spectrum.png", res["plot_bytes"], "plot", "Vibrational IR Spectrum Plot")
    
    interpretation = (
        f"IR spectrum simulated successfully. Registered spectrum plot artifact. "
        f"Calculated {len(res['results']['frequencies_cm1'])} vibrational modes."
    )
    
    return make_success_response(
        results=res["results"],
        interpretation=interpretation,
        warnings=res["warnings"],
        artifacts=[plot_art],
        meta={
            "molecule_id": molecule_id,
            "method": f"{method}/{basis}"
        }
    )
