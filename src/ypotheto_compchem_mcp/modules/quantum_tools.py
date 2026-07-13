
from ypotheto_compchem_mcp.artifacts import register_artifact
from ypotheto_compchem_mcp.chemistry.qm_engine import (
    PYSCF_AVAILABLE,
    optimize_geometry_engine,
    run_pyscf_properties_engine,
    run_single_point_engine,
)
from ypotheto_compchem_mcp.chemistry.qm_engine import (
    estimate_time_seconds as _estimate_time_seconds,
)
from ypotheto_compchem_mcp.envelope import (
    build_provenance,
    make_error_response,
    make_success_response,
    mcp_tool_decorator,
)
from ypotheto_compchem_mcp.errors import BackendUnavailableError
from ypotheto_compchem_mcp.jobs import job_manager
from ypotheto_compchem_mcp.server import mcp
from ypotheto_compchem_mcp.workspace import get_workspace_id

_PYSCF_UNAVAILABLE_HINT = "pip install ypotheto-compchem-mcp[qm], or run inside the project's Docker image which includes it."

def _finalize_run_single_point(res: dict, molecule_id: str, method: str, functional: str, basis: str) -> dict:
    """Shared post-engine work (report artifact, interpretation, provenance) for
    run_single_point - called by BOTH the sync path and the async job runner so
    a job submitted with run_async=True gets the same envelope a sync caller
    would, instead of just the raw engine return value."""
    import json
    report_bytes = json.dumps(res, indent=2).encode("utf-8")
    report_art = register_artifact(f"{molecule_id}_qm_report.json", report_bytes, "report", "Single Point Energy Report")

    interpretation = (
        f"Single-point calculation completed. Total Energy = {res['results']['energy_ev']:.4f} eV "
        f"({res['results']['energy_hartree']:.6f} Hartree). "
        f"HOMO-LUMO Gap = {res['results']['homo_lumo_gap_ev']:.4f} eV. "
        f"Dipole Moment (X, Y, Z) = {[round(x, 4) for x in res['results']['dipole_moment_debye']]} Debye."
    )

    return make_success_response(
        results=res["results"],
        interpretation=interpretation,
        warnings=res["warnings"],
        artifacts=[report_art],
        meta={
            "molecule_id": molecule_id,
            "method": f"{method}/{functional}/{basis}",
            "provenance": build_provenance("pyscf", method=method, functional=functional, basis=basis)
        }
    )

def run_single_point_job(workspace_id, molecule_id, method, functional, basis, charge, spin, solvent):
    """Composed job function registered with the durable job queue: runs the
    engine AND the finalize step, so a background job's envelope matches the
    sync path (report artifact, interpretation, provenance) instead of just
    whatever run_single_point_engine returns on its own."""
    res = run_single_point_engine(workspace_id, molecule_id, method, functional, basis, charge, spin, solvent)
    return _finalize_run_single_point(res, molecule_id, method, functional, basis)

def _finalize_optimize_geometry(res: dict, molecule_id: str, method: str, functional: str, basis: str) -> dict:
    xyz_bytes = res["xyz_block"].encode("utf-8")
    sdf_bytes = res["sdf_block"].encode("utf-8")

    opt_mol_id = res["results"]["optimized_molecule_id"]
    xyz_art = register_artifact(f"{opt_mol_id}.xyz", xyz_bytes, "structure", "Optimized Coordinates (XYZ)")
    sdf_art = register_artifact(f"{opt_mol_id}.sdf", sdf_bytes, "structure", "Optimized Coordinates (SDF)")

    interpretation = (
        f"Geometry optimization converged in {res['results']['steps']} steps. "
        f"Final Energy = {res['results']['final_energy_ev']:.4f} eV. "
        f"New optimized molecule handle registered: {opt_mol_id}."
    )

    return make_success_response(
        results=res["results"],
        interpretation=interpretation,
        warnings=res["warnings"],
        artifacts=[xyz_art, sdf_art],
        meta={
            "molecule_id": molecule_id,
            "optimized_molecule_id": opt_mol_id,
            "method": f"{method}/{functional}/{basis}",
            "provenance": build_provenance("pyscf", method=method, functional=functional, basis=basis)
        }
    )

def run_optimize_geometry_job(
    workspace_id, molecule_id, method, functional, basis, charge, spin, max_steps, solvent, progress_callback=None
):
    """`progress_callback` is deliberately the LAST, keyword-only-in-practice
    parameter (not threaded through positionally like the sync path does):
    jobs.py's worker injects it via `kwargs["progress_callback"] = ...` after
    inspecting this function's signature. The previous code submitted
    optimize_geometry_engine directly with `None` already passed positionally
    for this same parameter, so that kwarg injection collided with it -
    `TypeError: optimize_geometry_engine() got multiple values for argument
    'progress_callback'` - meaning every async optimize_geometry call (the
    DEFAULT run_async=True) crashed and silently failed the job. Keeping this
    parameter last and never passing a positional placeholder for it in the
    submit_job call below avoids that collision."""
    res = optimize_geometry_engine(
        workspace_id, molecule_id, method, functional, basis, charge, spin, max_steps, progress_callback, solvent
    )
    return _finalize_optimize_geometry(res, molecule_id, method, functional, basis)

def _finalize_run_pyscf_properties(res: dict, molecule_id: str, method: str, functional: str, basis: str) -> dict:
    return make_success_response(
        results=res["results"],
        interpretation=res["interpretation"],
        warnings=res.get("warnings", []),
        artifacts=res["results"].get("artifacts", []),
        meta={
            "molecule_id": molecule_id,
            "method": f"{method}/{functional}/{basis}",
            "provenance": build_provenance("pyscf", method=method, functional=functional, basis=basis)
        }
    )

def run_pyscf_properties_job(workspace_id, molecule_id, method, functional, basis, charge, spin, properties, solvent):
    res = run_pyscf_properties_engine(workspace_id, molecule_id, method, functional, basis, charge, spin, properties, solvent)
    return _finalize_run_pyscf_properties(res, molecule_id, method, functional, basis)

@mcp.tool()
@mcp_tool_decorator
def estimate_calculation_time(molecule_id: str, method: str = "DFT", basis: str = "sto-3g") -> dict:
    """
    Estimate the execution time for a quantum chemistry calculation before running it.
    Use to check if a calculation is short enough to run synchronously or if it should be queued.
    
    Parameters:
    - molecule_id: The stored molecule handle (e.g. mol_a1b2c3d4)
    - method: The target method (HF or DFT)
    - basis: The basis set (e.g. sto-3g, 6-31g*)
    """
    workspace_id = get_workspace_id()
    est = _estimate_time_seconds(workspace_id, molecule_id, method, basis)
    
    results = {
        "molecule_id": molecule_id,
        "method": method,
        "basis": basis,
        "estimated_time_seconds": est,
        "run_mode_recommendation": "sync" if est < 10 else "async"
    }
    
    interpretation = (
        f"Calculations using {method}/{basis} on molecule {molecule_id} are estimated to take "
        f"approximately {est} seconds. We recommend running this calculation "
        f"{'synchronously' if est < 10 else 'asynchronously in the background'}."
    )
    
    return make_success_response(results, interpretation)

@mcp.tool()
@mcp_tool_decorator
def run_single_point(
    molecule_id: str,
    method: str = "DFT",
    functional: str = "B3LYP",
    basis: str = "sto-3g",
    charge: int = 0,
    spin: int = 0,
    run_async: bool = False,
    solvent: str | None = None
) -> dict:
    """
    Compute single-point energy, dipole moments, HOMO/LUMO energies, and Mulliken charges.
    Uses PySCF quantum chemistry engine.
    
    Parameters:
    - molecule_id: The stored molecule handle (e.g. mol_a1b2c3d4)
    - method: Method type, either 'DFT' or 'HF'
    - functional: XC functional (only used for DFT, e.g. B3LYP, PBE)
    - basis: Orbital basis set (e.g. sto-3g, 6-31g*)
    - charge: Net molecular charge (default is 0)
    - spin: Spin state 2S (number of unpaired electrons, default is 0)
    - run_async: If true, runs calculation in background and returns job ID immediately.
    - solvent: Implicit solvent model name (e.g. water, methanol, benzene)
    """
    if not PYSCF_AVAILABLE:
        raise BackendUnavailableError(
            "PySCF is not installed or available on this system host.",
            hint=_PYSCF_UNAVAILABLE_HINT
        )
        
    workspace_id = get_workspace_id()
    
    # Run preflight checks
    from ypotheto_compchem_mcp.chemistry.builder_engine import load_molecule_from_workspace
    from ypotheto_compchem_mcp.chemistry.preflight import (
        validate_basis_set_coverage,
        validate_charge_spin_multiplicity,
    )
    try:
        mol = load_molecule_from_workspace(workspace_id, molecule_id)
    except Exception as e:
        return make_error_response("MOLECULE_NOT_FOUND", f"Could not load molecule {molecule_id}: {str(e)}")
        
    ok, err = validate_charge_spin_multiplicity(mol, charge, spin + 1)
    if not ok:
        return make_error_response("INVALID_CHARGE_SPIN", err)
        
    ok, err = validate_basis_set_coverage(mol, basis)
    if not ok:
        return make_error_response("UNSUPPORTED_BASIS_SET", err)
        
    est_sec = _estimate_time_seconds(workspace_id, molecule_id, method, basis)
    
    if run_async or est_sec >= 10:
        # Submit to background executor
        job = job_manager.submit_job(
            workspace_id,
            run_single_point_job,
            est_sec,
            workspace_id,
            molecule_id,
            method,
            functional,
            basis,
            charge,
            spin,
            solvent
        )
        results = {
            "job_id": job.job_id,
            "status": job.status,
            "estimated_time_seconds": job.estimated_time_seconds,
            "message": f"Submitted to background thread. Check progress using get_job_status('{job.job_id}')."
        }
        interpretation = (
            f"The calculation is estimated to take {est_sec} seconds and has been submitted to the background. "
            f"Job ID: {job.job_id}. Check back shortly."
        )
        return make_success_response(results, interpretation)

    # Synchronous execution
    res = run_single_point_engine(workspace_id, molecule_id, method, functional, basis, charge, spin, solvent)
    return _finalize_run_single_point(res, molecule_id, method, functional, basis)

@mcp.tool()
@mcp_tool_decorator
def optimize_geometry(
    molecule_id: str,
    method: str = "DFT",
    functional: str = "B3LYP",
    basis: str = "sto-3g",
    charge: int = 0,
    spin: int = 0,
    max_steps: int = 50,
    run_async: bool = True,
    solvent: str | None = None
) -> dict:
    """
    Relax molecule coordinates using ASE LBFGS optimizer coupled with PySCF energy/gradients.
    
    Parameters:
    - molecule_id: The stored molecule handle (e.g. mol_a1b2c3d4)
    - method: Method type, either 'DFT' or 'HF'
    - functional: XC functional (only used for DFT, e.g. B3LYP)
    - basis: Orbital basis set (e.g. sto-3g, 6-31g*)
    - charge: Net molecular charge (default is 0)
    - spin: Spin state 2S (number of unpaired electrons, default is 0)
    - max_steps: Maximum LBFGS optimization steps (default 50)
    - run_async: If true, runs optimization in background (strongly recommended, default is True).
    - solvent: Implicit solvent model name (e.g. water, methanol, benzene)
    """
    if not PYSCF_AVAILABLE:
        raise BackendUnavailableError(
            "PySCF is not installed or available on this system host.",
            hint=_PYSCF_UNAVAILABLE_HINT
        )
        
    workspace_id = get_workspace_id()
    
    # Run preflight checks
    from ypotheto_compchem_mcp.chemistry.builder_engine import load_molecule_from_workspace
    from ypotheto_compchem_mcp.chemistry.preflight import (
        validate_basis_set_coverage,
        validate_charge_spin_multiplicity,
    )
    try:
        mol = load_molecule_from_workspace(workspace_id, molecule_id)
    except Exception as e:
        return make_error_response("MOLECULE_NOT_FOUND", f"Could not load molecule {molecule_id}: {str(e)}")
        
    ok, err = validate_charge_spin_multiplicity(mol, charge, spin + 1)
    if not ok:
        return make_error_response("INVALID_CHARGE_SPIN", err)
        
    ok, err = validate_basis_set_coverage(mol, basis)
    if not ok:
        return make_error_response("UNSUPPORTED_BASIS_SET", err)
        
    # Optimize takes longer: roughly multiply single point time by ~15 steps
    est_sec = _estimate_time_seconds(workspace_id, molecule_id, method, basis) * 15
    
    if run_async or est_sec >= 10:
        # NOTE: no positional placeholder for progress_callback here (see
        # run_optimize_geometry_job's docstring) - jobs.py injects it as a
        # kwarg since it detects the parameter in that function's signature.
        job = job_manager.submit_job(
            workspace_id,
            run_optimize_geometry_job,
            est_sec,
            workspace_id,
            molecule_id,
            method,
            functional,
            basis,
            charge,
            spin,
            max_steps,
            solvent
        )
        results = {
            "job_id": job.job_id,
            "status": job.status,
            "estimated_time_seconds": job.estimated_time_seconds,
            "message": f"Submitted geometry optimization. Poll progress via get_job_status('{job.job_id}')."
        }
        interpretation = (
            f"Geometry optimization submitted to background (estimated duration: {est_sec} seconds). "
            f"Job ID: {job.job_id}. Poll status to retrieve relaxed coordinates when complete."
        )
        return make_success_response(results, interpretation)

    # Synchronous execution
    res = optimize_geometry_engine(workspace_id, molecule_id, method, functional, basis, charge, spin, max_steps, None, solvent)
    return _finalize_optimize_geometry(res, molecule_id, method, functional, basis)

@mcp.tool()
@mcp_tool_decorator
def run_pyscf_properties(
    molecule_id: str,
    method: str = "DFT",
    functional: str = "B3LYP",
    basis: str = "sto-3g",
    charge: int = 0,
    spin: int = 0,
    properties: list[str] = ["mulliken", "loewdin", "esp", "homo_lumo_cubes"],  # noqa: B006 - MCP tool param: the default must appear in the advertised tool schema, and this list is never mutated in place.
    run_async: bool = True,
    solvent: str | None = None
) -> dict:
    """
    Perform advanced electronic structure calculations to compute properties like
    Mulliken and Loewdin populations, Electrostatic Potential (ESP) cubes, and HOMO/LUMO orbital cubes.
    
    Parameters:
    - molecule_id: The stored molecule handle (e.g. mol_a1b2c3d4)
    - method: Method type ('DFT' or 'HF')
    - functional: XC functional (only used for DFT, e.g. B3LYP, PBE)
    - basis: Orbital basis set (e.g. sto-3g, 6-31g*)
    - charge: Net molecular charge (default is 0)
    - spin: Spin state 2S (number of unpaired electrons, default is 0)
    - properties: List of properties to compute ('mulliken', 'loewdin', 'esp', 'homo_lumo_cubes')
    - run_async: If true, runs calculation in background and returns job ID.
    - solvent: Implicit solvent model name (e.g. water, methanol, benzene)
    """
    if not PYSCF_AVAILABLE:
        raise BackendUnavailableError(
            "PySCF is not installed or available on this system host.",
            hint=_PYSCF_UNAVAILABLE_HINT
        )
        
    workspace_id = get_workspace_id()
    
    # Run preflight checks
    from ypotheto_compchem_mcp.chemistry.builder_engine import load_molecule_from_workspace
    from ypotheto_compchem_mcp.chemistry.preflight import (
        validate_basis_set_coverage,
        validate_charge_spin_multiplicity,
    )
    try:
        mol = load_molecule_from_workspace(workspace_id, molecule_id)
    except Exception as e:
        return make_error_response("MOLECULE_NOT_FOUND", f"Could not load molecule {molecule_id}: {str(e)}")
        
    ok, err = validate_charge_spin_multiplicity(mol, charge, spin + 1)
    if not ok:
        return make_error_response("INVALID_CHARGE_SPIN", err)
        
    ok, err = validate_basis_set_coverage(mol, basis)
    if not ok:
        return make_error_response("UNSUPPORTED_BASIS_SET", err)
        
    est_sec = _estimate_time_seconds(workspace_id, molecule_id, method, basis)
    if "homo_lumo_cubes" in properties or "esp" in properties:
        est_sec += 5
        
    if run_async or est_sec >= 10:
        job = job_manager.submit_job(
            workspace_id,
            run_pyscf_properties_job,
            est_sec,
            workspace_id,
            molecule_id,
            method,
            functional,
            basis,
            charge,
            spin,
            properties,
            solvent
        )
        results = {
            "job_id": job.job_id,
            "status": job.status,
            "estimated_time_seconds": job.estimated_time_seconds,
            "message": f"Submitted advanced properties calculation. Poll status via get_job_status('{job.job_id}')."
        }
        interpretation = (
            f"The advanced properties calculation is estimated to take {est_sec} seconds and has been submitted. "
            f"Job ID: {job.job_id}. Check back shortly."
        )
        return make_success_response(results, interpretation)

    res = run_pyscf_properties_engine(
        workspace_id, molecule_id, method, functional, basis, charge, spin, properties, solvent
    )
    return _finalize_run_pyscf_properties(res, molecule_id, method, functional, basis)

@mcp.tool()
@mcp_tool_decorator
def get_job_status(job_id: str) -> dict:
    """
    Check progress or fetch results of a background calculation job.
    Use when polling running optimizations, molecular dynamics, or quantum calculations.
    
    Parameters:
    - job_id: The background job handle returned by async submissions (e.g. job_9b3e1a)
    """
    workspace_id = get_workspace_id()
    job = job_manager.get_job(workspace_id, job_id)
    if job is None:
        return make_error_response("JOB_NOT_FOUND", f"No job with ID '{job_id}' found in workspace.")
        
    job_dict = job.to_dict()
    
    if job.status == "completed":
        interpretation = f"Job '{job_id}' has completed successfully. Results and artifacts are ready."
    elif job.status == "failed":
        err_msg = job.error.get("message", "Unknown error") if job.error else "Unknown error"
        interpretation = f"Job '{job_id}' failed. Error details: {err_msg}"
    else:
        interpretation = (
            f"Job '{job_id}' is currently running. "
            f"Elapsed time: {job.elapsed_time_seconds} seconds (Estimate: {job.estimated_time_seconds} seconds). "
            f"Progress: {job.progress_message}"
        )
        
    return make_success_response(
        results=job_dict,
        interpretation=interpretation,
        meta={"job_id": job_id, "status": job.status}
    )
