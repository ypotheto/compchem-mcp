from typing import List, Dict, Any, Optional
from ypotheto_compchem_mcp.server import mcp
from ypotheto_compchem_mcp.envelope import mcp_tool_decorator, make_success_response, make_error_response
from ypotheto_compchem_mcp.errors import ValidationError
from ypotheto_compchem_mcp.workspace import get_workspace_id
from ypotheto_compchem_mcp.jobs import job_manager
from ypotheto_compchem_mcp.chemistry.thermo_engine import (
    run_mixture_flash_engine,
    run_reactor_kinetics_engine,
    calculate_transport_properties_engine,
    CLAPEYRON_AVAILABLE,
    CANTERA_AVAILABLE
)

@mcp.tool()
@mcp_tool_decorator
def run_mixture_flash(
    components: List[str],
    mole_fractions: List[float],
    temperature_k: float,
    pressure_pa: float,
    model: str = "PC-SAFT",
    flash_type: str = "VLE",
    run_async: bool = False
) -> dict:
    """
    Perform flash equilibrium calculations for a mixture using Clapeyron.jl.
    Calculates phase fractions (vapor/liquid) and phase compositions.
    
    Parameters:
    - components: List of component names (e.g. ["water", "ethanol"])
    - mole_fractions: Mole fraction of each component in the feed (must sum to 1.0)
    - temperature_k: Temperature in Kelvin
    - pressure_pa: Pressure in Pascals
    - model: Equation of State model name ('PC-SAFT', 'PR', or 'SRK')
    - flash_type: Flash equilibrium type ('VLE' or 'LLE')
    - run_async: If true, runs calculation in background and returns job ID.
    """
    if not CLAPEYRON_AVAILABLE:
        raise RuntimeError("Clapeyron.jl/juliacall is not available or Julia environment is not set up.")

    if len(components) != len(mole_fractions):
        raise ValidationError(
            f"components ({len(components)}) and mole_fractions ({len(mole_fractions)}) must have the same length."
        )

    if abs(sum(mole_fractions) - 1.0) > 1e-4:
        return make_error_response("INVALID_ARGUMENT", "Mole fractions must sum to 1.0.")
        
    workspace_id = get_workspace_id()
    est_sec = 5
    
    if run_async:
        job = job_manager.submit_job(
            workspace_id,
            run_mixture_flash_engine,
            est_sec,
            workspace_id,
            components,
            mole_fractions,
            temperature_k,
            pressure_pa,
            model,
            flash_type
        )
        return make_success_response(
            results={
                "job_id": job.job_id,
                "status": job.status,
                "estimated_time_seconds": job.estimated_time_seconds,
                "message": f"Submitted flash calculation. Poll status via get_job_status('{job.job_id}')."
            },
            interpretation=f"Flash calculation submitted to background. Job ID: {job.job_id}."
        )
        
    res = run_mixture_flash_engine(workspace_id, components, mole_fractions, temperature_k, pressure_pa, model, flash_type)
    if not res["ok"]:
        return make_error_response(res["error"]["code"], res["error"]["message"])
        
    return make_success_response(
        results=res["results"],
        interpretation=res["interpretation"],
        meta={"model": model, "flash_type": flash_type}
    )

@mcp.tool()
@mcp_tool_decorator
def run_reactor_kinetics(
    mechanism: str,
    initial_state: Dict[str, Any],
    reactor_type: str = "batch",
    residence_time_s: float = 1.0,
    steps: int = 100,
    run_async: bool = True
) -> dict:
    """
    Simulate chemical kinetics and species concentrations over time using Cantera.
    Useful for combustion, gas phase kinetics, and catalyst reactor simulations.
    
    Parameters:
    - mechanism: Mechanism filename or built-in identifier (e.g. 'gri30.yaml')
    - initial_state: Initial state dict (e.g. {"temperature": 1000, "pressure": 101325, "X": "CH4:1, O2:2, N2:7.52"})
    - reactor_type: Reactor configuration ('batch' or 'constant_pressure')
    - residence_time_s: Simulation duration in seconds (default is 1.0)
    - steps: Number of integration steps to output (default is 100)
    - run_async: If true, runs simulation in background (strongly recommended, default is True).
    """
    if not CANTERA_AVAILABLE:
        raise RuntimeError("Cantera is not available on this host.")
        
    workspace_id = get_workspace_id()
    est_sec = 5
    
    if run_async:
        job = job_manager.submit_job(
            workspace_id,
            run_reactor_kinetics_engine,
            est_sec,
            workspace_id,
            mechanism,
            initial_state,
            reactor_type,
            residence_time_s,
            steps
        )
        return make_success_response(
            results={
                "job_id": job.job_id,
                "status": job.status,
                "estimated_time_seconds": job.estimated_time_seconds,
                "message": f"Submitted reactor simulation. Poll status via get_job_status('{job.job_id}')."
            },
            interpretation=f"Reactor simulation submitted to background. Job ID: {job.job_id}."
        )
        
    res = run_reactor_kinetics_engine(workspace_id, mechanism, initial_state, reactor_type, residence_time_s, steps)
    if not res["ok"]:
        return make_error_response("REACTOR_SIMULATION_FAILED", res["error"]["message"])
        
    return make_success_response(
        results=res["results"],
        interpretation=res["interpretation"],
        artifacts=res.get("artifacts", []),
        meta={"mechanism": mechanism, "reactor_type": reactor_type}
    )

@mcp.tool()
@mcp_tool_decorator
def calculate_transport_properties(
    components: List[str],
    mole_fractions: List[float],
    temperature_k: float,
    pressure_pa: float,
    model: str = "Cantera"
) -> dict:
    """
    Calculate viscosity, thermal conductivity, and binary diffusion coefficients.
    
    Parameters:
    - components: List of species names (e.g. ["CH4", "O2"])
    - mole_fractions: Mole fractions of components (must sum to 1.0)
    - temperature_k: Temperature in Kelvin
    - pressure_pa: Pressure in Pascals
    - model: Underlying model engine ('Cantera')
    """
    if len(components) != len(mole_fractions):
        raise ValidationError(
            f"components ({len(components)}) and mole_fractions ({len(mole_fractions)}) must have the same length."
        )

    if abs(sum(mole_fractions) - 1.0) > 1e-4:
        return make_error_response("INVALID_ARGUMENT", "Mole fractions must sum to 1.0.")

    res = calculate_transport_properties_engine(components, mole_fractions, temperature_k, pressure_pa, model)
    if not res["ok"]:
        return make_error_response(res["error"]["code"], res["error"]["message"])
        
    return make_success_response(
        results=res["results"],
        interpretation=res["interpretation"]
    )
