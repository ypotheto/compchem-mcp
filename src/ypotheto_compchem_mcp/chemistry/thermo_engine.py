import os
import sys
import logging
from typing import Any, Dict, List, Optional

from ypotheto_compchem_mcp.errors import ValidationError

logger = logging.getLogger(__name__)

# 1. Initialize Cantera availability
try:
    import cantera as ct
    CANTERA_AVAILABLE = True
except ImportError:
    CANTERA_AVAILABLE = False

# 2. Initialize Clapeyron/Julia availability
try:
    from juliacall import Main as jl
    # Perform standard imports in Julia
    jl.seval("using Clapeyron")
    
    # Define a robust helper in Julia to do flash calculations
    # and return easily extractable types (tuple of floats/vectors)
    jl.seval("""
    function python_vle_flash(model, p, T, z)
        res = tp_flash(model, p, T, z)
        # tp_flash returns (liquid_fraction, vapor_fraction, liquid_x, vapor_y) or equivalent FlashResult
        # Let's extract properties in a clean, standard way
        x = res[1]
        n = res[2]
        tot = sum(n)
        return (n[1]/tot, n[2]/tot, collect(x[:, 1]), collect(x[:, 2]))
    end
    """)
    CLAPEYRON_AVAILABLE = True
except Exception as e:
    logger.warning(f"Clapeyron.jl/juliacall could not be initialized: {str(e)}")
    CLAPEYRON_AVAILABLE = False

def run_mixture_flash_engine(
    workspace_id: str,
    components: List[str],
    mole_fractions: List[float],
    temperature_k: float,
    pressure_pa: float,
    model_name: str = "PC-SAFT",
    flash_type: str = "VLE"
) -> Dict[str, Any]:
    """
    Perform a thermodynamic flash calculation on a mixture using Clapeyron.jl.
    """
    if not CLAPEYRON_AVAILABLE:
        raise RuntimeError("Clapeyron.jl/juliacall is not available on this host.")
        
    comps = [c.lower() for c in components]
    model_name_upper = model_name.upper().replace("-", "")
    
    comps_jl_str = ", ".join(f'"{c}"' for c in comps)
    model_init_code = f'{model_name_upper}([{comps_jl_str}])'
    
    try:
        model = jl.seval(model_init_code)
    except Exception as e:
        return {
            "ok": False,
            "error": {
                "code": "MODEL_INIT_FAILED",
                "message": f"Failed to initialize Clapeyron model '{model_name}': {str(e)}"
            }
        }
        
    try:
        # Pass variables to Julia Main
        jl.model = model
        jl.p = float(pressure_pa)
        jl.T = float(temperature_k)
        jl.z = [float(x) for x in mole_fractions]
        
        # Run flash
        frac_liq, frac_vap, liq_x, vap_y = jl.seval("python_vle_flash(model, p, T, z)")
        
        results = {
            "liquid_fraction": float(frac_liq),
            "vapor_fraction": float(frac_vap),
            "liquid_mole_fractions": [float(x) for x in liq_x],
            "vapor_mole_fractions": [float(y) for y in vap_y],
            "temperature_k": temperature_k,
            "pressure_pa": pressure_pa
        }
        
        vap_moles = results["vapor_mole_fractions"]
        liq_moles = results["liquid_mole_fractions"]
        interpretation = (
            f"Flash calculation completed successfully using {model_name}.\n"
            f"Vapor fraction = {results['vapor_fraction'] * 100:.1f}%, Liquid fraction = {results['liquid_fraction'] * 100:.1f}%.\n"
            f"Vapor composition: {', '.join(f'{components[i]}: {vap_moles[i]:.3f}' for i in range(len(components)))}\n"
            f"Liquid composition: {', '.join(f'{components[i]}: {liq_moles[i]:.3f}' for i in range(len(components)))}"
        )
        
        return {
            "ok": True,
            "results": results,
            "interpretation": interpretation
        }
    except Exception as e:
        return {
            "ok": False,
            "error": {
                "code": "FLASH_CALCULATION_FAILED",
                "message": f"Equilibrium flash calculation failed: {str(e)}."
            }
        }

def run_reactor_kinetics_engine(
    workspace_id: str,
    mechanism: str,
    initial_state: Dict[str, Any],
    reactor_type: str = "batch",
    residence_time_s: float = 1.0,
    steps: int = 100
) -> Dict[str, Any]:
    """
    Simulate chemical kinetics and species concentrations over time using Cantera.
    """
    if not CANTERA_AVAILABLE:
        raise RuntimeError("Cantera is not available on this host.")
        
    try:
        gas = ct.Solution(mechanism)
    except Exception as e:
        try:
            gas = ct.Solution("gri30.yaml")
        except Exception:
            raise RuntimeError(f"Could not load mechanism '{mechanism}': {str(e)}")
            
    T = initial_state.get("temperature", 300.0)
    P = initial_state.get("pressure", 101325.0)
    X = initial_state.get("X", "")
    
    gas.TPX = T, P, X
    
    if reactor_type.lower() == "cstr" or reactor_type.lower() == "constant_pressure":
        r = ct.IdealGasConstPressureReactor(gas)
    else:
        r = ct.IdealGasReactor(gas)
        
    sim = ct.ReactorNet([r])
    
    time = 0.0
    dt = residence_time_s / steps
    
    times = []
    temperatures = []
    pressures = []
    species_names = gas.species_names
    concentrations = {name: [] for name in species_names}
    
    for _ in range(steps):
        time += dt
        sim.advance(time)
        times.append(time)
        temperatures.append(float(r.thermo.T))
        pressures.append(float(r.thermo.P))
        for name in species_names:
            concentrations[name].append(float(r.thermo[name].X[0]))
            
    major_species = []
    for name in species_names:
        max_val = max(concentrations[name])
        if max_val > 0.01:
            major_species.append(name)
            
    plot_art = None
    try:
        import matplotlib.pyplot as plt
        import io
        from ypotheto_compchem_mcp.artifacts import register_artifact
        
        plt.figure(figsize=(8, 5))
        for name in major_species:
            plt.plot(times, concentrations[name], label=name)
        plt.xlabel("Time (s)")
        plt.ylabel("Mole Fraction")
        plt.title("Reactor Kinetics Concentration Profile")
        plt.legend()
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150)
        plt.close()
        
        plot_art = register_artifact("reactor_kinetics_profile.png", buf.getvalue(), "plot", "Reactor Kinetics Species Profile")
    except Exception as e:
        logger.warning(f"Failed to generate reactor kinetics plot: {str(e)}")
        
    results = {
        "times": times,
        "temperatures": temperatures,
        "pressures": pressures,
        "species_mole_fractions": {name: concentrations[name] for name in major_species},
        "final_state": {
            "temperature": float(r.thermo.T),
            "pressure": float(r.thermo.P),
            "X": {name: float(r.thermo[name].X[0]) for name in major_species}
        }
    }
    
    interpretation = (
        f"Cantera reactor simulation completed successfully.\n"
        f"Reactor type: {reactor_type.upper()} Batch Reactor.\n"
        f"Final Temperature = {r.thermo.T:.2f} K, Pressure = {r.thermo.P:.2f} Pa.\n"
        f"Major species at end: {', '.join(f'{k}: {v:.3f}' for k, v in results['final_state']['X'].items())}."
    )
    
    artifacts = [plot_art] if plot_art else []
    
    return {
        "ok": True,
        "results": results,
        "interpretation": interpretation,
        "artifacts": artifacts
    }

def calculate_transport_properties_engine(
    components: List[str],
    mole_fractions: List[float],
    temperature_k: float,
    pressure_pa: float,
    model: str = "Cantera"
) -> Dict[str, Any]:
    """
    Calculate viscosity, thermal conductivity, and binary diffusion coefficients.
    """
    if model.lower() == "cantera":
        if not CANTERA_AVAILABLE:
            raise RuntimeError("Cantera is not available on this host.")
            
        try:
            gas = ct.Solution("gri30.yaml")
            comp_dict = {c.upper(): f for c, f in zip(components, mole_fractions)}
            comp_str = ", ".join(f"{k}:{v}" for k, v in comp_dict.items())
            
            gas.TPX = temperature_k, pressure_pa, comp_str
            
            viscosity = float(gas.viscosity)
            thermal_cond = float(gas.thermal_conductivity)
            diff_coeffs = gas.binary_diff_coeffs.tolist()
            
            results = {
                "viscosity_pa_s": viscosity,
                "thermal_conductivity_w_m_k": thermal_cond,
                "binary_diffusion_coefficients_m2_s": diff_coeffs,
                "components": components
            }
            
            interpretation = (
                f"Transport properties calculated using Cantera transport models at {temperature_k} K, {pressure_pa} Pa.\n"
                f"Viscosity = {viscosity:.3e} Pa*s.\n"
                f"Thermal Conductivity = {thermal_cond:.4f} W/(m*K)."
            )
            
            return {
                "ok": True,
                "results": results,
                "interpretation": interpretation
            }
        except Exception as e:
            return {
                "ok": False,
                "error": {
                    "code": "CANTERA_TRANSPORT_FAILED",
                    "message": f"Cantera transport calculation failed: {str(e)}"
                }
            }
    else:
        raise ValidationError(
            f"Unknown transport model '{model}'.",
            hint="Supported model values are: 'Cantera'."
        )
