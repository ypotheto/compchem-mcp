import pytest
from unittest.mock import MagicMock, patch
import sys

@pytest.fixture(autouse=True)
def mock_julia_and_cantera():
    mock_jl = MagicMock()
    mock_jl.Main.seval.side_effect = lambda code: (
        (0.4, 0.6, [0.1, 0.9], [0.8, 0.2]) if "python_vle_flash" in code or "tp_flash" in code
        else None
    )
    
    mock_ct = MagicMock()
    mock_gas = MagicMock()
    mock_gas.viscosity = 1.8e-5
    mock_gas.thermal_conductivity = 0.026
    import numpy as np
    mock_gas.binary_diff_coeffs = np.array([[1.0, 0.2], [0.2, 1.0]])
    mock_gas.species_names = ["CH4", "O2", "CO2", "H2O", "N2"]
    
    mock_thermo = MagicMock()
    mock_thermo.T = 1000.0
    mock_thermo.P = 101325.0
    mock_spec = MagicMock()
    mock_spec.X = np.array([0.5])
    mock_thermo.__getitem__.return_value = mock_spec
    
    mock_reactor = MagicMock()
    mock_reactor.thermo = mock_thermo
    
    mock_ct.Solution.return_value = mock_gas
    mock_ct.IdealGasReactor.return_value = mock_reactor
    mock_ct.IdealGasConstPressureReactor.return_value = mock_reactor
    
    with patch.dict(sys.modules, {"juliacall": mock_jl, "cantera": mock_ct}):
        import importlib
        import ypotheto_compchem_mcp.chemistry.thermo_engine as te
        import ypotheto_compchem_mcp.modules.thermo_tools as tt
        importlib.reload(te)
        importlib.reload(tt)
        yield te, tt

def test_run_mixture_flash_sync(mock_julia_and_cantera):
    te, tt = mock_julia_and_cantera
    
    te.CLAPEYRON_AVAILABLE = True
    tt.CLAPEYRON_AVAILABLE = True
    
    res = tt.run_mixture_flash(
        components=["ethanol", "water"],
        mole_fractions=[0.4, 0.6],
        temperature_k=350.0,
        pressure_pa=101325.0,
        model="PC-SAFT",
        flash_type="VLE",
        run_async=False
    )
    
    assert res["ok"] is True
    results = res["results"]
    assert results["vapor_fraction"] == 0.6
    assert results["liquid_fraction"] == 0.4
    assert results["liquid_mole_fractions"] == [0.1, 0.9]
    assert results["vapor_mole_fractions"] == [0.8, 0.2]

def test_run_mixture_flash_async(mock_julia_and_cantera):
    te, tt = mock_julia_and_cantera
    
    te.CLAPEYRON_AVAILABLE = True
    tt.CLAPEYRON_AVAILABLE = True
    
    res = tt.run_mixture_flash(
        components=["ethanol", "water"],
        mole_fractions=[0.4, 0.6],
        temperature_k=350.0,
        pressure_pa=101325.0,
        model="PC-SAFT",
        flash_type="VLE",
        run_async=True
    )
    
    assert res["ok"] is True
    assert "job_id" in res["results"]
    assert "estimated_time_seconds" in res["results"]

def test_run_reactor_kinetics_sync(mock_julia_and_cantera):
    te, tt = mock_julia_and_cantera
    
    te.CANTERA_AVAILABLE = True
    tt.CANTERA_AVAILABLE = True
    
    res = tt.run_reactor_kinetics(
        mechanism="gri30.yaml",
        initial_state={"temperature": 1000.0, "pressure": 101325.0, "X": "CH4:1, O2:2, N2:7.52"},
        reactor_type="batch",
        residence_time_s=1.0,
        steps=10,
        run_async=False
    )
    
    assert res["ok"] is True
    results = res["results"]
    assert "final_state" in results
    assert results["final_state"]["temperature"] == 1000.0
    assert results["final_state"]["pressure"] == 101325.0

def test_calculate_transport_properties(mock_julia_and_cantera):
    te, tt = mock_julia_and_cantera
    
    te.CANTERA_AVAILABLE = True
    
    res = tt.calculate_transport_properties(
        components=["CH4", "O2"],
        mole_fractions=[0.3, 0.7],
        temperature_k=300.0,
        pressure_pa=101325.0,
        model="Cantera"
    )
    
    assert res["ok"] is True
    results = res["results"]
    assert results["viscosity_pa_s"] == 1.8e-5
    assert results["thermal_conductivity_w_m_k"] == 0.026
    assert len(results["binary_diffusion_coefficients_m2_s"]) == 2
