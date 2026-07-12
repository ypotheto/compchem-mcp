from ypotheto_compchem_mcp.chemistry.polymer_engine import (
    _parse_lammps_thermo_log,
    _parse_lammps_thermo_log_file,
)

CANNED_LAMMPS_LOG = """LAMMPS (2 Aug 2023)
Reading data file ...
Setting up Verlet run ...
Step          Temp          Press          PotEng         KinEng         TotEng        Density
         0   300           1.2345        -120.5          0.895         -119.605       0.87654321
       100   298.5         1.1998        -125.3          0.891         -124.409       0.87700000
       200   301.2         1.2101        -128.75         0.899         -127.851       0.87812345
Loop time of 0.123456 on 1 procs for 200 steps with 12 atoms

Performance: 1.000 ns/day, 24.000 hours/ns, 100.000 timesteps/s
"""

CORRUPT_LAMMPS_LOG = """LAMMPS (2 Aug 2023)
ERROR: Something went wrong during setup
Segmentation fault (core dumped)
"""

# Mirrors the real generated sim.in script structure: a pre-equilibration
# `run 0` (before the ensemble fix is even applied) followed by the real
# production `run {steps}`. LAMMPS reprints the thermo header for each `run`
# invocation, so this has TWO header/table blocks - the parser must read the
# LAST one (the production run), not the first (pre-equilibration, step 0).
TWO_RUN_LAMMPS_LOG = """LAMMPS (2 Aug 2023)
Reading data file ...
Setting up Verlet run ...
Step          Temp          Press          PotEng         KinEng         TotEng        Density
         0   300           1.5            -20.0           0.9           -19.1          0.5
Loop time of 0.001 on 1 procs for 0 steps with 8 atoms

Setting up Verlet run ...
Step          Temp          Press          PotEng         KinEng         TotEng        Density
         0   300           1.5            -20.0           0.9           -19.1          0.5
       100   298.0         1.2            -125.3          0.891         -124.409       0.877
       200   301.2         1.21           -128.75         0.899         -127.851       0.87812345
Loop time of 0.123456 on 1 procs for 200 steps with 8 atoms

Performance: 1.000 ns/day, 24.000 hours/ns, 100.000 timesteps/s
"""


def test_parses_final_thermo_row_from_canned_log():
    result = _parse_lammps_thermo_log(CANNED_LAMMPS_LOG)
    assert result is not None
    assert result["potential_energy_kcal_mol"] == -128.75
    assert result["final_density_g_cm3"] == 0.87812345


def test_parses_production_run_table_not_preequilibration_table():
    """Regression test: the parser must not report the pre-equilibration
    `run 0` row (PotEng=-20.0) as the final result when a second, real
    production `run {steps}` table follows it."""
    result = _parse_lammps_thermo_log(TWO_RUN_LAMMPS_LOG)
    assert result is not None
    assert result["potential_energy_kcal_mol"] == -128.75
    assert result["final_density_g_cm3"] == 0.87812345


def test_returns_none_for_corrupt_or_missing_thermo_output():
    result = _parse_lammps_thermo_log(CORRUPT_LAMMPS_LOG)
    assert result is None


def test_returns_none_for_empty_log():
    assert _parse_lammps_thermo_log("") is None


def test_file_variant_parses_final_thermo_row_from_log_on_disk(tmp_path):
    """The file-streaming variant must agree with the in-memory parser, since
    production runs read the LAMMPS log from disk instead of buffering all of
    stdout in memory."""
    log_path = tmp_path / "lammps.log"
    log_path.write_text(TWO_RUN_LAMMPS_LOG, encoding="utf-8")

    result = _parse_lammps_thermo_log_file(str(log_path))
    assert result is not None
    assert result["potential_energy_kcal_mol"] == -128.75
    assert result["final_density_g_cm3"] == 0.87812345


def test_file_variant_returns_none_for_missing_file(tmp_path):
    missing_path = tmp_path / "does_not_exist.log"
    assert _parse_lammps_thermo_log_file(str(missing_path)) is None
