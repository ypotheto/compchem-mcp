# Changelog

All notable changes to this project are documented here.

## [0.6.0] - 2026-07-12

**Phase 1 of the excellence plan (`planning/excellence_plan.md`): scientific integrity.**
The server previously returned physically meaningless results with `ok: true` in several
situations — this release makes every result honest about how it was actually computed.

### Fixed
- Fixed a `NameError` (`loew_charges` should have been `loewdin_charges`) that crashed every
  successful synchronous `run_pyscf_properties` call while building its interpretation string.
- Removed silent Lennard-Jones potential fallbacks that masqueraded as real results:
  - `run_transition_state_search` / `run_neb_calculation`: xTB-unavailable now raises a clean
    `BACKEND_UNAVAILABLE` error instead of silently optimizing against a toy LJ potential.
  - `run_mlff_optimization` / `run_mlff_molecular_dynamics`: CHGNet/MACE-unavailable now raises
    `BACKEND_UNAVAILABLE` instead of silently reporting "optimized successfully using CHGNet"
    against LJ energies. Also fixed the MACE integration itself, which called
    `MACECalculator(default_dtype="float32")` with no model path and could never construct
    successfully — every "MACE" run was silently falling back to LJ. Now uses
    `mace_off(default_dtype="float32")`, MACE's pretrained foundation-model loader.
  - `run_periodic_dft`: xTB-unavailable and PySCF-PBC-unavailable now raise
    `BACKEND_UNAVAILABLE`/`CALCULATION_FAILED` instead of silently substituting LJ.
- `run_lammps_simulation`: the ASE-LJ fallback (used only when no LAMMPS binary is available) now
  discloses itself via `results.engine_used` and a `warnings` entry, and its interpretation string
  no longer claims "LAMMPS simulation completed successfully" when LAMMPS never ran.
- `run_lammps_simulation`: real LAMMPS runs now parse actual thermo output (potential energy,
  density) from the LAMMPS log instead of returning hardcoded placeholder values
  (`-150.0 kcal/mol`, `0.9 g/cm3`) regardless of the real result.
- `calculate_transport_properties`: unsupported models now return a clean `INVALID_ARGUMENT`
  instead of an unhandled `NotImplementedError`.
- `run_mixture_flash` / `calculate_transport_properties`: added validation that `components` and
  `mole_fractions` have matching lengths. (Note: a prior review had claimed `run_mixture_flash`
  was artificially restricted to binary mixtures — that was a misreading of Clapeyron.jl's
  `tp_flash` matrix convention; N-component flashes already worked correctly and continue to.)
- Background (async) jobs now preserve a typed error's real code and hint (e.g.
  `BACKEND_UNAVAILABLE` with an actionable install hint) instead of collapsing every failure into
  a generic `INTERNAL_JOB_ERROR` — both the DB-backed and thread-fallback job execution paths call
  engine functions directly and previously discarded this information.
- `xtb_engine.py` (and its `xtb_tools`/`ensemble_tools` callers) now raise the same typed
  `BackendUnavailableError` as the other engines above, instead of a plain `RuntimeError`.
- LAMMPS subprocess failures now log the captured `stderr` instead of discarding it.

### Added
- `src/ypotheto_compchem_mcp/errors.py`: typed error taxonomy (`CompchemError` base plus
  `ValidationError`, `MoleculeNotFoundError`, `JobNotFoundError`, `QuotaExceededError`,
  `BackendUnavailableError`, `CalculationFailedError`), wired into the tool-call envelope decorator.
- `results.method_used` / `results.engine_used` fields on tools that can run under more than one
  backend, so a client can always tell which method actually produced a result.
- Test suite grew from 62 to 83 tests, including regression coverage for the LAMMPS thermo-table
  parsing (a two-`run`-table log, matching the real generated script structure) and for typed
  errors propagating through both the sync tool-call path and the async job path.

## [0.1.2] - [0.5.0] - prior history

Phased implementation of the core server: molecule builder/cheminformatics, PySCF QM
(single-point/optimization/vibrations), xTB/CREST semi-empirical methods and conformer ensembles,
periodic crystal I/O and symmetry analysis, polymer/amorphous-cell packing and LAMMPS MD,
transition-state/NEB search, engineering thermodynamics (Clapeyron/Cantera), Hansen solubility
parameters, and MLFF (CHGNet/MACE) support. See `git log` for the detailed commit history of this
range.
