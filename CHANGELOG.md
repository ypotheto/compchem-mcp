# Changelog

All notable changes to this project are documented here.

## [Unreleased] - Phase 6: advisor & guidance layer

### Added
- `src/ypotheto_compchem_mcp/content/concepts.yaml`: 30 plain-language
  explanations of core computational chemistry concepts (basis sets, DFT
  functional choice, HF vs. DFT, GFN2-xTB vs. DFT, when to trust an MLFF,
  ZPE, Gibbs thermochemistry, conformer ensembles, Boltzmann weighting,
  transition states, NEB, activation barriers, HOMO/LUMO gap, Mulliken vs.
  Löwdin charges, Hansen solubility parameters, Ra distance, VLE flash,
  k-points, surface slabs, adsorption sites, Lipinski's Rule of Five,
  tautomers, standardization, MD ensembles, radius of gyration, RDF, MSD,
  imaginary frequencies, spin multiplicity, charge state).
- New `modules/advisor_tools.py` with two tools and four MCP prompts:
  - `explain_concept(concept)`: looks up one concept, or lists all available
    keys when called with an empty string.
  - `recommend_workflow(goal, molecule_id=None)`: deterministic
    keyword-matched tool chains (e.g. "activation barrier" ->
    build_molecule_from_smiles -> run_scientific_preflight ->
    optimize_geometry -> run_neb_calculation -> calculate_vibrations) with a
    rationale and caveats per chain; when `molecule_id` is given, tailors the
    top recommendation with a caveat suggesting xTB/MLFF over DFT for
    molecules whose estimated DFT/STO-3G runtime is slow (reusing the
    existing `estimate_calculation_time` heuristic).
  - MCP prompts `compute_reaction_barrier`, `characterize_a_molecule`,
    `screen_solvent_compatibility`, `simulate_polymer_properties` - guided,
    parameterized multi-step workflow instructions for a client LLM,
    modeled on statistician-mcp's `advisor.py`.
- `pyyaml` added as an explicit core dependency (was previously an
  undeclared transitive dependency - the concepts loader now genuinely
  needs it at runtime).

### Fixed
- Found (not fixed - out of scope for this phase, flagged as a follow-up)
  a pre-existing bug while writing this phase's tests: building a molecule
  from a pathologically long linear SMILES chain (e.g. 60 carbons) crashes
  `build_molecule_from_smiles_engine`'s random-coordinate embedding fallback
  with a raw `Boost.Python.ArgumentError` instead of embedding or failing
  cleanly.

## [Unreleased] - Phase 5: packaging, docs, CI hygiene

### Added
- `scripts/gen_tool_catalog.py`: regenerates the README's "Tool Catalog
  Overview" table from the actual `mcp.list_tools()` registrations, so it
  can't silently drift out of sync again (it had drifted to list 26 of 40
  real tools).
- Implemented PDB export in `get_3d_coordinates` (via `Chem.MolToPDBBlock`)
  — the README already claimed PDB support that the tool didn't actually
  have.
- README documents the extras install matrix and every new environment
  variable from Phases 3–4 (`COMPCHEM_ALLOWED_ORIGINS`,
  `COMPCHEM_REQUEST_TIMEOUT_SECONDS`, `COMPCHEM_ARTIFACT_URL_EXPIRY_SECONDS`,
  `COMPCHEM_DATABASE_URL`, `COMPCHEM_SPACES_*`).
- `tests/test_integration.py`: a `@pytest.mark.integration` tier that runs
  real xtb/PySCF/CREST/Packmol/LAMMPS calculations (auto-skipped per-test via
  `shutil.which`/`importlib.util.find_spec` when the backend isn't
  installed), checked against known reference values (GFN2-xTB water ≈
  -5.070 Ha, HF/STO-3G water ≈ -74.96 Ha). The project Docker image (`docker
  build --target test`) is the canonical venue with every backend installed.

### Changed
- Split heavy backends out of the hard dependency list into optional extras
  (`pyproject.toml`): `[qm]` (pyscf, cclib), `[mlff]` (chgnet, mace-torch),
  `[thermo]` (cantera, juliacall), `[md]` (MDAnalysis), `[ts]` (sella),
  `[db]` (psycopg2-binary), `[s3]` (boto3), `[all]` = union. Matches the
  `*_AVAILABLE` guards the code already had; a core-only install can now
  start the server and serve the RDKit/ASE tool subset without any of these.
- Fixed the `pyclapeyron` dependency lie: declared but never imported
  (`thermo_engine.py` actually uses `juliacall`) — dropped the former,
  declared the latter. Dropped `geometric`, an unused dependency (never
  imported anywhere in the codebase).
- Every `BackendUnavailableError` hint that names a pip-installable package
  now says `pip install ypotheto-compchem-mcp[extra]` instead of a bare
  "install X", so the calling LLM gets an actionable, exact command.
- Added `[tool.ruff]`, `[tool.mypy]`, `[tool.pytest.ini_options]` config
  blocks and a `dev` extra, matching statistician-mcp's setup. Dynamic
  versioning: `__version__` in `__init__.py` is now the single source of
  truth (`pyproject.toml` no longer hardcodes a separate version).
- `ruff check src/ tests/` is now clean: fixed unsorted/unused imports,
  legacy `Dict`/`Optional` typing syntax, bare `raise` inside `except`
  clauses (losing the original traceback), mutable default arguments,
  unused loop variables, and `zip()` calls without `strict=`.

### Fixed
- `qm_engine.py` imported `cclib` unconditionally at module level — this
  forced `cclib` to be installed just to import the QM tool module at all,
  even without PySCF. Made lazy (only needed for the Docker-runner log-
  parsing path).
- `sella` was the only optional backend with no availability guard at all —
  a missing install crashed with a raw `ImportError` deep inside the engine.
  Added a proper `SELLA_AVAILABLE` flag and `BackendUnavailableError` with an
  install hint, matching every other backend's pattern.
- `pack_amorphous_cell_engine`'s fallback packer (used when Packmol isn't
  available) computed a random polar angle (`phi`) for 3D molecule
  orientation but never applied it — every packed molecule ended up rotated
  only about one fixed axis (Z), biasing the packing away from a true random
  orientation. Now uses `scipy.spatial.transform.Rotation.random()` for a
  genuinely uniform random 3D rotation.
- `run_mlff_molecular_dynamics_engine` created a temp directory via
  `tempfile.mkdtemp()` that was never used (the trajectory is built
  in-memory) or cleaned up, leaking an empty directory on disk every run.
- **Near-miss caught before landing**: a mechanical `ruff check --fix` pass
  silently deleted `server.py`'s entire block of `from
  ypotheto_compchem_mcp.modules import ...` statements, since nothing in the
  file references the imported module names directly (they're side-effect
  imports — each module's `@mcp.tool()` decorators register against the
  shared `mcp` instance just by being imported). This would have reduced the
  running server to a single tool (`ping`) out of 40. Caught by the new
  `scripts/gen_tool_catalog.py` generator suddenly reporting 1 tool instead
  of 40; restored with `# noqa` annotations and a comment warning against
  ever "cleaning up" this block again.
- **`Dockerfile` has never actually been able to build**: its runtime stage
  ran `apt-get install ... julia ...`, but Debian bookworm's repos don't
  carry a `julia` package at all (`E: Package 'julia' has no installation
  candidate`). Found by actually building the image rather than trusting the
  file read cleanly. Fixed by installing Julia from the official generic
  linux-x86_64 tarball, the same pattern the Dockerfile already used for the
  CREST binary.
- Once the above was fixed, the build failed again one step later at the
  CREST download itself: the pinned URL
  (`crest-gnu-v3.0.2-x86_64.tar.xz`) 404s — crest-lab renamed their v3.0.2
  release assets (now `crest-gnu-12-ubuntu-latest.tar.xz`) at some point after
  this line was written. The new asset also changed its internal layout (a
  `crest/` directory containing the binary, rather than the binary at the
  tarball root), so the extraction command needed `--strip-components=1`
  too, not just a new URL - verified by actually downloading and extracting
  the real asset before touching the Dockerfile again.
- A third build failure surfaced one step later still: `uv pip install -e
  ".[all]"` in the `builder` stage failed with `OSError: Error getting the
  version from source 'regex': file does not exist:
  src/ypotheto_compchem_mcp/__init__.py`. A direct regression from this same
  phase's own switch to dynamic versioning (`[tool.hatch.version] path =
  "src/ypotheto_compchem_mcp/__init__.py"`) - the `builder` stage only ever
  `COPY`'d `pyproject.toml`, never `src/`, which was harmless while the
  version was a static string in the TOML file but broke as soon as
  hatchling needed to read `__init__.py` to resolve it. Fixed by copying
  `src/` and `README.md` (also referenced via `readme = "README.md"`) into
  the `builder` stage before the install step.
- The next build succeeded, but running the resulting image
  (`docker run ypotheto-compchem-mcp:test`) failed with `No module named
  pytest`, despite the build log showing `pytest` installed successfully.
  Root cause: `uv venv` (used to create `.venv` in the `builder` stage) does
  not seed a `pip` binary into the venv, so the `test` stage's `RUN pip
  install -e ".[dev]"` silently fell through `PATH` to the base image's
  system `pip` and installed the dev dependencies into system
  site-packages - invisible to `/app/.venv/bin/python`, which only sees the
  venv's own site-packages. Fixed by using `uv pip install` (the same tool
  the `builder` stage already uses) instead of bare `pip`.
- With `pytest` now actually reachable, test collection itself failed for
  every module that (transitively) imports `builder_engine.py`:
  `ImportError: libXrender.so.1: cannot open shared object file`. RDKit's
  `Chem.Draw` submodule links against libXrender at import time even in a
  headless server with nothing to render to, and the slim runtime base image
  doesn't ship it. Fixed by adding `libxrender1`, `libxext6`, and `libsm6` to
  the `runner` stage's apt packages.
- With collection fixed, `import ypotheto_compchem_mcp.server` (transitively
  importing `juliacall` via `thermo_engine.py`) crashed with `ERROR: could
  not load library "/usr/local/bin/../lib/julia/sys.so": ... No such file or
  directory`. Root cause: the Julia install used a symlink
  (`/usr/local/bin/julia` -> `/opt/julia-1.10.5/bin/julia`), and `juliacall`
  loads `libjulia` by `dlopen`, computing the sysimage path relative to
  wherever it opened the library from - that computation doesn't resolve
  symlinks the way the OS's process loader does for a plain shell
  invocation of `julia`, so it looked for the sysimage under
  `/usr/local/lib/julia/` (doesn't exist) instead of the real
  `/opt/julia-1.10.5/lib/julia/` (does exist). Running `julia` directly from
  a shell gave no hint of the problem since that path *is* resolved
  correctly for direct execution - only `juliacall`'s library-loading path
  math broke. Fixed by adding `/opt/julia-1.10.5/bin` to `PATH` directly
  instead of symlinking into `/usr/local/bin`.
- With all five of the above fixed, `docker run ypotheto-compchem-mcp:test`
  (i.e. `pytest -m integration` with every real backend installed) passes:
  `5 passed, 143 deselected` - GFN2-xTB, PySCF HF/STO-3G, CREST, Packmol, and
  LAMMPS all verified against real physics, not mocks.

### Known issue
- `mypy src/` cannot currently complete: the installed `rdkit` package ships
  a malformed bundled type stub (`rdkit-stubs/Chem/rdMolDescriptors.pyi`
  contains a C++-codegen artifact, `rdkit.rdBase._vectunsigned int`, with a
  literal space in the type name) that fails to parse and aborts the whole
  mypy invocation before checking any project code. Confirmed as a genuine
  upstream RDKit packaging bug, not something project config can route
  around (PEP 561 stub-only packages are parsed for their public interface
  regardless of `follow_imports`). A `[[tool.mypy.overrides]]` skip for
  `rdkit.*` is in place in case a different environment/rdkit version
  doesn't hit this.

## [Unreleased] - Phase 4: output discipline & async-job parity

### Added
- `utils/limits.py::cap_series()`: uniform decimation helper, `(values,
  max_points=200) -> (decimated, was_truncated)`.
- Bounded inline output for four tools whose response size scaled with a
  user-controlled parameter — full data now goes to an artifact, inline
  results carry only a bounded preview plus the artifact link:
  - `run_reactor_kinetics`: full timeseries → CSV artifact, ≤200-point
    decimated preview inline, `results.truncated` flag.
  - `run_conformer_search`: all conformer geometries → one multi-record SDF
    artifact; `xyz_block` kept inline only for the lowest-energy conformer.
  - `run_ensemble_thermochemistry`: full per-conformer frequency tables →
    JSON artifact; only the lowest-Gibbs conformer's first 20 frequencies
    are shown inline, alongside summary stats for every conformer.
  - `get_3d_coordinates`: content over 50 KB is omitted inline (artifact-only
    + a warning) instead of returned uncapped.

### Fixed
- **Async/sync envelope-shape inconsistency** in `jobs.py`: `get_job_status`
  used to return differently-nested result structures depending on whether
  `COMPCHEM_DATABASE_URL` was set — the DB-backed execution path stored the
  *whole* envelope as `JobState.results`, while the thread-fallback path
  stored only the inner `results` dict. Both paths now extract fields
  through one shared `_envelope_to_job_fields()` helper.
- `database.py` never issued `CREATE SCHEMA IF NOT EXISTS compchem` before
  creating tables inside it — would fail outright against a genuinely fresh
  database.
- **Async jobs silently dropped artifacts/interpretation/provenance** that
  the synchronous tool wrapper builds after calling the engine, because
  async submission calls the raw `*_engine` function directly. Audited all
  17 job-submitting tools; found 9 that genuinely lose data this way
  (`run_single_point`, `optimize_geometry`, `run_pyscf_properties`,
  `calculate_vibrations`, `simulate_ir_spectrum`, `run_molecular_dynamics`,
  `pack_amorphous_cell`, `run_transition_state_search`,
  `run_ensemble_thermochemistry`) and gave each a `finalize_<tool>` +
  composed `run_<tool>_job` function registered in the durable job queue, so
  async and sync now return identical output. The other 8 registered
  engines were already thin envelope pass-throughs with nothing to lose.
- **`optimize_geometry` was silently crashing every default-configuration
  async call** (`run_async=True` is its default): its job submission passed
  `None` positionally for `progress_callback` while the job runner also
  re-injected it as a keyword, raising `TypeError: got multiple values for
  argument 'progress_callback'` on every invocation. Fixed by making
  `progress_callback` the function's own trailing keyword parameter instead
  of threading it through positionally.

## [Unreleased] - Phase 3: HTTP & storage security hardening

### Fixed
- **Security bug**: `serve_artifact` took `workspace_id` from the URL path
  with only a `".."` substring check — any authenticated caller could read
  any other workspace's artifacts by guessing or reusing a path. Now
  compares the path's `workspace_id` against the caller's own auth-resolved
  workspace and 404s (not 403, to avoid confirming existence) on mismatch.
- **Signed artifact URLs replace the raw shared secret in `?t=`**: the
  previous `?t={api_token}` query parameter leaked the shared secret into
  chat transcripts, logs, and referrer headers. Signed URLs (`?exp=&sig=`,
  HMAC, 7-day default expiry) are self-authenticating and bypass the Bearer
  check for that route; `?t=` query-param auth was removed entirely.
- **`AuthMiddleware` rewritten as plain ASGI** (was `BaseHTTPMiddleware`,
  which buffers the whole response and breaks streaming — the `/mcp` mount
  is a streamable-HTTP app). Added a POST-scoped `TimeoutMiddleware`
  (`COMPCHEM_REQUEST_TIMEOUT_SECONDS`, default 120s) so a hung tool call
  can't hang the connection forever, without ever interrupting a GET.
- **CORS locked down**: `allow_origins=["*"]` replaced with
  `COMPCHEM_ALLOWED_ORIGINS` (empty by default — no browser clients exist,
  so same-origin only is the correct default).
- **Storage traversal/concurrency hardening**: replaced the naive `".." in
  path` substring check with a proper lexical validator (rejects absolute
  paths, drive letters, backslashes, and any `..` path segment); moved
  `mkdir` out of the read path (only writes should create directories);
  added retry-on-`PermissionError` around reads/writes/deletes (Windows
  antivirus/indexer transiently locks files); `delete_file` is now
  idempotent; `list_files` tolerates files vanishing mid-walk; `boto3`
  import is now lazy (a core install without the `[s3]` extra can still
  import the storage module).
- Test-isolation fixture now also neutralizes `api_token` in addition to the
  database URL and Spaces bucket it already pinned, so tests can never
  silently inherit the real shared secret from a developer's `.env`.
- Wired up FastMCP's DNS-rebinding protection (`TransportSecuritySettings`),
  derived from `COMPCHEM_PUBLIC_BASE_URL`.
- **Found and fixed a pre-existing, unrelated production bug** while wiring
  the above: `http_app.py` mounted `mcp.streamable_http_app()` inside its
  own `Starlette` app without ever wiring that sub-app's `lifespan` —
  Starlette's default lifespan handler doesn't recurse into a `Mount()`'d
  sub-app's own lifespan, so the streamable-HTTP session manager's task
  group never started. Every real request to the actual FastMCP endpoint
  (`/mcp/mcp` — it registers its own route at `/mcp` *relative to* the outer
  `/mcp` mount) had been failing with `RuntimeError: Task group is not
  initialized` since this `Mount` was first introduced. No existing test
  caught it because they only ever exercised the single-nested `/mcp` path,
  which 404s before reaching that code.

## [Unreleased] - Phase 2: envelope, observability, provenance

### Added
- `meta.provenance` on QM/xTB/vibrations tool responses (backend name,
  version, method, functional, basis) so a client can tell exactly how a
  number was computed.
- `compute_credits_cost` surfaced in the scientific-preflight envelope
  (results + interpretation) as advisory-only cost guidance for the calling
  LLM — no enforcement, no billing.

### Changed
- Centralized matplotlib figure creation into `utils/plotting.py` so every
  plotting tool shares one place to create figures and close leaked ones on
  error, instead of each engine importing `matplotlib.pyplot` directly.
- Unified every chemistry engine's error style onto the typed
  `CompchemError` taxonomy from Phase 1 (some engines still raised bare
  exceptions or returned ad hoc `{"ok": False, ...}` dicts directly).
- Deduplicated the xTB-unavailable check across the kinetics and periodic
  engines into a shared `chemistry/_backend_checks.py` helper.

### Fixed
- LAMMPS stdout was buffered entirely in memory for long production runs;
  now streamed to a log file.
- A mislabeled `energy_hartree` unit-conversion bug in periodic DFT.

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
