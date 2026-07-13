# Multi-stage build to keep the final image slim
FROM python:3.12-slim-bookworm AS builder

# Install build dependencies required for compiling PySCF and other C-extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    gfortran \
    cmake \
    libblas-dev \
    liblapack-dev \
    libopenblas-dev \
    make \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:0.11.28 /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Copy dependency files and source. `src/` and README.md must be present
# BEFORE the install step below: pyproject.toml uses dynamic versioning
# (hatchling reads __version__ out of src/ypotheto_compchem_mcp/__init__.py
# to resolve the package version at build time), and `readme = "README.md"`
# is likewise read during metadata preparation. This previously worked with
# only `pyproject.toml` present because the version used to be a static
# string in the TOML file itself - switching to dynamic versioning without
# copying src/ into this stage broke `uv pip install -e .` outright
# ("OSError: file does not exist: src/ypotheto_compchem_mcp/__init__.py").
COPY pyproject.toml README.md .
COPY src/ ./src/

# Create virtual environment and install dependencies (including compilation of PySCF).
# Heavy backends (PySCF, Cantera/juliacall, CHGNet/MACE, MDAnalysis, Sella, psycopg2, boto3) are
# optional extras (see pyproject.toml) since a core-only install shouldn't require them - but this
# image is the canonical integration-test venue (see planning/HUMAN_TASKS.md item 4), so it installs
# everything via the [all] extra. PySCF is still installed as an explicit first step so uv compiles
# it from source during build rather than as an incidental part of the extras resolution.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv .venv && \
    uv pip install pyscf && \
    uv pip install -e ".[all]"

# --- Final lightweight runtime image ---
FROM python:3.12-slim-bookworm AS runner

# Install runtime dependencies (OpenBLAS is needed for PySCF compiled binaries, and curl/ca-certificates/xz-utils for crest/julia, packmol and lammps for polymers).
# Julia is NOT installed via apt: Debian bookworm's repos don't carry a `julia`
# package at all ("has no installation candidate") - this line previously
# failed the build outright. Installed from the official tarball instead,
# same pattern as the CREST binary download just below.
# libxrender1/libxext6/libsm6 are needed even in this headless server: RDKit's
# `rdkit.Chem.Draw` submodule (used by builder_engine.py for on-the-fly PDB/
# image generation) links against libXrender at import time regardless of
# whether anything is actually rendered to a display, and this slim base
# image doesn't ship it - every module that imports builder_engine.py failed
# with "ImportError: libXrender.so.1: cannot open shared object file".
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas-dev \
    libxrender1 \
    libxext6 \
    libsm6 \
    xtb \
    curl \
    ca-certificates \
    xz-utils \
    packmol \
    lammps \
    && rm -rf /var/lib/apt/lists/*

# Download and install CREST binary. Both the release asset name and the
# tarball's internal layout have changed upstream since this line was first
# written: the old asset name (crest-gnu-v3.0.2-x86_64.tar.xz) 404s - the
# current v3.0.2 asset is crest-gnu-12-ubuntu-latest.tar.xz - and it contains
# a crest/ directory (crest/crest, crest/LICENSE, ...), not a bare binary at
# the tarball root, so --strip-components=1 is needed to land crest directly
# in /usr/local/bin instead of /usr/local/bin/crest/crest.
RUN curl -fsSL -o /tmp/crest.tar.xz https://github.com/crest-lab/crest/releases/download/v3.0.2/crest-gnu-12-ubuntu-latest.tar.xz && \
    tar -C /usr/local/bin --strip-components=1 -xf /tmp/crest.tar.xz crest/crest && \
    chmod +x /usr/local/bin/crest && \
    rm /tmp/crest.tar.xz

# Download and install Julia (official generic linux x86_64 tarball - no apt package exists).
# Put the real bin/ directory on PATH rather than symlinking julia into
# /usr/local/bin: `juliacall` (the Python<->Julia bridge used by
# thermo_engine.py) dlopen()s libjulia directly rather than exec'ing the
# `julia` binary as a subprocess, and computes its sysimage path relative to
# wherever it dlopen'd the library from - that computation does not resolve
# symlinks the way the OS's process loader does for a plain `julia -e ...`
# invocation. A symlink at /usr/local/bin/julia therefore made juliacall look
# for the sysimage at /usr/local/lib/julia/sys.so (doesn't exist) instead of
# the real /opt/julia-1.10.5/lib/julia/sys.so, failing every `import
# juliacall` with "could not load library ... sys.so: cannot open shared
# object file" - even though running `julia` directly from a shell worked
# fine and gave no hint of the problem.
RUN curl -fsSL -o /tmp/julia.tar.gz https://julialang-s3.julialang.org/bin/linux/x64/1.10/julia-1.10.5-linux-x86_64.tar.gz && \
    tar -C /opt -xzf /tmp/julia.tar.gz && \
    rm /tmp/julia.tar.gz
ENV PATH="/opt/julia-1.10.5/bin:$PATH"

# Pre-install Clapeyron Julia library for fast container initialization
RUN julia -e 'import Pkg; Pkg.add("Clapeyron")'

# Set working directory
WORKDIR /app

# Copy virtual environment and source code from builder
COPY --from=builder /app/.venv /app/.venv
COPY src/ /app/src/
COPY README.md /app/

# Environment variables
ENV PATH="/app/.venv/bin:$PATH"
ENV COMPCHEM_DATA_DIR="/data"
ENV COMPCHEM_PORT="8348"

# Expose port
EXPOSE 8348

# Run the server via HTTP transport by default
ENTRYPOINT ["python", "-m", "ypotheto_compchem_mcp", "--transport", "http"]

# --- Integration-test image: this Docker image is the canonical venue for
# `pytest -m integration` (every heavy binary/backend above is installed here,
# unlike a bare dev machine) - see planning/HUMAN_TASKS.md item 4. Kept as a
# separate stage so the default `docker build` production target above stays
# lean; build this one explicitly with `docker build --target test`.
FROM runner AS test
COPY --from=ghcr.io/astral-sh/uv:0.11.28 /uv /uvx /bin/
COPY pyproject.toml .
COPY tests/ /app/tests/
# `uv venv` (used in the builder stage) does not seed a `pip` binary into the
# venv, so plain `pip install` here would silently fall through PATH to the
# base image's system pip and install dev deps into system site-packages -
# invisible to /app/.venv/bin/python, which only sees the venv's own
# site-packages. That happened here: the build reported "Successfully
# installed pytest-9.1.1" yet `python -m pytest` failed with "No module
# named pytest". Using `uv pip install` (same tool the builder stage uses)
# targets the active venv correctly instead.
RUN uv pip install -e ".[dev]"
ENTRYPOINT ["python", "-m", "pytest", "-m", "integration"]
