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

# Copy dependency files
COPY pyproject.toml .

# Create virtual environment and install dependencies (including compilation of PySCF)
# We also install pyscf directly here so that uv compiles it from source during build
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv .venv && \
    uv pip install pyscf && \
    uv pip install -e .

# --- Final lightweight runtime image ---
FROM python:3.12-slim-bookworm AS runner

# Install runtime dependencies (OpenBLAS is needed for PySCF compiled binaries, and curl/ca-certificates/xz-utils for crest, julia for clapeyron, packmol and lammps for polymers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas-dev \
    xtb \
    curl \
    ca-certificates \
    xz-utils \
    julia \
    packmol \
    lammps \
    && rm -rf /var/lib/apt/lists/*

# Download and install CREST binary
RUN curl -fsSL -o /tmp/crest.tar.xz https://github.com/crest-lab/crest/releases/download/v3.0.2/crest-gnu-v3.0.2-x86_64.tar.xz && \
    tar -C /usr/local/bin -xf /tmp/crest.tar.xz && \
    chmod +x /usr/local/bin/crest && \
    rm /tmp/crest.tar.xz

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
