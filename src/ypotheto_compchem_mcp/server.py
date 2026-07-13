from dataclasses import dataclass
from urllib.parse import urlparse

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

from ypotheto_compchem_mcp import __version__
from ypotheto_compchem_mcp.config import Settings


def _build_transport_security(settings: Settings) -> TransportSecuritySettings:
    """FastMCP constructed without explicit transport security auto-enables
    DNS-rebinding protection with a localhost-only Host allowlist - correct for
    the local-dev threat model it targets, but it 421s every request that
    carries a real public hostname, so a hosted deployment must allowlist its
    own public host explicitly. Derived from COMPCHEM_PUBLIC_BASE_URL (which a
    hosted deployment must set anyway, for artifact URLs); localhost stays
    allowed so local runs and tests behave as before."""
    allowed_hosts = ["localhost", "localhost:*", "127.0.0.1", "127.0.0.1:*"]
    allowed_origins = ["http://localhost:*", "http://127.0.0.1:*"]
    if settings.public_base_url:
        parsed = urlparse(settings.public_base_url)
        if parsed.netloc:
            allowed_hosts.append(parsed.netloc)
            allowed_origins.append(f"{parsed.scheme}://{parsed.netloc}")
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=allowed_hosts,
        allowed_origins=allowed_origins,
    )


def ping() -> str:
    """
    Check if the Ypotheto Computational Chemistry MCP Server is responsive.
    Use when verifying connection health.
    """
    return f"pong from ypotheto-compchem-mcp version {__version__}"


@dataclass
class ServerBundle:
    mcp: FastMCP
    settings: Settings


def create_server(settings: Settings) -> ServerBundle:
    """Build a fresh FastMCP server instance with every tool/prompt module
    registered against it - explicit dependency injection instead of a
    module-global `mcp` + import-side-effect registration (see CHANGELOG.md,
    Phase 8). Each caller (cli.py, http_app.py, tests) gets its own
    independent instance rather than sharing hidden global state; nothing
    stops two callers from passing the same `settings` singleton (the normal
    case for the real running server), but nothing requires it either."""
    mcp = FastMCP("ypotheto-compchem", transport_security=_build_transport_security(settings))
    mcp.tool()(ping)

    from ypotheto_compchem_mcp.modules import (
        advisor_tools,
        builder_tools,
        cheminformatics_tools,
        dynamics_tools,
        ensemble_tools,
        kinetics_tools,
        mlff_tools,
        periodic_tools,
        polymer_tools,
        quantum_tools,
        scientific_preflight_tools,
        solubility_tools,
        thermo_tools,
        vibrations_tools,
        xtb_tools,
    )

    advisor_tools.register_advisor_tools(mcp)
    builder_tools.register_builder_tools(mcp)
    cheminformatics_tools.register_cheminformatics_tools(mcp)
    dynamics_tools.register_dynamics_tools(mcp)
    ensemble_tools.register_ensemble_tools(mcp)
    kinetics_tools.register_kinetics_tools(mcp)
    mlff_tools.register_mlff_tools(mcp)
    periodic_tools.register_periodic_tools(mcp)
    polymer_tools.register_polymer_tools(mcp)
    quantum_tools.register_quantum_tools(mcp)
    scientific_preflight_tools.register_scientific_preflight_tools(mcp)
    solubility_tools.register_solubility_tools(mcp)
    thermo_tools.register_thermo_tools(mcp)
    vibrations_tools.register_vibrations_tools(mcp)
    xtb_tools.register_xtb_tools(mcp)

    return ServerBundle(mcp=mcp, settings=settings)
