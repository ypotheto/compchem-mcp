from urllib.parse import urlparse

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

from ypotheto_compchem_mcp import __version__
from ypotheto_compchem_mcp.config import settings


def _build_transport_security() -> TransportSecuritySettings:
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

# Create FastMCP server
mcp = FastMCP("ypotheto-compchem", transport_security=_build_transport_security())

@mcp.tool()
def ping() -> str:
    """
    Check if the Ypotheto Computational Chemistry MCP Server is responsive.
    Use when verifying connection health.
    """
    return f"pong from ypotheto-compchem-mcp version {__version__}"

# Import modules below to register their tools on the mcp instance above.
# These are deliberate side-effect-only imports: each module's tool
# decorators register against the shared instance just by being imported,
# and nothing in this file references the imported names directly. Do not
# remove any of these, including via an automated lint auto-fix pass - one
# such pass previously deleted this entire block since it looked unused on
# a naive syntactic check, silently breaking every tool except the one
# defined directly above (see CHANGELOG.md, Phase 5). The lint suppression
# comments on each line below are load-bearing, not decorative.
from ypotheto_compchem_mcp.modules import builder_tools  # noqa: F401,E402,I001
from ypotheto_compchem_mcp.modules import cheminformatics_tools  # noqa: F401,E402
from ypotheto_compchem_mcp.modules import dynamics_tools  # noqa: F401,E402
from ypotheto_compchem_mcp.modules import ensemble_tools  # noqa: F401,E402
from ypotheto_compchem_mcp.modules import kinetics_tools  # noqa: F401,E402
from ypotheto_compchem_mcp.modules import mlff_tools  # noqa: F401,E402
from ypotheto_compchem_mcp.modules import periodic_tools  # noqa: F401,E402
from ypotheto_compchem_mcp.modules import polymer_tools  # noqa: F401,E402
from ypotheto_compchem_mcp.modules import quantum_tools  # noqa: F401,E402
from ypotheto_compchem_mcp.modules import scientific_preflight_tools  # noqa: F401,E402
from ypotheto_compchem_mcp.modules import solubility_tools  # noqa: F401,E402
from ypotheto_compchem_mcp.modules import thermo_tools  # noqa: F401,E402
from ypotheto_compchem_mcp.modules import vibrations_tools  # noqa: F401,E402
from ypotheto_compchem_mcp.modules import xtb_tools  # noqa: F401,E402
