from mcp.server.fastmcp import FastMCP
from ypotheto_compchem_mcp import __version__

# Create FastMCP server
mcp = FastMCP("ypotheto-compchem")

@mcp.tool()
def ping() -> str:
    """
    Check if the Ypotheto Computational Chemistry MCP Server is responsive.
    Use when verifying connection health.
    """
    return f"pong from ypotheto-compchem-mcp version {__version__}"

# Import modules to register their tools on the mcp instance
from ypotheto_compchem_mcp.modules import builder_tools
from ypotheto_compchem_mcp.modules import cheminformatics_tools
from ypotheto_compchem_mcp.modules import quantum_tools
from ypotheto_compchem_mcp.modules import vibrations_tools
from ypotheto_compchem_mcp.modules import dynamics_tools
from ypotheto_compchem_mcp.modules import periodic_tools
from ypotheto_compchem_mcp.modules import polymer_tools
from ypotheto_compchem_mcp.modules import solubility_tools
from ypotheto_compchem_mcp.modules import scientific_preflight_tools
from ypotheto_compchem_mcp.modules import xtb_tools
from ypotheto_compchem_mcp.modules import ensemble_tools
from ypotheto_compchem_mcp.modules import thermo_tools
from ypotheto_compchem_mcp.modules import kinetics_tools
from ypotheto_compchem_mcp.modules import mlff_tools

