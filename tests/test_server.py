from ypotheto_compchem_mcp.server import ping
from ypotheto_compchem_mcp import __version__

def test_ping():
    result = ping()
    assert "pong" in result
    assert __version__ in result
