from ypotheto_compchem_mcp import __version__
from ypotheto_compchem_mcp.server import ping


def test_ping():
    result = ping()
    assert "pong" in result
    assert __version__ in result
