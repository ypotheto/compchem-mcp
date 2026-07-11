from ypotheto_compchem_mcp.server import ping

def test_ping():
    result = ping()
    assert "pong" in result
    assert "0.1.0" in result
