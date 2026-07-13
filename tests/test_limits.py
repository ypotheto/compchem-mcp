from ypotheto_compchem_mcp.utils.limits import cap_series


def test_cap_series_no_truncation_when_under_limit():
    values = list(range(50))
    decimated, truncated = cap_series(values, max_points=200)
    assert decimated == values
    assert truncated is False

def test_cap_series_truncates_oversized_input():
    values = list(range(1000))
    decimated, truncated = cap_series(values, max_points=200)
    assert truncated is True
    assert len(decimated) <= 200
    assert decimated[0] == 0
    assert decimated[-1] == 999

def test_cap_series_is_monotonic_and_uniform():
    values = list(range(500))
    decimated, _ = cap_series(values, max_points=100)
    assert decimated == sorted(decimated)
    assert len(set(decimated)) == len(decimated)

def test_cap_series_exact_boundary_not_truncated():
    values = list(range(200))
    decimated, truncated = cap_series(values, max_points=200)
    assert decimated == values
    assert truncated is False

def test_cap_series_empty_input():
    decimated, truncated = cap_series([], max_points=200)
    assert decimated == []
    assert truncated is False
