

def cap_series(values: list, max_points: int = 200) -> tuple[list, bool]:
    """Uniformly decimate `values` to at most `max_points` entries, always
    keeping the first and last point. Returns `(decimated, was_truncated)`.

    Policy: full data belongs in an artifact; the inline envelope only carries
    a bounded preview plus the artifact link, so a user-controlled `steps`/
    length parameter can't blow up the inline response size."""
    n = len(values)
    if n <= max_points:
        return list(values), False
    if max_points <= 1:
        return [values[0]], True
    step = (n - 1) / (max_points - 1)
    indices = sorted({round(i * step) for i in range(max_points)})
    return [values[i] for i in indices], True
