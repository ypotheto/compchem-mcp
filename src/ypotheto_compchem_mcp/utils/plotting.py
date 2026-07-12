import io
from typing import Any, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "font.size": 10,
        "font.family": "sans-serif",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)


def new_figure(nrows: int = 1, ncols: int = 1, figsize: Tuple[float, float] = (7.0, 4.5)) -> Any:
    """Return `(fig, ax)` (or `(fig, axes_array)` for nrows*ncols > 1) via
    `plt.subplots`. Centralizing figure creation here keeps every other module from
    needing to import `matplotlib.pyplot` directly (and thus from racing to set the backend)."""
    return plt.subplots(nrows, ncols, figsize=figsize)


def new_3d_figure(figsize: Tuple[float, float] = (6.5, 5.0)) -> Tuple[Any, Any]:
    """Return `(fig, ax)` with `ax` a 3D-projection Axes, for surface/wireframe plots."""
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection="3d")
    return fig, ax


def render_png(fig: Any) -> bytes:
    """Serialize a figure to PNG bytes and close it."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    return buf.getvalue()


def close_all_open_figures() -> None:
    """
    Safety net for a plotting tool that raises between figure creation and
    render_png (the only place a figure normally gets closed) - matplotlib
    keeps every created figure registered globally until explicitly closed,
    so an uncaught exception there would otherwise leak memory for the life
    of the server process. Called from mcp_tool_decorator's exception
    handlers, which run for every tool regardless of whether it plots.
    """
    plt.close("all")
