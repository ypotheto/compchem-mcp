import matplotlib

from ypotheto_compchem_mcp.utils.plotting import (
    close_all_open_figures,
    new_3d_figure,
    new_figure,
    render_png,
)


def test_agg_backend_is_set():
    assert matplotlib.get_backend().lower() == "agg"


def test_new_figure_and_render_png():
    fig, ax = new_figure()
    ax.plot([0, 1], [0, 1])
    png_bytes = render_png(fig)
    assert isinstance(png_bytes, bytes)
    assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"


def test_new_3d_figure():
    fig, ax = new_3d_figure()
    assert ax.name == "3d"
    render_png(fig)


def test_close_all_open_figures_clears_registry():
    import matplotlib.pyplot as plt
    new_figure()
    new_figure()
    assert len(plt.get_fignums()) >= 2
    close_all_open_figures()
    assert plt.get_fignums() == []
