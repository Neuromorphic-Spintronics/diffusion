"""Resource lifecycle checks for plotting helpers."""

import matplotlib
import matplotlib.pyplot as plt
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from diffusionsde.plotting import configure_matplotlib, save_figure


def test_latex_probe_failure_restores_backend_and_clears_probe(monkeypatch):
    original_backend = matplotlib.get_backend()
    existing_figure, _ = plt.subplots()
    original_figures = set(plt.get_fignums())
    cleared = []
    original_clear = Figure.clear

    def fail_draw(self):
        raise RuntimeError("LaTeX unavailable")

    def capture_clear(self, *args, **kwargs):
        cleared.append(self)
        return original_clear(self, *args, **kwargs)

    monkeypatch.setattr(FigureCanvasAgg, "draw", fail_draw)
    monkeypatch.setattr(Figure, "clear", capture_clear)

    assert configure_matplotlib(use_latex=True) is False

    assert matplotlib.get_backend() == original_backend
    assert set(plt.get_fignums()) == original_figures
    assert cleared
    assert cleared[-1] is not existing_figure
    assert matplotlib.rcParams["text.usetex"] is False
    plt.close(existing_figure)


def test_saved_figure_closes_even_when_write_fails(monkeypatch, tmp_path):
    fig, _ = plt.subplots()
    figure_number = fig.number

    def fail_save(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(fig, "savefig", fail_save)
    with pytest.raises(OSError, match="disk full"):
        save_figure(fig, tmp_path / "failed.pdf")

    assert figure_number not in plt.get_fignums()
