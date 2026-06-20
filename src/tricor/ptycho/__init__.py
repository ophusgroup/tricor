"""Pseudo-ptychography → local-structure training data for tricor.

Build (input, target) pairs for learning local g2 / g3 from
ptychographic reconstructions:

* **target** — window-weighted, asymptote-to-1 g2 / g3 measured directly
  from the atomic structure (:mod:`tricor.ptycho.correlations`), over a
  soft Hann (xy) + Gaussian (z) window (:mod:`tricor.ptycho.weighting`).
* **input** — a blurred, depth-resolved projected potential in radians
  (:mod:`tricor.ptycho.potential`, requires the optional ``abtem``
  dependency — install with ``pip install -e '.[training]'``).

An :func:`~tricor.ptycho.widget.PtychoExplorer` anywidget ties the two
together for interactive inspection.

``correlations`` and ``weighting`` have no heavy dependencies; ``abtem``
is imported lazily only when the potential pipeline is used.
"""

from __future__ import annotations

from .correlations import LocalCorrelations, clear_random_cache, local_correlations
from .weighting import WindowSpec, gaussian_z, hann_window_xy, window_weights

__all__ = [
    "WindowSpec",
    "window_weights",
    "hann_window_xy",
    "gaussian_z",
    "local_correlations",
    "LocalCorrelations",
    "clear_random_cache",
]


def __getattr__(name: str):
    # Lazy access to the abtem-backed and widget modules so that
    # ``import tricor.ptycho`` stays light and abtem-free.
    if name in ("potential_stack", "PotentialStack", "blur_stack", "scattering_power"):
        from . import potential

        return getattr(potential, name)
    if name in ("PtychoExplorer",):
        from . import widget

        return getattr(widget, name)
    if name in ("plot_scattering_power", "plot_training_pair"):
        from . import plotting

        return getattr(plotting, name)
    if name in ("sliding_window_pairs", "TrainingPair"):
        from . import training

        return getattr(training, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
