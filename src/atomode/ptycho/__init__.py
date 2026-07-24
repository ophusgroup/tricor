"""Pseudo-ptychography → local-structure training data for atomode.

Build (input, target) pairs for learning local g2 / g3 from
ptychographic reconstructions:

* **target** — window-weighted, asymptote-to-1 g2 / g3 measured directly
  from the atomic structure (:mod:`atomode.ptycho.correlations`), over a
  soft circular-Hann (xy) + Gaussian (z) window
  (:mod:`atomode.ptycho.weighting`).
* **input** — a blurred, depth-resolved projected potential in radians
  (:mod:`atomode.ptycho.potential`, requires the optional ``abtem``
  dependency — install with ``pip install -e '.[training]'``).

An :func:`~atomode.ptycho.widget.PtychoExplorer` anywidget ties the two
together for interactive inspection.

``correlations`` and ``weighting`` have no heavy dependencies; ``abtem``
is imported lazily only when the potential pipeline is used.
"""

from __future__ import annotations

from .correlations import LocalCorrelations, clear_random_cache, local_correlations
from .weighting import (
    WindowSpec,
    gaussian_z,
    hann_radial,
    hann_window_xy,
    window_weights,
    windowed_image,
)

__all__ = [
    "WindowSpec",
    "window_weights",
    "windowed_image",
    "hann_radial",
    "hann_window_xy",
    "gaussian_z",
    "local_correlations",
    "LocalCorrelations",
    "clear_random_cache",
]


def __getattr__(name: str):
    # Lazy access to the abtem-backed and widget modules so that
    # ``import atomode.ptycho`` stays light and abtem-free.
    if name in ("potential_stack", "PotentialStack", "blur_stack", "scattering_power"):
        from . import potential

        return getattr(potential, name)
    if name in ("PtychoExplorer",):
        from . import widget

        return getattr(widget, name)
    if name in ("HRTEMExplorer",):
        from . import hrtem_widget

        return getattr(hrtem_widget, name)
    if name in ("plot_scattering_power", "plot_training_pair", "plot_hrtem_pair"):
        from . import plotting

        return getattr(plotting, name)
    if name in ("sliding_window_pairs", "TrainingPair"):
        from . import training

        return getattr(training, name)
    if name in ("graded_supercell",):
        from . import structures

        return getattr(structures, name)
    if name in (
        "ExitWaveStack", "exit_wave_stack", "ctf_image", "hrtem_image",
        "default_defocus", "radial_average", "hrtem_input", "hrtem_window_image",
    ):
        from . import hrtem

        return getattr(hrtem, name)
    if name in ("sliding_window_pairs_hrtem",):
        from . import hrtem_training

        return getattr(hrtem_training, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
