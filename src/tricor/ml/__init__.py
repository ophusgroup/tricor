"""Machine-learning acceleration for ``Supercell.generate``.

A per-material EGNN learns to map a Voronoi-tiled atomic
configuration directly to the FIRE-quenched final positions,
conditioned on the ``grain_size`` regime parameter.  At inference
the user gets a single forward pass instead of hundreds of FIRE
iterations — ~100–500× faster on cells big enough to benefit (≥ 50 Å
boxes).

Public surface
--------------

>>> from tricor.ml import predict_positions, load_model
>>> model = load_model("sio2")
>>> new_positions = predict_positions(
...     model,
...     voronoi_positions=tile_xyz,
...     species_idx=species,
...     box_dim=(50., 50., 50.),
...     grain_size=20.0,
... )

End-to-end via the existing ``Supercell.generate`` API is exposed by
the ``backend="ml"`` / ``"ml+fire"`` flag (see
``src/tricor/supercell.py``).
"""

from __future__ import annotations

try:
    import torch  # noqa: F401
    HAS_TORCH = True
except ImportError:  # pragma: no cover
    HAS_TORCH = False

if HAS_TORCH:
    from .egnn import EGNN, EGNNLayer
    from .inference import load_model, predict_positions

    __all__ = [
        "EGNN",
        "EGNNLayer",
        "load_model",
        "predict_positions",
        "HAS_TORCH",
    ]
else:  # pragma: no cover
    __all__ = ["HAS_TORCH"]
