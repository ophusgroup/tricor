"""Per-edge shell_target injection variant of the relaxml surrogate.

This package supersedes ``tricor.relaxml.shelltgt_phys`` for cases
where shell_target conditioning needs to actually influence
predictions (cross-composition, cross-polymorph, cross-coordination
tests).  See ``model.py`` for the architectural rationale and
``edge_features.py`` for the per-edge feature schema.

Re-uses ``tricor.relaxml.data_shelltgt`` unchanged.
"""

from .edge_features import (
    COORD_SCALE,
    NUM_PER_EDGE_FEATURES,
    SIGMA_SCALE,
    TARGET_R_SCALE,
    build_per_edge_shell_target,
)
from .model import LitRelaxML, RelaxMLModel

__all__ = [
    "LitRelaxML",
    "RelaxMLModel",
    "build_per_edge_shell_target",
    "NUM_PER_EDGE_FEATURES",
    "TARGET_R_SCALE",
    "SIGMA_SCALE",
    "COORD_SCALE",
]
