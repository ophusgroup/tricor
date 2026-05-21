"""Physics-feature variant of the shell_target-conditioned relaxml surrogate.

Drop-in replacement for ``tricor.relaxml.model_shelltgt`` that swaps every
``nn.Embedding(MAX_Z, dim)`` lookup for a shared MLP over a fixed
periodic-table feature table.  See ``model.py`` and ``species.py`` for
details.

Re-uses ``tricor.relaxml.data_shelltgt`` unchanged for the data layer.
"""

from .model import LitRelaxML, RelaxMLModel
from .species import (
    MAX_Z,
    N_FEATURES,
    SCALAR_COLUMNS,
    SpeciesEncoder,
    build_periodic_table_features,
)

__all__ = [
    "LitRelaxML",
    "RelaxMLModel",
    "SpeciesEncoder",
    "MAX_Z",
    "N_FEATURES",
    "SCALAR_COLUMNS",
    "build_periodic_table_features",
]
