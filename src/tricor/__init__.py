"""Public package interface for tricor."""

from ._plotting import (
    export_g2_compare_html,
    export_overview_html,
    plot_g2_compare,
)
from .g3 import G3Distribution
from .shells import CoordinationShellTarget
from .supercell import Supercell

__all__ = [
    "CoordinationShellTarget",
    "G3Distribution",
    "Supercell",
    "__version__",
    "export_g2_compare_html",
    "export_overview_html",
    "plot_g2_compare",
]

__version__ = "0.1.0"
