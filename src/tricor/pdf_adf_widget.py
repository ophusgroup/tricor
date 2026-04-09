"""Interactive anywidget explorer for differentiable PDF g2(r) and ADF(phi) data.

Provides a two-panel visualization matching tricor's widget style:
  - Top panel: g2(r) pair distribution function for the species pairs
    associated with the selected triplet type
  - Bottom panel: ADF(phi) angular distribution for the selected triplet

Usage:
    calc = DifferentiablePDFADF(r_max=8.0, r_step=0.05, phi_num_bins=90,
                                 sigma_r=0.15, sigma_phi=0.1, species=[14])
    g2, adf = calc.compute(positions, species, cell)
    widget = PDFADFWidget(calc, g2, adf)
    widget  # display in notebook
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import anywidget
import numpy as np
import traitlets

if TYPE_CHECKING:
    import torch

_STATIC_DIR = Path(__file__).parent / "static"
_EPS = 1e-12


def _to_numpy(tensor_or_array) -> np.ndarray:
    """Convert a torch tensor or numpy array to a numpy array."""
    if hasattr(tensor_or_array, "detach"):
        return tensor_or_array.detach().cpu().numpy()
    return np.asarray(tensor_or_array)


class PDFADFWidget(anywidget.AnyWidget):
    """Interactive two-panel explorer for differentiable PDF and ADF data."""

    _esm = _STATIC_DIR / "pdf_adf_explorer.js"
    _css = _STATIC_DIR / "pdf_adf_explorer.css"

    triplet_labels = traitlets.List(trait=traitlets.Unicode()).tag(sync=True)
    triplet_index = traitlets.Int(0).tag(sync=True)
    normalize = traitlets.Bool(True).tag(sync=True)
    r = traitlets.List(trait=traitlets.Float()).tag(sync=True)
    phi_deg = traitlets.List(trait=traitlets.Float()).tag(sync=True)
    g2_profile = traitlets.List(trait=traitlets.Float()).tag(sync=True)
    g2_profile_2 = traitlets.List(trait=traitlets.Float()).tag(sync=True)
    adf_profile = traitlets.List(trait=traitlets.Float()).tag(sync=True)
    g2_label_1 = traitlets.Unicode("").tag(sync=True)
    g2_label_2 = traitlets.Unicode("").tag(sync=True)
    status = traitlets.Unicode("").tag(sync=True)

    def __init__(
        self,
        calculator,
        g2,
        adf,
        *,
        triplet_index: int = 0,
        normalize: bool = True,
    ) -> None:
        """Create the explorer widget.

        Args:
            calculator: A DifferentiablePDFADF instance (provides grid
                        definitions, species info, and label properties).
            g2: The g2 output tensor/array with shape
                (num_species, num_species, num_r).
            adf: The ADF output tensor/array with shape
                 (num_triplets, phi_num_bins).
            triplet_index: Initial triplet type to display.
            normalize: If True, show g(r) normalized by r^2 and ADF
                       normalized to a probability density.
        """
        self._calc = calculator
        self._g2 = _to_numpy(g2).astype(np.float64)
        self._adf = _to_numpy(adf).astype(np.float64)
        self._suspend = True
        super().__init__()

        self.triplet_labels = list(calculator.triplet_labels)
        self.r = _to_numpy(calculator.r_grid).astype(float).tolist()
        phi_rad = _to_numpy(calculator.phi_grid).astype(float)
        self.phi_deg = np.rad2deg(phi_rad).tolist()
        self.normalize = bool(normalize)
        self.triplet_index = int(
            np.clip(triplet_index, 0, max(len(self.triplet_labels) - 1, 0))
        )
        self._update_payload()

        self.observe(self._on_change, names=["triplet_index", "normalize"])
        self._suspend = False

    def _on_change(self, _change) -> None:
        if self._suspend:
            return
        self._suspend = True
        self._update_payload()
        self._suspend = False

    def _update_payload(self) -> None:
        calc = self._calc
        g3_index = _to_numpy(calc.g3_index)
        idx = self.triplet_index
        center_ind, neigh1_ind, neigh2_ind = g3_index[idx]

        r = np.asarray(self.r, dtype=np.float64)

        # g2 profiles for the two neighbor species of this triplet type
        g2_1 = self._g2[center_ind, neigh1_ind].copy()
        g2_2 = self._g2[center_ind, neigh2_ind].copy()

        if self.normalize:
            # Normalize by r^2 to get standard g(r) shape, then scale
            # so the long-range tail approaches 1
            g2_1 = self._normalize_g2(g2_1, r)
            g2_2 = self._normalize_g2(g2_2, r)

        # ADF profile
        adf_profile = self._adf[idx].copy()
        if self.normalize and adf_profile.sum() > _EPS:
            # Normalize to probability density
            dphi = np.deg2rad(self.phi_deg[1] - self.phi_deg[0]) if len(self.phi_deg) > 1 else 1.0
            adf_profile = adf_profile / (adf_profile.sum() * dphi)

        # Species labels
        pair_labels = list(calc.pair_labels)
        num_sp = calc.num_species
        label_1 = pair_labels[center_ind * num_sp + neigh1_ind]
        label_2 = pair_labels[center_ind * num_sp + neigh2_ind]

        self.g2_profile = g2_1.tolist()
        self.g2_profile_2 = g2_2.tolist()
        self.g2_label_1 = label_1
        self.g2_label_2 = label_2
        self.adf_profile = adf_profile.tolist()

        triplet_label = self.triplet_labels[idx]
        self.status = f"{triplet_label} | g2: {label_1}"
        if label_1 != label_2:
            self.status += f", {label_2}"

    @staticmethod
    def _normalize_g2(g2: np.ndarray, r: np.ndarray) -> np.ndarray:
        """Normalize raw g2 counts by r^2 and scale tail to ~1."""
        g2_norm = g2 / np.maximum(r ** 2, _EPS)
        # Use the outer 30% as the "tail" for scaling
        tail_start = int(0.7 * len(r))
        tail = g2_norm[tail_start:]
        finite = tail[np.isfinite(tail) & (tail > 0)]
        scale = float(np.mean(finite)) if finite.size else 1.0
        if scale <= _EPS:
            scale = 1.0
        return g2_norm / scale

    def update_data(self, g2, adf) -> None:
        """Update the displayed data (e.g., after re-computing with new positions).

        Args:
            g2: New g2 tensor/array, same shape as original.
            adf: New ADF tensor/array, same shape as original.
        """
        self._g2 = _to_numpy(g2).astype(np.float64)
        self._adf = _to_numpy(adf).astype(np.float64)
        self._suspend = True
        self._update_payload()
        self._suspend = False
