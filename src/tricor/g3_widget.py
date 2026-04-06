"""Interactive anywidget explorer for g3 data."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import anywidget
import numpy as np
import traitlets

if TYPE_CHECKING:
    from .g3 import G3Distribution


_STATIC_DIR = Path(__file__).parent / "static"
_EPS = 1e-12


class G3PlotWidget(anywidget.AnyWidget):
    """Interactive two-panel explorer for angle-binned three-body data."""

    _esm = _STATIC_DIR / "g3_explorer.js"
    _css = _STATIC_DIR / "g3_explorer.css"

    triplet_labels = traitlets.List(trait=traitlets.Unicode()).tag(sync=True)
    triplet_index = traitlets.Int(0).tag(sync=True)
    normalize = traitlets.Bool(True).tag(sync=True)
    r = traitlets.List(trait=traitlets.Float()).tag(sync=True)
    r_edges = traitlets.List(trait=traitlets.Float()).tag(sync=True)
    phi_deg = traitlets.List(trait=traitlets.Float()).tag(sync=True)
    phi_edges_deg = traitlets.List(trait=traitlets.Float()).tag(sync=True)
    pair_profile = traitlets.List(trait=traitlets.Float()).tag(sync=True)
    slice_image = traitlets.List(trait=traitlets.Float()).tag(sync=True)
    slice_shape = traitlets.List(trait=traitlets.Int()).tag(sync=True)
    selection_min = traitlets.Float(0.0).tag(sync=True)
    selection_max = traitlets.Float(0.0).tag(sync=True)
    sigma_r = traitlets.Float(0.0).tag(sync=True)
    sigma_phi = traitlets.Float(0.0).tag(sync=True)
    slice_max = traitlets.Float(-1.0).tag(sync=True)
    status = traitlets.Unicode("").tag(sync=True)

    def __init__(
        self,
        distribution: "G3Distribution",
        *,
        triplet_index: int = 0,
        normalize: bool = True,
    ) -> None:
        if distribution.g3 is None:
            raise ValueError("Measure g3 before creating a plot widget.")
        if distribution.g3.ndim != 4:
            raise ValueError("The interactive widget expects g3 with shape (triplet, r, r, phi).")

        self._distribution = distribution
        self._suspend_callbacks = True
        super().__init__()

        self.triplet_labels = list(distribution.pair_labels)
        self.r = distribution.r.astype(float).tolist()
        self.r_edges = distribution.bin_edges.astype(float).tolist()
        self.phi_deg = distribution.phi_deg.astype(float).tolist()
        self.phi_edges_deg = np.rad2deg(distribution.phi_edges).astype(float).tolist()
        self.normalize = bool(normalize)
        self.sigma_r = 0.0
        self.sigma_phi = 0.0
        self.slice_max = -1.0
        self.triplet_index = int(np.clip(triplet_index, 0, len(self.triplet_labels) - 1))
        self._set_default_shell()
        self._update_payload()

        self.observe(self._on_triplet_index, names="triplet_index")
        self.observe(
            self._on_controls,
            names=["selection_min", "selection_max", "normalize", "sigma_r", "sigma_phi"],
        )
        self._suspend_callbacks = False

    def _on_triplet_index(self, _change: traitlets.Bunch) -> None:
        if self._suspend_callbacks:
            return
        self._suspend_callbacks = True
        self._set_default_shell()
        self._update_payload()
        self._suspend_callbacks = False

    def _on_controls(self, _change: traitlets.Bunch) -> None:
        if self._suspend_callbacks:
            return
        self._suspend_callbacks = True
        self._clamp_shell()
        self._update_payload()
        self._suspend_callbacks = False

    def _triplet_data(self) -> np.ndarray:
        return np.asarray(self._distribution.g3[self.triplet_index], dtype=np.float64)

    def _pair_profile_raw_from_distribution(self, distribution: "G3Distribution") -> np.ndarray:
        g2 = getattr(distribution, "g2", None)
        if g2 is not None and hasattr(distribution, "g3_index"):
            center_ind, neigh1_ind, neigh2_ind = distribution.g3_index[self.triplet_index]
            profile = np.asarray(g2[center_ind, neigh1_ind], dtype=np.float64)
            profile += np.asarray(g2[center_ind, neigh2_ind], dtype=np.float64)
            return profile
        triplet_data = np.asarray(distribution.g3[self.triplet_index], dtype=np.float64)
        return triplet_data.sum(axis=(1, 2)) + triplet_data.sum(axis=(0, 2))

    def _default_shell_profile_raw(self) -> np.ndarray:
        distribution = self._distribution
        g2 = getattr(distribution, "g2", None)
        if g2 is not None:
            return self._pair_profile_raw_from_distribution(distribution)

        source = getattr(distribution, "source_distribution", None)
        if source is not None and source.g3 is not None:
            return self._pair_profile_raw_from_distribution(source)
        return self._pair_profile_raw_from_distribution(distribution)

    def _pair_profile_raw(self, triplet_data: np.ndarray) -> np.ndarray:
        return self._pair_profile_raw_from_distribution(self._distribution)

    def _pair_profile_for_display(self, triplet_data: np.ndarray) -> np.ndarray:
        profile = self._pair_profile_raw(triplet_data)
        if not self.normalize:
            return self._smooth_profile_for_display(profile)

        radius = np.asarray(self._distribution.r, dtype=np.float64)
        profile = profile / np.maximum(radius * radius, _EPS)
        profile = self._smooth_profile_for_display(profile)
        tail = profile[self._tail_mask(radius)]
        finite = tail[np.isfinite(tail)]
        scale = float(np.mean(finite)) if finite.size else 1.0
        if scale <= _EPS:
            scale = 1.0
        return profile / scale

    def _slice_image_raw(self, triplet_data: np.ndarray) -> np.ndarray:
        shell_mask = self._shell_mask()
        image = triplet_data[shell_mask, :, :].sum(axis=0)
        image += triplet_data[:, shell_mask, :].sum(axis=1)
        return image.T

    def _slice_image_for_display(self, triplet_data: np.ndarray) -> np.ndarray:
        image = self._slice_image_raw(triplet_data)
        if not self.normalize:
            return self._smooth_image_for_display(image)

        radius = np.asarray(self._distribution.r, dtype=np.float64)
        phi_rad = np.deg2rad(np.asarray(self.phi_deg, dtype=np.float64))
        phi_factor = np.maximum(np.sin(phi_rad), 1e-3)[:, None]
        radial_factor = np.maximum(radius * radius, _EPS)[None, :]
        image = image / (phi_factor * radial_factor)
        image = self._smooth_image_for_display(image)
        tail = image[:, self._tail_mask(radius)]
        finite = tail[np.isfinite(tail)]
        scale = float(np.mean(finite)) if finite.size else 1.0
        if scale <= _EPS:
            scale = 1.0
        return image / scale

    def _tail_mask(self, radius: np.ndarray) -> np.ndarray:
        if radius.size == 0:
            return np.zeros(0, dtype=bool)
        target_r_max = getattr(self._distribution, "target_r_max", None)
        if target_r_max is not None:
            start = float(target_r_max)
        else:
            start = 0.7 * float(radius[-1] + self._distribution.r_step / 2.0)
        mask = radius >= start
        if not np.any(mask):
            mask[-max(1, radius.size // 4):] = True
        return mask

    def _safe_scale(self, values: np.ndarray) -> float:
        finite = values[np.isfinite(values)]
        positive = finite[finite > 0]
        if positive.size == 0:
            return 1.0
        return float(np.mean(positive))

    def _smooth_profile(self, profile: np.ndarray) -> np.ndarray:
        kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float64)
        kernel /= kernel.sum()
        return np.convolve(profile, kernel, mode="same")

    def _gaussian_kernel(self, sigma_bins: float) -> np.ndarray:
        if sigma_bins <= 1e-12:
            return np.array([1.0], dtype=np.float64)
        radius = max(1, int(np.ceil(3.0 * sigma_bins)))
        x = np.arange(-radius, radius + 1, dtype=np.float64)
        kernel = np.exp(-0.5 * (x / sigma_bins) ** 2)
        kernel /= np.sum(kernel)
        return kernel

    def _smooth_along_axis(
        self,
        values: np.ndarray,
        *,
        sigma_bins: float,
        axis: int,
    ) -> np.ndarray:
        if sigma_bins <= 1e-12:
            return values
        kernel = self._gaussian_kernel(sigma_bins)
        pad = kernel.size // 2
        pad_width = [(0, 0)] * values.ndim
        pad_width[axis] = (pad, pad)
        padded = np.pad(values, pad_width, mode="reflect")
        return np.apply_along_axis(
            lambda row: np.convolve(row, kernel, mode="valid"),
            axis,
            padded,
        )

    def _smooth_profile_for_display(self, profile: np.ndarray) -> np.ndarray:
        sigma_r_bins = float(self.sigma_r) / max(float(self._distribution.r_step), _EPS)
        return self._smooth_along_axis(profile, sigma_bins=sigma_r_bins, axis=0)

    def _smooth_image_for_display(self, image: np.ndarray) -> np.ndarray:
        sigma_r_bins = float(self.sigma_r) / max(float(self._distribution.r_step), _EPS)
        phi_step_deg = 180.0 / max(int(self._distribution.phi_num_bins), 1)
        sigma_phi_bins = float(self.sigma_phi) / max(phi_step_deg, _EPS)
        image = self._smooth_along_axis(image, sigma_bins=sigma_phi_bins, axis=0)
        image = self._smooth_along_axis(image, sigma_bins=sigma_r_bins, axis=1)
        return image

    def _set_default_shell(self) -> None:
        raw_profile = self._default_shell_profile_raw()
        profile = raw_profile.copy()
        if self.normalize:
            radius = np.asarray(self._distribution.r, dtype=np.float64)
            profile = profile / np.maximum(radius * radius, _EPS)
            profile = self._smooth_profile_for_display(profile)
            tail = profile[self._tail_mask(radius)]
            finite = tail[np.isfinite(tail)]
            scale = float(np.mean(finite)) if finite.size else 1.0
            if scale <= _EPS:
                scale = 1.0
            profile = profile / scale
        else:
            profile = self._smooth_profile_for_display(profile)
        smooth = self._smooth_profile(profile)
        finite = np.nan_to_num(smooth, nan=0.0, posinf=0.0, neginf=0.0)
        positive = np.flatnonzero(finite > 0)
        if positive.size == 0:
            left_bin = 0
            right_bin = min(1, len(self.r) - 1)
        else:
            start = int(positive[0])
            peak_bin = None
            for idx in range(max(start, 1), finite.size - 1):
                if finite[idx] >= finite[idx - 1] and finite[idx] > finite[idx + 1]:
                    peak_bin = idx
                    break
            if peak_bin is None:
                peak_bin = int(start + np.argmax(finite[start:]))

            threshold = max(1e-12, 0.01 * float(finite[peak_bin]))
            leading = np.flatnonzero(finite[:peak_bin + 1] > threshold)
            if leading.size:
                left_bin = int(leading[0])
            else:
                left_bin = start

            right_bin = min(finite.size - 1, peak_bin + 1)
            for idx in range(peak_bin + 1, finite.size - 1):
                if finite[idx] <= finite[idx - 1] and finite[idx] <= finite[idx + 1]:
                    right_bin = idx
                    break

        bin_edges = np.asarray(self._distribution.bin_edges, dtype=np.float64)
        self.selection_min = float(bin_edges[left_bin])
        self.selection_max = float(bin_edges[right_bin + 1])
        self._clamp_shell()

    def _clamp_shell(self) -> None:
        bin_edges = np.asarray(self._distribution.bin_edges, dtype=np.float64)
        min_edge = float(bin_edges[0])
        max_edge = float(bin_edges[-1])
        self.selection_min = float(np.clip(self.selection_min, min_edge, max_edge))
        self.selection_max = float(np.clip(self.selection_max, min_edge, max_edge))
        if self.selection_max <= self.selection_min:
            self.selection_max = min(max_edge, self.selection_min + self._distribution.r_step)

    def _shell_mask(self) -> np.ndarray:
        edges = np.asarray(self._distribution.bin_edges, dtype=np.float64)
        mask = (edges[:-1] < self.selection_max) & (edges[1:] > self.selection_min)
        if not np.any(mask):
            centers = np.asarray(self._distribution.r, dtype=np.float64)
            nearest = int(np.argmin(np.abs(centers - 0.5 * (self.selection_min + self.selection_max))))
            mask[nearest] = True
        return mask

    def _update_payload(self) -> None:
        triplet_data = self._triplet_data()
        pair_profile = self._pair_profile_for_display(triplet_data)
        slice_image = self._slice_image_for_display(triplet_data)
        shell_mask = self._shell_mask()
        shell_bins = np.flatnonzero(shell_mask)
        shell_label = (
            f"{self.selection_min:.2f}-{self.selection_max:.2f} A"
            f" ({shell_bins[0]}:{shell_bins[-1] + 1})"
        )
        self.pair_profile = pair_profile.astype(float).tolist()
        self.slice_image = slice_image.astype(float).ravel().tolist()
        self.slice_shape = list(slice_image.shape)
        self.status = f"{self.triplet_labels[self.triplet_index]} | shell {shell_label}"
