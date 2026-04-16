"""Three-body distribution measurement and target construction."""

from __future__ import annotations

import sys
from typing import Any

import numpy as np
from ase.atoms import Atoms
from ase.data import chemical_symbols


_EPS = 1e-12


class _TextProgressBar:
    """Minimal text progress bar for long-running measurements."""

    def __init__(self, total: int, *, label: str = "Progress", width: int = 28) -> None:
        self.total = max(int(total), 1)
        self.label = label
        self.width = max(int(width), 10)
        self.current = 0
        self._last_units = -1

    def update(self, current: int) -> None:
        """Advance the bar to the requested current step."""
        self.current = int(np.clip(current, 0, self.total))
        filled = int(round(self.width * self.current / self.total))
        if filled == self._last_units and self.current < self.total:
            return
        self._last_units = filled
        bar = "#" * filled + "-" * (self.width - filled)
        percent = 100.0 * self.current / self.total
        print(
            f"\r{self.label}: [{bar}] {self.current}/{self.total} ({percent:5.1f}%)",
            end="",
            file=sys.stdout,
            flush=True,
        )

    def close(self) -> None:
        """Finish the bar and move to the next line."""
        if self.current < self.total:
            self.update(self.total)
        print(file=sys.stdout, flush=True)


def _resolve_optional_alias(
    value: float | None,
    alias: float | None,
    *,
    name: str,
    alias_name: str,
) -> float | None:
    if value is None:
        return alias
    if alias is None:
        return value
    if not np.isclose(value, alias):
        raise ValueError(f"Received conflicting values for {name} and {alias_name}.")
    return value


class G3Distribution:
    """Container for measured or transformed rooted three-body distributions."""

    def __init__(
        self,
        source: Atoms | "G3Distribution",
        r_step: float | None = None,
        r_max: float | None = None,
        *,
        target_r_min: float | None = None,
        target_r_max: float | None = None,
        r_sigma: float | None = None,
        r_sigma_at: float | None = None,
        phi_sigma_deg: float | None = None,
        label: str | None = None,
        r_min: float | None = None,
        blur_sigma: float | None = None,
        **kwargs: Any,
    ) -> None:
        """Create a measured distribution from atoms or a target from another distribution.

        Parameters
        ----------
        source
            Either the reference `ase.Atoms` object to measure, or an existing
            `G3Distribution` that will be transformed into a target.
        r_step, r_max
            Optional measurement grid settings. These are typically supplied to
            `measure_g3()`, but are accepted here for convenience and backward
            compatibility.
        target_r_min, target_r_max
            Transition window used when constructing a target distribution.
        r_sigma, r_sigma_at, phi_sigma_deg
            Optional blur settings used only when constructing a target
            distribution.
        label
            Human-readable label used in reprs and interactive plots.
        r_min, blur_sigma
            Legacy aliases retained for compatibility with older notebooks.
        **kwargs
            Additional metadata stored on the distribution for future use.
        """
        self.label = label or "g3"
        self.metadata: dict[str, Any] = dict(kwargs)
        self.source_distribution: G3Distribution | None = None
        self.atoms: Atoms | None = None
        self.g3: np.ndarray | None = None
        self.g2: np.ndarray | None = None
        self.summary: dict[str, Any] = {}
        self.history: list[dict[str, Any]] = []
        self.is_target = False
        self.target_r_min = _resolve_optional_alias(
            target_r_min,
            r_min,
            name="target_r_min",
            alias_name="r_min",
        )
        self.target_r_max = None if target_r_max is None else float(target_r_max)
        self.r_sigma = _resolve_optional_alias(
            r_sigma,
            blur_sigma,
            name="r_sigma",
            alias_name="blur_sigma",
        )
        self.r_sigma_at = None if r_sigma_at is None else float(r_sigma_at)
        self.phi_sigma_deg = None if phi_sigma_deg is None else float(phi_sigma_deg)
        # Legacy alias retained for backward compatibility with existing notebooks.
        self.blur_sigma = self.r_sigma

        if isinstance(source, G3Distribution):
            self._init_from_distribution(source, r_step=r_step, r_max=r_max)
        elif isinstance(source, Atoms):
            self._init_from_atoms(source, r_step=r_step, r_max=r_max)
        else:
            raise TypeError(
                "G3Distribution expects an ase.Atoms object or another G3Distribution."
            )

    def _init_from_atoms(
        self,
        atoms: Atoms,
        *,
        r_step: float | None,
        r_max: float | None,
    ) -> None:
        self.atoms = atoms.copy()
        self.r_step = None if r_step is None else float(r_step)
        self.r_max = None if r_max is None else float(r_max)
        self.num_r = None
        self.r = None
        self.r_num = None
        self.bin_edges = None
        self.bin_centers = None
        self.phi_num_bins = None
        self.phi_edges = None
        self.phi_step = None
        self.phi = None
        self.phi_deg = None
        self.species = np.unique(self.atoms.numbers)
        self.species_pairs: list[Any] = []
        self.pair_labels: list[str] = []
        self.g2_labels: list[str] = []
        self.num_sites = len(self.atoms)
        self.num_species = int(self.species.size)
        self.history.append(
            {
                "event": "initialized",
                "kind": "measured",
                "num_atoms": len(self.atoms),
            }
        )

    def _init_from_distribution(
        self,
        distribution: "G3Distribution",
        *,
        r_step: float | None,
        r_max: float | None,
    ) -> None:
        self.source_distribution = distribution
        self.atoms = distribution.atoms.copy() if distribution.atoms is not None else None
        self.r_step = distribution.r_step if r_step is None else float(r_step)
        self.r_max = distribution.r_max if r_max is None else float(r_max)
        self.num_r = getattr(distribution, "num_r", None)
        self.bin_edges = (
            None if getattr(distribution, "bin_edges", None) is None
            else distribution.bin_edges.copy()
        )
        self.bin_centers = (
            None if getattr(distribution, "bin_centers", None) is None
            else distribution.bin_centers.copy()
        )
        self.species = np.array(distribution.species, copy=True)
        self.species_pairs = list(distribution.species_pairs)
        self.pair_labels = list(distribution.pair_labels)
        self.g2_labels = list(getattr(distribution, "g2_labels", []))
        self.r = None if getattr(distribution, "r", None) is None else distribution.r.copy()
        self.r_num = getattr(distribution, "r_num", self.num_r)
        self.g2 = (
            None
            if getattr(distribution, "g2", None) is None
            else np.array(distribution.g2, copy=True)
        )
        if getattr(distribution, "phi_num_bins", None) is not None:
            self.phi_num_bins = distribution.phi_num_bins
            self.phi_edges = distribution.phi_edges.copy()
            self.phi_step = distribution.phi_step
            self.phi = distribution.phi.copy()
            self.phi_deg = distribution.phi_deg.copy()
        else:
            self.phi_num_bins = None
            self.phi_edges = None
            self.phi_step = None
            self.phi = None
            self.phi_deg = None
        if hasattr(distribution, "g3_index"):
            self.g3_index = np.array(distribution.g3_index, copy=True)
        if hasattr(distribution, "g3_lookup"):
            self.g3_lookup = np.array(distribution.g3_lookup, copy=True)
        self.is_target = True

        if self.r_step is None or self.r_max is None:
            raise ValueError("Source distribution must define r_step and r_max before targeting.")
        if not np.isclose(self.r_step, distribution.r_step):
            raise ValueError("Target distributions must currently reuse the source r_step.")
        if not np.isclose(self.r_max, distribution.r_max):
            raise ValueError("Target distributions must currently reuse the source r_max.")
        if self.target_r_max is None:
            self.target_r_max = float(self.r_max)
        self._validate_target_parameters()

        self._ensure_source_g3()
        if distribution.g3 is not None:
            self.g3 = self._make_target_array(distribution.g3)
            if distribution.g2 is not None:
                self.g2 = self._make_target_g2_array(distribution.g2)
            self.summary = dict(distribution.summary)
            self.summary.update(
                {
                    "kind": "target",
                    "target_r_min": self.target_r_min,
                    "target_r_max": self.target_r_max,
                    "r_sigma": self.r_sigma,
                    "r_sigma_at": self.r_sigma_at,
                    "phi_sigma_deg": self.phi_sigma_deg,
                }
            )

        self.history.extend(distribution.history)
        self.history.append(
            {
                "event": "initialized",
                "kind": "target",
                "target_r_min": self.target_r_min,
                "target_r_max": self.target_r_max,
                "r_sigma": self.r_sigma,
                "r_sigma_at": self.r_sigma_at,
                "phi_sigma_deg": self.phi_sigma_deg,
            }
        )

    def _ensure_source_g3(self) -> None:
        if self.source_distribution is None or self.source_distribution.g3 is not None:
            return
        if self.source_distribution.r_step is None or self.source_distribution.r_max is None:
            raise ValueError(
                "Measure the source distribution before constructing a target distribution."
            )
        self.source_distribution.measure_g3()

    def _validate_target_parameters(self) -> None:
        if self.target_r_min is not None:
            self.target_r_min = float(self.target_r_min)
            if self.target_r_min < 0:
                raise ValueError("target_r_min must be non-negative.")
        if self.target_r_max is not None:
            self.target_r_max = float(self.target_r_max)
            if self.target_r_max <= 0:
                raise ValueError("target_r_max must be positive.")
            if self.r_max is not None and self.target_r_max > float(self.r_max) + _EPS:
                raise ValueError("target_r_max must lie within the measured radial grid.")
        if self.target_r_min is not None and self.target_r_max is not None:
            if self.target_r_min > self.target_r_max + _EPS:
                raise ValueError("target_r_min must be less than or equal to target_r_max.")
        if self.r_sigma is not None and self.r_sigma < 0:
            raise ValueError("r_sigma must be non-negative.")
        if self.r_sigma_at is not None and self.r_sigma_at <= 0:
            raise ValueError("r_sigma_at must be positive.")
        if self.phi_sigma_deg is not None and self.phi_sigma_deg < 0:
            raise ValueError("phi_sigma_deg must be non-negative.")

    def _scaled_sigma_at_radius(
        self,
        base_sigma: float | None,
        radius: float,
    ) -> float:
        if base_sigma is None or base_sigma <= _EPS:
            return 0.0
        radius = max(float(radius), 0.0)
        if self.r_sigma_at is None:
            return float(base_sigma) * radius
        return float(base_sigma) * radius / max(float(self.r_sigma_at), _EPS)

    def _gaussian_kernel(self, sigma_bins: float) -> np.ndarray:
        if sigma_bins <= _EPS:
            return np.array([1.0], dtype=np.float64)
        radius = max(1, int(np.ceil(3.0 * sigma_bins)))
        x = np.arange(-radius, radius + 1, dtype=np.float64)
        kernel = np.exp(-0.5 * (x / sigma_bins) ** 2)
        kernel /= np.sum(kernel)
        return kernel

    def _convolve_reflect_axis(
        self,
        values: np.ndarray,
        *,
        kernel: np.ndarray,
        axis: int,
    ) -> np.ndarray:
        if kernel.size == 1:
            return values
        pad = kernel.size // 2
        pad_width = [(0, 0)] * values.ndim
        pad_width[axis] = (pad, pad)
        padded = np.pad(values, pad_width, mode="reflect")
        return np.apply_along_axis(
            lambda row: np.convolve(row, kernel, mode="valid"),
            axis,
            padded,
        )

    def _radial_blur_kernel(self) -> np.ndarray:
        radii = np.asarray(self.bin_centers, dtype=np.float64)
        kernel = np.zeros((radii.size, radii.size), dtype=np.float64)
        for ind, radius in enumerate(radii):
            sigma = self._scaled_sigma_at_radius(self.r_sigma, radius)
            if sigma <= _EPS:
                kernel[ind, ind] = 1.0
                continue
            row = np.exp(-0.5 * ((radii - radius) / sigma) ** 2)
            row_sum = np.sum(row)
            if row_sum <= _EPS:
                kernel[ind, ind] = 1.0
            else:
                kernel[ind] = row / row_sum
        return kernel

    def _blur_phi_reduced(self, reduced_g3: np.ndarray) -> np.ndarray:
        if self.phi_sigma_deg is None or self.phi_sigma_deg <= _EPS:
            return reduced_g3
        phi_step_deg = np.rad2deg(float(self.phi_step))
        radii = np.asarray(self.bin_centers, dtype=np.float64)
        max_radius_index = np.maximum.outer(
            np.arange(radii.size, dtype=np.intp),
            np.arange(radii.size, dtype=np.intp),
        )
        blurred = np.empty_like(reduced_g3)

        for radius_index, radius in enumerate(radii):
            mask = max_radius_index == radius_index
            sigma_deg = self._scaled_sigma_at_radius(self.phi_sigma_deg, radius)
            kernel = self._gaussian_kernel(sigma_deg / max(phi_step_deg, _EPS))
            selected = reduced_g3[:, mask, :]
            blurred[:, mask, :] = self._convolve_reflect_axis(
                selected,
                kernel=kernel,
                axis=2,
            )

        return blurred

    def _blur_r_reduced(self, reduced_g3: np.ndarray) -> np.ndarray:
        if self.r_sigma is None or self.r_sigma <= _EPS:
            return reduced_g3
        kernel = self._radial_blur_kernel()
        blurred = np.einsum("ai,tijp->tajp", kernel, reduced_g3, optimize=True)
        return np.einsum("bj,tajp->tabp", kernel, blurred, optimize=True)

    def _ideal_density_shape(self) -> np.ndarray:
        radii = np.asarray(self.bin_centers, dtype=np.float64)
        phi_weight = np.maximum(np.sin(np.asarray(self.phi, dtype=np.float64)), _EPS)
        return (
            np.square(radii)[:, None, None]
            * np.square(radii)[None, :, None]
            * phi_weight[None, None, :]
        )

    def _ideal_g2_shape(self) -> np.ndarray:
        radii = np.asarray(self.bin_centers, dtype=np.float64)
        return np.square(radii)

    def _far_field_mask(self) -> np.ndarray:
        radii = np.asarray(self.bin_centers, dtype=np.float64)
        r01, r02 = np.meshgrid(radii, radii, indexing="ij")
        r_eff = np.maximum(r01, r02)
        if self.target_r_max is not None and self.target_r_min is not None:
            start = float(self.target_r_max)
        else:
            start = 0.75 * float(self.r_max)
        mask = r_eff >= start
        if not np.any(mask):
            start = radii[max(0, radii.size - max(1, radii.size // 4))]
            mask = r_eff >= start
        if not np.any(mask):
            mask = np.ones_like(r_eff, dtype=bool)
        return mask

    def _ideal_amplitudes(self, source_g3: np.ndarray) -> np.ndarray:
        shape = self._ideal_density_shape()
        radial_mask = self._far_field_mask()
        tail_mask = np.broadcast_to(radial_mask[:, :, None], shape.shape)
        shape_tail_mean = float(np.mean(shape[tail_mask]))
        shape_mean = float(np.mean(shape))
        amplitudes = np.empty(source_g3.shape[0], dtype=np.float64)

        for triplet_ind in range(source_g3.shape[0]):
            channel = np.asarray(source_g3[triplet_ind], dtype=np.float64)
            amplitude = float(np.mean(channel[tail_mask])) / max(shape_tail_mean, _EPS)
            if not np.isfinite(amplitude) or amplitude <= _EPS:
                amplitude = float(np.mean(channel)) / max(shape_mean, _EPS)
            if not np.isfinite(amplitude) or amplitude <= _EPS:
                amplitude = 1.0
            amplitudes[triplet_ind] = amplitude

        return amplitudes

    def _ideal_g3_raw(self, source_g3: np.ndarray) -> np.ndarray:
        shape = self._ideal_density_shape()
        amplitudes = self._ideal_amplitudes(source_g3)
        return amplitudes[:, None, None, None] * shape[None, :, :, :]

    def _ideal_pair_amplitudes(self, source_g2: np.ndarray) -> np.ndarray:
        shape = self._ideal_g2_shape()
        radii = np.asarray(self.bin_centers, dtype=np.float64)
        if self.target_r_max is not None:
            mask = radii >= float(self.target_r_max)
        else:
            mask = radii >= 0.75 * float(self.r_max)
        if not np.any(mask):
            mask[-max(1, radii.size // 4):] = True
        shape_tail_mean = float(np.mean(shape[mask]))
        shape_mean = float(np.mean(shape))
        amplitudes = np.empty(source_g2.shape[:2], dtype=np.float64)

        for ind0 in range(source_g2.shape[0]):
            for ind1 in range(source_g2.shape[1]):
                channel = np.asarray(source_g2[ind0, ind1], dtype=np.float64)
                amplitude = float(np.mean(channel[mask])) / max(shape_tail_mean, _EPS)
                if not np.isfinite(amplitude) or amplitude <= _EPS:
                    amplitude = float(np.mean(channel)) / max(shape_mean, _EPS)
                if not np.isfinite(amplitude) or amplitude <= _EPS:
                    amplitude = 1.0
                amplitudes[ind0, ind1] = amplitude

        return amplitudes

    def _ideal_g2_raw(self, source_g2: np.ndarray) -> np.ndarray:
        shape = self._ideal_g2_shape()
        amplitudes = self._ideal_pair_amplitudes(source_g2)
        return amplitudes[:, :, None] * shape[None, None, :]

    def _reduce_g3(self, raw_g3: np.ndarray, ideal_g3: np.ndarray) -> np.ndarray:
        return np.asarray(raw_g3, dtype=np.float64) / np.maximum(ideal_g3, _EPS)

    def _unreduce_g3(self, reduced_g3: np.ndarray, ideal_g3: np.ndarray) -> np.ndarray:
        return (np.asarray(reduced_g3, dtype=np.float64) * ideal_g3).astype(np.float32)

    def _reduce_g2(self, raw_g2: np.ndarray, ideal_g2: np.ndarray) -> np.ndarray:
        return np.asarray(raw_g2, dtype=np.float64) / np.maximum(ideal_g2, _EPS)

    def _unreduce_g2(self, reduced_g2: np.ndarray, ideal_g2: np.ndarray) -> np.ndarray:
        return (np.asarray(reduced_g2, dtype=np.float64) * ideal_g2).astype(np.float32)

    def _target_mix(self) -> np.ndarray:
        if self.target_r_min is None or self.target_r_max is None:
            return np.zeros((self.r_num, self.r_num, 1), dtype=np.float64)
        radii = np.asarray(self.bin_centers, dtype=np.float64)
        r01, r02 = np.meshgrid(radii, radii, indexing="ij")
        r_eff = np.maximum(r01, r02)
        width = max(float(self.target_r_max) - float(self.target_r_min), float(self.r_step))
        scaled = np.clip((r_eff - float(self.target_r_min)) / width, 0.0, 1.0)
        mix = scaled * scaled * (3.0 - 2.0 * scaled)
        return mix[:, :, None]

    def _target_mix_1d(self) -> np.ndarray:
        if self.target_r_min is None or self.target_r_max is None:
            return np.zeros(self.r_num, dtype=np.float64)
        radii = np.asarray(self.bin_centers, dtype=np.float64)
        width = max(float(self.target_r_max) - float(self.target_r_min), float(self.r_step))
        scaled = np.clip((radii - float(self.target_r_min)) / width, 0.0, 1.0)
        return scaled * scaled * (3.0 - 2.0 * scaled)

    def _blur_r_reduced_g2(self, reduced_g2: np.ndarray) -> np.ndarray:
        if self.r_sigma is None or self.r_sigma <= _EPS:
            return reduced_g2
        kernel = self._radial_blur_kernel()
        return np.einsum("ij,abj->abi", kernel, reduced_g2, optimize=True)

    def _make_target_array(self, source_g3: np.ndarray) -> np.ndarray:
        if source_g3.ndim != 4 or getattr(self, "phi_num_bins", None) is None:
            raise ValueError("Target construction currently expects angle-binned g3 data.")

        ideal_g3 = self._ideal_g3_raw(source_g3)
        reduced_g3 = self._reduce_g3(source_g3, ideal_g3)
        reduced_g3 = self._blur_phi_reduced(reduced_g3)
        reduced_g3 = self._blur_r_reduced(reduced_g3)
        mix = self._target_mix()
        target_reduced = (1.0 - mix[None, :, :, :]) * reduced_g3 + mix[None, :, :, :]
        return self._unreduce_g3(target_reduced, ideal_g3)

    def _make_target_g2_array(self, source_g2: np.ndarray) -> np.ndarray:
        ideal_g2 = self._ideal_g2_raw(source_g2)
        reduced_g2 = self._reduce_g2(source_g2, ideal_g2)
        reduced_g2 = self._blur_r_reduced_g2(reduced_g2)
        mix = self._target_mix_1d()
        target_reduced = (1.0 - mix[None, None, :]) * reduced_g2 + mix[None, None, :]
        return self._unreduce_g2(target_reduced, ideal_g2)

    def measure_g3(
        self,
        r_max: float | None = None,
        r_step: float | None = None,
        phi_num_bins: int = 90,
        plot_g3: bool = False,
        return_g3: bool = False,
        show_progress: bool = False,
        progress_label: str | None = None,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Measure the raw rooted three-body distribution.

        Parameters
        ----------
        r_max
            Maximum radial distance included in the radial histogram grid.
        r_step
            Radial bin width. `r_max / r_step` must be an integer.
        phi_num_bins
            Number of angular bins spanning 0 to pi.
        plot_g3
            If `True`, prepare the measured object for immediate inspection with
            `plot_g3()`.
        return_g3
            If `True`, also return the radial and angular bin centers along with the
            measured raw histogram.
        show_progress
            If `True`, display a simple text progress bar while the origin-centered
            triplet histograms are accumulated.
        progress_label
            Optional label shown next to the progress bar.

        Returns
        -------
        np.ndarray or tuple
            The raw non-reduced histogram with shape
            `(num_triplets, num_r, num_r, phi_num_bins)`. If `return_g3` is `True`,
            this method returns `(g3, r, phi)` where `r` and `phi` are the radial and
            angular bin centers.
        """

        # Coordinates
        if r_max is None:
            r_max = getattr(self, "r_max", None)
        if r_step is None:
            r_step = getattr(self, "r_step", None)
        if r_max is None or r_step is None:
            raise ValueError("r_max and r_step must be provided at least once.")
        if r_step <= 0 or r_max <= 0:
            raise ValueError("r_step and r_max must be positive.")

        num_r_float = r_max / r_step
        num_r = int(round(num_r_float))
        if not np.isclose(num_r_float, num_r):
            raise ValueError("r_max must be divisible by r_step.")

        self.r_max = r_max
        self.r_step = r_step
        self.r = np.arange(num_r, dtype=float) * r_step + 0.5 * r_step
        self.r_num = num_r
        self.num_r = num_r
        self.bin_centers = self.r
        self.bin_edges = np.arange(self.r_num + 1, dtype=float) * self.r_step
        self.phi_num_bins = int(phi_num_bins)
        self.phi_edges = np.linspace(0.0, np.pi, self.phi_num_bins + 1)
        self.phi_step = self.phi_edges[1] - self.phi_edges[0]
        self.phi = self.phi_edges[:-1] + 0.5 * self.phi_step
        self.phi_deg = np.rad2deg(self.phi)

        # lattice parameters
        u = self.atoms.cell[0]
        v = self.atoms.cell[1]
        w = self.atoms.cell[2]
        scaled_positions = self.atoms.get_scaled_positions()
        numbers = self.atoms.numbers

        # unique atomic numbers and number of pairs
        self.species = np.unique(numbers)
        self.num_sites = scaled_positions.shape[0]
        self.num_species = self.species.size
        self.num_triplets = (self.num_species**2) * (self.num_species + 1) // 2
        self.g3_index = np.zeros((self.num_triplets, 3), dtype=np.intp)
        origin_species_index = np.searchsorted(self.species, numbers)

        # Make rooted triplet index with unordered neighbors:
        # [center, neigh_1, neigh_2] with neigh_1 <= neigh_2
        # For a binary system this gives:
        # [0,0,0], [0,0,1], [0,1,1], [1,0,0], [1,0,1], [1,1,1]
        triplet_ind = 0
        for center_ind in range(self.num_species):
            for neigh1_ind in range(self.num_species):
                for neigh2_ind in range(neigh1_ind, self.num_species):
                    self.g3_index[triplet_ind, 0] = center_ind
                    self.g3_index[triplet_ind, 1] = neigh1_ind
                    self.g3_index[triplet_ind, 2] = neigh2_ind
                    triplet_ind += 1

        # Fast lookup from (center, neigh1, neigh2) to the flattened g3 channel.
        # Neighbor order is symmetrized so (i, j) and (j, i) map to the same channel.
        self.g3_lookup = -np.ones(
            (self.num_species, self.num_species, self.num_species),
            dtype=np.intp,
        )
        for triplet_ind, (center_ind, neigh1_ind, neigh2_ind) in enumerate(self.g3_index):
            self.g3_lookup[center_ind, neigh1_ind, neigh2_ind] = triplet_ind
            self.g3_lookup[center_ind, neigh2_ind, neigh1_ind] = triplet_ind
        species_labels = [chemical_symbols[int(spec)] for spec in self.species]
        self.pair_labels = [
            f"{species_labels[ind0]} | {species_labels[ind1]} {species_labels[ind2]}"
            for ind0, ind1, ind2 in self.g3_index
        ]
        self.species_pairs = [tuple(int(v) for v in triplet) for triplet in self.g3_index]
        self.g2_labels = [
            f"{species_labels[ind0]}-{species_labels[ind1]}"
            for ind0 in range(self.num_species)
            for ind1 in range(self.num_species)
        ]

        # determine required cell tiling
        dists = np.sum(
            np.array([
                u,
                v,
                w,
                u+v,
                u-v,
                u+w,
                u-w,
                v+w,
                v-w,
                u+v+w,
                u+v-w,
                u-v+w,
                u-v-w,
            ]
            )**2,
            axis=1
        )
        dist_min = np.sqrt(np.min(dists))
        dist_max = np.sqrt(np.max(dists))
        self.num_tile = int(np.ceil(self.r_max / dist_min) + 1)

        # tile and crop unit cells
        a, b, c, index = np.meshgrid(
            np.arange(-self.num_tile, self.num_tile + 1, dtype=np.intp),
            np.arange(-self.num_tile, self.num_tile + 1, dtype=np.intp),
            np.arange(-self.num_tile, self.num_tile + 1, dtype=np.intp),
            np.arange(self.num_sites, dtype=np.intp),
        )
        tile_species = numbers[index.ravel()]
        tile_xyz = (
            (a.ravel()[:, None] + scaled_positions[index.ravel(), 0][:, None]) * u[None, :]
            + (b.ravel()[:, None] + scaled_positions[index.ravel(), 1][:, None]) * v[None, :]
            + (c.ravel()[:, None] + scaled_positions[index.ravel(), 2][:, None]) * w[None, :]
        )
        keep = np.sum(tile_xyz**2, axis=1) < (self.r_max + dist_max) ** 2
        self.tile_species = tile_species[keep]
        self.tile_xyz = tile_xyz[keep, :]

        # subsets of tiled coordinates and origin coordinates by species
        xyz_all = []
        for spec in self.species:
            sub = self.tile_species == spec
            xyz_all.append(self.tile_xyz[sub, :])

        origin_xyz = (
            scaled_positions[:, 0][:, None] * u[None, :]
            + scaled_positions[:, 1][:, None] * v[None, :]
            + scaled_positions[:, 2][:, None] * w[None, :]
        )
        origin_xyz_by_species = []
        for ind0 in range(self.num_species):
            origin_xyz_by_species.append(origin_xyz[origin_species_index == ind0, :])

        self.origin_xyz = origin_xyz
        self.origin_species_index = origin_species_index
        self.origin_xyz_by_species = origin_xyz_by_species
        self.xyz_all = xyz_all

        # init g3
        self.g3count = np.zeros(
            (self.num_triplets, self.r_num, self.r_num, self.phi_num_bins),
            dtype=np.int64,
        )
        self.g2count = np.zeros(
            (self.num_species, self.num_species, self.r_num),
            dtype=np.int64,
        )

        r_max_sq = float(self.r_max * self.r_max)
        zero_tol = max(1e-12, (1e-9 * self.r_step) ** 2)
        flat_size = self.r_num * self.r_num * self.phi_num_bins
        triplets_by_center = [
            np.where(self.g3_index[:, 0] == ind0)[0]
            for ind0 in range(self.num_species)
        ]
        progress = None
        processed_origins = 0
        if show_progress:
            progress = _TextProgressBar(
                self.num_sites,
                label=progress_label or f"Measuring {self.label}",
            )
            progress.update(0)

        # calculate g3 as the sum over all unit-cell origins, grouped by center species
        for ind0 in range(self.num_species):
            for xyz0 in origin_xyz_by_species[ind0]:
                vector_table: list[np.ndarray] = []
                radius_sq_table: list[np.ndarray] = []
                radius_bin_table: list[np.ndarray] = []

                for indn in range(self.num_species):
                    vectors = xyz_all[indn] - xyz0
                    radius_sq = np.einsum("ij,ij->i", vectors, vectors)
                    keep = (radius_sq > zero_tol) & (radius_sq < r_max_sq)
                    vectors = vectors[keep]
                    radius_sq = radius_sq[keep]
                    radius_bin = np.floor(np.sqrt(radius_sq) / self.r_step).astype(np.intp)
                    keep_bin = radius_bin < self.r_num
                    vector_table.append(vectors[keep_bin])
                    radius_sq_table.append(radius_sq[keep_bin])
                    radius_bin_table.append(radius_bin[keep_bin])
                    if radius_bin_table[-1].size:
                        counts_2b = np.bincount(radius_bin_table[-1], minlength=self.r_num)
                        self.g2count[ind0, indn] += counts_2b

                for ind in triplets_by_center[ind0]:
                    _, ind1, ind2 = self.g3_index[ind]
                    v01 = vector_table[ind1]
                    v02 = vector_table[ind2]
                    r01_sq = radius_sq_table[ind1]
                    r02_sq = radius_sq_table[ind2]
                    r01_bin = radius_bin_table[ind1]
                    r02_bin = radius_bin_table[ind2]

                    if v01.size == 0 or v02.size == 0:
                        continue

                    dot = v01 @ v02.T
                    denom = np.sqrt(r01_sq[:, None] * r02_sq[None, :])
                    cos_phi = np.clip(dot / denom, -1.0, 1.0)
                    phi_bin = np.floor(np.arccos(cos_phi) / self.phi_step).astype(np.intp)
                    np.clip(phi_bin, 0, self.phi_num_bins - 1, out=phi_bin)

                    rr_index = (
                        (r01_bin[:, None] * self.r_num + r02_bin[None, :]) * self.phi_num_bins
                    )
                    linear = rr_index + phi_bin

                    if ind1 == ind2:
                        valid = np.ones(linear.shape, dtype=bool)
                        np.fill_diagonal(valid, False)
                        linear = linear[valid]

                    counts = np.bincount(linear.ravel(), minlength=flat_size)
                    self.g3count[ind] += counts.reshape(
                        self.r_num,
                        self.r_num,
                        self.phi_num_bins,
                    )

                    if ind1 != ind2:
                        rr_index_sym = (
                            (r02_bin[None, :] * self.r_num + r01_bin[:, None]) * self.phi_num_bins
                        )
                        linear_sym = rr_index_sym + phi_bin
                        counts_sym = np.bincount(linear_sym.ravel(), minlength=flat_size)
                        self.g3count[ind] += counts_sym.reshape(
                            self.r_num,
                            self.r_num,
                            self.phi_num_bins,
                        )
                processed_origins += 1
                if progress is not None:
                    progress.update(processed_origins)

        if progress is not None:
            progress.close()

        self.g3 = self.g3count
        self.g2 = self.g2count
        self.summary = {
            "kind": "measured",
            "num_atoms": len(self.atoms),
            "num_species": int(self.num_species),
            "num_triplets": int(self.num_triplets),
            "g2_shape": tuple(self.g2.shape),
            "num_origins": int(self.num_sites),
            "shape": tuple(self.g3.shape),
            "r_max": float(self.r_max),
            "r_step": float(self.r_step),
            "phi_num_bins": int(self.phi_num_bins),
        }
        self.history.append(
            {
                "event": "measure_g3",
                "shape": tuple(self.g3.shape),
                "phi_num_bins": int(self.phi_num_bins),
            }
        )
        if plot_g3:
            self.plot_g3()

        if return_g3:
            return self.g3, self.r, self.phi
        return self.g3

    def target_g3(
        self,
        *,
        target_r_min: float,
        target_r_max: float,
        r_sigma: float | None = None,
        r_sigma_at: float | None = None,
        phi_sigma_deg: float | None = None,
        label: str | None = None,
        **kwargs: Any,
    ) -> "G3Distribution":
        """Construct a transformed target distribution from the current raw g3.

        The target is built in reduced coordinates, where the random reference
        distribution is proportional to `r01^2 * r02^2 * sin(phi)`. The source
        distribution is first reduced by this ideal-density factor, optionally
        blurred in `phi` and in the two radial directions, and then smoothly mixed
        toward `1.0` between `target_r_min` and `target_r_max`. The returned object
        stores the transformed result back in the original raw, non-reduced form.

        Parameters
        ----------
        target_r_min
            Radius where the smooth transition away from the measured reduced
            distribution begins.
        target_r_max
            Radius where the reduced target has fully transitioned to the random
            limit of `1.0`.
        r_sigma
            Radial blur width, in Angstrom, evaluated at `r_sigma_at`. The effective
            radial blur grows linearly with radius. If `r_sigma_at` is omitted,
            `r_sigma` is interpreted as a linear slope so that `sigma_r(r) = r_sigma * r`.
        r_sigma_at
            Shared reference radius where the radial blur equals `r_sigma` and the
            angular blur equals `phi_sigma_deg`. If omitted, both blur widths grow
            linearly from zero using `r_sigma` and `phi_sigma_deg` as slopes.
        phi_sigma_deg
            Angular Gaussian blur width in degrees, evaluated at `r_sigma_at`.
            Reflection is used at `phi = 0` and `phi = 180` degrees. If
            `r_sigma_at` is omitted, `phi_sigma_deg` is interpreted as a linear
            slope so that `sigma_phi(r) = phi_sigma_deg * r`.
        label
            Optional label for the returned target distribution.

        Returns
        -------
        G3Distribution
            A new distribution containing the transformed raw target histogram.
        """
        return G3Distribution(
            self,
            target_r_min=target_r_min,
            target_r_max=target_r_max,
            r_sigma=r_sigma,
            r_sigma_at=r_sigma_at,
            phi_sigma_deg=phi_sigma_deg,
            label=label or f"{self.label}-target",
            **kwargs,
        )

    def plot_g3(
        self,
        pair: int | str = 0,
        *,
        normalize: bool = True,
    ):
        """Return an interactive anywidget explorer for a rooted triplet channel.

        Parameters
        ----------
        pair
            Either the integer triplet index or a triplet label such as
            `"Si-Si-C"`, where the center atom is shown in the middle.
        normalize
            If `True`, display reduced-density views that approach `1.0` in the
            random long-range limit.

        Returns
        -------
        G3PlotWidget
            Interactive widget with a `(phi, r)` slice view and a linked rooted
            two-body shell selector.
        """
        self._ensure_plot_data()
        pair_index = self._resolve_pair_index(pair)
        if self.g3.ndim != 4 or getattr(self, "phi_num_bins", None) is None:
            raise ValueError("Interactive plotting currently expects angle-binned g3 data.")
        from .g3_widget import G3PlotWidget

        return G3PlotWidget(
            self,
            triplet_index=pair_index,
            normalize=normalize,
        )

    def _ensure_plot_data(self) -> None:
        if self.g3 is not None:
            return
        if self.source_distribution is not None:
            self._ensure_source_g3()
            if self.source_distribution is None or self.source_distribution.g3 is None:
                raise ValueError("Unable to derive target g3 from the source distribution.")
            self.g3 = self._make_target_array(self.source_distribution.g3)
            return
        self.measure_g3()

    def _resolve_pair_index(self, pair: int | str) -> int:
        labels = getattr(self, "pair_labels", None)
        if labels is None:
            labels = []
        if isinstance(pair, int):
            if pair < 0 or pair >= len(labels):
                raise IndexError("Pair index is out of range.")
            return pair
        if pair not in labels:
            raise KeyError(f"Unknown pair label {pair!r}. Available labels: {labels}")
        return labels.index(pair)

    def __repr__(self) -> str:
        kind = "target" if self.is_target else "measured"
        labels = getattr(self, "pair_labels", [])
        bin_count = getattr(self, "num_r", getattr(self, "r_num", None))
        return (
            f"G3Distribution(label={self.label!r}, kind={kind!r}, "
            f"pairs={len(labels)}, bins={bin_count})"
        )
