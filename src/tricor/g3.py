"""Three-body distribution scaffolding."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from ase.atoms import Atoms


def _resolve_sigma(
    blur_sigma: float | None,
    r_sigma: float | None,
) -> float | None:
    if blur_sigma is None:
        return r_sigma
    if r_sigma is None:
        return blur_sigma
    if not np.isclose(blur_sigma, r_sigma):
        raise ValueError("Received conflicting values for blur_sigma and r_sigma.")
    return blur_sigma


class G3Distribution:
    """Dummy container for measured or transformed three-body distributions."""

    def __init__(
        self,
        source: Atoms | "G3Distribution",
        r_step: float | None = None,
        r_max: float | None = None,
        *,
        r_min: float | None = None,
        blur_sigma: float | None = None,
        r_sigma: float | None = None,
        label: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.label = label or "g3"
        self.metadata: dict[str, Any] = dict(kwargs)
        self.source_distribution: G3Distribution | None = None
        self.atoms: Atoms | None = None
        self.g3: np.ndarray | None = None
        self.summary: dict[str, Any] = {}
        self.history: list[dict[str, Any]] = []
        self.is_target = False
        self.r_min = r_min
        self.blur_sigma = _resolve_sigma(blur_sigma, r_sigma)

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
        if r_step is None or r_max is None:
            raise ValueError("r_step and r_max are required when initializing from Atoms.")
        if r_step <= 0 or r_max <= 0:
            raise ValueError("r_step and r_max must be positive.")

        num_r_float = r_max / r_step
        num_r = int(round(num_r_float))
        if not np.isclose(num_r_float, num_r):
            raise ValueError("r_max must be divisible by r_step.")

        self.atoms = atoms.copy()
        self.r_step = float(r_step)
        self.r_max = float(r_max)
        self.num_r = num_r
        self.bin_edges = np.linspace(0.0, self.r_max, self.num_r + 1)
        self.bin_centers = self.bin_edges[:-1] + 0.5 * self.r_step
        self.species = sorted(set(self.atoms.get_chemical_symbols()))
        self.species_pairs = [
            (left, right)
            for index, left in enumerate(self.species)
            for right in self.species[index:]
        ]
        self.pair_labels = [f"{left}-{right}" for left, right in self.species_pairs]
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
        self.num_r = distribution.num_r
        self.bin_edges = distribution.bin_edges.copy()
        self.bin_centers = distribution.bin_centers.copy()
        self.species = list(distribution.species)
        self.species_pairs = list(distribution.species_pairs)
        self.pair_labels = list(distribution.pair_labels)
        self.is_target = True

        if not np.isclose(self.r_step, distribution.r_step):
            raise ValueError("Target distributions must currently reuse the source r_step.")
        if not np.isclose(self.r_max, distribution.r_max):
            raise ValueError("Target distributions must currently reuse the source r_max.")

        self._ensure_source_g3()
        if distribution.g3 is not None:
            self.g3 = self._make_target_array(distribution.g3)
            self.summary = dict(distribution.summary)
            self.summary.update(
                {
                    "kind": "target",
                    "r_min": self.r_min,
                    "blur_sigma": self.blur_sigma,
                }
            )

        self.history.extend(distribution.history)
        self.history.append(
            {
                "event": "initialized",
                "kind": "target",
                "r_min": self.r_min,
                "blur_sigma": self.blur_sigma,
            }
        )

    def _ensure_source_g3(self) -> None:
        if self.source_distribution is None or self.source_distribution.g3 is not None:
            return
        self.source_distribution.measure_g3()

    def _make_target_array(self, source_g3: np.ndarray) -> np.ndarray:
        centers = self.bin_centers.astype(np.float32, copy=False)
        r01, r02, r12 = np.meshgrid(centers, centers, centers, indexing="ij")
        average_radius = (r01 + r02 + r12) / 3.0

        shell = np.square(centers + 0.5 * self.r_step).astype(np.float32, copy=False)
        uniform = (
            shell[:, None, None] * shell[None, :, None] * shell[None, None, :]
        ).astype(np.float32, copy=False)
        uniform /= float(uniform.mean())

        if self.r_min is None:
            envelope = np.zeros_like(average_radius, dtype=np.float32)
        else:
            width = max(self.r_max - self.r_min, self.r_step)
            scaled = np.clip((average_radius - self.r_min) / width, 0.0, 1.0)
            envelope = scaled * scaled * (3.0 - 2.0 * scaled)

        blur_strength = 0.0 if self.blur_sigma is None else float(self.blur_sigma)
        cumulative_blur = np.clip(
            blur_strength * average_radius / max(self.r_max, self.r_step),
            0.0,
            0.95,
        )
        mix = np.maximum(envelope, cumulative_blur).astype(np.float32, copy=False)

        target = np.empty_like(source_g3, dtype=np.float32)
        for pair_index in range(source_g3.shape[0]):
            pair_uniform = uniform * float(np.mean(source_g3[pair_index]))
            target[pair_index] = (
                (1.0 - mix) * source_g3[pair_index].astype(np.float32, copy=False)
                + mix * pair_uniform
            )
        return target

    def measure_g3(
        self, 
        r_max,
        r_step,
        plot_g3 = True,
    ):
        """
        Measure 3 body distribution
        """

        # Coordinates
        self.r_max = r_max
        self.r_step = r_step
        self.r = np.arange(0.0,r_max,r_step) + r_step/2
        self.r_num = self.r.size

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
        self.num_triplets = (self.num_species**2)*(self.num_species+1)//2
        self.g3_index = np.zeros((self.num_triplets,3),dtype='int')

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
        self.num_tile = (np.ceil(self.r_max / dist_min) + 1).astype('int')

        # tile and crop unit cells
        a,b,c,index = np.meshgrid(
            np.arange(self.num_tile,dtype='int'),
            np.arange(self.num_tile,dtype='int'),
            np.arange(self.num_tile,dtype='int'),
            np.arange(self.num_sites,dtype='int'),
        )
        tile_species = numbers[index.ravel()]
        tile_xyz = (a.ravel()[:,None]+scaled_positions[index.ravel(),0][:,None])*u[None,:] \
            + (b.ravel()[:,None]+scaled_positions[index.ravel(),1][:,None])*v[None,:] \
            + (c.ravel()[:,None]+scaled_positions[index.ravel(),2][:,None])*w[None,:]
        keep = np.sum(tile_xyz**2,axis=1) < (self.r_max + dist_max)**2
        self.tile_species = tile_species[keep]
        self.tile_xyz = tile_xyz[keep,:]

        # calculate g3



        # """Populate a synthetic g3 grid with the right shape for notebook work."""
        # if self.atoms is None:
        #     raise ValueError("This distribution has no atomic structure to measure.")

        # centers = self.bin_centers.astype(np.float32, copy=False)
        # r01, r02, r12 = np.meshgrid(centers, centers, centers, indexing="ij")
        # pair_count = len(self.species_pairs)
        # g3 = np.empty((pair_count, self.num_r, self.num_r, self.num_r), dtype=np.float32)

        # radial_decay = np.exp(-(r01 + r02 + r12) / max(self.r_max * 0.75, self.r_step))
        # shell_center = 0.35 * self.r_max
        # shell_width = max(0.2 * self.r_max, self.r_step)
        # shell_peak = np.exp(
        #     -(
        #         np.square(r01 - shell_center)
        #         + np.square(r02 - 1.2 * shell_center)
        #         + np.square(r12 - 1.4 * shell_center)
        #     )
        #     / (2.0 * shell_width * shell_width)
        # )
        # oscillation = 0.5 * (1.0 + np.cos((r01 - r02) / max(self.r_step, 1e-8)))

        # for pair_index, _pair in enumerate(self.species_pairs):
        #     scale = amplitude * (1.0 + 0.2 * pair_index)
        #     g3[pair_index] = baseline + scale * radial_decay * (0.35 + 0.65 * oscillation)
        #     g3[pair_index] += 0.15 * (pair_index + 1) * shell_peak

        # self.g3 = g3
        # self.summary = {
        #     "kind": "measured",
        #     "num_atoms": len(self.atoms),
        #     "num_species": len(self.species),
        #     "num_pairs": len(self.species_pairs),
        #     "shape": tuple(g3.shape),
        #     "amplitude": amplitude,
        #     "baseline": baseline,
        # }
        # if kwargs:
        #     self.summary["measure_kwargs"] = dict(kwargs)
        # self.history.append(
        #     {
        #         "event": "measure_g3",
        #         "shape": tuple(g3.shape),
        #         "kwargs": dict(kwargs),
        #     }
        # )
        # return g3

    def target_g3(
        self,
        *,
        r_min: float,
        blur_sigma: float | None = None,
        r_sigma: float | None = None,
        label: str | None = None,
        **kwargs: Any,
    ) -> "G3Distribution":
        """Return a new target distribution derived from the current one."""
        return G3Distribution(
            self,
            r_min=r_min,
            blur_sigma=blur_sigma,
            r_sigma=r_sigma,
            label=label or f"{self.label}-target",
            **kwargs,
        )

    def plot_g3(
        self,
        pair: int | str = 0,
        *,
        slice_index: int | None = None,
        ax: plt.Axes | None = None,
        cmap: str = "viridis",
        title: str | None = None,
    ) -> plt.Axes:
        """Plot a single r12 slice of one pair channel."""
        self._ensure_plot_data()

        pair_index = self._resolve_pair_index(pair)
        if slice_index is None:
            slice_index = self.num_r // 2
        slice_index = int(np.clip(slice_index, 0, self.num_r - 1))

        if ax is None:
            _, ax = plt.subplots(figsize=(6, 5))

        plane = self.g3[pair_index, :, :, slice_index]
        image = ax.imshow(
            plane,
            origin="lower",
            extent=(0.0, self.r_max, 0.0, self.r_max),
            aspect="equal",
            cmap=cmap,
        )
        ax.set_xlabel("r01")
        ax.set_ylabel("r02")
        ax.set_title(
            title
            or f"{self.label}: {self.pair_labels[pair_index]} at r12={self.bin_centers[slice_index]:.2f}"
        )
        ax.figure.colorbar(image, ax=ax, label="g3")
        return ax

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
        if isinstance(pair, int):
            if pair < 0 or pair >= len(self.species_pairs):
                raise IndexError("Pair index is out of range.")
            return pair
        if pair not in self.pair_labels:
            raise KeyError(f"Unknown pair label {pair!r}. Available labels: {self.pair_labels}")
        return self.pair_labels.index(pair)

    def __repr__(self) -> str:
        kind = "target" if self.is_target else "measured"
        return (
            f"G3Distribution(label={self.label!r}, kind={kind!r}, "
            f"pairs={len(self.species_pairs)}, bins={self.num_r})"
        )
