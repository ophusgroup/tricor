"""Three-body distribution scaffolding."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from ase.atoms import Atoms
from ase.data import chemical_symbols


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
        self.r = distribution.r.copy() if hasattr(distribution, "r") else self.bin_centers.copy()
        self.r_num = getattr(distribution, "r_num", self.num_r)
        if hasattr(distribution, "phi_num_bins"):
            self.phi_num_bins = distribution.phi_num_bins
            self.phi_edges = distribution.phi_edges.copy()
            self.phi_step = distribution.phi_step
            self.phi = distribution.phi.copy()
            self.phi_deg = distribution.phi_deg.copy()
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
        if source_g3.ndim == 4 and getattr(self, "phi_num_bins", None) is not None:
            centers = self.bin_centers.astype(np.float32, copy=False)
            r01, r02 = np.meshgrid(centers, centers, indexing="ij")
            average_radius = (r01 + r02) / 2.0

            shell = np.square(centers + 0.5 * self.r_step).astype(np.float32, copy=False)
            uniform = (shell[:, None] * shell[None, :]).astype(np.float32, copy=False)
            uniform /= float(uniform.mean())
            uniform = uniform[:, :, None]

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
            mix = mix[:, :, None]

            target = np.empty_like(source_g3, dtype=np.float32)
            for pair_index in range(source_g3.shape[0]):
                pair_uniform = uniform * float(np.mean(source_g3[pair_index]))
                target[pair_index] = (
                    (1.0 - mix) * source_g3[pair_index].astype(np.float32, copy=False)
                    + mix * pair_uniform
                )
            return target

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
        r_max: float | None = None,
        r_step: float | None = None,
        phi_num_bins: int = 90,
        plot_g3: bool = True,
    ) -> np.ndarray:
        """
        Measure 3 body distribution
        """

        # Coordinates
        if r_max is None:
            r_max = getattr(self, "r_max", None)
        if r_step is None:
            r_step = getattr(self, "r_step", None)
        if r_max is None or r_step is None:
            raise ValueError("r_max and r_step must be provided at least once.")

        self.r_max = r_max
        self.r_step = r_step
        self.r = np.arange(0.0, r_max, r_step) + r_step / 2
        self.r_num = self.r.size
        self.num_r = self.r_num
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
        self.num_triplets = (self.num_species**2)*(self.num_species+1)//2
        self.g3_index = np.zeros((self.num_triplets,3),dtype='int')

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
            dtype='int',
        )
        for triplet_ind, (center_ind, neigh1_ind, neigh2_ind) in enumerate(self.g3_index):
            self.g3_lookup[center_ind, neigh1_ind, neigh2_ind] = triplet_ind
            self.g3_lookup[center_ind, neigh2_ind, neigh1_ind] = triplet_ind
        species_labels = [chemical_symbols[int(spec)] for spec in self.species]
        self.pair_labels = [
            f"{species_labels[ind0]}{species_labels[ind1]}{species_labels[ind2]}"
            for ind0, ind1, ind2 in self.g3_index
        ]
        self.species_pairs = list(self.g3_index)

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
            np.arange(-self.num_tile,self.num_tile+1,dtype='int'),
            np.arange(-self.num_tile,self.num_tile+1,dtype='int'),
            np.arange(-self.num_tile,self.num_tile+1,dtype='int'),
            np.arange(self.num_sites,dtype='int'),
        )
        tile_species = numbers[index.ravel()]
        tile_xyz = (a.ravel()[:,None]+scaled_positions[index.ravel(),0][:,None])*u[None,:] \
            + (b.ravel()[:,None]+scaled_positions[index.ravel(),1][:,None])*v[None,:] \
            + (c.ravel()[:,None]+scaled_positions[index.ravel(),2][:,None])*w[None,:]
        keep = np.sum(tile_xyz**2,axis=1) < (self.r_max + dist_max)**2
        self.tile_species = tile_species[keep]
        self.tile_xyz = tile_xyz[keep,:]

        # subsets of coordinates by type
        xyz_all = []
        for spec in self.species:
            sub = self.tile_species==spec
            xyz_all.append(self.tile_xyz[sub,:])

        # origin coordinates
        xyz0 = np.zeros((self.num_species,3))
        for ind,xyz_spec in enumerate(xyz_all):
            ind_min = np.argmin(
                np.sum(xyz_spec**2,axis=1)
            )
            xyz0[ind,:] = xyz_spec[ind_min,:]
        self.xyz0 = xyz0
        self.xyz_all = xyz_all

        # init g3
        self.g3count = np.zeros(
            (self.num_triplets, self.r_num, self.r_num, self.phi_num_bins),
            dtype=np.int64,
        )

        r_max_sq = float(self.r_max * self.r_max)
        zero_tol = max(1e-12, (1e-9 * self.r_step) ** 2)
        vector_table: list[list[np.ndarray]] = []
        radius_sq_table: list[list[np.ndarray]] = []
        radius_bin_table: list[list[np.ndarray]] = []

        # Precompute neighbors for each center-species / neighbor-species combination.
        for ind0 in range(self.num_species):
            vectors_by_species: list[np.ndarray] = []
            radius_sq_by_species: list[np.ndarray] = []
            radius_bin_by_species: list[np.ndarray] = []
            for indn in range(self.num_species):
                vectors = xyz_all[indn] - xyz0[ind0]
                radius_sq = np.einsum("ij,ij->i", vectors, vectors)
                keep = (radius_sq > zero_tol) & (radius_sq < r_max_sq)
                vectors = vectors[keep]
                radius_sq = radius_sq[keep]
                radius_bin = np.floor(np.sqrt(radius_sq) / self.r_step).astype(np.intp)
                keep_bin = radius_bin < self.r_num
                vectors_by_species.append(vectors[keep_bin])
                radius_sq_by_species.append(radius_sq[keep_bin])
                radius_bin_by_species.append(radius_bin[keep_bin])
            vector_table.append(vectors_by_species)
            radius_sq_table.append(radius_sq_by_species)
            radius_bin_table.append(radius_bin_by_species)

        flat_size = self.r_num * self.r_num * self.phi_num_bins

        # calculate g3
        for ind in range(self.num_triplets):
            ind0, ind1, ind2 = self.g3_index[ind]
            v01 = vector_table[ind0][ind1]
            v02 = vector_table[ind0][ind2]
            r01_sq = radius_sq_table[ind0][ind1]
            r02_sq = radius_sq_table[ind0][ind2]
            r01_bin = radius_bin_table[ind0][ind1]
            r02_bin = radius_bin_table[ind0][ind2]

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
            self.g3count[ind] += counts.reshape(self.r_num, self.r_num, self.phi_num_bins)

            # Explicitly mirror mixed-species neighbor channels so the stored array is
            # symmetric in the two radial axes.
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

        self.g3 = self.g3count
        self.summary = {
            "kind": "measured",
            "num_atoms": len(self.atoms),
            "num_species": int(self.num_species),
            "num_triplets": int(self.num_triplets),
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
        return self.g3

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
        """Plot a single angular or radial slice of one channel."""
        self._ensure_plot_data()

        pair_index = self._resolve_pair_index(pair)
        if self.g3.ndim == 4 and getattr(self, "phi_num_bins", None) is not None:
            if slice_index is None:
                slice_index = self.phi_num_bins // 2
            slice_index = int(np.clip(slice_index, 0, self.phi_num_bins - 1))
        else:
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
        if self.g3.ndim == 4 and getattr(self, "phi_num_bins", None) is not None:
            slice_label = f"phi={self.phi_deg[slice_index]:.1f} deg"
        else:
            slice_label = f"r12={self.bin_centers[slice_index]:.2f}"
        ax.set_title(title or f"{self.label}: {self.pair_labels[pair_index]} at {slice_label}")
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
