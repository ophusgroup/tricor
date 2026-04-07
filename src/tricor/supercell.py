"""Random supercell initialization and local-update Monte Carlo scaffolding."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any
import warnings

import numpy as np
from ase.atoms import Atoms
from ase.neighborlist import neighbor_list

from .g3 import G3Distribution, _EPS, _TextProgressBar


class Supercell:
    """Random supercell scaffold driven by a target :class:`G3Distribution`."""

    def __init__(
        self,
        distribution: G3Distribution,
        cell_dim_angstroms: float | Sequence[float] | None = None,
        *,
        relative_density: float = 1.0,
        measure_g3: bool = False,
        plot_g3_compare: bool = False,
        label: str | None = None,
        rng_seed: int | None = None,
        cell_dim: float | Sequence[float] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a random supercell with the target composition and density.

        Parameters
        ----------
        distribution
            Target distribution that the eventual Monte Carlo engine will try to
            match.
        cell_dim_angstroms
            Physical supercell lengths in Angstrom along the source lattice
            vectors. A single scalar produces a cubic box, while a length-3
            sequence specifies `(La, Lb, Lc)`.
        relative_density
            Total number density relative to the crystalline reference cell. A
            value of `1.0` preserves the crystalline density, while values below
            one reduce the number of atoms placed into the requested box.
        measure_g3
            If `True`, immediately measure the random supercell `g2/g3` on the
            target grid during initialization. Leaving this as `False` is often
            faster when the next step is a repulsion-style preconditioning pass.
        plot_g3_compare
            If `True`, immediately display an interactive comparison between the
            random supercell and the target distribution when running inside
            IPython/Jupyter.
        label
            Human-readable label for summaries and repr output.
        rng_seed
            Optional random seed for reproducible initialization and Monte Carlo
            trial moves.
        **kwargs
            Extra metadata stored for future Monte Carlo options.
        """
        if relative_density <= 0:
            raise ValueError("relative_density must be positive.")
        if distribution.atoms is None:
            raise ValueError("Supercell construction requires a distribution with source atoms.")

        if cell_dim_angstroms is None:
            if cell_dim is None:
                raise TypeError("cell_dim_angstroms must be provided.")
            cell_dim_angstroms = cell_dim
            warnings.warn(
                "`cell_dim` is deprecated; use `cell_dim_angstroms`.",
                DeprecationWarning,
                stacklevel=2,
            )
        elif cell_dim is not None:
            normalized_lengths = self._normalize_cell_dim_angstroms(cell_dim_angstroms)
            normalized_alias = self._normalize_cell_dim_angstroms(cell_dim)
            if not np.allclose(normalized_lengths, normalized_alias):
                raise ValueError("Received conflicting values for cell_dim_angstroms and cell_dim.")
            cell_dim_angstroms = normalized_lengths

        self.target_distribution = distribution
        self.distribution = distribution
        self.target_distribution._ensure_plot_data()

        self.cell_dim_angstroms = self._normalize_cell_dim_angstroms(cell_dim_angstroms)
        self.relative_density = float(relative_density)
        self.label = label or "supercell"
        self.metadata: dict[str, Any] = dict(kwargs)
        self.rng = np.random.default_rng(rng_seed)

        self.measure_r_max = float(self.target_distribution.r_max)
        self.measure_r_step = float(self.target_distribution.r_step)
        self.measure_phi_num_bins = int(self.target_distribution.phi_num_bins)

        self.reference_atoms = self.target_distribution.atoms.copy()
        self.atoms = self._build_random_atoms()
        self.current_distribution: G3Distribution | None = None
        self.mc_history: dict[str, np.ndarray] | None = None
        self.best_score: float | None = None
        self.last_temperature: float | None = None
        self.current_cost: float | None = None

        self._cell_matrix = np.asarray(self.atoms.cell.array, dtype=np.float64)
        self._cell_inverse = np.linalg.inv(self._cell_matrix)
        self._r_max_sq = float(self.measure_r_max * self.measure_r_max)
        self._zero_tol = max(1e-12, (1e-9 * self.measure_r_step) ** 2)
        self._species = np.array(self.target_distribution.species, copy=True)
        self._num_species = int(self._species.size)
        self._num_triplets = int(self.target_distribution.g3_index.shape[0])
        self._r_num = int(self.target_distribution.r_num)
        self._phi_num_bins = int(self.target_distribution.phi_num_bins)
        self._phi_step = float(self.target_distribution.phi_step)
        self._triplets_by_center = [
            np.where(self.target_distribution.g3_index[:, 0] == ind0)[0]
            for ind0 in range(self._num_species)
        ]
        self._flat_triplet_size = self._r_num * self._r_num * self._phi_num_bins
        self._atom_species_index = np.searchsorted(self._species, self.atoms.numbers)

        if measure_g3 or plot_g3_compare:
            self.measure_g3(show_progress=True)
            self._initialize_mc_state()

        if plot_g3_compare:
            self._display_compare_widget()

    def _normalize_cell_dim_angstroms(
        self,
        cell_dim_angstroms: float | Sequence[float],
    ) -> tuple[float, float, float]:
        """Validate and normalize the requested supercell lengths in Angstrom."""
        if isinstance(cell_dim_angstroms, (int, float)):
            if float(cell_dim_angstroms) <= 0:
                raise ValueError("cell_dim_angstroms must be positive.")
            length = float(cell_dim_angstroms)
            return (length, length, length)

        dims = tuple(float(value) for value in cell_dim_angstroms)
        if len(dims) != 3 or any(value <= 0 for value in dims):
            raise ValueError(
                "cell_dim_angstroms must be a scalar or a length-3 sequence of positive values."
            )
        return dims

    def _build_supercell_cell(self) -> np.ndarray:
        """Scale the source lattice vectors to the requested physical box lengths."""
        reference_cell = np.asarray(self.reference_atoms.cell.array, dtype=np.float64)
        reference_lengths = np.linalg.norm(reference_cell, axis=1)
        if np.any(reference_lengths <= _EPS):
            raise ValueError("Reference cell must have non-zero lattice-vector lengths.")
        scale = np.asarray(self.cell_dim_angstroms, dtype=np.float64) / reference_lengths
        return reference_cell * scale[:, None]

    def _target_species_counts(self, target_volume: float) -> tuple[np.ndarray, np.ndarray]:
        """Return the closest exact-stoichiometry atom counts for the requested box."""
        numbers = np.asarray(self.reference_atoms.numbers, dtype=np.int64)
        species, counts = np.unique(numbers, return_counts=True)
        divisor = int(np.gcd.reduce(counts.astype(np.int64)))
        reduced_counts = counts.astype(np.int64) // max(divisor, 1)
        atoms_per_formula_unit = int(np.sum(reduced_counts))
        reference_density = len(numbers) / max(float(self.reference_atoms.cell.volume), _EPS)
        target_num_atoms = float(target_volume) * reference_density * self.relative_density
        num_formula_units = max(
            1,
            int(round(target_num_atoms / max(atoms_per_formula_unit, 1))),
        )
        return species.astype(np.int64), (reduced_counts * num_formula_units).astype(np.int64)

    def _build_random_atoms(self) -> Atoms:
        """Construct a random-coordinate supercell at the requested box and density."""
        cell = self._build_supercell_cell()
        target_volume = float(abs(np.linalg.det(cell)))
        species, counts = self._target_species_counts(target_volume)
        numbers = np.repeat(species, counts.astype(np.intp))
        self.rng.shuffle(numbers)
        scaled_positions = self.rng.random((len(numbers), 3))

        atoms = Atoms(
            numbers=numbers,
            cell=cell,
            pbc=self.reference_atoms.pbc,
            scaled_positions=scaled_positions,
        )
        atoms.info["relative_density"] = self.relative_density
        atoms.info["cell_dim_angstroms"] = self.cell_dim_angstroms
        return atoms

    def _distribution_scale(self, *, order: int) -> float:
        """Scale target raw histograms onto the current supercell density basis."""
        target_num_origins = int(
            self.target_distribution.summary.get(
                "num_origins",
                len(self.target_distribution.atoms),
            )
        )
        current_num_origins = int(
            self.current_distribution.summary.get(
                "num_origins",
                len(self.atoms),
            )
        )
        target_density = target_num_origins / max(
            float(self.target_distribution.atoms.cell.volume),
            _EPS,
        )
        current_density = current_num_origins / max(float(self.atoms.cell.volume), _EPS)
        density_ratio = current_density / max(target_density, _EPS)
        origin_ratio = current_num_origins / max(target_num_origins, 1)
        return float(origin_ratio * density_ratio**order)

    def _initialize_mc_state(self) -> None:
        """Cache neighbor tables, target histograms, and loss vectors."""
        if self.current_distribution is None:
            raise ValueError("Measure the supercell distribution before initializing MC state.")

        self._neighbor_indices = self._build_neighbor_indices()
        self._origin_contribution_cache = [
            self._origin_sparse_contribution(origin_index)
            for origin_index in range(len(self.atoms))
        ]
        self._target_g2_flat = (
            np.asarray(self.target_distribution.g2, dtype=np.float64).reshape(-1)
            * self._distribution_scale(order=1)
        )
        self._target_g3_flat = (
            np.asarray(self.target_distribution.g3, dtype=np.float64).reshape(-1)
            * self._distribution_scale(order=2)
        )
        self._current_g2_flat = self.current_distribution.g2.reshape(-1)
        self._current_g3_flat = self.current_distribution.g3.reshape(-1)
        self._g2_diff_flat = self._current_g2_flat.astype(np.float64) - self._target_g2_flat
        self._g3_diff_flat = self._current_g3_flat.astype(np.float64) - self._target_g3_flat
        self.current_cost = float(np.dot(self._g3_diff_flat, self._g3_diff_flat))
        self.best_score = self.current_cost

    def _build_neighbor_indices(self) -> list[np.ndarray]:
        """Build a fixed-cutoff neighbor table for the current coordinates."""
        num_atoms = len(self.atoms)
        i, j = neighbor_list(
            "ij",
            self.atoms,
            self.measure_r_max,
            self_interaction=False,
        )
        order = np.argsort(i, kind="stable")
        i = i[order]
        j = j[order]
        counts = np.bincount(i, minlength=num_atoms)
        neighbors: list[np.ndarray] = []
        start = 0
        for count in counts:
            neighbors.append(np.array(j[start:start + count], dtype=np.intp, copy=True))
            start += count
        return neighbors

    def _minimum_image_vectors(
        self,
        origin_position: np.ndarray,
        target_positions: np.ndarray,
    ) -> np.ndarray:
        """Return minimum-image displacement vectors from origin to the targets."""
        delta = target_positions - origin_position[None, :]
        frac = delta @ self._cell_inverse
        frac -= np.rint(frac)
        return frac @ self._cell_matrix

    def _wrap_position(self, position: np.ndarray) -> np.ndarray:
        """Wrap a Cartesian position back into the periodic supercell."""
        frac = np.asarray(position, dtype=np.float64) @ self._cell_inverse
        frac %= 1.0
        return frac @ self._cell_matrix

    def _wrap_positions(self, positions: np.ndarray) -> np.ndarray:
        """Wrap an array of Cartesian positions back into the periodic supercell."""
        frac = np.asarray(positions, dtype=np.float64) @ self._cell_inverse
        frac %= 1.0
        return frac @ self._cell_matrix

    def _query_neighbors_for_position(
        self,
        atom_index: int,
        origin_position: np.ndarray,
    ) -> np.ndarray:
        """Find all neighbors within `r_max` of a proposed origin position."""
        vectors = self._minimum_image_vectors(origin_position, self.atoms.positions)
        radius_sq = np.einsum("ij,ij->i", vectors, vectors)
        keep = (radius_sq > self._zero_tol) & (radius_sq < self._r_max_sq)
        keep[atom_index] = False
        return np.flatnonzero(keep).astype(np.intp)

    def _repel_trial_position(
        self,
        atom_index: int,
        position: np.ndarray,
        r_min_nn: float | None,
        *,
        max_iter: int = 12,
    ) -> tuple[np.ndarray, bool, int]:
        """Push a trial position away from any atoms closer than `r_min_nn`."""
        if r_min_nn is None or r_min_nn <= 0.0:
            return self._wrap_position(position), True, 0

        r_min_nn = float(r_min_nn)
        position = self._wrap_position(position)
        close_sq = r_min_nn * r_min_nn

        for iter_count in range(1, max_iter + 1):
            vectors = self._minimum_image_vectors(position, self.atoms.positions)
            radius_sq = np.einsum("ij,ij->i", vectors, vectors)
            keep = (radius_sq > self._zero_tol) & (radius_sq < close_sq)
            keep[atom_index] = False
            if not np.any(keep):
                return position, True, iter_count - 1

            close_vectors = vectors[keep]
            close_radius = np.sqrt(np.maximum(radius_sq[keep], _EPS))
            overlap = r_min_nn - close_radius
            direction = -close_vectors / close_radius[:, None]
            displacement = np.sum(direction * overlap[:, None], axis=0)
            position = self._wrap_position(position + displacement)

        vectors = self._minimum_image_vectors(position, self.atoms.positions)
        radius_sq = np.einsum("ij,ij->i", vectors, vectors)
        keep = (radius_sq > self._zero_tol) & (radius_sq < close_sq)
        keep[atom_index] = False
        return position, not np.any(keep), max_iter

    def _origin_sparse_contribution(
        self,
        origin_index: int,
        *,
        neighbor_indices: np.ndarray | None = None,
        origin_position: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute sparse local g2/g3 counts for a single origin atom."""
        if neighbor_indices is None:
            neighbor_indices = self._neighbor_indices[origin_index]
        neighbor_indices = np.asarray(neighbor_indices, dtype=np.intp)
        if neighbor_indices.size == 0:
            empty_i = np.empty(0, dtype=np.intp)
            empty_c = np.empty(0, dtype=np.int64)
            return empty_i, empty_c, empty_i, empty_c

        positions = self.atoms.positions
        if origin_position is None:
            origin_position = positions[origin_index]

        vectors = self._minimum_image_vectors(origin_position, positions[neighbor_indices])
        radius_sq = np.einsum("ij,ij->i", vectors, vectors)
        keep = (radius_sq > self._zero_tol) & (radius_sq < self._r_max_sq)
        if not np.all(keep):
            neighbor_indices = neighbor_indices[keep]
            vectors = vectors[keep]
            radius_sq = radius_sq[keep]

        if neighbor_indices.size == 0:
            empty_i = np.empty(0, dtype=np.intp)
            empty_c = np.empty(0, dtype=np.int64)
            return empty_i, empty_c, empty_i, empty_c

        radius = np.sqrt(radius_sq)
        radius_bin = np.floor(radius / self.measure_r_step).astype(np.intp)
        keep_bin = radius_bin < self._r_num
        if not np.all(keep_bin):
            neighbor_indices = neighbor_indices[keep_bin]
            vectors = vectors[keep_bin]
            radius_sq = radius_sq[keep_bin]
            radius_bin = radius_bin[keep_bin]

        if neighbor_indices.size == 0:
            empty_i = np.empty(0, dtype=np.intp)
            empty_c = np.empty(0, dtype=np.int64)
            return empty_i, empty_c, empty_i, empty_c

        neighbor_species = self._atom_species_index[neighbor_indices]
        center_species = int(self._atom_species_index[origin_index])

        pair_linear = (
            (center_species * self._num_species + neighbor_species) * self._r_num + radius_bin
        )
        g2_indices, g2_counts = np.unique(pair_linear, return_counts=True)

        g3_parts: list[np.ndarray] = []
        for triplet_index in self._triplets_by_center[center_species]:
            _, species_1, species_2 = self.target_distribution.g3_index[triplet_index]

            mask_1 = neighbor_species == species_1
            mask_2 = neighbor_species == species_2
            if not np.any(mask_1) or not np.any(mask_2):
                continue

            v01 = vectors[mask_1]
            v02 = vectors[mask_2]
            r01_sq = radius_sq[mask_1]
            r02_sq = radius_sq[mask_2]
            r01_bin = radius_bin[mask_1]
            r02_bin = radius_bin[mask_2]

            dot = v01 @ v02.T
            denom = np.sqrt(r01_sq[:, None] * r02_sq[None, :])
            cos_phi = np.clip(dot / np.maximum(denom, _EPS), -1.0, 1.0)
            phi_bin = np.floor(np.arccos(cos_phi) / self._phi_step).astype(np.intp)
            np.clip(phi_bin, 0, self._phi_num_bins - 1, out=phi_bin)

            rr_index = (
                (r01_bin[:, None] * self._r_num + r02_bin[None, :]) * self._phi_num_bins
            )
            linear = rr_index + phi_bin

            if species_1 == species_2:
                valid = np.ones(linear.shape, dtype=bool)
                np.fill_diagonal(valid, False)
                linear_values = linear[valid].ravel()
                if linear_values.size:
                    g3_parts.append(triplet_index * self._flat_triplet_size + linear_values)
                continue

            linear_values = linear.ravel()
            if linear_values.size:
                g3_parts.append(triplet_index * self._flat_triplet_size + linear_values)

            rr_index_sym = (
                (r02_bin[None, :] * self._r_num + r01_bin[:, None]) * self._phi_num_bins
            )
            linear_sym = (rr_index_sym + phi_bin).ravel()
            if linear_sym.size:
                g3_parts.append(triplet_index * self._flat_triplet_size + linear_sym)

        if not g3_parts:
            empty_i = np.empty(0, dtype=np.intp)
            empty_c = np.empty(0, dtype=np.int64)
            return g2_indices.astype(np.intp), g2_counts.astype(np.int64), empty_i, empty_c

        g3_linear = np.concatenate(g3_parts)
        g3_indices, g3_counts = np.unique(g3_linear, return_counts=True)
        return (
            g2_indices.astype(np.intp),
            g2_counts.astype(np.int64),
            g3_indices.astype(np.intp),
            g3_counts.astype(np.int64),
        )

    def _center_moved_sparse_contribution(
        self,
        center_index: int,
        moved_atom: int,
        *,
        moved_position: np.ndarray,
        neighbor_indices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute only the g2/g3 terms that involve a moved neighbor around one center."""
        neighbor_indices = np.asarray(neighbor_indices, dtype=np.intp)
        if center_index == moved_atom or neighbor_indices.size == 0:
            empty_i = np.empty(0, dtype=np.intp)
            empty_c = np.empty(0, dtype=np.int64)
            return empty_i, empty_c, empty_i, empty_c

        if not np.any(neighbor_indices == moved_atom):
            empty_i = np.empty(0, dtype=np.intp)
            empty_c = np.empty(0, dtype=np.int64)
            return empty_i, empty_c, empty_i, empty_c

        other_neighbors = neighbor_indices[neighbor_indices != moved_atom]
        center_position = self.atoms.positions[center_index]

        moved_vector = self._minimum_image_vectors(
            center_position,
            np.asarray(moved_position, dtype=np.float64)[None, :],
        )[0]
        moved_radius_sq = float(np.dot(moved_vector, moved_vector))
        if moved_radius_sq <= self._zero_tol or moved_radius_sq >= self._r_max_sq:
            empty_i = np.empty(0, dtype=np.intp)
            empty_c = np.empty(0, dtype=np.int64)
            return empty_i, empty_c, empty_i, empty_c

        moved_radius_bin = int(np.floor(np.sqrt(moved_radius_sq) / self.measure_r_step))
        if moved_radius_bin >= self._r_num:
            empty_i = np.empty(0, dtype=np.intp)
            empty_c = np.empty(0, dtype=np.int64)
            return empty_i, empty_c, empty_i, empty_c

        center_species = int(self._atom_species_index[center_index])
        moved_species = int(self._atom_species_index[moved_atom])
        g2_index = np.array(
            [
                (
                    (center_species * self._num_species + moved_species) * self._r_num
                    + moved_radius_bin
                )
            ],
            dtype=np.intp,
        )
        g2_count = np.array([1], dtype=np.int64)

        if other_neighbors.size == 0:
            empty_i = np.empty(0, dtype=np.intp)
            empty_c = np.empty(0, dtype=np.int64)
            return g2_index, g2_count, empty_i, empty_c

        other_vectors = self._minimum_image_vectors(center_position, self.atoms.positions[other_neighbors])
        other_radius_sq = np.einsum("ij,ij->i", other_vectors, other_vectors)
        keep = (other_radius_sq > self._zero_tol) & (other_radius_sq < self._r_max_sq)
        if not np.any(keep):
            empty_i = np.empty(0, dtype=np.intp)
            empty_c = np.empty(0, dtype=np.int64)
            return g2_index, g2_count, empty_i, empty_c

        other_neighbors = other_neighbors[keep]
        other_vectors = other_vectors[keep]
        other_radius_sq = other_radius_sq[keep]
        other_radius_bin = np.floor(np.sqrt(other_radius_sq) / self.measure_r_step).astype(np.intp)
        keep_bin = other_radius_bin < self._r_num
        if not np.any(keep_bin):
            empty_i = np.empty(0, dtype=np.intp)
            empty_c = np.empty(0, dtype=np.int64)
            return g2_index, g2_count, empty_i, empty_c

        other_neighbors = other_neighbors[keep_bin]
        other_vectors = other_vectors[keep_bin]
        other_radius_sq = other_radius_sq[keep_bin]
        other_radius_bin = other_radius_bin[keep_bin]
        other_species = self._atom_species_index[other_neighbors]

        dot = other_vectors @ moved_vector
        denom = np.sqrt(np.maximum(other_radius_sq * moved_radius_sq, _EPS))
        cos_phi = np.clip(dot / denom, -1.0, 1.0)
        phi_bin = np.floor(np.arccos(cos_phi) / self._phi_step).astype(np.intp)
        np.clip(phi_bin, 0, self._phi_num_bins - 1, out=phi_bin)

        triplet_index = self.target_distribution.g3_lookup[
            center_species,
            moved_species,
            other_species,
        ]
        linear_primary = (
            triplet_index.astype(np.intp) * self._flat_triplet_size
            + ((moved_radius_bin * self._r_num + other_radius_bin) * self._phi_num_bins + phi_bin)
        )
        linear_symmetric = (
            triplet_index.astype(np.intp) * self._flat_triplet_size
            + ((other_radius_bin * self._r_num + moved_radius_bin) * self._phi_num_bins + phi_bin)
        )
        g3_linear = np.concatenate([linear_primary, linear_symmetric]).astype(np.intp, copy=False)
        g3_indices, g3_counts = np.unique(g3_linear, return_counts=True)
        return (
            g2_index,
            g2_count,
            g3_indices.astype(np.intp),
            g3_counts.astype(np.int64),
        )

    def _combine_sparse(
        self,
        parts: list[tuple[np.ndarray, np.ndarray]],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Aggregate repeated sparse indices by summing their counts."""
        valid_parts = [(idx, val) for idx, val in parts if idx.size]
        if not valid_parts:
            return np.empty(0, dtype=np.intp), np.empty(0, dtype=np.int64)

        indices = np.concatenate([idx for idx, _ in valid_parts])
        values = np.concatenate([val for _, val in valid_parts]).astype(np.int64, copy=False)
        unique, inverse = np.unique(indices, return_inverse=True)
        summed = np.bincount(inverse, weights=values).astype(np.int64)
        keep = summed != 0
        return unique[keep].astype(np.intp), summed[keep]

    def _sparse_delta(
        self,
        old_indices: np.ndarray,
        old_values: np.ndarray,
        new_indices: np.ndarray,
        new_values: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return sparse `new - old` counts."""
        return self._combine_sparse(
            [
                (new_indices, new_values.astype(np.int64, copy=False)),
                (old_indices, -old_values.astype(np.int64, copy=False)),
            ]
        )

    def _proposal_neighbor_indices(
        self,
        center_index: int,
        moved_atom: int,
        old_neighbor_set: set[int],
        new_neighbor_set: set[int],
        new_neighbors_for_moved_atom: np.ndarray,
    ) -> np.ndarray:
        """Return the proposed neighbor list for an affected center."""
        if center_index == moved_atom:
            return new_neighbors_for_moved_atom

        current = self._neighbor_indices[center_index]
        was_neighbor = center_index in old_neighbor_set
        is_neighbor = center_index in new_neighbor_set
        if was_neighbor == is_neighbor:
            return current
        if is_neighbor:
            return np.append(current, moved_atom).astype(np.intp, copy=False)
        return current[current != moved_atom]

    def _prepare_move_delta(
        self,
        atom_index: int,
        new_position: np.ndarray,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        list[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """Compute sparse g2/g3 deltas for a proposed atomic displacement."""
        old_position = np.array(self.atoms.positions[atom_index], copy=True)
        old_neighbors = self._neighbor_indices[atom_index]
        new_neighbors = self._query_neighbors_for_position(atom_index, new_position)
        old_neighbor_set = set(old_neighbors.tolist())
        new_neighbor_set = set(new_neighbors.tolist())
        affected = np.unique(
            np.concatenate(
                [
                    np.array([atom_index], dtype=np.intp),
                    old_neighbors,
                    new_neighbors,
                ]
            )
        )

        moved_old_cache = self._origin_contribution_cache[atom_index]
        moved_new_cache = self._origin_sparse_contribution(
            atom_index,
            neighbor_indices=new_neighbors,
            origin_position=new_position,
        )
        moved_delta_g2_idx, moved_delta_g2_val = self._sparse_delta(
            moved_old_cache[0],
            moved_old_cache[1],
            moved_new_cache[0],
            moved_new_cache[1],
        )
        moved_delta_g3_idx, moved_delta_g3_val = self._sparse_delta(
            moved_old_cache[2],
            moved_old_cache[3],
            moved_new_cache[2],
            moved_new_cache[3],
        )
        delta_g2_parts: list[tuple[np.ndarray, np.ndarray]] = [
            (moved_delta_g2_idx, moved_delta_g2_val)
        ]
        delta_g3_parts: list[tuple[np.ndarray, np.ndarray]] = [
            (moved_delta_g3_idx, moved_delta_g3_val)
        ]
        cache_delta_updates: list[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

        for center_index in affected:
            center_index = int(center_index)
            if center_index == atom_index:
                continue

            current_neighbors = self._neighbor_indices[center_index]
            proposal_neighbors = self._proposal_neighbor_indices(
                center_index,
                atom_index,
                old_neighbor_set,
                new_neighbor_set,
                new_neighbors,
            )

            old_partial = self._center_moved_sparse_contribution(
                center_index,
                atom_index,
                moved_position=old_position,
                neighbor_indices=current_neighbors,
            )
            new_partial = self._center_moved_sparse_contribution(
                center_index,
                atom_index,
                moved_position=new_position,
                neighbor_indices=proposal_neighbors,
            )

            delta_center_g2_idx, delta_center_g2_val = self._sparse_delta(
                old_partial[0],
                old_partial[1],
                new_partial[0],
                new_partial[1],
            )
            delta_center_g3_idx, delta_center_g3_val = self._sparse_delta(
                old_partial[2],
                old_partial[3],
                new_partial[2],
                new_partial[3],
            )
            delta_g2_parts.append((delta_center_g2_idx, delta_center_g2_val))
            delta_g3_parts.append((delta_center_g3_idx, delta_center_g3_val))
            cache_delta_updates.append(
                (
                    center_index,
                    delta_center_g2_idx,
                    delta_center_g2_val,
                    delta_center_g3_idx,
                    delta_center_g3_val,
                )
            )

        delta_g2_idx, delta_g2_val = self._combine_sparse(delta_g2_parts)
        delta_g3_idx, delta_g3_val = self._combine_sparse(delta_g3_parts)
        return (
            affected,
            new_neighbors,
            moved_new_cache,
            cache_delta_updates,
            delta_g2_idx,
            delta_g2_val,
            delta_g3_idx,
            delta_g3_val,
        )

    def _apply_neighbor_updates(
        self,
        atom_index: int,
        affected: np.ndarray,
        new_neighbors: np.ndarray,
    ) -> None:
        """Commit the accepted neighbor-table changes after a move."""
        old_neighbor_set = set(self._neighbor_indices[atom_index].tolist())
        new_neighbor_set = set(new_neighbors.tolist())
        self._neighbor_indices[atom_index] = np.array(new_neighbors, dtype=np.intp, copy=True)

        for center_index in affected:
            center_index = int(center_index)
            if center_index == atom_index:
                continue
            was_neighbor = center_index in old_neighbor_set
            is_neighbor = center_index in new_neighbor_set
            if was_neighbor == is_neighbor:
                continue
            current = self._neighbor_indices[center_index]
            if is_neighbor:
                self._neighbor_indices[center_index] = np.append(current, atom_index).astype(
                    np.intp,
                    copy=False,
                )
            else:
                self._neighbor_indices[center_index] = current[current != atom_index]

    def _apply_origin_cache_updates(
        self,
        atom_index: int,
        moved_new_cache: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        cache_delta_updates: list[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    ) -> None:
        """Commit updated sparse per-origin contributions after an accepted move."""
        self._origin_contribution_cache[int(atom_index)] = moved_new_cache
        for (
            center_index,
            delta_g2_idx,
            delta_g2_val,
            delta_g3_idx,
            delta_g3_val,
        ) in cache_delta_updates:
            old_cache = self._origin_contribution_cache[int(center_index)]
            new_full_g2_idx, new_full_g2_val = self._combine_sparse(
                [
                    (old_cache[0], old_cache[1]),
                    (delta_g2_idx, delta_g2_val),
                ]
            )
            new_full_g3_idx, new_full_g3_val = self._combine_sparse(
                [
                    (old_cache[2], old_cache[3]),
                    (delta_g3_idx, delta_g3_val),
                ]
            )
            self._origin_contribution_cache[int(center_index)] = (
                new_full_g2_idx,
                new_full_g2_val,
                new_full_g3_idx,
                new_full_g3_val,
            )

    def measure_g3(
        self,
        *,
        force: bool = False,
        show_progress: bool = False,
    ) -> G3Distribution:
        """Measure the current random supercell on the target distribution grid.

        The current implementation always uses the full target discretization so
        the supercell and target histograms can be compared bin-for-bin in raw
        count space during local Monte Carlo updates.

        Parameters
        ----------
        force
            If `True`, discard any cached measurement and recompute the current
            supercell distribution.
        show_progress
            If `True`, display a text progress bar while the supercell histogram
            is accumulated.
        """
        if self.current_distribution is not None and not force:
            return self.current_distribution

        measured = G3Distribution(self.atoms, label=f"{self.label}-measured")
        measured.measure_g3(
            r_max=self.measure_r_max,
            r_step=self.measure_r_step,
            phi_num_bins=self.measure_phi_num_bins,
            show_progress=show_progress,
            progress_label=f"Measuring g3 in {self.label}",
        )
        self.current_distribution = measured
        return measured

    def sync_g3(self, *, show_progress: bool = True) -> G3Distribution:
        """Recompute the supercell g2/g3 from scratch and rebuild MC caches."""
        measured = self.measure_g3(force=True, show_progress=show_progress)
        self._initialize_mc_state()
        return measured

    def plot_g3_compare(
        self,
        pair: int | str = 0,
        *,
        normalize: bool = True,
    ):
        """Return an interactive comparison between the current supercell and target."""
        current = self.measure_g3()
        pair_index = self.target_distribution._resolve_pair_index(pair)

        from .g3_compare_widget import G3CompareWidget

        return G3CompareWidget(
            current_distribution=current,
            target_distribution=self.target_distribution,
            triplet_index=pair_index,
            normalize=normalize,
            supercell_title=f"{self.label} g3 slice",
            target_title=f"{self.target_distribution.label} g3 slice",
            status_prefix=(
                f"density {self.relative_density:.3f} | "
                "cell_dim_angstroms "
                f"{self.cell_dim_angstroms[0]:.2f}x"
                f"{self.cell_dim_angstroms[1]:.2f}x"
                f"{self.cell_dim_angstroms[2]:.2f} A"
            ),
        )

    def _display_compare_widget(self) -> None:
        """Display the comparison widget immediately when running in IPython."""
        try:
            from IPython.display import display
        except Exception:
            return
        display(self.plot_g3_compare())

    def plot_monte_carlo(self):
        """Plot the recorded Monte Carlo cost history using Matplotlib."""
        if self.mc_history is None:
            raise ValueError("Run monte_carlo() before plotting the history.")

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.8, 4.2))
        ax.plot(self.mc_history["step"], self.mc_history["cost"], lw=1.8, label="cost")
        ax.plot(self.mc_history["step"], self.mc_history["best_cost"], lw=1.4, label="best")
        ax.set_xlabel("step")
        ax.set_ylabel("cost")
        ax.set_title("Monte Carlo cost history")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        return fig, ax

    def repulsion(
        self,
        num_steps: int = 24,
        *,
        step_size: float | None = None,
        cutoff: float | None = None,
        sync_g3: bool = True,
        show_progress: bool = True,
    ) -> dict[str, float | int]:
        """Spread atoms apart by repeatedly stepping opposite each site's nearest neighbor.

        This is a lightweight preconditioning step for the random initialization.
        Each sweep finds the nearest periodic neighbor of every atom, moves each
        atom by a small Cartesian step opposite that bond direction, wraps the new
        coordinates back into the supercell, and optionally remeasures `g2/g3`
        afterward so Monte Carlo starts from a synchronized state.

        Parameters
        ----------
        num_steps
            Number of nearest-neighbor repulsion sweeps.
        step_size
            Cartesian step length applied on each sweep. If omitted, a small
            fraction of the mean inter-atomic spacing is used.
        cutoff
            Neighbor-search cutoff. If omitted, a density-based value is chosen
            automatically and any atoms with no neighbor inside that radius fall
            back to an exact minimum-image search.
        sync_g3
            If `True`, rebuild the measured supercell `g2/g3` after the repulsion
            sweeps so later Monte Carlo moves start from a synchronized histogram.
        show_progress
            If `True`, display text progress bars for the repulsion sweeps and the
            optional `g2/g3` resynchronization.

        Returns
        -------
        dict[str, float | int]
            Summary containing the sweep count, step size, and nearest-neighbor
            distances before and after the repulsion pass.
        """
        num_steps = int(num_steps)
        if num_steps <= 0:
            raise ValueError("num_steps must be positive.")

        num_atoms = len(self.atoms)
        mean_spacing = float((self.atoms.cell.volume / max(num_atoms, 1)) ** (1.0 / 3.0))
        if step_size is None:
            step_size = 0.08 * mean_spacing
        step_size = float(step_size)
        if step_size <= 0:
            raise ValueError("step_size must be positive.")

        cell_lengths = np.linalg.norm(self._cell_matrix, axis=1)
        max_cutoff = max(0.45 * float(np.min(cell_lengths)), self.measure_r_step)
        if cutoff is None:
            cutoff = min(max_cutoff, max(3.0 * mean_spacing, 4.0 * step_size))
        cutoff = float(cutoff)
        if cutoff <= 0:
            raise ValueError("cutoff must be positive.")

        def nearest_neighbor_vectors() -> tuple[np.ndarray, np.ndarray]:
            i, j, d, D = neighbor_list(
                "ijdD",
                self.atoms,
                cutoff,
                self_interaction=False,
            )
            nearest_sq = np.full(num_atoms, np.inf, dtype=np.float64)
            nearest_vec = np.zeros((num_atoms, 3), dtype=np.float64)

            if i.size:
                dist_sq = d * d
                order = np.lexsort((dist_sq, i))
                i_sorted = i[order]
                dist_sq_sorted = dist_sq[order]
                vec_sorted = D[order]
                first = np.unique(i_sorted, return_index=True)[1]
                center_index = i_sorted[first]
                nearest_sq[center_index] = dist_sq_sorted[first]
                nearest_vec[center_index] = vec_sorted[first]

            missing = np.flatnonzero(~np.isfinite(nearest_sq))
            if missing.size:
                positions = self.atoms.positions
                for atom_index in missing:
                    vectors = self._minimum_image_vectors(positions[atom_index], positions)
                    radius_sq = np.einsum("ij,ij->i", vectors, vectors)
                    radius_sq[atom_index] = np.inf
                    nn_index = int(np.argmin(radius_sq))
                    nearest_sq[atom_index] = radius_sq[nn_index]
                    nearest_vec[atom_index] = vectors[nn_index]

            nearest_dist = np.sqrt(np.maximum(nearest_sq, 0.0))
            return nearest_dist, nearest_vec

        initial_nn, _ = nearest_neighbor_vectors()
        progress = None
        if show_progress:
            progress = _TextProgressBar(num_steps, label="Nearest-neighbor repulsion", width=28)
            progress.update(0)

        for step in range(1, num_steps + 1):
            _, nearest_vec = nearest_neighbor_vectors()
            norm = np.linalg.norm(nearest_vec, axis=1, keepdims=True)
            unit = np.divide(
                nearest_vec,
                np.maximum(norm, _EPS),
                out=np.zeros_like(nearest_vec),
            )
            self.atoms.positions[:] = self._wrap_positions(
                self.atoms.positions - step_size * unit
            )
            if progress is not None:
                progress.update(step)

        if progress is not None:
            progress.close()

        final_nn, _ = nearest_neighbor_vectors()

        if sync_g3:
            self.sync_g3(show_progress=show_progress)

        return {
            "num_steps": num_steps,
            "step_size": step_size,
            "cutoff": cutoff,
            "initial_nn_min": float(np.min(initial_nn)),
            "initial_nn_mean": float(np.mean(initial_nn)),
            "final_nn_min": float(np.min(final_nn)),
            "final_nn_mean": float(np.mean(final_nn)),
        }

    def monte_carlo(
        self,
        num_steps: int = 1_000,
        temperature: float = 0.0,
        *,
        jump_size: float | None = None,
        r_min_nn: float | None = None,
        attempt_prob: float = 1.0,
        plot_history: bool = True,
        show_progress: bool = True,
        swap_freq: float | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run a first local-update Monte Carlo loop on the random supercell.

        Parameters
        ----------
        num_steps
            Number of Monte Carlo steps to attempt.
        temperature
            Metropolis temperature. At `0`, only downhill or equal-`delta_cost`
            moves are accepted. At finite temperature, uphill moves are accepted
            with probability `exp(-delta_cost / temperature)`.
        jump_size
            Standard deviation of the Cartesian Gaussian displacement applied to
            trial moves. Defaults to the current radial bin width.
        r_min_nn
            Optional hard-core nearest-neighbor distance. Trial positions that
            land within this distance of any other atom are iteratively pushed
            away before their `g2/g3` deltas are evaluated.
        attempt_prob
            Probability of attempting a positional jump on any given Monte Carlo
            step. Set this to `1.0` to attempt one move per step.
        plot_history
            If `True`, immediately create a Matplotlib cost-history plot at the
            end of the run.
        show_progress
            If `True`, display a text progress bar over Monte Carlo steps.
        swap_freq
            Deprecated alias for `attempt_prob`, kept temporarily for backwards
            compatibility with older notebooks.
        **kwargs
            Additional Monte Carlo options preserved in the returned summary.

        Returns
        -------
        dict[str, Any]
            Summary of the Monte Carlo run, including the final and best costs.
        """
        num_steps = int(num_steps)
        if num_steps <= 0:
            raise ValueError("num_steps must be positive.")
        if temperature < 0:
            raise ValueError("temperature must be non-negative.")
        if swap_freq is not None:
            attempt_prob = float(swap_freq)
        if not (0.0 <= attempt_prob <= 1.0):
            raise ValueError("attempt_prob must lie between 0 and 1.")

        if jump_size is None:
            jump_size = float(self.measure_r_step)
        if jump_size <= 0:
            raise ValueError("jump_size must be positive.")
        if r_min_nn is not None and r_min_nn <= 0:
            raise ValueError("r_min_nn must be positive when provided.")

        self.measure_g3()
        if self.current_cost is None:
            self._initialize_mc_state()

        num_atoms = len(self.atoms)
        steps = np.arange(num_steps + 1, dtype=np.int32)
        cost_history = np.empty(num_steps + 1, dtype=np.float64)
        best_history = np.empty(num_steps + 1, dtype=np.float64)
        accepted_history = np.zeros(num_steps + 1, dtype=np.float32)
        attempted_history = np.zeros(num_steps + 1, dtype=np.float32)
        repelled_history = np.zeros(num_steps + 1, dtype=np.float32)
        unresolved_history = np.zeros(num_steps + 1, dtype=np.float32)

        current_cost = float(self.current_cost)
        best_cost = float(current_cost)
        cost_history[0] = current_cost
        best_history[0] = best_cost

        progress = None
        if show_progress:
            progress = _TextProgressBar(num_steps, label="Monte Carlo", width=28)
            progress.update(0)

        for step in range(1, num_steps + 1):
            accepted = False
            attempted = False
            repelled = False
            unresolved = False

            if self.rng.random() <= attempt_prob:
                attempted = True
                atom_index = int(self.rng.integers(0, num_atoms))
                old_position = np.array(self.atoms.positions[atom_index], copy=True)
                new_position = self._wrap_position(
                    old_position + self.rng.normal(scale=jump_size, size=3)
                )
                new_position, resolved, repel_iter = self._repel_trial_position(
                    atom_index,
                    new_position,
                    r_min_nn,
                )
                repelled = repel_iter > 0
                unresolved = not resolved

                if resolved:
                    (
                        affected,
                        new_neighbors,
                        moved_new_cache,
                        cache_delta_updates,
                        delta_g2_idx,
                        delta_g2_val,
                        delta_g3_idx,
                        delta_g3_val,
                    ) = self._prepare_move_delta(atom_index, new_position)

                    if delta_g3_idx.size:
                        delta_g3 = delta_g3_val.astype(np.float64, copy=False)
                        delta_cost = float(
                            2.0 * np.dot(self._g3_diff_flat[delta_g3_idx], delta_g3)
                            + np.dot(delta_g3, delta_g3)
                        )
                    else:
                        delta_cost = 0.0

                    if temperature == 0.0:
                        accepted = delta_cost <= 0.0
                    else:
                        accepted = delta_cost <= 0.0 or self.rng.random() < np.exp(
                            -delta_cost / max(temperature, _EPS)
                        )

                    if accepted:
                        self.atoms.positions[atom_index] = new_position
                        if delta_g2_idx.size:
                            delta_g2_int = delta_g2_val.astype(np.int64, copy=False)
                            self._current_g2_flat[delta_g2_idx] += delta_g2_int
                            self._g2_diff_flat[delta_g2_idx] += delta_g2_int.astype(np.float64)
                        if delta_g3_idx.size:
                            delta_g3_int = delta_g3_val.astype(np.int64, copy=False)
                            self._current_g3_flat[delta_g3_idx] += delta_g3_int
                            self._g3_diff_flat[delta_g3_idx] += delta_g3_int.astype(np.float64)
                        self._apply_neighbor_updates(atom_index, affected, new_neighbors)
                        self._apply_origin_cache_updates(
                            atom_index,
                            moved_new_cache,
                            cache_delta_updates,
                        )
                        current_cost += delta_cost
                        best_cost = min(best_cost, current_cost)

            cost_history[step] = current_cost
            best_history[step] = best_cost
            accepted_history[step] = float(accepted)
            attempted_history[step] = float(attempted)
            repelled_history[step] = float(repelled)
            unresolved_history[step] = float(unresolved)
            if progress is not None:
                progress.update(step)

        if progress is not None:
            progress.close()

        self.current_cost = float(current_cost)
        self.best_score = float(best_cost)
        self.last_temperature = float(temperature)
        self.current_distribution.atoms = self.atoms.copy()

        self.mc_history = {
            "step": steps,
            "cost": cost_history.astype(np.float64),
            "best_cost": best_history.astype(np.float64),
            "accepted": accepted_history,
            "attempted": attempted_history,
            "repelled": repelled_history,
            "unresolved": unresolved_history,
        }

        summary = {
            "num_steps": num_steps,
            "temperature": float(temperature),
            "jump_size": float(jump_size),
            "r_min_nn": None if r_min_nn is None else float(r_min_nn),
            "attempt_prob": float(attempt_prob),
            "attempted_moves": int(np.sum(attempted_history)),
            "accepted_moves": int(np.sum(accepted_history)),
            "repelled_moves": int(np.sum(repelled_history)),
            "unresolved_moves": int(np.sum(unresolved_history)),
            "final_cost": float(current_cost),
            "best_cost": float(best_cost),
            "cell_dim_angstroms": self.cell_dim_angstroms,
            "relative_density": self.relative_density,
            "measure_r_max": self.measure_r_max,
            "measure_r_step": self.measure_r_step,
            "measure_phi_num_bins": self.measure_phi_num_bins,
            "num_atoms": num_atoms,
        }
        if kwargs:
            summary["mc_kwargs"] = dict(kwargs)

        if plot_history:
            self.plot_monte_carlo()
        return summary

    def __repr__(self) -> str:
        atom_count = len(self.atoms)
        return (
            f"Supercell(label={self.label!r}, cell_dim_angstroms={self.cell_dim_angstroms}, "
            f"atoms={atom_count}, relative_density={self.relative_density:.3f}, "
            f"measure_r_max={self.measure_r_max:.3f}, best_score={self.best_score})"
        )
