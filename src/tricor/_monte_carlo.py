from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from ase.neighborlist import neighbor_list

from .g3 import G3Distribution, _EPS, _TextProgressBar
from .shells import CoordinationShellTarget

if TYPE_CHECKING:
    from .supercell import Supercell


class _MonteCarloMixin:
    # ------------------------------------------------------------------
    # Cost / weights
    # ------------------------------------------------------------------

    def _distribution_scale(self: "Supercell", *, order: int) -> float:
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

    def _build_g3_rr_weights(self: "Supercell") -> np.ndarray:
        """Build flattened radial-pair weights for the full-g3 Monte Carlo cost."""
        radii = np.asarray(self.target_distribution.bin_centers, dtype=np.float64)
        r01, r02 = np.meshgrid(radii, radii, indexing="ij")
        r_eff = np.maximum(r01, r02)
        if self.g3_weight_exponent <= _EPS:
            weights = np.ones_like(r_eff, dtype=np.float64)
        else:
            scaled = np.power(
                r_eff / max(self.g3_weight_r_scale, _EPS),
                self.g3_weight_exponent,
            )
            weights = self.g3_weight_floor + (1.0 - self.g3_weight_floor) / (1.0 + scaled)
        weights /= max(float(np.mean(weights)), _EPS)
        return weights.reshape(-1).astype(np.float64, copy=False)

    def _weighted_g3_cost(self: "Supercell", g3_diff_flat: np.ndarray) -> float:
        """Return the weighted full-g3 squared-error cost."""
        g3_diff_view = np.asarray(g3_diff_flat, dtype=np.float64).reshape(
            self._num_triplets,
            self._r_num * self._r_num,
            self._phi_num_bins,
        )
        return float(
            np.einsum(
                "r,trp,trp->",
                self._g3_rr_weights_flat,
                g3_diff_view,
                g3_diff_view,
                optimize=True,
            )
        )

    def _weighted_delta_cost(self: "Supercell", delta_g3_idx: np.ndarray, delta_g3_val: np.ndarray) -> float:
        """Return the change in weighted full-g3 cost for a sparse histogram delta."""
        if delta_g3_idx.size == 0:
            return 0.0
        delta_g3 = delta_g3_val.astype(np.float64, copy=False)
        rr_index = (
            np.asarray(delta_g3_idx, dtype=np.intp) % self._flat_triplet_size
        ) // self._phi_num_bins
        weights = self._g3_rr_weights_flat[rr_index]
        current = self._g3_diff_flat[delta_g3_idx]
        return float(
            2.0 * np.dot(weights * current, delta_g3)
            + np.dot(weights * delta_g3, delta_g3)
        )

    def _cell_face_spacings(self: "Supercell", cell_matrix: np.ndarray) -> np.ndarray:
        """Return periodic face spacings for the cell spanned by the row vectors."""
        inverse = np.linalg.inv(np.asarray(cell_matrix, dtype=np.float64))
        return 1.0 / np.maximum(np.linalg.norm(inverse, axis=0), _EPS)

    # ------------------------------------------------------------------
    # Spatial indexing
    # ------------------------------------------------------------------

    def _wrapped_fractional_positions(self: "Supercell", positions: np.ndarray) -> np.ndarray:
        """Return wrapped fractional coordinates in `[0, 1)` for Cartesian positions."""
        frac = np.asarray(positions, dtype=np.float64) @ self._cell_inverse
        frac %= 1.0
        return frac

    def _spatial_flat_index(self: "Supercell", cell_index: np.ndarray) -> int:
        """Flatten a 3D spatial-hash cell index."""
        return int(np.ravel_multi_index(tuple(np.asarray(cell_index, dtype=np.intp)), self._spatial_shape))

    def _spatial_cell_index_for_position(self: "Supercell", position: np.ndarray) -> np.ndarray:
        """Map a Cartesian position to the wrapped spatial-hash bin index."""
        frac = self._wrapped_fractional_positions(np.asarray(position, dtype=np.float64)[None, :])[0]
        cell_index = np.floor(frac * self._spatial_shape[None, :]).astype(np.intp)[0]
        np.minimum(cell_index, self._spatial_shape - 1, out=cell_index)
        return cell_index

    def _spatial_search_ranges(self: "Supercell", cutoff: float) -> np.ndarray:
        """Return the number of neighboring spatial bins to inspect along each axis."""
        cutoff = float(max(cutoff, 0.0))
        return np.maximum(
            1,
            np.ceil(cutoff / np.maximum(self._spatial_bin_face_spacings, _EPS)).astype(np.intp),
        )

    def _spatial_neighbor_offsets(self: "Supercell", search_ranges: np.ndarray) -> np.ndarray:
        """Return cached spatial-bin neighbor offsets for the requested search ranges."""
        key = tuple(int(value) for value in np.asarray(search_ranges, dtype=np.intp))
        offsets = self._spatial_offset_cache.get(key)
        if offsets is not None:
            return offsets
        mesh = np.meshgrid(
            np.arange(-key[0], key[0] + 1, dtype=np.intp),
            np.arange(-key[1], key[1] + 1, dtype=np.intp),
            np.arange(-key[2], key[2] + 1, dtype=np.intp),
            indexing="ij",
        )
        offsets = np.stack([axis.ravel() for axis in mesh], axis=1)
        self._spatial_offset_cache[key] = offsets
        return offsets

    def _rebuild_spatial_index(self: "Supercell") -> None:
        """Rebuild the periodic spatial hash for local Monte Carlo queries."""
        cell_face_spacings = self._cell_face_spacings(self._cell_matrix)
        self._spatial_shape = np.maximum(
            1,
            np.floor(cell_face_spacings / max(self.spatial_bin_size, _EPS)).astype(np.intp),
        )
        bin_matrix = self._cell_matrix / self._spatial_shape[:, None]
        self._spatial_bin_face_spacings = self._cell_face_spacings(bin_matrix)
        num_bins = int(np.prod(self._spatial_shape, dtype=np.int64))
        self._spatial_bins: list[list[int]] = [[] for _ in range(num_bins)]
        frac = self._wrapped_fractional_positions(self.atoms.positions)
        cell_indices = np.floor(frac * self._spatial_shape[None, :]).astype(np.intp)
        np.minimum(cell_indices, self._spatial_shape[None, :] - 1, out=cell_indices)
        self._atom_spatial_cells = cell_indices.astype(np.intp, copy=True)
        flat_indices = np.ravel_multi_index(cell_indices.T, self._spatial_shape)
        for atom_index, flat_index in enumerate(flat_indices):
            self._spatial_bins[int(flat_index)].append(int(atom_index))

    def _update_spatial_index_for_atom(
        self: "Supercell",
        atom_index: int,
        old_position: np.ndarray,
        new_position: np.ndarray,
    ) -> None:
        """Update the spatial hash after an accepted single-atom move."""
        old_cell = self._spatial_cell_index_for_position(old_position)
        new_cell = self._spatial_cell_index_for_position(new_position)
        if np.array_equal(old_cell, new_cell):
            self._atom_spatial_cells[int(atom_index)] = new_cell
            return
        old_flat = self._spatial_flat_index(old_cell)
        new_flat = self._spatial_flat_index(new_cell)
        self._spatial_bins[old_flat].remove(int(atom_index))
        self._spatial_bins[new_flat].append(int(atom_index))
        self._atom_spatial_cells[int(atom_index)] = new_cell

    def _candidate_indices_for_position(self: "Supercell", position: np.ndarray, cutoff: float) -> np.ndarray:
        """Return candidate atom indices from nearby spatial-hash bins."""
        base_cell = self._spatial_cell_index_for_position(position)
        search_ranges = self._spatial_search_ranges(cutoff)
        offsets = self._spatial_neighbor_offsets(search_ranges)
        bin_ids: set[int] = set()
        for offset in offsets:
            neighbor_cell = (base_cell + offset) % self._spatial_shape
            bin_ids.add(self._spatial_flat_index(neighbor_cell))
        candidates: list[int] = []
        for flat_index in bin_ids:
            candidates.extend(self._spatial_bins[flat_index])
        if not candidates:
            return np.empty(0, dtype=np.intp)
        return np.asarray(candidates, dtype=np.intp)

    def _query_local_environment(
        self: "Supercell",
        atom_index: int,
        origin_position: np.ndarray,
        cutoff: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return exact local neighbors, vectors, and squared radii within `cutoff`."""
        candidate_indices = self._candidate_indices_for_position(origin_position, cutoff)
        if candidate_indices.size == 0:
            empty_i = np.empty(0, dtype=np.intp)
            empty_f = np.empty((0, 3), dtype=np.float64)
            return empty_i, empty_f, np.empty(0, dtype=np.float64)
        keep_not_self = candidate_indices != int(atom_index)
        candidate_indices = candidate_indices[keep_not_self]
        if candidate_indices.size == 0:
            empty_i = np.empty(0, dtype=np.intp)
            empty_f = np.empty((0, 3), dtype=np.float64)
            return empty_i, empty_f, np.empty(0, dtype=np.float64)
        vectors = self._minimum_image_vectors(origin_position, self.atoms.positions[candidate_indices])
        radius_sq = np.einsum("ij,ij->i", vectors, vectors)
        cutoff_sq = float(cutoff * cutoff)
        keep = (radius_sq > self._zero_tol) & (radius_sq < cutoff_sq)
        if not np.any(keep):
            empty_i = np.empty(0, dtype=np.intp)
            empty_f = np.empty((0, 3), dtype=np.float64)
            return empty_i, empty_f, np.empty(0, dtype=np.float64)
        return (
            candidate_indices[keep].astype(np.intp, copy=False),
            vectors[keep],
            radius_sq[keep],
        )

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    def _nearest_neighbor_vectors(self: "Supercell", cutoff: float) -> tuple[np.ndarray, np.ndarray]:
        """Return each atom's nearest-neighbor distance and displacement vector."""
        num_atoms = len(self.atoms)
        i, j, d, D = neighbor_list(
            "ijdD",
            self.atoms,
            float(cutoff),
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

    def _minimum_image_position_delta(
        self: "Supercell",
        start_positions: np.ndarray,
        end_positions: np.ndarray,
    ) -> np.ndarray:
        """Return minimum-image Cartesian displacements from `start` to `end`."""
        delta = np.asarray(end_positions, dtype=np.float64) - np.asarray(start_positions, dtype=np.float64)
        frac = delta @ self._cell_inverse
        frac -= np.rint(frac)
        return frac @ self._cell_matrix
    # ------------------------------------------------------------------

    def _initialize_mc_state(self: "Supercell") -> None:
        """Cache neighbor tables, target histograms, and loss vectors."""
        if self.current_distribution is None:
            raise ValueError("Measure the supercell distribution before initializing MC state.")

        self._rebuild_spatial_index()
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
        self.current_cost = self._weighted_g3_cost(self._g3_diff_flat)
        self.best_score = self.current_cost

    def _build_neighbor_indices(self: "Supercell") -> list[np.ndarray]:
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
        self: "Supercell",
        origin_position: np.ndarray,
        target_positions: np.ndarray,
    ) -> np.ndarray:
        """Return minimum-image displacement vectors from origin to the targets."""
        delta = target_positions - origin_position[None, :]
        frac = delta @ self._cell_inverse
        frac -= np.rint(frac)
        return frac @ self._cell_matrix

    def _wrap_position(self: "Supercell", position: np.ndarray) -> np.ndarray:
        """Wrap a Cartesian position back into the periodic supercell."""
        frac = np.asarray(position, dtype=np.float64) @ self._cell_inverse
        frac %= 1.0
        return frac @ self._cell_matrix

    def _wrap_positions(self: "Supercell", positions: np.ndarray) -> np.ndarray:
        """Wrap an array of Cartesian positions back into the periodic supercell."""
        frac = np.asarray(positions, dtype=np.float64) @ self._cell_inverse
        frac %= 1.0
        return frac @ self._cell_matrix

    def _query_neighbors_for_position(
        self: "Supercell",
        atom_index: int,
        origin_position: np.ndarray,
    ) -> np.ndarray:
        """Find all neighbors within `r_max` of a proposed origin position."""
        neighbor_indices, _, _ = self._query_local_environment(
            atom_index,
            origin_position,
            self.measure_r_max,
        )
        return neighbor_indices

    def _repel_trial_position(
        self: "Supercell",
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

        for iter_count in range(1, max_iter + 1):
            _, close_vectors, close_radius_sq = self._query_local_environment(
                atom_index,
                position,
                r_min_nn,
            )
            if close_radius_sq.size == 0:
                return position, True, iter_count - 1

            close_radius = np.sqrt(np.maximum(close_radius_sq, _EPS))
            overlap = r_min_nn - close_radius
            direction = -close_vectors / close_radius[:, None]
            displacement = np.sum(direction * overlap[:, None], axis=0)
            position = self._wrap_position(position + displacement)

        _, _, close_radius_sq = self._query_local_environment(
            atom_index,
            position,
            r_min_nn,
        )
        return position, close_radius_sq.size == 0, max_iter

    def _random_perpendicular_vector(self: "Supercell", vector: np.ndarray) -> np.ndarray:
        """Return a random unit vector perpendicular to `vector`."""
        vector = np.asarray(vector, dtype=np.float64)
        candidate = self.rng.normal(size=3)
        candidate -= vector * (np.dot(candidate, vector) / max(np.dot(vector, vector), _EPS))
        norm = float(np.linalg.norm(candidate))
        if norm > _EPS:
            return candidate / norm
        fallback = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(fallback, vector))) / max(float(np.linalg.norm(vector)), _EPS) > 0.8:
            fallback = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        candidate = fallback - vector * (np.dot(fallback, vector) / max(np.dot(vector, vector), _EPS))
        norm = float(np.linalg.norm(candidate))
        if norm <= _EPS:
            return np.array([0.0, 0.0, 1.0], dtype=np.float64)
        return candidate / norm

    def _axis_angle_rotation_matrix(
        self: "Supercell",
        axis: np.ndarray,
        angle_rad: float,
    ) -> np.ndarray:
        """Return a column-action rotation matrix for an axis-angle rotation."""
        axis = np.asarray(axis, dtype=np.float64)
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm <= _EPS or abs(float(angle_rad)) <= _EPS:
            return np.eye(3, dtype=np.float64)
        axis = axis / axis_norm
        x, y, z = axis
        cos_a = float(np.cos(angle_rad))
        sin_a = float(np.sin(angle_rad))
        one_minus = 1.0 - cos_a
        return np.array(
            [
                [cos_a + x * x * one_minus, x * y * one_minus - z * sin_a, x * z * one_minus + y * sin_a],
                [y * x * one_minus + z * sin_a, cos_a + y * y * one_minus, y * z * one_minus - x * sin_a],
                [z * x * one_minus - y * sin_a, z * y * one_minus + x * sin_a, cos_a + z * z * one_minus],
            ],
            dtype=np.float64,
        )

    def _apply_rotation(
        self: "Supercell",
        vectors: np.ndarray,
        rotation: np.ndarray,
    ) -> np.ndarray:
        """Apply a column-action rotation matrix to one or more vectors."""
        vectors = np.asarray(vectors, dtype=np.float64)
        rotation = np.asarray(rotation, dtype=np.float64)
        if vectors.ndim == 1:
            return rotation @ vectors
        return vectors @ rotation.T

    def _random_rotation_matrix(self: "Supercell") -> np.ndarray:
        """Return a uniform random 3D rotation matrix."""
        q = self.rng.normal(size=4)
        q_norm = float(np.linalg.norm(q))
        if q_norm <= _EPS:
            return np.eye(3, dtype=np.float64)
        q /= q_norm
        w, x, y, z = q
        return np.array(
            [
                [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
                [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
                [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )

    def _rotation_align_vector(
        self: "Supercell",
        source: np.ndarray,
        target: np.ndarray,
        *,
        random_spin: bool,
    ) -> np.ndarray:
        """Return a rotation that maps `source` onto `target`."""
        source = np.asarray(source, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        source_norm = float(np.linalg.norm(source))
        target_norm = float(np.linalg.norm(target))
        if source_norm <= _EPS or target_norm <= _EPS:
            return self._random_rotation_matrix() if random_spin else np.eye(3, dtype=np.float64)
        source_unit = source / source_norm
        target_unit = target / target_norm
        cross = np.cross(source_unit, target_unit)
        cross_norm = float(np.linalg.norm(cross))
        dot = float(np.clip(np.dot(source_unit, target_unit), -1.0, 1.0))
        if cross_norm <= _EPS:
            if dot > 0.0:
                rotation = np.eye(3, dtype=np.float64)
            else:
                axis = self._random_perpendicular_vector(source_unit)
                rotation = self._axis_angle_rotation_matrix(axis, np.pi)
        else:
            axis = cross / cross_norm
            angle = float(np.arccos(dot))
            rotation = self._axis_angle_rotation_matrix(axis, angle)
        if random_spin:
            spin = float(self.rng.uniform(0.0, 2.0 * np.pi))
            rotation = self._axis_angle_rotation_matrix(target_unit, spin) @ rotation
        return rotation

    def _fit_rotation_to_vectors(
        self: "Supercell",
        source_vectors: np.ndarray,
        target_vectors: np.ndarray,
    ) -> np.ndarray:
        """Return the best-fit rotation mapping `source_vectors` onto `target_vectors`."""
        source_vectors = np.asarray(source_vectors, dtype=np.float64)
        target_vectors = np.asarray(target_vectors, dtype=np.float64)
        if source_vectors.shape != target_vectors.shape:
            raise ValueError("source_vectors and target_vectors must have matching shapes.")
        if source_vectors.ndim != 2 or source_vectors.shape[1] != 3:
            raise ValueError("Expected vectors with shape (n, 3).")
        if source_vectors.shape[0] == 0:
            return np.eye(3, dtype=np.float64)
        if source_vectors.shape[0] == 1:
            return self._rotation_align_vector(
                source_vectors[0],
                target_vectors[0],
                random_spin=False,
            )
        covariance = source_vectors.T @ target_vectors
        u_mat, _, vt_mat = np.linalg.svd(covariance)
        row_rotation = vt_mat.T @ u_mat.T
        if np.linalg.det(row_rotation) < 0.0:
            vt_mat[-1, :] *= -1.0
            row_rotation = vt_mat.T @ u_mat.T
        return row_rotation.T

    def _position_overlap_score(
        self: "Supercell",
        position: np.ndarray,
        species_index: int,
        positions: np.ndarray,
        atom_species_index: np.ndarray,
        shell_target: CoordinationShellTarget,
        *,
        ignore_atom: int | None = None,
    ) -> tuple[float, float]:
        """Return total and maximum hard-min overlap for a candidate position."""
        positions = np.asarray(positions, dtype=np.float64)
        if positions.size == 0:
            return 0.0, 0.0
        deltas = self._minimum_image_vectors(np.asarray(position, dtype=np.float64), positions)
        distance = np.linalg.norm(deltas, axis=1)
        if ignore_atom is not None and 0 <= int(ignore_atom) < distance.size:
            distance[int(ignore_atom)] = np.inf
        hard_min = shell_target.pair_hard_min[int(species_index), np.asarray(atom_species_index, dtype=np.intp)]
        overlap = hard_min - distance
        keep = overlap > 0.0
        if not np.any(keep):
            return 0.0, 0.0
        return float(np.sum(overlap[keep] * overlap[keep])), float(np.max(overlap[keep]))

    def _build_reciprocal_projection_matches(
        self: "Supercell",
        positions: np.ndarray,
        rotations: np.ndarray,
        shell_target: CoordinationShellTarget,
        motif_ids: np.ndarray,
        *,
        match_candidates_per_slot: int,
        assign_cutoff_scale: float,
        reciprocal_weight: float,
    ) -> dict[str, Any]:
        """Match motif slots using reciprocal slot compatibility on both atoms."""
        num_atoms = positions.shape[0]
        atom_indices_by_species = [
            np.flatnonzero(self._atom_species_index == species_index).astype(np.intp, copy=False)
            for species_index in range(self._num_species)
        ]
        capacities = np.array(
            [len(shell_target.motif_neighbor_species[int(motif_id)]) for motif_id in motif_ids],
            dtype=np.intp,
        )
        candidate_edges: list[
            tuple[
                float,
                int,
                int,
                int,
                int,
                np.ndarray,
                np.ndarray,
                float,
            ]
        ] = []
        site_records: list[tuple[int, int, int, np.ndarray]] = []

        for atom_i in range(num_atoms):
            species_i = int(self._atom_species_index[atom_i])
            motif_i = int(motif_ids[atom_i])
            motif_vectors_i = shell_target.motif_neighbor_vectors[motif_i]
            motif_species_i = shell_target.motif_neighbor_species[motif_i]
            if motif_vectors_i.shape[0] == 0:
                continue
            predicted_vectors_i = self._apply_rotation(motif_vectors_i, rotations[atom_i])
            for slot_i, neighbor_species in enumerate(motif_species_i):
                neighbor_species = int(neighbor_species)
                predicted_position = np.asarray(
                    positions[atom_i] + predicted_vectors_i[int(slot_i)],
                    dtype=np.float64,
                )
                site_records.append((int(atom_i), int(slot_i), neighbor_species, predicted_position))
                target_atoms = atom_indices_by_species[neighbor_species]
                if target_atoms.size == 0:
                    continue
                deltas = self._minimum_image_vectors(predicted_position, positions[target_atoms])
                distance_sq = np.einsum("ij,ij->i", deltas, deltas)
                self_mask = target_atoms == int(atom_i)
                if np.any(self_mask):
                    distance_sq[self_mask] = np.inf
                pair_peak = float(shell_target.pair_peak[species_i, neighbor_species])
                pair_sigma = float(shell_target.pair_sigma[species_i, neighbor_species])
                cutoff = float(max(assign_cutoff_scale * pair_peak, 2.5 * pair_sigma, 0.35))
                keep = np.isfinite(distance_sq) & (distance_sq <= cutoff * cutoff)
                if not np.any(keep):
                    continue
                local_atoms = target_atoms[keep]
                local_distance_sq = distance_sq[keep]
                order = np.argsort(local_distance_sq, kind="stable")[: int(max(match_candidates_per_slot, 1))]
                for local_index in order:
                    atom_j = int(local_atoms[local_index])
                    motif_j = int(motif_ids[atom_j])
                    reciprocal_slots = np.flatnonzero(
                        shell_target.motif_neighbor_species[motif_j] == species_i
                    )
                    if reciprocal_slots.size == 0:
                        continue
                    reciprocal_vectors = self._apply_rotation(
                        shell_target.motif_neighbor_vectors[motif_j][reciprocal_slots],
                        rotations[atom_j],
                    )
                    reciprocal_positions = positions[atom_j] + reciprocal_vectors
                    back_deltas = self._minimum_image_vectors(
                        positions[atom_i],
                        reciprocal_positions,
                    )
                    back_distance_sq = np.einsum("ij,ij->i", back_deltas, back_deltas)
                    best_recip_index = int(np.argmin(back_distance_sq))
                    best_slot_j = int(reciprocal_slots[best_recip_index])
                    best_back_position = np.asarray(
                        reciprocal_positions[best_recip_index],
                        dtype=np.float64,
                    )
                    source_distance_sq = float(local_distance_sq[local_index])
                    score = float(
                        source_distance_sq + reciprocal_weight * float(back_distance_sq[best_recip_index])
                    )
                    effective_distance = float(
                        np.sqrt(max(source_distance_sq, 0.0))
                        + np.sqrt(max(float(back_distance_sq[best_recip_index]), 0.0))
                    )
                    candidate_edges.append(
                        (
                            score,
                            int(atom_i),
                            int(slot_i),
                            int(atom_j),
                            int(best_slot_j),
                            np.asarray(predicted_position, dtype=np.float64),
                            best_back_position,
                            effective_distance,
                        )
                    )

        candidate_edges.sort(key=lambda item: item[0])
        outgoing: list[list[tuple[int, int, float]]] = [[] for _ in range(num_atoms)]
        incoming: list[list[tuple[int, int, np.ndarray, float]]] = [[] for _ in range(num_atoms)]
        bonded_sets: list[set[int]] = [set() for _ in range(num_atoms)]
        used_slot_keys: set[tuple[int, int]] = set()
        used_pairs: set[tuple[int, int]] = set()
        accepted_edges: list[tuple[int, int, int, int]] = []
        matched_edges = 0
        matched_score = 0.0

        for score, atom_i, slot_i, atom_j, slot_j, predicted_position, back_position, effective_distance in candidate_edges:
            source_key = (int(atom_i), int(slot_i))
            target_key = (int(atom_j), int(slot_j))
            pair_key = (min(int(atom_i), int(atom_j)), max(int(atom_i), int(atom_j)))
            if source_key in used_slot_keys or target_key in used_slot_keys or pair_key in used_pairs:
                continue
            used_slot_keys.add(source_key)
            used_slot_keys.add(target_key)
            used_pairs.add(pair_key)
            outgoing[atom_i].append((int(atom_j), int(slot_i), float(effective_distance)))
            outgoing[atom_j].append((int(atom_i), int(slot_j), float(effective_distance)))
            incoming[atom_j].append(
                (
                    int(atom_i),
                    int(slot_i),
                    np.asarray(predicted_position, dtype=np.float64),
                    float(effective_distance),
                )
            )
            incoming[atom_i].append(
                (
                    int(atom_j),
                    int(slot_j),
                    np.asarray(back_position, dtype=np.float64),
                    float(effective_distance),
                )
            )
            bonded_sets[atom_i].add(atom_j)
            bonded_sets[atom_j].add(atom_i)
            accepted_edges.append((int(atom_i), int(slot_i), int(atom_j), int(slot_j)))
            matched_edges += 1
            matched_score += float(score)

        matched_slot_count = np.zeros(num_atoms, dtype=np.intp)
        for atom_index, _ in used_slot_keys:
            matched_slot_count[int(atom_index)] += 1
        remaining = capacities - matched_slot_count
        unmatched_positions: list[np.ndarray] = []
        unmatched_species: list[int] = []
        for atom_i, slot_i, neighbor_species, predicted_position in site_records:
            if (int(atom_i), int(slot_i)) in used_slot_keys:
                continue
            unmatched_positions.append(np.asarray(predicted_position, dtype=np.float64))
            unmatched_species.append(int(neighbor_species))

        return {
            "outgoing": outgoing,
            "incoming": incoming,
            "bonded_sets": bonded_sets,
            "capacities": capacities,
            "remaining": remaining,
            "matched_edges": int(matched_edges),
            "matched_slots": int(len(used_slot_keys)),
            "unmatched_slots": int(np.sum(capacities) - len(used_slot_keys)),
            "matched_score": float(matched_score),
            "edges": accepted_edges,
            "unmatched_site_positions": (
                np.asarray(unmatched_positions, dtype=np.float64)
                if unmatched_positions
                else np.empty((0, 3), dtype=np.float64)
            ),
            "unmatched_site_species": (
                np.asarray(unmatched_species, dtype=np.intp)
                if unmatched_species
                else np.empty(0, dtype=np.intp)
            ),
        }

    def _cluster_projection_sites(
        self: "Supercell",
        site_positions: np.ndarray,
        site_species: np.ndarray,
        *,
        cluster_radius: float,
    ) -> dict[int, list[dict[str, Any]]]:
        """Cluster unmatched motif-site positions by species and periodic proximity."""
        site_positions = np.asarray(site_positions, dtype=np.float64)
        site_species = np.asarray(site_species, dtype=np.intp)
        clusters_by_species: dict[int, list[dict[str, Any]]] = {}
        if site_positions.size == 0 or site_species.size == 0:
            return clusters_by_species

        cluster_radius = float(cluster_radius)
        for species_index in range(self._num_species):
            indices = np.flatnonzero(site_species == species_index)
            if indices.size == 0:
                continue
            clusters: list[dict[str, Any]] = []
            for site_index in indices:
                position = np.asarray(site_positions[int(site_index)], dtype=np.float64)
                best_cluster_index: int | None = None
                best_distance = np.inf
                for cluster_index, cluster in enumerate(clusters):
                    delta = self._minimum_image_vectors(
                        np.asarray(cluster["position"], dtype=np.float64),
                        position[None, :],
                    )[0]
                    distance = float(np.linalg.norm(delta))
                    if distance <= cluster_radius and distance < best_distance:
                        best_cluster_index = int(cluster_index)
                        best_distance = distance
                if best_cluster_index is None:
                    clusters.append(
                        {
                            "position": np.asarray(self._wrap_position(position), dtype=np.float64),
                            "count": 1,
                        }
                    )
                    continue
                cluster = clusters[best_cluster_index]
                delta = self._minimum_image_vectors(
                    np.asarray(cluster["position"], dtype=np.float64),
                    position[None, :],
                )[0]
                new_count = int(cluster["count"]) + 1
                cluster["position"] = np.asarray(
                    self._wrap_position(
                        np.asarray(cluster["position"], dtype=np.float64) + delta / new_count
                    ),
                    dtype=np.float64,
                )
                cluster["count"] = new_count
            clusters.sort(key=lambda item: (-int(item["count"])))
            clusters_by_species[int(species_index)] = clusters
        return clusters_by_species

    def _recruit_projection_atoms(
        self: "Supercell",
        positions: np.ndarray,
        rotations: np.ndarray,
        shell_target: CoordinationShellTarget,
        matches: dict[str, Any],
        *,
        cluster_radius_scale: float,
        max_recruits_per_step: int,
        min_cluster_size: int,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Move underused atoms onto dense clusters of unmatched projected sites."""
        positions = np.asarray(positions, dtype=np.float64)
        rotations = np.asarray(rotations, dtype=np.float64)
        unmatched_positions = np.asarray(matches["unmatched_site_positions"], dtype=np.float64)
        unmatched_species = np.asarray(matches["unmatched_site_species"], dtype=np.intp)
        if unmatched_positions.size == 0 or max_recruits_per_step <= 0:
            return positions, rotations, 0

        cluster_radius = float(
            max(
                cluster_radius_scale * np.max(shell_target.pair_peak[shell_target.pair_mask])
                if np.any(shell_target.pair_mask)
                else 0.0,
                0.35,
            )
        )
        clusters_by_species = self._cluster_projection_sites(
            unmatched_positions,
            unmatched_species,
            cluster_radius=cluster_radius,
        )
        degrees = np.array([len(outgoing) for outgoing in matches["outgoing"]], dtype=np.intp)
        capacities = np.asarray(matches["capacities"], dtype=np.intp)
        positions_new = np.array(positions, copy=True)
        rotations_new = np.array(rotations, copy=True)
        recruited = 0

        for species_index in range(self._num_species):
            clusters = [
                cluster
                for cluster in clusters_by_species.get(int(species_index), [])
                if int(cluster["count"]) >= int(min_cluster_size)
            ]
            if not clusters:
                continue
            atom_indices = np.flatnonzero(self._atom_species_index == species_index).astype(np.intp, copy=False)
            if atom_indices.size == 0:
                continue
            deficit = capacities[atom_indices] - degrees[atom_indices]
            movable_mask = (deficit >= 2) | (degrees[atom_indices] <= 1)
            movable = atom_indices[movable_mask]
            if movable.size == 0:
                continue
            order = np.argsort(
                np.stack(
                    [
                        -deficit[movable_mask],
                        degrees[movable_mask],
                    ],
                    axis=1,
                ),
                axis=0,
            )
            movable = movable[np.lexsort((degrees[movable_mask], -deficit[movable_mask]))]
            used_atoms: set[int] = set()
            for cluster in clusters:
                if recruited >= int(max_recruits_per_step):
                    break
                target_position = np.asarray(cluster["position"], dtype=np.float64)
                chosen_atom: int | None = None
                chosen_overlap: float | None = None
                for atom_index in movable:
                    atom_index = int(atom_index)
                    if atom_index in used_atoms:
                        continue
                    _, max_overlap = self._position_overlap_score(
                        target_position,
                        int(species_index),
                        positions_new,
                        self._atom_species_index,
                        shell_target,
                        ignore_atom=atom_index,
                    )
                    if chosen_overlap is None or max_overlap < chosen_overlap:
                        chosen_atom = atom_index
                        chosen_overlap = float(max_overlap)
                if chosen_atom is None:
                    continue
                if chosen_overlap is not None:
                    pair_peak = float(np.max(shell_target.pair_peak[int(species_index)]))
                    if chosen_overlap > 0.45 * max(pair_peak, 1.0):
                        continue
                positions_new[chosen_atom] = self._wrap_position(target_position)
                rotations_new[chosen_atom] = self._random_rotation_matrix()
                used_atoms.add(int(chosen_atom))
                recruited += 1
        return positions_new, rotations_new, int(recruited)

    def _origin_sparse_contribution(
        self: "Supercell",
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
        self: "Supercell",
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
        self: "Supercell",
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
        self: "Supercell",
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
        self: "Supercell",
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
        self: "Supercell",
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
        self: "Supercell",
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
        self: "Supercell",
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

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def measure_g3(
        self: "Supercell",
        *,
        force: bool = False,
        show_progress: bool = True,
        backend: str = "auto",
        sample_fraction: float = 1.0,
        sample_rng_seed: int | None = None,
    ) -> G3Distribution:
        """Measure the current random supercell on the target distribution grid.

        Parameters
        ----------
        force
            If `True`, discard any cached measurement and recompute.
        show_progress
            If `True`, display a text progress bar while the supercell
            histogram is accumulated.
        backend
            Kernel selector forwarded to
            :meth:`G3Distribution.measure_g3`.  ``"auto"`` (default)
            picks the numba-parallel path when numba is installed.
        sample_fraction, sample_rng_seed
            Monte-Carlo origin subsampling.  ``sample_fraction < 1.0``
            iterates only a uniform random subset of origin atoms
            (preserves PBC tile, full neighbour set).  At 200³ Å set
            this to ~``(40/200)**3`` so the histogram cost matches a
            40³ Å reference cell.
        """
        if (
            self.current_distribution is not None
            and not force
            and sample_fraction == 1.0
        ):
            return self.current_distribution

        measured = G3Distribution(self.atoms, label=f"{self.label}-measured")
        measured.measure_g3(
            r_max=self.measure_r_max,
            r_step=self.measure_r_step,
            phi_num_bins=self.measure_phi_num_bins,
            show_progress=show_progress,
            progress_label=f"Measuring g3 in {self.label}",
            backend=backend,
            sample_fraction=sample_fraction,
            sample_rng_seed=sample_rng_seed,
        )
        self.current_distribution = measured
        return measured

    def sync_g3(self: "Supercell", *, show_progress: bool = True) -> G3Distribution:
        """Recompute the supercell's g2 + g3 from scratch and rebuild Monte-Carlo caches.

        Forces a fresh :meth:`measure_g3` (bypassing any cached
        result) and then re-initialises the per-atom contribution
        tables that :meth:`monte_carlo` uses to compute incremental
        ΔG3 / Δcost on each trial move.  Call after externally
        modifying ``self.atoms.positions`` (e.g. injecting
        a :func:`numpy.random.normal` thermal jitter) or after
        chaining a :meth:`shell_relax` between MC rounds — otherwise
        the MC caches refer to stale neighbour pairings and the
        proposed-move statistics drift.

        Parameters
        ----------
        show_progress : bool, optional
            Display a tqdm-style progress bar while the underlying
            :meth:`measure_g3` iterates.  Default ``True``.

        Returns
        -------
        G3Distribution
            The freshly measured distribution, also stored as
            ``self.current_distribution``.
        """
        measured = self.measure_g3(force=True, show_progress=show_progress)
        self._initialize_mc_state()
        return measured

