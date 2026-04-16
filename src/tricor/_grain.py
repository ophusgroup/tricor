"""Voronoi grain construction mixin for Supercell."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from ase.atoms import Atoms
from ase.neighborlist import neighbor_list

from .g3 import _EPS

if TYPE_CHECKING:
    from .shells import CoordinationShellTarget
    from .supercell import Supercell


class _GrainMixin:
    def _build_grain_atoms(
        self: "Supercell",
        shell_target: "CoordinationShellTarget",
        grain_size: float,
        crystalline_fraction: float = 1.0,
        displacement_sigma: float = 0.0,
        max_density_passes: int = 5,
    ) -> Atoms:
        """Build a supercell with crystalline grains via Voronoi construction.

        Steps:
            1. Place Voronoi seeds, mark fraction as crystalline.
            2. Tile crystal once into a spherical block (large enough
               for any grain).  For each crystalline cell: copy, shift
               origin, rotate, translate, Voronoi-crop.
            3. Remove overlapping atoms at grain boundaries.
            4. Fill amorphous cells with random positions.

        Parameters
        ----------
        shell_target
            First-shell coordination targets from the reference crystal.
        grain_size
            Diameter of crystalline grains in Angstrom.
        crystalline_fraction
            Fraction of Voronoi cells filled with crystal (0.0 to 1.0).
        displacement_sigma
            Gaussian displacement (Angstrom) applied to grain atoms.
        """
        cell = self._build_supercell_cell()
        cell_mat = np.asarray(cell, dtype=np.float64)
        cell_inv = np.linalg.inv(cell_mat)
        box_volume = float(abs(np.linalg.det(cell_mat)))
        species, counts = self._target_species_counts(box_volume)
        total_target = int(np.sum(counts))

        crystalline_fraction = float(np.clip(crystalline_fraction, 0.0, 1.0))

        # ---- Step 1: Voronoi seeds ----
        grain_radius = max(float(grain_size) * 0.5, 2.0)
        grain_volume = (4.0 / 3.0) * np.pi * grain_radius ** 3
        n_seeds = max(1, int(np.ceil(box_volume / grain_volume)))

        n_crystalline = max(0, min(n_seeds, int(np.round(n_seeds * crystalline_fraction))))
        is_crystalline_cell = np.zeros(n_seeds, dtype=bool)
        if n_crystalline > 0:
            chosen = self.rng.choice(n_seeds, size=n_crystalline, replace=False)
            is_crystalline_cell[chosen] = True

        seed_frac = self.rng.random((n_seeds, 3))
        seed_cart = seed_frac @ cell_mat

        rotations = np.empty((n_seeds, 3, 3), dtype=np.float64)
        for ig in range(n_seeds):
            if is_crystalline_cell[ig]:
                M = self.rng.standard_normal((3, 3))
                Q, R = np.linalg.qr(M)
                Q *= np.sign(np.linalg.det(Q))
                rotations[ig] = Q
            else:
                rotations[ig] = np.eye(3)

        # ---- Step 2: tile crystal ONCE into a spherical block ----
        ref_cell = np.asarray(self.reference_atoms.cell.array, dtype=np.float64)
        ref_pos = self.reference_atoms.positions.copy()
        ref_numbers = self.reference_atoms.numbers.copy()

        # Sphere radius: grain radius + 0.5 unit cells for origin shifts
        max_ref_length = float(np.max(np.linalg.norm(ref_cell, axis=1)))
        sphere_radius = grain_radius + max_ref_length

        # Tile enough to cover the sphere
        ref_lengths = np.linalg.norm(ref_cell, axis=1)
        ref_lengths = np.maximum(ref_lengths, _EPS)
        n_rep = int(np.ceil(sphere_radius / np.min(ref_lengths))) + 1

        shifts = []
        for ix in range(-n_rep, n_rep + 1):
            for iy in range(-n_rep, n_rep + 1):
                for iz in range(-n_rep, n_rep + 1):
                    shifts.append([ix, iy, iz])
        shifts = np.array(shifts, dtype=np.float64)
        shift_cart = shifts @ ref_cell

        tile_pos = (ref_pos[None, :, :] + shift_cart[:, None, :]).reshape(-1, 3)
        tile_nums = np.tile(ref_numbers, len(shifts))

        # Center and crop to sphere
        tile_center = np.mean(tile_pos, axis=0)
        tile_pos -= tile_center
        dist_from_origin = np.linalg.norm(tile_pos, axis=1)
        sphere_mask = dist_from_origin < sphere_radius
        tile_pos = tile_pos[sphere_mask]
        tile_nums = tile_nums[sphere_mask]

        # ---- For each grain: copy, shift, rotate, place, crop ----

        # Chunked PBC Voronoi assignment
        max_chunk = max(1, 25_000_000 // max(n_seeds, 1))

        def _voronoi(positions: np.ndarray) -> np.ndarray:
            n = len(positions)
            nearest = np.empty(n, dtype=np.intp)
            for start in range(0, n, max_chunk):
                end = min(start + max_chunk, n)
                delta = positions[start:end, None, :] - seed_cart[None, :, :]
                frac_d = delta @ cell_inv
                frac_d -= np.rint(frac_d)
                cart_d = frac_d @ cell_mat
                nearest[start:end] = np.argmin(
                    np.sum(cart_d ** 2, axis=2), axis=1,
                )
            return nearest

        all_pos: list[np.ndarray] = []
        all_nums: list[np.ndarray] = []
        all_gids: list[np.ndarray] = []

        for ig in range(n_seeds):
            if not is_crystalline_cell[ig]:
                continue

            # Copy the spherical block
            gpos = tile_pos.copy()
            gnums = tile_nums.copy()

            # Random fractional origin shift for sublattice symmetry
            frac_shift = (self.rng.random(3) - 0.5) @ ref_cell
            gpos += frac_shift

            # Rotate around origin
            gpos = gpos @ rotations[ig].T

            # Translate to grain seed
            gpos += seed_cart[ig]

            # Wrap into periodic box
            frac = gpos @ cell_inv
            frac %= 1.0
            gpos = frac @ cell_mat

            all_pos.append(gpos)
            all_nums.append(gnums)
            all_gids.append(np.full(len(gpos), ig, dtype=np.intp))

        # Voronoi crop: keep atoms only in their own grain's cell
        if all_pos:
            raw_pos = np.concatenate(all_pos, axis=0)
            raw_nums = np.concatenate(all_nums, axis=0)
            raw_gids = np.concatenate(all_gids, axis=0)

            nearest = _voronoi(raw_pos)
            keep = nearest == raw_gids
            positions = raw_pos[keep]
            numbers = raw_nums[keep]
            grain_ids = raw_gids[keep]
        else:
            positions = np.empty((0, 3), dtype=np.float64)
            numbers = np.empty(0, dtype=np.intp)
            grain_ids = np.empty(0, dtype=np.intp)

        # Apply thermal displacement to grain atoms
        if displacement_sigma > _EPS and len(positions) > 0:
            positions += self.rng.normal(0.0, displacement_sigma, size=positions.shape)
            frac = positions @ cell_inv
            frac %= 1.0
            positions = frac @ cell_mat

        # ---- Step 3: remove overlapping atoms ----
        pair_inner = np.asarray(shell_target.pair_inner, dtype=np.float64)
        pair_hard_min = np.asarray(shell_target.pair_hard_min, dtype=np.float64)
        overlap_thresh = float(np.max(np.maximum(pair_inner, pair_hard_min)))

        for _pass in range(10):
            if len(positions) == 0:
                break
            temp = Atoms(numbers=numbers, positions=positions,
                         cell=cell, pbc=self.reference_atoms.pbc)
            ov_i, ov_j, ov_d = neighbor_list("ijd", temp, overlap_thresh)
            if len(ov_i) == 0:
                break
            remove = set()
            for k in range(len(ov_i)):
                ai, aj = int(ov_i[k]), int(ov_j[k])
                if ai not in remove and aj not in remove:
                    remove.add(max(ai, aj))
            if not remove:
                break
            keep = np.ones(len(numbers), dtype=bool)
            keep[list(remove)] = False
            positions = positions[keep]
            numbers = numbers[keep]
            grain_ids = grain_ids[keep]

        # ---- Step 4: fill to target density ----
        # Place atoms in low-density regions (grain boundaries).
        # Strategy: find voids by generating candidates near existing
        # atoms with random offsets, then keep those that are far
        # enough from all existing atoms.
        species_frac = counts.astype(float) / max(total_target, 1)
        pair_peak_val = float(np.max(
            np.asarray(shell_target.pair_peak, dtype=np.float64),
        ))
        # Fill overlap threshold: starts at 0.85 * NN distance and
        # decreases each pass if density can't be reached.
        fill_thresh_start = pair_peak_val * 0.85
        fill_thresh_min = pair_peak_val * 0.55

        for _density_pass in range(max_density_passes):
            n_fill = max(0, total_target - len(positions))
            if n_fill == 0:
                break

            # Generate candidates: random offsets from existing atoms
            # at roughly the NN distance.  This places candidates in
            # the void regions near grain boundaries.
            n_candidates = n_fill * 10
            source_idx = self.rng.integers(0, len(positions), size=n_candidates)
            offsets = self.rng.normal(0.0, pair_peak_val * 0.6, size=(n_candidates, 3))
            cand_pos = positions[source_idx] + offsets

            # Wrap into box
            frac_cand = cand_pos @ cell_inv
            frac_cand %= 1.0
            cand_pos = frac_cand @ cell_mat

            # Check distances to all existing atoms
            combined = np.concatenate([positions, cand_pos], axis=0)
            combined_nums = np.concatenate([
                numbers,
                np.zeros(len(cand_pos), dtype=numbers.dtype),
            ])
            # Decrease threshold each pass to fill tighter voids
            t = _density_pass / max(max_density_passes - 1, 1)
            fill_thresh = fill_thresh_start + t * (fill_thresh_min - fill_thresh_start)

            temp = Atoms(numbers=combined_nums, positions=combined,
                         cell=cell, pbc=self.reference_atoms.pbc)
            ov_i, ov_j, ov_d = neighbor_list("ijd", temp, fill_thresh)

            # Reject candidates that overlap with existing atoms
            n_existing = len(positions)
            bad = set()
            for k in range(len(ov_i)):
                ai, aj = int(ov_i[k]), int(ov_j[k])
                if ai >= n_existing:
                    bad.add(ai - n_existing)
                if aj >= n_existing:
                    bad.add(aj - n_existing)

            good_mask = np.ones(len(cand_pos), dtype=bool)
            if bad:
                good_mask[list(bad)] = False
            good_pos = cand_pos[good_mask][:n_fill]

            if len(good_pos) == 0:
                break

            # Assign species proportionally
            fill_nums = np.repeat(
                species,
                np.round(species_frac * len(good_pos)).astype(int),
            )
            if len(fill_nums) < len(good_pos):
                fill_nums = np.concatenate([
                    fill_nums,
                    np.repeat(species[0:1], len(good_pos) - len(fill_nums)),
                ])
            fill_nums = fill_nums[:len(good_pos)]
            self.rng.shuffle(fill_nums)

            positions = np.concatenate([positions, good_pos], axis=0)
            numbers = np.concatenate([numbers, fill_nums], axis=0)
            grain_ids = np.concatenate([
                grain_ids,
                np.full(len(good_pos), -1, dtype=np.intp),
            ], axis=0)

        if len(positions) == 0:
            return self._build_random_atoms()

        atoms = Atoms(
            numbers=numbers,
            positions=positions,
            cell=cell,
            pbc=self.reference_atoms.pbc,
        )
        atoms.info["relative_density"] = self.relative_density
        atoms.info["cell_dim_angstroms"] = self.cell_dim_angstroms
        atoms.info["n_grains"] = n_crystalline
        atoms.info["grain_size"] = float(grain_size)
        atoms.info["crystalline_fraction"] = crystalline_fraction

        self._grain_ids = grain_ids
        self._grain_seeds = seed_cart.copy()
        return atoms
