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
    ) -> Atoms:
        """Build a supercell with crystalline grains via Voronoi construction.

        The box is Voronoi-tessellated into cells of diameter
        *grain_size*.  A fraction *crystalline_fraction* of cells are
        filled with randomly rotated copies of the reference crystal;
        the remaining cells are filled with random atom positions
        (amorphous, tagged with ``grain_id = -1``).

        Parameters
        ----------
        shell_target
            First-shell coordination targets from the reference crystal.
        grain_size
            Diameter of crystalline grains in Angstrom.
        crystalline_fraction
            Fraction of Voronoi cells filled with crystal (0.0 to 1.0).
            The rest are filled with random (amorphous) positions.
        displacement_sigma
            Gaussian displacement sigma (Angstrom) applied to atoms
            within crystalline grains.  Defaults to 0.
        """
        cell = self._build_supercell_cell()
        cell_mat = np.asarray(cell, dtype=np.float64)
        cell_inv = np.linalg.inv(cell_mat)
        box_volume = float(abs(np.linalg.det(cell_mat)))
        species, counts = self._target_species_counts(box_volume)
        total_target = int(np.sum(counts))

        crystalline_fraction = float(np.clip(crystalline_fraction, 0.0, 1.0))

        # --- grain parameters ---
        grain_radius = max(float(grain_size) * 0.5, 2.0)
        grain_volume = (4.0 / 3.0) * np.pi * grain_radius ** 3
        n_seeds = max(1, int(np.ceil(box_volume / grain_volume)))

        # Decide which Voronoi cells are crystalline vs amorphous
        n_crystalline = max(0, min(n_seeds, int(np.round(n_seeds * crystalline_fraction))))
        is_crystalline_cell = np.zeros(n_seeds, dtype=bool)
        if n_crystalline > 0:
            chosen = self.rng.choice(n_seeds, size=n_crystalline, replace=False)
            is_crystalline_cell[chosen] = True

        # --- place Voronoi seeds randomly ---
        seed_frac = self.rng.random((n_seeds, 3))
        seed_cart = seed_frac @ cell_mat  # (n_seeds, 3)

        # --- random rotation for each crystalline grain ---
        rotations = np.empty((n_seeds, 3, 3), dtype=np.float64)
        for ig in range(n_seeds):
            if is_crystalline_cell[ig]:
                M = self.rng.standard_normal((3, 3))
                Q, R = np.linalg.qr(M)
                Q *= np.sign(np.linalg.det(Q))
                rotations[ig] = Q
            else:
                rotations[ig] = np.eye(3)  # unused

        # --- build a crystal tile block centered at the origin ---
        # Tile the reference cell in +/- directions to cover a sphere
        # of radius grain_radius + buffer.  The number of reps is
        # based on the SHORTEST lattice combination (not just the
        # individual vectors) to handle non-orthogonal cells.
        ref_cell = np.asarray(self.reference_atoms.cell.array, dtype=np.float64)
        ref_pos = self.reference_atoms.positions.copy()
        ref_numbers = self.reference_atoms.numbers.copy()
        u, v, w = ref_cell[0], ref_cell[1], ref_cell[2]

        # Find the shortest lattice vector combination
        combos = [u, v, w, u+v, u-v, u+w, u-w, v+w, v-w,
                  u+v+w, u+v-w, u-v+w, u-v-w]
        min_length = min(np.linalg.norm(c) for c in combos)
        min_length = max(min_length, _EPS)

        tile_radius = grain_radius + min_length * 2
        n_rep = int(np.ceil(tile_radius / min_length)) + 1

        shifts = []
        for ix in range(-n_rep, n_rep + 1):
            for iy in range(-n_rep, n_rep + 1):
                for iz in range(-n_rep, n_rep + 1):
                    shifts.append([ix, iy, iz])
        shifts = np.array(shifts, dtype=np.float64)
        shift_cart = shifts @ ref_cell

        base_pos = (ref_pos[None, :, :] + shift_cart[:, None, :]).reshape(-1, 3)
        base_numbers = np.tile(ref_numbers, len(shifts))

        # --- chunked Voronoi assignment ---
        max_chunk_elements = 25_000_000
        chunk_size = max(1, max_chunk_elements // max(n_seeds, 1))

        def _chunked_voronoi(positions: np.ndarray) -> np.ndarray:
            """Return nearest-seed index for each position, PBC-aware."""
            n = len(positions)
            nearest = np.empty(n, dtype=np.intp)
            for start in range(0, n, chunk_size):
                end = min(start + chunk_size, n)
                delta = positions[start:end, None, :] - seed_cart[None, :, :]
                frac_d = delta @ cell_inv
                frac_d -= np.rint(frac_d)
                cart_d = frac_d @ cell_mat
                nearest[start:end] = np.argmin(
                    np.sum(cart_d ** 2, axis=2), axis=1,
                )
            return nearest

        # --- per-grain: rotate base block, place at seed, collect ---
        # Each grain gets its own properly rotated crystal tile.
        # This ensures both sublattices are tiled identically.
        all_raw_pos: list[np.ndarray] = []
        all_raw_numbers: list[np.ndarray] = []
        all_raw_gids: list[np.ndarray] = []

        for ig in range(n_seeds):
            if not is_crystalline_cell[ig]:
                continue
            # Random fractional shift of the crystal origin along
            # each lattice vector.  This randomises which sublattice
            # site lands nearest the grain seed so all species are
            # equally represented across grains.
            rand_frac = self.rng.random(3)
            origin_shift = (rand_frac[0] * 0.5 * u
                          + rand_frac[1] * 0.5 * v
                          + rand_frac[2] * 0.5 * w)
            shifted = base_pos + origin_shift
            # Rotate around origin, then translate to grain seed
            rotated = shifted @ rotations[ig].T
            grain_pos = rotated + seed_cart[ig]
            frac = grain_pos @ cell_inv
            frac %= 1.0
            grain_pos = frac @ cell_mat
            all_raw_pos.append(grain_pos)
            all_raw_numbers.append(base_numbers.copy())
            all_raw_gids.append(np.full(len(grain_pos), ig, dtype=np.intp))

        if not all_raw_pos:
            all_positions: list[np.ndarray] = []
            all_numbers: list[np.ndarray] = []
            all_grain_ids: list[np.ndarray] = []
        else:
            raw_pos = np.concatenate(all_raw_pos, axis=0)
            raw_numbers = np.concatenate(all_raw_numbers, axis=0)
            raw_gids = np.concatenate(all_raw_gids, axis=0)

            # Voronoi assignment: keep atoms only in their own
            # grain's cell (nearest seed must match the grain).
            nearest = _chunked_voronoi(raw_pos)
            keep = nearest == raw_gids
            all_positions = [raw_pos[keep]]
            all_numbers = [raw_numbers[keep]]
            all_grain_ids = [raw_gids[keep]]

        # --- fill amorphous Voronoi cells with random positions ---
        if n_crystalline < n_seeds:
            amorphous_volume_fraction = (n_seeds - n_crystalline) / max(n_seeds, 1)
            n_amorphous_target = int(np.round(total_target * amorphous_volume_fraction))

            if n_amorphous_target > 0:
                oversample = max(2, int(np.ceil(n_seeds / max(n_seeds - n_crystalline, 1))))
                n_candidates = n_amorphous_target * oversample
                cand_frac = self.rng.random((n_candidates, 3))
                cand_pos = cand_frac @ cell_mat

                nearest_c = _chunked_voronoi(cand_pos)
                in_amorphous = ~is_crystalline_cell[nearest_c]
                amorphous_pos = cand_pos[in_amorphous]

                if len(amorphous_pos) > n_amorphous_target:
                    amorphous_pos = amorphous_pos[:n_amorphous_target]

                # Assign species proportionally
                species_frac = counts.astype(float) / max(total_target, 1)
                amorphous_numbers = np.repeat(
                    species,
                    np.round(species_frac * len(amorphous_pos)).astype(int),
                )
                if len(amorphous_numbers) < len(amorphous_pos):
                    extra = np.repeat(species[0:1], len(amorphous_pos) - len(amorphous_numbers))
                    amorphous_numbers = np.concatenate([amorphous_numbers, extra])
                amorphous_numbers = amorphous_numbers[:len(amorphous_pos)]
                self.rng.shuffle(amorphous_numbers)

                all_positions.append(amorphous_pos)
                all_numbers.append(amorphous_numbers)
                all_grain_ids.append(np.full(len(amorphous_pos), -1, dtype=np.intp))

        if len(all_positions) == 0 or sum(len(p) for p in all_positions) == 0:
            return self._build_random_atoms()

        positions = np.concatenate(all_positions, axis=0)
        numbers = np.concatenate(all_numbers, axis=0)
        grain_ids = np.concatenate(all_grain_ids, axis=0)

        # --- remove overlapping atoms ---
        # Iteratively remove atoms closer than pair_inner until no
        # overlaps remain.  Each pass removes the higher-indexed atom
        # of each overlapping pair.
        pair_inner_arr = np.asarray(shell_target.pair_inner, dtype=np.float64)
        pair_hard_min = np.asarray(shell_target.pair_hard_min, dtype=np.float64)
        overlap_thresh = float(np.max(np.maximum(pair_inner_arr, pair_hard_min)))

        for _pass in range(10):  # safety limit
            temp_atoms = Atoms(
                numbers=numbers,
                positions=positions,
                cell=cell,
                pbc=self.reference_atoms.pbc,
            )
            ov_i, ov_j, ov_d = neighbor_list("ijd", temp_atoms, overlap_thresh)
            if len(ov_i) == 0:
                break
            # Mark one atom from each pair for removal
            remove = set()
            for k in range(len(ov_i)):
                ai, aj = int(ov_i[k]), int(ov_j[k])
                if ai not in remove and aj not in remove:
                    remove.add(max(ai, aj))
            if not remove:
                break
            keep_mask = np.ones(len(numbers), dtype=bool)
            keep_mask[list(remove)] = False
            positions = positions[keep_mask]
            numbers = numbers[keep_mask]
            grain_ids = grain_ids[keep_mask]

        # --- apply thermal displacements ---
        if displacement_sigma > _EPS:
            displacements = self.rng.normal(
                0.0, displacement_sigma, size=positions.shape,
            )
            positions += displacements
            # Re-wrap
            frac = positions @ cell_inv
            frac %= 1.0
            positions = frac @ cell_mat

        # --- adjust to target stoichiometry ---
        # Count current species
        unique_nums, current_counts = np.unique(numbers, return_counts=True)
        species_map = {int(s): int(c) for s, c in zip(species, counts)}

        # For each species, randomly keep/add to match target count
        final_positions: list[np.ndarray] = []
        final_numbers: list[int] = []
        final_grain_ids: list[int] = []

        for sp_num in species:
            sp_num = int(sp_num)
            target_count = species_map.get(sp_num, 0)
            sp_mask = numbers == sp_num
            sp_pos = positions[sp_mask]
            sp_gids = grain_ids[sp_mask]
            current_count = len(sp_pos)

            if current_count > target_count:
                # Remove excess atoms, preferring to remove grain atoms
                # over fill atoms to preserve crystalline_fraction
                excess = current_count - target_count
                grain_idx = np.where(sp_gids >= 0)[0]
                fill_idx = np.where(sp_gids < 0)[0]
                remove_idx: list[int] = []
                # Remove from grain atoms first if we have too many
                if len(grain_idx) > 0:
                    n_remove_grain = min(excess, len(grain_idx))
                    remove_idx.extend(
                        self.rng.choice(grain_idx, size=n_remove_grain, replace=False).tolist()
                    )
                if len(remove_idx) < excess and len(fill_idx) > 0:
                    n_more = min(excess - len(remove_idx), len(fill_idx))
                    remove_idx.extend(
                        self.rng.choice(fill_idx, size=n_more, replace=False).tolist()
                    )
                keep_mask_sp = np.ones(current_count, dtype=bool)
                keep_mask_sp[remove_idx] = False
                sp_pos = sp_pos[keep_mask_sp]
                sp_gids = sp_gids[keep_mask_sp]
            elif current_count < target_count:
                # Add random atoms to fill deficit
                deficit = target_count - current_count
                extra_frac = self.rng.random((deficit, 3))
                extra_pos = extra_frac @ cell_mat
                extra_gids = np.full(deficit, -1, dtype=np.intp)  # -1 = boundary fill
                sp_pos = np.concatenate([sp_pos, extra_pos], axis=0)
                sp_gids = np.concatenate([sp_gids, extra_gids], axis=0)

            for p, g in zip(sp_pos, sp_gids):
                final_positions.append(p)
                final_numbers.append(sp_num)
                final_grain_ids.append(int(g))

        final_positions_arr = np.array(final_positions, dtype=np.float64)
        final_numbers_arr = np.array(final_numbers, dtype=np.intp)
        final_grain_ids_arr = np.array(final_grain_ids, dtype=np.intp)

        # Shuffle to mix species (consistent with _build_random_atoms)
        shuffle_idx = self.rng.permutation(len(final_numbers_arr))
        final_positions_arr = final_positions_arr[shuffle_idx]
        final_numbers_arr = final_numbers_arr[shuffle_idx]
        final_grain_ids_arr = final_grain_ids_arr[shuffle_idx]

        # Final overlap removal pass (stoichiometry adjustment may
        # have added random atoms that overlap with existing ones)
        for _pass in range(10):
            temp = Atoms(
                numbers=final_numbers_arr,
                positions=final_positions_arr,
                cell=cell,
                pbc=self.reference_atoms.pbc,
            )
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
            keep = np.ones(len(final_numbers_arr), dtype=bool)
            keep[list(remove)] = False
            final_positions_arr = final_positions_arr[keep]
            final_numbers_arr = final_numbers_arr[keep]
            final_grain_ids_arr = final_grain_ids_arr[keep]

        atoms = Atoms(
            numbers=final_numbers_arr,
            positions=final_positions_arr,
            cell=cell,
            pbc=self.reference_atoms.pbc,
        )
        atoms.info["relative_density"] = self.relative_density
        atoms.info["cell_dim_angstroms"] = self.cell_dim_angstroms
        atoms.info["n_grains"] = n_crystalline
        atoms.info["grain_size"] = float(grain_size)
        atoms.info["crystalline_fraction"] = crystalline_fraction

        self._grain_ids = final_grain_ids_arr
        self._grain_seeds = seed_cart.copy()
        return atoms
