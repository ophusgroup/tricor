"""Random supercell initialization and local-update Monte Carlo scaffolding."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Callable

import numpy as np
from ase.atoms import Atoms
from ase.neighborlist import neighbor_list

from .g3 import G3Distribution, _EPS, _TextProgressBar
from .shells import CoordinationShellTarget


class Supercell:
    """Random supercell scaffold driven by a target :class:`G3Distribution`."""

    def __init__(
        self,
        distribution: G3Distribution,
        cell_dim_angstroms: float | Sequence[float],
        *,
        relative_density: float = 1.0,
        measure_g3: bool = False,
        plot_g3_compare: bool = False,
        label: str | None = None,
        rng_seed: int | None = None,
        g3_weight_r_scale: float | None = None,
        g3_weight_exponent: float = 2.0,
        g3_weight_floor: float = 0.1,
        spatial_bin_size: float | None = None,
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
        g3_weight_r_scale
            Characteristic radius in Angstrom that controls how strongly the
            Monte Carlo cost prioritizes short-range `g3` bins. Smaller values
            emphasize low-`r` structure more aggressively.
        g3_weight_exponent
            Power-law exponent used in the radial weighting curve for the
            full-`g3` cost.
        g3_weight_floor
            Minimum relative weight assigned to the farthest radial bins in the
            full-`g3` cost. Set this closer to `1.0` to flatten the weighting.
        spatial_bin_size
            Approximate spatial-hash bin size in Angstrom used for fast local
            proposal and overlap queries during Monte Carlo.
        **kwargs
            Extra metadata stored for future Monte Carlo options.
        """
        if "cell_dim" in kwargs:
            raise TypeError("Supercell.__init__() got an unexpected keyword argument 'cell_dim'")
        if relative_density <= 0:
            raise ValueError("relative_density must be positive.")
        if distribution.atoms is None:
            raise ValueError("Supercell construction requires a distribution with source atoms.")

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
        self.g3_weight_r_scale = (
            max(0.35 * self.measure_r_max, self.measure_r_step)
            if g3_weight_r_scale is None
            else float(g3_weight_r_scale)
        )
        self.g3_weight_exponent = float(g3_weight_exponent)
        self.g3_weight_floor = float(g3_weight_floor)
        self.spatial_bin_size = (
            max(self.measure_r_max, self.measure_r_step)
            if spatial_bin_size is None
            else float(spatial_bin_size)
        )
        if self.g3_weight_r_scale <= 0:
            raise ValueError("g3_weight_r_scale must be positive.")
        if self.g3_weight_exponent < 0:
            raise ValueError("g3_weight_exponent must be non-negative.")
        if not (0.0 < self.g3_weight_floor <= 1.0):
            raise ValueError("g3_weight_floor must lie in (0, 1].")
        if self.spatial_bin_size <= 0:
            raise ValueError("spatial_bin_size must be positive.")

        self.reference_atoms = self.target_distribution.atoms.copy()
        self._raw_distribution: G3Distribution = self.target_distribution
        self.atoms = self._build_random_atoms()
        self.current_distribution: G3Distribution | None = None
        self.mc_history: dict[str, np.ndarray] | None = None
        self.shell_history: dict[str, np.ndarray] | None = None
        self.shell_relax_history: dict[str, np.ndarray] | None = None
        self._grain_ids: np.ndarray | None = None
        self._grain_seeds: np.ndarray | None = None
        self.motif_history: dict[str, np.ndarray] | None = None
        self.motif_graph: dict[str, Any] | None = None
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
        self._g3_rr_weights_flat = self._build_g3_rr_weights()
        self._spatial_offset_cache: dict[tuple[int, int, int], np.ndarray] = {}
        self._rebuild_spatial_index()

        if measure_g3 or plot_g3_compare:
            self.measure_g3(show_progress=True)
            self._initialize_mc_state()

        if plot_g3_compare:
            self._display_compare_widget()

    @classmethod
    def from_atoms(
        cls,
        atoms: Atoms,
        cell_dim_angstroms: float | Sequence[float],
        *,
        r_max: float = 10.0,
        r_step: float = 0.2,
        phi_num_bins: int = 90,
        relative_density: float = 1.0,
        rng_seed: int | None = None,
        label: str | None = None,
        **kwargs: Any,
    ) -> "Supercell":
        """Create a random supercell directly from a reference crystal.

        This is a convenience constructor that measures the reference g3
        distribution internally, avoiding the need to create a
        :class:`G3Distribution` manually.  Use :meth:`generate` to build
        the desired structure (amorphous, nanocrystalline, etc.).

        Parameters
        ----------
        atoms
            Reference crystal structure (ASE Atoms object).
        cell_dim_angstroms
            Physical supercell lengths in Angstrom.  A single scalar
            produces a cubic box.
        r_max
            Maximum radius for the g3 measurement grid.
        r_step
            Radial bin width for the g3 measurement grid.
        phi_num_bins
            Number of angular bins for the g3 measurement grid.
        relative_density
            Density relative to the crystalline reference.
        rng_seed
            Random seed for reproducible initialization.
        label
            Human-readable label.
        **kwargs
            Forwarded to :meth:`Supercell.__init__`.

        Returns
        -------
        Supercell
            A new random supercell ready for :meth:`generate` or
            :meth:`shell_relax`.
        """
        dist = G3Distribution(atoms, label=label or "reference")
        dist.measure_g3(
            r_max=r_max,
            r_step=r_step,
            phi_num_bins=phi_num_bins,
            show_progress=False,
        )
        return cls(
            dist,
            cell_dim_angstroms=cell_dim_angstroms,
            relative_density=relative_density,
            measure_g3=False,
            rng_seed=rng_seed,
            label=label,
            **kwargs,
        )

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

    def _build_grain_atoms(
        self,
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

        # --- build rotated crystal tiling per grain ---
        # Each crystalline grain gets its own rotated copy of the
        # reference crystal, tiled just enough to cover the grain's
        # Voronoi cell.  All grains' atoms are then collected and a
        # single batched Voronoi assignment keeps only the atoms
        # belonging to each grain.
        ref_cell = np.asarray(self.reference_atoms.cell.array, dtype=np.float64)
        ref_pos = self.reference_atoms.positions.copy()
        ref_numbers = self.reference_atoms.numbers.copy()

        box_lengths = np.array(self.cell_dim_angstroms, dtype=np.float64)
        ref_lengths = np.linalg.norm(ref_cell, axis=1)
        ref_lengths = np.maximum(ref_lengths, _EPS)

        # Tile enough to cover the largest possible Voronoi cell
        # (upper bound: the full box, but for many grains each cell
        # is much smaller).  We tile to cover grain_radius + buffer.
        tile_radius = grain_radius + float(np.max(ref_lengths))
        n_reps = np.ceil(tile_radius / ref_lengths).astype(int)
        n_reps = np.maximum(n_reps, 1)

        shifts = []
        for ix in range(-n_reps[0], n_reps[0] + 1):
            for iy in range(-n_reps[1], n_reps[1] + 1):
                for iz in range(-n_reps[2], n_reps[2] + 1):
                    shifts.append([ix, iy, iz])
        shifts = np.array(shifts, dtype=np.float64)
        shift_cart = shifts @ ref_cell

        # Base tiled block centered at origin
        base_pos = (ref_pos[None, :, :] + shift_cart[:, None, :]).reshape(-1, 3)
        base_center = np.mean(base_pos, axis=0)
        base_pos -= base_center
        base_numbers = np.tile(ref_numbers, len(shifts))

        # For each crystalline grain: rotate, translate, wrap, collect
        all_raw_pos: list[np.ndarray] = []
        all_raw_numbers: list[np.ndarray] = []
        all_raw_grain_ids: list[np.ndarray] = []

        for ig in range(n_seeds):
            if not is_crystalline_cell[ig]:
                continue
            # Rotate the base block and place at seed
            rotated = base_pos @ rotations[ig].T
            grain_pos = rotated + seed_cart[ig]
            # Wrap into periodic box
            frac = grain_pos @ cell_inv
            frac %= 1.0
            grain_pos = frac @ cell_mat
            all_raw_pos.append(grain_pos)
            all_raw_numbers.append(base_numbers.copy())
            all_raw_grain_ids.append(np.full(len(grain_pos), ig, dtype=np.intp))

        if not all_raw_pos:
            all_positions: list[np.ndarray] = []
            all_numbers: list[np.ndarray] = []
            all_grain_ids: list[np.ndarray] = []
        else:
            # Concatenate all grain candidates
            raw_pos = np.concatenate(all_raw_pos, axis=0)
            raw_numbers = np.concatenate(all_raw_numbers, axis=0)
            raw_gids = np.concatenate(all_raw_grain_ids, axis=0)

            # Single batched Voronoi assignment: keep each atom only
            # if its grain's seed is the nearest seed.
            delta_all = raw_pos[:, None, :] - seed_cart[None, :, :]
            frac_delta = delta_all @ cell_inv
            frac_delta -= np.rint(frac_delta)
            cart_delta = frac_delta @ cell_mat
            dist_sq = np.sum(cart_delta ** 2, axis=2)
            nearest_seed = np.argmin(dist_sq, axis=1)

            keep = nearest_seed == raw_gids
            all_positions = [raw_pos[keep]]
            all_numbers = [raw_numbers[keep]]
            all_grain_ids = [raw_gids[keep]]

        # --- fill amorphous Voronoi cells with random positions ---
        if n_crystalline < n_seeds:
            amorphous_volume_fraction = (n_seeds - n_crystalline) / max(n_seeds, 1)
            n_amorphous_target = int(np.round(total_target * amorphous_volume_fraction))

            if n_amorphous_target > 0:
                # Over-sample random positions, keep those in amorphous cells
                oversample = max(2, int(np.ceil(n_seeds / max(n_seeds - n_crystalline, 1))))
                n_candidates = n_amorphous_target * oversample
                cand_frac = self.rng.random((n_candidates, 3))
                cand_pos = cand_frac @ cell_mat

                # Assign candidates to nearest seed
                delta_cand = cand_pos[:, None, :] - seed_cart[None, :, :]
                frac_delta_c = delta_cand @ cell_inv
                frac_delta_c -= np.rint(frac_delta_c)
                cart_delta_c = frac_delta_c @ cell_mat
                dist_sq_c = np.sum(cart_delta_c ** 2, axis=2)
                nearest_c = np.argmin(dist_sq_c, axis=1)

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

        # --- remove duplicate/overlapping atoms ---
        # Use a hard minimum distance to detect overlaps
        pair_hard_min = np.asarray(shell_target.pair_hard_min, dtype=np.float64)
        global_hard_min = float(np.min(pair_hard_min[pair_hard_min > _EPS])) if np.any(pair_hard_min > _EPS) else 0.5
        # Use 80% of hard min as overlap threshold
        overlap_thresh = global_hard_min * 0.8

        # Build a temporary Atoms to use neighbor_list
        temp_atoms = Atoms(
            numbers=numbers,
            positions=positions,
            cell=cell,
            pbc=self.reference_atoms.pbc,
        )
        ov_i, ov_j, ov_d = neighbor_list("ijd", temp_atoms, overlap_thresh)

        # Mark atoms to remove: for each overlapping pair, remove the one
        # with the higher index (arbitrary but consistent)
        remove = set()
        for k in range(len(ov_i)):
            if ov_i[k] < ov_j[k] and ov_i[k] not in remove:
                remove.add(int(ov_j[k]))

        if remove:
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

    @staticmethod
    def _broadening_to_weights(
        pair_peak_max: float,
        r_broadening: float | None = None,
        phi_broadening: float | None = None,
    ) -> dict[str, float]:
        """Map broadening parameters to shell_relax force weights.

        Parameters
        ----------
        pair_peak_max
            Maximum first-shell distance across all species pairs (Å).
        r_broadening
            Radial disorder σ in Å at the NN distance.  ``None`` uses a
            default bond_weight of 1.0.
        phi_broadening
            Angular disorder σ in degrees.  ``None`` uses a default
            angle_weight of 0.5.  180 effectively disables angle forces.

        Returns
        -------
        dict with ``bond_weight`` and ``angle_weight`` keys.
        """
        # --- radial broadening → bond_weight ---
        if r_broadening is not None and r_broadening > _EPS:
            r_norm = float(r_broadening) / max(pair_peak_max, _EPS)
            bond_weight = float(np.clip(0.3 / max(r_norm, 0.01), 0.3, 3.0))
        else:
            bond_weight = 1.0

        # --- angular broadening → angle_weight ---
        if phi_broadening is not None and phi_broadening > _EPS:
            angle_weight = float(np.clip(
                3.0 / max(float(phi_broadening), 1.0), 0.05, 2.0,
            ))
        else:
            angle_weight = 0.5

        return {
            "bond_weight": bond_weight,
            "angle_weight": angle_weight,
        }

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

    def _build_g3_rr_weights(self) -> np.ndarray:
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

    def _weighted_g3_cost(self, g3_diff_flat: np.ndarray) -> float:
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

    def _weighted_delta_cost(self, delta_g3_idx: np.ndarray, delta_g3_val: np.ndarray) -> float:
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

    def _cell_face_spacings(self, cell_matrix: np.ndarray) -> np.ndarray:
        """Return periodic face spacings for the cell spanned by the row vectors."""
        inverse = np.linalg.inv(np.asarray(cell_matrix, dtype=np.float64))
        return 1.0 / np.maximum(np.linalg.norm(inverse, axis=0), _EPS)

    def _wrapped_fractional_positions(self, positions: np.ndarray) -> np.ndarray:
        """Return wrapped fractional coordinates in `[0, 1)` for Cartesian positions."""
        frac = np.asarray(positions, dtype=np.float64) @ self._cell_inverse
        frac %= 1.0
        return frac

    def _spatial_flat_index(self, cell_index: np.ndarray) -> int:
        """Flatten a 3D spatial-hash cell index."""
        return int(np.ravel_multi_index(tuple(np.asarray(cell_index, dtype=np.intp)), self._spatial_shape))

    def _spatial_cell_index_for_position(self, position: np.ndarray) -> np.ndarray:
        """Map a Cartesian position to the wrapped spatial-hash bin index."""
        frac = self._wrapped_fractional_positions(np.asarray(position, dtype=np.float64)[None, :])[0]
        cell_index = np.floor(frac * self._spatial_shape[None, :]).astype(np.intp)[0]
        np.minimum(cell_index, self._spatial_shape - 1, out=cell_index)
        return cell_index

    def _spatial_search_ranges(self, cutoff: float) -> np.ndarray:
        """Return the number of neighboring spatial bins to inspect along each axis."""
        cutoff = float(max(cutoff, 0.0))
        return np.maximum(
            1,
            np.ceil(cutoff / np.maximum(self._spatial_bin_face_spacings, _EPS)).astype(np.intp),
        )

    def _spatial_neighbor_offsets(self, search_ranges: np.ndarray) -> np.ndarray:
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

    def _rebuild_spatial_index(self) -> None:
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
        self,
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

    def _candidate_indices_for_position(self, position: np.ndarray, cutoff: float) -> np.ndarray:
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
        self,
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

    def _nearest_neighbor_vectors(self, cutoff: float) -> tuple[np.ndarray, np.ndarray]:
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
        self,
        start_positions: np.ndarray,
        end_positions: np.ndarray,
    ) -> np.ndarray:
        """Return minimum-image Cartesian displacements from `start` to `end`."""
        delta = np.asarray(end_positions, dtype=np.float64) - np.asarray(start_positions, dtype=np.float64)
        frac = delta @ self._cell_inverse
        frac -= np.rint(frac)
        return frac @ self._cell_matrix

    def _capture_teacher_snapshot(
        self,
        *,
        stage_code: int,
        step: int,
        accepted_moves: int,
        attempted_moves: int,
    ) -> dict[str, Any]:
        """Capture the current state for teacher-trajectory export."""
        nearest_dist, _ = self._nearest_neighbor_vectors(self.measure_r_max)
        return {
            "positions": np.array(self.atoms.positions, dtype=np.float32, copy=True),
            "stage_code": int(stage_code),
            "step": int(step),
            "accepted_moves": int(accepted_moves),
            "attempted_moves": int(attempted_moves),
            "cost_g3_weighted": float(self.current_cost),
            "cost_g3_unweighted": float(np.dot(self._g3_diff_flat, self._g3_diff_flat)),
            "nn_min": float(np.min(nearest_dist)),
            "nn_mean": float(np.mean(nearest_dist)),
        }

    def _assemble_teacher_rollout(
        self,
        *,
        snapshots: list[dict[str, Any]],
        target_id: str,
        repulsion_summary: dict[str, Any] | None,
        monte_carlo_summary: dict[str, Any],
        snapshot_stride_accepted: int,
        r_min_nn: float | None,
    ) -> dict[str, Any]:
        """Pack captured teacher snapshots into array form for serialization."""
        if len(snapshots) < 2:
            raise ValueError("Teacher rollout requires at least two captured snapshots.")
        positions = np.stack([snap["positions"] for snap in snapshots], axis=0).astype(np.float32)
        delta_positions = self._minimum_image_position_delta(positions[:-1], positions[1:]).astype(
            np.float32,
            copy=False,
        )
        return {
            "atom_numbers": np.asarray(self.atoms.numbers, dtype=np.int16),
            "cell_matrix": np.asarray(self._cell_matrix, dtype=np.float32),
            "target_g3": np.asarray(self.target_distribution.g3, dtype=np.float32),
            "target_g2": np.asarray(self.target_distribution.g2, dtype=np.float32),
            "target_weight_rr": self._g3_rr_weights_flat.reshape(self._r_num, self._r_num).astype(
                np.float32,
                copy=False,
            ),
            "positions": positions,
            "delta_positions": delta_positions,
            "snapshot_stage_codes": np.asarray(
                [snap["stage_code"] for snap in snapshots],
                dtype=np.int8,
            ),
            "snapshot_steps": np.asarray([snap["step"] for snap in snapshots], dtype=np.int32),
            "snapshot_accepted_moves": np.asarray(
                [snap["accepted_moves"] for snap in snapshots],
                dtype=np.int32,
            ),
            "snapshot_attempted_moves": np.asarray(
                [snap["attempted_moves"] for snap in snapshots],
                dtype=np.int32,
            ),
            "snapshot_cost_g3_weighted": np.asarray(
                [snap["cost_g3_weighted"] for snap in snapshots],
                dtype=np.float32,
            ),
            "snapshot_cost_g3_unweighted": np.asarray(
                [snap["cost_g3_unweighted"] for snap in snapshots],
                dtype=np.float32,
            ),
            "snapshot_nn_min": np.asarray([snap["nn_min"] for snap in snapshots], dtype=np.float32),
            "snapshot_nn_mean": np.asarray([snap["nn_mean"] for snap in snapshots], dtype=np.float32),
            "relative_density": np.asarray(self.relative_density, dtype=np.float32),
            "r_min_nn": np.asarray(np.nan if r_min_nn is None else float(r_min_nn), dtype=np.float32),
            "r_max": np.asarray(self.measure_r_max, dtype=np.float32),
            "r_step": np.asarray(self.measure_r_step, dtype=np.float32),
            "phi_num_bins": np.asarray(self.measure_phi_num_bins, dtype=np.int32),
            "g3_weight_r_scale": np.asarray(self.g3_weight_r_scale, dtype=np.float32),
            "g3_weight_exponent": np.asarray(self.g3_weight_exponent, dtype=np.float32),
            "g3_weight_floor": np.asarray(self.g3_weight_floor, dtype=np.float32),
            "snapshot_stride_accepted": np.asarray(snapshot_stride_accepted, dtype=np.int32),
            "repulsion_steps": np.asarray(
                0 if repulsion_summary is None else int(repulsion_summary["num_steps"]),
                dtype=np.int32,
            ),
            "mc_steps": np.asarray(monte_carlo_summary["num_steps"], dtype=np.int32),
            "target_id": np.asarray(target_id),
        }

    def _save_teacher_rollout_npz(self, rollout: dict[str, Any], path: Path) -> Path:
        """Write a teacher rollout to a compressed NumPy archive."""
        np.savez_compressed(path, **rollout)
        return path

    def _save_teacher_rollout_hdf5(self, rollout: dict[str, Any], path: Path) -> Path:
        """Write a teacher rollout to HDF5."""
        try:
            import h5py
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Saving teacher rollouts as HDF5 requires the optional dependency `h5py`."
            ) from exc

        with h5py.File(path, "w") as h5:
            h5.attrs["stage_code_labels"] = "0=random,1=repulsion,2=mc,3=mc_final"
            for key, value in rollout.items():
                array = np.asarray(value)
                if array.ndim == 0:
                    item = array.item()
                    if isinstance(item, str):
                        h5.attrs[key] = item
                    else:
                        h5.attrs[key] = item
                    continue
                chunks = None
                if key in {"positions", "delta_positions"} and array.ndim == 3:
                    chunks = (1, array.shape[1], array.shape[2])
                h5.create_dataset(
                    key,
                    data=array,
                    compression="gzip",
                    shuffle=True,
                    chunks=chunks,
                )
        return path

    def _save_teacher_rollout_zarr(self, rollout: dict[str, Any], path: Path) -> Path:
        """Write a teacher rollout to a zipped Zarr store."""
        try:
            import zarr
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Saving teacher rollouts as Zarr requires the optional dependency `zarr`."
            ) from exc

        with zarr.storage.ZipStore(str(path), mode="w") as store:
            root = zarr.open_group(store=store, mode="w")
            root.attrs["stage_code_labels"] = "0=random,1=repulsion,2=mc,3=mc_final"
            for key, value in rollout.items():
                array = np.asarray(value)
                if array.ndim == 0:
                    root.attrs[key] = array.item()
                    continue
                chunks = None
                if key in {"positions", "delta_positions"} and array.ndim == 3:
                    chunks = (1, array.shape[1], array.shape[2])
                root.create_dataset(
                    key,
                    data=array,
                    shape=array.shape,
                    chunks=chunks,
                )
        return path

    def _save_teacher_rollout(
        self,
        rollout: dict[str, Any],
        path: str | Path,
        *,
        output_format: str,
    ) -> Path:
        """Save a teacher rollout using the requested backend."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        format_key = output_format.lower()
        if format_key == "npz":
            return self._save_teacher_rollout_npz(rollout, path)
        if format_key in {"hdf5", "h5"}:
            return self._save_teacher_rollout_hdf5(rollout, path)
        if format_key == "zarr":
            return self._save_teacher_rollout_zarr(rollout, path)
        raise ValueError("output_format must be one of 'npz', 'hdf5', or 'zarr'.")

    def _initialize_mc_state(self) -> None:
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
        neighbor_indices, _, _ = self._query_local_environment(
            atom_index,
            origin_position,
            self.measure_r_max,
        )
        return neighbor_indices

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

    def _shell_window_membership(
        self,
        radius: np.ndarray,
        inner: float,
        outer: float,
        sigma: float,
    ) -> np.ndarray:
        """Return a smooth shell-membership weight between `inner` and `outer`."""
        if outer <= inner:
            return np.zeros_like(np.asarray(radius, dtype=np.float64))
        sigma = float(max(sigma, 0.25 * self.measure_r_step, 1e-3))
        radius = np.asarray(radius, dtype=np.float64)
        left_arg = np.clip((radius - inner) / sigma, -60.0, 60.0)
        right_arg = np.clip((outer - radius) / sigma, -60.0, 60.0)
        left = 1.0 / (1.0 + np.exp(-left_arg))
        right = 1.0 / (1.0 + np.exp(-right_arg))
        return left * right

    def _random_perpendicular_vector(self, vector: np.ndarray) -> np.ndarray:
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

    def _rotate_vector_about_axis(
        self,
        vector: np.ndarray,
        axis: np.ndarray,
        angle_rad: float,
    ) -> np.ndarray:
        """Rotate `vector` about `axis` using Rodrigues' rotation formula."""
        vector = np.asarray(vector, dtype=np.float64)
        axis = np.asarray(axis, dtype=np.float64)
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm <= _EPS or abs(float(angle_rad)) <= _EPS:
            return np.array(vector, copy=True)
        axis_unit = axis / axis_norm
        cos_a = float(np.cos(angle_rad))
        sin_a = float(np.sin(angle_rad))
        return (
            vector * cos_a
            + np.cross(axis_unit, vector) * sin_a
            + axis_unit * np.dot(axis_unit, vector) * (1.0 - cos_a)
        )

    def _query_local_environment_override(
        self,
        atom_index: int,
        origin_position: np.ndarray,
        cutoff: float,
        *,
        moved_atom: int | None = None,
        moved_position: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return local neighbors while optionally overriding one atom's position."""
        candidate_indices = self._candidate_indices_for_position(origin_position, cutoff)
        if moved_atom is not None and moved_atom != int(atom_index):
            moved_atom_array = np.array([int(moved_atom)], dtype=np.intp)
            if candidate_indices.size == 0:
                candidate_indices = moved_atom_array
            else:
                candidate_indices = np.unique(
                    np.concatenate([candidate_indices.astype(np.intp, copy=False), moved_atom_array])
                )
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

        candidate_positions = np.array(self.atoms.positions[candidate_indices], dtype=np.float64, copy=True)
        if moved_atom is not None and moved_position is not None:
            moved_mask = candidate_indices == int(moved_atom)
            if np.any(moved_mask):
                candidate_positions[moved_mask] = np.asarray(moved_position, dtype=np.float64)

        vectors = self._minimum_image_vectors(np.asarray(origin_position, dtype=np.float64), candidate_positions)
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

    def _shell_center_state(
        self,
        center_index: int,
        shell_target: CoordinationShellTarget,
        *,
        moved_atom: int | None = None,
        moved_position: np.ndarray | None = None,
        count_weight: float,
        radius_weight: float,
        angle_weight: float,
        overlap_weight: float,
        return_details: bool = True,
    ) -> dict[str, Any] | float:
        """Return the local first-shell loss and descriptors around one center atom."""
        center_index = int(center_index)
        center_species = int(self._atom_species_index[center_index])
        center_position = (
            np.asarray(moved_position, dtype=np.float64)
            if moved_atom is not None and center_index == int(moved_atom)
            else np.asarray(self.atoms.positions[center_index], dtype=np.float64)
        )

        cutoff = float(shell_target.max_pair_outer_by_center[center_species])
        count_values = np.zeros(self._num_species, dtype=np.float64)
        count_residual = -np.asarray(shell_target.coordination_target[center_species], dtype=np.float64)
        per_species_radius_loss = np.zeros(self._num_species, dtype=np.float64)
        shell_weights = np.empty(0, dtype=np.float64)
        neighbor_indices = np.empty(0, dtype=np.intp)
        neighbor_species = np.empty(0, dtype=np.intp)
        vectors = np.empty((0, 3), dtype=np.float64)
        radii = np.empty(0, dtype=np.float64)
        count_loss = 0.0
        radius_loss = 0.0
        overlap_loss = 0.0

        if cutoff > 0.0:
            neighbor_indices, vectors, radius_sq = self._query_local_environment_override(
                center_index,
                center_position,
                cutoff,
                moved_atom=moved_atom,
                moved_position=moved_position,
            )
            if neighbor_indices.size:
                radii = np.sqrt(np.maximum(radius_sq, _EPS))
                neighbor_species = self._atom_species_index[neighbor_indices]
                shell_weights = np.zeros_like(radii, dtype=np.float64)

                for neigh_species_index in range(self._num_species):
                    target_count = float(shell_target.coordination_target[center_species, neigh_species_index])
                    if not shell_target.pair_mask[center_species, neigh_species_index]:
                        continue
                    species_mask = neighbor_species == neigh_species_index
                    if not np.any(species_mask):
                        scale = max(target_count, 1.0)
                        count_loss += ((0.0 - target_count) / scale) ** 2
                        continue

                    local_radii = radii[species_mask]
                    sigma_r = float(shell_target.pair_sigma[center_species, neigh_species_index])
                    weights = self._shell_window_membership(
                        local_radii,
                        float(shell_target.pair_inner[center_species, neigh_species_index]),
                        float(shell_target.pair_outer[center_species, neigh_species_index]),
                        sigma_r,
                    )
                    shell_weights[species_mask] = weights
                    count_values[neigh_species_index] = float(np.sum(weights))
                    count_residual[neigh_species_index] = count_values[neigh_species_index] - target_count

                    scale = max(target_count, 1.0)
                    count_loss += (count_residual[neigh_species_index] / scale) ** 2

                    if np.any(weights > 1e-8):
                        peak = float(shell_target.pair_peak[center_species, neigh_species_index])
                        radius_term = float(np.sum(weights * ((local_radii - peak) / max(sigma_r, _EPS)) ** 2) / scale)
                        per_species_radius_loss[neigh_species_index] = radius_term
                        radius_loss += radius_term

                    hard_min = float(shell_target.pair_hard_min[center_species, neigh_species_index])
                    if hard_min > 0.0:
                        overlap = np.maximum(hard_min - local_radii, 0.0) / max(sigma_r, _EPS)
                        overlap_loss += float(np.sum(overlap * overlap))
            else:
                for neigh_species_index in range(self._num_species):
                    if not shell_target.pair_mask[center_species, neigh_species_index]:
                        continue
                    target_count = float(shell_target.coordination_target[center_species, neigh_species_index])
                    scale = max(target_count, 1.0)
                    count_loss += ((0.0 - target_count) / scale) ** 2

        angle_triplet_loss = np.zeros(shell_target.angle_index.shape[0], dtype=np.float64)
        angle_loss = 0.0
        if neighbor_indices.size:
            active_triplets = np.flatnonzero(shell_target.angle_index[:, 0] == center_species)
            for triplet_index in active_triplets:
                _, species_1, species_2 = shell_target.angle_index[triplet_index]
                mask_1 = (neighbor_species == species_1) & (shell_weights > 1e-3)
                mask_2 = (neighbor_species == species_2) & (shell_weights > 1e-3)
                if not np.any(mask_1) or not np.any(mask_2):
                    continue

                vector_1 = vectors[mask_1]
                vector_2 = vectors[mask_2]
                weight_1 = shell_weights[mask_1]
                weight_2 = shell_weights[mask_2]
                radius_1_sq = np.einsum("ij,ij->i", vector_1, vector_1)
                radius_2_sq = np.einsum("ij,ij->i", vector_2, vector_2)
                dot = vector_1 @ vector_2.T
                denom = np.sqrt(np.maximum(radius_1_sq[:, None] * radius_2_sq[None, :], _EPS))
                cos_phi = np.clip(dot / denom, -1.0, 1.0)
                phi_bin = np.floor(np.arccos(cos_phi) / max(float(shell_target.phi_edges[1] - shell_target.phi_edges[0]), _EPS)).astype(np.intp)
                np.clip(phi_bin, 0, shell_target.phi_num_bins - 1, out=phi_bin)

                if species_1 == species_2:
                    if vector_1.shape[0] < 2:
                        continue
                    pair_weights = weight_1[:, None] * weight_2[None, :]
                    upper = np.triu_indices(phi_bin.shape[0], k=1)
                    phi_values = phi_bin[upper]
                    pair_values = pair_weights[upper]
                else:
                    phi_values = phi_bin.ravel()
                    pair_values = (weight_1[:, None] * weight_2[None, :]).ravel()

                local_mass = float(np.sum(pair_values))
                if local_mass <= _EPS:
                    continue
                local_hist = np.bincount(
                    phi_values,
                    weights=pair_values,
                    minlength=shell_target.phi_num_bins,
                ).astype(np.float64, copy=False)
                local_hist /= local_mass
                target_hist = np.asarray(shell_target.angle_target[triplet_index], dtype=np.float64)
                target_mass = float(shell_target.angle_pair_mass_target[triplet_index])
                triplet_loss = float(target_mass * np.dot(local_hist - target_hist, local_hist - target_hist))
                angle_triplet_loss[triplet_index] = triplet_loss
                angle_loss += triplet_loss

        component_values = {
            "count": float(count_loss),
            "radius": float(radius_loss),
            "angle": float(angle_loss),
            "overlap": float(overlap_loss),
        }
        component_scores = {
            "count": float(count_weight * count_loss),
            "radius": float(radius_weight * radius_loss),
            "angle": float(angle_weight * angle_loss),
            "overlap": float(overlap_weight * overlap_loss),
        }
        total_loss = float(sum(component_scores.values()))
        if not return_details:
            return total_loss
        return {
            "center_index": center_index,
            "center_species": center_species,
            "center_position": center_position,
            "neighbor_indices": neighbor_indices,
            "neighbor_species": neighbor_species,
            "vectors": vectors,
            "radii": radii,
            "shell_weights": shell_weights,
            "count_values": count_values,
            "count_residual": count_residual,
            "per_species_radius_loss": per_species_radius_loss,
            "angle_triplet_loss": angle_triplet_loss,
            "component_values": component_values,
            "component_scores": component_scores,
            "total_loss": total_loss,
        }

    def _shell_affected_centers(
        self,
        atom_index: int,
        old_position: np.ndarray,
        new_position: np.ndarray,
        cutoff: float,
    ) -> np.ndarray:
        """Return center atoms whose first-shell descriptors change after a move."""
        affected = [np.array([int(atom_index)], dtype=np.intp)]
        if cutoff > 0.0:
            affected.append(self._candidate_indices_for_position(old_position, cutoff))
            affected.append(self._candidate_indices_for_position(new_position, cutoff))
        return np.unique(np.concatenate(affected)).astype(np.intp, copy=False)

    def _build_shell_candidates(
        self,
        center_state: dict[str, Any],
        shell_target: CoordinationShellTarget,
        *,
        move_scale: float,
        trials_per_step: int,
        recruit_cutoff_scale: float,
        max_angle_step_deg: float,
    ) -> list[tuple[int, np.ndarray, str]]:
        """Return guided trial moves for the highest-loss local shell motif."""
        center_index = int(center_state["center_index"])
        center_species = int(center_state["center_species"])
        center_position = np.asarray(center_state["center_position"], dtype=np.float64)
        neighbor_indices = np.asarray(center_state["neighbor_indices"], dtype=np.intp)
        neighbor_species = np.asarray(center_state["neighbor_species"], dtype=np.intp)
        vectors = np.asarray(center_state["vectors"], dtype=np.float64)
        radii = np.asarray(center_state["radii"], dtype=np.float64)
        shell_weights = np.asarray(center_state["shell_weights"], dtype=np.float64)
        count_residual = np.asarray(center_state["count_residual"], dtype=np.float64)
        radius_loss = np.asarray(center_state["per_species_radius_loss"], dtype=np.float64)
        angle_triplet_loss = np.asarray(center_state["angle_triplet_loss"], dtype=np.float64)
        component_scores = dict(center_state["component_scores"])

        mode_priority = max(component_scores, key=component_scores.get)
        if component_scores[mode_priority] <= _EPS:
            mode_priority = "random"

        moved_atom: int | None = None
        anchor_positions: list[np.ndarray] = []
        mode_name = mode_priority

        if mode_priority == "overlap" and radii.size:
            hard_min = shell_target.pair_hard_min[center_species, neighbor_species]
            shortfall = np.maximum(hard_min - radii, 0.0)
            if np.any(shortfall > 0.0):
                local_index = int(np.argmax(shortfall))
                moved_atom = int(neighbor_indices[local_index])
                direction = vectors[local_index] / max(float(radii[local_index]), _EPS)
                sigma_r = float(shell_target.pair_sigma[center_species, neighbor_species[local_index]])
                target_radius = max(
                    float(shell_target.pair_peak[center_species, neighbor_species[local_index]]),
                    float(shell_target.pair_hard_min[center_species, neighbor_species[local_index]] + 0.75 * sigma_r),
                )
                anchor_positions.append(self._wrap_position(center_position + direction * target_radius))

        if moved_atom is None and mode_priority in {"count", "radius"}:
            if mode_priority == "count" and np.any(count_residual > 0.15):
                species_choice = int(np.argmax(count_residual))
                candidate_mask = (neighbor_species == species_choice) & (shell_weights > 0.05)
                if np.any(candidate_mask):
                    local_candidates = np.flatnonzero(candidate_mask)
                    local_index = int(local_candidates[np.argmax(shell_weights[local_candidates])])
                    moved_atom = int(neighbor_indices[local_index])
                    direction = vectors[local_index] / max(float(radii[local_index]), _EPS)
                    sigma_r = float(shell_target.pair_sigma[center_species, species_choice])
                    target_radius = float(shell_target.pair_outer[center_species, species_choice] + 0.8 * sigma_r)
                    anchor_positions.append(self._wrap_position(center_position + direction * target_radius))
                    mode_name = "count_excess"
            if moved_atom is None and mode_priority == "count" and np.any(count_residual < -0.15):
                species_choice = int(np.argmin(count_residual))
                recruit_cutoff = float(
                    max(
                        shell_target.pair_outer[center_species, species_choice] * recruit_cutoff_scale,
                        shell_target.pair_peak[center_species, species_choice] + 2.5 * shell_target.pair_sigma[center_species, species_choice],
                    )
                )
                recruit_indices, recruit_vectors, recruit_sq = self._query_local_environment_override(
                    center_index,
                    center_position,
                    recruit_cutoff,
                )
                recruit_radii = np.sqrt(np.maximum(recruit_sq, _EPS))
                recruit_species = self._atom_species_index[recruit_indices] if recruit_indices.size else np.empty(0, dtype=np.intp)
                candidate_mask = recruit_species == species_choice
                if np.any(candidate_mask):
                    local_candidates = np.flatnonzero(candidate_mask)
                    peak = float(shell_target.pair_peak[center_species, species_choice])
                    local_index = int(local_candidates[np.argmin(np.abs(recruit_radii[local_candidates] - peak))])
                    moved_atom = int(recruit_indices[local_index])
                    direction = recruit_vectors[local_index] / max(float(recruit_radii[local_index]), _EPS)
                    anchor_positions.append(self._wrap_position(center_position + direction * peak))
                    mode_name = "count_deficit"
            if moved_atom is None and mode_priority == "radius" and np.any(radius_loss > 0.0):
                species_choice = int(np.argmax(radius_loss))
                candidate_mask = (neighbor_species == species_choice) & (shell_weights > 0.05)
                if np.any(candidate_mask):
                    local_candidates = np.flatnonzero(candidate_mask)
                    peak = float(shell_target.pair_peak[center_species, species_choice])
                    sigma_r = float(shell_target.pair_sigma[center_species, species_choice])
                    local_score = shell_weights[local_candidates] * np.abs(radii[local_candidates] - peak) / max(sigma_r, _EPS)
                    local_index = int(local_candidates[np.argmax(local_score)])
                    moved_atom = int(neighbor_indices[local_index])
                    direction = vectors[local_index] / max(float(radii[local_index]), _EPS)
                    anchor_positions.append(self._wrap_position(center_position + direction * peak))

        if moved_atom is None and mode_priority == "angle" and np.any(angle_triplet_loss > 0.0):
            triplet_index = int(np.argmax(angle_triplet_loss))
            _, species_1, species_2 = shell_target.angle_index[triplet_index]
            mask_1 = (neighbor_species == species_1) & (shell_weights > 0.05)
            mask_2 = (neighbor_species == species_2) & (shell_weights > 0.05)
            target_angle = float(np.deg2rad(shell_target.angle_mode_deg[triplet_index]))
            if np.any(mask_1) and np.any(mask_2):
                idx_1 = np.flatnonzero(mask_1)
                idx_2 = np.flatnonzero(mask_2)
                vector_1 = vectors[idx_1]
                vector_2 = vectors[idx_2]
                dot = vector_1 @ vector_2.T
                norm_1 = np.sqrt(np.maximum(np.einsum("ij,ij->i", vector_1, vector_1), _EPS))
                norm_2 = np.sqrt(np.maximum(np.einsum("ij,ij->i", vector_2, vector_2), _EPS))
                denom = norm_1[:, None] * norm_2[None, :]
                angles = np.arccos(np.clip(dot / denom, -1.0, 1.0))
                weight_matrix = shell_weights[idx_1][:, None] * shell_weights[idx_2][None, :]
                if species_1 == species_2:
                    upper = np.triu_indices(angles.shape[0], k=1)
                    if upper[0].size:
                        angle_score = weight_matrix[upper] * np.abs(angles[upper] - target_angle)
                        best_index = int(np.argmax(angle_score))
                        local_1 = int(idx_1[upper[0][best_index]])
                        local_2 = int(idx_2[upper[1][best_index]])
                    else:
                        local_1 = local_2 = -1
                else:
                    best_pair = np.unravel_index(int(np.argmax(weight_matrix * np.abs(angles - target_angle))), angles.shape)
                    local_1 = int(idx_1[best_pair[0]])
                    local_2 = int(idx_2[best_pair[1]])
                if local_1 >= 0 and local_2 >= 0:
                    dev_1 = abs(
                        radii[local_1]
                        - float(shell_target.pair_peak[center_species, neighbor_species[local_1]])
                    )
                    dev_2 = abs(
                        radii[local_2]
                        - float(shell_target.pair_peak[center_species, neighbor_species[local_2]])
                    )
                    moved_local = local_1 if dev_1 >= dev_2 else local_2
                    fixed_local = local_2 if moved_local == local_1 else local_1
                    moved_atom = int(neighbor_indices[moved_local])
                    moved_vector = vectors[moved_local]
                    fixed_vector = vectors[fixed_local]
                    current_angle = float(
                        np.arccos(
                            np.clip(
                                np.dot(moved_vector, fixed_vector)
                                / max(float(np.linalg.norm(moved_vector) * np.linalg.norm(fixed_vector)), _EPS),
                                -1.0,
                                1.0,
                            )
                        )
                    )
                    delta_angle = float(
                        np.clip(
                            target_angle - current_angle,
                            -np.deg2rad(max_angle_step_deg),
                            np.deg2rad(max_angle_step_deg),
                        )
                    )
                    axis = np.cross(moved_vector, fixed_vector)
                    if np.linalg.norm(axis) <= _EPS:
                        axis = self._random_perpendicular_vector(moved_vector)
                    peak = float(shell_target.pair_peak[center_species, neighbor_species[moved_local]])
                    for sign in (1.0, -1.0):
                        rotated = self._rotate_vector_about_axis(moved_vector, axis, sign * delta_angle)
                        rotated_norm = float(np.linalg.norm(rotated))
                        if rotated_norm <= _EPS:
                            continue
                        anchor_positions.append(
                            self._wrap_position(center_position + rotated * (peak / rotated_norm))
                        )

        if moved_atom is None:
            moved_atom = center_index
            mode_name = "random"
            anchor_positions.append(
                self._wrap_position(
                    center_position + self.rng.normal(scale=max(move_scale, self.measure_r_step), size=3)
                )
            )

        candidates: list[tuple[int, np.ndarray, str]] = []
        old_position = np.asarray(self.atoms.positions[moved_atom], dtype=np.float64)
        if not anchor_positions:
            anchor_positions.append(
                self._wrap_position(old_position + self.rng.normal(scale=max(move_scale, self.measure_r_step), size=3))
            )
        unique_anchors = []
        for anchor in anchor_positions:
            if not any(np.allclose(anchor, existing) for existing in unique_anchors):
                unique_anchors.append(np.asarray(anchor, dtype=np.float64))
        jitter_scale = max(0.35 * move_scale, 0.2 * self.measure_r_step)
        for anchor in unique_anchors:
            candidates.append((moved_atom, self._wrap_position(anchor), mode_name))
        while len(candidates) < int(max(trials_per_step, 1)):
            anchor = unique_anchors[len(candidates) % len(unique_anchors)]
            displacement = anchor - center_position
            displacement = self._minimum_image_vectors(center_position, displacement[None, :] + center_position[None, :])[0]
            if np.linalg.norm(displacement) <= _EPS:
                tangent = self.rng.normal(scale=jitter_scale, size=3)
            elif mode_name.startswith("angle"):
                tangent = self._random_perpendicular_vector(displacement) * self.rng.normal(scale=jitter_scale)
            else:
                tangent = self.rng.normal(scale=jitter_scale, size=3)
            candidates.append((moved_atom, self._wrap_position(anchor + tangent), mode_name))
        return candidates

    def _axis_angle_rotation_matrix(
        self,
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
        self,
        vectors: np.ndarray,
        rotation: np.ndarray,
    ) -> np.ndarray:
        """Apply a column-action rotation matrix to one or more vectors."""
        vectors = np.asarray(vectors, dtype=np.float64)
        rotation = np.asarray(rotation, dtype=np.float64)
        if vectors.ndim == 1:
            return rotation @ vectors
        return vectors @ rotation.T

    def _random_rotation_matrix(self) -> np.ndarray:
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
        self,
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
        self,
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

    def _random_cell_position(self) -> np.ndarray:
        """Return a random wrapped Cartesian point inside the periodic cell."""
        return self.rng.random(3) @ self._cell_matrix

    def _position_overlap_score(
        self,
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

    def _assign_motif_templates(
        self,
        shell_target: CoordinationShellTarget,
    ) -> np.ndarray:
        """Assign a reference-shell motif template to each atom in the supercell."""
        num_atoms = len(self.atoms)
        motif_ids = np.empty(num_atoms, dtype=np.intp)
        motif_center_species = np.asarray(shell_target.motif_center_species, dtype=np.intp)
        for species_index in range(self._num_species):
            atom_indices = np.flatnonzero(self._atom_species_index == species_index)
            if atom_indices.size == 0:
                continue
            candidates = np.flatnonzero(motif_center_species == species_index)
            if candidates.size == 0:
                raise ValueError(
                    f"No motif templates are available for species index {species_index}."
                )
            sequence: list[int] = []
            while len(sequence) < int(atom_indices.size):
                sequence.extend(self.rng.permutation(candidates).tolist())
            self.rng.shuffle(atom_indices)
            motif_ids[atom_indices] = np.asarray(sequence[: atom_indices.size], dtype=np.intp)
        return motif_ids

    def _pair_cross_species_slots(
        self,
        slots_a: list[tuple[int, int]],
        slots_b: list[tuple[int, int]],
        existing_pairs: set[tuple[int, int]],
    ) -> tuple[list[tuple[int, int, int, int]], int]:
        """Greedily pair compatible directed slots across two species."""
        self.rng.shuffle(slots_a)
        self.rng.shuffle(slots_b)
        paired: list[tuple[int, int, int, int]] = []
        available_b = list(slots_b)
        unmatched = 0
        for atom_a, slot_a in slots_a:
            match_index: int | None = None
            for candidate_index, (atom_b, slot_b) in enumerate(available_b):
                pair_key = (min(atom_a, atom_b), max(atom_a, atom_b))
                if atom_a == atom_b or pair_key in existing_pairs:
                    continue
                match_index = candidate_index
                break
            if match_index is None:
                unmatched += 1
                continue
            atom_b, slot_b = available_b.pop(match_index)
            existing_pairs.add((min(atom_a, atom_b), max(atom_a, atom_b)))
            paired.append((atom_a, slot_a, atom_b, slot_b))
        unmatched += len(available_b)
        return paired, unmatched

    def _pair_same_species_slots(
        self,
        slots: list[tuple[int, int]],
        existing_pairs: set[tuple[int, int]],
    ) -> tuple[list[tuple[int, int, int, int]], int]:
        """Greedily pair directed slots within a single species class."""
        self.rng.shuffle(slots)
        paired: list[tuple[int, int, int, int]] = []
        available = list(slots)
        unmatched = 0
        while available:
            atom_a, slot_a = available.pop()
            match_index: int | None = None
            for candidate_index, (atom_b, slot_b) in enumerate(available):
                pair_key = (min(atom_a, atom_b), max(atom_a, atom_b))
                if atom_a == atom_b or pair_key in existing_pairs:
                    continue
                match_index = candidate_index
                break
            if match_index is None:
                unmatched += 1
                continue
            atom_b, slot_b = available.pop(match_index)
            existing_pairs.add((min(atom_a, atom_b), max(atom_a, atom_b)))
            paired.append((atom_a, slot_a, atom_b, slot_b))
        return paired, unmatched

    def _build_motif_graph_data(
        self,
        shell_target: CoordinationShellTarget,
        motif_ids: np.ndarray,
        edges: Sequence[tuple[int, int, int, int]],
        *,
        num_atoms: int,
        rotations: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Build adjacency, degree, and bond metadata for a motif edge list."""
        motif_ids = np.asarray(motif_ids, dtype=np.intp)
        target_degree = np.array(
            [len(shell_target.motif_neighbor_species[int(motif_id)]) for motif_id in motif_ids],
            dtype=np.intp,
        )
        edge_count = len(edges)
        edge_i = np.empty(edge_count, dtype=np.intp)
        edge_j = np.empty(edge_count, dtype=np.intp)
        slot_i = np.empty(edge_count, dtype=np.intp)
        slot_j = np.empty(edge_count, dtype=np.intp)
        target_length = np.empty(edge_count, dtype=np.float64)
        actual_degree = np.zeros(int(num_atoms), dtype=np.intp)
        adjacency: list[list[tuple[int, int, int, int]]] = [[] for _ in range(int(num_atoms))]
        bonded_sets: list[set[int]] = [set() for _ in range(int(num_atoms))]

        for edge_index, (atom_i, local_slot_i, atom_j, local_slot_j) in enumerate(edges):
            atom_i = int(atom_i)
            atom_j = int(atom_j)
            local_slot_i = int(local_slot_i)
            local_slot_j = int(local_slot_j)
            edge_i[edge_index] = atom_i
            edge_j[edge_index] = atom_j
            slot_i[edge_index] = local_slot_i
            slot_j[edge_index] = local_slot_j
            vec_i = shell_target.motif_neighbor_vectors[int(motif_ids[atom_i])][local_slot_i]
            vec_j = shell_target.motif_neighbor_vectors[int(motif_ids[atom_j])][local_slot_j]
            target_length[edge_index] = 0.5 * (
                float(np.linalg.norm(vec_i)) + float(np.linalg.norm(vec_j))
            )
            actual_degree[atom_i] += 1
            actual_degree[atom_j] += 1
            adjacency[atom_i].append((atom_j, local_slot_i, local_slot_j, edge_index))
            adjacency[atom_j].append((atom_i, local_slot_j, local_slot_i, edge_index))
            bonded_sets[atom_i].add(atom_j)
            bonded_sets[atom_j].add(atom_i)

        graph = {
            "edge_i": edge_i,
            "edge_j": edge_j,
            "slot_i": slot_i,
            "slot_j": slot_j,
            "target_length": target_length,
            "target_degree": target_degree,
            "actual_degree": actual_degree,
            "adjacency": adjacency,
            "bonded_sets": bonded_sets,
            "unmatched_slots": int(max(np.sum(target_degree) - 2 * edge_count, 0)),
        }
        if rotations is not None:
            graph["rotations"] = np.asarray(rotations, dtype=np.float64)
        return graph

    def _fit_motif_graph_rotations(
        self,
        positions: np.ndarray,
        shell_target: CoordinationShellTarget,
        motif_ids: np.ndarray,
        graph: dict[str, Any],
        *,
        fallback_rotations: np.ndarray | None = None,
    ) -> np.ndarray:
        """Fit one local frame per atom from the current bonded motif graph."""
        positions = np.asarray(positions, dtype=np.float64)
        motif_ids = np.asarray(motif_ids, dtype=np.intp)
        num_atoms = positions.shape[0]
        rotations = np.empty((num_atoms, 3, 3), dtype=np.float64)
        if fallback_rotations is None and "rotations" in graph:
            fallback_rotations = np.asarray(graph["rotations"], dtype=np.float64)
        for atom_index in range(num_atoms):
            adjacency = graph["adjacency"][atom_index]
            if not adjacency:
                if fallback_rotations is not None and fallback_rotations.shape[0] == num_atoms:
                    rotations[atom_index] = fallback_rotations[atom_index]
                else:
                    rotations[atom_index] = self._random_rotation_matrix()
                continue
            motif_vectors = shell_target.motif_neighbor_vectors[int(motif_ids[atom_index])]
            slot_indices = np.array([entry[1] for entry in adjacency], dtype=np.intp)
            neighbor_indices = np.array([entry[0] for entry in adjacency], dtype=np.intp)
            template_vectors = motif_vectors[slot_indices]
            current_vectors = self._minimum_image_vectors(
                positions[atom_index],
                positions[neighbor_indices],
            )
            rotations[atom_index] = self._fit_rotation_to_vectors(template_vectors, current_vectors)
        return rotations

    def _augment_motif_graph_closures(
        self,
        positions: np.ndarray,
        shell_target: CoordinationShellTarget,
        motif_ids: np.ndarray,
        graph: dict[str, Any],
        *,
        closure_distance_scale: float,
        max_passes: int = 1,
    ) -> dict[str, Any]:
        """Greedily add missing motif bonds whose open slots already align in space."""
        positions = np.asarray(positions, dtype=np.float64)
        motif_ids = np.asarray(motif_ids, dtype=np.intp)
        closure_distance_scale = float(closure_distance_scale)
        if closure_distance_scale <= 0.0:
            return graph
        max_passes = max(int(max_passes), 1)

        edge_list = [
            (
                int(graph["edge_i"][edge_index]),
                int(graph["slot_i"][edge_index]),
                int(graph["edge_j"][edge_index]),
                int(graph["slot_j"][edge_index]),
            )
            for edge_index in range(graph["edge_i"].size)
        ]
        rotations = self._fit_motif_graph_rotations(
            positions,
            shell_target,
            motif_ids,
            graph,
        )

        for _ in range(max_passes):
            trial_graph = self._build_motif_graph_data(
                shell_target,
                motif_ids,
                edge_list,
                num_atoms=positions.shape[0],
                rotations=rotations,
            )
            used_slots = [
                np.zeros(
                    len(shell_target.motif_neighbor_species[int(motif_ids[atom_index])]),
                    dtype=bool,
                )
                for atom_index in range(positions.shape[0])
            ]
            for atom_i, slot_i, atom_j, slot_j in edge_list:
                used_slots[int(atom_i)][int(slot_i)] = True
                used_slots[int(atom_j)][int(slot_j)] = True

            candidate_edges: list[tuple[float, int, int, int, int]] = []
            for atom_i in range(positions.shape[0]):
                species_i = int(self._atom_species_index[atom_i])
                motif_i = int(motif_ids[atom_i])
                for local_slot_i, neighbor_species in enumerate(
                    shell_target.motif_neighbor_species[motif_i]
                ):
                    if used_slots[atom_i][local_slot_i]:
                        continue
                    vec_i = self._apply_rotation(
                        shell_target.motif_neighbor_vectors[motif_i][local_slot_i],
                        rotations[atom_i],
                    )
                    for atom_j in range(atom_i + 1, positions.shape[0]):
                        if int(self._atom_species_index[atom_j]) != int(neighbor_species):
                            continue
                        if atom_j in trial_graph["bonded_sets"][atom_i]:
                            continue
                        motif_j = int(motif_ids[atom_j])
                        reciprocal_slots = np.flatnonzero(
                            (~used_slots[atom_j])
                            & (
                                shell_target.motif_neighbor_species[motif_j]
                                == species_i
                            )
                        )
                        if reciprocal_slots.size == 0:
                            continue
                        actual = self._minimum_image_vectors(
                            positions[atom_i],
                            positions[atom_j][None, :],
                        )[0]
                        for local_slot_j in reciprocal_slots:
                            vec_j = self._apply_rotation(
                                shell_target.motif_neighbor_vectors[motif_j][int(local_slot_j)],
                                rotations[atom_j],
                            )
                            desired = 0.5 * (vec_i - vec_j)
                            mismatch = vec_i + vec_j
                            score = float(
                                np.linalg.norm(actual - desired)
                                + 0.25 * np.linalg.norm(mismatch)
                            )
                            pair_peak = 0.5 * (
                                float(np.linalg.norm(vec_i)) + float(np.linalg.norm(vec_j))
                            )
                            pair_sigma = float(
                                shell_target.pair_sigma[species_i, int(neighbor_species)]
                            )
                            tolerance = float(
                                max(
                                    closure_distance_scale * pair_peak,
                                    2.0 * pair_sigma,
                                    0.25,
                                )
                            )
                            if score <= tolerance:
                                candidate_edges.append(
                                    (
                                        score,
                                        int(atom_i),
                                        int(local_slot_i),
                                        int(atom_j),
                                        int(local_slot_j),
                                    )
                                )

            if not candidate_edges:
                break

            candidate_edges.sort(key=lambda item: item[0])
            added_any = False
            for _, atom_i, slot_i, atom_j, slot_j in candidate_edges:
                if used_slots[atom_i][slot_i] or used_slots[atom_j][slot_j]:
                    continue
                if atom_j in trial_graph["bonded_sets"][atom_i]:
                    continue
                used_slots[atom_i][slot_i] = True
                used_slots[atom_j][slot_j] = True
                trial_graph["bonded_sets"][atom_i].add(atom_j)
                trial_graph["bonded_sets"][atom_j].add(atom_i)
                edge_list.append((atom_i, slot_i, atom_j, slot_j))
                added_any = True

            if not added_any:
                break

            updated_graph = self._build_motif_graph_data(
                shell_target,
                motif_ids,
                edge_list,
                num_atoms=positions.shape[0],
                rotations=rotations,
            )
            rotations = self._fit_motif_graph_rotations(
                positions,
                shell_target,
                motif_ids,
                updated_graph,
                fallback_rotations=rotations,
            )

        return self._build_motif_graph_data(
            shell_target,
            motif_ids,
            edge_list,
            num_atoms=positions.shape[0],
            rotations=rotations,
        )

    def _build_random_motif_graph(
        self,
        shell_target: CoordinationShellTarget,
        motif_ids: np.ndarray,
        *,
        num_retries: int,
    ) -> dict[str, Any]:
        """Build a random bond graph that respects motif neighbor species as closely as possible."""
        motif_neighbor_species = shell_target.motif_neighbor_species
        target_degree = np.array(
            [len(motif_neighbor_species[int(motif_id)]) for motif_id in motif_ids],
            dtype=np.intp,
        )
        best_graph: dict[str, Any] | None = None
        best_matched_slots = -1

        for _ in range(max(int(num_retries), 1)):
            slots_by_pair: dict[tuple[int, int], list[tuple[int, int]]] = {}
            for atom_index, motif_id in enumerate(motif_ids):
                center_species = int(self._atom_species_index[atom_index])
                for slot_index, neighbor_species in enumerate(motif_neighbor_species[int(motif_id)]):
                    key = (center_species, int(neighbor_species))
                    slots_by_pair.setdefault(key, []).append((int(atom_index), int(slot_index)))

            existing_pairs: set[tuple[int, int]] = set()
            edges: list[tuple[int, int, int, int]] = []
            unmatched_slots = 0
            for species_a in range(self._num_species):
                for species_b in range(species_a, self._num_species):
                    if species_a == species_b:
                        paired, unmatched = self._pair_same_species_slots(
                            list(slots_by_pair.get((species_a, species_a), [])),
                            existing_pairs,
                        )
                        edges.extend(paired)
                        unmatched_slots += unmatched
                    else:
                        paired, unmatched = self._pair_cross_species_slots(
                            list(slots_by_pair.get((species_a, species_b), [])),
                            list(slots_by_pair.get((species_b, species_a), [])),
                            existing_pairs,
                        )
                        edges.extend(paired)
                        unmatched_slots += unmatched

            matched_slots = int(2 * len(edges))
            if matched_slots > best_matched_slots:
                best_matched_slots = matched_slots
                best_graph = self._build_motif_graph_data(
                    shell_target,
                    motif_ids,
                    edges,
                    num_atoms=len(self.atoms),
                )
            if matched_slots == int(np.sum(target_degree)):
                break

        if best_graph is None:
            raise RuntimeError("Failed to build a motif bond graph.")
        return best_graph

    def _build_grown_motif_graph(
        self,
        shell_target: CoordinationShellTarget,
        *,
        merge_distance_scale: float,
        show_progress: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        """Grow a random network of local motifs with bond-axis spin matching."""
        remaining_counts = np.bincount(self._atom_species_index, minlength=self._num_species).astype(np.intp)
        motif_center_species = np.asarray(shell_target.motif_center_species, dtype=np.intp)
        motif_candidates_by_species = {
            species_index: np.flatnonzero(motif_center_species == species_index).astype(np.intp, copy=False)
            for species_index in range(self._num_species)
        }
        spin_num_angles = 12
        seed_position_trials = 12
        rotation_refresh_interval = 24

        positions_list: list[np.ndarray] = []
        species_list: list[int] = []
        motif_ids_list: list[int] = []
        rotations_list: list[np.ndarray] = []
        used_slots_list: list[np.ndarray] = []
        expanded_list: list[bool] = []
        bonded_sets_list: list[set[int]] = []
        edges: list[tuple[int, int, int, int]] = []
        target_num_atoms = int(np.sum(remaining_counts))
        build_progress = None
        if show_progress and target_num_atoms > 0:
            build_progress = _TextProgressBar(target_num_atoms, label="Motif build", width=28)
            build_progress.update(0)

        def choose_motif(
            species_index: int,
            required_neighbor_species: int | None = None,
        ) -> int:
            candidates = motif_candidates_by_species.get(int(species_index))
            if candidates is None or candidates.size == 0:
                raise ValueError(f"No motifs are available for species index {species_index}.")
            candidate_ids = np.asarray(candidates, dtype=np.intp)
            self.rng.shuffle(candidate_ids)
            if required_neighbor_species is not None:
                filtered = candidate_ids[
                    [
                        np.any(
                            shell_target.motif_neighbor_species[int(motif_id)]
                            == int(required_neighbor_species)
                        )
                        for motif_id in candidate_ids
                    ]
                ]
                if filtered.size:
                    candidate_ids = filtered
            return int(candidate_ids[0])

        def add_atom(
            species_index: int,
            motif_id: int,
            position: np.ndarray,
            rotation: np.ndarray,
        ) -> int:
            atom_index = len(positions_list)
            positions_list.append(np.asarray(position, dtype=np.float64))
            species_list.append(int(species_index))
            motif_ids_list.append(int(motif_id))
            rotations_list.append(np.asarray(rotation, dtype=np.float64))
            used_slots_list.append(
                np.zeros(
                    len(shell_target.motif_neighbor_species[int(motif_id)]),
                    dtype=bool,
                )
            )
            expanded_list.append(False)
            bonded_sets_list.append(set())
            remaining_counts[int(species_index)] -= 1
            if build_progress is not None:
                build_progress.update(len(positions_list))
            return atom_index

        def open_slots() -> list[tuple[int, int]]:
            slots: list[tuple[int, int]] = []
            for atom_index, used in enumerate(used_slots_list):
                free = np.flatnonzero(~used)
                slots.extend((int(atom_index), int(slot_index)) for slot_index in free)
            return slots

        def slot_vector(atom_index: int, slot_index: int, rotation: np.ndarray | None = None) -> np.ndarray:
            motif_id = int(motif_ids_list[atom_index])
            if rotation is None:
                rotation = rotations_list[atom_index]
            return self._apply_rotation(
                shell_target.motif_neighbor_vectors[motif_id][int(slot_index)],
                rotation,
            )

        def bond_score(
            atom_i: int,
            slot_i: int,
            atom_j: int,
            slot_j: int,
            *,
            position_i: np.ndarray | None = None,
            position_j: np.ndarray | None = None,
            rotation_i: np.ndarray | None = None,
            rotation_j: np.ndarray | None = None,
        ) -> tuple[float, float]:
            if position_i is None:
                position_i = np.asarray(positions_list[atom_i], dtype=np.float64)
            if position_j is None:
                position_j = np.asarray(positions_list[atom_j], dtype=np.float64)
            vec_i = slot_vector(atom_i, slot_i, rotation_i)
            motif_j = int(motif_ids_list[atom_j])
            if rotation_j is None:
                rotation_j = rotations_list[atom_j]
            vec_j = self._apply_rotation(
                shell_target.motif_neighbor_vectors[motif_j][int(slot_j)],
                rotation_j,
            )
            actual = self._minimum_image_vectors(position_i, position_j[None, :])[0]
            desired = 0.5 * (vec_i - vec_j)
            mismatch = vec_i + vec_j
            score = float(np.linalg.norm(actual - desired) + 0.25 * np.linalg.norm(mismatch))
            target_length = 0.5 * (float(np.linalg.norm(vec_i)) + float(np.linalg.norm(vec_j)))
            return score, target_length

        def overlap_penalty(
            species_index: int,
            position: np.ndarray,
            *,
            ignore_atoms: set[int] | None = None,
        ) -> tuple[float, float]:
            if not positions_list:
                return 0.0, 0.0
            ignore_atoms = ignore_atoms or set()
            all_positions = np.asarray(positions_list, dtype=np.float64)
            all_species = np.asarray(species_list, dtype=np.intp)
            deltas = self._minimum_image_vectors(np.asarray(position, dtype=np.float64), all_positions)
            distance = np.linalg.norm(deltas, axis=1)
            hard_min = shell_target.pair_hard_min[int(species_index), all_species]
            if ignore_atoms:
                ignore_idx = np.fromiter(ignore_atoms, dtype=np.intp, count=len(ignore_atoms))
                distance[ignore_idx] = np.inf
            overlap = hard_min - distance
            keep = overlap > 0.0
            if not np.any(keep):
                return 0.0, 0.0
            return float(np.sum(overlap[keep] * overlap[keep])), float(np.max(overlap[keep]))

        def build_open_slot_lookup() -> dict[tuple[int, int], list[tuple[int, int]]]:
            lookup: dict[tuple[int, int], list[tuple[int, int]]] = {}
            for atom_index, used in enumerate(used_slots_list):
                free = np.flatnonzero(~used)
                motif_id = int(motif_ids_list[atom_index])
                center_species = int(species_list[atom_index])
                neighbor_species = shell_target.motif_neighbor_species[motif_id]
                for slot_index in free:
                    key = (center_species, int(neighbor_species[int(slot_index)]))
                    lookup.setdefault(key, []).append((int(atom_index), int(slot_index)))
            return lookup

        def refresh_rotations() -> None:
            if not edges:
                return
            partial_motif_ids = np.asarray(motif_ids_list, dtype=np.intp)
            partial_graph = self._build_motif_graph_data(
                shell_target,
                partial_motif_ids,
                edges,
                num_atoms=len(positions_list),
                rotations=np.asarray(rotations_list, dtype=np.float64),
            )
            fitted = self._fit_motif_graph_rotations(
                np.asarray(positions_list, dtype=np.float64),
                shell_target,
                partial_motif_ids,
                partial_graph,
                fallback_rotations=np.asarray(rotations_list, dtype=np.float64),
            )
            for atom_index in range(len(rotations_list)):
                rotations_list[atom_index] = fitted[atom_index]

        def add_edge(atom_i: int, slot_i: int, atom_j: int, slot_j: int) -> None:
            if used_slots_list[int(atom_i)][int(slot_i)] or used_slots_list[int(atom_j)][int(slot_j)]:
                return
            if int(atom_j) in bonded_sets_list[int(atom_i)]:
                return
            used_slots_list[int(atom_i)][int(slot_i)] = True
            used_slots_list[int(atom_j)][int(slot_j)] = True
            bonded_sets_list[int(atom_i)].add(int(atom_j))
            bonded_sets_list[int(atom_j)].add(int(atom_i))
            edges.append((int(atom_i), int(slot_i), int(atom_j), int(slot_j)))

        def add_seed_atom() -> bool:
            available_species = np.flatnonzero(remaining_counts > 0)
            if available_species.size == 0:
                return False
            species_index = int(self.rng.choice(available_species))
            motif_id = choose_motif(species_index)
            best_position: np.ndarray | None = None
            best_penalty: tuple[float, float] | None = None
            for _ in range(seed_position_trials):
                candidate_position = self._random_cell_position()
                penalty = overlap_penalty(species_index, candidate_position)
                if best_penalty is None or penalty < best_penalty:
                    best_penalty = penalty
                    best_position = candidate_position
                    if penalty[0] <= _EPS:
                        break
            if best_position is None:
                best_position = self._random_cell_position()
            add_atom(
                species_index,
                motif_id,
                self._wrap_position(best_position),
                self._random_rotation_matrix(),
            )
            return True

        def best_existing_match(
            atom_i: int,
            slot_i: int,
            open_slot_lookup: dict[tuple[int, int], list[tuple[int, int]]],
        ) -> tuple[int, int] | None:
            species_i = int(species_list[atom_i])
            motif_i = int(motif_ids_list[atom_i])
            neighbor_species = int(shell_target.motif_neighbor_species[motif_i][slot_i])
            best_match: tuple[int, int] | None = None
            best_score = np.inf
            for atom_j, slot_j in open_slot_lookup.get((neighbor_species, species_i), []):
                atom_j = int(atom_j)
                slot_j = int(slot_j)
                if atom_j == atom_i:
                    continue
                if used_slots_list[atom_j][slot_j]:
                    continue
                if atom_j in bonded_sets_list[atom_i]:
                    continue
                score, target_length = bond_score(atom_i, slot_i, atom_j, slot_j)
                pair_sigma = float(shell_target.pair_sigma[species_i, neighbor_species])
                tolerance = float(
                    max(
                        merge_distance_scale * target_length,
                        2.0 * pair_sigma,
                        0.25,
                    )
                )
                if score <= tolerance and score < best_score:
                    best_score = score
                    best_match = (atom_j, slot_j)
            return best_match

        def best_new_atom_action(
            atom_i: int,
            slot_i: int,
            open_slot_lookup: dict[tuple[int, int], list[tuple[int, int]]],
        ) -> dict[str, Any] | None:
            species_i = int(species_list[atom_i])
            motif_i = int(motif_ids_list[atom_i])
            neighbor_species = int(shell_target.motif_neighbor_species[motif_i][slot_i])
            if remaining_counts[neighbor_species] <= 0:
                return None

            source_position = np.asarray(positions_list[atom_i], dtype=np.float64)
            source_vector = slot_vector(atom_i, slot_i)
            bond_axis = -source_vector
            bond_axis_norm = float(np.linalg.norm(bond_axis))
            if bond_axis_norm <= _EPS:
                return None

            motif_candidates = np.asarray(
                motif_candidates_by_species.get(neighbor_species, np.empty(0, dtype=np.intp)),
                dtype=np.intp,
            )
            if motif_candidates.size == 0:
                return None
            self.rng.shuffle(motif_candidates)

            best_action: dict[str, Any] | None = None
            for motif_j in motif_candidates:
                reciprocal_slots = np.flatnonzero(
                    shell_target.motif_neighbor_species[int(motif_j)] == species_i
                )
                if reciprocal_slots.size == 0:
                    continue
                for slot_j in reciprocal_slots:
                    base_rotation = self._rotation_align_vector(
                        shell_target.motif_neighbor_vectors[int(motif_j)][int(slot_j)],
                        bond_axis,
                        random_spin=False,
                    )
                    spin_offset = float(self.rng.uniform(0.0, 2.0 * np.pi))
                    for spin_index in range(spin_num_angles):
                        spin_angle = spin_offset + (2.0 * np.pi * spin_index) / spin_num_angles
                        rotation_spin = self._axis_angle_rotation_matrix(bond_axis, spin_angle)
                        rotation_j = rotation_spin @ base_rotation
                        vec_back = self._apply_rotation(
                            shell_target.motif_neighbor_vectors[int(motif_j)][int(slot_j)],
                            rotation_j,
                        )
                        displacement = 0.5 * (source_vector - vec_back)
                        position_j = self._wrap_position(source_position + displacement)
                        overlap_value, max_overlap = overlap_penalty(
                            neighbor_species,
                            position_j,
                            ignore_atoms={atom_i},
                        )
                        if max_overlap > float(
                            0.45 * shell_target.pair_peak[species_i, neighbor_species]
                        ):
                            continue

                        candidate_matches: list[tuple[float, int, int, int]] = []
                        motif_neighbor_species = shell_target.motif_neighbor_species[int(motif_j)]
                        for new_slot, target_species in enumerate(motif_neighbor_species):
                            if int(new_slot) == int(slot_j):
                                continue
                            vec_new = self._apply_rotation(
                                shell_target.motif_neighbor_vectors[int(motif_j)][int(new_slot)],
                                rotation_j,
                            )
                            lookup_key = (int(target_species), neighbor_species)
                            for atom_k, slot_k in open_slot_lookup.get(lookup_key, []):
                                if atom_k == atom_i:
                                    continue
                                if atom_k >= len(species_list):
                                    continue
                                if used_slots_list[atom_k][slot_k]:
                                    continue
                                vec_k = slot_vector(int(atom_k), int(slot_k))
                                actual = self._minimum_image_vectors(
                                    position_j,
                                    np.asarray(positions_list[int(atom_k)], dtype=np.float64)[None, :],
                                )[0]
                                desired = 0.5 * (vec_new - vec_k)
                                mismatch = vec_new + vec_k
                                score = float(
                                    np.linalg.norm(actual - desired)
                                    + 0.25 * np.linalg.norm(mismatch)
                                )
                                target_length = 0.5 * (
                                    float(np.linalg.norm(vec_new)) + float(np.linalg.norm(vec_k))
                                )
                                pair_sigma = float(
                                    shell_target.pair_sigma[neighbor_species, int(target_species)]
                                )
                                tolerance = float(
                                    max(
                                        merge_distance_scale * target_length,
                                        2.0 * pair_sigma,
                                        0.25,
                                    )
                                )
                                if score <= tolerance:
                                    candidate_matches.append(
                                        (score, int(new_slot), int(atom_k), int(slot_k))
                                    )

                        candidate_matches.sort(key=lambda item: item[0])
                        chosen_matches: list[tuple[int, int, int]] = []
                        used_new_slots: set[int] = {int(slot_j)}
                        used_existing_atoms: set[int] = set()
                        total_match_error = 0.0
                        for score, new_slot, atom_k, slot_k in candidate_matches:
                            if new_slot in used_new_slots or atom_k in used_existing_atoms:
                                continue
                            used_new_slots.add(new_slot)
                            used_existing_atoms.add(atom_k)
                            chosen_matches.append((new_slot, atom_k, slot_k))
                            total_match_error += float(score)

                        action = {
                            "source_atom": int(atom_i),
                            "source_slot": int(slot_i),
                            "new_species": int(neighbor_species),
                            "new_motif": int(motif_j),
                            "new_slot": int(slot_j),
                            "new_position": np.asarray(position_j, dtype=np.float64),
                            "new_rotation": np.asarray(rotation_j, dtype=np.float64),
                            "extra_matches": chosen_matches,
                            "overlap_penalty": float(overlap_value),
                            "max_overlap": float(max_overlap),
                            "match_error": float(total_match_error),
                            "match_count": int(len(chosen_matches)),
                        }
                        if best_action is None:
                            best_action = action
                            continue
                        best_key = (
                            int(best_action["match_count"]),
                            -float(best_action["overlap_penalty"]),
                            -float(best_action["match_error"]),
                        )
                        action_key = (
                            int(action["match_count"]),
                            -float(action["overlap_penalty"]),
                            -float(action["match_error"]),
                        )
                        if action_key > best_key:
                            best_action = action
            return best_action

        if not add_seed_atom():
            raise RuntimeError("Motif growth failed to seed the first atom.")
        expand_queue: list[int] = [0]
        while (len(positions_list) < target_num_atoms) or any(not flag for flag in expanded_list):
            if not expand_queue:
                pending = [index for index, flag in enumerate(expanded_list) if not flag]
                if pending:
                    expand_queue.extend([int(value) for value in self.rng.permutation(np.asarray(pending, dtype=np.intp))])
                elif len(positions_list) < target_num_atoms and add_seed_atom():
                    expand_queue.append(len(positions_list) - 1)
                else:
                    break

            center_atom = int(expand_queue.pop(0))
            if expanded_list[center_atom]:
                continue

            center_progress = False
            free_slots = np.flatnonzero(~used_slots_list[center_atom])
            for slot_i in free_slots:
                slot_i = int(slot_i)
                if used_slots_list[center_atom][slot_i]:
                    continue
                open_slot_lookup = build_open_slot_lookup()
                existing_match = best_existing_match(center_atom, slot_i, open_slot_lookup)
                if existing_match is not None:
                    add_edge(center_atom, slot_i, existing_match[0], existing_match[1])
                    center_progress = True
                    continue

                action: dict[str, Any] | None = None
                if len(positions_list) < target_num_atoms:
                    action = best_new_atom_action(center_atom, slot_i, open_slot_lookup)

                if action is None and len(positions_list) < target_num_atoms:
                    species_i = int(species_list[center_atom])
                    motif_i = int(motif_ids_list[center_atom])
                    neighbor_species = int(shell_target.motif_neighbor_species[motif_i][slot_i])
                    if remaining_counts[neighbor_species] > 0:
                        motif_j = choose_motif(neighbor_species, required_neighbor_species=species_i)
                        reciprocal_slots = np.flatnonzero(
                            shell_target.motif_neighbor_species[int(motif_j)] == species_i
                        )
                        if reciprocal_slots.size:
                            reciprocal_slot = int(reciprocal_slots[0])
                            source_vector = slot_vector(center_atom, slot_i)
                            rotation_j = self._rotation_align_vector(
                                shell_target.motif_neighbor_vectors[int(motif_j)][reciprocal_slot],
                                -source_vector,
                                random_spin=True,
                            )
                            vec_back = self._apply_rotation(
                                shell_target.motif_neighbor_vectors[int(motif_j)][reciprocal_slot],
                                rotation_j,
                            )
                            position_j = self._wrap_position(
                                np.asarray(positions_list[center_atom], dtype=np.float64)
                                + 0.5 * (source_vector - vec_back)
                            )
                            overlap_value, max_overlap = overlap_penalty(
                                neighbor_species,
                                position_j,
                                ignore_atoms={center_atom},
                            )
                            if max_overlap <= float(
                                0.60 * shell_target.pair_peak[species_i, neighbor_species]
                            ):
                                action = {
                                    "source_atom": center_atom,
                                    "source_slot": slot_i,
                                    "new_species": neighbor_species,
                                    "new_motif": int(motif_j),
                                    "new_slot": reciprocal_slot,
                                    "new_position": np.asarray(position_j, dtype=np.float64),
                                    "new_rotation": np.asarray(rotation_j, dtype=np.float64),
                                    "extra_matches": [],
                                    "overlap_penalty": float(overlap_value),
                                    "max_overlap": float(max_overlap),
                                    "match_error": 0.0,
                                    "match_count": 0,
                                }

                if action is None:
                    continue

                new_atom = add_atom(
                    int(action["new_species"]),
                    int(action["new_motif"]),
                    np.asarray(action["new_position"], dtype=np.float64),
                    np.asarray(action["new_rotation"], dtype=np.float64),
                )
                expand_queue.append(int(new_atom))
                add_edge(center_atom, slot_i, new_atom, int(action["new_slot"]))
                for new_slot, atom_k, slot_k in action["extra_matches"]:
                    add_edge(new_atom, int(new_slot), int(atom_k), int(slot_k))
                center_progress = True
                if edges and (len(positions_list) % rotation_refresh_interval == 0):
                    refresh_rotations()

            expanded_list[center_atom] = True
            if (not center_progress) and len(positions_list) < target_num_atoms and not expand_queue:
                if add_seed_atom():
                    expand_queue.append(len(positions_list) - 1)

        if build_progress is not None:
            build_progress.close()

        if not positions_list:
            raise RuntimeError("Motif growth failed to place any atoms.")

        positions = np.asarray(positions_list, dtype=np.float64)
        species_array = np.asarray(species_list, dtype=np.intp)
        motif_ids = np.asarray(motif_ids_list, dtype=np.intp)
        numbers = self._species[species_array]
        graph = self._build_motif_graph_data(
            shell_target,
            motif_ids,
            edges,
            num_atoms=len(positions),
            rotations=np.asarray(rotations_list, dtype=np.float64),
        )
        return positions, numbers.astype(np.int64, copy=False), motif_ids, graph

    def _initial_motif_graph_positions(
        self,
        shell_target: CoordinationShellTarget,
        motif_ids: np.ndarray,
        graph: dict[str, Any],
    ) -> np.ndarray:
        """Embed the random motif graph into 3D using propagated local motif directions."""
        num_atoms = len(self.atoms)
        positions = np.zeros((num_atoms, 3), dtype=np.float64)
        rotations = np.zeros((num_atoms, 3, 3), dtype=np.float64)
        placed = np.zeros(num_atoms, dtype=bool)
        min_root_separation = float(
            max(
                0.6 * np.median(graph["target_length"]) if graph["target_length"].size else 0.0,
                np.max(shell_target.pair_hard_min[shell_target.pair_mask]) if np.any(shell_target.pair_mask) else 0.0,
            )
        )

        for root_atom in range(num_atoms):
            if placed[root_atom]:
                continue
            candidate_position = self._random_cell_position()
            if np.any(placed) and min_root_separation > 0.0:
                for _ in range(32):
                    vec = self._minimum_image_vectors(candidate_position, positions[placed])
                    if vec.size == 0 or np.all(np.linalg.norm(vec, axis=1) >= min_root_separation):
                        break
                    candidate_position = self._random_cell_position()
            positions[root_atom] = candidate_position
            rotations[root_atom] = self._random_rotation_matrix()
            placed[root_atom] = True
            queue = [int(root_atom)]
            while queue:
                center_atom = int(queue.pop(0))
                motif_vectors = shell_target.motif_neighbor_vectors[int(motif_ids[center_atom])]
                for neighbor_atom, local_slot, neighbor_slot, _ in graph["adjacency"][center_atom]:
                    if placed[neighbor_atom]:
                        continue
                    bond_vector = self._apply_rotation(motif_vectors[int(local_slot)], rotations[center_atom])
                    positions[neighbor_atom] = self._wrap_position(positions[center_atom] + bond_vector)
                    neighbor_vectors = shell_target.motif_neighbor_vectors[int(motif_ids[neighbor_atom])]
                    if neighbor_vectors.shape[0] > int(neighbor_slot):
                        rotations[neighbor_atom] = self._rotation_align_vector(
                            neighbor_vectors[int(neighbor_slot)],
                            -bond_vector,
                            random_spin=True,
                        )
                    else:
                        rotations[neighbor_atom] = self._random_rotation_matrix()
                    placed[neighbor_atom] = True
                    queue.append(int(neighbor_atom))

        if num_atoms:
            jitter = 0.08 * np.median(graph["target_length"]) if graph["target_length"].size else 0.0
            if jitter > 0.0:
                positions = self._wrap_positions(
                    positions + self.rng.normal(scale=jitter, size=positions.shape)
                )
        return positions

    def _motif_vote_relax_step(
        self,
        positions: np.ndarray,
        shell_target: CoordinationShellTarget,
        motif_ids: np.ndarray,
        graph: dict[str, Any],
        *,
        step_size: float,
        spring_step: float,
        inertia: float,
        repulsion_step: float,
        max_repulsion_distance: float,
    ) -> tuple[np.ndarray, float, float]:
        """Perform one motif-voting relaxation step."""
        num_atoms = positions.shape[0]
        prediction_sum = inertia * positions
        prediction_weight = np.full(num_atoms, inertia, dtype=np.float64)
        bond_loss = 0.0

        for atom_index in range(num_atoms):
            adjacency = graph["adjacency"][atom_index]
            if not adjacency:
                continue
            motif_vectors = shell_target.motif_neighbor_vectors[int(motif_ids[atom_index])]
            slot_indices = np.array([entry[1] for entry in adjacency], dtype=np.intp)
            neighbor_indices = np.array([entry[0] for entry in adjacency], dtype=np.intp)
            template_vectors = motif_vectors[slot_indices]
            current_vectors = self._minimum_image_vectors(positions[atom_index], positions[neighbor_indices])
            rotation = self._fit_rotation_to_vectors(template_vectors, current_vectors)
            predicted_vectors = self._apply_rotation(template_vectors, rotation)
            diff = current_vectors - predicted_vectors
            bond_loss += float(np.sum(diff * diff))

            for local_index, neighbor_atom in enumerate(neighbor_indices):
                predicted_position = self._wrap_position(positions[atom_index] + predicted_vectors[local_index])
                delta = self._minimum_image_vectors(positions[neighbor_atom], predicted_position[None, :])[0]
                prediction_sum[neighbor_atom] += positions[neighbor_atom] + delta
                prediction_weight[neighbor_atom] += 1.0

        new_positions = positions + step_size * (
            prediction_sum / np.maximum(prediction_weight[:, None], _EPS) - positions
        )
        if spring_step > 0.0 and graph["edge_i"].size:
            spring_delta = np.zeros_like(new_positions)
            edge_count = graph["edge_i"].size
            for edge_index in range(edge_count):
                atom_i = int(graph["edge_i"][edge_index])
                atom_j = int(graph["edge_j"][edge_index])
                target_length = float(graph["target_length"][edge_index])
                rij = self._minimum_image_vectors(new_positions[atom_i], new_positions[atom_j][None, :])[0]
                distance = float(np.linalg.norm(rij))
                if distance <= _EPS:
                    continue
                correction = 0.5 * (target_length - distance) * rij / distance
                spring_delta[atom_i] -= correction
                spring_delta[atom_j] += correction
                bond_loss += (distance - target_length) ** 2
            degree_scale = np.maximum(graph["actual_degree"].astype(np.float64), 1.0)[:, None]
            new_positions = new_positions + spring_step * spring_delta / degree_scale
        new_positions = self._wrap_positions(new_positions)

        self.atoms.positions[:] = new_positions
        self._rebuild_spatial_index()
        repulsion_delta = np.zeros_like(new_positions)
        overlap_loss = 0.0
        if max_repulsion_distance > 0.0 and repulsion_step > 0.0:
            for atom_index in range(num_atoms):
                neighbor_indices, vectors, radius_sq = self._query_local_environment(
                    atom_index,
                    new_positions[atom_index],
                    max_repulsion_distance,
                )
                if neighbor_indices.size == 0:
                    continue
                radius = np.sqrt(np.maximum(radius_sq, _EPS))
                nonbond_mask = np.array(
                    [neighbor not in graph["bonded_sets"][atom_index] for neighbor in neighbor_indices],
                    dtype=bool,
                )
                if not np.any(nonbond_mask):
                    continue
                neighbor_indices = neighbor_indices[nonbond_mask]
                vectors = vectors[nonbond_mask]
                radius = radius[nonbond_mask]
                species_i = int(self._atom_species_index[atom_index])
                species_j = self._atom_species_index[neighbor_indices]
                hard_min = shell_target.pair_hard_min[species_i, species_j]
                overlap = hard_min - radius
                keep = overlap > 0.0
                if not np.any(keep):
                    continue
                unit = -vectors[keep] / radius[keep][:, None]
                repulsion_delta[atom_index] += np.sum(unit * overlap[keep][:, None], axis=0)
                overlap_loss += float(np.sum(overlap[keep] * overlap[keep]))
            if np.any(repulsion_delta):
                new_positions = self._wrap_positions(new_positions + repulsion_step * repulsion_delta)
                self.atoms.positions[:] = new_positions
                self._rebuild_spatial_index()

        return new_positions, bond_loss, overlap_loss

    def _build_dynamic_motif_matches(
        self,
        positions: np.ndarray,
        rotations: np.ndarray,
        shell_target: CoordinationShellTarget,
        motif_ids: np.ndarray,
        *,
        match_candidates_per_slot: int,
    ) -> dict[str, Any]:
        """Greedily match predicted motif slots to actual atoms with capacity limits."""
        num_atoms = positions.shape[0]
        atom_indices_by_species = [
            np.flatnonzero(self._atom_species_index == species_index).astype(np.intp, copy=False)
            for species_index in range(self._num_species)
        ]
        capacities = np.array(
            [len(shell_target.motif_neighbor_species[int(motif_id)]) for motif_id in motif_ids],
            dtype=np.intp,
        )
        remaining = capacities.astype(np.intp, copy=True)
        candidate_edges: list[tuple[float, int, int, int, np.ndarray]] = []

        for atom_index in range(num_atoms):
            motif_vectors = shell_target.motif_neighbor_vectors[int(motif_ids[atom_index])]
            motif_species = shell_target.motif_neighbor_species[int(motif_ids[atom_index])]
            if motif_vectors.shape[0] == 0:
                continue
            predicted_vectors = self._apply_rotation(motif_vectors, rotations[atom_index])
            for slot_index, neighbor_species in enumerate(motif_species):
                target_atoms = atom_indices_by_species[int(neighbor_species)]
                if target_atoms.size == 0:
                    continue
                predicted_position = positions[atom_index] + predicted_vectors[slot_index]
                deltas = self._minimum_image_vectors(predicted_position, positions[target_atoms])
                distance_sq = np.einsum("ij,ij->i", deltas, deltas)
                self_mask = target_atoms == int(atom_index)
                if np.any(self_mask):
                    distance_sq[self_mask] = np.inf
                order = np.argsort(distance_sq, kind="stable")[: int(max(match_candidates_per_slot, 1))]
                for candidate_index in order:
                    if not np.isfinite(distance_sq[candidate_index]):
                        continue
                    candidate_edges.append(
                        (
                            float(distance_sq[candidate_index]),
                            int(atom_index),
                            int(slot_index),
                            int(target_atoms[candidate_index]),
                            np.asarray(predicted_position, dtype=np.float64),
                        )
                    )

        candidate_edges.sort(key=lambda item: item[0])
        outgoing: list[list[tuple[int, int, float]]] = [[] for _ in range(num_atoms)]
        incoming: list[list[tuple[int, int, np.ndarray, float]]] = [[] for _ in range(num_atoms)]
        bonded_sets: list[set[int]] = [set() for _ in range(num_atoms)]
        used_sources: set[tuple[int, int]] = set()
        used_pairs: set[tuple[int, int]] = set()
        matched_slots = 0
        matched_distance_sq = 0.0

        for distance_sq, atom_i, slot_i, atom_j, predicted_position in candidate_edges:
            source_key = (int(atom_i), int(slot_i))
            pair_key = (min(int(atom_i), int(atom_j)), max(int(atom_i), int(atom_j)))
            if source_key in used_sources or pair_key in used_pairs or remaining[atom_j] <= 0:
                continue
            used_sources.add(source_key)
            used_pairs.add(pair_key)
            remaining[atom_j] -= 1
            distance = float(np.sqrt(max(distance_sq, 0.0)))
            outgoing[atom_i].append((int(atom_j), int(slot_i), distance))
            incoming[atom_j].append(
                (
                    int(atom_i),
                    int(slot_i),
                    np.asarray(predicted_position, dtype=np.float64),
                    distance,
                )
            )
            bonded_sets[atom_i].add(atom_j)
            bonded_sets[atom_j].add(atom_i)
            matched_slots += 1
            matched_distance_sq += float(distance_sq)

        return {
            "outgoing": outgoing,
            "incoming": incoming,
            "bonded_sets": bonded_sets,
            "capacities": capacities,
            "remaining": remaining,
            "matched_slots": int(matched_slots),
            "unmatched_slots": int(np.sum(capacities) - matched_slots),
            "matched_distance_sq": float(matched_distance_sq),
        }

    def _build_reciprocal_projection_matches(
        self,
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
        self,
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
        self,
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

    def _motif_match_relax_step(
        self,
        positions: np.ndarray,
        rotations: np.ndarray,
        shell_target: CoordinationShellTarget,
        motif_ids: np.ndarray,
        matches: dict[str, Any],
        *,
        step_size: float,
        repulsion_step: float,
        inertia: float,
        max_repulsion_distance: float,
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        """Relax positions and local motif frames using dynamic fuzzy motif matches."""
        num_atoms = positions.shape[0]
        prediction_sum = inertia * positions
        prediction_weight = np.full(num_atoms, inertia, dtype=np.float64)

        for atom_index in range(num_atoms):
            for _, _, predicted_position, distance in matches["incoming"][atom_index]:
                weight = 1.0 / max(1.0 + distance * distance, 1e-6)
                delta = self._minimum_image_vectors(
                    positions[atom_index],
                    np.asarray(predicted_position, dtype=np.float64)[None, :],
                )[0]
                prediction_sum[atom_index] += weight * (positions[atom_index] + delta)
                prediction_weight[atom_index] += weight

        new_positions = positions + step_size * (
            prediction_sum / np.maximum(prediction_weight[:, None], _EPS) - positions
        )
        new_positions = self._wrap_positions(new_positions)

        self.atoms.positions[:] = new_positions
        self._rebuild_spatial_index()
        overlap_loss = 0.0
        if max_repulsion_distance > 0.0 and repulsion_step > 0.0:
            repulsion_delta = np.zeros_like(new_positions)
            for atom_index in range(num_atoms):
                neighbor_indices, vectors, radius_sq = self._query_local_environment(
                    atom_index,
                    new_positions[atom_index],
                    max_repulsion_distance,
                )
                if neighbor_indices.size == 0:
                    continue
                radius = np.sqrt(np.maximum(radius_sq, _EPS))
                nonbond_mask = np.array(
                    [neighbor not in matches["bonded_sets"][atom_index] for neighbor in neighbor_indices],
                    dtype=bool,
                )
                if not np.any(nonbond_mask):
                    continue
                neighbor_indices = neighbor_indices[nonbond_mask]
                vectors = vectors[nonbond_mask]
                radius = radius[nonbond_mask]
                species_i = int(self._atom_species_index[atom_index])
                species_j = self._atom_species_index[neighbor_indices]
                hard_min = shell_target.pair_hard_min[species_i, species_j]
                overlap = hard_min - radius
                keep = overlap > 0.0
                if not np.any(keep):
                    continue
                unit = -vectors[keep] / radius[keep][:, None]
                repulsion_delta[atom_index] += np.sum(unit * overlap[keep][:, None], axis=0)
                overlap_loss += float(np.sum(overlap[keep] * overlap[keep]))
            if np.any(repulsion_delta):
                new_positions = self._wrap_positions(new_positions + repulsion_step * repulsion_delta)
                self.atoms.positions[:] = new_positions
                self._rebuild_spatial_index()

        new_rotations = np.array(rotations, copy=True)
        match_loss = 0.0
        for atom_index in range(num_atoms):
            outgoing = matches["outgoing"][atom_index]
            if not outgoing:
                continue
            motif_vectors = shell_target.motif_neighbor_vectors[int(motif_ids[atom_index])]
            slot_indices = np.array([entry[1] for entry in outgoing], dtype=np.intp)
            neighbor_indices = np.array([entry[0] for entry in outgoing], dtype=np.intp)
            template_vectors = motif_vectors[slot_indices]
            current_vectors = self._minimum_image_vectors(new_positions[atom_index], new_positions[neighbor_indices])
            new_rotations[atom_index] = self._fit_rotation_to_vectors(template_vectors, current_vectors)
            predicted_vectors = self._apply_rotation(template_vectors, new_rotations[atom_index])
            diff = current_vectors - predicted_vectors
            match_loss += float(np.sum(diff * diff))

        return new_positions, new_rotations, match_loss, overlap_loss

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

    def generate_teacher_rollout(
        self,
        *,
        repulsion_steps: int = 10,
        repulsion_step_size: float | None = None,
        repulsion_cutoff: float | None = None,
        mc_steps: int = 1_000,
        temperature: float = 0.0,
        jump_size: float | None = None,
        r_min_nn: float | None = None,
        attempt_prob: float = 1.0,
        snapshot_stride_accepted: int = 40,
        target_id: str | None = None,
        output_path: str | Path | None = None,
        output_format: str = "npz",
        show_progress: bool = True,
    ) -> dict[str, Any] | Path:
        """Generate a coarse teacher trajectory for model training.

        This method keeps the exact repulsion and Monte Carlo logic intact, but
        records sparse snapshots of the trajectory so a later model can learn
        coordinate updates that move structures toward the target `g3`.
        """
        snapshot_stride_accepted = int(snapshot_stride_accepted)
        if snapshot_stride_accepted <= 0:
            raise ValueError("snapshot_stride_accepted must be positive.")

        target_id = target_id or self.target_distribution.label
        snapshots: list[dict[str, Any]] = []

        self.sync_g3(show_progress=show_progress)
        snapshots.append(
            self._capture_teacher_snapshot(
                stage_code=0,
                step=0,
                accepted_moves=0,
                attempted_moves=0,
            )
        )

        repulsion_summary: dict[str, Any] | None = None
        if int(repulsion_steps) > 0:
            repulsion_summary = self.repulsion(
                num_steps=int(repulsion_steps),
                step_size=repulsion_step_size,
                cutoff=repulsion_cutoff,
                sync_g3=False,
                show_progress=show_progress,
            )
            self.sync_g3(show_progress=show_progress)
            snapshots.append(
                self._capture_teacher_snapshot(
                    stage_code=1,
                    step=int(repulsion_steps),
                    accepted_moves=0,
                    attempted_moves=0,
                )
            )

        def capture_mc_snapshot(meta: dict[str, Any]) -> None:
            snapshots.append(
                self._capture_teacher_snapshot(
                    stage_code=2,
                    step=int(meta["step"]),
                    accepted_moves=int(meta["accepted_moves"]),
                    attempted_moves=int(meta["attempted_moves"]),
                )
            )

        monte_carlo_summary = self.monte_carlo(
            num_steps=int(mc_steps),
            temperature=temperature,
            jump_size=jump_size,
            r_min_nn=r_min_nn,
            attempt_prob=attempt_prob,
            plot_history=False,
            show_progress=show_progress,
            _snapshot_stride_accepted=snapshot_stride_accepted,
            _snapshot_callback=capture_mc_snapshot,
        )
        snapshots.append(
            self._capture_teacher_snapshot(
                stage_code=3,
                step=int(mc_steps),
                accepted_moves=int(monte_carlo_summary["accepted_moves"]),
                attempted_moves=int(monte_carlo_summary["attempted_moves"]),
            )
        )

        rollout = self._assemble_teacher_rollout(
            snapshots=snapshots,
            target_id=target_id,
            repulsion_summary=repulsion_summary,
            monte_carlo_summary=monte_carlo_summary,
            snapshot_stride_accepted=snapshot_stride_accepted,
            r_min_nn=r_min_nn,
        )
        if output_path is None:
            return rollout
        return self._save_teacher_rollout(rollout, output_path, output_format=output_format)

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

    def plot_monte_carlo(
        self,
        *,
        log_y: bool = False,
        show_run_boundaries: bool = True,
    ):
        """Plot the recorded Monte Carlo cost history using Matplotlib."""
        if self.mc_history is None:
            raise ValueError("Run monte_carlo() before plotting the history.")

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.8, 4.2))
        ax.plot(self.mc_history["step"], self.mc_history["cost"], lw=1.8, label="cost")
        ax.plot(self.mc_history["step"], self.mc_history["best_cost"], lw=1.4, label="best")
        if show_run_boundaries and "run_index" in self.mc_history:
            run_index = np.asarray(self.mc_history["run_index"], dtype=np.int32)
            step = np.asarray(self.mc_history["step"], dtype=np.int32)
            change_points = np.flatnonzero(np.diff(run_index) > 0)
            for point_index in change_points:
                ax.axvline(
                    float(step[point_index + 1]),
                    color="0.65",
                    lw=0.9,
                    ls="--",
                    alpha=0.7,
                )
        if log_y:
            positive_cost = np.asarray(self.mc_history["cost"], dtype=np.float64)
            positive_cost = positive_cost[positive_cost > 0.0]
            if positive_cost.size:
                ax.set_yscale("log")
        ax.set_xlabel("step")
        ax.set_ylabel("cost")
        ax.set_title("Monte Carlo cost history")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        return fig, ax

    def plot_shell_refine(
        self,
        *,
        log_y: bool = False,
    ):
        """Plot the recorded shell-refinement loss history using Matplotlib."""
        if self.shell_history is None:
            raise ValueError("Run shell_refine() before plotting the history.")

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.8, 4.2))
        ax.plot(self.shell_history["step"], self.shell_history["loss"], lw=1.8, label="loss")
        ax.plot(self.shell_history["step"], self.shell_history["best_loss"], lw=1.4, label="best")
        if log_y:
            positive_loss = np.asarray(self.shell_history["loss"], dtype=np.float64)
            positive_loss = positive_loss[positive_loss > 0.0]
            if positive_loss.size:
                ax.set_yscale("log")
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.set_title("Shell refinement history")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        return fig, ax

    # ------------------------------------------------------------------
    # shell_relax: vectorized spring-network relaxation
    # ------------------------------------------------------------------

    def shell_relax(
        self,
        shell_target: CoordinationShellTarget,
        num_steps: int = 200,
        *,
        bond_weight: float = 1.0,
        angle_weight: float = 0.5,
        repulsion_weight: float = 2.0,
        step_size: float = 0.1,
        step_decay: float = 0.995,
        neighbor_update_interval: int = 10,
        neighbor_cutoff_scale: float = 1.5,
        max_force_clip: float = 2.0,
        show_progress: bool = True,
    ) -> dict[str, Any]:
        """Relax random positions to match first-shell targets using spring forces.

        Moves **all atoms simultaneously** each step via three vectorized force
        terms: bond springs toward the target nearest-neighbor distance, angle
        springs toward the target bond angle, and soft repulsion to eliminate
        overlaps and close-packed background.  Bond topology (K-nearest
        assignment) is rebuilt periodically using ASE's ``neighbor_list``.

        Parameters
        ----------
        shell_target
            First-shell coordination targets extracted from the reference
            crystal via :meth:`CoordinationShellTarget.from_atoms`.
        num_steps
            Number of relaxation sweeps.
        bond_weight
            Strength of the harmonic spring pulling bonded neighbors toward
            ``pair_peak`` distance.
        angle_weight
            Strength of the angular spring pushing bond angles toward
            ``angle_mode_deg``.
        repulsion_weight
            Strength of the short-range repulsive force below ``pair_hard_min``.
        step_size
            Initial maximum displacement per step (Angstrom).
        step_decay
            Multiplicative decay applied to *step_size* each iteration.
        neighbor_update_interval
            Rebuild the bond topology every this many steps.
        neighbor_cutoff_scale
            Neighbor search cutoff as a multiple of ``max_pair_outer``.
        max_force_clip
            Per-atom force magnitude is clipped to this value before
            integration to keep the dynamics stable.
        show_progress
            Display a text progress bar.

        Returns
        -------
        dict[str, Any]
            Summary with parameters and final/initial loss values.
        """
        num_atoms = len(self.atoms)
        species_idx = self._atom_species_index  # (num_atoms,) int
        cell_inv = self._cell_inverse
        cell_mat = self._cell_matrix

        # --- extract targets from shell_target ---
        coord_target = np.asarray(shell_target.coordination_target, dtype=np.float64)
        pair_peak = np.asarray(shell_target.pair_peak, dtype=np.float64)
        pair_hard_min = np.asarray(shell_target.pair_hard_min, dtype=np.float64)
        angle_mode_rad = np.deg2rad(
            np.asarray(shell_target.angle_mode_deg, dtype=np.float64)
        )
        angle_lookup = np.asarray(shell_target.angle_lookup, dtype=np.intp)
        cutoff = float(shell_target.max_pair_outer * neighbor_cutoff_scale)

        # K nearest neighbors per atom (total across all neighbor species)
        k_per_species = np.zeros(shell_target.species.size, dtype=np.intp)
        for s in range(shell_target.species.size):
            k_per_species[s] = int(np.round(coord_target[s].sum()))

        # Repulsion radii: hard core (overlap prevention) and non-bonded
        # shell clearance (eliminates close-packed background).
        pair_outer = np.asarray(shell_target.pair_outer, dtype=np.float64)
        hard_core = pair_hard_min.copy()
        mask_zero = hard_core < _EPS
        hard_core[mask_zero] = 0.4 * pair_peak[mask_zero]
        global_floor = float(np.min(pair_peak[pair_peak > _EPS])) * 0.35 if np.any(pair_peak > _EPS) else 1.0
        hard_core[hard_core < _EPS] = global_floor
        # Non-bonded atoms are pushed beyond this radius to create a
        # gap between the first and second coordination shells.  The
        # scale is modest (1.25x peak) to avoid fighting the density
        # constraint — even a small gap eliminates the close-packed
        # angular background in g3.
        nonbond_push = pair_peak * 1.25
        nonbond_push[nonbond_push < _EPS] = float(np.max(pair_peak)) * 1.25

        # --- grain-aware force scaling ---
        # When _grain_ids is set, interior atoms are frozen to preserve
        # crystalline order; boundary atoms get full relaxation forces.
        grain_ids = self._grain_ids
        grain_seeds = self._grain_seeds
        if (
            grain_ids is not None
            and grain_seeds is not None
            and len(grain_ids) == num_atoms
        ):
            is_boundary = np.ones(num_atoms, dtype=bool)  # start all boundary
            _grain_boundary_detected = [False]

            def _detect_boundary_atoms() -> None:
                """Mark atoms as boundary using distance to grain boundary.

                For each atom, boundary_depth = half the gap between the
                distance to the nearest foreign seed and the distance to
                its own seed.  Atoms deep inside a grain (boundary_depth
                > threshold) are interior; the rest are boundary.
                """
                if _grain_boundary_detected[0]:
                    return
                _grain_boundary_detected[0] = True

                pos = self.atoms.positions
                n_seeds = len(grain_seeds)
                boundary_width = float(np.max(pair_peak)) * 0.5

                # PBC distances from each atom to every seed
                delta = pos[:, None, :] - grain_seeds[None, :, :]  # (N, n_seeds, 3)
                frac_d = delta @ cell_inv
                frac_d -= np.rint(frac_d)
                cart_d = frac_d @ cell_mat
                dist_to_seeds = np.sqrt(np.sum(cart_d ** 2, axis=2))  # (N, n_seeds)

                is_boundary[:] = True  # default boundary
                for ia in range(num_atoms):
                    gid = grain_ids[ia]
                    if gid < 0:
                        continue  # amorphous fill → always boundary
                    dist_own = dist_to_seeds[ia, gid]
                    # Distance to nearest OTHER seed
                    dists_copy = dist_to_seeds[ia].copy()
                    dists_copy[gid] = np.inf
                    dist_other = np.min(dists_copy)
                    boundary_depth = (dist_other - dist_own) * 0.5
                    if boundary_depth > boundary_width:
                        is_boundary[ia] = False

            # Interior atoms: forces zeroed completely (frozen)
            interior_force_scale = 0.0
        else:
            grain_ids = None
            is_boundary = None
            _grain_boundary_detected = None

            def _detect_boundary_atoms() -> None:
                pass

            interior_force_scale = 1.0

        # --- vectorized minimum-image helper for paired arrays ---
        def min_image(delta: np.ndarray) -> np.ndarray:
            frac = delta @ cell_inv
            frac -= np.rint(frac)
            return frac @ cell_mat

        # --- neighbor rebuild ---
        bond_i = np.empty(0, dtype=np.intp)
        bond_j = np.empty(0, dtype=np.intp)
        bond_r_target = np.empty(0, dtype=np.float64)
        tri_center = np.empty(0, dtype=np.intp)
        tri_a = np.empty(0, dtype=np.intp)
        tri_b = np.empty(0, dtype=np.intp)
        tri_phi_target = np.empty(0, dtype=np.float64)
        bonded_set: set[tuple[int, int]] = set()

        def rebuild_topology() -> None:
            nonlocal bond_i, bond_j, bond_r_target
            nonlocal tri_center, tri_a, tri_b, tri_phi_target
            nonlocal bonded_set

            nl_i, nl_j, nl_d = neighbor_list("ijd", self.atoms, cutoff)

            # Symmetric bond matching with angular awareness: greedily
            # build a K-regular bond graph.  Candidates are sorted by
            # distance; each candidate is accepted only if the new bond
            # makes angles ≥ min_accept_angle with all existing bonds at
            # BOTH endpoints.  This eliminates close-packed clusters.
            bond_count = np.zeros(num_atoms, dtype=np.intp)
            k_atom = np.array(
                [int(k_per_species[species_idx[a]]) for a in range(num_atoms)],
                dtype=np.intp,
            )

            # Pre-compute displacement vectors for all neighbor pairs
            nl_vecs = min_image(
                self.atoms.positions[nl_j] - self.atoms.positions[nl_i]
            )
            nl_hats = nl_vecs / np.maximum(nl_d, _EPS)[:, None]

            # Sort candidates by distance (nearest first)
            dist_order = np.argsort(nl_d)

            _bond_i_list: list[int] = []
            _bond_j_list: list[int] = []
            _bond_rt_list: list[float] = []
            bonded_set = set()
            bonded_neighbors: list[list[int]] = [[] for _ in range(num_atoms)]
            # Store unit vectors of existing bonds per atom for angle check
            bond_hats_per_atom: list[list[np.ndarray]] = [[] for _ in range(num_atoms)]

            min_accept_angle = np.deg2rad(60.0)  # reject bonds with < 60° to existing

            for idx in dist_order:
                ai = int(nl_i[idx])
                aj = int(nl_j[idx])
                if bond_count[ai] >= k_atom[ai] or bond_count[aj] >= k_atom[aj]:
                    continue
                if (ai, aj) in bonded_set:
                    continue

                hat_ij = nl_hats[idx]
                hat_ji = -hat_ij

                # Check angular compatibility with existing bonds at ai
                accept = True
                for existing_hat in bond_hats_per_atom[ai]:
                    cos_a = np.dot(hat_ij, existing_hat)
                    if cos_a > np.cos(min_accept_angle):  # angle < min_accept
                        accept = False
                        break
                if not accept:
                    continue

                # Check angular compatibility at aj
                for existing_hat in bond_hats_per_atom[aj]:
                    cos_a = np.dot(hat_ji, existing_hat)
                    if cos_a > np.cos(min_accept_angle):
                        accept = False
                        break
                if not accept:
                    continue

                s_ai = species_idx[ai]
                s_aj = species_idx[aj]
                _bond_i_list.append(ai)
                _bond_j_list.append(aj)
                _bond_rt_list.append(float(pair_peak[s_ai, s_aj]))
                bonded_set.add((ai, aj))
                bonded_set.add((aj, ai))
                bonded_neighbors[ai].append(aj)
                bonded_neighbors[aj].append(ai)
                bond_hats_per_atom[ai].append(hat_ij.copy())
                bond_hats_per_atom[aj].append(hat_ji.copy())
                bond_count[ai] += 1
                bond_count[aj] += 1

            # Second pass: fill remaining unsatisfied atoms with
            # distance-only matching (relaxing angle constraint)
            for idx in dist_order:
                ai = int(nl_i[idx])
                aj = int(nl_j[idx])
                if bond_count[ai] >= k_atom[ai] or bond_count[aj] >= k_atom[aj]:
                    continue
                if (ai, aj) in bonded_set:
                    continue
                s_ai = species_idx[ai]
                s_aj = species_idx[aj]
                _bond_i_list.append(ai)
                _bond_j_list.append(aj)
                _bond_rt_list.append(float(pair_peak[s_ai, s_aj]))
                bonded_set.add((ai, aj))
                bonded_set.add((aj, ai))
                bonded_neighbors[ai].append(aj)
                bonded_neighbors[aj].append(ai)
                bond_count[ai] += 1
                bond_count[aj] += 1

            bond_i = np.array(_bond_i_list, dtype=np.intp)
            bond_j = np.array(_bond_j_list, dtype=np.intp)
            bond_r_target = np.array(_bond_rt_list, dtype=np.float64)

            # Build triplet arrays from bonded neighbors
            _tc: list[int] = []
            _ta: list[int] = []
            _tb: list[int] = []
            _tp: list[float] = []
            for atom in range(num_atoms):
                bn = bonded_neighbors[atom]
                if len(bn) < 2:
                    continue
                s_center = species_idx[atom]
                for ia in range(len(bn)):
                    for ib in range(ia + 1, len(bn)):
                        s_a = species_idx[bn[ia]]
                        s_b = species_idx[bn[ib]]
                        # Ensure canonical order for angle lookup
                        if s_a <= s_b:
                            triplet_idx = int(angle_lookup[s_center, s_a, s_b])
                        else:
                            triplet_idx = int(angle_lookup[s_center, s_b, s_a])
                        phi_t = float(angle_mode_rad[triplet_idx])
                        _tc.append(atom)
                        _ta.append(int(bn[ia]))
                        _tb.append(int(bn[ib]))
                        _tp.append(phi_t)

            tri_center = np.array(_tc, dtype=np.intp)
            tri_a = np.array(_ta, dtype=np.intp)
            tri_b = np.array(_tb, dtype=np.intp)
            tri_phi_target = np.array(_tp, dtype=np.float64)

        # --- history arrays ---
        loss_history = np.zeros(num_steps + 1, dtype=np.float64)
        best_loss_history = np.zeros(num_steps + 1, dtype=np.float64)
        bond_loss_history = np.zeros(num_steps + 1, dtype=np.float64)
        angle_loss_history = np.zeros(num_steps + 1, dtype=np.float64)
        repulsion_loss_history = np.zeros(num_steps + 1, dtype=np.float64)

        current_step = float(step_size)
        velocity = np.zeros((num_atoms, 3), dtype=np.float64)
        momentum = 0.8  # momentum damping factor
        best_positions = self.atoms.positions.copy()
        best_loss = np.inf

        if show_progress:
            progress = _TextProgressBar(num_steps, label="Shell relax", width=28)
        else:
            progress = None

        # --- main loop ---
        for step in range(num_steps + 1):
            pos = self.atoms.positions  # (num_atoms, 3) — live reference

            # Rebuild bond topology periodically
            if step % neighbor_update_interval == 0:
                rebuild_topology()
                if step == 0:
                    _detect_boundary_atoms()

            # ---------- compute forces ----------
            force = np.zeros((num_atoms, 3), dtype=np.float64)

            # 1) Bond springs
            bond_loss = 0.0
            if bond_i.size > 0:
                bond_vec = min_image(pos[bond_j] - pos[bond_i])
                bond_r = np.linalg.norm(bond_vec, axis=1)
                bond_r_safe = np.maximum(bond_r, _EPS)
                bond_hat = bond_vec / bond_r_safe[:, None]
                delta_r = bond_r - bond_r_target
                bond_loss = float(np.mean(delta_r ** 2))
                f_bond = (bond_weight * delta_r)[:, None] * bond_hat
                np.add.at(force, bond_i, f_bond)
                np.add.at(force, bond_j, -f_bond)

            # 2) Angle springs
            angle_loss = 0.0
            if tri_center.size > 0:
                vec_a = min_image(pos[tri_a] - pos[tri_center])
                vec_b = min_image(pos[tri_b] - pos[tri_center])
                r_a = np.linalg.norm(vec_a, axis=1)
                r_b = np.linalg.norm(vec_b, axis=1)
                r_a_safe = np.maximum(r_a, _EPS)
                r_b_safe = np.maximum(r_b, _EPS)
                hat_a = vec_a / r_a_safe[:, None]
                hat_b = vec_b / r_b_safe[:, None]

                cos_phi = np.sum(hat_a * hat_b, axis=1)
                cos_phi = np.clip(cos_phi, -1.0 + 1e-7, 1.0 - 1e-7)
                phi = np.arccos(cos_phi)
                sin_phi = np.sqrt(1.0 - cos_phi ** 2)
                sin_phi_safe = np.maximum(sin_phi, 1e-7)

                delta_phi = phi - tri_phi_target
                angle_loss = float(np.mean(delta_phi ** 2))

                perp_a = (hat_b - cos_phi[:, None] * hat_a) / sin_phi_safe[:, None]
                perp_b = (hat_a - cos_phi[:, None] * hat_b) / sin_phi_safe[:, None]

                f_angle_a = (angle_weight * delta_phi / r_a_safe)[:, None] * perp_a
                f_angle_b = (angle_weight * delta_phi / r_b_safe)[:, None] * perp_b

                np.add.at(force, tri_a, f_angle_a)
                np.add.at(force, tri_b, f_angle_b)
                np.add.at(force, tri_center, -(f_angle_a + f_angle_b))

            # 3) Repulsion: hard core + non-bonded shell clearance
            repulsion_loss = 0.0
            rep_cutoff = float(np.max(nonbond_push)) * 1.2
            rep_i_all, rep_j_all, rep_d_all, rep_D_all = neighbor_list(
                "ijdD", self.atoms, rep_cutoff,
            )
            if rep_i_all.size > 0:
                s_i = species_idx[rep_i_all]
                s_j = species_idx[rep_j_all]
                r_safe = np.maximum(rep_d_all, _EPS)
                rep_hat = rep_D_all / r_safe[:, None]

                # a) Hard core overlap prevention
                r_hard = hard_core[s_i, s_j]
                hard_ratio = r_hard / r_safe
                hard_mask = hard_ratio > 1.0
                hard_mag = np.zeros_like(r_safe)
                hard_mag[hard_mask] = repulsion_weight * 2.0 * (hard_ratio[hard_mask] - 1.0) ** 2

                # b) Non-bonded clearance
                _pair_keys = rep_i_all.astype(np.int64) * num_atoms + rep_j_all.astype(np.int64)
                _bonded_keys = set(
                    int(a) * num_atoms + int(b) for a, b in bonded_set
                )
                is_bonded = np.array(
                    [int(k) in _bonded_keys for k in _pair_keys], dtype=bool,
                )
                r_push = nonbond_push[s_i, s_j]
                push_ratio = r_push / r_safe
                nonbond_mask = (~is_bonded) & (push_ratio > 1.0)
                nonbond_mag = np.zeros_like(r_safe)
                nonbond_mag[nonbond_mask] = (
                    repulsion_weight * (push_ratio[nonbond_mask] - 1.0) ** 2
                )

                total_rep_mag = hard_mag + nonbond_mag
                active = total_rep_mag > 0.0
                repulsion_loss = float(np.sum(hard_mask)) + 0.1 * float(np.sum(nonbond_mask))

                if np.any(active):
                    f_rep = total_rep_mag[:, None] * rep_hat
                    np.add.at(force, rep_i_all, -f_rep)
                    np.add.at(force, rep_j_all, f_rep)

            # ---------- record loss ----------
            total_loss = bond_loss + angle_loss + repulsion_loss / max(num_atoms, 1)
            loss_history[step] = total_loss
            bond_loss_history[step] = bond_loss
            angle_loss_history[step] = angle_loss
            repulsion_loss_history[step] = repulsion_loss
            if total_loss < best_loss:
                best_loss = total_loss
                best_positions = pos.copy()
            best_loss_history[step] = best_loss

            # ---------- integrate (skip on last step) ----------
            if step < num_steps:
                # Freeze interior grain atoms: zero force and velocity
                if is_boundary is not None:
                    interior_mask = ~is_boundary
                    if np.any(interior_mask):
                        force[interior_mask] = 0.0
                        velocity[interior_mask] = 0.0

                force_mag = np.linalg.norm(force, axis=1)
                clip_mask = force_mag > max_force_clip
                if np.any(clip_mask):
                    force[clip_mask] *= (max_force_clip / force_mag[clip_mask])[:, None]

                # FIRE-inspired: reset velocity on direction reversal
                vf_dot = np.sum(velocity * force)
                if vf_dot < 0:
                    velocity[:] = 0.0
                else:
                    velocity = momentum * velocity + current_step * force

                new_pos = pos + velocity

                frac = new_pos @ cell_inv
                frac %= 1.0
                self.atoms.positions = frac @ cell_mat

                current_step *= step_decay

            if progress is not None:
                progress.update(step)

        if progress is not None:
            progress.update(num_steps)

        # Restore best positions
        frac_best = best_positions @ cell_inv
        frac_best %= 1.0
        self.atoms.positions = frac_best @ cell_mat

        # Store history
        step_arr = np.arange(num_steps + 1, dtype=np.int32)
        self.shell_relax_history = {
            "step": step_arr,
            "loss": loss_history,
            "best_loss": best_loss_history,
            "bond_loss": bond_loss_history,
            "angle_loss": angle_loss_history,
            "repulsion_loss": repulsion_loss_history,
        }

        # Invalidate caches
        self.current_distribution = None
        self.current_cost = None
        self.mc_history = None
        self.last_temperature = None
        self._rebuild_spatial_index()

        summary: dict[str, Any] = {
            "num_steps": int(num_steps),
            "bond_weight": float(bond_weight),
            "angle_weight": float(angle_weight),
            "repulsion_weight": float(repulsion_weight),
            "step_size": float(step_size),
            "step_decay": float(step_decay),
            "neighbor_update_interval": int(neighbor_update_interval),
            "neighbor_cutoff_scale": float(neighbor_cutoff_scale),
            "initial_loss": float(loss_history[0]),
            "final_loss": float(loss_history[-1]),
            "best_loss": float(best_loss),
            "num_atoms": num_atoms,
        }
        return summary

    def plot_shell_relax(
        self,
        *,
        log_y: bool = False,
    ):
        """Plot the recorded shell-relax loss history using Matplotlib."""
        if self.shell_relax_history is None:
            raise ValueError("Run shell_relax() before plotting the history.")

        import matplotlib.pyplot as plt

        hist = self.shell_relax_history
        fig, ax = plt.subplots(figsize=(6.8, 4.2))
        ax.plot(hist["step"], hist["loss"], lw=1.8, label="total loss")
        ax.plot(hist["step"], hist["best_loss"], lw=1.4, ls="--", label="best")
        ax.plot(hist["step"], hist["bond_loss"], lw=1.0, alpha=0.7, label="bond")
        ax.plot(hist["step"], hist["angle_loss"], lw=1.0, alpha=0.7, label="angle")
        ax.plot(
            hist["step"],
            hist["repulsion_loss"] / max(len(self.atoms), 1),
            lw=1.0, alpha=0.7, label="repulsion (per atom)",
        )
        if log_y:
            positive = hist["loss"][hist["loss"] > 0.0]
            if positive.size:
                ax.set_yscale("log")
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.set_title("Shell relax history")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        return fig, ax

    def plot_structure(
        self,
        shell_target: "CoordinationShellTarget | None" = None,
        *,
        output: str | None = None,
        width: int = 1024,
        height: int = 1024,
        fps: int = 60,
        duration: float = 6.0,
        elevation: float = 15.0,
        atom_size: float = 10.0,
        bond_cutoff: float | None = None,
        show_cell: bool = True,
        show_atoms: bool = True,
        background: str = "white",
        colormap: str = "copper",
        tetrahedral_thresh: float = 0.4,
        show_progress: bool = True,
    ):
        """Render a bond-centric rotating 3D view of the atomic structure.

        Bonds are the primary visual: crystalline (tetrahedral) bonds
        are drawn thick and coloured by depth; boundary / amorphous
        bonds are drawn faint.  Atoms are optional small dots.  The
        animation performs a full periodic 360-degree rotation.

        Classification follows the MATLAB ``plotAtoms02`` convention:
        an atom is *crystalline* if it has exactly K nearest neighbours
        within *bond_cutoff* **and** the mean displacement of those
        neighbours is less than *tetrahedral_thresh* (i.e. the local
        coordination is symmetric / tetrahedral).

        Parameters
        ----------
        shell_target
            First-shell targets.  Used to set *bond_cutoff* and the
            coordination number K automatically.
        output
            File path for a ``.mp4`` (recommended) or ``.gif``.
            ``None`` shows a static figure.
        width, height
            Frame size in pixels.
        fps
            Frames per second (GIF only).
        duration
            Total GIF length in seconds.  Rotation is always exactly
            360 degrees so the loop is seamless.
        elevation
            Camera elevation in degrees.
        atom_size
            Matplotlib scatter marker size.  Set to 0 to hide atoms.
        bond_cutoff
            NN bond length cutoff in Angstrom.
        show_cell
            Draw the periodic cell outline.
        show_atoms
            Draw atom dots.
        background
            Figure background colour.
        colormap
            Matplotlib colormap for crystalline bonds (coloured by
            depth / y-coordinate after rotation, like MATLAB ``bone``).
        tetrahedral_thresh
            Maximum norm of mean NN displacement vector for an atom to
            be classified as crystalline.  Smaller = stricter.
        show_progress
            Print frame counter during GIF rendering.
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Line3DCollection

        pos = self.atoms.positions.copy()
        cell_mat = np.asarray(self.atoms.cell.array, dtype=np.float64)
        cell_inv = np.linalg.inv(cell_mat)
        num_atoms = len(self.atoms)

        # --- bond cutoff ---
        if bond_cutoff is None:
            if shell_target is not None:
                # Use a generous cutoff (pair_peak + 3*sigma or 1.2x)
                pair_peak_max = float(np.max(
                    np.asarray(shell_target.pair_peak, dtype=np.float64),
                ))
                bond_cutoff = pair_peak_max * 1.2
            else:
                bond_cutoff = 3.0

        if shell_target is not None:
            coord_target = np.asarray(shell_target.coordination_target, dtype=np.float64)
            species_idx = self._atom_species_index
            k_per_atom = np.array([
                int(np.round(coord_target[species_idx[a]].sum()))
                for a in range(num_atoms)
            ], dtype=np.intp)
        else:
            k_per_atom = np.full(num_atoms, 4, dtype=np.intp)

        # --- find bonds ---
        bi_all, bj_all, bd_all = neighbor_list("ijd", self.atoms, bond_cutoff)

        def _min_image(delta: np.ndarray) -> np.ndarray:
            frac = delta @ cell_inv
            frac -= np.rint(frac)
            return frac @ cell_mat

        # --- classify atoms as crystalline ---
        # Two criteria (either makes an atom crystalline):
        # 1) Tetrahedral check (MATLAB style): K NN within cutoff and
        #    symmetric coordination (mean displacement < thresh).
        #    Allow K-1 to K+1 neighbors for tolerance.
        # 2) Grain interior: deep inside a crystalline Voronoi cell.
        is_crystalline_atom = np.zeros(num_atoms, dtype=bool)

        # Criterion 1: tetrahedral / symmetric coordination
        for a in range(num_atoms):
            mask = bi_all == a
            nn_count = int(np.sum(mask))
            k_target = int(k_per_atom[a])
            if nn_count < max(k_target - 1, 1) or nn_count > k_target + 1:
                continue
            # Use K nearest for the displacement check
            dists_a = bd_all[mask]
            js_a = bj_all[mask]
            order = np.argsort(dists_a)[:k_target]
            dxyz = _min_image(pos[js_a[order]] - pos[a])
            mean_disp = np.linalg.norm(np.mean(dxyz, axis=0))
            if mean_disp < tetrahedral_thresh:
                is_crystalline_atom[a] = True

        # Criterion 2: grain interior atoms (always crystalline)
        grain_ids = self._grain_ids
        grain_seeds = self._grain_seeds
        if grain_ids is not None and grain_seeds is not None:
            pp_max = float(np.max(
                np.asarray(shell_target.pair_peak, dtype=np.float64),
            )) if shell_target is not None else 2.5
            bw = pp_max * 0.5
            delta_seeds = pos[:, None, :] - grain_seeds[None, :, :]
            frac_ds = delta_seeds @ cell_inv
            frac_ds -= np.rint(frac_ds)
            cart_ds = frac_ds @ cell_mat
            dist_to_seeds = np.sqrt(np.sum(cart_ds ** 2, axis=2))
            for ia in range(num_atoms):
                gid = grain_ids[ia]
                if gid < 0:
                    continue
                d_own = dist_to_seeds[ia, gid]
                dists_copy = dist_to_seeds[ia].copy()
                dists_copy[gid] = np.inf
                d_other = float(np.min(dists_copy))
                if (d_other - d_own) * 0.5 > bw:
                    is_crystalline_atom[ia] = True

        # Keep i < j for unique bonds
        mask_ij = bi_all < bj_all
        bi, bj = bi_all[mask_ij], bj_all[mask_ij]

        # Bond is crystalline if BOTH endpoints are crystalline
        bond_is_cryst = is_crystalline_atom[bi] & is_crystalline_atom[bj]

        # Bond segment endpoints (minimum-image)
        bond_vecs = _min_image(pos[bj] - pos[bi])
        bond_starts = pos[bi]
        bond_ends = bond_starts + bond_vecs

        # --- center everything ---
        a_vec, b_vec, c_vec = cell_mat[0], cell_mat[1], cell_mat[2]
        center = 0.5 * (a_vec + b_vec + c_vec)
        pos_c = pos - center
        bstart_c = bond_starts - center
        bend_c = bond_ends - center

        # Cell outline edges
        o = -center
        cell_corners = [
            o, o + a_vec, o + b_vec, o + c_vec,
            o + a_vec + b_vec, o + a_vec + c_vec, o + b_vec + c_vec,
            o + a_vec + b_vec + c_vec,
        ]
        cell_edge_pairs = [
            (0, 1), (0, 2), (0, 3), (1, 4), (1, 5), (2, 4),
            (2, 6), (3, 5), (3, 6), (4, 7), (5, 7), (6, 7),
        ]
        cell_segs = [(cell_corners[i], cell_corners[j]) for i, j in cell_edge_pairs]

        # --- colormap for crystalline bonds (depth-coloured) ---
        cmap = plt.get_cmap(colormap)

        cryst_mask = bond_is_cryst
        bnd_mask = ~bond_is_cryst
        extent = float(np.max(np.abs(pos_c))) * 1.15

        dpi = 100
        figsize = (width / dpi, height / dpi)

        def _rotate_2d(pts: np.ndarray, theta: float) -> np.ndarray:
            """Rotate x-y columns by theta radians (in-place friendly)."""
            c, s = np.cos(theta), np.sin(theta)
            x_new = pts[:, 0] * c - pts[:, 1] * s
            y_new = pts[:, 0] * s + pts[:, 1] * c
            out = pts.copy()
            out[:, 0] = x_new
            out[:, 1] = y_new
            return out

        def _draw_frame(theta_rad: float) -> "plt.Figure":
            fig = plt.figure(figsize=figsize, dpi=dpi)
            ax = fig.add_subplot(111, projection="3d")
            ax.set_facecolor(background)
            fig.patch.set_facecolor(background)

            # Perspective projection (like MATLAB camproj('perspective'))
            try:
                ax.set_proj_type("persp", focal_length=0.25)
            except (TypeError, AttributeError):
                pass  # older matplotlib

            # Rotate bond endpoints in x-y plane (like MATLAB)
            bs_r = _rotate_2d(bstart_c, theta_rad)
            be_r = _rotate_2d(bend_c, theta_rad)

            # --- boundary bonds: very faint ---
            if np.any(bnd_mask):
                segs_b = list(zip(bs_r[bnd_mask], be_r[bnd_mask]))
                lc_b = Line3DCollection(
                    segs_b, linewidths=0.3,
                    colors=(0.0, 0.0, 0.0, 0.05),
                )
                ax.add_collection3d(lc_b)

            # --- crystalline bonds: depth-coloured + depth-width ---
            # Camera at +x (azim=0): larger rotated-x = closer = brighter + thicker
            if np.any(cryst_mask):
                segs_cr = list(zip(bs_r[cryst_mask], be_r[cryst_mask]))
                mid_x_rot = 0.5 * (bs_r[cryst_mask, 0] + be_r[cryst_mask, 0])
                norm_depth = (mid_x_rot + extent) / max(2.0 * extent, _EPS)
                norm_depth = np.clip(norm_depth, 0, 1)
                cryst_colors = cmap(norm_depth)
                # Linewidth: 0.4 at back, 1.8 at front
                cryst_lw = 0.4 + 1.4 * norm_depth
                lc_c = Line3DCollection(
                    segs_cr, linewidths=cryst_lw, colors=cryst_colors,
                )
                ax.add_collection3d(lc_c)

            # --- cell outline ---
            if show_cell:
                cell_segs_r = []
                for s, e in cell_segs:
                    s_r = _rotate_2d(s.reshape(1, 3), theta_rad)[0]
                    e_r = _rotate_2d(e.reshape(1, 3), theta_rad)[0]
                    cell_segs_r.append((s_r, e_r))
                lc_cell = Line3DCollection(
                    cell_segs_r, linewidths=1.5, colors="k", alpha=0.5,
                )
                ax.add_collection3d(lc_cell)

            # --- atoms (tiny dots) ---
            if show_atoms and atom_size > 0:
                pos_r = _rotate_2d(pos_c, theta_rad)
                ax.scatter(
                    pos_r[:, 0], pos_r[:, 1], pos_r[:, 2],
                    s=atom_size, c="k", alpha=0.15,
                    edgecolors="none", depthshade=False,
                )

            # Zoom: moderate tightening to reduce whitespace
            zoom = extent * 0.92
            ax.set_xlim(-zoom, zoom)
            ax.set_ylim(-zoom, zoom)
            ax.set_zlim(-zoom, zoom)
            ax.set_box_aspect([1, 1, 1])
            ax.view_init(elev=elevation, azim=0)  # azim fixed; we rotate data
            ax.axis("off")
            fig.subplots_adjust(left=-0.08, right=1.08, bottom=-0.08, top=1.08)
            return fig

        # --- static display or animation ---
        if output is None:
            return _draw_frame(theta_rad=np.deg2rad(45.0))

        # Full 360-degree periodic rotation
        n_frames = int(fps * duration)
        thetas = np.linspace(0, 2 * np.pi, n_frames, endpoint=False)

        if show_progress:
            progress = _TextProgressBar(n_frames, label="Rendering", width=28)
        else:
            progress = None

        is_mp4 = str(output).lower().endswith(".mp4")

        if is_mp4:
            # MP4 via ffmpeg subprocess — true 60fps
            import subprocess
            import io

            # Frame dimensions from figure size (no bbox_inches="tight"
            # for the raw pipe — keeps pixel count deterministic).
            fw, fh = width, height

            ffmpeg_cmd = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "rawvideo",
                "-pix_fmt", "rgba",
                "-s", f"{fw}x{fh}",
                "-r", str(fps),
                "-i", "pipe:0",
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-preset", "fast",
                "-crf", "18",
                str(output),
            ]
            proc = subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )

            for i, th in enumerate(thetas):
                fig_i = _draw_frame(th)
                buf_i = io.BytesIO()
                fig_i.savefig(buf_i, format="raw", dpi=dpi,
                              facecolor=background)
                plt.close(fig_i)
                proc.stdin.write(buf_i.getvalue())
                if progress is not None:
                    progress.update(i + 1)

            proc.stdin.close()
            proc.wait()
        else:
            # GIF fallback via Pillow
            from PIL import Image
            import io

            frames: list[Image.Image] = []
            for i, th in enumerate(thetas):
                fig = _draw_frame(th)
                buf = io.BytesIO()
                fig.savefig(
                    buf, format="png", dpi=dpi,
                    bbox_inches="tight", pad_inches=0,
                    facecolor=background,
                )
                plt.close(fig)
                buf.seek(0)
                frames.append(Image.open(buf).copy())
                buf.close()
                if progress is not None:
                    progress.update(i + 1)

            frame_duration_ms = int(1000 / fps)
            frames[0].save(
                output,
                save_all=True,
                append_images=frames[1:],
                duration=frame_duration_ms,
                loop=0,
                optimize=True,
            )

        if progress is not None:
            progress.update(n_frames)
        return output

    def generate(
        self,
        shell_target: "CoordinationShellTarget",
        num_steps: int = 200,
        *,
        grain_size: float | None = None,
        crystalline_fraction: float = 1.0,
        r_broadening: float | None = None,
        phi_broadening: float | None = None,
        show_progress: bool = True,
        **shell_relax_kwargs: Any,
    ) -> dict[str, Any]:
        """Generate a disordered supercell from liquid to nanocrystalline.

        Covers the full spectrum of disorder:

        * **Liquid** — ``grain_size=None, phi_broadening=180``:
          only nearest-neighbor distances enforced, angles free.
        * **Amorphous** — ``grain_size=None, r_broadening=0.1,
          phi_broadening=10``: short-range order with tunable
          distance and angle sharpness.
        * **Short-range order** — ``grain_size=5, crystalline_fraction=0.3``:
          small crystalline clusters in an amorphous matrix.
        * **Mixed** — ``grain_size=15, crystalline_fraction=0.5``:
          50 % crystalline grains, 50 % amorphous fill.
        * **Nanocrystalline** — ``grain_size=15, crystalline_fraction=1.0``:
          grains fill the entire box with thin disordered boundaries.

        Also builds a matching *target_g3* distribution (auto-derived
        from the construction parameters) and stores it on
        :attr:`target_distribution` for use with :meth:`plot_g3_compare`.

        Parameters
        ----------
        shell_target
            First-shell coordination targets from the reference crystal.
        num_steps
            Number of relaxation sweeps.
        grain_size
            Diameter of crystalline grains in Angstrom.  ``None`` means
            no grains — start from random positions (amorphous/liquid).
        crystalline_fraction
            Volume fraction filled by crystalline grains (0–1).  Only
            used when *grain_size* is set.  The remaining volume is
            filled with random (amorphous) positions.
        r_broadening
            Radial disorder σ in Angstrom at the nearest-neighbor
            distance.  Controls how tightly bond lengths are enforced.
            ``None`` uses a default.  Larger values → more distance
            freedom.  Also sets the radial blur for the target g3.
        phi_broadening
            Angular disorder σ in degrees.  Controls how tightly bond
            angles are enforced.  ``None`` uses a default.  180 means
            angles are essentially free (liquid-like).  Small values
            (e.g. 3) enforce sharp tetrahedral angles (diamond-like).
            Also sets the angular blur for the target g3.
        show_progress
            Display a text progress bar.
        **shell_relax_kwargs
            Additional keyword arguments forwarded to :meth:`shell_relax`.
            Explicit ``bond_weight`` / ``angle_weight`` override the
            auto-derived values from broadening parameters.

        Returns
        -------
        dict[str, Any]
            Summary dict with regime, loss values, and construction
            parameters.
        """
        pair_peak = np.asarray(shell_target.pair_peak, dtype=np.float64)
        pair_peak_max = float(np.max(pair_peak[pair_peak > _EPS])) if np.any(pair_peak > _EPS) else 2.5
        max_pair_outer = float(shell_target.max_pair_outer)
        g3_r_max = float(self.measure_r_max)
        r_step = float(self.measure_r_step)

        # --- compute force weights from broadening ---
        auto_weights = self._broadening_to_weights(
            pair_peak_max, r_broadening, phi_broadening,
        )
        for key, val in auto_weights.items():
            shell_relax_kwargs.setdefault(key, val)

        # --- construct atoms ---
        use_grains = grain_size is not None and float(grain_size) > 0.0

        if use_grains:
            disp_sigma = float(r_broadening) if (r_broadening is not None and r_broadening > _EPS) else 0.0

            self.atoms = self._build_grain_atoms(
                shell_target,
                grain_size=float(grain_size),
                crystalline_fraction=crystalline_fraction,
                displacement_sigma=disp_sigma,
            )

            # Refresh cached arrays after rebuilding atoms
            self._cell_matrix = np.asarray(self.atoms.cell.array, dtype=np.float64)
            self._cell_inverse = np.linalg.inv(self._cell_matrix)
            self._atom_species_index = np.searchsorted(
                self._species, self.atoms.numbers,
            )
            self._rebuild_spatial_index()

        # --- auto-derive target_g3 from construction params ---
        # For grains: use actual Voronoi cell sizes, not the requested
        # grain_size (which may differ due to packing).
        if use_grains and self._grain_ids is not None:
            gids = self._grain_ids
            cryst_gids = gids[gids >= 0]
            if len(cryst_gids) > 0:
                vals, cnts = np.unique(cryst_gids, return_counts=True)
                ref_density = len(self.reference_atoms) / max(
                    float(self.reference_atoms.cell.volume), _EPS,
                )
                grain_diams = 2.0 * np.cbrt(3.0 * cnts / ref_density / (4.0 * np.pi))
                median_diam = float(np.median(grain_diams))
                target_r_min = median_diam * 0.4
                target_r_max = median_diam * 0.7
            else:
                target_r_min = max_pair_outer
                target_r_max = target_r_min + 4.0
        else:
            target_r_min = max_pair_outer
            target_r_max = target_r_min + 4.0

        # Clamp to g3 grid range
        target_r_max = min(target_r_max, g3_r_max - r_step)
        target_r_min = min(target_r_min, target_r_max - r_step)

        # Build target distribution from the raw measured g3
        self.target_distribution = self._raw_distribution.target_g3(
            target_r_min=target_r_min,
            target_r_max=target_r_max,
            r_sigma=r_broadening,
            r_sigma_at=pair_peak_max,
            phi_sigma_deg=phi_broadening,
            label="target",
        )

        # --- relax ---
        summary = self.shell_relax(
            shell_target,
            num_steps=num_steps,
            show_progress=show_progress,
            **shell_relax_kwargs,
        )

        # --- summary ---
        if use_grains:
            summary["regime"] = "nanocrystalline" if crystalline_fraction >= 0.9 else "mixed"
            summary["n_grains"] = int(self.atoms.info.get("n_grains", 0))
            summary["grain_size"] = float(grain_size)
            summary["crystalline_fraction"] = crystalline_fraction
        else:
            summary["regime"] = "amorphous"
        summary["r_broadening"] = r_broadening
        summary["phi_broadening"] = phi_broadening
        summary["target_r_min"] = target_r_min
        summary["target_r_max"] = target_r_max

        return summary

    def plot_motif_training(
        self,
        *,
        log_y: bool = False,
    ):
        """Plot the recorded motif-training relaxation history using Matplotlib."""
        if self.motif_history is None:
            raise ValueError("Run motif_training() before plotting the history.")

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.8, 4.2))
        ax.plot(self.motif_history["step"], self.motif_history["loss"], lw=1.8, label="loss")
        ax.plot(self.motif_history["step"], self.motif_history["best_loss"], lw=1.4, label="best")
        if log_y:
            positive_loss = np.asarray(self.motif_history["loss"], dtype=np.float64)
            positive_loss = positive_loss[positive_loss > 0.0]
            if positive_loss.size:
                ax.set_yscale("log")
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.set_title("Motif training history")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        return fig, ax

    def motif_training(
        self,
        shell_target: CoordinationShellTarget,
        *,
        num_graph_retries: int = 12,
        num_relax_steps: int = 800,
        step_size: float = 0.35,
        spring_step: float = 0.30,
        inertia: float = 0.25,
        repulsion_step: float = 0.12,
        overlap_weight: float = 10.0,
        plot_history: bool = False,
        show_progress: bool = True,
    ) -> dict[str, Any]:
        """Build a random motif network and relax it toward the reference first shell.

        Unlike `shell_refine()`, this method does not begin from a random close-packed
        point cloud. It assigns each atom a crystal-derived first-shell motif, builds a
        random bond graph from those motifs, embeds that graph into 3D, and then
        iteratively relaxes positions by aligning local neighbor clusters.
        """
        if not np.array_equal(np.asarray(shell_target.species, dtype=np.int64), self._species):
            raise ValueError("shell_target species must match the Supercell composition ordering.")
        num_graph_retries = int(num_graph_retries)
        if num_graph_retries <= 0:
            raise ValueError("num_graph_retries must be positive.")
        num_relax_steps = int(num_relax_steps)
        if num_relax_steps <= 0:
            raise ValueError("num_relax_steps must be positive.")
        step_size = float(step_size)
        if step_size <= 0.0:
            raise ValueError("step_size must be positive.")
        spring_step = float(spring_step)
        if spring_step < 0.0:
            raise ValueError("spring_step must be non-negative.")
        inertia = float(inertia)
        if inertia < 0.0:
            raise ValueError("inertia must be non-negative.")
        repulsion_step = float(repulsion_step)
        if repulsion_step < 0.0:
            raise ValueError("repulsion_step must be non-negative.")
        overlap_weight = float(overlap_weight)
        if overlap_weight < 0.0:
            raise ValueError("overlap_weight must be non-negative.")

        motif_ids = self._assign_motif_templates(shell_target)
        graph = self._build_random_motif_graph(
            shell_target,
            motif_ids,
            num_retries=num_graph_retries,
        )
        positions = self._initial_motif_graph_positions(shell_target, motif_ids, graph)
        self.atoms.positions[:] = positions
        shell_bin_size = float(max(self.measure_r_step, shell_target.max_pair_outer))
        if self.spatial_bin_size > shell_bin_size * 1.05:
            self.spatial_bin_size = shell_bin_size
        self._rebuild_spatial_index()

        max_repulsion_distance = float(np.max(shell_target.pair_hard_min)) if np.any(shell_target.pair_mask) else 0.0
        loss_history = np.empty(num_relax_steps + 1, dtype=np.float64)
        best_history = np.empty(num_relax_steps + 1, dtype=np.float64)
        bond_history = np.empty(num_relax_steps + 1, dtype=np.float64)
        overlap_history = np.empty(num_relax_steps + 1, dtype=np.float64)
        if num_relax_steps > 0:
            loss_history[0] = np.nan
            best_history[0] = np.nan
            bond_history[0] = np.nan
            overlap_history[0] = np.nan
        else:
            loss_history[0] = 0.0
            best_history[0] = 0.0
            bond_history[0] = 0.0
            overlap_history[0] = 0.0

        progress = None
        if show_progress:
            progress = _TextProgressBar(num_relax_steps, label="Motif training", width=28)
            progress.update(0)

        best_loss = np.inf
        best_positions = np.array(positions, copy=True)
        for step in range(1, num_relax_steps + 1):
            positions, bond_loss, overlap_loss = self._motif_vote_relax_step(
                positions,
                shell_target,
                motif_ids,
                graph,
                step_size=step_size,
                spring_step=spring_step,
                inertia=inertia,
                repulsion_step=repulsion_step,
                max_repulsion_distance=max_repulsion_distance,
            )
            total_loss = float(bond_loss + overlap_weight * overlap_loss)
            if total_loss < best_loss:
                best_loss = total_loss
                best_positions = np.array(positions, copy=True)
            loss_history[step] = total_loss
            best_history[step] = best_loss
            bond_history[step] = float(bond_loss)
            overlap_history[step] = float(overlap_loss)
            if progress is not None:
                progress.update(step)

        if progress is not None:
            progress.close()

        self.atoms.positions[:] = self._wrap_positions(best_positions)
        self._rebuild_spatial_index()
        self.current_distribution = None
        self.current_cost = None
        self.mc_history = None
        self.shell_history = None
        self.last_temperature = None
        self.motif_graph = {
            **graph,
            "motif_ids": np.asarray(motif_ids, dtype=np.intp, copy=True),
        }
        self.motif_history = {
            "step": np.arange(num_relax_steps + 1, dtype=np.int32),
            "loss": loss_history,
            "best_loss": best_history,
            "bond_loss": bond_history,
            "overlap_loss": overlap_history,
        }

        actual_degree = np.asarray(graph["actual_degree"], dtype=np.intp)
        target_degree = np.asarray(graph["target_degree"], dtype=np.intp)
        matched_fraction = float(
            np.sum(actual_degree) / max(float(np.sum(target_degree)), 1.0)
        )
        summary = {
            "num_graph_retries": int(num_graph_retries),
            "num_relax_steps": int(num_relax_steps),
            "step_size": float(step_size),
            "spring_step": float(spring_step),
            "inertia": float(inertia),
            "repulsion_step": float(repulsion_step),
            "overlap_weight": float(overlap_weight),
            "unmatched_slots": int(graph["unmatched_slots"]),
            "matched_fraction": matched_fraction,
            "mean_target_degree": float(np.mean(target_degree)) if target_degree.size else 0.0,
            "mean_actual_degree": float(np.mean(actual_degree)) if actual_degree.size else 0.0,
            "initial_loss": float(loss_history[1]),
            "final_loss": float(loss_history[-1]),
            "best_loss": float(best_loss),
            "num_atoms": len(self.atoms),
            "cell_dim_angstroms": self.cell_dim_angstroms,
            "relative_density": self.relative_density,
        }
        if plot_history:
            self.plot_motif_training()
        return summary

    def motif_match_training(
        self,
        shell_target: CoordinationShellTarget,
        *,
        num_steps: int = 800,
        step_size: float = 0.55,
        inertia: float = 0.15,
        repulsion_step: float = 0.12,
        overlap_weight: float = 10.0,
        unmatched_weight: float = 25.0,
        match_candidates_per_slot: int = 6,
        plot_history: bool = False,
        show_progress: bool = True,
    ) -> dict[str, Any]:
        """Assemble a random motif soup by fuzzy matching predicted shell atoms.

        This starts from randomly positioned atoms, assigns each atom a reference-shell
        motif, then repeatedly matches predicted neighbor positions onto actual atoms of
        the correct species with per-atom capacity limits. The topology adapts every
        step, which is more flexible than a fixed random bond graph for open networks
        such as Si.
        """
        if not np.array_equal(np.asarray(shell_target.species, dtype=np.int64), self._species):
            raise ValueError("shell_target species must match the Supercell composition ordering.")
        num_steps = int(num_steps)
        if num_steps <= 0:
            raise ValueError("num_steps must be positive.")
        step_size = float(step_size)
        if step_size <= 0.0:
            raise ValueError("step_size must be positive.")
        inertia = float(inertia)
        if inertia < 0.0:
            raise ValueError("inertia must be non-negative.")
        repulsion_step = float(repulsion_step)
        if repulsion_step < 0.0:
            raise ValueError("repulsion_step must be non-negative.")
        overlap_weight = float(overlap_weight)
        unmatched_weight = float(unmatched_weight)
        if overlap_weight < 0.0 or unmatched_weight < 0.0:
            raise ValueError("overlap_weight and unmatched_weight must be non-negative.")
        match_candidates_per_slot = int(match_candidates_per_slot)
        if match_candidates_per_slot <= 0:
            raise ValueError("match_candidates_per_slot must be positive.")

        motif_ids = self._assign_motif_templates(shell_target)
        positions = np.array(self.atoms.positions, dtype=np.float64, copy=True)
        rotations = np.stack(
            [self._random_rotation_matrix() for _ in range(len(self.atoms))],
            axis=0,
        )
        shell_bin_size = float(max(self.measure_r_step, shell_target.max_pair_outer))
        if self.spatial_bin_size > shell_bin_size * 1.05:
            self.spatial_bin_size = shell_bin_size
        self.atoms.positions[:] = self._wrap_positions(positions)
        self._rebuild_spatial_index()

        max_repulsion_distance = float(np.max(shell_target.pair_hard_min)) if np.any(shell_target.pair_mask) else 0.0
        loss_history = np.empty(num_steps + 1, dtype=np.float64)
        best_history = np.empty(num_steps + 1, dtype=np.float64)
        match_history = np.empty(num_steps + 1, dtype=np.float64)
        overlap_history = np.empty(num_steps + 1, dtype=np.float64)
        unmatched_history = np.empty(num_steps + 1, dtype=np.float64)
        matched_fraction_history = np.empty(num_steps + 1, dtype=np.float64)
        loss_history[0] = np.nan
        best_history[0] = np.nan
        match_history[0] = np.nan
        overlap_history[0] = np.nan
        unmatched_history[0] = np.nan
        matched_fraction_history[0] = np.nan

        progress = None
        if show_progress:
            progress = _TextProgressBar(num_steps, label="Motif match", width=28)
            progress.update(0)

        best_loss = np.inf
        best_positions = np.array(positions, copy=True)
        best_matches: dict[str, Any] | None = None
        for step in range(1, num_steps + 1):
            matches = self._build_dynamic_motif_matches(
                positions,
                rotations,
                shell_target,
                motif_ids,
                match_candidates_per_slot=match_candidates_per_slot,
            )
            positions, rotations, match_loss, overlap_loss = self._motif_match_relax_step(
                positions,
                rotations,
                shell_target,
                motif_ids,
                matches,
                step_size=step_size,
                repulsion_step=repulsion_step,
                inertia=inertia,
                max_repulsion_distance=max_repulsion_distance,
            )
            matched_fraction = float(
                matches["matched_slots"] / max(float(np.sum(matches["capacities"])), 1.0)
            )
            total_loss = float(
                match_loss
                + overlap_weight * overlap_loss
                + unmatched_weight * float(matches["unmatched_slots"])
            )
            if total_loss < best_loss:
                best_loss = total_loss
                best_positions = np.array(positions, copy=True)
                best_matches = matches
            loss_history[step] = total_loss
            best_history[step] = best_loss
            match_history[step] = float(match_loss)
            overlap_history[step] = float(overlap_loss)
            unmatched_history[step] = float(matches["unmatched_slots"])
            matched_fraction_history[step] = matched_fraction
            if progress is not None:
                progress.update(step)

        if progress is not None:
            progress.close()

        self.atoms.positions[:] = self._wrap_positions(best_positions)
        self._rebuild_spatial_index()
        self.current_distribution = None
        self.current_cost = None
        self.mc_history = None
        self.shell_history = None
        self.last_temperature = None
        self.motif_graph = {
            "motif_ids": np.asarray(motif_ids, dtype=np.intp, copy=True),
            "matches": best_matches,
        }
        self.motif_history = {
            "step": np.arange(num_steps + 1, dtype=np.int32),
            "loss": loss_history,
            "best_loss": best_history,
            "bond_loss": match_history,
            "overlap_loss": overlap_history,
            "unmatched_slots": unmatched_history,
            "matched_fraction": matched_fraction_history,
        }

        final_matches = best_matches if best_matches is not None else self._build_dynamic_motif_matches(
            self.atoms.positions,
            rotations,
            shell_target,
            motif_ids,
            match_candidates_per_slot=match_candidates_per_slot,
        )
        summary = {
            "num_steps": int(num_steps),
            "step_size": float(step_size),
            "inertia": float(inertia),
            "repulsion_step": float(repulsion_step),
            "overlap_weight": float(overlap_weight),
            "unmatched_weight": float(unmatched_weight),
            "match_candidates_per_slot": int(match_candidates_per_slot),
            "matched_fraction": float(
                final_matches["matched_slots"] / max(float(np.sum(final_matches["capacities"])), 1.0)
            ),
            "unmatched_slots": int(final_matches["unmatched_slots"]),
            "initial_loss": float(loss_history[1]),
            "final_loss": float(loss_history[-1]),
            "best_loss": float(best_loss),
            "num_atoms": len(self.atoms),
            "cell_dim_angstroms": self.cell_dim_angstroms,
            "relative_density": self.relative_density,
        }
        if plot_history:
            self.plot_motif_training()
        return summary

    def motif_growth_training(
        self,
        shell_target: CoordinationShellTarget,
        *,
        merge_distance_scale: float = 0.18,
        num_relax_steps: int = 12,
        step_size: float = 0.0,
        spring_step: float = 0.12,
        inertia: float = 0.0,
        repulsion_step: float = 0.10,
        overlap_weight: float = 12.0,
        plot_history: bool = False,
        show_progress: bool = True,
    ) -> dict[str, Any]:
        """Grow random local motifs, then narrow the shell without broad re-optimization.

        The random motif-unit initializer already captures the correct first-shell
        angular character, but it can leave accidental nonbonded background at short
        distances. This method therefore applies only a narrow cleanup pass:
        motif bonds are gently pulled toward their template shell while nonbonded
        overlaps are repelled. The default intentionally avoids the broader relaxers
        that tended to wash the shell back toward random close packing.
        """
        if not np.array_equal(np.asarray(shell_target.species, dtype=np.int64), self._species):
            raise ValueError("shell_target species must match the Supercell composition ordering.")
        merge_distance_scale = float(merge_distance_scale)
        if merge_distance_scale <= 0.0:
            raise ValueError("merge_distance_scale must be positive.")
        num_relax_steps = int(num_relax_steps)
        if num_relax_steps < 0:
            raise ValueError("num_relax_steps must be non-negative.")
        step_size = float(step_size)
        spring_step = float(spring_step)
        inertia = float(inertia)
        repulsion_step = float(repulsion_step)
        overlap_weight = float(overlap_weight)
        if (
            step_size < 0.0
            or spring_step < 0.0
            or inertia < 0.0
            or repulsion_step < 0.0
            or overlap_weight < 0.0
        ):
            raise ValueError("Invalid cleanup parameters for motif_growth_training.")

        positions, numbers, motif_ids, graph = self._build_grown_motif_graph(
            shell_target,
            merge_distance_scale=merge_distance_scale,
            show_progress=show_progress,
        )
        if positions.shape[0] != len(self.atoms):
            raise RuntimeError(
                f"Motif growth created {positions.shape[0]} atoms, expected {len(self.atoms)}."
            )

        self.atoms = Atoms(
            numbers=numbers,
            cell=self._cell_matrix,
            pbc=self.reference_atoms.pbc,
            positions=self._wrap_positions(positions),
        )
        self._atom_species_index = np.searchsorted(self._species, self.atoms.numbers)
        shell_bin_size = float(max(self.measure_r_step, shell_target.max_pair_outer))
        if self.spatial_bin_size > shell_bin_size * 1.05:
            self.spatial_bin_size = shell_bin_size
        self._rebuild_spatial_index()

        max_repulsion_distance = float(np.max(shell_target.pair_hard_min)) if np.any(shell_target.pair_mask) else 0.0
        loss_history = np.empty(num_relax_steps + 1, dtype=np.float64)
        best_history = np.empty(num_relax_steps + 1, dtype=np.float64)
        bond_history = np.empty(num_relax_steps + 1, dtype=np.float64)
        overlap_history = np.empty(num_relax_steps + 1, dtype=np.float64)
        if num_relax_steps > 0:
            loss_history[0] = np.nan
            best_history[0] = np.nan
            bond_history[0] = np.nan
            overlap_history[0] = np.nan
        else:
            loss_history[0] = 0.0
            best_history[0] = 0.0
            bond_history[0] = 0.0
            overlap_history[0] = 0.0

        progress = None
        if show_progress and num_relax_steps > 0:
            progress = _TextProgressBar(num_relax_steps, label="Motif cleanup", width=28)
            progress.update(0)

        positions = np.array(self.atoms.positions, copy=True)
        best_positions = np.array(positions, copy=True)
        best_loss = np.inf

        for step in range(1, num_relax_steps + 1):
            positions, bond_loss, overlap_loss = self._motif_vote_relax_step(
                positions,
                shell_target,
                motif_ids,
                graph,
                step_size=step_size,
                spring_step=spring_step,
                inertia=inertia,
                repulsion_step=repulsion_step,
                max_repulsion_distance=max_repulsion_distance,
            )
            self.atoms.positions[:] = self._wrap_positions(positions)
            self._rebuild_spatial_index()
            total_loss = float(bond_loss + overlap_weight * overlap_loss)
            if total_loss < best_loss:
                best_loss = total_loss
                best_positions = np.array(self.atoms.positions, copy=True)
            loss_history[step] = total_loss
            best_history[step] = best_loss
            bond_history[step] = float(bond_loss)
            overlap_history[step] = float(overlap_loss)
            if progress is not None:
                progress.update(step)

        if progress is not None:
            progress.close()

        self.atoms.positions[:] = self._wrap_positions(best_positions)
        self._rebuild_spatial_index()
        self.current_distribution = None
        self.current_cost = None
        self.mc_history = None
        self.shell_history = None
        self.last_temperature = None
        self.motif_graph = {
            **graph,
            "motif_ids": np.asarray(motif_ids, dtype=np.intp, copy=True),
        }
        self.motif_history = {
            "step": np.arange(num_relax_steps + 1, dtype=np.int32),
            "loss": loss_history,
            "best_loss": best_history,
            "bond_loss": bond_history,
            "overlap_loss": overlap_history,
        }

        summary = {
            "merge_distance_scale": float(merge_distance_scale),
            "num_relax_steps": int(num_relax_steps),
            "step_size": float(step_size),
            "spring_step": float(spring_step),
            "inertia": float(inertia),
            "repulsion_step": float(repulsion_step),
            "overlap_weight": float(overlap_weight),
            "unmatched_slots": int(graph["unmatched_slots"]),
            "matched_fraction": float(
                np.sum(graph["actual_degree"]) / max(float(np.sum(graph["target_degree"])), 1.0)
            ),
            "mean_target_degree": float(np.mean(graph["target_degree"])) if graph["target_degree"].size else 0.0,
            "mean_actual_degree": float(np.mean(graph["actual_degree"])) if graph["actual_degree"].size else 0.0,
            "initial_loss": float(loss_history[1]) if num_relax_steps > 0 else np.nan,
            "final_loss": float(loss_history[num_relax_steps]) if num_relax_steps > 0 else np.nan,
            "best_loss": float(best_loss) if np.isfinite(best_loss) else np.nan,
            "num_atoms": len(self.atoms),
            "cell_dim_angstroms": self.cell_dim_angstroms,
            "relative_density": self.relative_density,
        }
        if plot_history:
            self.plot_motif_training()
        return summary

    def shell_refine(
        self,
        shell_target: CoordinationShellTarget,
        num_steps: int = 4_000,
        *,
        move_scale: float | None = None,
        trials_per_step: int = 8,
        temperature: float = 0.0,
        selection_power: float = 1.5,
        count_weight: float = 2.0,
        radius_weight: float = 1.0,
        angle_weight: float = 8.0,
        overlap_weight: float = 20.0,
        recruit_cutoff_scale: float = 1.6,
        max_angle_step_deg: float = 20.0,
        full_eval_candidates: int = 3,
        plot_history: bool = False,
        show_progress: bool = True,
    ) -> dict[str, Any]:
        """Refine the random supercell using first-shell counts, radii, and angles.

        This optimizer does not use `target_g3`. It only tries to match the
        species-aware first-shell structure extracted from `shell_target`, which
        makes it a better teacher for later ML training data generation.
        """
        if not np.array_equal(np.asarray(shell_target.species, dtype=np.int64), self._species):
            raise ValueError("shell_target species must match the Supercell composition ordering.")
        num_steps = int(num_steps)
        if num_steps <= 0:
            raise ValueError("num_steps must be positive.")
        trials_per_step = int(trials_per_step)
        if trials_per_step <= 0:
            raise ValueError("trials_per_step must be positive.")
        temperature = float(temperature)
        if temperature < 0.0:
            raise ValueError("temperature must be non-negative.")
        selection_power = float(selection_power)
        if selection_power < 0.0:
            raise ValueError("selection_power must be non-negative.")
        if recruit_cutoff_scale <= 1.0:
            raise ValueError("recruit_cutoff_scale must be greater than 1.")
        if max_angle_step_deg <= 0.0:
            raise ValueError("max_angle_step_deg must be positive.")
        full_eval_candidates = int(full_eval_candidates)
        if full_eval_candidates <= 0:
            raise ValueError("full_eval_candidates must be positive.")

        if move_scale is None:
            valid_peaks = np.asarray(shell_target.pair_peak[shell_target.pair_mask], dtype=np.float64)
            if valid_peaks.size:
                move_scale = max(0.12 * float(np.median(valid_peaks)), 0.8 * self.measure_r_step)
            else:
                move_scale = float(self.measure_r_step)
        move_scale = float(move_scale)
        if move_scale <= 0.0:
            raise ValueError("move_scale must be positive.")

        shell_spatial_bin_size = float(max(self.measure_r_step, shell_target.max_pair_outer))
        if self.spatial_bin_size > shell_spatial_bin_size * 1.05:
            self.spatial_bin_size = shell_spatial_bin_size
            self._rebuild_spatial_index()

        num_atoms = len(self.atoms)
        local_loss = np.empty(num_atoms, dtype=np.float64)
        for atom_index in range(num_atoms):
            local_loss[atom_index] = self._shell_center_state(
                atom_index,
                shell_target,
                count_weight=count_weight,
                radius_weight=radius_weight,
                angle_weight=angle_weight,
                overlap_weight=overlap_weight,
                return_details=False,
            )
        total_loss = float(np.sum(local_loss))
        best_loss = float(total_loss)

        step_history = np.arange(num_steps + 1, dtype=np.int32)
        loss_history = np.empty(num_steps + 1, dtype=np.float64)
        best_history = np.empty(num_steps + 1, dtype=np.float64)
        accepted_history = np.zeros(num_steps + 1, dtype=np.float32)
        center_history = np.full(num_steps + 1, -1, dtype=np.int32)
        moved_history = np.full(num_steps + 1, -1, dtype=np.int32)
        mode_history = np.full(num_steps + 1, -1, dtype=np.int32)
        accepted_moves = 0
        mode_codes = {
            "random": 0,
            "overlap": 1,
            "count_excess": 2,
            "count_deficit": 3,
            "radius": 4,
            "angle": 5,
        }

        loss_history[0] = total_loss
        best_history[0] = best_loss

        progress = None
        if show_progress:
            progress = _TextProgressBar(num_steps, label="Shell refine", width=28)
            progress.update(0)

        any_accept = False
        for step in range(1, num_steps + 1):
            weights = np.power(np.maximum(local_loss, 1e-8), selection_power)
            weight_sum = float(np.sum(weights))
            if not np.isfinite(weight_sum) or weight_sum <= 0.0:
                weights = np.ones(num_atoms, dtype=np.float64) / max(num_atoms, 1)
            else:
                weights /= weight_sum

            center_index = int(self.rng.choice(num_atoms, p=weights))
            center_state = self._shell_center_state(
                center_index,
                shell_target,
                count_weight=count_weight,
                radius_weight=radius_weight,
                angle_weight=angle_weight,
                overlap_weight=overlap_weight,
            )
            candidates = self._build_shell_candidates(
                center_state,
                shell_target,
                move_scale=move_scale,
                trials_per_step=trials_per_step,
                recruit_cutoff_scale=float(recruit_cutoff_scale),
                max_angle_step_deg=float(max_angle_step_deg),
            )

            screened_candidates: list[tuple[float, int, np.ndarray, str]] = []
            for moved_atom, new_position, mode_name in candidates:
                screen_centers = (
                    np.array([center_index], dtype=np.intp)
                    if int(moved_atom) == center_index
                    else np.array([center_index, int(moved_atom)], dtype=np.intp)
                )
                old_screen = float(np.sum(local_loss[screen_centers]))
                new_screen = 0.0
                for affected_center in screen_centers:
                    new_screen += float(
                        self._shell_center_state(
                            int(affected_center),
                            shell_target,
                            moved_atom=int(moved_atom),
                            moved_position=np.asarray(new_position, dtype=np.float64),
                            count_weight=count_weight,
                            radius_weight=radius_weight,
                            angle_weight=angle_weight,
                            overlap_weight=overlap_weight,
                            return_details=False,
                        )
                    )
                screened_candidates.append(
                    (
                        float(new_screen - old_screen),
                        int(moved_atom),
                        np.asarray(new_position, dtype=np.float64),
                        mode_name,
                    )
                )

            screened_candidates.sort(key=lambda item: item[0])
            shortlist = screened_candidates[: min(full_eval_candidates, len(screened_candidates))]

            best_candidate: tuple[int, np.ndarray, str] | None = None
            best_candidate_losses: np.ndarray | None = None
            best_candidate_affected: np.ndarray | None = None
            best_delta = np.inf
            for _, moved_atom, new_position, mode_name in shortlist:
                old_position = np.array(self.atoms.positions[moved_atom], copy=True)
                affected = self._shell_affected_centers(
                    moved_atom,
                    old_position,
                    np.asarray(new_position, dtype=np.float64),
                    float(shell_target.max_pair_outer),
                )
                old_sum = float(np.sum(local_loss[affected]))
                new_losses = np.empty(affected.size, dtype=np.float64)
                for affected_idx, affected_center in enumerate(affected):
                    new_losses[affected_idx] = self._shell_center_state(
                        int(affected_center),
                        shell_target,
                        moved_atom=moved_atom,
                            moved_position=np.asarray(new_position, dtype=np.float64),
                            count_weight=count_weight,
                            radius_weight=radius_weight,
                            angle_weight=angle_weight,
                            overlap_weight=overlap_weight,
                            return_details=False,
                        )
                delta = float(np.sum(new_losses) - old_sum)
                if delta < best_delta:
                    best_delta = delta
                    best_candidate = (int(moved_atom), np.asarray(new_position, dtype=np.float64), mode_name)
                    best_candidate_losses = new_losses
                    best_candidate_affected = affected

            accepted = False
            moved_atom = -1
            mode_name = "random"
            if best_candidate is not None and best_candidate_losses is not None and best_candidate_affected is not None:
                moved_atom = int(best_candidate[0])
                new_position = best_candidate[1]
                mode_name = best_candidate[2]
                if temperature == 0.0:
                    accepted = best_delta <= 0.0
                else:
                    accepted = best_delta <= 0.0 or self.rng.random() < np.exp(
                        -best_delta / max(temperature, _EPS)
                    )
                if accepted:
                    any_accept = True
                    old_position = np.array(self.atoms.positions[moved_atom], copy=True)
                    self.atoms.positions[moved_atom] = new_position
                    self._update_spatial_index_for_atom(moved_atom, old_position, new_position)
                    local_loss[best_candidate_affected] = best_candidate_losses
                    total_loss += best_delta
                    best_loss = min(best_loss, total_loss)
                    accepted_moves += 1

            loss_history[step] = total_loss
            best_history[step] = best_loss
            accepted_history[step] = float(accepted)
            center_history[step] = center_index
            moved_history[step] = moved_atom
            mode_history[step] = int(mode_codes.get(mode_name, 0))
            if progress is not None:
                progress.update(step)

        if progress is not None:
            progress.close()

        self.shell_history = {
            "step": step_history,
            "loss": loss_history,
            "best_loss": best_history,
            "accepted": accepted_history,
            "center_index": center_history,
            "moved_atom": moved_history,
            "mode_code": mode_history,
        }

        if any_accept:
            self.current_distribution = None
            self.current_cost = None
            self.mc_history = None
            self.last_temperature = None

        summary = {
            "num_steps": num_steps,
            "move_scale": float(move_scale),
            "trials_per_step": trials_per_step,
            "temperature": float(temperature),
            "selection_power": float(selection_power),
            "count_weight": float(count_weight),
            "radius_weight": float(radius_weight),
            "angle_weight": float(angle_weight),
            "overlap_weight": float(overlap_weight),
            "full_eval_candidates": int(full_eval_candidates),
            "spatial_bin_size": float(self.spatial_bin_size),
            "accepted_moves": int(accepted_moves),
            "acceptance_rate": float(accepted_moves / max(num_steps, 1)),
            "initial_loss": float(loss_history[0]),
            "final_loss": float(loss_history[-1]),
            "best_loss": float(best_history[-1]),
            "cell_dim_angstroms": self.cell_dim_angstroms,
            "relative_density": self.relative_density,
            "num_atoms": num_atoms,
        }
        if plot_history:
            self.plot_shell_refine()
        return summary

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

        initial_nn, _ = self._nearest_neighbor_vectors(cutoff)
        progress = None
        if show_progress:
            progress = _TextProgressBar(num_steps, label="Nearest-neighbor repulsion", width=28)
            progress.update(0)

        for step in range(1, num_steps + 1):
            _, nearest_vec = self._nearest_neighbor_vectors(cutoff)
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

        self._rebuild_spatial_index()
        final_nn, _ = self._nearest_neighbor_vectors(cutoff)

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
        _snapshot_stride_accepted: int | None = None,
        _snapshot_callback: Callable[[dict[str, Any]], None] | None = None,
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
        if _snapshot_stride_accepted is not None and int(_snapshot_stride_accepted) <= 0:
            raise ValueError("_snapshot_stride_accepted must be positive when provided.")

        self.measure_g3()
        if self.current_cost is None:
            self._initialize_mc_state()

        num_atoms = len(self.atoms)
        if self.mc_history is None:
            history_step_offset = 0
            run_index = 0
            best_cost_seed = float(self.current_cost)
        else:
            history_step_offset = int(self.mc_history["step"][-1])
            run_index = int(self.mc_history.get("run_index", np.zeros(1, dtype=np.int32))[-1]) + 1
            best_cost_seed = min(float(self.current_cost), float(self.mc_history["best_cost"][-1]))

        steps_local = np.arange(num_steps + 1, dtype=np.int32)
        steps = history_step_offset + steps_local
        cost_history = np.empty(num_steps + 1, dtype=np.float64)
        best_history = np.empty(num_steps + 1, dtype=np.float64)
        accepted_history = np.zeros(num_steps + 1, dtype=np.float32)
        attempted_history = np.zeros(num_steps + 1, dtype=np.float32)
        repelled_history = np.zeros(num_steps + 1, dtype=np.float32)
        unresolved_history = np.zeros(num_steps + 1, dtype=np.float32)
        run_history = np.full(num_steps + 1, run_index, dtype=np.int32)
        step_in_run_history = steps_local.astype(np.int32, copy=False)
        jump_size_history = np.full(num_steps + 1, float(jump_size), dtype=np.float32)
        temperature_history = np.full(num_steps + 1, float(temperature), dtype=np.float32)

        current_cost = float(self.current_cost)
        best_cost = float(best_cost_seed)
        cost_history[0] = current_cost
        best_history[0] = best_cost
        accepted_moves_total = 0
        attempted_moves_total = 0

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
                attempted_moves_total += 1
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

                    delta_cost = self._weighted_delta_cost(delta_g3_idx, delta_g3_val)

                    if temperature == 0.0:
                        accepted = delta_cost <= 0.0
                    else:
                        accepted = delta_cost <= 0.0 or self.rng.random() < np.exp(
                            -delta_cost / max(temperature, _EPS)
                        )

                    if accepted:
                        self.atoms.positions[atom_index] = new_position
                        self._update_spatial_index_for_atom(atom_index, old_position, new_position)
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
                        accepted_moves_total += 1
                        if (
                            _snapshot_callback is not None
                            and _snapshot_stride_accepted is not None
                            and accepted_moves_total % int(_snapshot_stride_accepted) == 0
                        ):
                            _snapshot_callback(
                                {
                                    "step": step,
                                    "accepted_moves": accepted_moves_total,
                                    "attempted_moves": attempted_moves_total,
                                    "current_cost": current_cost,
                                }
                            )

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

        new_history = {
            "step": steps.astype(np.int32, copy=False),
            "step_in_run": step_in_run_history,
            "run_index": run_history,
            "cost": cost_history.astype(np.float64),
            "best_cost": best_history.astype(np.float64),
            "accepted": accepted_history,
            "attempted": attempted_history,
            "repelled": repelled_history,
            "unresolved": unresolved_history,
            "jump_size": jump_size_history,
            "temperature": temperature_history,
        }
        if self.mc_history is None:
            self.mc_history = new_history
        else:
            self.mc_history = {
                key: np.concatenate(
                    [
                        np.asarray(self.mc_history[key]),
                        np.asarray(value)[1:],
                    ]
                )
                for key, value in new_history.items()
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
            "g3_weight_r_scale": float(self.g3_weight_r_scale),
            "g3_weight_exponent": float(self.g3_weight_exponent),
            "g3_weight_floor": float(self.g3_weight_floor),
            "final_cost": float(current_cost),
            "best_cost": float(best_cost),
            "cell_dim_angstroms": self.cell_dim_angstroms,
            "relative_density": self.relative_density,
            "measure_r_max": self.measure_r_max,
            "measure_r_step": self.measure_r_step,
            "measure_phi_num_bins": self.measure_phi_num_bins,
            "history_total_points": int(self.mc_history["step"].size),
            "history_final_step": int(self.mc_history["step"][-1]),
            "history_num_runs": int(self.mc_history["run_index"][-1] + 1),
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
            f"measure_r_max={self.measure_r_max:.3f}, "
            f"g3_weight_r_scale={self.g3_weight_r_scale:.3f}, best_score={self.best_score})"
        )
