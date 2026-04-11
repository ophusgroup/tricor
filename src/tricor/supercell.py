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
        self.shell_relax_history: dict[str, np.ndarray] | None = None
        self._grain_ids: np.ndarray | None = None
        self._grain_seeds: np.ndarray | None = None
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

        # --- tile reference crystal once to fill the box ---
        ref_cell = np.asarray(self.reference_atoms.cell.array, dtype=np.float64)
        ref_pos = self.reference_atoms.positions.copy()
        ref_numbers = self.reference_atoms.numbers.copy()

        box_lengths = np.array(self.cell_dim_angstroms, dtype=np.float64)
        ref_lengths = np.linalg.norm(ref_cell, axis=1)
        ref_lengths = np.maximum(ref_lengths, _EPS)
        n_reps = np.ceil(box_lengths / ref_lengths).astype(int)
        n_reps = np.maximum(n_reps, 1)

        shifts = []
        for ix in range(n_reps[0]):
            for iy in range(n_reps[1]):
                for iz in range(n_reps[2]):
                    shifts.append([ix, iy, iz])
        shifts = np.array(shifts, dtype=np.float64)
        shift_cart = shifts @ ref_cell

        tiled_pos = (ref_pos[None, :, :] + shift_cart[:, None, :]).reshape(-1, 3)
        tiled_numbers = np.tile(ref_numbers, len(shifts))

        # Wrap into supercell box
        frac_tiled = tiled_pos @ cell_inv
        frac_tiled %= 1.0
        tiled_pos = frac_tiled @ cell_mat
        n_tiled = len(tiled_pos)

        # --- chunked Voronoi assignment ---
        # For many seeds, the (n_atoms, n_seeds, 3) array can be huge.
        # Process atoms in chunks to bound memory at ~100 MB.
        max_chunk_elements = 25_000_000  # ~100 MB for float64 × 3
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

        # Assign each tiled atom to its nearest seed
        nearest_seed = _chunked_voronoi(tiled_pos)

        # Keep only atoms in crystalline cells
        in_crystalline = is_crystalline_cell[nearest_seed]
        cryst_pos = tiled_pos[in_crystalline].copy()
        cryst_numbers = tiled_numbers[in_crystalline].copy()
        cryst_seeds = nearest_seed[in_crystalline]

        # Rotate each grain's atoms around its seed center
        for ig in range(n_seeds):
            if not is_crystalline_cell[ig]:
                continue
            mask_ig = cryst_seeds == ig
            if not np.any(mask_ig):
                continue
            delta = cryst_pos[mask_ig] - seed_cart[ig]
            frac_d = delta @ cell_inv
            frac_d -= np.rint(frac_d)
            delta = frac_d @ cell_mat
            cryst_pos[mask_ig] = seed_cart[ig] + delta @ rotations[ig].T

        # Wrap rotated positions
        frac_c = cryst_pos @ cell_inv
        frac_c %= 1.0
        cryst_pos = frac_c @ cell_mat

        all_positions: list[np.ndarray] = [cryst_pos]
        all_numbers: list[np.ndarray] = [cryst_numbers]
        all_grain_ids: list[np.ndarray] = [cryst_seeds.astype(np.intp)]

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

        # --- remove duplicate/overlapping atoms ---
        # Use the shell inner boundary as overlap threshold to prevent
        # any bonds shorter than the physical minimum.
        pair_inner_arr = np.asarray(shell_target.pair_inner, dtype=np.float64)
        pair_hard_min = np.asarray(shell_target.pair_hard_min, dtype=np.float64)
        overlap_thresh = float(np.max(np.maximum(pair_inner_arr, pair_hard_min)))

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
        repulsion_weight: float = 3.0,
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
        # Hard core: use max of pair_hard_min and pair_inner to
        # prevent any bonds shorter than the shell inner boundary.
        pair_inner = np.asarray(shell_target.pair_inner, dtype=np.float64)
        hard_core = np.maximum(pair_hard_min, pair_inner)
        mask_zero = hard_core < _EPS
        hard_core[mask_zero] = 0.4 * pair_peak[mask_zero]
        global_floor = float(np.min(pair_peak[pair_peak > _EPS])) * 0.4 if np.any(pair_peak > _EPS) else 1.0
        hard_core[hard_core < _EPS] = global_floor
        # Non-bonded atoms are pushed beyond this radius to create a
        # clean gap between the first and second coordination shells.
        # For Si, 2nd shell is at ~3.84Å (sqrt(8/3) * pair_peak).
        # Push non-bonded atoms to at least 1.5x pair_peak to
        # eliminate close-packed triplets from nearby non-bonded pairs.
        nonbond_push = pair_peak * 1.5
        nonbond_push[nonbond_push < _EPS] = float(np.max(pair_peak)) * 1.5

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
                n_seeds_local = len(grain_seeds)
                boundary_width = float(np.max(pair_peak)) * 0.5

                is_boundary[:] = True  # default boundary

                # Process in chunks to bound memory
                _bchunk = max(1, 25_000_000 // max(n_seeds_local, 1))
                for start in range(0, num_atoms, _bchunk):
                    end = min(start + _bchunk, num_atoms)
                    delta = pos[start:end, None, :] - grain_seeds[None, :, :]
                    frac_d = delta @ cell_inv
                    frac_d -= np.rint(frac_d)
                    cart_d = frac_d @ cell_mat
                    dist_chunk = np.sqrt(np.sum(cart_d ** 2, axis=2))

                    for ia_local in range(end - start):
                        ia = start + ia_local
                        gid = grain_ids[ia]
                        if gid < 0:
                            continue
                        dist_own = dist_chunk[ia_local, gid]
                        dists_row = dist_chunk[ia_local].copy()
                        dists_row[gid] = np.inf
                        dist_other = float(np.min(dists_row))
                        if (dist_other - dist_own) * 0.5 > boundary_width:
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
                # Strong hard core: linear + quadratic for stiff wall
                hr = hard_ratio[hard_mask] - 1.0
                hard_mag[hard_mask] = repulsion_weight * 4.0 * (hr + hr ** 2)

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
                # Linear + quadratic: strong near boundary, stronger close in
                pr = push_ratio[nonbond_mask] - 1.0
                nonbond_mag[nonbond_mask] = repulsion_weight * (pr + pr ** 2)

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
        colormap: str = "Reds",
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
            # Camera at +x (azim=0): larger rotated-x = closer.
            # Use squared depth for stronger contrast: close bonds pop,
            # far bonds fade to near-white via the Reds colormap.
            if np.any(cryst_mask):
                segs_cr = list(zip(bs_r[cryst_mask], be_r[cryst_mask]))
                mid_x_rot = 0.5 * (bs_r[cryst_mask, 0] + be_r[cryst_mask, 0])
                norm_depth = (mid_x_rot + extent) / max(2.0 * extent, _EPS)
                norm_depth = np.clip(norm_depth, 0, 1)
                # Square for stronger contrast: far → ~0 (white), close → 1 (bold)
                norm_sq = norm_depth ** 2
                # Colormap range 0.05–0.95 to avoid pure white and clipping
                cryst_colors = cmap(0.05 + 0.9 * norm_sq)
                # Linewidth: 0.2 at back, 2.5 at front
                cryst_lw = 0.2 + 2.3 * norm_sq
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

            ax.set_xlim(-extent, extent)
            ax.set_ylim(-extent, extent)
            ax.set_zlim(-extent, extent)
            ax.set_box_aspect([1, 1, 1])
            ax.view_init(elev=elevation, azim=0)  # azim fixed; we rotate data
            ax.axis("off")
            fig.subplots_adjust(left=-0.05, right=1.05, bottom=-0.05, top=1.05)
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

        # --- enforce minimum grain size ---
        # Multi-shell order requires grains large enough to contain
        # complete 2nd/3rd coordination shells plus a boundary buffer.
        use_grains = grain_size is not None and float(grain_size) > 0.0
        user_grain_size = float(grain_size) if use_grains else 0.0

        if use_grains:
            second_shell_est = max_pair_outer * 1.4
            min_grain_size = 2.0 * (second_shell_est + pair_peak_max)
            grain_size_clamped = max(float(grain_size), min_grain_size)

            # Inflate to compensate for boundary disorder:
            # effective crystalline core ≈ construction_size - 2*boundary_loss
            boundary_loss = pair_peak_max * 1.5
            construction_grain_size = grain_size_clamped + 2.0 * boundary_loss

            disp_sigma = float(r_broadening) if (r_broadening is not None and r_broadening > _EPS) else 0.0

            self.atoms = self._build_grain_atoms(
                shell_target,
                grain_size=construction_grain_size,
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
        if use_grains:
            # Use the user's requested grain_size for the target
            # (not the inflated construction size)
            target_r_min = user_grain_size * 0.4
            target_r_max = user_grain_size * 0.7
        else:
            # No grains: only 1st shell is enforced, tight fade to random
            target_r_min = max_pair_outer
            target_r_max = target_r_min + 1.5

        # Clamp to g3 grid range
        target_r_max = min(target_r_max, g3_r_max - r_step)
        target_r_min = min(target_r_min, target_r_max - r_step)

        # Build target distribution from the raw measured g3.
        # The target blur must account for grain boundary disorder
        # (which is always present regardless of broadening params).
        # Minimum blur: inversely proportional to grain size — smaller
        # grains have more boundary fraction, so more blur.
        if use_grains:
            # Boundary disorder: gentle minimum blur, mostly angular
            boundary_blur_r = 0.15 * pair_peak_max / max(user_grain_size, pair_peak_max)
            boundary_blur_phi = 10.0 * pair_peak_max / max(user_grain_size, pair_peak_max)
            user_r = float(r_broadening) * 0.3 if (r_broadening is not None and r_broadening > _EPS) else 0.0
            user_phi = float(phi_broadening) * 0.3 if (phi_broadening is not None and phi_broadening > _EPS) else 0.0
            target_r_sigma = max(user_r, boundary_blur_r)
            target_phi_sigma = max(user_phi, boundary_blur_phi)
        else:
            target_r_sigma = float(r_broadening) * 0.3 if (r_broadening is not None and r_broadening > _EPS) else None
            target_phi_sigma = float(phi_broadening) * 0.3 if (phi_broadening is not None and phi_broadening > _EPS) else None
        self.target_distribution = self._raw_distribution.target_g3(
            target_r_min=target_r_min,
            target_r_max=target_r_max,
            r_sigma=target_r_sigma,
            r_sigma_at=pair_peak_max,
            phi_sigma_deg=target_phi_sigma,
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
            summary["grain_size"] = user_grain_size
            summary["construction_grain_size"] = construction_grain_size
            summary["crystalline_fraction"] = crystalline_fraction
        else:
            summary["regime"] = "amorphous"
        summary["r_broadening"] = r_broadening
        summary["phi_broadening"] = phi_broadening
        summary["target_r_min"] = target_r_min
        summary["target_r_max"] = target_r_max

        return summary

    # ------------------------------------------------------------------
    # Monte Carlo and repulsion utilities
    # ------------------------------------------------------------------

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
