"""Supercell class — disordered atomic structure generation and optimization.

The heavy lifting is split across mixin modules:
    _grain.py        — Voronoi grain construction
    _shell_relax.py  — vectorized spring-network relaxation
    _plotting.py     — visualization (plot_structure, plot_g3_compare, …)
    _monte_carlo.py  — Monte Carlo engine, spatial indexing, teacher rollout
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from ase.atoms import Atoms

from .g3 import G3Distribution, _EPS
from .shells import CoordinationShellTarget

from ._grain import _GrainMixin
from ._shell_relax import _ShellRelaxMixin
from ._plotting import _PlottingMixin
from ._monte_carlo import _MonteCarloMixin


class Supercell(_GrainMixin, _ShellRelaxMixin, _PlottingMixin, _MonteCarloMixin):
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
            target grid during initialization.
        plot_g3_compare
            If `True`, immediately display an interactive comparison between the
            random supercell and the target distribution.
        label
            Human-readable label for summaries and repr output.
        rng_seed
            Optional random seed for reproducible initialization.
        g3_weight_r_scale
            Characteristic radius in Angstrom that controls how strongly the
            Monte Carlo cost prioritizes short-range `g3` bins.
        g3_weight_exponent
            Power-law exponent used in the radial weighting curve for the
            full-`g3` cost.
        g3_weight_floor
            Minimum relative weight assigned to the farthest radial bins.
        spatial_bin_size
            Approximate spatial-hash bin size in Angstrom.
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

        self.reference_atoms = self._to_orthogonal_cell(
            self.target_distribution.atoms,
        )
        self._shell_target: Any | None = None
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

    # ------------------------------------------------------------------
    # Cell geometry utilities
    # ------------------------------------------------------------------

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
        """Build an orthogonal supercell with the requested dimensions.

        Always returns a diagonal cell matrix regardless of the
        reference crystal's lattice vectors.
        """
        return np.diag(np.asarray(self.cell_dim_angstroms, dtype=np.float64))

    @staticmethod
    def _to_orthogonal_cell(atoms: Atoms) -> Atoms:
        """Convert to an orthogonal (diagonal) cell if needed.

        Tiles the primitive cell and wraps into the smallest
        axis-aligned box, removing duplicates.  If the cell is
        already orthogonal, returns a copy unchanged.
        """
        cell = np.asarray(atoms.cell.array, dtype=np.float64)
        off_diag = cell - np.diag(np.diag(cell))
        if np.allclose(off_diag, 0, atol=1e-6):
            return atoms.copy()

        # Find the smallest orthogonal box: use the max absolute
        # Cartesian extent of each lattice vector column.
        # For FCC [[0,a/2,a/2],[a/2,0,a/2],[a/2,a/2,0]]:
        # max per column = [a/2, a/2, a/2], so box = [a, a, a].
        a_orth = 2.0 * np.max(np.abs(cell), axis=0)
        orth_cell = np.diag(a_orth)
        orth_inv = np.linalg.inv(orth_cell)

        # Tile generously
        ref_lengths = np.linalg.norm(cell, axis=1)
        n_reps = np.ceil(a_orth / np.maximum(ref_lengths, 1e-10)).astype(int) + 1
        shifts = []
        for ix in range(-n_reps[0], n_reps[0] + 1):
            for iy in range(-n_reps[1], n_reps[1] + 1):
                for iz in range(-n_reps[2], n_reps[2] + 1):
                    shifts.append([ix, iy, iz])
        shifts = np.array(shifts, dtype=np.float64)
        shift_cart = shifts @ cell

        pos = atoms.positions
        nums = atoms.numbers
        tiled_pos = (pos[None, :, :] + shift_cart[:, None, :]).reshape(-1, 3)
        tiled_nums = np.tile(nums, len(shifts))

        # Keep atoms inside the orthogonal box
        frac = tiled_pos @ orth_inv
        eps = 1e-6
        inside = np.all((frac >= -eps) & (frac < 1.0 - eps), axis=1)
        tiled_pos = tiled_pos[inside]
        tiled_nums = tiled_nums[inside]

        # Remove duplicates
        frac_inside = tiled_pos @ orth_inv
        frac_rounded = np.round(frac_inside, decimals=5)
        _, unique_idx = np.unique(frac_rounded, axis=0, return_index=True)
        tiled_pos = tiled_pos[unique_idx]
        tiled_nums = tiled_nums[unique_idx]

        result = Atoms(
            numbers=tiled_nums,
            positions=tiled_pos,
            cell=orth_cell,
            pbc=atoms.pbc,
        )
        return result

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

    # ------------------------------------------------------------------
    # generate: unified structure generation
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Recommended presets for Si (diamond cubic)
    # ------------------------------------------------------------------

    PRESETS: dict[str, dict[str, Any]] = {
        "liquid": dict(
            relative_density=0.96, num_steps=100,
            grain_size=None,
            bond_weight=0.4, angle_weight=0.5,
            repulsion_weight=0.5,
            hard_core_scale=0.75, nonbond_push_scale=0.7,
        ),
        "amorphous": dict(
            relative_density=0.96, num_steps=150,
            grain_size=6.0,
            bond_weight=1.2, angle_weight=0.6,
            hard_core_scale=0.9, nonbond_push_scale=0.8,
            displacement_sigma=0.08,
        ),
        "SRO": dict(
            relative_density=0.96, num_steps=200,
            grain_size=10.0,
            bond_weight=2.2, angle_weight=1.0,
            hard_core_scale=0.95, nonbond_push_scale=0.9,
            displacement_sigma=0.04,
        ),
        "MRO": dict(
            relative_density=0.96, num_steps=150,
            grain_size=13.0,
            bond_weight=1.9, angle_weight=0.9,
            hard_core_scale=0.95, nonbond_push_scale=0.9,
            displacement_sigma=0.04,
        ),
        "MRO_more": dict(
            relative_density=0.96, num_steps=150,
            grain_size=18.0,
            bond_weight=2.0, angle_weight=1.0,
            hard_core_scale=0.95, nonbond_push_scale=0.9,
            displacement_sigma=0.04,
        ),
        "nanocrystalline_10": dict(
            relative_density=0.96, num_steps=200,
            grain_size=15.0,
            bond_weight=2.8, angle_weight=1.3,
            displacement_sigma=0.02,
        ),
        "nanocrystalline_20": dict(
            relative_density=0.96, num_steps=150,
            grain_size=20.0,
            bond_weight=3.0, angle_weight=1.5,
            displacement_sigma=0.02,
        ),
    }

    def generate(
        self,
        shell_target: "CoordinationShellTarget",
        num_steps: int = 200,
        *,
        grain_size: float | None = None,
        crystalline_fraction: float = 1.0,
        bond_weight: float = 1.0,
        angle_weight: float = 0.5,
        repulsion_weight: float = 3.0,
        hard_core_scale: float = 1.0,
        nonbond_push_scale: float = 1.0,
        displacement_sigma: float = 0.0,
        show_progress: bool = True,
        **shell_relax_kwargs: Any,
    ) -> dict[str, Any]:
        """Generate a disordered supercell from liquid to nanocrystalline.

        Covers the full spectrum of disorder by combining Voronoi grain
        construction with spring-network relaxation.  See
        :attr:`PRESETS` for recommended parameter sets for Si.

        Parameters
        ----------
        shell_target
            First-shell coordination targets from the reference crystal.
        num_steps
            Number of relaxation sweeps.
        grain_size
            Diameter of crystalline grains in Angstrom.  ``None`` means
            no grains — start from random positions (liquid/amorphous).
        crystalline_fraction
            Volume fraction filled by crystalline grains (0–1).  Only
            used when *grain_size* is set.  The remaining volume is
            filled with random (amorphous) positions.
        bond_weight
            Harmonic spring strength pulling bonded neighbours toward
            the target bond distance.  Larger = tighter distances.
        angle_weight
            Spring strength pushing bond angles toward the target angle.
            Larger = tighter angles.  Near-zero = liquid-like freedom.
        displacement_sigma
            Gaussian displacement (Angstrom) applied to atoms within
            crystalline grains as thermal broadening.  0 = no jitter.
        show_progress
            Display a text progress bar.
        **shell_relax_kwargs
            Additional keyword arguments forwarded to :meth:`shell_relax`
            (e.g. ``repulsion_weight``, ``hard_core_scale``, ``step_size``).

        Returns
        -------
        dict[str, Any]
            Summary dict with regime, construction parameters, and
            relaxation loss values.
        """
        self._shell_target = shell_target
        pair_peak = np.asarray(shell_target.pair_peak, dtype=np.float64)
        pair_peak_max = float(np.max(pair_peak[pair_peak > _EPS])) if np.any(pair_peak > _EPS) else 2.5

        # --- construct atoms ---
        use_grains = grain_size is not None and float(grain_size) > 0.0

        if use_grains:
            self.atoms = self._build_grain_atoms(
                shell_target,
                grain_size=float(grain_size),
                crystalline_fraction=crystalline_fraction,
                displacement_sigma=displacement_sigma,
            )

            # Refresh cached arrays after rebuilding atoms
            self._cell_matrix = np.asarray(self.atoms.cell.array, dtype=np.float64)
            self._cell_inverse = np.linalg.inv(self._cell_matrix)
            self._atom_species_index = np.searchsorted(
                self._species, self.atoms.numbers,
            )
            self._rebuild_spatial_index()

        # --- relax ---
        summary = self.shell_relax(
            shell_target,
            num_steps=num_steps,
            bond_weight=bond_weight,
            angle_weight=angle_weight,
            repulsion_weight=repulsion_weight,
            hard_core_scale=hard_core_scale,
            nonbond_push_scale=nonbond_push_scale,
            show_progress=show_progress,
            **shell_relax_kwargs,
        )

        # --- summary ---
        ref_density = len(self.target_distribution.atoms) / max(
            float(self.target_distribution.atoms.cell.volume), _EPS,
        )
        actual_density = len(self.atoms) / max(float(self.atoms.cell.volume), _EPS)
        actual_relative = actual_density / max(ref_density, _EPS)

        if use_grains:
            summary["regime"] = "nanocrystalline" if crystalline_fraction >= 0.9 else "mixed"
            summary["n_grains"] = int(self.atoms.info.get("n_grains", 0))
            summary["grain_size"] = float(grain_size)
            summary["crystalline_fraction"] = crystalline_fraction
        else:
            summary["regime"] = "amorphous"
        summary["num_atoms"] = len(self.atoms)
        summary["target_density"] = self.relative_density
        summary["actual_density"] = float(f"{actual_relative:.4f}")

        return summary

    def __repr__(self) -> str:
        atom_count = len(self.atoms)
        return (
            f"Supercell(label={self.label!r}, cell_dim_angstroms={self.cell_dim_angstroms}, "
            f"atoms={atom_count}, relative_density={self.relative_density:.3f}, "
            f"measure_r_max={self.measure_r_max:.3f}, "
            f"g3_weight_r_scale={self.g3_weight_r_scale:.3f}, best_score={self.best_score})"
        )
