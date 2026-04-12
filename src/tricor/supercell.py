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

    # ------------------------------------------------------------------
    # generate: unified structure generation
    # ------------------------------------------------------------------

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

        * **Liquid** — ``grain_size=None, phi_broadening=25``:
          only nearest-neighbor distances enforced, angles loosely constrained.
        * **Amorphous** — ``grain_size=4, r_broadening=0.2, phi_broadening=12``:
          short-range order with tunable distance and angle sharpness.
        * **Short-range order** — ``grain_size=12, crystalline_fraction=0.5``:
          small crystalline clusters in an amorphous matrix.
        * **Mixed** — ``grain_size=18, crystalline_fraction=0.5``:
          50 % crystalline grains, 50 % amorphous fill.
        * **Nanocrystalline** — ``grain_size=25, crystalline_fraction=1.0``:
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
        use_grains = grain_size is not None and float(grain_size) > 0.0
        user_grain_size = float(grain_size) if use_grains else 0.0

        if use_grains:
            min_grain_size = pair_peak_max * 3.0
            grain_size_clamped = max(float(grain_size), min_grain_size)

            boundary_loss = pair_peak_max * 0.75
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
            target_r_min = max(user_grain_size * 0.4, max_pair_outer + 1.0)
            target_r_max = max(user_grain_size * 0.7, target_r_min + 2.0)
        else:
            target_r_min = max_pair_outer
            target_r_max = target_r_min + 1.5

        # Clamp to g3 grid range
        target_r_max = min(target_r_max, g3_r_max - r_step)
        target_r_min = min(target_r_min, target_r_max - r_step)

        # Build target distribution with boundary-aware blur
        if use_grains:
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

    def __repr__(self) -> str:
        atom_count = len(self.atoms)
        return (
            f"Supercell(label={self.label!r}, cell_dim_angstroms={self.cell_dim_angstroms}, "
            f"atoms={atom_count}, relative_density={self.relative_density:.3f}, "
            f"measure_r_max={self.measure_r_max:.3f}, "
            f"g3_weight_r_scale={self.g3_weight_r_scale:.3f}, best_score={self.best_score})"
        )
