"""Supercell class - disordered atomic structure generation and optimization.

The heavy lifting is split across mixin modules:
    _grain.py - Voronoi grain construction
    _shell_relax.py - vectorized spring-network relaxation
    _plotting.py - visualization (plot_structure, plot_g3_compare, …)
    _monte_carlo.py - Monte Carlo engine, spatial indexing, teacher rollout
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
from ._resample import _ResampleMixin


class Supercell(
    _GrainMixin,
    _ShellRelaxMixin,
    _PlottingMixin,
    _MonteCarloMixin,
    _ResampleMixin,
):
    """Random supercell scaffold driven by a target :class:`G3Distribution`."""

    def __init__(
        self,
        distribution: G3Distribution,
        cell_dim_angstroms: float | Sequence[float],
        *,
        relative_density: float = 0.96,
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
        # Optional override: when a composite shell target has more
        # virtual species than the atomic-number species (e.g. sp2_C
        # and sp3_C sharing atomic number 6), the relaxer and the
        # polyhedra-friendly plotting paths consult this array instead
        # of ``_atom_species_index``.  Set from ``_build_grain_atoms``
        # when ``grain_sources`` is supplied, or manually via
        # ``Supercell.generate(..., atom_species_index=...)``.
        self._atom_shell_species_index: np.ndarray | None = None
        self._grain_source: np.ndarray | None = None
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
        relative_density: float = 0.96,
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

    def _normalize_cell_dim_angstroms(self, cell_dim_angstroms):
        """Validate and normalize the requested supercell lattice.

        Accepts:
            - a scalar (cubic box),
            - a length-3 sequence of positive edge lengths (orthogonal box),
            - a 3 x 3 array of cell vectors (general triclinic box).
        """
        if isinstance(cell_dim_angstroms, (int, float)):
            if float(cell_dim_angstroms) <= 0:
                raise ValueError("cell_dim_angstroms must be positive.")
            length = float(cell_dim_angstroms)
            return (length, length, length)

        arr = np.asarray(cell_dim_angstroms, dtype=float)
        if arr.ndim == 2 and arr.shape == (3, 3):
            if abs(np.linalg.det(arr)) < 1e-8:
                raise ValueError("cell_dim_angstroms 3x3 matrix is singular.")
            return arr.copy()

        if arr.ndim == 1 and arr.shape[0] == 3:
            if np.any(arr <= 0):
                raise ValueError(
                    "cell_dim_angstroms edge lengths must be positive."
                )
            return tuple(float(v) for v in arr)

        raise ValueError(
            "cell_dim_angstroms must be a scalar, a length-3 sequence, "
            "or a 3x3 matrix of cell vectors.",
        )

    def _build_supercell_cell(self) -> np.ndarray:
        """Build the supercell cell matrix.

        ``cell_dim_angstroms`` can be supplied as either
        - a 3-tuple of edge lengths (orthogonal cell), or
        - a full 3x3 array of cell vectors (rows = cell vectors).

        The second form lets a hexagonal / triclinic reference be tiled
        into a supercell that shares its lattice geometry so that quartz,
        etc., tile cleanly through the periodic boundaries.
        """
        dim = np.asarray(self.cell_dim_angstroms, dtype=np.float64)
        if dim.ndim == 1 and dim.shape[0] == 3:
            return np.diag(dim)
        if dim.ndim == 2 and dim.shape == (3, 3):
            return dim.copy()
        raise ValueError(
            f"cell_dim_angstroms must be shape (3,) or (3, 3), got {dim.shape!r}"
        )

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
            num_steps=100,
            grain_size=None,
            bond_weight=0.4, angle_weight=0.5,
            repulsion_weight=0.5,
            hard_core_scale=0.75, nonbond_push_scale=0.7,
        ),
        "amorphous": dict(
            num_steps=150,
            grain_size=6.0,
            bond_weight=1.2, angle_weight=0.6,
            repulsion_weight=1.5,
            hard_core_scale=0.9, nonbond_push_scale=0.5,
            displacement_sigma=0.08,
        ),
        "SRO": dict(
            num_steps=200,
            grain_size=10.0,
            bond_weight=2.2, angle_weight=1.0,
            repulsion_weight=2.0,
            hard_core_scale=0.95, nonbond_push_scale=0.6,
            displacement_sigma=0.04,
        ),
        "MRO": dict(
            num_steps=150,
            grain_size=13.0,
            bond_weight=1.9, angle_weight=0.9,
            repulsion_weight=2.5,
            hard_core_scale=0.95, nonbond_push_scale=0.7,
            displacement_sigma=0.04,
        ),
        "LRO": dict(
            num_steps=150,
            grain_size=18.0,
            bond_weight=2.0, angle_weight=1.0,
            hard_core_scale=0.95, nonbond_push_scale=0.9,
            displacement_sigma=0.04,
        ),
        "nanocrystalline": dict(
            num_steps=150,
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
        atom_species_index: np.ndarray | None = None,
        grain_sources: "list[dict] | None" = None,
        # Build-time grain-orientation refinement (recommended for
        # directional-bond materials like Si — see
        # :meth:`refine_initial_orientations`).  ``True`` runs a
        # cheap topology-free coordinate-descent over per-grain
        # rotations BEFORE the FIRE quench; ``False`` skips it (the
        # historical default).
        refine_orientations: bool = False,
        refine_orientations_kwargs: "dict | None" = None,
        show_progress: bool = True,
        # ─── ML backend ──────────────────────────────────────────────
        backend: str = "fire",
        ml_model: "Any" = None,
        ml_fire_cleanup_steps: int = 0,
        ml_chunk_size: int | None = None,
        # When >0, run the EGNN as an *iterative* next-step predictor
        # (model trained on TricorMLStepDataset).  Each forward pass
        # applies ~one FIRE step's worth of relaxation; the loop runs
        # ``ml_iterative_steps`` times.  Replaces the one-shot
        # ``predict_positions`` path inside the ML backend.
        ml_iterative_steps: int = 0,
        ml_iterative_momentum: float = 0.0,
        ml_iterative_step_clip: float | None = None,
        # Repulsion-projection sweeps applied AFTER each EGNN iteration
        # to push any sub-hard-core pairs apart, keeping the
        # configuration physical between steps so the model stays in
        # its training distribution.  ``0`` (default) disables.
        ml_repulsion_iters_per_step: int = 0,
        # One-shot repulsion cleanup after the full iter loop — drives
        # any residual sub-NN bond count to zero.  10–20 is typical.
        ml_final_repulsion_iters: int = 0,
        # ────────────────────────────────────────────────────────────
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
            no grains - start from random positions (liquid/amorphous).
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
        backend
            Which relaxation kernel to use after the (optional) Voronoi
            tile + orientation refinement.  One of:

            - ``"fire"`` *(default)* — the classical spring-network FIRE
              quench (``shell_relax``).  Bit-identical to historical
              behaviour.
            - ``"ml"`` — single forward pass of the per-material EGNN
              given by ``ml_model``.  Skips ``refine_orientations`` and
              ``shell_relax`` entirely.  Use when you want the speed.
            - ``"ml+fire"`` — ML forward pass, then
              ``ml_fire_cleanup_steps`` of classical FIRE as a safety net
              (verifies bond / angle / repulsion losses, catches edge
              cases where the ML model produced a sub-NN overlap).
        ml_model
            Either a path to an EGNN checkpoint (loaded via
            :func:`tricor.ml.load_model`) or a pre-loaded model object.
            Required when ``backend != "fire"``.
        ml_fire_cleanup_steps
            How many FIRE iterations to run after the ML forward pass
            when ``backend == "ml+fire"``.  Ignored otherwise.
            Recommended values: 5–30 depending on how strict the
            downstream acceptance gate is.
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
                grain_sources=grain_sources,
            )

            # Refresh cached arrays after rebuilding atoms
            self._cell_matrix = np.asarray(self.atoms.cell.array, dtype=np.float64)
            self._cell_inverse = np.linalg.inv(self._cell_matrix)
            self._atom_species_index = np.searchsorted(
                self._species, self.atoms.numbers,
            )
            self._rebuild_spatial_index()
            # If the caller passed an explicit per-atom virtual species
            # index, it takes precedence over whatever _build_grain_atoms
            # set.  (The grain builder writes an index when grain_sources
            # is non-None; an explicit kwarg here lets the caller
            # override that.)
            if atom_species_index is not None:
                asp = np.asarray(atom_species_index, dtype=np.intp)
                if asp.shape[0] != len(self.atoms):
                    raise ValueError(
                        f"atom_species_index length ({asp.shape[0]}) must "
                        f"match atom count ({len(self.atoms)}) after "
                        f"grain construction."
                    )
                self._atom_shell_species_index = asp
        else:
            # Liquid path: atoms came from _build_random_atoms() at init
            # time with purely-random positions.  Pre-separate only
            # severe overlaps (< 0.35 * hard_min ~ one-third a bond),
            # leaving the rest for shell_relax's soft repulsion spring
            # to handle smoothly.  A harder push would pile up a sharp
            # non-physical spike at exactly the cutoff radius (that's
            # exactly the artefact the user saw in the Cu liquid
            # panel).
            from ._grain import _push_close_pairs_apart
            hard_min = float(np.min(
                np.asarray(shell_target.pair_hard_min, dtype=np.float64)
            ))
            push_cutoff = 0.35 * hard_min
            self.atoms.positions = _push_close_pairs_apart(
                self.atoms.positions,
                self.atoms.numbers,
                self.atoms.cell.array,
                pbc=self.atoms.pbc,
                push_cutoff=push_cutoff,
                max_iter=40,
            )
            self._rebuild_spatial_index()
            if atom_species_index is not None:
                asp = np.asarray(atom_species_index, dtype=np.intp)
                if asp.shape[0] != len(self.atoms):
                    raise ValueError(
                        f"atom_species_index length ({asp.shape[0]}) must "
                        f"match atom count ({len(self.atoms)})."
                    )
                self._atom_shell_species_index = asp

        # --- optional build-time orientation refinement ---
        # Runs BEFORE the global FIRE quench so the FIRE starts from a
        # better basin.  Only meaningful when grains exist.  See
        # :meth:`refine_initial_orientations` for the algorithm.  When
        # enabled we first retile every grain at its initial rotation
        # T=0 — this resets atoms to canonical lattice positions
        # (undoing any thermal displacement from
        # ``displacement_sigma`` and giving refinement a clean
        # starting state that matches what trial retiles produce).
        # Skip orientation refinement when the ML backend is doing the
        # relaxation — the EGNN forward pass is expected to fix grain
        # boundaries in one shot, and we benchmarked it without the
        # SO(3) search.
        if (refine_orientations
                and backend == "fire"
                and getattr(self, "_grain_ids", None) is not None
                and getattr(self, "_grain_cells", None) is not None):
            from ._resample import _retile_grain

            grain_ids_arr = np.asarray(self._grain_ids, dtype=np.intp)
            grain_seeds_arr = np.asarray(self._grain_seeds,
                                         dtype=np.float64)
            voronoi_cells_l = self._grain_cells
            masters_l = self._grain_masters
            grain_source_l = self._grain_source
            if grain_source_l is None:
                grain_source_l = np.zeros(len(grain_seeds_arr),
                                          dtype=np.intp)
            is_crystalline_arr = np.asarray(
                self._grain_is_crystalline, dtype=bool,
            )
            box_dim_arr = np.asarray(self._grain_box_dim,
                                     dtype=np.float64)
            for g in [int(_g) for _g in
                      np.unique(grain_ids_arr[grain_ids_arr >= 0])
                      if is_crystalline_arr[int(_g)]]:
                gm = (grain_ids_arr == g)
                target_n = int(np.sum(gm))
                if target_n == 0:
                    continue
                src_idx = int(grain_source_l[g])
                m = masters_l[src_idx]
                retile = _retile_grain(
                    master_positions=np.asarray(
                        m["positions"], dtype=np.float64),
                    master_numbers=np.asarray(
                        m["numbers"], dtype=np.int64),
                    voronoi_cell=voronoi_cells_l[g],
                    seed_world=grain_seeds_arr[g],
                    box_dim=box_dim_arr,
                    rotation=self._grain_rotations_initial[g],
                    translation=np.zeros(3),
                    target_n=target_n,
                )
                if retile is not None:
                    self.atoms.positions[gm] = retile[0]
                    # Update species: multi-species grains re-order
                    # atoms during retile so we must also assign the
                    # corresponding ``numbers`` array (otherwise an O
                    # position lands at a Si atom slot etc.).
                    self.atoms.numbers[gm] = retile[1]
            # Refresh the species-index cache after re-tiling.
            self._atom_species_index = np.searchsorted(
                self._species, self.atoms.numbers,
            )
            self._rebuild_spatial_index()

            r_kwargs = dict(refine_orientations_kwargs or {})
            r_kwargs.setdefault("bond_weight", float(bond_weight))
            r_kwargs.setdefault("angle_weight", float(angle_weight))
            r_kwargs.setdefault("repulsion_weight", float(repulsion_weight))
            r_kwargs.setdefault("hard_core_scale", float(hard_core_scale))
            r_kwargs.setdefault("nonbond_push_scale", float(nonbond_push_scale))
            r_kwargs.setdefault("show_progress", bool(show_progress))
            self.refine_initial_orientations(shell_target, **r_kwargs)

        # --- relax ---
        if backend not in ("fire", "ml", "ml+fire"):
            raise ValueError(
                f"backend must be one of 'fire', 'ml', 'ml+fire'; "
                f"got {backend!r}",
            )
        if backend in ("ml", "ml+fire"):
            if ml_model is None:
                raise ValueError(
                    f"backend={backend!r} requires ml_model "
                    "(an EGNN instance or checkpoint path)",
                )
            from .ml.inference import (
                load_model as _ml_load_model,
                predict_and_optionally_relax,
            )
            if isinstance(ml_model, (str, bytes)) or hasattr(ml_model, "__fspath__"):
                ml_model = _ml_load_model(ml_model)
            n_fire_cleanup = (
                int(ml_fire_cleanup_steps) if backend == "ml+fire" else 0
            )
            # Forward shell_relax_kwargs to the FIRE cleanup pass too
            # so per-material hard_core / nonbond / repulsion stay
            # consistent.
            shell_relax_kw = dict(shell_relax_kwargs)
            shell_relax_kw.setdefault("bond_weight", float(bond_weight))
            shell_relax_kw.setdefault("angle_weight", float(angle_weight))
            shell_relax_kw.setdefault("repulsion_weight", float(repulsion_weight))
            shell_relax_kw.setdefault("hard_core_scale", float(hard_core_scale))
            shell_relax_kw.setdefault("nonbond_push_scale", float(nonbond_push_scale))
            predict_and_optionally_relax(
                self, shell_target, ml_model,
                grain_size=(float(grain_size) if grain_size is not None else 0.0),
                fire_cleanup_steps=n_fire_cleanup,
                chunk_size=ml_chunk_size,
                iterative_steps=int(ml_iterative_steps),
                iterative_momentum=float(ml_iterative_momentum),
                iterative_step_clip=ml_iterative_step_clip,
                iterative_repulsion_per_step=int(ml_repulsion_iters_per_step),
                final_repulsion_iters=int(ml_final_repulsion_iters),
                **shell_relax_kw,
            )
            # Build a minimal summary dict so the rest of generate()
            # doesn't need branching to render the regime / density.
            summary = {
                "backend": backend,
                "ml_fire_cleanup_steps": n_fire_cleanup,
                "ml_iterative_steps": int(ml_iterative_steps),
            }
        else:
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
