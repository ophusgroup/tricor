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
        angle_weight: float = 1.0,
        repulsion_weight: float = 3.0,
        hard_core_scale: float = 1.0,
        nonbond_push_scale: float = 1.0,
        displacement_sigma: float = 0.0,
        atom_species_index: np.ndarray | None = None,
        grain_sources: "list[dict] | None" = None,
        seeds: np.ndarray | None = None,
        # Build-time grain-orientation refinement (recommended for
        # directional-bond materials like Si — see
        # :meth:`refine_initial_orientations`).  ``True`` runs a
        # cheap topology-free coordinate-descent over per-grain
        # rotations BEFORE the FIRE quench; ``False`` skips it (the
        # historical default).
        refine_orientations: bool = False,
        refine_orientations_kwargs: "dict | None" = None,
        protect_crystallites: bool = False,
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
            no grains - start from random positions (liquid/amorphous).
        crystalline_fraction
            Volume fraction filled by crystalline grains (0–1).  Only
            used when *grain_size* is set.  The remaining volume is
            filled with random (amorphous) positions.
        protect_crystallites
            Opt-in (default ``False``, so existing behaviour is unchanged).
            When ``True``, *relative_density* describes the AMORPHOUS region
            only and the crystalline grains stay at full crystal density: the
            random fill is scaled by *relative_density*, the exact-count trim
            draws only from amorphous atoms, and overlap removal always
            sacrifices the amorphous atom of a straddling pair.  Without it the
            global trim punches vacancies into the grains — at 10 Å grains with
            ``relative_density=0.78`` the grain came out at 85.6% of crystal
            density with CN(Al–O) 4.785 against the 6.000 a perfect corundum
            gives.  Requires ``crystalline_fraction < 1``.  Note that
            :meth:`bond_relax` still moves every atom; pass its ``freeze_mask``
            to hold the grains through that step too.
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
        use_grains = (grain_size is not None and float(grain_size) > 0.0) or seeds is not None

        # Initialise the grain bookkeeping that `crystalline_mask` reads, but
        # ONLY if it has never been set.  `__init__` sets `_grain_ids = None`,
        # so `is None` distinguishes "this object has never had grains" from
        # "there is a grain build to preserve".
        #
        # Why initialise at all: grain_size=0 / None (the liquid control) used
        # to leave both attributes at None, so `crystalline_mask` returned None
        # and every caller had to guard it.  scripts/macerelax/expval/
        # build_grain_matrix.py:295 did not, and raised AttributeError after
        # the seed build, G3 measurement and full pack had already run.  With
        # `_grain_ids` all -1 and `_grain_is_crystalline` empty the property
        # returns all-False, the correct reading for a cell built with no
        # grains.  It still returns None before any generate() call, so the
        # documented unknown-vs-zero distinction survives.
        #
        # Why NOT unconditionally: an earlier version of this reset ran on
        # every call, on the theory that a second generate() with no grains
        # would otherwise report the previous build's crystallinity.  That is
        # backwards.  `generate()`'s no-grain path never rebuilds
        # `self.atoms` -- `_build_random_atoms` is called only at __init__
        # (line 136) -- so after generate(grain_size=8) then
        # generate(grain_size=None) the coordinates are BYTE-IDENTICAL to the
        # grain build (verified: 270 atoms, max coordinate change 0.0 A) and
        # the old mask was describing them correctly.  The unconditional reset
        # reported 0 crystalline atoms for a cell that still contained a
        # 99-atom grain.  The grain path overwrites both attributes itself, so
        # preserving them here costs nothing.
        if getattr(self, "_grain_ids", None) is None:
            self._grain_ids = np.full(len(self.atoms), -1, dtype=np.intp)
            self._grain_is_crystalline = np.zeros(0, dtype=bool)

        if use_grains:
            self.atoms = self._build_grain_atoms(
                shell_target,
                grain_size=float(grain_size) if grain_size else 0.0,
                crystalline_fraction=crystalline_fraction,
                displacement_sigma=displacement_sigma,
                grain_sources=grain_sources,
                seeds=seeds,
                protect_crystallites=protect_crystallites,
            )
            # Remember the exact build parameters so
            # ``refine_initial_orientations`` can re-run the full
            # grain-assembly procedure with trial rotations.
            self._grain_build_params = dict(
                grain_size=float(grain_size) if grain_size else 0.0,
                crystalline_fraction=float(crystalline_fraction),
                displacement_sigma=float(displacement_sigma),
                grain_sources=grain_sources,
                seeds=seeds,
                protect_crystallites=bool(protect_crystallites),
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
            # atoms came from _build_random_atoms() at init
            # time with purely-random positions.  Pre-separate only
            # severe overlaps (< 0.35 * hard_min ~ one-third a bond),
            # leaving the rest for shell_relax's soft repulsion spring
            # to handle smoothly.  A harder push would pile up a sharp
            # non-physical spike at exactly the cutoff radius (that's
            # exactly the artefact the user saw in the Cu liquid
            # panel).
            #
            # Skip when num_steps=0 (caller opted out of FIRE) — in
            # that case the caller is handling relaxation themselves
            # (typically via :meth:`bond_relax`) which does its own
            # overlap separation.  At 200³ Å this saves ~2 min.
            if int(num_steps) > 0:
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
        if (refine_orientations
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
        if int(num_steps) > 0:
            # protect_crystallites guarded the grains during BUILDING (padding
            # exclusion, straddling deletion sacrificing the amorphous atom)
            # and then handed everything to shell_relax unmasked, which relaxed
            # the crystallites it had just protected.  The caller could not
            # intervene: `crystalline_mask` does not exist until generate()
            # returns.  Latent so far only because every current caller passes
            # num_steps=0 and runs its own bond_relax -- it goes live for
            # anyone using the default num_steps=200.
            # Full crystalline mask, not crystalline_interior_mask: the latter
            # freezes only 0.4-34.5% of crystalline atoms depending on grain
            # count (see _XTAL_COLLISION_FRAC in _grain.py), so it would leave
            # the grains substantially free -- the opposite of the intent here.
            if protect_crystallites and "freeze_mask" not in shell_relax_kwargs:
                _fm = self.crystalline_mask
                if _fm is not None and _fm.any():
                    shell_relax_kwargs["freeze_mask"] = _fm
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
        else:
            # num_steps == 0: caller has opted out of FIRE.  Skip
            # shell_relax entirely — it would otherwise spend ~90 s
            # at 200³ Å rebuilding the bond topology before running
            # zero relaxation steps.  Caller is presumably running
            # their own relaxation afterwards (e.g. cell.bond_relax()).
            summary = {
                "backend": "fire",
                "num_steps": 0,
            }

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


    @property
    def crystalline_mask(self):
        """Per-atom boolean: was this atom placed inside a crystalline grain?

        Built from the generator's own bookkeeping (``_grain_is_crystalline``
        indexed by ``_grain_ids``), so it is exact rather than detected.  Atoms
        added by the exact-count padding carry ``grain_id = -1`` and come back
        ``False``, which is correct — they are random fill.  ``None`` if no
        grain build has run.
        """
        gid = getattr(self, "_grain_ids", None)
        isx = getattr(self, "_grain_is_crystalline", None)
        if gid is None or isx is None:
            return None
        gid = np.asarray(gid)
        isx = np.asarray(isx, dtype=bool)
        m = np.zeros(len(gid), dtype=bool)
        if isx.size:
            ok = gid >= 0
            m[ok] = isx[gid[ok]]
        return m

    def crystalline_interior_mask(self, shell_target=None):
        """Crystalline atoms MINUS the grain-boundary shell.

        The mask to freeze during relaxation.  Freezing every crystalline
        atom (``crystalline_mask``) makes cross-wall contacts permanent:
        adjacent grains meet at Voronoi walls with random relative
        orientation, atoms land at arbitrary separations, and if both are
        frozen no iteration count can separate them.  That is what forced
        ``_XTAL_COLLISION_FRAC`` up to 1.0 -- deleting one atom of every
        sub-floor cross-wall pair -- which costs 15-43% of the crystalline
        atoms depending on grain count.

        Freezing only the interiors lets the boundary shell relax instead.
        Measured (SiO2, 24 A, cf 0.75, rd 0.9, 28 grains, bond_relax 40):

            FRAC 1.00, freeze all       keep 0.715, min x-x 1.591, short +0.000
            FRAC 0.60, freeze all       keep 0.976, min x-x 0.885, short +1.173
            FRAC 0.60, freeze interiors keep 0.976, min x-x 1.572, short +0.012

        i.e. interior-only freezing recovers 26 points of crystalline density
        at no cost in contact quality.  Returns ``None`` when no grain build
        has run, matching :attr:`crystalline_mask`.
        """
        xt = self.crystalline_mask
        if xt is None:
            return None
        xt = np.asarray(xt, dtype=bool)
        seeds = getattr(self, "_grain_seeds", None)
        gids = getattr(self, "_grain_ids", None)
        if seeds is None or gids is None or not xt.any():
            return xt
        from ._thermal_mc import detect_grain_boundary_atoms
        ppm = None
        if shell_target is not None:
            ppm = float(np.max(np.asarray(shell_target.pair_peak)))
        is_boundary = detect_grain_boundary_atoms(
            self.atoms, gids, seeds, pair_peak_max=ppm,
        )
        return xt & ~np.asarray(is_boundary, dtype=bool)

    def bond_relax(
        self,
        shell_target,
        n_iter: int = 40,
        attract_frac: float = 0.2,
        repel_frac: float = 1.0,
        max_step: float = 0.2,
        device=None,
        freeze_mask=None,
    ) -> None:
        """Combined attract-to-bond-peak + repel-from-hard-core sweep.

        A fast O(N log N) alternative to a full FIRE relaxation for
        cleaning up Voronoi-tiled or ML-predicted positions.  Each
        iteration:

        - Pulls bonded species pairs (those with non-zero
          ``shell_target.coordination_target``) toward
          ``shell_target.pair_peak``.
        - Pushes any pair below ``shell_target.pair_hard_min`` apart.

        Mutates ``self.atoms.positions`` in place.  Uses
        :class:`scipy.spatial.cKDTree` so cost scales linearly with N
        at constant density — at 200³ Å × 600 k atoms, ~1 s/iter on
        CPU.  Drives Si-O to its 1.61 Å peak and non-bonded pairs
        (Si-Si, O-O) onto their hard-core walls in 40-80 iterations.

        Parameters
        ----------
        shell_target
            The :class:`CoordinationShellTarget` whose
            ``pair_peak`` / ``pair_hard_min`` / ``pair_outer`` /
            ``coordination_target`` matrices drive the forces.
        n_iter
            Number of sweeps.  40 typically reaches the bond peak to
            within 0.01 Å.
        attract_frac
            Per-sweep gap-closing fraction toward the bond peak.
        repel_frac
            Per-sweep gap-closing fraction away from the hard-core
            wall.  ``1.0`` (default) closes the gap in one shot,
            with ``max_step`` providing the safety against overshoot
            in dense regions.
        max_step
            Per-atom displacement cap (Å) per sweep.
        freeze_mask
            Optional ``(N,)`` bool array of atoms to hold fixed.  When
            given, the sweep is driven one iteration at a time and the
            masked positions are restored after EVERY sweep -- restoring
            only at the end would let mobile atoms relax against frozen
            neighbours that had themselves drifted (measured at n_iter=20
            on a 20 Å corundum grain: rms 0.41 Å / max 1.03 Å of grain
            motion, and the 3-4 Å shell dropping from 100% six-fold to
            91%).  Measured cost of the frozen path relative to the
            unfrozen one: 1.00x at 1.9k atoms, 1.05x at 4.4k, 1.03x at
            8.6k -- within noise, and NOT growing with N.  (It was
            1.12-1.26x and growing before the per-sweep spatial-hash
            rebuild was removed from this loop.)
            For a grain-in-matrix build, freeze
            :meth:`crystalline_interior_mask`, NOT
            :attr:`crystalline_mask`: freezing the grain-boundary shell as
            well makes sub-floor cross-wall contacts permanent.  See
            ``_XTAL_COLLISION_FRAC`` in ``_grain.py``.
        device
            Optional torch device.  ``None`` (default) runs the pure-
            NumPy CPU path.  A torch device (``"cuda"``, ``"cuda:0"``,
            etc.) dispatches to a hybrid GPU implementation that keeps
            cKDTree pair finding on CPU but moves the per-pair force
            computation and scatter to GPU.  Drops wall-clock from
            ~17 s → ~3-4 s at 100×100×400 / 358 k atoms.  Output
            positions agree with the CPU path to ~1e-4 Å (FP-order
            differences in ``index_add_`` vs ``np.add.at``); atom
            count is unchanged.
        """
        species_idx = (
            getattr(self, "_atom_shell_species_index", None)
            if getattr(self, "_atom_shell_species_index", None) is not None
            else self._atom_species_index
        )
        box = np.diag(np.asarray(self.atoms.cell.array, dtype=np.float64))
        pos = np.asarray(self.atoms.positions, dtype=np.float64)
        kwargs = dict(
            positions=pos, box=box, species_idx=np.asarray(species_idx),
            pair_peak=np.asarray(shell_target.pair_peak, dtype=np.float64),
            pair_hard_min=np.asarray(shell_target.pair_hard_min, dtype=np.float64),
            pair_outer=np.asarray(shell_target.pair_outer, dtype=np.float64),
            coordination_target=np.asarray(
                shell_target.coordination_target, dtype=np.float64),
            n_iter=int(n_iter),
            attract_frac=float(attract_frac),
            repel_frac=float(repel_frac),
            max_step=float(max_step),
        )
        if freeze_mask is None:
            if device is None:
                from ._pair_relax import _bond_relax_sweep
                pos = _bond_relax_sweep(**kwargs)
            else:
                from ._pair_relax import _bond_relax_sweep_torch
                pos = _bond_relax_sweep_torch(**kwargs, device=device)
            self.atoms.positions = pos
        else:
            # Frozen atoms must be restored after EVERY sweep, not once at the
            # end: the sweep is iterative internally, so restoring only at the
            # end would let the mobile atoms relax against frozen neighbours
            # that had themselves drifted.  Measured at n_iter=20 on a 20 Å
            # corundum grain, the unfrozen path moves grain atoms by rms 0.41 Å
            # / max 1.03 Å and drops the 3–4 Å shell from 100% six-fold to 91%.
            fm = np.asarray(freeze_mask, dtype=bool)
            if fm.shape != (len(self.atoms),):
                raise ValueError(
                    f"freeze_mask has shape {fm.shape}, expected "
                    f"({len(self.atoms)},)"
                )
            one = dict(kwargs)
            one["n_iter"] = 1
            for _ in range(n_iter):
                held = np.asarray(self.atoms.positions,
                                  dtype=np.float64)[fm].copy()
                one["positions"] = np.asarray(self.atoms.positions,
                                              dtype=np.float64)
                if device is None:
                    from ._pair_relax import _bond_relax_sweep
                    pos = _bond_relax_sweep(**one)
                else:
                    from ._pair_relax import _bond_relax_sweep_torch
                    pos = _bond_relax_sweep_torch(**one, device=device)
                pos = np.asarray(pos, dtype=np.float64)
                pos[fm] = held
                self.atoms.positions = pos
                # NO per-sweep _rebuild_spatial_index() here.  It builds
                # `_spatial_bins`, the periodic hash for local Monte Carlo
                # queries, which only _monte_carlo.py reads -- _pair_relax.py
                # never references it, and each sweep builds its own cKDTree
                # from the positions passed in.  So rebuilding it inside the
                # loop was pure overhead: n_iter list-of-lists builds over N
                # atoms.  The single rebuild after the loop leaves the hash
                # consistent for any later MC call, which is all that is
                # needed.
        self._rebuild_spatial_index()

    def enforce_hard_core(
        self,
        shell_target,
        n_iter: int = 40,
        push_fraction: float = 0.5,
    ) -> None:
        """Geometric projection step that clears hard-core overlaps.

        Iteratively finds pairs below
        ``shell_target.pair_hard_min`` (via :class:`scipy.spatial.cKDTree`)
        and pushes each violating pair apart along their bond vector
        by ``push_fraction × deficit``.  Pure geometry - no force
        springs - so it cannot pull a pair through its wall the way
        the FIRE finisher's bond springs can.  Use this as a final
        cleanup whenever you suspect FIRE's bond-spring forces have
        compressed pairs below their hard-core distance in dense
        regions (a real failure mode at 100+ Å cells).

        Mutates ``self.atoms.positions`` in place.  Cost: ~1 s per
        sweep at 200³ Å × 600 k atoms; converges in roughly
        ``O(log(initial_deficit / push_fraction))`` sweeps for
        moderate overlaps, more for severe ones from a fresh Voronoi
        tile.  Defaults are tuned to clear NB 01 / NB 02-scale
        overlaps in a single call.

        Parameters
        ----------
        shell_target
            The :class:`CoordinationShellTarget` whose
            ``pair_hard_min`` matrix sets the wall distances.
        n_iter
            Number of projection sweeps.  Early-terminates if no
            violations remain.
        push_fraction
            Per-iter fraction of the deficit to close.  ``0.5``
            (default) is the natural choice - both atoms move
            symmetrically and meet in the middle.  Larger values
            risk overshoot; smaller values just need more sweeps.
        """
        from ._pair_relax import _enforce_hard_core

        species_idx = (
            getattr(self, "_atom_shell_species_index", None)
            if getattr(self, "_atom_shell_species_index", None) is not None
            else self._atom_species_index
        )
        box = np.diag(np.asarray(self.atoms.cell.array, dtype=np.float64))
        pos = np.asarray(self.atoms.positions, dtype=np.float64)
        pos = _enforce_hard_core(
            pos,
            box,
            np.asarray(species_idx),
            pair_hard_min=np.asarray(
                shell_target.pair_hard_min, dtype=np.float64),
            n_iter=int(n_iter),
            push_fraction=float(push_fraction),
        )
        self.atoms.positions = pos
        self._rebuild_spatial_index()

    def __repr__(self) -> str:
        atom_count = len(self.atoms)
        return (
            f"Supercell(label={self.label!r}, cell_dim_angstroms={self.cell_dim_angstroms}, "
            f"atoms={atom_count}, relative_density={self.relative_density:.3f}, "
            f"measure_r_max={self.measure_r_max:.3f}, "
            f"g3_weight_r_scale={self.g3_weight_r_scale:.3f}, best_score={self.best_score})"
        )
