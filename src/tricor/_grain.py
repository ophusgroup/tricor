"""Voronoi grain construction mixin for Supercell.

This module ports the 3D Voronoi tiling algorithm from
``tests/tiling3d.py`` into tricor's grain construction.  The key
properties of the algorithm:

- **Exact geometric Voronoi** via ``scipy.spatial.Voronoi`` on a 3 x 3 x 3
  replica of the seed points, giving a closed cell for every central seed.
- **Exact convex-hull membership test** for which atoms belong to each
  grain, using the cell's face equations.  No nearest-seed approximation
  that drops atoms near an incommensurate periodic image.
- A single **master atom block**: the reference crystal is tiled once
  out to a sphere that covers the largest Voronoi cell (`grain_radius`
  = farthest cell vertex from a seed); each grain rotates that block
  and crops by its cell.
- The final ``np.mod(positions + seed, box)`` places every grain's
  atoms inside ``[0, L)^3``, so the resulting supercell contains
  exactly the atoms that make up the periodic tiling.

See ``tiling3d.py`` / ``voronoi02.ipynb`` in ``tests/`` for the
standalone reference implementation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from ase.atoms import Atoms

from .g3 import _EPS

if TYPE_CHECKING:
    from .shells import CoordinationShellTarget
    from .supercell import Supercell


# --------------------------------------------------------------------------
# Module-level helpers - ported from tests/tiling3d.py
# --------------------------------------------------------------------------


def _unique_rows(points: np.ndarray, decimals: int = 12) -> np.ndarray:
    """Return rows of *points* after rounding to *decimals* decimals."""
    points = np.asarray(points, dtype=float)
    if len(points) == 0:
        return points
    rounded = np.round(points, decimals=decimals)
    _, unique_idx = np.unique(rounded, axis=0, return_index=True)
    return points[np.sort(unique_idx)]


def _periodic_voronoi_3d(box: np.ndarray, seeds: np.ndarray) -> list[dict]:
    """Compute the periodic Voronoi cells of *seeds* in a rectangular box.

    Uses the standard 27-replica trick: the central replica's cells are
    closed (all vertices finite) because the surrounding replicas act as
    neighbour points.

    Each returned cell dict has ``vertices`` (relative to its seed),
    ``equations`` and ``simplices`` from a :class:`scipy.spatial.ConvexHull`
    of those vertices, and ``volume``.
    """
    from scipy.spatial import ConvexHull, Voronoi

    box = np.asarray(box, dtype=float)
    seeds = np.asarray(seeds, dtype=float)
    num_grains = len(seeds)

    shifts = np.array(
        [
            (i * box[0], j * box[1], k * box[2])
            for i in (-1, 0, 1)
            for j in (-1, 0, 1)
            for k in (-1, 0, 1)
        ],
        dtype=float,
    )
    tiled_points = np.concatenate([seeds + shift for shift in shifts], axis=0)
    vor = Voronoi(tiled_points)

    central_block = int(np.where((shifts == (0.0, 0.0, 0.0)).all(axis=1))[0][0])
    central_offset = central_block * num_grains

    cells = []
    for i, seed in enumerate(seeds):
        region_index = vor.point_region[central_offset + i]
        region = vor.regions[region_index]
        if -1 in region or len(region) == 0:
            raise RuntimeError("Unexpected infinite Voronoi region")

        rel_vertices = _unique_rows(vor.vertices[region] - seed)
        if len(rel_vertices) < 4:
            raise RuntimeError("Voronoi cell has too few vertices for a 3D hull")

        hull = ConvexHull(rel_vertices)
        cells.append(
            {
                "vertices": rel_vertices,
                "equations": hull.equations.copy(),
                "simplices": hull.simplices.copy(),
                "volume": float(hull.volume),
            }
        )
    return cells


def _grain_radius_3d(cells: list[dict]) -> float:
    """Largest seed-to-vertex distance across all Voronoi cells."""
    radius = 0.0
    for cell in cells:
        radius = max(radius, float(np.max(np.linalg.norm(cell["vertices"], axis=1))))
    return radius


def _lattice_repeat_spacing_3d(lattice_vectors: np.ndarray) -> float:
    """Smallest non-zero Bravais translation ``|i u + j v + k w|``."""
    lattice_vectors = np.asarray(lattice_vectors, dtype=float)
    coeffs = np.array(
        [
            (i, j, k)
            for i in (-1, 0, 1)
            for j in (-1, 0, 1)
            for k in (-1, 0, 1)
            if not (i == 0 and j == 0 and k == 0)
        ],
        dtype=float,
    )
    translations = coeffs @ lattice_vectors
    spacing = float(np.min(np.linalg.norm(translations, axis=1)))
    if spacing <= 0.0:
        raise ValueError("lattice_vectors must span a non-degenerate 3D cell")
    return spacing


def _build_master_atom_block_3d(
    lattice_vectors: np.ndarray,
    basis_frac: np.ndarray,
    numbers: np.ndarray,
    radius: float,
) -> dict:
    """Tile the reference basis and crop to a sphere of *radius*.

    Returns ``{"positions": (M, 3), "numbers": (M,)}``.
    """
    lattice_vectors = np.asarray(lattice_vectors, dtype=float)
    basis_frac = np.asarray(basis_frac, dtype=float)
    numbers = np.asarray(numbers, dtype=np.int64)

    spacing = _lattice_repeat_spacing_3d(lattice_vectors)
    num_tile = int(np.ceil(radius / max(spacing, _EPS))) + 1

    a_idx = np.arange(-num_tile, num_tile + 1)
    basis_idx = np.arange(len(numbers))
    a, b, c, inds = np.meshgrid(a_idx, a_idx, a_idx, basis_idx, indexing="ij")
    frac = (
        np.column_stack((a.ravel(), b.ravel(), c.ravel()))
        + basis_frac[inds.ravel()]
    )
    points = frac @ lattice_vectors
    keep = np.sum(points ** 2, axis=1) <= radius ** 2 + 1e-12
    return {
        "positions": points[keep],
        "numbers": numbers[inds.ravel()[keep]],
    }


def _points_in_cell(points: np.ndarray, cell: dict, tol: float = 1e-8) -> np.ndarray:
    """Boolean mask: True if ``points @ normals.T + offsets <= tol``
    for every face of the cell."""
    equations = np.asarray(cell["equations"], dtype=float)
    normals = equations[:, :3]
    offsets = equations[:, 3]
    return np.all(points @ normals.T + offsets <= tol, axis=1)


def _cell_tetrahedra(cell: dict) -> tuple[np.ndarray, np.ndarray]:
    """Decompose a Voronoi cell into tetrahedra rooted at its centroid.

    Returns ``(tetrahedra, volumes)`` where each tet is (4, 3) and volumes
    are the (positive) tet volumes.  Degenerate (zero-volume) tets are
    dropped.
    """
    interior_point = np.mean(cell["vertices"], axis=0)
    face_triangles = cell["vertices"][cell["simplices"]]
    tetrahedra = np.concatenate(
        [
            np.repeat(interior_point[None, None, :], len(face_triangles), axis=0),
            face_triangles,
        ],
        axis=1,
    )
    edge1 = tetrahedra[:, 1] - tetrahedra[:, 0]
    edge2 = tetrahedra[:, 2] - tetrahedra[:, 0]
    edge3 = tetrahedra[:, 3] - tetrahedra[:, 0]
    volumes = abs(np.einsum("ij,ij->i", np.cross(edge1, edge2), edge3)) / 6.0
    keep = volumes > 1e-12
    return tetrahedra[keep], volumes[keep]


def _sample_points_in_cell(cell: dict, num_points: int, rng: np.random.Generator) -> np.ndarray:
    """Draw *num_points* uniform random samples inside the Voronoi cell."""
    if num_points <= 0:
        return np.empty((0, 3), dtype=float)
    tetrahedra, volumes = _cell_tetrahedra(cell)
    if len(tetrahedra) == 0:
        return np.empty((0, 3), dtype=float)
    tetra_idx = rng.choice(
        len(tetrahedra), size=num_points, p=volumes / volumes.sum()
    )
    chosen = tetrahedra[tetra_idx]
    weights = rng.exponential(size=(num_points, 4))
    weights /= weights.sum(axis=1, keepdims=True)
    return np.einsum("ni,nij->nj", weights, chosen)


def _random_rotation_matrices(num_grains: int, rng: np.random.Generator) -> np.ndarray:
    """Sample *num_grains* random rotation matrices uniformly on SO(3)."""
    from scipy.spatial.transform import Rotation

    rotations = Rotation.random(num_grains, random_state=rng)
    matrices = rotations.as_matrix()
    if num_grains == 1:
        matrices = matrices[None, :, :]
    return matrices


# --------------------------------------------------------------------------
# Mixin
# --------------------------------------------------------------------------


class _GrainMixin:
    def _build_grain_atoms(
        self: "Supercell",
        shell_target: "CoordinationShellTarget",
        grain_size: float,
        crystalline_fraction: float = 1.0,
        displacement_sigma: float = 0.0,
        max_density_passes: int = 5,
    ) -> Atoms:
        """Build a supercell with crystalline grains via Voronoi tiling.

        Algorithm (see ``tests/tiling3d.py`` for the standalone
        reference implementation):

        1. Place ``num_grains = ceil(V_box / V_grain)`` random seeds.
        2. Compute the periodic Voronoi cells via ``scipy.spatial.Voronoi``
           on a 27-replica copy of the seeds.
        3. ``grain_radius`` = farthest Voronoi vertex from any seed.
        4. Tile the reference basis out to a sphere of
           ``grain_radius`` (single master atom block).
        5. For each grain: random SO(3) rotation of the master block,
           convex-hull membership filter against the grain's Voronoi
           cell, shift by seed, wrap into ``[0, L)^3``.
        6. For non-crystalline grains, sample random positions
           uniformly inside the Voronoi cell and assign species from
           the reference composition.
        7. Optional Gaussian thermal displacement on all atoms.
        """
        cell_mat = self._build_supercell_cell()
        cell_mat = np.asarray(cell_mat, dtype=np.float64)
        if not np.allclose(cell_mat, np.diag(np.diag(cell_mat)), atol=1e-6):
            raise ValueError(
                "Grain construction currently only supports orthogonal "
                "supercells; got a non-diagonal cell matrix.",
            )
        box_dim = np.diag(cell_mat)
        cell_inv = np.linalg.inv(cell_mat)

        ref_cell = np.asarray(self.reference_atoms.cell.array, dtype=np.float64)
        ref_basis_frac = np.asarray(
            self.reference_atoms.get_scaled_positions(wrap=True),
            dtype=np.float64,
        )
        ref_numbers = np.asarray(self.reference_atoms.numbers, dtype=np.int64)

        # ---- 1. Seeds ----
        grain_radius_user = max(float(grain_size) * 0.5, 2.0)
        V_box = float(np.prod(box_dim))
        V_grain = (4.0 / 3.0) * np.pi * grain_radius_user ** 3
        num_grains = max(1, int(np.ceil(V_box / V_grain)))
        seeds = self.rng.random((num_grains, 3)) * box_dim

        # ---- 2. Periodic Voronoi cells ----
        cells = _periodic_voronoi_3d(box_dim, seeds)

        # ---- 3. Master atom block ----
        radius = _grain_radius_3d(cells)
        master = _build_master_atom_block_3d(
            ref_cell, ref_basis_frac, ref_numbers, radius,
        )

        # ---- 4. Decide which grains are crystalline ----
        crystalline_fraction = float(np.clip(crystalline_fraction, 0.0, 1.0))
        num_crystalline = int(np.round(crystalline_fraction * num_grains))
        is_crystalline = np.zeros(num_grains, dtype=bool)
        if num_crystalline > 0:
            chosen = self.rng.permutation(num_grains)[:num_crystalline]
            is_crystalline[chosen] = True

        # ---- 5. Rotations: random SO(3) except for the single-grain
        # ---- spans-the-whole-box case, where identity keeps the
        # ---- rotated lattice commensurate with PBC wrap-around.
        single_box_grain = (
            int(np.sum(is_crystalline)) <= 1
            and grain_radius_user >= 0.5 * float(np.min(box_dim))
        )
        if single_box_grain:
            rotations = np.broadcast_to(np.eye(3), (num_grains, 3, 3)).copy()
        else:
            rotations = _random_rotation_matrices(num_grains, self.rng)

        # ---- 6. Fill each grain ----
        ref_volume = float(abs(np.linalg.det(ref_cell)))
        species_density = float(len(ref_numbers) / max(ref_volume, _EPS))

        # Per-species probabilities (for amorphous sampling) preserve the
        # reference composition on average.
        unique_species, species_counts = np.unique(ref_numbers, return_counts=True)
        species_probs = species_counts.astype(float) / float(species_counts.sum())

        positions_all: list[np.ndarray] = []
        numbers_all: list[np.ndarray] = []
        grain_ids_all: list[np.ndarray] = []

        for i, (seed, cell) in enumerate(zip(seeds, cells)):
            if is_crystalline[i]:
                rotated = master["positions"] @ rotations[i].T
                keep = _points_in_cell(rotated, cell)
                pos = rotated[keep]
                num = master["numbers"][keep]
            else:
                expected = species_density * cell["volume"]
                n_atoms = int(np.floor(expected))
                if self.rng.random() < expected - n_atoms:
                    n_atoms += 1
                pos = _sample_points_in_cell(cell, n_atoms, self.rng)
                num = self.rng.choice(
                    unique_species, size=n_atoms, p=species_probs,
                ).astype(np.int64)

            pos = np.mod(pos + seed, box_dim)
            positions_all.append(pos)
            numbers_all.append(num)
            grain_ids_all.append(np.full(len(pos), i, dtype=np.intp))

        positions = np.concatenate(positions_all, axis=0) if positions_all else (
            np.empty((0, 3), dtype=np.float64)
        )
        numbers = np.concatenate(numbers_all, axis=0) if numbers_all else (
            np.empty(0, dtype=np.int64)
        )
        grain_ids = np.concatenate(grain_ids_all, axis=0) if grain_ids_all else (
            np.empty(0, dtype=np.intp)
        )

        # ---- 7a. Remove grain-boundary overlaps ----
        # Rotated neighbouring grains can leave two atoms within << 1 A
        # of each other at their shared face; delete one of each such
        # pair.  We prefer to delete the atom whose species is most in
        # excess of its reference-scaled target; that keeps step 7c
        # (exact target enforcement) from doing extra work, and tends
        # to preserve stoichiometry at boundaries.  Threshold = 0.9 x
        # the reference's smallest hard-core distance.
        hard_min_scalar = float(
            np.min(np.asarray(shell_target.pair_hard_min, dtype=np.float64))
        )
        dup_cutoff = max(0.5, 0.9 * hard_min_scalar)

        # Target per-species counts: reference stoichiometry scaled by
        # (V_box / V_ref) * relative_density.  Matches the liquid path
        # (Supercell._target_species_counts), so liquid / amorphous /
        # crystalline regimes all land at the same Si and O counts.
        v_ratio = V_box / max(ref_volume, _EPS)
        rel_density = float(getattr(self, "relative_density", 1.0))
        target_by_z: dict[int, int] = {
            int(z): int(round(float(c) * v_ratio * rel_density))
            for z, c in zip(unique_species, species_counts)
        }

        if len(positions) > 0:
            from ase.neighborlist import neighbor_list

            probe = Atoms(
                numbers=numbers, positions=positions,
                cell=cell_mat, pbc=self.reference_atoms.pbc,
            )
            ov_i, ov_j, _ = neighbor_list("ijd", probe, dup_cutoff)
            # Running species counts so priority updates as we remove.
            z_counts = {int(z): int(np.sum(numbers == z)) for z in unique_species}
            remove: set[int] = set()
            for k in range(len(ov_i)):
                ai, aj = int(ov_i[k]), int(ov_j[k])
                if ai in remove or aj in remove:
                    continue
                zi, zj = int(numbers[ai]), int(numbers[aj])
                if zi == zj:
                    pick = ai if self.rng.random() < 0.5 else aj
                else:
                    # Delete from the species that's most over target.
                    over_i = z_counts[zi] - target_by_z[zi]
                    over_j = z_counts[zj] - target_by_z[zj]
                    if over_i > over_j:
                        pick = ai
                    elif over_j > over_i:
                        pick = aj
                    else:
                        pick = ai if self.rng.random() < 0.5 else aj
                remove.add(pick)
                z_counts[int(numbers[pick])] -= 1
            if remove:
                mask = np.ones(len(numbers), dtype=bool)
                mask[list(remove)] = False
                positions = positions[mask]
                numbers = numbers[mask]
                grain_ids = grain_ids[mask]

        # ---- 7b. Enforce exact per-species target counts ----
        # Bring each species to its reference-scaled target by randomly
        # dropping surplus or padding with new random atoms.  Padding
        # atoms try to respect the hard-core separation but accept
        # looser placement if the retry budget is exhausted (rare for
        # small shortfalls).  This guarantees identical Si and O counts
        # across every SiO2 regime so direct g3 / relax comparisons are
        # apples-to-apples.
        cell_inv_local = np.linalg.inv(cell_mat)
        for z, target in target_by_z.items():
            idx = np.where(numbers == z)[0]
            current = int(len(idx))
            if current > target:
                drop = self.rng.choice(
                    idx, size=current - target, replace=False,
                )
                keep = np.ones(len(numbers), dtype=bool)
                keep[drop] = False
                positions = positions[keep]
                numbers = numbers[keep]
                grain_ids = grain_ids[keep]
            elif current < target:
                n_missing = target - current
                added = np.empty((0, 3), dtype=np.float64)
                tries = 0
                max_tries = max(500, 50 * n_missing)
                while len(added) < n_missing and tries < max_tries:
                    trial = self.rng.random(3) * box_dim
                    ok = True
                    if len(positions) > 0:
                        delta = positions - trial
                        frac = delta @ cell_inv_local
                        frac -= np.round(frac)
                        mi = frac @ cell_mat
                        if float(np.min(np.sum(mi * mi, axis=1))) < dup_cutoff ** 2:
                            ok = False
                    if ok and len(added) > 0:
                        delta = added - trial
                        frac = delta @ cell_inv_local
                        frac -= np.round(frac)
                        mi = frac @ cell_mat
                        if float(np.min(np.sum(mi * mi, axis=1))) < dup_cutoff ** 2:
                            ok = False
                    if ok:
                        added = np.vstack([added, trial])
                    tries += 1
                while len(added) < n_missing:
                    # Retry budget exhausted: accept looser placement.
                    added = np.vstack([added, self.rng.random(3) * box_dim])
                positions = np.concatenate([positions, added], axis=0)
                numbers = np.concatenate(
                    [numbers, np.full(n_missing, z, dtype=np.int64)],
                )
                grain_ids = np.concatenate(
                    [grain_ids, np.full(n_missing, -1, dtype=np.intp)],
                )

        # ---- 7c. Optional thermal displacement ----
        if displacement_sigma > _EPS and len(positions) > 0:
            positions = positions + self.rng.normal(
                0.0, displacement_sigma, size=positions.shape,
            )
            frac = positions @ cell_inv
            frac %= 1.0
            positions = frac @ cell_mat

        atoms = Atoms(
            numbers=numbers,
            positions=positions,
            cell=cell_mat,
            pbc=self.reference_atoms.pbc,
        )
        atoms.info["relative_density"] = self.relative_density
        atoms.info["cell_dim_angstroms"] = self.cell_dim_angstroms
        atoms.info["n_grains"] = int(np.sum(is_crystalline))
        atoms.info["grain_size"] = float(grain_size)
        atoms.info["crystalline_fraction"] = float(crystalline_fraction)
        atoms.info["grain_radius"] = float(radius)

        self._grain_ids = grain_ids
        self._grain_seeds = seeds.copy()
        return atoms
