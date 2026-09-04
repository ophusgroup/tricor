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


_XTAL_COLLISION_FRAC = 1.00

if TYPE_CHECKING:
    from .shells import CoordinationShellTarget
    from .supercell import Supercell

# The fast grain path sizes its master atom block from a statistical
# envelope of the largest Voronoi cell radius. See the
# ``can_fast`` branch of ``_GrainMixin._build_grain_atoms``.
_FAST_PATH_EXACT_RADIUS_MAX_GRAINS = 64

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


def _grain_assign_fast(
    master_positions: np.ndarray,
    master_numbers: np.ndarray,
    seeds: np.ndarray,
    rotations: np.ndarray,
    box_dim: np.ndarray,
    source_offset: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized single-source grain assembly via nearest-seed assignment.

    Mathematically equivalent to the original per-grain loop that filters
    rotated master-block atoms through each Voronoi cell's convex-hull
    equations — nearest-seed assignment IS the Voronoi diagram by
    definition.  Avoids both ``scipy.spatial.Voronoi`` (~47 s for amorphous
    on a 100×100×400 box) and the Python-overhead-bound per-grain loop
    (~137 s on the same trajectory).

    Returns ``(positions, numbers, grain_ids, shell_species_idx)`` in the
    same layout as the slow path so the downstream overlap-removal /
    species-target / push-pairs steps can run unchanged.
    """
    from scipy.spatial import cKDTree

    G = int(seeds.shape[0])
    K = int(master_positions.shape[0])

    # Rotate the master block by each grain's rotation.  einsum:
    #   rotated[g, k, j] = sum_i master_positions[k, i] * rotations[g, j, i]
    rotated = np.einsum("ki,gji->gkj", master_positions, rotations)

    # Translate by each grain's seed and wrap into [0, box_dim).
    world_pos = np.mod(rotated + seeds[:, None, :], box_dim)

    # Nearest-seed assignment with periodic Euclidean metric.  cKDTree's
    # ``boxsize`` triggers the torus metric — exactly what Voronoi face
    # equations would produce.
    tree = cKDTree(seeds, boxsize=box_dim)
    flat = world_pos.reshape(-1, 3)
    _, nearest = tree.query(flat, k=1)
    nearest = nearest.reshape(G, K)

    grain_idx_col = np.arange(G, dtype=np.intp)[:, None]
    keep = nearest == grain_idx_col

    positions = world_pos[keep]
    numbers = np.broadcast_to(master_numbers, (G, K))[keep]
    grain_ids = np.broadcast_to(grain_idx_col, (G, K))[keep].astype(np.intp)
    shell_species_idx = np.full(
        positions.shape[0], int(source_offset), dtype=np.intp,
    )
    return positions, numbers, grain_ids, shell_species_idx


def _sample_padding_atoms(
    existing_positions: np.ndarray,
    box_dim: np.ndarray,
    n_missing: int,
    pad_min_sep: float,
    rng: np.random.Generator,
    *,
    max_rounds: int = 200,
    oversample: float = 4.0,
    exclude=None,
    pair_cutoff: "np.ndarray | None" = None,
    species_index: "np.ndarray | None" = None,
    pad_row: int = 0,
    backoff: float = 0.96,
) -> tuple[np.ndarray, int, float]:
    """Batched rejection sampling for step 7b padding.

    Equivalent in spirit to the original serial trial-and-error loop:
    pick uniform-random positions inside the box that respect
    ``pad_min_sep`` against every existing atom AND every padding atom
    already accepted in this call.  But instead of doing one trial per
    iteration with an O(N) distance scan, this generates trial atoms in
    big batches and filters them with ``scipy.spatial.cKDTree`` (torus
    metric via ``boxsize=box_dim``).  Drops step 7b padding wall-clock
    from ~131 s → ~3 s for amorphous at 100×100×400.

    ``exclude`` is an optional ``f(points) -> bool array`` marking trial
    positions that must be rejected regardless of spacing.  Used by
    ``protect_crystallites`` to keep padding out of the crystalline Voronoi
    cells: this sampler draws uniformly over the whole box, so the exact-count
    shortfall otherwise lands random atoms INSIDE the crystallites as
    interstitials.  Measured on a 20 Å corundum grain, 61 padding atoms sat
    inside its hull and pushed the interior to CN 6.13-6.29 against the 6.000 a
    perfect crystal gives — crystal-crystal pairs alone were exactly 6.000 /
    100% six-fold, which is how the interstitials were identified.

    PAIR-RESOLVED spacing.  ``pad_min_sep`` is a single scalar derived from
    ``min(pair_hard_min)``, i.e. one pair's floor imposed on every pair.  For
    corundum that is Al-O 1.710, so 0.8 x it lets an O-O land at 1.368 A when
    its true floor is 2.326 A -- tighter than the 1.710 A pack that produced
    the peroxide defects in the first place.  When ``pair_cutoff`` (an (S, S)
    matrix), ``species_index`` (existing atom -> species row) and ``pad_row``
    (the species being padded) are supplied, each trial is tested against the
    real per-pair floor instead.  Species trees are built once and reused;
    only the scale changes between rounds.

    GRACEFUL BACKOFF.  The old version ran 8
    fixed rounds and left the caller to loose-place the shortfall with NO
    spacing test at all -- measured on Al2O3 cf=0.95 the batched sampler placed
    0 of 535 atoms, so all 535 went in unchecked.  Instead the floor is relaxed
    geometrically every 10 rounds with NO lower clamp, and the achieved scale is
    RETURNED so the caller can record how far it had to give.  Relaxing the
    floor is always preferable to the old behaviour of handing the shortfall to
    an unchecked random placement.

    Returns ``(accepted_positions, total_trials, effective_scale)``;
    ``effective_scale < 1`` means the floor was relaxed to reach the count.
    """
    from scipy.spatial import cKDTree

    if n_missing <= 0:
        return np.empty((0, 3), dtype=np.float64), 0, 1.0

    use_pairs = pair_cutoff is not None and species_index is not None
    trees_by_species: list = []
    tree_existing = None
    if use_pairs:
        pair_cutoff = np.asarray(pair_cutoff, dtype=np.float64)
        species_index = np.asarray(species_index, dtype=np.intp)
        for srow in range(pair_cutoff.shape[0]):
            sel = species_index == srow
            trees_by_species.append(
                cKDTree(existing_positions[sel], boxsize=box_dim)
                if sel.any() else None
            )
        self_sep = pair_cutoff[pad_row, pad_row]
    else:
        tree_existing = (
            cKDTree(existing_positions, boxsize=box_dim)
            if len(existing_positions) > 0 else None
        )
        self_sep = pad_min_sep

    accepted = np.empty((0, 3), dtype=np.float64)
    total_trials = 0
    scale = 1.0
    # `max_rounds` is the budget after which the decayed floor stops being
    # meaningfully restrictive; `_HARD_ROUNDS` is the absolute ceiling.  We keep
    # going past max_rounds because a short count changes composition AND
    # density and is unrecoverable, whereas a padding atom is amorphous and
    # free to move, so a tight pair involving one is separated by the
    # relaxation that follows -- verified in isolation: a free atom 1.5 A from
    # a FROZEN crystalline partner reaches its 2.975 A floor in 40 bond_relax
    # sweeps with the frozen atom immobile, because `skip_mask` gives the free
    # partner the whole displacement.  That repair does stall in an over-dense
    # cell (measured: relative_density 1.2-1.8 leaves pairs ~0.3 A sub-floor
    # indefinitely), so the guarantee is "placed, and separable at sane
    # density", not "placed and always clean".
    #
    # Two independent exits, because the two failure modes are different:
    #   * `_blocked` -- 40 CONSECUTIVE rounds in which `exclude` rejected the
    #     entire batch.  That means the unprotected volume is saturated and no
    #     amount of further decay helps, so stop early rather than grind.
    #   * `_HARD_ROUNDS` -- an absolute ceiling, so the worst case is bounded
    #     no matter how the accept fraction behaves.  Without it the `_blocked`
    #     reset on every placement makes the true bound O(40 * n_missing);
    #     measured by direct call at 12,409 rounds / 1.8e8 trials before this
    #     ceiling existed.
    _HARD_ROUNDS = 4 * max_rounds
    _blocked = 0
    _round = -1
    while _round + 1 < _HARD_ROUNDS and _blocked < 40:
        _round += 1
        need = n_missing - len(accepted)
        if need <= 0:
            break
        if _round and _round % 10 == 0:
            # UNCLAMPED, deliberately.  A clamp turns "floor relaxed to 0.84"
            # into "N atoms placed with no spacing test at all", which is
            # strictly worse and much harder to notice.  Backing off always
            # beats giving up: the achieved scale is returned and recorded in
            # atoms.info["padding_report"], so an over-dense box reports how
            # far it had to give instead of hiding unchecked atoms.
            scale *= backoff
        # Over-sample by ``oversample`` to absorb a typical ~25 %
        # rejection rate in a single round.  Floor at 256 so very
        # small ``need`` values still amortize the KDTree query.
        n_trial = max(int(np.ceil(oversample * need)), 256)
        trials = rng.random((n_trial, 3)) * box_dim
        total_trials += n_trial

        # 0) Drop trials inside an excluded region (protected crystallites).
        if exclude is not None and len(trials):
            keep_ex = ~np.asarray(exclude(trials), dtype=bool)
            trials = trials[keep_ex]
            if trials.shape[0] == 0:
                # The ONLY failure the floor decay cannot fix: every candidate
                # landed inside a protected crystallite.  Padding deliberately
                # never enters those cells, so if the unprotected volume is
                # saturated the count cannot be filled at any scale.
                _blocked += 1
                continue

        # 1) Drop trials that collide with EXISTING atoms, per-pair.  One
        #    nearest-neighbour query per species row: the floor a trial must
        #    clear depends on WHICH species it is near, so a single k=1 query
        #    against all atoms cannot express it.
        cand = trials
        if use_pairs:
            for srow, tree_s in enumerate(trees_by_species):
                if tree_s is None or cand.shape[0] == 0:
                    continue
                d_nn, _ = tree_s.query(cand, k=1)
                cand = cand[d_nn >= pair_cutoff[pad_row, srow] * scale]
        elif tree_existing is not None:
            d_nn, _ = tree_existing.query(trials, k=1)
            cand = trials[d_nn >= pad_min_sep * scale]
        if cand.shape[0] == 0:
            continue

        # 2) Drop trials that collide with PREVIOUSLY-ACCEPTED padding
        #    atoms.  Cheap KDTree on the small accepted set.
        if len(accepted) > 0:
            tree_acc = cKDTree(accepted, boxsize=box_dim)
            d_nn_acc, _ = tree_acc.query(cand, k=1)
            cand = cand[d_nn_acc >= self_sep * scale]
        if cand.shape[0] == 0:
            continue

        # 3) Drop intra-batch collisions.  ``query_pairs`` returns the
        #    upper-triangular (i, j) for i<j; greedy mark j as dropped.
        #    Skipping this would let two close trials both slip through
        #    the same round, which the original serial loop would have
        #    caught.
        if len(cand) > 1:
            tree_cand = cKDTree(cand, boxsize=box_dim)
            pairs = tree_cand.query_pairs(
                self_sep * scale, output_type="ndarray",
            )
            if len(pairs) > 0:
                to_drop = np.zeros(len(cand), dtype=bool)
                # Sort pairs by first index so the greedy choice (drop
                # j when i is still alive) keeps a deterministic
                # winner for chains of overlapping trials.
                for i, j in pairs[np.argsort(pairs[:, 0])]:
                    if not to_drop[i] and not to_drop[j]:
                        to_drop[j] = True
                cand = cand[~to_drop]

        # Take just enough to hit the target.
        take = min(need, cand.shape[0])
        accepted = (
            cand[:take] if accepted.shape[0] == 0
            else np.concatenate([accepted, cand[:take]], axis=0)
        )
        _blocked = 0

    return accepted, total_trials, scale


def _push_close_pairs_apart(
    positions: np.ndarray,
    numbers: np.ndarray,
    cell_mat: np.ndarray,
    *,
    pbc,
    push_cutoff: float,
    max_iter: int = 40,
    pair_cutoff: "np.ndarray | None" = None,
    species_index: "np.ndarray | None" = None,
    skip_mask: "np.ndarray | None" = None,
) -> np.ndarray:
    """Iteratively push any pair closer than its cutoff out to that cutoff.

    ``push_cutoff`` is the scalar fallback.  When ``pair_cutoff`` (an (S, S)
    matrix) and ``species_index`` (atom -> species row) are supplied, each pair
    gets its OWN target instead.

    Why that matters: the caller used to collapse ``pair_hard_min`` to
    ``np.min(...)`` and push every species pair to that one number.  For
    corundum that is min(Al-Al 2.513, Al-O 1.710, O-O 2.326) = 1.710, so O-O
    was left 0.62 A too close and Al-Al 0.80 A too close -- by construction,
    in every pack.  Those residual contacts then became the MACE wall's floor
    (pack min - margin), which licensed a peroxide collapse to 1.58 A.

    ``skip_mask`` marks atoms that must not be moved; a pair is skipped only
    when BOTH of its atoms are masked.

    Cheap geometric pre-conditioner used BEFORE shell_relax.  Without
    it, close pairs introduced by Voronoi-grain overlap padding or by
    random placement at near-target density can sit below the hard-
    core wall deep enough that shell_relax never escapes them (the
    surrounding bond springs hold them in).

    Implementation note: uses :class:`scipy.spatial.cKDTree` (with the
    ``boxsize`` PBC parameter).  The previous ``ase.neighbor_list``
    backend re-built the cell list inside the loop and spent ~12 s per
    call at 600 k atoms — 40 iterations of that was 8 min just for
    the random-placement separation at 200³ Å.  cKDTree builds in
    ~1.5 s and the query for tiny cutoffs is much cheaper.
    """
    from scipy.spatial import cKDTree

    use_pairs = pair_cutoff is not None and species_index is not None
    if use_pairs:
        pair_cutoff = np.asarray(pair_cutoff, dtype=np.float64)
        species_index = np.asarray(species_index, dtype=np.intp)
        query_r = pair_cutoff.max()
    else:
        query_r = push_cutoff
    if len(positions) == 0 or query_r <= 0:
        return positions
    positions = np.asarray(positions, dtype=np.float64).copy()
    cell_mat = np.asarray(cell_mat, dtype=np.float64)
    # cKDTree's PBC support needs an orthorhombic boxsize.  All tricor
    # supercells are orthorhombic by construction, so the diagonal of
    # cell_mat is the box.  Fall back to non-PBC if any pbc=False (rare).
    box = np.diag(cell_mat).astype(np.float64)
    use_pbc = bool(np.all(np.asarray(pbc)))
    for _ in range(int(max_iter)):
        if use_pbc:
            wrap = positions - np.floor(positions / box) * box
            tree = cKDTree(wrap, boxsize=box)
        else:
            wrap = positions
            tree = cKDTree(wrap)
        pairs = tree.query_pairs(query_r, output_type="ndarray")
        if len(pairs) == 0:
            break
        ii = pairs[:, 0]
        jj = pairs[:, 1]
        if use_pairs:
            target = pair_cutoff[species_index[ii], species_index[jj]]
        else:
            target = np.full(len(ii), push_cutoff)
        if skip_mask is not None:
            frozen_pair = skip_mask[ii] & skip_mask[jj]
            if frozen_pair.any():
                keep = ~frozen_pair
                ii, jj, target = ii[keep], jj[keep], target[keep]
                if len(ii) == 0:
                    break
        delta = wrap[jj] - wrap[ii]
        if use_pbc:
            delta -= np.round(delta / box) * box
        d = np.linalg.norm(delta, axis=1).clip(min=1e-9)
        active = d < target
        if not active.any():
            break
        ii, jj, delta, d, target = (ii[active], jj[active], delta[active],
                                    d[active], target[active])
        needed = target - d
        unit = delta / d[:, None]
        if skip_mask is None:
            fi = fj = np.full(len(ii), 0.5)
        else:
            # A protected (crystalline) atom must not move, so its free partner
            # absorbs the entire separation.  Splitting it 50/50 as before is
            # what distorted the frozen grains: the push was enforcing the much
            # larger pair-resolved separations by dragging lattice atoms, which
            # created MORE short crystalline-crystalline contacts than it fixed
            # (measured 78 -> 660).  Pairs with both atoms protected were
            # already dropped above.
            wi = (~skip_mask[ii]).astype(np.float64)
            wj = (~skip_mask[jj]).astype(np.float64)
            tot = np.clip(wi + wj, 1e-12, None)
            fi, fj = wi / tot, wj / tot
        np.add.at(positions, jj, (fj * needed)[:, None] * unit)
        np.add.at(positions, ii, -(fi * needed)[:, None] * unit)
        if use_pbc:
            positions -= np.floor(positions / box) * box
    return positions


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
        grain_sources: "list[dict] | None" = None,
        rotations_override: "np.ndarray | None" = None,
        seeds: "np.ndarray | None" = None,
        protect_crystallites: bool = False,
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

        # Build the list of per-grain source crystals.  Default is a
        # single source = the reference atoms on the supercell.  When
        # ``grain_sources`` is supplied (e.g. for sp²/sp³ carbon
        # blends), each grain samples a source by weight.  Each entry
        # carries an ``atoms``, a ``species_offset`` (used to populate
        # the per-atom virtual-species index for the relaxer), and a
        # ``weight``.
        if grain_sources is None:
            sources = [
                {
                    "atoms": self.reference_atoms,
                    "species_offset": 0,
                    "weight": 1.0,
                }
            ]
        else:
            if len(grain_sources) == 0:
                raise ValueError("grain_sources must be a non-empty list.")
            sources = list(grain_sources)
        source_weights = np.asarray(
            [float(s.get("weight", 1.0)) for s in sources], dtype=np.float64
        )
        if not np.all(source_weights >= 0):
            raise ValueError("grain_sources weights must be non-negative.")
        if source_weights.sum() <= 0:
            raise ValueError("grain_sources weights must sum to > 0.")
        source_probs = source_weights / source_weights.sum()
        multi_source = len(sources) > 1

        # Legacy reference (used for density / padding / composition
        # heuristics below).  We pick the first source by convention;
        # it only drives stoichiometric rounding, not crystalline
        # content.  For carbon sp²/sp³ both sources are pure C so this
        # is fine.
        ref_cell = np.asarray(sources[0]["atoms"].cell.array, dtype=np.float64)
        ref_basis_frac = np.asarray(
            sources[0]["atoms"].get_scaled_positions(wrap=True),
            dtype=np.float64,
        )
        ref_numbers = np.asarray(sources[0]["atoms"].numbers, dtype=np.int64)

        # ---- 1. Seeds ----
        grain_radius_user = max(float(grain_size) * 0.5, 2.0)
        V_box = float(np.prod(box_dim))
        # Whether the caller supplied seeds (upstream's graded order-gradient
        # builder); captured before ``seeds`` is reassigned below so the
        # fast-path gate can exclude the supplied-seed case.
        seeds_supplied = seeds is not None
        if seeds is not None:
            seeds = np.asarray(seeds, dtype=np.float64)
            num_grains = len(seeds)
            if num_grains == 0:
                raise ValueError("seeds must contain at least one seed.")
        else:
            V_grain = (4.0 / 3.0) * np.pi * grain_radius_user ** 3
            num_grains = max(1, int(np.ceil(V_box / V_grain)))
            seeds = self.rng.random((num_grains, 3)) * box_dim

        # ---- Fast-path dispatch ----
        # Every production preset has crystalline_fraction = 1.0 (default)
        # and a single source.  In that regime we don't need exact Voronoi
        # cell faces — nearest-seed assignment IS the Voronoi diagram by
        # definition.  Skipping scipy.spatial.Voronoi + the per-grain
        # Python loop drops _build_grain_atoms wall-clock from 184 s →
        # ~3 s for amorphous at 100×100×400 (35 000 grains) and from 26
        # s → ~1 s for crystalline_30.  See scratch/profile_pack*.out for
        # the attribution.
        #
        # Multi-source, partial-crystalline, or overridden-input runs
        # still use the original slow path below.
        # ``_USE_FAST_GRAIN_PATH`` is a class-level escape hatch: setting
        # ``Supercell._USE_FAST_GRAIN_PATH = False`` from outside forces
        # the original Voronoi + per-grain loop, which is useful for
        # head-to-head validation (see scratch/validate_fast_path.py).
        # Defaults to True so production code uses the fast path
        # transparently.
        use_fast = getattr(type(self), "_USE_FAST_GRAIN_PATH", True)
        # Below FAST_MIN_GRAINS the fast path's statistical master-block
        # radius can under-fill the largest grain (negligible/sub-atomic at
        # production cell sizes, but a real void for few-grain / small cells
        # -- see project_fast_packing_allsizes_todo).  Use exact Voronoi below.
        FAST_MIN_GRAINS = 200
        can_fast = (
            use_fast
            and not seeds_supplied
            and not multi_source
            and float(crystalline_fraction) >= 0.9999
            and num_grains >= FAST_MIN_GRAINS
        )

        if can_fast:
            # Master-block radius must cover the longest Voronoi cell at
            # this seed density.  For N seeds in volume V the typical
            # cell radius is r_typ = (3V / (4 pi N))^(1/3) and the
            # empirical max scales as r_typ * (log N)^(1/3) for Poisson
            # seeds.  1.3× that envelope catches the long tail; the
            # KDTree drop step rejects any extras cleanly.  ``max`` with
            # ``grain_radius_user`` guards the few-grains case where the
            # statistical estimate would underrate the requested grain
            # size.
            r_typ = (3.0 * V_box / (4.0 * np.pi * num_grains)) ** (1.0 / 3.0)
            radius = max(
                grain_radius_user,
                1.3 * r_typ * (float(np.log(max(num_grains, 2))) ** (1.0 / 3.0)),
            )

            master = _build_master_atom_block_3d(
                ref_cell, ref_basis_frac, ref_numbers, radius,
            )
            master["species_offset"] = 0
            masters = [master]

            # All grains crystalline; trivial source assignment.
            is_crystalline = np.ones(num_grains, dtype=bool)
            grain_source = np.zeros(num_grains, dtype=np.intp)
            # _grain_cells is unused in the fast path (refinement is
            # tied to the slow path's exact cell metadata; production
            # always passes refine_orientations=False).  Setting it
            # None is the documented "fast path" sentinel — see
            # refine_initial_orientations docstring.
            cells = None
            single_box_grain = False

            rotations = _random_rotation_matrices(num_grains, self.rng)

            positions, numbers, grain_ids, shell_species_idx = (
                _grain_assign_fast(
                    master["positions"], master["numbers"],
                    seeds, rotations, box_dim, source_offset=0,
                )
            )

            # Jump past the slow-path Voronoi / loop / concat block —
            # skip to step 7a (overlap removal) below.
            num_grains_total = num_grains
            # Stash a placeholder for the variables read after the loop
            # so the code below behaves as if the slow path had run.
            unique_species, species_counts = np.unique(
                ref_numbers, return_counts=True,
            )
            species_probs = (species_counts.astype(float)
                             / float(species_counts.sum()))
            ref_volume = float(abs(np.linalg.det(ref_cell)))
            species_density = float(
                len(ref_numbers) / max(ref_volume, _EPS)
            )
        else:
            # ---- 2. Periodic Voronoi cells (slow path) ----
            cells = _periodic_voronoi_3d(box_dim, seeds)

            # ---- 3. Master atom block (one per source) ----
            radius = _grain_radius_3d(cells)
            masters: list[dict] = []
            for src in sources:
                src_cell = np.asarray(src["atoms"].cell.array, dtype=np.float64)
                src_basis = np.asarray(
                    src["atoms"].get_scaled_positions(wrap=True), dtype=np.float64
                )
                src_numbers = np.asarray(src["atoms"].numbers, dtype=np.int64)
                master = _build_master_atom_block_3d(
                    src_cell, src_basis, src_numbers, radius,
                )
                # Tag each master with its source's species_offset so the
                # orientation-refinement retile can restore the correct
                # virtual-species index for the rotated grain.  Without
                # this, multi-source composite cells (sp²/sp³ carbon,
                # SiO₂/Si₃N₄ blends, ...) lose all virtual-species
                # information after refinement because every atom carries
                # the SAME atomic number — searchsorted(self._species,
                # numbers) returns 0 for every atom and tags them all as
                # the first virtual species.
                master["species_offset"] = int(src.get("species_offset", 0))
                masters.append(master)

            # Per-grain source assignment: draw by weight (or uniform for
            # legacy single-source).  Store on self so the trajectory
            # exporter can introspect which grain is which type.
            num_grains_total = len(cells)
            if multi_source:
                grain_source = self.rng.choice(
                    len(sources), size=num_grains_total, p=source_probs,
                ).astype(np.intp)
            else:
                grain_source = np.zeros(num_grains_total, dtype=np.intp)

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
                # Every grain shares the same seed offset so that the tiles
                # produced by the (identity-rotated) master block match at
                # the Voronoi cell boundaries - without this, each grain
                # has a different random offset and adjacent grains produce
                # mismatched copies of the same lattice, creating boundary
                # distortions that ruin the crystalline structure.
                shared_seed = seeds[0].copy()
                seeds = np.broadcast_to(shared_seed, seeds.shape).copy()
            elif rotations_override is not None:
                # Caller supplies the per-grain rotation set (orientation
                # refinement re-runs this builder with a fixed RNG seed and
                # trial rotations).  Skip the random draw so the RNG stream
                # stays aligned across calls.
                rotations = np.asarray(rotations_override, dtype=np.float64).copy()
            else:
                rotations = _random_rotation_matrices(num_grains, self.rng)

            # ---- 6. Fill each grain ----
            ref_volume = float(abs(np.linalg.det(ref_cell)))
            species_density = float(len(ref_numbers) / max(ref_volume, _EPS))
            # read early: the amorphous fill below needs it (see
            # ``protect_crystallites``); the exact-count block re-reads the
            # same attribute further down.
            _rel_density = getattr(self, "relative_density", 1.0)

            # Per-species probabilities (for amorphous sampling) preserve the
            # reference composition on average.
            unique_species, species_counts = np.unique(ref_numbers, return_counts=True)
            species_probs = species_counts.astype(float) / float(species_counts.sum())

            # For multi-source builds the exact-count enforcement below
            # would otherwise trim every regime to sources[0]'s density,
            # destroying the denser phase (e.g. diamond atoms trimmed down
            # to graphite density when sources[0] is graphite).  Use a
            # weight-averaged reference density and a weight-averaged
            # formula-unit count so each regime's target atom count scales
            # with its actual phase mix.
            if multi_source:
                weighted_density = 0.0
                weighted_ref_volume = 0.0
                weighted_formula_count = 0.0
                for ki, src in enumerate(sources):
                    _sc = np.asarray(src["atoms"].cell.array, dtype=np.float64)
                    _svol = float(abs(np.linalg.det(_sc)))
                    _snum = int(len(src["atoms"].numbers))
                    _w = float(source_probs[ki])
                    weighted_density += _w * (_snum / max(_svol, _EPS))
                    weighted_ref_volume += _w * _svol
                    weighted_formula_count += _w * _snum
                species_density = float(weighted_density)

            positions_all: list[np.ndarray] = []
            numbers_all: list[np.ndarray] = []
            grain_ids_all: list[np.ndarray] = []
            shell_species_all: list[np.ndarray] = []

            for i in range(len(seeds)):
                seed, cell = seeds[i], cells[i]
                src_idx = int(grain_source[i])
                master = masters[src_idx]
                src_offset = int(sources[src_idx]["species_offset"])
                if is_crystalline[i]:
                    rotated = master["positions"] @ rotations[i].T
                    keep = _points_in_cell(rotated, cell)
                    pos = rotated[keep]
                    num = master["numbers"][keep]
                    # All atoms from this grain get the source's species
                    # offset (plus 0 for single-species sources).
                    shell_species = np.full(
                        len(pos), src_offset, dtype=np.intp
                    )
                else:
                    # ``protect_crystallites``: relative_density describes the
                    # AMORPHOUS region only, so scale the random fill here
                    # instead of thinning the whole box afterwards.  Without
                    # this the amorphous regions are filled at full crystal
                    # density and the global trim then removes atoms at random
                    # from the crystallites too -- measured at 10 A grains,
                    # rd 0.78: grain density 85.6% of crystal, CN(Al-O) 4.785
                    # against the 6.000 a perfect crystal gives.
                    _fill_scale = _rel_density if protect_crystallites else 1.0
                    expected = species_density * _fill_scale * cell["volume"]
                    n_atoms = int(np.floor(expected))
                    if self.rng.random() < expected - n_atoms:
                        n_atoms += 1
                    pos = _sample_points_in_cell(cell, n_atoms, self.rng)
                    num = self.rng.choice(
                        unique_species, size=n_atoms, p=species_probs,
                    ).astype(np.int64)
                    shell_species = np.full(
                        n_atoms, src_offset, dtype=np.intp
                    )

                pos = np.mod(pos + seed, box_dim)
                positions_all.append(pos)
                numbers_all.append(num)
                grain_ids_all.append(np.full(len(pos), i, dtype=np.intp))
                shell_species_all.append(shell_species)

            positions = np.concatenate(positions_all, axis=0) if positions_all else (
                np.empty((0, 3), dtype=np.float64)
            )
            numbers = np.concatenate(numbers_all, axis=0) if numbers_all else (
                np.empty(0, dtype=np.int64)
            )
            grain_ids = np.concatenate(grain_ids_all, axis=0) if grain_ids_all else (
                np.empty(0, dtype=np.intp)
            )
            shell_species_idx = np.concatenate(shell_species_all, axis=0) if shell_species_all else (
                np.empty(0, dtype=np.intp)
            )

        # ---- 7a. Remove grain-boundary overlaps ----
        # Rotated neighbouring grains can leave two atoms almost on top
        # of each other at their shared face; delete one of each such
        # pair.  We prefer to delete the atom whose species is most in
        # excess of its reference-scaled target; that keeps step 7b
        # (exact target enforcement) from doing extra work, and tends
        # to preserve stoichiometry at boundaries.  Threshold = 0.55 x
        # the reference's smallest hard-core distance - aggressive
        # enough to catch real overlaps but lax enough that "merely
        # distorted" pairs at ~0.7 x hard_min survive to be relaxed by
        # shell_relax, keeping the initial atom count close to target
        # and avoiding the close pairs that random padding would
        # otherwise introduce when the box is already packed.
        hard_min_scalar = float(
            np.min(np.asarray(shell_target.pair_hard_min, dtype=np.float64))
        )
        # Pair-RESOLVED floors.  `hard_min_scalar` above collapses the whole
        # (S, S) matrix to its smallest entry, which for a multi-species
        # reference is one particular pair's floor applied to every pair.  Keep
        # the scalar for the legacy code paths, but push and collide against
        # the real matrix.
        _pair_hard = np.asarray(shell_target.pair_hard_min, dtype=np.float64)
        _shell_species = [int(z) for z in np.asarray(shell_target.species)]
        _z_to_row = {z: i for i, z in enumerate(_shell_species)}
        _sp_idx = np.array([_z_to_row.get(int(z), 0) for z in numbers],
                           dtype=np.intp)
        # VALIDATE, do not silently degrade.  This used to be a boolean
        # `_pair_ok` that fell back to the collapsed scalar whenever it was
        # False.  Measured across all 20 catalogue seeds, the two conditions
        # that could indicate a malformed matrix (dimension mismatch,
        # non-finite entries) NEVER fired -- the only condition that ever
        # fired was `shape[0] > 1`, which is False for every single-species
        # reference (C, Ge, Si) where a 1x1 matrix is perfectly valid.  So the
        # fallback protected nothing and instead disabled pair-resolved
        # spacing for the elemental seeds, which left 18 crystalline-
        # crystalline collisions down to 1.085 A in an a-Si cf=0.25 build.
        # A malformed matrix here is a bug in CoordinationShellTarget, and
        # quietly substituting min(pair_hard_min) would hide it.
        if _pair_hard.shape[0] != len(_shell_species):
            raise ValueError(
                f"pair_hard_min is {_pair_hard.shape} but the shell target "
                f"declares {len(_shell_species)} species; refusing to guess "
                f"a per-pair floor.")
        if not np.isfinite(_pair_hard).all():
            raise ValueError(
                "pair_hard_min contains non-finite entries; refusing to guess "
                "a per-pair floor.")
        # Padding telemetry: which species needed the floor relaxed, and how
        # many atoms (if any) still went in with no spacing test at all.
        _pad_report: dict = {}
        # For single-box-grain (a single coherent tile covering the
        # whole supercell), skip overlap removal + random padding
        # entirely: the only "close pairs" are PBC-wrap artefacts where
        # the reference lattice is incommensurate with the supercell.
        # Deleting them wrecks the FCC geometry (we'd have to
        # random-pad the shortfall back in); instead we rely on step
        # 7b.5's push pass.
        #
        # For multi-grain *crystalline* builds, keep overlap removal
        # but use a much tighter cutoff so we only delete genuine
        # atom-on-atom collisions from adjacent rotated grains, not
        # merely-distorted boundary neighbours.  Pairs at ~0.7-0.9 x
        # hard_min are boundary distortions that the push step +
        # shell_relax can fix without destroying the crystalline
        # interiors.  pad_min_sep stays at the old tight value so any
        # random padding still respects the hard-core exclusion.
        skip_overlap_removal = bool(single_box_grain)
        is_crystalline_build = (
            int(np.sum(is_crystalline)) > 0
            and crystalline_fraction >= 0.9
        )
        if is_crystalline_build:
            # Moderate dup_cutoff: aggressive enough to clear
            # sub-boundary collisions (where atoms from adjacent
            # rotated grains land almost on top of each other) while
            # lax enough that shell_relax can still spread out the
            # merely-distorted boundary pairs in the 0.7-0.9 x hard_min
            # range.  Paired with random-position padding below to
            # guarantee a uniform per-species atom count across the
            # whole regime ladder.
            dup_cutoff = max(0.5, 0.7 * hard_min_scalar)
            _dup_frac = 0.7
        else:
            dup_cutoff = max(0.5, 0.9 * hard_min_scalar)
            _dup_frac = 0.9
        pad_min_sep = max(0.5, 0.8 * hard_min_scalar)
        # PAIR-RESOLVED duplicate threshold.  `dup_cutoff` above is
        # `_dup_frac * np.min(pair_hard_min)` -- the smallest entry of the
        # whole matrix applied to every pair -- and it was the last remaining
        # scalar distance threshold: push, crystal/crystal collision and
        # padding all already use `_pair_hard`.  That left a band between the
        # global scalar and a given pair's own floor in which nothing was even
        # examined for deletion, so the pair-resolved push handled it instead
        # by moving atoms.  Illustration on mp-1196402_Mn6Ga29: dup_cutoff is
        # 0.9 * min(pair_hard_min) = 0.9 * 2.2181 = 1.996 A there, while the
        # Ga-Ga floor is 2.3872 A, so a Ga-Ga pair in the 1.996-2.3872 A band
        # was never examined for deletion and fell to the pair-resolved push
        # instead.  (2.3872 A is the CLAMPED floor from shells.from_atoms;
        # before that clamp it was the unclamped covalent value, 2.440 A.)
        # `dup_cutoff` is retained because `push_cutoff` below still reads it.
        _dup_matrix = np.maximum(0.5, _dup_frac * _pair_hard)

        # Target per-species counts: reference stoichiometry scaled by
        # (V_box / V_ref) * relative_density.  Rounding is done via
        # formula-unit count so the ratio across species stays exact
        # (matches Supercell._target_species_counts - critical for
        # multi-species compounds like SrTiO3 where simple per-species
        # rounding would drift off stoichiometry by an atom or two).
        #
        # For multi-source grain builds (e.g. graphite+diamond carbon),
        # the ``species_density`` variable above was already
        # weight-averaged across sources, so we scale the target total
        # by that density directly (rather than by the sources[0]
        # ref_volume alone).  For the single-source case both paths
        # give the same answer.
        rel_density = float(getattr(self, "relative_density", 1.0))

        # ``protect_crystallites``: the crystallites are already at crystal
        # density and must not be counted against a relative-density-scaled
        # target, or the surplus drop below will thin them.  Budget them at
        # their actual count and apply relative_density only to the amorphous
        # volume.
        _protect = protect_crystallites
        _xtal_of = None
        if _protect and cells is None:
            # The fast path is only taken when crystalline_fraction >= 0.9999,
            # i.e. every grain is crystalline and there is no amorphous region
            # for relative_density to describe.  Fail loudly rather than
            # silently ignoring the flag.
            raise ValueError(
                "protect_crystallites=True requires a mixed build "
                "(crystalline_fraction < 1): with every grain crystalline "
                "there is no amorphous region for relative_density to apply "
                "to, and the fast tiling path does not compute Voronoi cell "
                "volumes."
            )
        if _protect:
            _isx_arr = np.asarray(is_crystalline, dtype=bool)

            def _xtal_of(gids):
                m = np.zeros(len(gids), dtype=bool)
                if _isx_arr.size:
                    ok = gids >= 0
                    m[ok] = _isx_arr[gids[ok]]
                return m

            _V_xtal = float(sum(float(c["volume"])
                                for c, f in zip(cells, _isx_arr) if f))

            # Half-space test against each protected grain's Voronoi hull, in
            # that grain's seed-local frame.  ``equations`` is [n | d] with
            # n·x + d <= 0 inside, the scipy ConvexHull convention.
            _prot = [(np.asarray(seeds[gi], dtype=np.float64),
                      np.asarray(cells[gi]["equations"], dtype=np.float64))
                     for gi in np.flatnonzero(_isx_arr)
                     if cells[gi] is not None
                     and np.asarray(cells[gi]["equations"]).ndim == 2]

            def _in_protected_cell(points):
                pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
                bad = np.zeros(len(pts), dtype=bool)
                for _seed, _eq in _prot:
                    x = pts - _seed
                    x -= box_dim * np.round(x / box_dim)
                    bad |= (x @ _eq[:, :3].T + _eq[:, 3] <= 0.0).all(axis=1)
                return bad
        _reduced_counts = species_counts.astype(np.int64)
        _divisor = int(np.gcd.reduce(_reduced_counts)) if _reduced_counts.size else 1
        _reduced_counts = _reduced_counts // max(_divisor, 1)
        _atoms_per_formula = int(np.sum(_reduced_counts))
        if _protect:
            # amorphous share only; crystallites added back per species below
            _target_total = float(species_density * (V_box - _V_xtal)
                                  * rel_density)
        elif multi_source:
            _target_total = float(species_density * V_box * rel_density)
        else:
            v_ratio = V_box / max(ref_volume, _EPS)
            _target_total = float(len(ref_numbers)) * v_ratio * rel_density
        _num_formula_units = max(1, int(round(_target_total / max(_atoms_per_formula, 1))))
        target_by_z: dict[int, int] = {
            int(z): int(_reduced_counts[i] * _num_formula_units)
            for i, z in enumerate(unique_species)
        }
        _xtal0_by_z: dict[int, int] = {}
        if _protect:
            # The crystalline share has to be in `target_by_z` BEFORE the
            # removal block below, because that block's species priority reads
            # it ("delete from the species most over target", ~line 1518).
            # But it is the PRE-removal count, and step 7b would treat every
            # crystalline atom deleted at a grain wall as a shortfall to pad --
            # so it is re-taken after removal, further down.
            _xm0 = _xtal_of(grain_ids)
            _xtal0_by_z = {int(z): int(np.sum(numbers[_xm0] == z))
                           for z in target_by_z}
            for z in list(target_by_z):
                target_by_z[z] += _xtal0_by_z[int(z)]

        if len(positions) > 0 and not skip_overlap_removal:
            from scipy.spatial import cKDTree

            # Box is guaranteed orthogonal by the check at the top of
            # this function, so cKDTree's torus-metric boxsize is safe
            # here.  query_pairs returns each (i, j<k) once, vs ase's
            # neighbor_list which returns each pair in both orderings
            # — the downstream remove-set loop tolerates either.  At
            # 196 k atoms / 1.56 Å cutoff this drops the call from
            # ~3.2 s → ~0.1 s.
            tree = cKDTree(positions, boxsize=box_dim)
            # Query radius must cover BOTH tests: the scalar `dup_cutoff` used
            # for amorphous pairs (unchanged), and the pair-resolved
            # crystalline-collision limit, which is larger.  Querying only at
            # dup_cutoff meant an O-O grain collision at 1.6-1.86 A was never
            # even examined -- 1.539 vs a 0.80 x 2.326 = 1.861 A limit.
            # Cover the largest pair-resolved duplicate threshold, not the
            # scalar: querying at `dup_cutoff` would never even return the
            # pairs the matrix is meant to catch.
            _query_r = float(_dup_matrix.max())
            if _protect:
                _query_r = max(_query_r,
                               _XTAL_COLLISION_FRAC * _pair_hard.max())
            pair_arr = tree.query_pairs(_query_r, output_type="ndarray")
            if len(pair_arr) > 0:
                ov_i = pair_arr[:, 0]
                ov_j = pair_arr[:, 1]
                _dv = positions[ov_j] - positions[ov_i]
                _dv -= box_dim * np.round(_dv / box_dim)
                _pair_d = np.linalg.norm(_dv, axis=1)
            else:
                ov_i = np.empty(0, dtype=np.intp)
                ov_j = np.empty(0, dtype=np.intp)
                _pair_d = np.empty(0, dtype=np.float64)
            # Running species counts so priority updates as we remove.
            z_counts = {int(z): int(np.sum(numbers == z)) for z in unique_species}
            _xm = _xtal_of(grain_ids) if _protect else None
            remove: set[int] = set()
            for k in range(len(ov_i)):
                ai, aj = int(ov_i[k]), int(ov_j[k])
                if ai in remove or aj in remove:
                    continue
                _is_xx = _xm is not None and _xm[ai] and _xm[aj]
                _dup_lim = _dup_matrix[_sp_idx[ai], _sp_idx[aj]]
                if not _is_xx and _pair_d[k] >= _dup_lim:
                    # Outside THIS pair's duplicate threshold.  Previously the
                    # scalar `dup_cutoff`, which for a multi-species reference
                    # is one pair's floor imposed on all of them.
                    continue
                if _xm is not None:
                    # A crystalline/amorphous straddling pair always loses the
                    # amorphous atom.  A crystalline/crystalline pair used to be
                    # left alone entirely, on the grounds that deleting either
                    # punches a vacancy into the lattice.  But those pairs are
                    # never repaired downstream: bond_relax runs with the
                    # crystallites frozen, so a pair with BOTH atoms frozen
                    # cannot move at any iteration count (measured: unchanged
                    # from 0 to 800 sweeps), and it then sets the MACE wall's
                    # floor for the whole cell.  A genuine collision -- two
                    # atoms from adjacent grains nearly on top of each other --
                    # is worse than a vacancy, so delete one of those; merely
                    # distorted boundary pairs are still left alone.
                    xi, xj = _xm[ai], _xm[aj]
                    if xi and xj:
                        _lim = (_XTAL_COLLISION_FRAC
                                * _pair_hard[_sp_idx[ai], _sp_idx[aj]])
                        if _pair_d[k] >= _lim:
                            continue
                        pick = ai if self.rng.random() < 0.5 else aj
                        remove.add(pick)
                        z_counts[int(numbers[pick])] -= 1
                        continue
                    if xi != xj:
                        pick = aj if xi else ai
                        remove.add(pick)
                        z_counts[int(numbers[pick])] -= 1
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
                shell_species_idx = shell_species_idx[mask]

        # ---- 7a.9 Re-take the crystalline share AFTER overlap removal ----
        # Adjacent crystalline grains meet at Voronoi walls with random
        # relative orientation, so atoms across a wall land at arbitrary
        # separations and _XTAL_COLLISION_FRAC (= 1.0, the FULL pair floor)
        # deletes one atom of every sub-floor cross-wall pair.  That thins a
        # shell either side of every wall, and the amount scales with total
        # wall area, i.e. with grain COUNT: measured on SiO2, 24 A box,
        # cf 0.75, rd 0.9 -- 4 grains lose 15% of the crystalline atoms,
        # 12 grains 22%, 28 grains 28%, 92 grains 43%.
        #
        # `target_by_z` still carries the PRE-removal crystalline count, so
        # step 7b below reads those deletions as a shortfall and pads them.
        # Padding runs with `exclude=_in_protected_cell`, so every replacement
        # lands in the AMORPHOUS region -- mass moves out of the grains and
        # into the matrix.  (An earlier version of this comment cited "a 213
        # atom crystalline deficit against a 237 atom amorphous excess, the
        # gap being formula-unit rounding".  Do not rely on that: a deficit
        # and an excess of different size cannot both be right, and the
        # 24-atom gap did not reproduce.  Judge the fix on the matrix density
        # ratio below, which does.)  The matrix reaches 1.99x the
        # requested `relative_density` while the grains sit at 0.715 of
        # crystal density, breaking the documented protect_crystallites
        # contract that `relative_density` describes the amorphous region.
        #
        # Re-taking the share here means a crystalline atom lost to a genuine
        # wall collision is simply not replaced, which is the correct
        # behaviour: two grains meeting at a wall cannot both keep every atom.
        # It does NOT restore crystalline density -- that is set by
        # _XTAL_COLLISION_FRAC and is a separate physics choice.
        if _protect and target_by_z:
            _xm1 = _xtal_of(grain_ids)
            for z in list(target_by_z):
                _now = int(np.sum(numbers[_xm1] == z))
                target_by_z[z] += _now - _xtal0_by_z.get(int(z), _now)

        # ---- 7b. Enforce exact per-species target counts ----
        # Bring each species to its reference-scaled target by randomly
        # dropping surplus or padding with new random atoms.  Padding
        # atoms try to respect the hard-core separation but accept
        # looser placement if the retry budget is exhausted (rare for
        # small shortfalls).  Skipped for single-box-grain where the
        # coherent FCC tile should be preserved intact: random padding
        # would break the FCC order at the padded positions, and the
        # visible atom count differs from the liquid-path target by at
        # most a few percent (incommensurate-box wrap artefact), which
        # we accept.
        cell_inv_local = np.linalg.inv(cell_mat)
        # Only the single-box-grain path skips padding entirely (it
        # preserves a coherent tile by construction).  Multi-grain
        # crystalline builds DO need padding to hit the target atom
        # count; we pair it with the tight dup_cutoff above and an
        # aggressive retry budget below so new atoms respect the hard-
        # core spacing and boundary distortions aren't over-destroyed.
        skip_padding = skip_overlap_removal
        if skip_overlap_removal:
            target_by_z = {}   # skip exact-count adjustment
        for z, target in target_by_z.items():
            idx = np.where(numbers == z)[0]
            current = int(len(idx))
            if current > target:
                pool = idx
                if _protect:
                    # trim the amorphous region only
                    _xm = _xtal_of(grain_ids)
                    pool = idx[~_xm[idx]]
                n_drop = min(current - target, len(pool))
                drop = self.rng.choice(pool, size=n_drop, replace=False)
                keep = np.ones(len(numbers), dtype=bool)
                keep[drop] = False
                positions = positions[keep]
                numbers = numbers[keep]
                grain_ids = grain_ids[keep]
                shell_species_idx = shell_species_idx[keep]
            elif current < target and not skip_padding:
                n_missing = target - current
                # Batched cKDTree-based rejection sampling.  Replaces
                # the original serial trial loop (which spent ~130 s
                # for amorphous at 100×100×400 with n_missing≈9k and
                # N_existing≈174k due to its per-trial O(N) distance
                # scan).
                _pad_sp_idx = np.array(
                    [_z_to_row.get(int(_zz), 0) for _zz in numbers],
                    dtype=np.intp)
                added, _, _pad_scale = _sample_padding_atoms(
                    positions, box_dim, n_missing, pad_min_sep, self.rng,
                    exclude=(_in_protected_cell if _protect else None),
                    pair_cutoff=_pair_hard,
                    species_index=_pad_sp_idx,
                    pad_row=_z_to_row.get(int(z), 0),
                )
                # Record any relaxation of the padding floor, and any atom that
                # still had to be placed with NO spacing test, so a build that
                # gave ground says so instead of looking clean.
                if _pad_scale < 1.0:
                    _pad_report.setdefault("relaxed", {})[int(z)] = float(_pad_scale)
                # NO loose-placement fallback.  It placed atoms with no spacing
                # test whatsoever -- measured 535 of 535 on Al2O3 cf=0.95 -- so
                # an over-dense box silently produced a corrupt structure that
                # looked complete.  The floor now backs off without limit; if
                # the sampler still cannot fit them the box genuinely has no
                # room, so say so and carry on short of the target count.
                _n_short = max(0, n_missing - len(added))
                if _n_short:
                    _pad_report.setdefault("short", {})[int(z)] = _n_short
                    print(
                        f"  [tricor] padding: could not place {_n_short} of "
                        f"{n_missing} atoms of Z={int(z)} (floor reached "
                        f"{_pad_scale:.3f} x pair_hard_min). This is a VOLUME "
                        f"failure, not a spacing one: padding never enters a "
                        f"protected crystallite, so when the unprotected "
                        f"volume is saturated no amount of floor relaxation "
                        f"helps -- the sampler stops after 40 consecutive "
                        f"fully-excluded rounds rather than grinding. Continuing "
                        f"{_n_short} short, which changes composition and "
                        f"density.", flush=True,
                    )
                n_missing = len(added)
                if n_missing == 0:
                    continue
                positions = np.concatenate([positions, added], axis=0)
                numbers = np.concatenate(
                    [numbers, np.full(n_missing, z, dtype=np.int64)],
                )
                grain_ids = np.concatenate(
                    [grain_ids, np.full(n_missing, -1, dtype=np.intp)],
                )
                # Padded atoms get assigned to the source of the
                # nearest existing atom (for multi-source builds) or
                # source 0 (single-source - unchanged behaviour).
                if multi_source and len(positions) > n_missing:
                    existing_pos = positions[:-n_missing]
                    existing_shell = shell_species_idx
                    pad_pos = added
                    # Nearest existing atom per padded atom.
                    delta = existing_pos[:, None, :] - pad_pos[None, :, :]
                    frac = delta @ cell_inv_local
                    frac -= np.round(frac)
                    mi = frac @ cell_mat
                    d2 = np.sum(mi * mi, axis=-1)
                    nearest = np.argmin(d2, axis=0)
                    pad_shell = existing_shell[nearest].astype(np.intp)
                else:
                    pad_shell = np.full(n_missing, 0, dtype=np.intp)
                shell_species_idx = np.concatenate(
                    [shell_species_idx, pad_shell]
                )

        # ---- 7b.5. Push any residual close pairs apart ----
        # `numbers` may have changed length via overlap removal and padding,
        # so the species index has to be rebuilt here rather than reused.
        _sp_idx_after_removal = np.array(
            [_z_to_row.get(int(z), 0) for z in numbers], dtype=np.intp)
        # For crystalline grain builds, only push pairs below the tight
        # dup_cutoff so boundary distortions (at 0.5-0.9 x hard_min)
        # survive the pre-conditioner and get resolved by shell_relax
        # via bond + repulsion springs.  For non-crystalline / liquid-
        # path cases we still push to hard_min.
        push_cutoff = dup_cutoff if is_crystalline_build else hard_min_scalar
        # Pair-resolved push for EVERY build, crystalline included.  This was
        # gated on `not is_crystalline_build` to preserve the tight scalar
        # cutoff there, but that left the cf >= 0.9 regime pushing every pair to
        # min(pair_hard_min) -- one pair's floor applied to all of them, which
        # is the exact collapse this matrix exists to undo.  Crystalline atoms
        # are pinned by `skip_mask` when protect_crystallites is on, and a pair
        # with BOTH atoms protected is skipped outright, so crystal interiors
        # are untouched; only the free atoms move.
        _push_matrix = _pair_hard
        positions = _push_close_pairs_apart(
            positions, numbers, cell_mat, pbc=self.reference_atoms.pbc,
            push_cutoff=push_cutoff, max_iter=40,
            pair_cutoff=_push_matrix,
            species_index=(_sp_idx_after_removal if _push_matrix is not None
                           else None),
            skip_mask=(_xtal_of(grain_ids) if _protect else None),
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
        if _pad_report:
            atoms.info["padding_report"] = _pad_report
        atoms.info["grain_radius"] = float(radius)

        self._grain_ids = grain_ids
        self._grain_seeds = seeds.copy()
        # Persist Voronoi cells + master atom blocks so
        # ``Supercell.refine_grains`` can re-tile a single grain
        # without recomputing the global Voronoi tessellation.  Each
        # entry of ``_grain_cells`` is the cell dict from
        # ``_periodic_voronoi_3d`` (vertices, hull equations,
        # simplices, volume, all in seed-local coordinates).
        # ``_grain_masters`` is the per-source tiled-out reference
        # atom block ({"positions", "numbers"}); ``_grain_source``
        # tells refine_grains which source each grain came from.
        self._grain_cells = cells
        self._grain_box_dim = box_dim.copy()
        self._grain_masters = masters
        self._grain_master_lattice = ref_cell.copy()
        self._grain_radius_value = float(radius)
        self._grain_is_crystalline = is_crystalline.copy()
        self._grain_rotations_initial = rotations.copy()
        # Publish the per-atom shell-species assignment so the
        # relaxer can pull graphite grains toward sp² targets and
        # diamond grains toward sp³ targets.  Only set when
        # multi-source was requested; otherwise callers fall back to
        # searchsorted on atomic numbers.
        if multi_source:
            self._atom_shell_species_index = shell_species_idx
            self._grain_source = grain_source
        else:
            self._atom_shell_species_index = None
            self._grain_source = grain_source  # always save (refine uses it)
        return atoms
