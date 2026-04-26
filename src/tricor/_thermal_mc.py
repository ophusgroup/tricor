"""Temperature-dependent Metropolis Monte-Carlo relaxation.

Sits alongside :meth:`Supercell.shell_relax`: same spring-network
energy (bond + angle + repulsion from ``shell_target``), but moves
atoms by Metropolis accept/reject with a temperature schedule.  Three
move types:

- **Displacement**: pick an atom, propose a Gaussian random
  displacement ``Δr ~ N(0, σ² I_3)``, accept with probability
  ``min(1, exp(-ΔE / T))``.
- **Species swap**: pick two atoms of different species, exchange
  virtual-species labels (positions unchanged), accept the same way.
  For SiO₂ this is a Si↔O position swap (labels move, physics is
  identical); for the carbon composite target this is a
  coordination-flip move (sp²↔sp³ at fixed position).
- **Smart / force-biased displacement**: propose via one forward-Euler
  step of the over-damped Langevin equation, with a proposal-density
  correction on the acceptance so detailed balance is preserved.

Energy definition (match shell_relax's forces as gradients):

- Bond:  E_bond = ½ k_bond (r - r_target)²
- Angle: E_angle = ½ k_angle (φ - φ_target)² / 3
  (the 1/3 matches shell_relax's per-atom cost share for triplets.)
- Hard core: E_hard(r) = 4 k_rep r_wall (h - ln(1+h))  for r < r_wall,
  h = r_wall/r - 1.  0 for r ≥ r_wall.  Its gradient matches
  ``4 k_rep (h + h²)`` from shell_relax.
- Non-bonded clearance: E_push(r) = k_rep r_wall (h - ln(1+h))
  for non-bonded pairs inside the push-wall.

Captured state lives on ``cell.thermal_relax_history`` with the same
layout as ``shell_relax_history``, so ``export_trajectory_html``
works unchanged.
"""

from __future__ import annotations

import math
import time
from typing import Callable

import numpy as np

try:
    import numba
    from numba import njit
    HAS_NUMBA = True
except ImportError:  # pragma: no cover
    HAS_NUMBA = False


# ---------------------------------------------------------------------------
# Local-energy kernels (numba)
# ---------------------------------------------------------------------------


if HAS_NUMBA:

    @njit(cache=False, fastmath=False, inline="always")
    def _min_image(dx, dy, dz, cell_mat, cell_inv):
        """Return the minimum-image displacement of ``(dx, dy, dz)``."""
        # Fractional coordinates.
        fx = cell_inv[0, 0] * dx + cell_inv[1, 0] * dy + cell_inv[2, 0] * dz
        fy = cell_inv[0, 1] * dx + cell_inv[1, 1] * dy + cell_inv[2, 1] * dz
        fz = cell_inv[0, 2] * dx + cell_inv[1, 2] * dy + cell_inv[2, 2] * dz
        fx -= math.floor(fx + 0.5)
        fy -= math.floor(fy + 0.5)
        fz -= math.floor(fz + 0.5)
        mx = cell_mat[0, 0] * fx + cell_mat[1, 0] * fy + cell_mat[2, 0] * fz
        my = cell_mat[0, 1] * fx + cell_mat[1, 1] * fy + cell_mat[2, 1] * fz
        mz = cell_mat[0, 2] * fx + cell_mat[1, 2] * fy + cell_mat[2, 2] * fz
        return mx, my, mz

    @njit(cache=False, fastmath=False)
    def _repulsion_energy_pair(
        r, r_wall, k_rep,
    ):
        """Integrated repulsion potential.  Returns ``0`` for r ≥ r_wall.

        Exact primitive of ``F = 4 k_rep (h + h²)`` (hard core) /
        ``F = k_rep (h + h²)`` (non-bonded); the caller picks ``k_rep``
        to include the 4× for hard-core or 1× for non-bonded.
        """
        if r >= r_wall or r <= 0.0:
            return 0.0
        h = r_wall / r - 1.0
        # E = k_rep * r_wall * (h - log1p(h)).  log1p(h) = ln(1 + h).
        return k_rep * r_wall * (h - math.log1p(h))

    @njit(cache=False, fastmath=False)
    def _local_energy(
        atom_i,
        trial_x, trial_y, trial_z,
        positions,
        species_idx,
        cell_mat, cell_inv,
        # Bond CSR: bonds_atom_start[a:a+1], bonds_atom_list -> bond indices
        bond_atom_start, bond_atom_list,
        bond_i, bond_j, bond_r_target,
        k_bond,
        # Triplet CSR: triplets_atom_start, triplets_atom_list -> triplet idx
        tri_atom_start, tri_atom_list,
        tri_center, tri_a, tri_b, tri_phi_target,
        k_angle,
        # Repulsion (cached CSR neighbour list; see
        # _build_thermal_topology).
        rep_atom_start, rep_atom_list,
        hard_core, nonbond_push,
        k_rep,
        bonded_flat,  # sorted int64 keys: i * num_atoms + j with i < j
        num_atoms,
        # Position-restraint term: ½ k_restraint ‖r - r_initial‖² per atom.
        # Min-image-corrected so the restraint doesn't drag atoms across
        # the cell when they wrap.  k_restraint = 0 disables the term.
        r_initial,
        k_restraint,
    ):
        """Scalar energy of all bond / angle / repulsion terms involving
        ``atom_i`` when atom_i sits at (trial_x, trial_y, trial_z).
        ``positions`` is the current whole-system array (with atom_i at
        its OLD position if we haven't moved it yet)."""
        total = 0.0

        # --- bonds involving atom_i ---
        bs = bond_atom_start[atom_i]
        be = bond_atom_start[atom_i + 1]
        for k in range(bs, be):
            bi = bond_atom_list[k]
            a = bond_i[bi]
            b = bond_j[bi]
            other = b if a == atom_i else a
            dx = positions[other, 0] - trial_x
            dy = positions[other, 1] - trial_y
            dz = positions[other, 2] - trial_z
            mx, my, mz = _min_image(dx, dy, dz, cell_mat, cell_inv)
            r = math.sqrt(mx * mx + my * my + mz * mz)
            delta = r - bond_r_target[bi]
            total += 0.5 * k_bond * delta * delta

        # --- triplets involving atom_i (as centre, a, or b) ---
        ts = tri_atom_start[atom_i]
        te = tri_atom_start[atom_i + 1]
        for k in range(ts, te):
            ti = tri_atom_list[k]
            c = tri_center[ti]
            a_ = tri_a[ti]
            b_ = tri_b[ti]
            # Positions of the three atoms, using trial for atom_i.
            if c == atom_i:
                cx = trial_x; cy = trial_y; cz = trial_z
            else:
                cx = positions[c, 0]; cy = positions[c, 1]; cz = positions[c, 2]
            if a_ == atom_i:
                ax = trial_x; ay = trial_y; az = trial_z
            else:
                ax = positions[a_, 0]; ay = positions[a_, 1]; az = positions[a_, 2]
            if b_ == atom_i:
                bx = trial_x; by = trial_y; bz = trial_z
            else:
                bx = positions[b_, 0]; by = positions[b_, 1]; bz = positions[b_, 2]
            # Vectors centre -> a and centre -> b, min-image.
            vax = ax - cx; vay = ay - cy; vaz = az - cz
            vax, vay, vaz = _min_image(vax, vay, vaz, cell_mat, cell_inv)
            vbx = bx - cx; vby = by - cy; vbz = bz - cz
            vbx, vby, vbz = _min_image(vbx, vby, vbz, cell_mat, cell_inv)
            ra = math.sqrt(vax * vax + vay * vay + vaz * vaz)
            rb = math.sqrt(vbx * vbx + vby * vby + vbz * vbz)
            if ra < 1e-10 or rb < 1e-10:
                continue
            cos_phi = (vax * vbx + vay * vby + vaz * vbz) / (ra * rb)
            if cos_phi > 1.0:
                cos_phi = 1.0
            elif cos_phi < -1.0:
                cos_phi = -1.0
            phi = math.acos(cos_phi)
            dphi = phi - tri_phi_target[ti]
            # Full triplet energy (½ k dphi², no /3).  When atom_i
            # moves, the whole triplet's energy changes by
            # E_new - E_old, so ΔE must track the full triplet energy.
            # (The /3 factor appears only in shell_relax's atom_cost
            # when the triplet cost is split across its three atoms.)
            total += 0.5 * k_angle * dphi * dphi

        # --- repulsion involving atom_i (cached CSR neighbour list) ---
        # The cache is built at rebuild time by
        # ``_build_thermal_topology`` via ASE's neighbor_list (spatial
        # hash).  We iterate only over the atoms within ``rep_cutoff``
        # of atom_i at that time; ``rep_cutoff`` includes a margin so
        # atoms drifting a few step_sigma between rebuilds remain
        # captured.
        sp_i = species_idx[atom_i]
        rs = rep_atom_start[atom_i]
        re = rep_atom_start[atom_i + 1]
        for idx in range(rs, re):
            j = rep_atom_list[idx]
            if j == atom_i:
                continue
            dx = positions[j, 0] - trial_x
            dy = positions[j, 1] - trial_y
            dz = positions[j, 2] - trial_z
            mx, my, mz = _min_image(dx, dy, dz, cell_mat, cell_inv)
            r2 = mx * mx + my * my + mz * mz
            sp_j = species_idx[j]
            r_hard = hard_core[sp_i, sp_j]
            r_push = nonbond_push[sp_i, sp_j]
            rwall_sq = r_push * r_push
            if r2 >= rwall_sq:
                continue
            r = math.sqrt(r2)
            total += _repulsion_energy_pair(r, r_hard, 4.0 * k_rep)
            if atom_i < j:
                lo = atom_i; hi = j
            else:
                lo = j; hi = atom_i
            key = np.int64(lo) * np.int64(num_atoms) + np.int64(hi)
            left = 0
            right = bonded_flat.shape[0]
            is_bonded = False
            while left < right:
                mid = (left + right) // 2
                if bonded_flat[mid] == key:
                    is_bonded = True
                    break
                elif bonded_flat[mid] < key:
                    left = mid + 1
                else:
                    right = mid
            if not is_bonded:
                total += _repulsion_energy_pair(r, r_push, k_rep)

        # --- position restraint (atom_i only) ---
        if k_restraint > 0.0:
            dx = trial_x - r_initial[atom_i, 0]
            dy = trial_y - r_initial[atom_i, 1]
            dz = trial_z - r_initial[atom_i, 2]
            mx, my, mz = _min_image(dx, dy, dz, cell_mat, cell_inv)
            total += 0.5 * k_restraint * (mx * mx + my * my + mz * mz)
        return total

    @njit(cache=False, fastmath=False)
    def _local_force(
        atom_i,
        trial_x, trial_y, trial_z,
        positions,
        species_idx,
        cell_mat, cell_inv,
        bond_atom_start, bond_atom_list,
        bond_i, bond_j, bond_r_target,
        k_bond,
        tri_atom_start, tri_atom_list,
        tri_center, tri_a, tri_b, tri_phi_target,
        k_angle,
        rep_atom_start, rep_atom_list,
        hard_core, nonbond_push,
        k_rep,
        bonded_flat,
        num_atoms,
        r_initial,
        k_restraint,
    ):
        """Force on atom_i (3-vector) when it sits at (trial_x, trial_y,
        trial_z).  Matches shell_relax's bond + angle + repulsion
        forces exactly — same formulas.
        """
        fx = 0.0; fy = 0.0; fz = 0.0

        # --- bonds ---
        bs = bond_atom_start[atom_i]
        be = bond_atom_start[atom_i + 1]
        for k in range(bs, be):
            bi = bond_atom_list[k]
            a = bond_i[bi]
            b = bond_j[bi]
            other = b if a == atom_i else a
            dx = positions[other, 0] - trial_x
            dy = positions[other, 1] - trial_y
            dz = positions[other, 2] - trial_z
            mx, my, mz = _min_image(dx, dy, dz, cell_mat, cell_inv)
            r = math.sqrt(mx * mx + my * my + mz * mz)
            if r < 1e-10:
                continue
            # F_i = k (r - r_t) * hat_ij  (hat points i->j)
            coeff = k_bond * (r - bond_r_target[bi]) / r
            fx += coeff * mx
            fy += coeff * my
            fz += coeff * mz

        # --- angles ---
        ts = tri_atom_start[atom_i]
        te = tri_atom_start[atom_i + 1]
        for k in range(ts, te):
            ti = tri_atom_list[k]
            c = tri_center[ti]
            ai = tri_a[ti]
            bi = tri_b[ti]
            # Get each atom's position, using trial for atom_i.
            if c == atom_i:
                cx = trial_x; cy = trial_y; cz = trial_z
            else:
                cx = positions[c, 0]; cy = positions[c, 1]; cz = positions[c, 2]
            if ai == atom_i:
                ax = trial_x; ay = trial_y; az = trial_z
            else:
                ax = positions[ai, 0]; ay = positions[ai, 1]; az = positions[ai, 2]
            if bi == atom_i:
                bx = trial_x; by = trial_y; bz = trial_z
            else:
                bx = positions[bi, 0]; by = positions[bi, 1]; bz = positions[bi, 2]
            vax = ax - cx; vay = ay - cy; vaz = az - cz
            vax, vay, vaz = _min_image(vax, vay, vaz, cell_mat, cell_inv)
            vbx = bx - cx; vby = by - cy; vbz = bz - cz
            vbx, vby, vbz = _min_image(vbx, vby, vbz, cell_mat, cell_inv)
            ra = math.sqrt(vax * vax + vay * vay + vaz * vaz)
            rb = math.sqrt(vbx * vbx + vby * vby + vbz * vbz)
            if ra < 1e-10 or rb < 1e-10:
                continue
            cos_phi = (vax * vbx + vay * vby + vaz * vbz) / (ra * rb)
            if cos_phi > 1.0:
                cos_phi = 1.0
            elif cos_phi < -1.0:
                cos_phi = -1.0
            phi = math.acos(cos_phi)
            sin_phi = math.sqrt(max(1e-20, 1.0 - cos_phi * cos_phi))
            dphi = phi - tri_phi_target[ti]
            hax = vax / ra; hay = vay / ra; haz = vaz / ra
            hbx = vbx / rb; hby = vby / rb; hbz = vbz / rb
            # e_perp_a = (hat_b - c * hat_a) / sin_phi
            eax = (hbx - cos_phi * hax) / sin_phi
            eay = (hby - cos_phi * hay) / sin_phi
            eaz = (hbz - cos_phi * haz) / sin_phi
            # e_perp_b = (hat_a - c * hat_b) / sin_phi
            ebx = (hax - cos_phi * hbx) / sin_phi
            eby = (hay - cos_phi * hby) / sin_phi
            ebz = (haz - cos_phi * hbz) / sin_phi
            coeff_a = k_angle * dphi / ra
            coeff_b = k_angle * dphi / rb
            fax = coeff_a * eax; fay = coeff_a * eay; faz = coeff_a * eaz
            fbx = coeff_b * ebx; fby = coeff_b * eby; fbz = coeff_b * ebz
            if ai == atom_i:
                fx += fax; fy += fay; fz += faz
            if bi == atom_i:
                fx += fbx; fy += fby; fz += fbz
            if c == atom_i:
                fx -= (fax + fbx); fy -= (fay + fby); fz -= (faz + fbz)

        # --- repulsion (cached CSR neighbour list) ---
        sp_i = species_idx[atom_i]
        rs = rep_atom_start[atom_i]
        re = rep_atom_start[atom_i + 1]
        for idx in range(rs, re):
            j = rep_atom_list[idx]
            if j == atom_i:
                continue
            dx = positions[j, 0] - trial_x
            dy = positions[j, 1] - trial_y
            dz = positions[j, 2] - trial_z
            mx, my, mz = _min_image(dx, dy, dz, cell_mat, cell_inv)
            r2 = mx * mx + my * my + mz * mz
            sp_j = species_idx[j]
            r_hard = hard_core[sp_i, sp_j]
            r_push = nonbond_push[sp_i, sp_j]
            r_wall_max = r_push if r_push > r_hard else r_hard
            if r2 >= r_wall_max * r_wall_max:
                continue
            r = math.sqrt(r2)
            # Hard core (always).
            if r < r_hard and r > 1e-10:
                h = r_hard / r - 1.0
                # Force magnitude on i: 4 k_rep (h + h²), direction
                # AWAY from j (i.e., along -hat_ij = hat_ji).
                mag = 4.0 * k_rep * (h + h * h)
                fx -= mag * mx / r
                fy -= mag * my / r
                fz -= mag * mz / r
            # Non-bonded push (if not bonded).
            if r < r_push and r > 1e-10:
                if atom_i < j:
                    lo = atom_i; hi = j
                else:
                    lo = j; hi = atom_i
                key = np.int64(lo) * np.int64(num_atoms) + np.int64(hi)
                left = 0; right = bonded_flat.shape[0]
                is_bonded = False
                while left < right:
                    mid = (left + right) // 2
                    if bonded_flat[mid] == key:
                        is_bonded = True
                        break
                    elif bonded_flat[mid] < key:
                        left = mid + 1
                    else:
                        right = mid
                if not is_bonded:
                    h = r_push / r - 1.0
                    mag = k_rep * (h + h * h)
                    fx -= mag * mx / r
                    fy -= mag * my / r
                    fz -= mag * mz / r

        # --- position restraint: F_i = -k_restraint · (r_i - r_initial_i) ---
        # min-image-corrected so wrap-around doesn't cause a huge tug.
        if k_restraint > 0.0:
            dx = trial_x - r_initial[atom_i, 0]
            dy = trial_y - r_initial[atom_i, 1]
            dz = trial_z - r_initial[atom_i, 2]
            mx, my, mz = _min_image(dx, dy, dz, cell_mat, cell_inv)
            fx -= k_restraint * mx
            fy -= k_restraint * my
            fz -= k_restraint * mz
        return fx, fy, fz


# ---------------------------------------------------------------------------
# Topology builder (Python-side; one-shot per rebuild interval)
# ---------------------------------------------------------------------------


def detect_grain_boundary_atoms(
    atoms,
    grain_ids: np.ndarray,
    grain_seeds: np.ndarray,
    *,
    boundary_width: float | None = None,
    pair_peak_max: float | None = None,
) -> np.ndarray:
    """Return a bool mask of atoms on a grain boundary.

    An atom is ``boundary`` if the gap between the distance to its own
    grain seed and the distance to its nearest foreign seed is less
    than ``boundary_width``.  Deep-interior atoms (gap bigger) come
    back as ``False``.

    Mirrors the logic in :meth:`Supercell.shell_relax`.

    Parameters
    ----------
    atoms
        ASE ``Atoms`` object.
    grain_ids
        Per-atom integer grain index (``-1`` = amorphous / not in any grain).
    grain_seeds
        ``(num_grains, 3)`` array of grain seed positions.
    boundary_width
        Width of the "boundary layer" in Å.  Defaults to
        ``0.5 × max(pair_peak)`` from the shell target (pass via
        ``pair_peak_max``) or ~1.3 Å when no shell target is available.
    pair_peak_max
        If ``boundary_width`` is ``None``, compute it from this.

    Returns
    -------
    is_boundary : (num_atoms,) bool
        ``True`` for boundary atoms.  Use ``~is_boundary`` to get the
        interior mask for freezing.
    """
    num_atoms = len(atoms)
    grain_ids = np.asarray(grain_ids, dtype=np.intp)
    grain_seeds = np.asarray(grain_seeds, dtype=np.float64)
    if boundary_width is None:
        bw = 1.3 if pair_peak_max is None else 0.5 * float(pair_peak_max)
    else:
        bw = float(boundary_width)

    cell_mat = np.asarray(atoms.cell.array, dtype=np.float64)
    cell_inv = np.linalg.inv(cell_mat)
    pos = np.asarray(atoms.positions, dtype=np.float64)

    is_boundary = np.ones(num_atoms, dtype=bool)
    n_seeds = grain_seeds.shape[0]
    if n_seeds < 2:
        return is_boundary  # one grain - no boundary to detect

    _bchunk = max(1, 25_000_000 // max(n_seeds, 1))
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
            if (dist_other - dist_own) * 0.5 > bw:
                is_boundary[ia] = False
    return is_boundary


def _rotation_matrix(axis: np.ndarray, theta: float) -> np.ndarray:
    """Rodrigues' formula for a rotation about a unit axis by angle theta."""
    ax = axis / max(float(np.linalg.norm(axis)), 1e-12)
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    C = 1.0 - c
    x, y, z = float(ax[0]), float(ax[1]), float(ax[2])
    return np.array([
        [c + x * x * C,     x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C,     y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ], dtype=np.float64)


def _try_grain_rigid_moves(
    positions: np.ndarray,
    species_idx: np.ndarray,
    grain_ids: np.ndarray,
    cell_mat: np.ndarray,
    cell_inv: np.ndarray,
    topo: dict,
    k_bond: float, k_angle: float, k_rep: float,
    T: float,
    sigma_rot: float,
    sigma_trans: float,
    rng: np.random.Generator,
    r_initial: np.ndarray,
    k_restraint: float,
) -> tuple[int, int]:
    """Try one rigid-body transform per grain.

    For each grain with > 0 atoms, propose a small rotation about
    its centre of mass plus a small translation; compute total
    energy before and after via :func:`_total_energy_fast`; accept
    with Metropolis probability at temperature ``T``.

    Returns ``(n_accepted, n_tried)``.

    Intra-grain bonds and triplets are invariant under rigid
    transforms, so only cross-grain interactions drive ΔE - which
    is exactly what lets grains collectively re-orient to relieve
    grain-boundary strain.  Amorphous atoms (``grain_ids == -1``)
    are never moved by this path (they're not a rigid body).
    """
    num_atoms = positions.shape[0]
    unique_grains = [int(g) for g in np.unique(grain_ids) if int(g) >= 0]
    n_acc = 0; n_try = 0

    # Compute initial total energy once.
    E_curr, *_ = _total_energy_fast(
        positions, species_idx, cell_mat, cell_inv,
        topo["bond_i"], topo["bond_j"], topo["bond_r_target"], float(k_bond),
        topo["tri_center"], topo["tri_a"], topo["tri_b"], topo["tri_phi_target"],
        float(k_angle),
        topo["rep_atom_start"], topo["rep_atom_list"],
        topo["hard_core"], topo["nonbond_push"], float(k_rep),
        topo["bonded_flat"], num_atoms,
        r_initial, float(k_restraint),
    )

    for gid in unique_grains:
        mask = (grain_ids == gid)
        if int(np.sum(mask)) < 2:
            continue
        n_try += 1
        # Compute centre of mass (minimum-image-safe via the first
        # atom as reference).
        ref = positions[np.argmax(mask)]  # first atom of this grain
        rel = positions[mask] - ref
        frac = rel @ cell_inv
        frac -= np.rint(frac)
        rel_wrapped = frac @ cell_mat
        com = ref + rel_wrapped.mean(axis=0)

        # Sample rotation (axis-uniform on sphere, angle Gaussian).
        axis = rng.standard_normal(3)
        theta = float(rng.standard_normal()) * float(sigma_rot)
        R = _rotation_matrix(axis, theta)
        # Sample translation.
        dt = rng.standard_normal(3) * float(sigma_trans)

        # Apply: r_new = R · (r - com) + com + dt
        old_pos = positions[mask].copy()
        positions[mask] = (old_pos - com) @ R.T + com + dt[None, :]

        E_new, *_ = _total_energy_fast(
            positions, species_idx, cell_mat, cell_inv,
            topo["bond_i"], topo["bond_j"], topo["bond_r_target"], float(k_bond),
            topo["tri_center"], topo["tri_a"], topo["tri_b"], topo["tri_phi_target"],
            float(k_angle),
            topo["rep_atom_start"], topo["rep_atom_list"],
            topo["hard_core"], topo["nonbond_push"], float(k_rep),
            topo["bonded_flat"], num_atoms,
            r_initial, float(k_restraint),
        )
        dE = float(E_new - E_curr)
        if dE <= 0.0 or rng.random() < math.exp(-dE / max(float(T), 1e-12)):
            # Accept: positions already updated, take the new energy.
            E_curr = E_new
            n_acc += 1
        else:
            # Revert.
            positions[mask] = old_pos
    return n_acc, n_try


def _rebuild_rep_neighbors(atoms, rep_cutoff: float) -> tuple[np.ndarray, np.ndarray]:
    """Rebuild just the rep-neighbour CSR (cheaper than rebuilding
    the full bond topology).  Uses ASE's spatial-hashed
    ``neighbor_list``.  Returns ``(rep_atom_start, rep_atom_list)``.
    """
    from ase.neighborlist import neighbor_list
    num_atoms = len(atoms)
    ri, rj = neighbor_list("ij", atoms, float(rep_cutoff))
    ri = np.asarray(ri, dtype=np.intp)
    rj = np.asarray(rj, dtype=np.intp)
    if ri.size == 0:
        start = np.zeros(num_atoms + 1, dtype=np.intp)
        return start, np.zeros(0, dtype=np.intp)
    order = np.argsort(ri, kind="stable")
    ri_s = ri[order]
    rj_s = rj[order]
    counts = np.bincount(ri_s, minlength=num_atoms)
    rep_atom_start = np.zeros(num_atoms + 1, dtype=np.intp)
    rep_atom_start[1:] = np.cumsum(counts)
    return rep_atom_start, rj_s.astype(np.intp)


def _build_thermal_topology(
    atoms,
    species_idx: np.ndarray,
    shell_target,
    *,
    hard_core_scale: float,
    nonbond_push_scale: float,
) -> dict:
    """Build the per-atom bond / triplet / species-pair index arrays.

    Mirrors shell_relax's ``rebuild_topology`` in spirit but returns
    plain numpy arrays + a sorted bonded-pair lookup table, ready for
    the numba kernel.  Called once per thermal_relax rebuild
    (typically every ``neighbor_update_interval`` sweeps).
    """
    from ase.neighborlist import neighbor_list

    _EPS = 1e-12
    num_atoms = len(atoms)
    species = np.asarray(shell_target.species, dtype=np.int64)
    num_sp = int(species.size)
    coord_target = np.asarray(shell_target.coordination_target, dtype=np.float64)
    pair_peak = np.asarray(shell_target.pair_peak, dtype=np.float64)
    pair_hard_min = np.asarray(shell_target.pair_hard_min, dtype=np.float64)
    pair_inner = np.asarray(shell_target.pair_inner, dtype=np.float64)
    angle_mode_deg = np.asarray(shell_target.angle_mode_deg, dtype=np.float64)
    angle_lookup = np.asarray(shell_target.angle_lookup, dtype=np.intp)
    angle_enabled_mask = np.asarray(
        getattr(shell_target, "angle_enabled_mask",
                np.ones(angle_mode_deg.size, dtype=bool)),
        dtype=bool,
    )

    hard_core = np.maximum(pair_hard_min, pair_inner) * float(hard_core_scale)
    mask_zero = hard_core < _EPS
    hard_core[mask_zero] = 0.4 * pair_peak[mask_zero]
    global_floor = float(np.min(pair_peak[pair_peak > _EPS])) * 0.4 if np.any(pair_peak > _EPS) else 1.0
    hard_core[hard_core < _EPS] = global_floor
    nonbond_push = pair_peak * 1.5 * float(nonbond_push_scale)
    nonbond_push[nonbond_push < _EPS] = float(np.max(pair_peak)) * 1.5

    # ---- bond graph (greedy, like shell_relax) ----
    cutoff = float(shell_target.max_pair_outer * 1.2)
    nl_i, nl_j, nl_d = neighbor_list("ijd", atoms, cutoff)
    coord_target_int = np.round(coord_target).astype(np.intp)
    k_per_species = np.array(
        [int(np.round(coord_target[s].sum())) for s in range(num_sp)],
        dtype=np.intp,
    )
    k_atom = np.array(
        [int(k_per_species[species_idx[a]]) for a in range(num_atoms)],
        dtype=np.intp,
    )

    cell_mat = np.asarray(atoms.cell.array, dtype=np.float64)
    cell_inv = np.linalg.inv(cell_mat)

    def _min_image_np(delta):
        frac = delta @ cell_inv
        frac -= np.rint(frac)
        return frac @ cell_mat

    nl_vecs = _min_image_np(
        atoms.positions[nl_j] - atoms.positions[nl_i]
    )
    nl_hats = nl_vecs / np.maximum(nl_d, _EPS)[:, None]

    dist_order = np.argsort(nl_d)
    bond_i_list: list[int] = []
    bond_j_list: list[int] = []
    bond_rt_list: list[float] = []
    bonded_set: set[tuple[int, int]] = set()
    bonded_neighbors: list[list[int]] = [[] for _ in range(num_atoms)]
    bond_hats_per_atom: list[list[np.ndarray]] = [[] for _ in range(num_atoms)]
    bond_count = np.zeros(num_atoms, dtype=np.intp)
    bond_count_pair = np.zeros((num_atoms, num_sp), dtype=np.intp)
    min_accept_angle = np.deg2rad(60.0)

    def _species_pair_ok(ai, aj):
        si, sj = species_idx[ai], species_idx[aj]
        if bond_count_pair[ai, sj] >= coord_target_int[si, sj]:
            return False
        if bond_count_pair[aj, si] >= coord_target_int[sj, si]:
            return False
        return True

    def _angle_ok(ai, aj, hat_ij):
        for existing in bond_hats_per_atom[ai]:
            if float(np.dot(existing, hat_ij)) > np.cos(min_accept_angle):
                continue
        for h_ in bond_hats_per_atom[ai]:
            if float(np.dot(h_, hat_ij)) > np.cos(min_accept_angle):
                return False
        for h_ in bond_hats_per_atom[aj]:
            if float(np.dot(h_, -hat_ij)) > np.cos(min_accept_angle):
                return False
        return True

    def _accept_bond(ai, aj, hat_ij, dist):
        si, sj = int(species_idx[ai]), int(species_idx[aj])
        bond_i_list.append(int(ai))
        bond_j_list.append(int(aj))
        bond_rt_list.append(float(pair_peak[si, sj]))
        key = (int(ai), int(aj)) if ai < aj else (int(aj), int(ai))
        bonded_set.add(key)
        bond_count[ai] += 1
        bond_count[aj] += 1
        bond_count_pair[ai, sj] += 1
        bond_count_pair[aj, si] += 1
        bonded_neighbors[ai].append(int(aj))
        bonded_neighbors[aj].append(int(ai))
        bond_hats_per_atom[ai].append(hat_ij.astype(np.float64))
        bond_hats_per_atom[aj].append((-hat_ij).astype(np.float64))

    # First pass with angle check.
    for k in dist_order:
        ai = int(nl_i[k]); aj = int(nl_j[k])
        if ai >= aj:
            continue
        if bond_count[ai] >= k_atom[ai] or bond_count[aj] >= k_atom[aj]:
            continue
        if not _species_pair_ok(ai, aj):
            continue
        if not _angle_ok(ai, aj, nl_hats[k]):
            continue
        _accept_bond(ai, aj, nl_hats[k], float(nl_d[k]))

    # Second pass: relax angle constraint for still-unbonded atoms.
    for k in dist_order:
        ai = int(nl_i[k]); aj = int(nl_j[k])
        if ai >= aj:
            continue
        if bond_count[ai] >= k_atom[ai] or bond_count[aj] >= k_atom[aj]:
            continue
        if not _species_pair_ok(ai, aj):
            continue
        _accept_bond(ai, aj, nl_hats[k], float(nl_d[k]))

    bond_i = np.asarray(bond_i_list, dtype=np.intp)
    bond_j = np.asarray(bond_j_list, dtype=np.intp)
    bond_r_target = np.asarray(bond_rt_list, dtype=np.float64)
    num_bonds = bond_i.shape[0]

    # CSR: bonds-by-atom.
    atom_bond_count = np.zeros(num_atoms + 1, dtype=np.intp)
    for bi in range(num_bonds):
        atom_bond_count[bond_i[bi] + 1] += 1
        atom_bond_count[bond_j[bi] + 1] += 1
    bond_atom_start = np.cumsum(atom_bond_count)
    bond_atom_list = np.zeros(bond_atom_start[-1], dtype=np.intp)
    write = np.zeros(num_atoms, dtype=np.intp)
    for bi in range(num_bonds):
        a = bond_i[bi]; b = bond_j[bi]
        bond_atom_list[bond_atom_start[a] + write[a]] = bi; write[a] += 1
        bond_atom_list[bond_atom_start[b] + write[b]] = bi; write[b] += 1

    # ---- triplet graph ----
    angle_mode_rad = np.deg2rad(angle_mode_deg)
    tri_c_list: list[int] = []
    tri_a_list: list[int] = []
    tri_b_list: list[int] = []
    tri_pt_list: list[float] = []
    for atom in range(num_atoms):
        bn = bonded_neighbors[atom]
        if len(bn) < 2:
            continue
        s_c = int(species_idx[atom])
        for ia in range(len(bn)):
            for ib in range(ia + 1, len(bn)):
                s_a = int(species_idx[bn[ia]])
                s_b = int(species_idx[bn[ib]])
                if s_a <= s_b:
                    tidx = int(angle_lookup[s_c, s_a, s_b])
                else:
                    tidx = int(angle_lookup[s_c, s_b, s_a])
                if tidx < 0 or not angle_enabled_mask[tidx]:
                    continue
                tri_c_list.append(int(atom))
                tri_a_list.append(int(bn[ia]))
                tri_b_list.append(int(bn[ib]))
                tri_pt_list.append(float(angle_mode_rad[tidx]))

    tri_center = np.asarray(tri_c_list, dtype=np.intp)
    tri_a = np.asarray(tri_a_list, dtype=np.intp)
    tri_b = np.asarray(tri_b_list, dtype=np.intp)
    tri_phi_target = np.asarray(tri_pt_list, dtype=np.float64)
    num_triplets = tri_center.shape[0]

    # CSR: triplets-by-atom (centre, a, or b).
    atom_tri_count = np.zeros(num_atoms + 1, dtype=np.intp)
    for ti in range(num_triplets):
        atom_tri_count[tri_center[ti] + 1] += 1
        atom_tri_count[tri_a[ti] + 1] += 1
        atom_tri_count[tri_b[ti] + 1] += 1
    tri_atom_start = np.cumsum(atom_tri_count)
    tri_atom_list = np.zeros(tri_atom_start[-1], dtype=np.intp)
    write = np.zeros(num_atoms, dtype=np.intp)
    for ti in range(num_triplets):
        for a_ in (tri_center[ti], tri_a[ti], tri_b[ti]):
            tri_atom_list[tri_atom_start[a_] + write[a_]] = ti
            write[a_] += 1

    # ---- sorted bonded-pair flat list for O(log N) lookup in the kernel ----
    bonded_flat = np.array(
        sorted(
            int(lo) * int(num_atoms) + int(hi) for (lo, hi) in bonded_set
        ),
        dtype=np.int64,
    )

    # ---- repulsion neighbour list (CSR; ASE's spatial hash) ----
    # Cutoff picks the larger of the two walls + a safety margin so
    # atoms that drift a few step_sigma between rebuilds still land
    # in the cached list.  Rebuild frequency is controlled by
    # ``rep_neighbor_update_interval`` on the driver.
    # Generous margin: typical atom drift over ``rep_neighbor_update_interval``
    # sweeps is a few tenths of an Å; 1.0 Å beyond the wall is safe.
    rep_cutoff = float(max(float(np.max(hard_core)), float(np.max(nonbond_push))) * 1.15 + 1.0)
    ri, rj = neighbor_list("ij", atoms, rep_cutoff)
    ri = np.asarray(ri, dtype=np.intp)
    rj = np.asarray(rj, dtype=np.intp)
    # Sort by ri to group by atom; build CSR.
    order = np.argsort(ri, kind="stable")
    ri_s = ri[order]
    rj_s = rj[order]
    counts = np.bincount(ri_s, minlength=num_atoms)
    rep_atom_start = np.zeros(num_atoms + 1, dtype=np.intp)
    rep_atom_start[1:] = np.cumsum(counts)
    rep_atom_list = rj_s.astype(np.intp)

    return {
        "bond_i": bond_i,
        "bond_j": bond_j,
        "bond_r_target": bond_r_target,
        "bond_atom_start": bond_atom_start.astype(np.intp),
        "bond_atom_list": bond_atom_list,
        "tri_center": tri_center,
        "tri_a": tri_a,
        "tri_b": tri_b,
        "tri_phi_target": tri_phi_target,
        "tri_atom_start": tri_atom_start.astype(np.intp),
        "tri_atom_list": tri_atom_list,
        "bonded_flat": bonded_flat,
        "hard_core": hard_core,
        "nonbond_push": nonbond_push,
        "pair_peak": pair_peak,
        "num_species": num_sp,
        # Cached repulsion neighbour list (CSR).
        "rep_atom_start": rep_atom_start,
        "rep_atom_list": rep_atom_list,
        "rep_cutoff": rep_cutoff,
    }


# ---------------------------------------------------------------------------
# Sweep driver (Python + numba hot loop)
# ---------------------------------------------------------------------------


if HAS_NUMBA:

    @njit(cache=False, fastmath=False)
    def _thermal_mc_sweep(
        positions,
        species_idx,
        cell_mat, cell_inv,
        # Topology arrays
        bond_atom_start, bond_atom_list,
        bond_i, bond_j, bond_r_target,
        tri_atom_start, tri_atom_list,
        tri_center, tri_a, tri_b, tri_phi_target,
        bonded_flat,
        rep_atom_start, rep_atom_list,
        hard_core, nonbond_push,
        # Weights
        k_bond, k_angle, k_rep,
        # MC params
        T,
        step_sigma,
        smart_dt,
        prob_displace,      # cumulative: [prob_displace, prob_displace+prob_swap]
        prob_swap_cum,     # cumulative
        num_trials,
        rng_seed,
        freeze_mask,        # (num_atoms,) bool: True = atom is frozen
        # Position restraint (global tether to a reference config)
        r_initial,          # (num_atoms, 3) float64 reference positions
        k_restraint,        # float64 spring constant (0.0 = disabled)
    ):
        """One MC sweep: ``num_trials`` trial moves.  Returns
        ``(n_acc_d, n_try_d, n_acc_s, n_try_s, n_acc_m, n_try_m)``
        for displacement / swap / smart move counts.
        """
        np.random.seed(rng_seed)
        num_atoms = positions.shape[0]
        n_acc_d = 0; n_try_d = 0
        n_acc_s = 0; n_try_s = 0
        n_acc_m = 0; n_try_m = 0

        for _ in range(num_trials):
            move_roll = np.random.random()
            if move_roll < prob_displace:
                # ---- Displacement ----
                i = np.random.randint(0, num_atoms)
                if freeze_mask[i]:
                    continue  # frozen atoms don't move
                old_x = positions[i, 0]
                old_y = positions[i, 1]
                old_z = positions[i, 2]
                e_old = _local_energy(
                    i, old_x, old_y, old_z, positions, species_idx,
                    cell_mat, cell_inv,
                    bond_atom_start, bond_atom_list,
                    bond_i, bond_j, bond_r_target, k_bond,
                    tri_atom_start, tri_atom_list,
                    tri_center, tri_a, tri_b, tri_phi_target, k_angle,
                    rep_atom_start, rep_atom_list,
                    hard_core, nonbond_push, k_rep,
                    bonded_flat, num_atoms,
                    r_initial, k_restraint,
                )
                # Trial: Gaussian 3-vector.
                dx = step_sigma * np.random.randn()
                dy = step_sigma * np.random.randn()
                dz = step_sigma * np.random.randn()
                new_x = old_x + dx
                new_y = old_y + dy
                new_z = old_z + dz
                e_new = _local_energy(
                    i, new_x, new_y, new_z, positions, species_idx,
                    cell_mat, cell_inv,
                    bond_atom_start, bond_atom_list,
                    bond_i, bond_j, bond_r_target, k_bond,
                    tri_atom_start, tri_atom_list,
                    tri_center, tri_a, tri_b, tri_phi_target, k_angle,
                    rep_atom_start, rep_atom_list,
                    hard_core, nonbond_push, k_rep,
                    bonded_flat, num_atoms,
                    r_initial, k_restraint,
                )
                dE = e_new - e_old
                n_try_d += 1
                if dE <= 0.0 or np.random.random() < math.exp(-dE / max(T, 1e-12)):
                    positions[i, 0] = new_x
                    positions[i, 1] = new_y
                    positions[i, 2] = new_z
                    n_acc_d += 1
            elif move_roll < prob_swap_cum:
                # ---- Species swap ----
                i = np.random.randint(0, num_atoms)
                if freeze_mask[i]:
                    continue
                # Find a j with different species AND not frozen;
                # up to 16 retries.
                j = -1
                for _ in range(16):
                    jj = np.random.randint(0, num_atoms)
                    if (
                        jj != i
                        and species_idx[jj] != species_idx[i]
                        and not freeze_mask[jj]
                    ):
                        j = jj
                        break
                if j < 0:
                    continue
                # Evaluate energy of i and j BEFORE swap.
                e_before = (
                    _local_energy(
                        i, positions[i, 0], positions[i, 1], positions[i, 2],
                        positions, species_idx, cell_mat, cell_inv,
                        bond_atom_start, bond_atom_list,
                        bond_i, bond_j, bond_r_target, k_bond,
                        tri_atom_start, tri_atom_list,
                        tri_center, tri_a, tri_b, tri_phi_target, k_angle,
                        rep_atom_start, rep_atom_list,
                        hard_core, nonbond_push, k_rep,
                        bonded_flat, num_atoms,
                        r_initial, k_restraint,
                    )
                    + _local_energy(
                        j, positions[j, 0], positions[j, 1], positions[j, 2],
                        positions, species_idx, cell_mat, cell_inv,
                        bond_atom_start, bond_atom_list,
                        bond_i, bond_j, bond_r_target, k_bond,
                        tri_atom_start, tri_atom_list,
                        tri_center, tri_a, tri_b, tri_phi_target, k_angle,
                        rep_atom_start, rep_atom_list,
                        hard_core, nonbond_push, k_rep,
                        bonded_flat, num_atoms,
                        r_initial, k_restraint,
                    )
                )
                # Swap labels.
                sp_i = species_idx[i]; sp_j = species_idx[j]
                species_idx[i] = sp_j
                species_idx[j] = sp_i
                e_after = (
                    _local_energy(
                        i, positions[i, 0], positions[i, 1], positions[i, 2],
                        positions, species_idx, cell_mat, cell_inv,
                        bond_atom_start, bond_atom_list,
                        bond_i, bond_j, bond_r_target, k_bond,
                        tri_atom_start, tri_atom_list,
                        tri_center, tri_a, tri_b, tri_phi_target, k_angle,
                        rep_atom_start, rep_atom_list,
                        hard_core, nonbond_push, k_rep,
                        bonded_flat, num_atoms,
                        r_initial, k_restraint,
                    )
                    + _local_energy(
                        j, positions[j, 0], positions[j, 1], positions[j, 2],
                        positions, species_idx, cell_mat, cell_inv,
                        bond_atom_start, bond_atom_list,
                        bond_i, bond_j, bond_r_target, k_bond,
                        tri_atom_start, tri_atom_list,
                        tri_center, tri_a, tri_b, tri_phi_target, k_angle,
                        rep_atom_start, rep_atom_list,
                        hard_core, nonbond_push, k_rep,
                        bonded_flat, num_atoms,
                        r_initial, k_restraint,
                    )
                )
                dE = e_after - e_before
                n_try_s += 1
                if dE <= 0.0 or np.random.random() < math.exp(-dE / max(T, 1e-12)):
                    n_acc_s += 1
                else:
                    # Revert.
                    species_idx[i] = sp_i
                    species_idx[j] = sp_j
            else:
                # ---- Smart MC: force-biased Langevin proposal + Metropolis correction ----
                # Self-tuning: use ``step_sigma`` as both the noise
                # magnitude AND the drift cap.  Internal dt derived
                # so noise_sigma = sqrt(2 dt) = step_sigma (same
                # per-step amplitude as classic MC), and the drift
                # ``(dt/T) F`` is clipped to magnitude ``step_sigma``
                # so it cannot dwarf the noise at low T.
                i = np.random.randint(0, num_atoms)
                if freeze_mask[i]:
                    continue
                old_x = positions[i, 0]; old_y = positions[i, 1]; old_z = positions[i, 2]
                fx_old, fy_old, fz_old = _local_force(
                    i, old_x, old_y, old_z, positions, species_idx,
                    cell_mat, cell_inv,
                    bond_atom_start, bond_atom_list,
                    bond_i, bond_j, bond_r_target, k_bond,
                    tri_atom_start, tri_atom_list,
                    tri_center, tri_a, tri_b, tri_phi_target, k_angle,
                    rep_atom_start, rep_atom_list,
                    hard_core, nonbond_push, k_rep,
                    bonded_flat, num_atoms,
                    r_initial, k_restraint,
                )
                noise_sigma = step_sigma
                dt_eff = 0.5 * step_sigma * step_sigma  # so sqrt(2 dt) = sigma
                dt_over_T = dt_eff / max(T, 1e-12)
                # Raw drift = (dt/T) * F, clipped to |drift| ≤ step_sigma.
                drift_x_raw = dt_over_T * fx_old
                drift_y_raw = dt_over_T * fy_old
                drift_z_raw = dt_over_T * fz_old
                dnorm = math.sqrt(
                    drift_x_raw * drift_x_raw
                    + drift_y_raw * drift_y_raw
                    + drift_z_raw * drift_z_raw
                )
                if dnorm > step_sigma and dnorm > 1e-12:
                    sc = step_sigma / dnorm
                    drift_x = drift_x_raw * sc
                    drift_y = drift_y_raw * sc
                    drift_z = drift_z_raw * sc
                else:
                    drift_x = drift_x_raw
                    drift_y = drift_y_raw
                    drift_z = drift_z_raw
                dx = drift_x + noise_sigma * np.random.randn()
                dy = drift_y + noise_sigma * np.random.randn()
                dz = drift_z + noise_sigma * np.random.randn()
                new_x = old_x + dx; new_y = old_y + dy; new_z = old_z + dz
                # ΔE local.
                e_old = _local_energy(
                    i, old_x, old_y, old_z, positions, species_idx,
                    cell_mat, cell_inv,
                    bond_atom_start, bond_atom_list,
                    bond_i, bond_j, bond_r_target, k_bond,
                    tri_atom_start, tri_atom_list,
                    tri_center, tri_a, tri_b, tri_phi_target, k_angle,
                    rep_atom_start, rep_atom_list,
                    hard_core, nonbond_push, k_rep,
                    bonded_flat, num_atoms,
                    r_initial, k_restraint,
                )
                e_new = _local_energy(
                    i, new_x, new_y, new_z, positions, species_idx,
                    cell_mat, cell_inv,
                    bond_atom_start, bond_atom_list,
                    bond_i, bond_j, bond_r_target, k_bond,
                    tri_atom_start, tri_atom_list,
                    tri_center, tri_a, tri_b, tri_phi_target, k_angle,
                    rep_atom_start, rep_atom_list,
                    hard_core, nonbond_push, k_rep,
                    bonded_flat, num_atoms,
                    r_initial, k_restraint,
                )
                dE = e_new - e_old
                # Force at the new position for the reverse-proposal term.
                fx_new, fy_new, fz_new = _local_force(
                    i, new_x, new_y, new_z, positions, species_idx,
                    cell_mat, cell_inv,
                    bond_atom_start, bond_atom_list,
                    bond_i, bond_j, bond_r_target, k_bond,
                    tri_atom_start, tri_atom_list,
                    tri_center, tri_a, tri_b, tri_phi_target, k_angle,
                    rep_atom_start, rep_atom_list,
                    hard_core, nonbond_push, k_rep,
                    bonded_flat, num_atoms,
                    r_initial, k_restraint,
                )
                # Apply the SAME drift-clamp to the reverse-proposal
                # drift, so the Metropolis correction stays consistent
                # with the actual (clamped) proposal distribution.
                bdrift_x_raw = dt_over_T * fx_new
                bdrift_y_raw = dt_over_T * fy_new
                bdrift_z_raw = dt_over_T * fz_new
                bnorm = math.sqrt(
                    bdrift_x_raw * bdrift_x_raw
                    + bdrift_y_raw * bdrift_y_raw
                    + bdrift_z_raw * bdrift_z_raw
                )
                if bnorm > step_sigma and bnorm > 1e-12:
                    sc = step_sigma / bnorm
                    bdrift_x = bdrift_x_raw * sc
                    bdrift_y = bdrift_y_raw * sc
                    bdrift_z = bdrift_z_raw * sc
                else:
                    bdrift_x = bdrift_x_raw
                    bdrift_y = bdrift_y_raw
                    bdrift_z = bdrift_z_raw
                # Log-proposal ratio:
                # log(q(old|new) / q(new|old)) =
                #    (|forward|² - |backward|²) / (4 dt_eff)
                # forward  = (new - old) - drift(F_old)
                # backward = (old - new) - drift(F_new) = -(dx) - drift(F_new)
                fdx = dx - drift_x
                fdy = dy - drift_y
                fdz = dz - drift_z
                bdx = -dx - bdrift_x
                bdy = -dy - bdrift_y
                bdz = -dz - bdrift_z
                log_q_ratio = (
                    (fdx * fdx + fdy * fdy + fdz * fdz)
                    - (bdx * bdx + bdy * bdy + bdz * bdz)
                ) / (4.0 * dt_eff)
                inv_T = 1.0 / max(T, 1e-12)
                log_alpha = -dE * inv_T + log_q_ratio
                n_try_m += 1
                if log_alpha >= 0.0 or np.random.random() < math.exp(log_alpha):
                    positions[i, 0] = new_x
                    positions[i, 1] = new_y
                    positions[i, 2] = new_z
                    n_acc_m += 1

        return n_acc_d, n_try_d, n_acc_s, n_try_s, n_acc_m, n_try_m

    @njit(cache=False, fastmath=False)
    def _total_energy_fast(
        positions,
        species_idx,
        cell_mat, cell_inv,
        bond_i, bond_j, bond_r_target,
        k_bond,
        tri_center, tri_a, tri_b, tri_phi_target,
        k_angle,
        rep_atom_start, rep_atom_list,
        hard_core, nonbond_push,
        k_rep,
        bonded_flat,
        num_atoms,
        r_initial,
        k_restraint,
    ):
        """Whole-system energy via the cached rep-neighbour CSR.
        Counts each bond / triplet / pair exactly once.  O(bonds +
        triplets + N × K) rather than O(N²).

        ``r_initial`` and ``k_restraint`` add a global position-tether
        term ``½ k_restraint Σ ‖r_i - r_initial_i‖²`` that anchors atoms
        to a reference configuration without freezing them.
        """
        # Bonds.
        e_bond = 0.0
        for bi in range(bond_i.shape[0]):
            a = bond_i[bi]
            b = bond_j[bi]
            dx = positions[b, 0] - positions[a, 0]
            dy = positions[b, 1] - positions[a, 1]
            dz = positions[b, 2] - positions[a, 2]
            mx, my, mz = _min_image(dx, dy, dz, cell_mat, cell_inv)
            r = math.sqrt(mx * mx + my * my + mz * mz)
            delta = r - bond_r_target[bi]
            e_bond += 0.5 * k_bond * delta * delta

        # Angles.  Each triplet counted once (with its full energy).
        e_angle = 0.0
        for ti in range(tri_center.shape[0]):
            c = tri_center[ti]
            ai = tri_a[ti]
            bi = tri_b[ti]
            vax = positions[ai, 0] - positions[c, 0]
            vay = positions[ai, 1] - positions[c, 1]
            vaz = positions[ai, 2] - positions[c, 2]
            vax, vay, vaz = _min_image(vax, vay, vaz, cell_mat, cell_inv)
            vbx = positions[bi, 0] - positions[c, 0]
            vby = positions[bi, 1] - positions[c, 1]
            vbz = positions[bi, 2] - positions[c, 2]
            vbx, vby, vbz = _min_image(vbx, vby, vbz, cell_mat, cell_inv)
            ra = math.sqrt(vax * vax + vay * vay + vaz * vaz)
            rb = math.sqrt(vbx * vbx + vby * vby + vbz * vbz)
            if ra < 1e-10 or rb < 1e-10:
                continue
            cos_phi = (vax * vbx + vay * vby + vaz * vbz) / (ra * rb)
            if cos_phi > 1.0:
                cos_phi = 1.0
            elif cos_phi < -1.0:
                cos_phi = -1.0
            phi = math.acos(cos_phi)
            dphi = phi - tri_phi_target[ti]
            e_angle += 0.5 * k_angle * dphi * dphi

        # Repulsion.  Each unordered pair (i, j) appears twice in the
        # CSR (both directions).  Count only i < j.
        e_rep = 0.0
        for i in range(num_atoms):
            sp_i = species_idx[i]
            rs = rep_atom_start[i]
            re = rep_atom_start[i + 1]
            for idx in range(rs, re):
                j = rep_atom_list[idx]
                if j <= i:
                    continue
                dx = positions[j, 0] - positions[i, 0]
                dy = positions[j, 1] - positions[i, 1]
                dz = positions[j, 2] - positions[i, 2]
                mx, my, mz = _min_image(dx, dy, dz, cell_mat, cell_inv)
                r2 = mx * mx + my * my + mz * mz
                sp_j = species_idx[j]
                r_hard = hard_core[sp_i, sp_j]
                r_push = nonbond_push[sp_i, sp_j]
                rwall = r_push if r_push > r_hard else r_hard
                if r2 >= rwall * rwall:
                    continue
                r = math.sqrt(r2)
                e_rep += _repulsion_energy_pair(r, r_hard, 4.0 * k_rep)
                # Non-bonded push if not bonded.
                key = np.int64(i) * np.int64(num_atoms) + np.int64(j)
                left = 0
                right = bonded_flat.shape[0]
                is_bonded = False
                while left < right:
                    mid = (left + right) // 2
                    if bonded_flat[mid] == key:
                        is_bonded = True
                        break
                    elif bonded_flat[mid] < key:
                        left = mid + 1
                    else:
                        right = mid
                if not is_bonded:
                    e_rep += _repulsion_energy_pair(r, r_push, k_rep)

        # Position restraint.  ½ k_restraint Σ ‖r_i - r_initial_i‖² with
        # min-image correction so wrap-around doesn't cause spurious tugs.
        e_restraint = 0.0
        if k_restraint > 0.0:
            for i in range(num_atoms):
                dx = positions[i, 0] - r_initial[i, 0]
                dy = positions[i, 1] - r_initial[i, 1]
                dz = positions[i, 2] - r_initial[i, 2]
                mx, my, mz = _min_image(dx, dy, dz, cell_mat, cell_inv)
                e_restraint += 0.5 * k_restraint * (mx * mx + my * my + mz * mz)
        return (
            e_bond + e_angle + e_rep + e_restraint,
            e_bond,
            e_angle,
            e_rep,
        )


# ---------------------------------------------------------------------------
# Python-level total-energy evaluator (for history capture)
# ---------------------------------------------------------------------------


def _total_energy(
    positions: np.ndarray,
    species_idx: np.ndarray,
    cell_mat: np.ndarray,
    cell_inv: np.ndarray,
    topo: dict,
    k_bond: float,
    k_angle: float,
    k_rep: float,
    r_initial: np.ndarray | None = None,
    k_restraint: float = 0.0,
) -> tuple[float, float, float, float]:
    """Return (total, bond, angle, repulsion) summed across the whole
    system.  Used for capture; not in the hot loop.

    When ``k_restraint > 0`` the position-restraint term
    ``½ k_restraint Σ ‖r_i - r_initial_i‖²`` (min-image-corrected) is
    added to ``total``.  The component breakdown still excludes the
    restraint so callers can compute it as
    ``total - bond - angle - rep``.
    """
    num_atoms = positions.shape[0]

    # Bond
    bond_energy = 0.0
    bi = topo["bond_i"]; bj = topo["bond_j"]; brt = topo["bond_r_target"]
    if bi.size:
        d = positions[bj] - positions[bi]
        frac = d @ cell_inv
        frac -= np.rint(frac)
        d = frac @ cell_mat
        r = np.linalg.norm(d, axis=1)
        bond_energy = float(np.sum(0.5 * k_bond * (r - brt) ** 2))

    # Angle
    angle_energy = 0.0
    tc = topo["tri_center"]; ta = topo["tri_a"]; tb = topo["tri_b"]
    tpt = topo["tri_phi_target"]
    if tc.size:
        va = positions[ta] - positions[tc]
        vb = positions[tb] - positions[tc]
        va_f = va @ cell_inv; va_f -= np.rint(va_f); va = va_f @ cell_mat
        vb_f = vb @ cell_inv; vb_f -= np.rint(vb_f); vb = vb_f @ cell_mat
        ra = np.linalg.norm(va, axis=1)
        rb = np.linalg.norm(vb, axis=1)
        cos_phi = np.clip(
            np.einsum("ij,ij->i", va, vb) / np.maximum(ra * rb, 1e-12),
            -1.0, 1.0,
        )
        phi = np.arccos(cos_phi)
        dphi = phi - tpt
        # Full triplet energy (no /3); must match _local_energy.
        angle_energy = float(np.sum(0.5 * k_angle * dphi ** 2))

    # Repulsion (O(N²) brute force)
    rep_energy = 0.0
    hard_core = topo["hard_core"]; nonbond_push = topo["nonbond_push"]
    # Build bonded-set for quick membership.
    bonded = set()
    for k in range(bi.size):
        lo = int(bi[k]); hi = int(bj[k])
        if lo > hi: lo, hi = hi, lo
        bonded.add((lo, hi))
    for i in range(num_atoms):
        sp_i = int(species_idx[i])
        for j in range(i + 1, num_atoms):
            sp_j = int(species_idx[j])
            d = positions[j] - positions[i]
            frac = d @ cell_inv
            frac -= np.rint(frac)
            d = frac @ cell_mat
            r = float(np.linalg.norm(d))
            r_hard = float(hard_core[sp_i, sp_j])
            r_push = float(nonbond_push[sp_i, sp_j])
            if r < r_hard:
                h = r_hard / r - 1.0
                rep_energy += 4.0 * k_rep * r_hard * (h - math.log1p(h))
            if r < r_push and (i, j) not in bonded:
                h = r_push / r - 1.0
                rep_energy += k_rep * r_push * (h - math.log1p(h))

    # Position restraint
    restraint_energy = 0.0
    if k_restraint > 0.0 and r_initial is not None:
        d = positions - r_initial
        frac = d @ cell_inv
        frac -= np.rint(frac)
        d = frac @ cell_mat
        restraint_energy = float(0.5 * k_restraint * np.sum(d * d))

    total = bond_energy + angle_energy + rep_energy + restraint_energy
    return total, bond_energy, angle_energy, rep_energy


def _temperature_schedule(
    T_schedule, num_sweeps: int, T_start: float, T_end: float, hold_sweeps: int,
) -> np.ndarray:
    """Return the temperature at every sweep."""
    if callable(T_schedule):
        return np.array([float(T_schedule(s)) for s in range(num_sweeps)],
                        dtype=np.float64)
    if T_schedule == "hold":
        return np.full(num_sweeps, float(T_start), dtype=np.float64)
    if T_schedule == "anneal":
        T = np.full(num_sweeps, float(T_end), dtype=np.float64)
        hold_n = min(int(hold_sweeps), num_sweeps)
        T[:hold_n] = float(T_start)
        anneal_n = num_sweeps - hold_n
        if anneal_n > 0:
            T[hold_n:] = np.linspace(
                float(T_start), float(T_end), anneal_n, endpoint=True,
                dtype=np.float64,
            )
        return T
    raise ValueError(f"Unknown T_schedule: {T_schedule!r}")


def thermal_relax_impl(
    atoms,
    species_idx: np.ndarray,
    shell_target,
    *,
    num_sweeps: int = 1000,
    T_schedule="anneal",
    T_start: float = 0.05,
    T_end: float = 0.001,
    hold_sweeps: int = 200,
    step_sigma: float = 0.05,
    smart_dt: float = 0.02,
    adapt_step: bool = True,
    target_accept: float = 0.4,
    move_probs: dict | None = None,
    bond_weight: float = 1.0,
    angle_weight: float = 0.5,
    repulsion_weight: float = 3.0,
    k_restraint: float = 0.0,
    hard_core_scale: float = 1.0,
    nonbond_push_scale: float = 1.0,
    neighbor_update_interval: int = 100,
    rep_neighbor_update_interval: int = 20,
    capture_stride: int = 10,
    capture_trajectory: bool = True,
    restore_best: bool = True,
    freeze_mask: np.ndarray | None = None,
    grain_ids: np.ndarray | None = None,
    grain_move_interval: int = 1,
    grain_sigma_rot: float = 0.01,
    grain_sigma_trans: float = 0.01,
    rng_seed: int | None = None,
    show_progress: bool = True,
) -> dict:
    """Run thermal Monte-Carlo on ``atoms`` against ``shell_target``.

    Returns a history dict that can be assigned to
    ``cell.thermal_relax_history``.  Mutates ``atoms.positions`` and
    ``species_idx`` in place (the ``_atom_shell_species_index`` of the
    owning Supercell).
    """
    if not HAS_NUMBA:
        raise RuntimeError(
            "thermal_relax requires numba.  Install: uv add numba."
        )
    rng = np.random.default_rng(rng_seed)
    move_probs = move_probs or {"displace": 1.0, "swap": 0.0, "smart": 0.0}
    p_d = float(move_probs.get("displace", 0.0))
    p_s = float(move_probs.get("swap", 0.0))
    p_m = float(move_probs.get("smart", 0.0))
    # Disable swap if only one species is present.
    if int(np.unique(species_idx).size) < 2 and p_s > 0:
        p_d += p_s
        p_s = 0.0
    tot = p_d + p_s + p_m
    if tot <= 0:
        raise ValueError("move_probs must sum to > 0.")
    p_d /= tot; p_s /= tot; p_m /= tot
    prob_displace_cum = p_d
    prob_swap_cum = p_d + p_s

    num_atoms = len(atoms)
    num_trials = num_atoms  # one sweep = N trial moves
    positions = np.ascontiguousarray(atoms.positions, dtype=np.float64)
    species_idx = np.ascontiguousarray(species_idx, dtype=np.intp)

    # Position-restraint reference: a frozen snapshot of the starting
    # positions.  Energy contribution is ``½ k_restraint Σ ‖r_i - r_initial_i‖²``
    # (min-image-corrected); ``k_restraint == 0`` disables the term
    # but the reference array still has to be passed to the kernel.
    r_initial = positions.copy()
    k_restraint = float(k_restraint)

    if freeze_mask is None:
        freeze_mask_arr = np.zeros(num_atoms, dtype=np.bool_)
    else:
        freeze_mask_arr = np.ascontiguousarray(freeze_mask, dtype=np.bool_)
        if freeze_mask_arr.shape != (num_atoms,):
            raise ValueError(
                f"freeze_mask must have shape ({num_atoms},); "
                f"got {freeze_mask_arr.shape}"
            )
    cell_mat = np.ascontiguousarray(atoms.cell.array, dtype=np.float64)
    cell_inv = np.ascontiguousarray(np.linalg.inv(cell_mat), dtype=np.float64)

    # Initial topology.
    topo = _build_thermal_topology(
        atoms, species_idx, shell_target,
        hard_core_scale=hard_core_scale,
        nonbond_push_scale=nonbond_push_scale,
    )

    # Temperature schedule (one value per sweep).
    T_sched = _temperature_schedule(
        T_schedule, num_sweeps, T_start, T_end, hold_sweeps,
    )

    # History buffers.
    n_capture = (num_sweeps + capture_stride - 1) // capture_stride + 1
    sweep_hist = np.zeros(n_capture, dtype=np.intp)
    T_hist = np.zeros(n_capture, dtype=np.float64)
    cost_hist = np.zeros(n_capture, dtype=np.float64)
    cost_bond_hist = np.zeros(n_capture, dtype=np.float64)
    cost_angle_hist = np.zeros(n_capture, dtype=np.float64)
    cost_rep_hist = np.zeros(n_capture, dtype=np.float64)
    cost_restraint_hist = np.zeros(n_capture, dtype=np.float64)
    accept_hist = np.zeros(n_capture, dtype=np.float64)
    sigma_hist = np.zeros(n_capture, dtype=np.float64)
    best_cost_hist = np.zeros(n_capture, dtype=np.float64)
    if capture_trajectory:
        traj_hist = np.zeros((n_capture, num_atoms, 3), dtype=np.float32)
    else:
        traj_hist = None

    # Capture the starting snapshot.
    total, e_b, e_a, e_r = _total_energy_fast(
        positions, species_idx, cell_mat, cell_inv,
        topo["bond_i"], topo["bond_j"], topo["bond_r_target"], float(bond_weight),
        topo["tri_center"], topo["tri_a"], topo["tri_b"], topo["tri_phi_target"],
        float(angle_weight),
        topo["rep_atom_start"], topo["rep_atom_list"],
        topo["hard_core"], topo["nonbond_push"], float(repulsion_weight),
        topo["bonded_flat"], num_atoms,
        r_initial, k_restraint,
    )
    best_cost = total
    best_positions = positions.copy()
    best_species_idx = species_idx.copy()

    cap_idx = 0
    def _capture(sweep_idx, T_now, acc_rate, sigma):
        nonlocal cap_idx
        sweep_hist[cap_idx] = sweep_idx
        T_hist[cap_idx] = T_now
        cost_hist[cap_idx] = total
        cost_bond_hist[cap_idx] = e_b
        cost_angle_hist[cap_idx] = e_a
        cost_rep_hist[cap_idx] = e_r
        # Restraint energy = total - bond - angle - rep (the fast
        # kernel folds restraint into ``total`` but reports the three
        # potential components separately).
        cost_restraint_hist[cap_idx] = total - e_b - e_a - e_r
        accept_hist[cap_idx] = acc_rate
        sigma_hist[cap_idx] = sigma
        best_cost_hist[cap_idx] = best_cost
        if traj_hist is not None:
            traj_hist[cap_idx] = positions.astype(np.float32)
        cap_idx += 1

    _capture(0, float(T_sched[0]), 0.0, float(step_sigma))

    # ---- sweeps ----
    progress = None
    if show_progress:
        from .g3 import _TextProgressBar
        progress = _TextProgressBar(num_sweeps, label="Thermal MC")
        progress.update(0)

    sweep_acc_d = 0; sweep_try_d = 0
    sweep_acc_s = 0; sweep_try_s = 0
    sweep_acc_m = 0; sweep_try_m = 0
    sweep_acc_g = 0; sweep_try_g = 0
    sigma = float(step_sigma)

    # Grain-rigid moves are opt-in via grain_ids.  When a multi-grain
    # structure is present, a single rotation of a grain costs zero
    # intra-grain energy (rigid transforms preserve distances) but
    # lets the grain re-orient to lower GB strain.  For SRO/MRO this
    # is often the dominant degree of freedom.
    do_grain_moves = (
        grain_ids is not None
        and grain_move_interval > 0
        and int(np.unique(grain_ids[grain_ids >= 0]).size) >= 2
    )
    if do_grain_moves:
        grain_ids_arr = np.asarray(grain_ids, dtype=np.intp)

    for sweep in range(num_sweeps):
        T_now = float(T_sched[sweep])
        sub_rng = int(rng.integers(0, 2**31 - 1))

        a_d, t_d, a_s, t_s, a_m, t_m = _thermal_mc_sweep(
            positions, species_idx,
            cell_mat, cell_inv,
            topo["bond_atom_start"], topo["bond_atom_list"],
            topo["bond_i"], topo["bond_j"], topo["bond_r_target"],
            topo["tri_atom_start"], topo["tri_atom_list"],
            topo["tri_center"], topo["tri_a"], topo["tri_b"],
            topo["tri_phi_target"],
            topo["bonded_flat"],
            topo["rep_atom_start"], topo["rep_atom_list"],
            topo["hard_core"], topo["nonbond_push"],
            float(bond_weight), float(angle_weight), float(repulsion_weight),
            T_now, sigma, float(smart_dt),
            prob_displace_cum, prob_swap_cum,
            int(num_trials), sub_rng,
            freeze_mask_arr,
            r_initial, float(k_restraint),
        )
        sweep_acc_d += a_d; sweep_try_d += t_d
        sweep_acc_s += a_s; sweep_try_s += t_s
        sweep_acc_m += a_m; sweep_try_m += t_m

        # Grain-rigid moves: run once every ``grain_move_interval``
        # sweeps.  Try each grain once per round.
        if do_grain_moves and (sweep + 1) % grain_move_interval == 0:
            a_g, t_g = _try_grain_rigid_moves(
                positions, species_idx, grain_ids_arr,
                cell_mat, cell_inv, topo,
                float(bond_weight), float(angle_weight), float(repulsion_weight),
                T_now, float(grain_sigma_rot), float(grain_sigma_trans),
                rng,
                r_initial, float(k_restraint),
            )
            sweep_acc_g += a_g; sweep_try_g += t_g

        # Recompute cost + track best.
        total, e_b, e_a, e_r = _total_energy_fast(
            positions, species_idx, cell_mat, cell_inv,
            topo["bond_i"], topo["bond_j"], topo["bond_r_target"], float(bond_weight),
            topo["tri_center"], topo["tri_a"], topo["tri_b"], topo["tri_phi_target"],
            float(angle_weight),
            topo["rep_atom_start"], topo["rep_atom_list"],
            topo["hard_core"], topo["nonbond_push"], float(repulsion_weight),
            topo["bonded_flat"], num_atoms,
            r_initial, k_restraint,
        )
        if total < best_cost:
            best_cost = total
            best_positions = positions.copy()
            best_species_idx = species_idx.copy()

        # Adapt step size every 20 sweeps.
        if adapt_step and sweep > 0 and sweep % 20 == 0 and sweep_try_d > 0:
            rate = sweep_acc_d / max(sweep_try_d, 1)
            sigma *= float(np.clip(rate / target_accept, 0.7, 1.3))
            sweep_acc_d = 0; sweep_try_d = 0

        # Rebuild FULL topology (bond graph + triplets + rep list)
        # periodically - this is the expensive pass.
        if neighbor_update_interval > 0 and (sweep + 1) % neighbor_update_interval == 0:
            atoms.positions = positions
            topo = _build_thermal_topology(
                atoms, species_idx, shell_target,
                hard_core_scale=hard_core_scale,
                nonbond_push_scale=nonbond_push_scale,
            )
            total, e_b, e_a, e_r = _total_energy_fast(
                positions, species_idx, cell_mat, cell_inv,
                topo["bond_i"], topo["bond_j"], topo["bond_r_target"], float(bond_weight),
                topo["tri_center"], topo["tri_a"], topo["tri_b"], topo["tri_phi_target"],
                float(angle_weight),
                topo["rep_atom_start"], topo["rep_atom_list"],
                topo["hard_core"], topo["nonbond_push"], float(repulsion_weight),
                topo["bonded_flat"], num_atoms,
                r_initial, k_restraint,
            )
            # Re-evaluate best under the new topology (different bond
            # graph = different energy function; re-floor the best to
            # the new metric).
            if total < best_cost:
                best_cost = total
                best_positions = positions.copy()
                best_species_idx = species_idx.copy()
        elif (
            rep_neighbor_update_interval > 0
            and (sweep + 1) % rep_neighbor_update_interval == 0
        ):
            # Cheap refresh: update the rep-neighbour CSR only (keeps
            # the bond topology stale for now).  Rebuild runs ASE's
            # spatial-hashed neighbor_list - typically ~10 ms for
            # 1000-atom cells.
            atoms.positions = positions
            new_start, new_list = _rebuild_rep_neighbors(atoms, topo["rep_cutoff"])
            topo["rep_atom_start"] = new_start
            topo["rep_atom_list"] = new_list

        # Capture.
        if (sweep + 1) % capture_stride == 0 or sweep == num_sweeps - 1:
            try_tot = max(sweep_try_d + sweep_try_s + sweep_try_m, 1)
            acc_rate = (
                float(sweep_acc_d + sweep_acc_s + sweep_acc_m) / float(try_tot)
            )
            _capture(sweep + 1, T_now, acc_rate, sigma)

        if progress is not None:
            progress.update(sweep + 1)

    if progress is not None:
        progress.close()

    # Write final atoms.  By default we restore the *best* (lowest-
    # cost) configuration observed during the run, not the literal
    # last-sweep positions: at non-zero T the atoms vibrate around
    # the basin minimum, so final positions are thermally-smeared
    # even when the run has long since converged.  Using the best
    # snapshot gives sharp g(r) / g3 that matches the 3D view.  Pass
    # ``restore_best=False`` to keep the final-sweep positions
    # instead (useful for equilibration studies where the ensemble
    # matters, not a single snapshot).
    if restore_best:
        positions[:] = best_positions
        species_idx[:] = best_species_idx
    atoms.positions = positions

    # Trim history arrays.
    def _trim(a):
        return a[:cap_idx] if a is not None else None

    history = {
        "sweep": _trim(sweep_hist),
        "T": _trim(T_hist),
        "cost": _trim(cost_hist),
        "cost_bond": _trim(cost_bond_hist),
        "cost_angle": _trim(cost_angle_hist),
        "cost_rep": _trim(cost_rep_hist),
        "cost_restraint": _trim(cost_restraint_hist),
        "accept_rate": _trim(accept_hist),
        "step_sigma": _trim(sigma_hist),
        "best_cost": _trim(best_cost_hist),
        "trajectory": _trim(traj_hist) if traj_hist is not None else None,
        "best_positions": best_positions,
        "best_species_idx": best_species_idx,
        "final_positions": positions.copy(),
        "final_species_idx": species_idx.copy(),
        "T_schedule": T_sched,
    }
    return history
