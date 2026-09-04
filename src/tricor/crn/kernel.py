"""Compiled Keating patch kernel.

Profiling the WWW inner loop at 1000 atoms: 7.9 ms per attempted move, of which
93% is the local FIRE relaxation and 70% is patch_energy_forces alone -- called
62 times per attempt at ~139 us each.  A 50-atom patch is ~100 bonds and ~200
angles, microseconds of real arithmetic; at that array size numpy costs ~1.1 us
per call in dispatch overhead, so ~125 numpy calls' worth of overhead is the
actual cost.  Compiling the kernel removes it.

This matters because the target is a 500k-atom structure in under a minute.
Attempts scale linearly with N (measured: 7626 us/attempt at 216 atoms, 7929 at
1000, i.e. flat), so the problem is a constant factor, not the algorithm.
"""
from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True, fastmath=True)
def patch_ef(pos, bonds, angles, L, kb, ka_arr, d2, off_arr, want_forces):
    """Keating energy and forces for one patch. Mirrors patch_energy_forces.

    ka_arr and off_arr are per-ANGLE, so a valency-dependent stiffness or a
    per-coordination equilibrium angle costs nothing extra here.
    """
    m = pos.shape[0]
    F = np.zeros((m, 3))
    E = 0.0
    # --- bond stretch: sum (r.r - d^2)^2
    for b in range(bonds.shape[0]):
        i = bonds[b, 0]
        j = bonds[b, 1]
        dx = pos[j, 0] - pos[i, 0]
        dy = pos[j, 1] - pos[i, 1]
        dz = pos[j, 2] - pos[i, 2]
        dx -= L[0] * np.rint(dx / L[0])
        dy -= L[1] * np.rint(dy / L[1])
        dz -= L[2] * np.rint(dz / L[2])
        r2 = dx * dx + dy * dy + dz * dz
        s = r2 - d2
        E += kb * s * s
        if want_forces:
            g = 4.0 * kb * s
            F[i, 0] += g * dx
            F[i, 1] += g * dy
            F[i, 2] += g * dz
            F[j, 0] -= g * dx
            F[j, 1] -= g * dy
            F[j, 2] -= g * dz
    # --- bond bend: sum (r_ij . r_ik + c)^2
    for a in range(angles.shape[0]):
        c = angles[a, 0]
        j = angles[a, 1]
        k = angles[a, 2]
        ax = pos[j, 0] - pos[c, 0]
        ay = pos[j, 1] - pos[c, 1]
        az = pos[j, 2] - pos[c, 2]
        ax -= L[0] * np.rint(ax / L[0])
        ay -= L[1] * np.rint(ay / L[1])
        az -= L[2] * np.rint(az / L[2])
        bx = pos[k, 0] - pos[c, 0]
        by = pos[k, 1] - pos[c, 1]
        bz = pos[k, 2] - pos[c, 2]
        bx -= L[0] * np.rint(bx / L[0])
        by -= L[1] * np.rint(by / L[1])
        bz -= L[2] * np.rint(bz / L[2])
        u = ax * bx + ay * by + az * bz + off_arr[a]
        ka = ka_arr[a]
        E += ka * u * u
        if want_forces:
            g = 2.0 * ka * u
            F[j, 0] -= g * bx
            F[j, 1] -= g * by
            F[j, 2] -= g * bz
            F[k, 0] -= g * ax
            F[k, 1] -= g * ay
            F[k, 2] -= g * az
            F[c, 0] += g * (ax + bx)
            F[c, 1] += g * (ay + by)
            F[c, 2] += g * (az + bz)
    return E, F


@njit(cache=True, fastmath=True)
def relax_fire(pos, bonds, angles, free, L, kb, ka_arr, d2, off_arr,
               n_iter, e_target, check_from, c_f, dt0, dt_max,
               alpha0, f_inc, f_dec, f_alpha, n_min, fmax_stop, max_step):
    """FIRE relaxation of a patch, entirely inside the compiled kernel.

    Wiring only the energy/force evaluation into numba left 62 Python
    round-trips per attempted move, and profiling put the remaining cost there
    rather than in the arithmetic.  Running the whole loop compiled removes
    them.  Early rejection is kept inside so a doomed move still aborts as soon
    as E - c_f|F|^2 exceeds the threshold, which is where most of the saving
    comes from at low acceptance.

    Returns (energy, n_iterations, aborted).
    """
    nf = free.shape[0]
    v = np.zeros((nf, 3))
    dt = dt0
    alpha = alpha0
    n_pos = 0
    E = 0.0
    aborted = False
    for it in range(n_iter):
        E, F = patch_ef(pos, bonds, angles, L, kb, ka_arr, d2, off_arr, True)
        fmax = 0.0
        fsq = 0.0
        for a in range(nf):
            i = free[a]
            for c in range(3):
                fc = F[i, c]
                fsq += fc * fc
                if abs(fc) > fmax:
                    fmax = abs(fc)
        if fmax < fmax_stop:
            return E, it, False
        if e_target > -1.0e300 and it >= check_from:
            if E - c_f * fsq > e_target:
                return E, it, True
        P = 0.0
        for a in range(nf):
            i = free[a]
            P += (F[i, 0] * v[a, 0] + F[i, 1] * v[a, 1] + F[i, 2] * v[a, 2])
        if P > 0.0:
            vn = 0.0
            fn = 0.0
            for a in range(nf):
                i = free[a]
                vn += v[a, 0]**2 + v[a, 1]**2 + v[a, 2]**2
                fn += F[i, 0]**2 + F[i, 1]**2 + F[i, 2]**2
            vn = np.sqrt(vn)
            fn = np.sqrt(fn)
            if fn < 1e-12:
                fn = 1e-12
            for a in range(nf):
                i = free[a]
                for c in range(3):
                    v[a, c] = (1.0 - alpha) * v[a, c] + alpha * vn * F[i, c] / fn
            n_pos += 1
            if n_pos > n_min:
                dt = min(dt * f_inc, dt_max)
                alpha *= f_alpha
        else:
            n_pos = 0
            dt *= f_dec
            alpha = alpha0
            for a in range(nf):
                v[a, 0] = 0.0
                v[a, 1] = 0.0
                v[a, 2] = 0.0
        for a in range(nf):
            i = free[a]
            dx0 = 0.0
            dx1 = 0.0
            dx2 = 0.0
            for c in range(3):
                v[a, c] += dt * F[i, c]
            dx0 = dt * v[a, 0]
            dx1 = dt * v[a, 1]
            dx2 = dt * v[a, 2]
            # Per-atom displacement cap, as in the reference implementation.
            # Without it a single FIRE step can throw an atom clear across the
            # patch, and the two paths diverge: measured, accept/reject agreed
            # on only 91% of moves and the uncapped run ended at higher energy.
            nrm = np.sqrt(dx0 * dx0 + dx1 * dx1 + dx2 * dx2)
            if nrm > max_step:
                sc = max_step / nrm
                dx0 *= sc
                dx1 *= sc
                dx2 *= sc
            pos[i, 0] += dx0
            pos[i, 1] += dx1
            pos[i, 2] += dx2
    # Energy must be re-evaluated at the FINAL positions; returning the value
    # from the last force call reports the energy one step stale.
    E, _F = patch_ef(pos, bonds, angles, L, kb, ka_arr, d2, off_arr, False)
    return E, n_iter, aborted


# ── excluded-volume variants ────────────────────────────────────────────────
#
# The original kernels implement only the PUBLISHED Keating energy, so the fast
# path is guarded on k_rep == k_core == 0 and any run with excluded volume
# falls back to numpy and loses the ~7x speedup.  That guard was a limitation
# of what was written, not an algorithmic barrier: the non-bonded pair list is
# built OUTSIDE the kernel by patch_nonbonded (cKDTree) and passed in, and it
# is held fixed for the duration of a patch relaxation, so both extra terms are
# plain pairwise loops with no spatial search.
#
# This matters beyond convenience: the Keating angular term is degenerate for
# Z >= 6 (an octahedron and a 17-degree arrangement have equal energy), so
# every chemistry with 6-fold cations REQUIRES the non-bonded term -- meaning
# the whole oxide corpus was locked out of the fast path.

@njit(cache=True, fastmath=True)
def patch_ef_nb(pos, bonds, angles, L, kb, ka_arr, d2, off_arr, want_forces,
                nb_pairs, r_rep, k_rep, r_core, k_core, quartic):
    """patch_ef plus the short-range core and non-bonded repulsion.

    Mirrors patch_energy_forces' numpy branch term for term:
      core (on BONDS, r < r_core):  E += k_core*(r_core-r)^2
                                    f  = 2*k_core*(r_core-r)/r * rij
      repulsion (on nb_pairs):
        harmonic  E += k_rep*(r_rep-r)^2      f = 2*k_rep*(r_rep-r)/r * dv
        quartic   E += 0.5*k_rep*(r_rep^2-r^2)^2   f = 2*k_rep*(r_rep^2-r^2)*dv
      (quartic is von Alfthan PRB 68, 073203 (2003) Eq. 3.17.)
    """
    E, F = patch_ef(pos, bonds, angles, L, kb, ka_arr, d2, off_arr,
                    want_forces)

    if k_core != 0.0:
        for b in range(bonds.shape[0]):
            i = bonds[b, 0]
            j = bonds[b, 1]
            dx = pos[j, 0] - pos[i, 0]
            dy = pos[j, 1] - pos[i, 1]
            dz = pos[j, 2] - pos[i, 2]
            dx -= L[0] * np.rint(dx / L[0])
            dy -= L[1] * np.rint(dy / L[1])
            dz -= L[2] * np.rint(dz / L[2])
            r = np.sqrt(max(dx * dx + dy * dy + dz * dz, 1e-24))
            if r < r_core:
                dc = r_core - r
                E += k_core * dc * dc
                if want_forces:
                    g = 2.0 * k_core * dc / max(r, 1e-9)
                    # numpy applies +fc to bonds[:,1], -fc to bonds[:,0]
                    F[j, 0] += g * dx
                    F[j, 1] += g * dy
                    F[j, 2] += g * dz
                    F[i, 0] -= g * dx
                    F[i, 1] -= g * dy
                    F[i, 2] -= g * dz

    if k_rep != 0.0:
        for p in range(nb_pairs.shape[0]):
            i = nb_pairs[p, 0]
            j = nb_pairs[p, 1]
            dx = pos[j, 0] - pos[i, 0]
            dy = pos[j, 1] - pos[i, 1]
            dz = pos[j, 2] - pos[i, 2]
            dx -= L[0] * np.rint(dx / L[0])
            dy -= L[1] * np.rint(dy / L[1])
            dz -= L[2] * np.rint(dz / L[2])
            r = np.sqrt(dx * dx + dy * dy + dz * dz)
            if r < r_rep:
                if quartic:
                    dr = r_rep * r_rep - r * r
                    E += 0.5 * k_rep * dr * dr
                    g = 2.0 * k_rep * dr
                else:
                    dr = r_rep - r
                    E += k_rep * dr * dr
                    g = 2.0 * k_rep * dr / max(r, 1e-9)
                if want_forces:
                    F[j, 0] += g * dx
                    F[j, 1] += g * dy
                    F[j, 2] += g * dz
                    F[i, 0] -= g * dx
                    F[i, 1] -= g * dy
                    F[i, 2] -= g * dz
    return E, F


@njit(cache=True, fastmath=True)
def relax_fire_nb(pos, bonds, angles, free, L, kb, ka_arr, d2, off_arr,
                  n_iter, e_target, check_from, c_f, dt0, dt_max,
                  alpha0, f_inc, f_dec, f_alpha, n_min, fmax_stop,
                  max_step, nb_pairs, r_rep, k_rep, r_core, k_core,
                  quartic):
    """FIRE relaxation of a patch WITH excluded volume.

    Derived mechanically from relax_fire (signature + E/F call swapped)
    so the FIRE loop is identical by construction.  A hand-written copy
    diverged from the original and would have silently changed the
    optimiser.

    The nb pair list is fixed for the duration of a patch relaxation
    (patch_with_scenery builds it once), matching the numpy path.

    Original docstring follows.

    FIRE relaxation of a patch, entirely inside the compiled kernel.

    Wiring only the energy/force evaluation into numba left 62 Python
    round-trips per attempted move, and profiling put the remaining cost there
    rather than in the arithmetic.  Running the whole loop compiled removes
    them.  Early rejection is kept inside so a doomed move still aborts as soon
    as E - c_f|F|^2 exceeds the threshold, which is where most of the saving
    comes from at low acceptance.

    Returns (energy, n_iterations, aborted).
    """
    nf = free.shape[0]
    v = np.zeros((nf, 3))
    dt = dt0
    alpha = alpha0
    n_pos = 0
    E = 0.0
    aborted = False
    for it in range(n_iter):
        E, F = patch_ef_nb(pos, bonds, angles, L, kb, ka_arr, d2, off_arr, True, nb_pairs, r_rep, k_rep, r_core, k_core, quartic)
        fmax = 0.0
        fsq = 0.0
        for a in range(nf):
            i = free[a]
            for c in range(3):
                fc = F[i, c]
                fsq += fc * fc
                if abs(fc) > fmax:
                    fmax = abs(fc)
        if fmax < fmax_stop:
            return E, it, False
        if e_target > -1.0e300 and it >= check_from:
            if E - c_f * fsq > e_target:
                return E, it, True
        P = 0.0
        for a in range(nf):
            i = free[a]
            P += (F[i, 0] * v[a, 0] + F[i, 1] * v[a, 1] + F[i, 2] * v[a, 2])
        if P > 0.0:
            vn = 0.0
            fn = 0.0
            for a in range(nf):
                i = free[a]
                vn += v[a, 0]**2 + v[a, 1]**2 + v[a, 2]**2
                fn += F[i, 0]**2 + F[i, 1]**2 + F[i, 2]**2
            vn = np.sqrt(vn)
            fn = np.sqrt(fn)
            if fn < 1e-12:
                fn = 1e-12
            for a in range(nf):
                i = free[a]
                for c in range(3):
                    v[a, c] = (1.0 - alpha) * v[a, c] + alpha * vn * F[i, c] / fn
            n_pos += 1
            if n_pos > n_min:
                dt = min(dt * f_inc, dt_max)
                alpha *= f_alpha
        else:
            n_pos = 0
            dt *= f_dec
            alpha = alpha0
            for a in range(nf):
                v[a, 0] = 0.0
                v[a, 1] = 0.0
                v[a, 2] = 0.0
        for a in range(nf):
            i = free[a]
            dx0 = 0.0
            dx1 = 0.0
            dx2 = 0.0
            for c in range(3):
                v[a, c] += dt * F[i, c]
            dx0 = dt * v[a, 0]
            dx1 = dt * v[a, 1]
            dx2 = dt * v[a, 2]
            # Per-atom displacement cap, as in the reference implementation.
            # Without it a single FIRE step can throw an atom clear across the
            # patch, and the two paths diverge: measured, accept/reject agreed
            # on only 91% of moves and the uncapped run ended at higher energy.
            nrm = np.sqrt(dx0 * dx0 + dx1 * dx1 + dx2 * dx2)
            if nrm > max_step:
                sc = max_step / nrm
                dx0 *= sc
                dx1 *= sc
                dx2 *= sc
            pos[i, 0] += dx0
            pos[i, 1] += dx1
            pos[i, 2] += dx2
    # Energy must be re-evaluated at the FINAL positions; returning the value
    # from the last force call reports the energy one step stale.
    E, _F = patch_ef_nb(pos, bonds, angles, L, kb, ka_arr, d2, off_arr, False, nb_pairs, r_rep, k_rep, r_core, k_core, quartic)
    return E, n_iter, aborted
