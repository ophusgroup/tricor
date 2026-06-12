"""Calibrate spring-network stiffnesses against the MACE-MP0 potential.

:func:`calibrate_to_mace` measures, for a shell target's reference
crystal, the effective stiffness of every bonded species pair and
every enabled angle triplet on the MACE potential-energy surface, plus
a Morse fit of each bond's anharmonicity and a hard-core estimate from
the repulsive wall.  The reference cell is tiny (2-10 atoms), so the
whole calibration is ~20-40 MACE single points per pair/triplet type
and runs in seconds-to-a-minute per material on CPU.

Methods
-------
- ``pair_k_bond``: interatomic-Hessian projection.  Displace the
  representative neighbour ``j`` by ±delta along the bond axis and
  read the force change on ``i``:
  ``k = u . (F_i(+) - F_i(-)) / (2 delta)``.  This isolates the i-j
  force constant from the other springs attached to either atom
  (which contaminate a rigid energy scan's curvature).
- ``pair_morse_*``: rigid bond scan ``r = s * r0`` fitted with
  ``U = D (1 - exp(-a (r - r_e)))^2 + c``.  The scan curvature
  includes environment compliance, so the Morse parameters describe
  the *effective* bond curve seen along that internal coordinate.
- ``pair_hard_min_mace``: distance on the compressive branch of the
  scan where the energy rises ``hard_core_rise_eV`` above the
  minimum.
- ``angle_k``: rigid bend.  Rotate neighbour ``a`` about the axis
  through the centre perpendicular to the (a, c, b) plane; quadratic
  fit of ``E(phi)``.

All scans run on a periodic repeat of the reference cell with a
minimum image distance of ``min_image`` so the displaced atom does
not interact with its own images at first order (MACE per-layer
cutoff is 6 Å).

``mace-torch`` is an optional dependency, imported lazily.
"""

from __future__ import annotations

import dataclasses
import time
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .shells import CoordinationShellTarget


def _rodrigues(axis: np.ndarray, angle: float) -> np.ndarray:
    c, s = float(np.cos(angle)), float(np.sin(angle))
    C = 1.0 - c
    x, y, z = (float(v) for v in axis)
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])


def _load_mace(model: str, device: str, dtype: str):
    try:
        from mace.calculators import mace_mp
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "calibrate_to_mace requires the optional 'mace-torch' "
            "dependency:  pip install torch 'mace-torch>=0.3'"
        ) from exc
    return mace_mp(model=model, device=device, default_dtype=dtype)


def calibrate_to_mace(
    shell: "CoordinationShellTarget",
    *,
    model: str = "medium-mpa-0",
    anharmonic: bool = True,
    device: str = "cpu",
    dtype: str = "float64",
    min_image: float = 10.5,
    bond_scan: tuple[float, float] = (0.72, 1.35),
    n_bond_scan: int = 15,
    angle_scan_deg: float = 6.0,
    n_angle_scan: int = 9,
    fd_delta: float = 0.01,
    hard_core_rise_eV: float = 1.0,
    morse_fit_window_eV: float = 4.0,
    show_progress: bool = True,
) -> "CoordinationShellTarget":
    """Return a copy of ``shell`` with MACE-calibrated stiffness fields.

    Parameters
    ----------
    shell
        Target produced by :meth:`CoordinationShellTarget.from_atoms`.
        Composite targets (:meth:`from_targets`) are not supported —
        calibrate each sub-target before composing.
    model, device, dtype
        Forwarded to ``mace.calculators.mace_mp``.  ``float64``
        (default) keeps finite differences clean.
    anharmonic
        Fit Morse parameters from the bond scan (default ``True``).
        ``False`` skips the scan fit but still runs the scan for the
        hard-core estimate.
    min_image
        Minimum image distance (Å) of the periodic repeat the scans
        run in.
    bond_scan, n_bond_scan
        Scan range as multiples of the bond peak, and point count.
    angle_scan_deg, n_angle_scan
        Half-range (degrees) and point count of the bend scan.
    fd_delta
        Finite-difference displacement (Å) for the Hessian projection.
    hard_core_rise_eV
        Energy rise above the bond minimum defining
        ``pair_hard_min_mace``.
    """
    species = np.asarray(shell.species, dtype=np.int64)
    if len(species) != len(np.unique(species)):
        raise NotImplementedError(
            "calibrate_to_mace does not support composite targets "
            "(duplicated atomic numbers across virtual species).  "
            "Calibrate each sub-target before from_targets()."
        )

    ref = shell.atoms
    coord = np.asarray(shell.coordination_target, dtype=np.float64)
    pair_peak = np.asarray(shell.pair_peak, dtype=np.float64)
    pair_inner = np.asarray(shell.pair_inner, dtype=np.float64)
    pair_outer = np.asarray(shell.pair_outer, dtype=np.float64)
    angle_index = np.asarray(shell.angle_index, dtype=np.intp)
    angle_mode = np.asarray(shell.angle_mode_deg, dtype=np.float64)
    angle_mask = np.asarray(shell.angle_enabled_mask, dtype=bool)
    n_sp = len(species)

    calc = _load_mace(model, device, dtype)

    # Periodic repeat reaching the minimum image distance.
    lengths = np.linalg.norm(np.asarray(ref.cell.array), axis=1)
    reps = tuple(int(np.ceil(min_image / L)) for L in lengths)
    probe = ref.repeat(reps)
    probe.calc = calc
    base_pos = probe.positions.copy()
    cell_mat = np.asarray(probe.cell.array, dtype=np.float64)
    cell_inv = np.linalg.inv(cell_mat)
    sp_idx = np.searchsorted(species, probe.numbers)
    t_start = time.time()

    def energy(positions) -> float:
        probe.positions = positions
        return float(probe.get_potential_energy())

    def forces(positions) -> np.ndarray:
        probe.positions = positions
        return np.asarray(probe.get_forces(), dtype=np.float64)

    def mic(vec):
        frac = vec @ cell_inv
        frac -= np.round(frac)
        return frac @ cell_mat

    # Neighbour table on the repeat (undisplaced geometry).
    from ase.neighborlist import neighbor_list
    nl_i, nl_j, nl_d = neighbor_list(
        "ijd", probe, float(shell.max_pair_outer) * 1.05)

    # ---- bonded pair types ----
    K = np.full((n_sp, n_sp), np.nan)
    D = np.full((n_sp, n_sp), np.nan)
    A = np.full((n_sp, n_sp), np.nan)
    RE = np.full((n_sp, n_sp), np.nan)
    HARD = np.full((n_sp, n_sp), np.nan)
    info_pairs = {}

    for ia in range(n_sp):
        for ib in range(ia, n_sp):
            if coord[ia, ib] <= 0:
                continue
            r0 = pair_peak[ia, ib]
            m = (((sp_idx[nl_i] == ia) & (sp_idx[nl_j] == ib))
                 | ((sp_idx[nl_i] == ib) & (sp_idx[nl_j] == ia)))
            m &= (nl_d >= pair_inner[ia, ib]) & (nl_d <= pair_outer[ia, ib])
            if not m.any():
                continue
            k_best = int(np.flatnonzero(m)[np.argmin(np.abs(nl_d[m] - r0))])
            i_at, j_at = int(nl_i[k_best]), int(nl_j[k_best])
            vec = mic(base_pos[j_at] - base_pos[i_at])
            r_act = float(np.linalg.norm(vec))
            u = vec / r_act

            # Hessian projection: k = u . dF_i/d(r_j . u) . u
            pos = base_pos.copy()
            pos[j_at] = base_pos[j_at] + fd_delta * u
            f_plus = forces(pos)[i_at]
            pos[j_at] = base_pos[j_at] - fd_delta * u
            f_minus = forces(pos)[i_at]
            k_fd = float(u @ (f_plus - f_minus)) / (2.0 * fd_delta)

            # Rigid scan along the bond axis.
            s_grid = np.unique(np.concatenate([
                np.linspace(bond_scan[0], bond_scan[1], n_bond_scan),
                [1.0],
            ]))
            scan_r = s_grid * r_act
            scan_E = np.empty_like(scan_r)
            for n, r in enumerate(scan_r):
                pos = base_pos.copy()
                pos[j_at] = base_pos[j_at] + u * (r - r_act)
                scan_E[n] = energy(pos)
            scan_E -= scan_E.min()

            # Hard core: compressive branch crossing of the rise.
            i_min = int(np.argmin(scan_E))
            r_hard = np.nan
            comp_r, comp_E = scan_r[: i_min + 1], scan_E[: i_min + 1]
            above = comp_E >= hard_core_rise_eV
            if above.any() and not above.all():
                k_hi = int(np.flatnonzero(above)[-1])
                r1, r2 = comp_r[k_hi], comp_r[k_hi + 1]
                e1, e2 = comp_E[k_hi], comp_E[k_hi + 1]
                r_hard = float(r1 + (e1 - hard_core_rise_eV)
                               * (r2 - r1) / max(e1 - e2, 1e-12))

            d_fit = a_fit = re_fit = rms = np.nan
            if anharmonic:
                from scipy.optimize import curve_fit

                def morse(r, d, a, re, c):
                    return d * (1.0 - np.exp(-a * (r - re))) ** 2 + c

                # Fit inside an energy window: the relaxer samples the
                # region near the minimum, and a single Morse cannot
                # track a stiff ionic wall over many eV.
                fit_m = scan_E <= morse_fit_window_eV
                if fit_m.sum() < 6:
                    fit_m = scan_E <= np.sort(scan_E)[5]
                a0 = 1.8
                p0 = [max(abs(k_fd), 0.5) / (2 * a0 * a0), a0, r_act, 0.0]
                try:
                    popt, _ = curve_fit(
                        morse, scan_r[fit_m], scan_E[fit_m], p0=p0,
                        bounds=([1e-3, 0.3, r_act - 0.4, -1.0],
                                [200.0, 6.0, r_act + 0.4, 1.0]),
                        maxfev=20000)
                    d_fit, a_fit, re_fit = (float(popt[0]), float(popt[1]),
                                            float(popt[2]))
                    rms = float(np.sqrt(np.mean(
                        (morse(scan_r[fit_m], *popt) - scan_E[fit_m]) ** 2)))
                except Exception:
                    pass

            # Ionic spectator pairs (e.g. Sr-O) can have a negative
            # bare interatomic constant — lattice stability comes from
            # the other terms.  A non-positive constant is unusable as
            # a spring, so the pair is stored as NaN: the relaxer then
            # falls back to the neutral default stiffness for that
            # pair (a zero spring would remove the pair's cohesion
            # entirely).  The raw value stays in the info dict.
            k_store = k_fd if k_fd > 0 else float("nan")
            if not np.isfinite(k_store):
                # No usable harmonic constant → no trustworthy Morse
                # fit either; the pair runs on the harmonic default.
                d_fit = a_fit = re_fit = float("nan")
            for x, y in ((ia, ib), (ib, ia)):
                K[x, y] = k_store
                D[x, y], A[x, y], RE[x, y] = d_fit, a_fit, re_fit
                HARD[x, y] = r_hard
            labels = (shell.species_labels[ia], shell.species_labels[ib])
            info_pairs["-".join(labels)] = dict(
                r0=r0, r_scan_center=r_act, k_bond=k_store,
                k_bond_raw=k_fd, clamped_nonnegative=bool(k_fd < 0),
                morse_D=d_fit, morse_a=a_fit, morse_r=re_fit,
                fit_rms=rms, r_hard_mace=r_hard,
                scan_r=scan_r.tolist(), scan_E=scan_E.tolist(),
            )
            if show_progress:
                print(f"  bond {labels[0]}-{labels[1]}: "
                      f"k={k_store:.3f} eV/Å²  D={d_fit:.3f} eV  "
                      f"a={a_fit:.3f} 1/Å  r_hard={r_hard:.3f} Å  "
                      f"rms={rms:.4f}"
                      + (f"  [raw k={k_fd:.3f} clamped]" if k_fd < 0 else ""),
                      flush=True)

    # ---- angle triplet types ----
    AK = np.full(angle_mode.shape[0], np.nan)
    info_angles = {}
    # Bonded-neighbour lookup for representative-triplet search.
    bond_ok = (nl_d >= pair_inner[sp_idx[nl_i], sp_idx[nl_j]]) \
        & (nl_d <= pair_outer[sp_idx[nl_i], sp_idx[nl_j]]) \
        & (coord[sp_idx[nl_i], sp_idx[nl_j]] > 0)

    for t in range(angle_mode.shape[0]):
        if not angle_mask[t]:
            continue
        ci, na, nb = (int(angle_index[t, 0]), int(angle_index[t, 1]),
                      int(angle_index[t, 2]))
        if coord[ci, na] <= 0 or coord[ci, nb] <= 0:
            continue
        phi0 = np.deg2rad(angle_mode[t])
        # Representative triplet: centre of species ci with bonded
        # neighbours of species na, nb at an angle nearest the mode.
        best = None
        centers = np.flatnonzero(sp_idx == ci)
        for c_at in centers[:64]:
            sel = bond_ok & (nl_i == c_at)
            nbrs = nl_j[sel]
            cand_a = nbrs[sp_idx[nbrs] == na]
            cand_b = nbrs[sp_idx[nbrs] == nb]
            for a_at in cand_a[:6]:
                for b_at in cand_b[:6]:
                    if a_at == b_at:
                        continue
                    va = mic(base_pos[a_at] - base_pos[c_at])
                    vb = mic(base_pos[b_at] - base_pos[c_at])
                    cosp = float(va @ vb
                                 / (np.linalg.norm(va) * np.linalg.norm(vb)))
                    phi = float(np.arccos(np.clip(cosp, -1, 1)))
                    err = abs(phi - phi0)
                    if best is None or err < best[0]:
                        best = (err, int(c_at), int(a_at), int(b_at),
                                va, vb, phi)
            if best is not None and best[0] < np.deg2rad(2.0):
                break
        if best is None:
            continue
        _, c_at, a_at, b_at, va, vb, phi_act = best
        # Rotate the endpoint with fewer bonded neighbours — the bend
        # coordinate is the same either way, but the bond-stretch
        # contamination scales with the moving atom's coordination.
        deg_a = int(np.sum(bond_ok & (nl_i == a_at)))
        deg_b = int(np.sum(bond_ok & (nl_i == b_at)))
        if deg_b < deg_a:
            a_at, b_at = b_at, a_at
            va, vb = vb, va
        axis = np.cross(va, vb)
        norm = np.linalg.norm(axis)
        if norm < 1e-8:        # 180° triplet: any perpendicular axis
            trial = np.array([1.0, 0.0, 0.0])
            if abs(va @ trial) > 0.9 * np.linalg.norm(va):
                trial = np.array([0.0, 1.0, 0.0])
            axis = np.cross(va, trial)
            norm = np.linalg.norm(axis)
        axis /= norm

        eps = np.deg2rad(np.linspace(-angle_scan_deg, angle_scan_deg,
                                     n_angle_scan))
        E = np.empty_like(eps)
        for n, e in enumerate(eps):
            pos = base_pos.copy()
            pos[a_at] = base_pos[c_at] + _rodrigues(axis, e) @ va
            E[n] = energy(pos)
        E -= E.min()
        coeffs = np.polyfit(eps, E, 2)
        k_scan = float(2.0 * coeffs[0])

        # Rotating ``a`` also stretches a's other bonds; that
        # contribution to the scan curvature is exactly
        # ``sum_n k_bond[s_a, s_n] (v . u_an)^2`` with the rotation
        # velocity ``v = w x r_ca`` (|v| = r_ca).  Subtract it using
        # the Hessian-projected k_bond measured above.  Other-angle
        # couplings (angles sharing atom a or c) are not removed and
        # remain a known overestimate.
        v_rot = np.cross(axis, va)
        bond_sub = 0.0
        sel_a = bond_ok & (nl_i == a_at)
        for n_at in nl_j[sel_a]:
            if int(n_at) == c_at:
                continue
            k_an = K[int(sp_idx[a_at]), int(sp_idx[n_at])]
            if np.isnan(k_an):
                continue
            u_an = mic(base_pos[int(n_at)] - base_pos[a_at])
            u_an /= np.linalg.norm(u_an)
            bond_sub += k_an * float(v_rot @ u_an) ** 2
        k_angle = k_scan - bond_sub
        clamped = k_angle < 0
        if clamped:
            k_angle = 0.0
        AK[t] = k_angle
        labels = (shell.species_labels[na], shell.species_labels[ci],
                  shell.species_labels[nb])
        info_angles["-".join(labels)] = dict(
            mode_deg=float(angle_mode[t]), phi_rep_deg=float(np.rad2deg(phi_act)),
            k_angle=k_angle, k_scan_raw=k_scan,
            bond_stretch_subtracted=float(bond_sub),
            clamped_nonnegative=bool(clamped),
            fit_rms=float(np.sqrt(np.mean(
                (np.polyval(coeffs, eps) - E) ** 2))),
        )
        if show_progress:
            print(f"  angle {labels[0]}-{labels[1]}-{labels[2]} "
                  f"({angle_mode[t]:.1f}°): k={k_angle:.3f} eV/rad² "
                  f"(scan {k_scan:.3f} − bonds {bond_sub:.3f})"
                  + ("  [clamped]" if clamped else ""), flush=True)

    k_bonds = K[~np.isnan(K)]
    k_angles = AK[~np.isnan(AK)]
    info = dict(
        model=model, device=device, dtype=dtype,
        repeat=reps, n_atoms=len(probe),
        anharmonic=bool(anharmonic),
        pairs=info_pairs, angles=info_angles,
        mean_k_bond=float(k_bonds.mean()) if k_bonds.size else np.nan,
        mean_k_angle=float(k_angles.mean()) if k_angles.size else np.nan,
        angle_over_bond=(float(k_angles.mean() / k_bonds.mean())
                         if k_bonds.size and k_angles.size else np.nan),
        wall_time_s=float(time.time() - t_start),
    )
    if show_progress:
        print(f"  calibrated in {info['wall_time_s']:.1f}s — "
              f"mean k_bond={info['mean_k_bond']:.3f} eV/Å², "
              f"mean k_angle={info['mean_k_angle']:.3f} eV/rad², "
              f"angle/bond={info['angle_over_bond']:.3f}", flush=True)

    return dataclasses.replace(
        shell,
        pair_k_bond=K, pair_morse_D=D, pair_morse_a=A, pair_morse_r=RE,
        pair_hard_min_mace=HARD, angle_k=AK, mace_calibration=info,
    )
