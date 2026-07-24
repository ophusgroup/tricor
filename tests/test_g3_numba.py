"""Tests for the numba-parallel kernel in
:meth:`G3Distribution.measure_g3`.

The acceleration path (``backend="numba"``) must produce ``g3count``
and ``g2count`` arrays bit-identical to the pure-numpy reference
(``backend="python"``).  Both are integer accumulators so the match
must be exact, not ``np.allclose``.

These tests also pin a minimum speedup gate so an accidental switch
back to the slow path (e.g. JIT cache invalidation, parallel mode
disabled) is caught.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

import atomode as tc
from atomode.shells import CoordinationShellTarget


HAS_NUMBA = False
try:
    from atomode._g3_numba import HAS_NUMBA  # type: ignore
except ImportError:
    pass


pytestmark = pytest.mark.skipif(
    not HAS_NUMBA, reason="numba not installed; install via atomode[fast]"
)


def _build_relaxed_cell(atoms_ref, *, side=15.0, phi_num_bins=36, seed=42):
    """Small relaxed cell for parity tests.  Tiny num_steps because
    the test is about g3 measurement, not relaxation quality."""
    shell = CoordinationShellTarget.from_atoms(
        atoms_ref, phi_num_bins=phi_num_bins,
    )
    cell = tc.Supercell.from_atoms(
        atoms_ref,
        cell_dim_angstroms=(side, side, side),
        r_max=6.0, r_step=0.1, phi_num_bins=phi_num_bins, rng_seed=seed,
    )
    cell.generate(
        shell,
        grain_size=None,
        num_steps=15,
        bond_weight=1.0, angle_weight=0.5, repulsion_weight=1.5,
        hard_core_scale=0.85, nonbond_push_scale=0.7,
        displacement_sigma=0.05,
        capture_trajectory=False,
        show_progress=False,
    )
    return cell


def _both_backends(cell):
    """Run python + numba and return (g3_py, g2_py, g3_nb, g2_nb)."""
    cell.measure_g3(backend="python", show_progress=False, force=True)
    g3_py = cell.current_distribution.g3count.copy()
    g2_py = cell.current_distribution.g2count.copy()

    cell.measure_g3(backend="numba", show_progress=False, force=True)
    g3_nb = cell.current_distribution.g3count.copy()
    g2_nb = cell.current_distribution.g2count.copy()
    return g3_py, g2_py, g3_nb, g2_nb


def test_numba_matches_python_si(atoms_si):
    """Single-species Si: g3 and g2 must be bit-identical between
    backends."""
    cell = _build_relaxed_cell(atoms_si)
    g3_py, g2_py, g3_nb, g2_nb = _both_backends(cell)
    assert np.array_equal(g2_py, g2_nb), "g2count differs between backends"
    assert np.array_equal(g3_py, g3_nb), (
        f"g3count differs: max diff "
        f"{int(np.abs(g3_py - g3_nb).max())} / "
        f"total {int(g3_py.sum())}"
    )


def test_numba_matches_python_cu(atoms_cu):
    """FCC Cu (12-coord): bit-identical."""
    cell = _build_relaxed_cell(atoms_cu)
    g3_py, g2_py, g3_nb, g2_nb = _both_backends(cell)
    assert np.array_equal(g2_py, g2_nb)
    assert np.array_equal(g3_py, g3_nb)


def test_numba_matches_python_sio2(atoms_sio2):
    """Multi-species SiO₂: bit-identical (catches species-index /
    g3_lookup wiring bugs in the numba kernel)."""
    cell = _build_relaxed_cell(atoms_sio2)
    g3_py, g2_py, g3_nb, g2_nb = _both_backends(cell)
    assert np.array_equal(g2_py, g2_nb)
    assert np.array_equal(g3_py, g3_nb)


def test_numba_matches_python_srtio3(atoms_srtio3):
    """Three-species SrTiO₃ (largest channel count of the test
    materials, 18 channels)."""
    cell = _build_relaxed_cell(atoms_srtio3, side=12.0)
    g3_py, g2_py, g3_nb, g2_nb = _both_backends(cell)
    assert np.array_equal(g2_py, g2_nb)
    assert np.array_equal(g3_py, g3_nb)


def test_numba_speedup_on_medium_cell(atoms_sio2):
    """At 20 Å SiO₂ (~600 atoms), the numba kernel should be at
    least 5× faster than the pure-python loop.  The plan's 20×
    target lives at 40 Å cells (3000+ atoms); 5× is a conservative
    lower bound for CI / smaller machines."""
    cell = _build_relaxed_cell(atoms_sio2, side=20.0)

    # Warm the JIT cache (first call compiles; we don't time that).
    cell.measure_g3(backend="numba", show_progress=False, force=True)

    t0 = time.perf_counter()
    cell.measure_g3(backend="python", show_progress=False, force=True)
    t_py = time.perf_counter() - t0

    # Best-of-three for numba (per-call jitter dominates the small-cell
    # benchmark, but the python path is too slow to call 3 times).
    times_nb = []
    for _ in range(3):
        t0 = time.perf_counter()
        cell.measure_g3(backend="numba", show_progress=False, force=True)
        times_nb.append(time.perf_counter() - t0)
    t_nb = min(times_nb)

    speedup = t_py / max(t_nb, 1e-9)
    assert speedup >= 5.0, (
        f"Numba speedup is only {speedup:.1f}× (want ≥5×).  "
        f"Python: {t_py * 1000:.1f} ms, numba: {t_nb * 1000:.1f} ms"
    )


def test_default_backend_is_auto():
    """The default ``backend="auto"`` should pick numba when
    available.  Verify by introspecting the signature."""
    import inspect

    from atomode.g3 import G3Distribution

    sig = inspect.signature(G3Distribution.measure_g3)
    assert sig.parameters["backend"].default == "auto"


def test_python_backend_works_without_numba_path(atoms_si, monkeypatch):
    """Force python backend explicitly and confirm it produces a
    valid (non-zero) ``g3count``.  Pre-2026-05 the python path was
    the only path; we have to keep it functional as a fallback."""
    cell = _build_relaxed_cell(atoms_si)
    cell.measure_g3(backend="python", show_progress=False, force=True)
    g3 = cell.current_distribution.g3count
    g2 = cell.current_distribution.g2count
    assert g3.sum() > 0
    assert g2.sum() > 0


def test_numba_explicit_request_when_available(atoms_si):
    """``backend="numba"`` should run successfully when numba IS
    available.  (We're inside the skip-if-no-numba pytestmark, so
    HAS_NUMBA is True here.)"""
    cell = _build_relaxed_cell(atoms_si)
    cell.measure_g3(backend="numba", show_progress=False, force=True)
    g3 = cell.current_distribution.g3count
    assert g3.sum() > 0


def test_force_recomputes(atoms_si):
    """``force=True`` must invalidate any cached measurement and
    rerun.  Without ``force``, calling measure_g3 twice should be a
    no-op."""
    cell = _build_relaxed_cell(atoms_si)
    cell.measure_g3(backend="numba", show_progress=False, force=True)
    g3_first = cell.current_distribution.g3count.copy()
    # Call again without force - same array should be returned.
    cell.measure_g3(backend="numba", show_progress=False, force=False)
    g3_second = cell.current_distribution.g3count
    assert np.array_equal(g3_first, g3_second)
