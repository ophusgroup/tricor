"""Acceptance-gate tests for the ML backend on SiO₂.

Skipped unless a trained checkpoint exists at
``src/tricor/ml/data/sio2/checkpoint.pt``.  Run after
``scripts/ml_generate_data.py`` + ``python -m tricor.ml.train``.

Gates (per regime, on a single seed at 20³ Å — kept small so CI is fast):

1. **No NaNs.**  Predicted positions are all finite.
2. **No sub-NN bonds.**  Min interatomic distance > 0.55 × shortest
   target bond peak.  This is a soft gate; the FIRE-cleanup variant
   should pass at the 0.65 threshold.
3. **g3 fidelity.**  L2 distance of normalised g3 vs FIRE reference
   is < 1.0 (an empirical threshold; tighten as model improves).
4. **Atom count match.**  ML and FIRE must produce the same number of
   atoms (because they share the same Voronoi seed).

These are NOT the full "95% similarity" gates the user wants for
production — they're a regression net to catch obviously broken
checkpoints.  See ``scripts/ml_benchmark.py`` for the full sweep
across regimes / box sizes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

try:
    import torch  # noqa: F401
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

import tricor as tc
from tricor.shells import CoordinationShellTarget

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CKPT = _REPO_ROOT / "src/tricor/ml/data/sio2/checkpoint.pt"

pytestmark = [
    pytest.mark.skipif(not HAS_TORCH, reason="torch not installed"),
    pytest.mark.skipif(
        not DEFAULT_CKPT.is_file(),
        reason=f"no trained checkpoint at {DEFAULT_CKPT} "
        "— run scripts/ml_generate_data.py + tricor.ml.train first",
    ),
]


# Same regime → grain_size mapping as scripts/ml_generate_data.py
REGIMES_SIO2 = {
    "liquid":              None,
    "amorphous":           12.0,
    "short_range_order":   15.0,
    "medium_range_order":  20.0,
    "long_range_order":    26.0,
    "nanocrystalline":     35.0,
}

BOX_SIDE = 20.0  # small enough that the test runs in <30s per regime
RNG_SEED = 4242


def _build(atoms_sio2, regime: str):
    shell = CoordinationShellTarget.from_atoms(atoms_sio2, phi_num_bins=36)
    cell = tc.Supercell.from_atoms(
        atoms_sio2,
        cell_dim_angstroms=(BOX_SIDE, BOX_SIDE, BOX_SIDE),
        r_max=10.0, r_step=0.1, phi_num_bins=36, rng_seed=RNG_SEED,
    )
    return cell, shell


def _kw_for(regime: str) -> dict:
    """Get the per-regime generate-kwargs from regen_static_full."""
    import sys
    scripts_dir = _REPO_ROOT.parent / "tricor-docs" / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from regen_static_full import DISORDER_REGIMES
    _, kw = DISORDER_REGIMES[("silicon_dioxide", regime)]
    if isinstance(kw, str) and kw.startswith("preset:"):
        kw = tc.Supercell.PRESETS[kw.split(":", 1)[1]].copy()
    else:
        kw = dict(kw)
    kw.pop("freeze_grain_interiors", None)
    kw["show_progress"] = False
    return kw


def _normalised_g3(cell) -> np.ndarray:
    cell.measure_g3()
    g3 = cell.current_distribution.g3count.astype(np.float64)
    for c in range(g3.shape[0]):
        s = g3[c].sum()
        if s > 0:
            g3[c] = g3[c] / s
    return g3


def _min_pair_dist(atoms) -> float:
    d = atoms.get_all_distances(mic=True)
    np.fill_diagonal(d, np.inf)
    return float(d.min())


@pytest.mark.parametrize("regime", list(REGIMES_SIO2.keys()))
def test_ml_no_nans(atoms_sio2, regime):
    """ML output must be finite — sanity check."""
    cell, shell = _build(atoms_sio2, regime)
    cell.generate(
        shell, **_kw_for(regime),
        backend="ml", ml_model=str(DEFAULT_CKPT),
    )
    assert np.isfinite(cell.atoms.positions).all(), (
        f"ML produced non-finite positions for regime {regime!r}"
    )


@pytest.mark.parametrize("regime", ["amorphous", "medium_range_order",
                                    "nanocrystalline"])
def test_mlfire_no_sub_nn_bonds(atoms_sio2, regime):
    """ML+10-FIRE cleanup should never produce pairs below
    0.55 × shortest NN peak."""
    cell, shell = _build(atoms_sio2, regime)
    cell.generate(
        shell, **_kw_for(regime),
        backend="ml+fire", ml_model=str(DEFAULT_CKPT),
        ml_fire_cleanup_steps=15,
    )
    peaks = shell.pair_peak[shell.pair_peak > 1e-6]
    shortest_nn = float(peaks.min())
    threshold = 0.55 * shortest_nn
    min_d = _min_pair_dist(cell.atoms)
    assert min_d > threshold, (
        f"{regime}: min pair distance {min_d:.3f} Å < "
        f"{threshold:.3f} Å (0.55 × {shortest_nn:.3f}) — "
        "ML+FIRE failed to resolve atom overlap"
    )


@pytest.mark.parametrize("regime", ["amorphous", "medium_range_order",
                                    "nanocrystalline"])
def test_mlfire_g3_similarity(atoms_sio2, regime):
    """ML+10-FIRE should produce g3 within L2=1.0 of the FIRE reference."""
    cell_fire, shell_fire = _build(atoms_sio2, regime)
    cell_fire.generate(shell_fire, **_kw_for(regime))
    g3_fire = _normalised_g3(cell_fire)

    cell_ml, shell_ml = _build(atoms_sio2, regime)
    cell_ml.generate(
        shell_ml, **_kw_for(regime),
        backend="ml+fire", ml_model=str(DEFAULT_CKPT),
        ml_fire_cleanup_steps=15,
    )
    g3_ml = _normalised_g3(cell_ml)

    diff = g3_ml - g3_fire
    l2 = float(np.sqrt((diff * diff).sum()))
    assert l2 < 1.0, (
        f"{regime}: g3 L2 distance {l2:.3f} > 1.0 — "
        f"ML+FIRE diverged significantly from FIRE reference"
    )


@pytest.mark.parametrize("regime", ["short_range_order", "long_range_order"])
def test_ml_atom_count_matches_fire(atoms_sio2, regime):
    """ML and FIRE start from the same Voronoi seed so atom counts
    must match exactly."""
    cell_fire, shell_fire = _build(atoms_sio2, regime)
    cell_fire.generate(shell_fire, **_kw_for(regime))

    cell_ml, shell_ml = _build(atoms_sio2, regime)
    cell_ml.generate(
        shell_ml, **_kw_for(regime),
        backend="ml", ml_model=str(DEFAULT_CKPT),
    )
    assert len(cell_ml.atoms) == len(cell_fire.atoms), (
        f"{regime}: ML produced {len(cell_ml.atoms)} atoms but "
        f"FIRE produced {len(cell_fire.atoms)}"
    )
