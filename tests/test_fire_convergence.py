"""Smoke tests for FIRE convergence on multi-species cells.

Pre-2026-05 the multi-species FIRE quench did not converge on
``SiO₂`` / ``SrTiO₃`` because ``from_atoms`` extracted bond springs
for second-shell lattice-artefact pairs.  These tests verify the
post-fix behaviour: bond loss must drop monotonically (i.e. FIRE is
actually doing useful work) for SiO₂ MRO and SrTiO₃ MRO.
"""
from __future__ import annotations

import numpy as np
import pytest

import tricor as tc
from tricor.shells import CoordinationShellTarget


def _final_loss(cell, key):
    """Return the final value of one of the loss components from the
    most recent ``shell_relax_history``."""
    arr = np.asarray(cell.shell_relax_history.get(key, []))
    if arr.size == 0:
        return None
    return float(arr[-1])


def _initial_loss(cell, key):
    arr = np.asarray(cell.shell_relax_history.get(key, []))
    if arr.size == 0:
        return None
    return float(arr[0])


def test_fire_converges_si_amorphous(atoms_si):
    """Single-species Si should always converge — keep this as a
    sanity check that nothing bigger broke the pipeline."""
    shell = CoordinationShellTarget.from_atoms(atoms_si, phi_num_bins=36)
    cell = tc.Supercell.from_atoms(
        atoms_si,
        cell_dim_angstroms=(20.0, 20.0, 20.0),
        r_max=6.0, r_step=0.1, phi_num_bins=36, rng_seed=42,
    )
    cell.generate(
        shell,
        grain_size=None,
        num_steps=40,
        bond_weight=0.6, angle_weight=0.20, repulsion_weight=1.3,
        hard_core_scale=0.86, nonbond_push_scale=0.45,
        displacement_sigma=0.12,
        capture_trajectory=False,
        show_progress=False,
    )
    bond_initial = _initial_loss(cell, "bond_loss")
    bond_final = _final_loss(cell, "bond_loss")
    assert bond_initial is not None and bond_final is not None
    assert bond_final < bond_initial, (
        f"Si amorphous bond_loss did NOT drop: {bond_initial} → {bond_final}"
    )


def test_fire_converges_sio2_with_auto_filter(atoms_sio2):
    """SiO₂ is the canonical multi-species failure mode pre-fix.
    With the auto-filter on (default), bond_loss must drop
    substantially during FIRE — it didn't pre-fix because each Si had
    geometrically-incompatible springs to 4 O at 1.61 Å AND 4 Si at
    3.06 Å."""
    shell = CoordinationShellTarget.from_atoms(
        atoms_sio2, phi_num_bins=36,
    )
    # Verify the auto-filter is active (Si-Si and O-O zeroed).
    species_to_idx = {
        sym: i for i, sym in enumerate(shell.species_labels)
    }
    si = species_to_idx["Si"]
    o = species_to_idx["O"]
    assert shell.coordination_target[si, si] == 0.0
    assert shell.coordination_target[o, o] == 0.0
    cell = tc.Supercell.from_atoms(
        atoms_sio2,
        cell_dim_angstroms=(20.0, 20.0, 20.0),
        r_max=6.0, r_step=0.1, phi_num_bins=36, rng_seed=42,
    )
    cell.generate(
        shell,
        grain_size=8.0,
        num_steps=60,
        bond_weight=1.65, angle_weight=1.35, repulsion_weight=1.3,
        hard_core_scale=0.82, nonbond_push_scale=0.72,
        displacement_sigma=0.011,
        capture_trajectory=False,
        show_progress=False,
    )
    bond_initial = _initial_loss(cell, "bond_loss")
    bond_final = _final_loss(cell, "bond_loss")
    assert bond_final < bond_initial * 0.5, (
        f"SiO₂ MRO bond_loss should drop ≥50%; got "
        f"{bond_initial:.4f} → {bond_final:.4f} "
        f"({100 * bond_final / bond_initial:.0f}% remaining)"
    )


def test_fire_converges_srtio3_with_angle_whitelist(atoms_srtio3):
    """SrTiO₃ requires both the auto-filter (zero Sr-Sr / Ti-Ti / O-O
    / Sr-Ti bonds) AND an angle whitelist to silence the multi-modal
    Sr-centred angles.  With both, FIRE must converge on bond
    loss."""
    shell = (
        CoordinationShellTarget.from_atoms(atoms_srtio3, phi_num_bins=36)
        .with_angle_triplets([("Ti", "O", "O"), ("O", "Ti", "Ti")])
    )
    cell = tc.Supercell.from_atoms(
        atoms_srtio3,
        cell_dim_angstroms=(20.0, 20.0, 20.0),
        r_max=6.0, r_step=0.1, phi_num_bins=36, rng_seed=42,
    )
    cell.generate(
        shell,
        grain_size=8.0,
        num_steps=80,
        bond_weight=1.0, angle_weight=0.7, repulsion_weight=1.2,
        hard_core_scale=1.10, nonbond_push_scale=0.75,
        displacement_sigma=0.005,
        capture_trajectory=False,
        show_progress=False,
    )
    bond_initial = _initial_loss(cell, "bond_loss")
    bond_final = _final_loss(cell, "bond_loss")
    assert bond_final < bond_initial * 0.7, (
        f"SrTiO₃ MRO bond_loss should drop ≥30%; got "
        f"{bond_initial:.4f} → {bond_final:.4f}"
    )


def test_no_nan_or_inf_in_history(atoms_si):
    """A FIRE run that diverged would produce NaN / inf in any of the
    loss components.  Make sure the standard pipeline never does."""
    shell = CoordinationShellTarget.from_atoms(atoms_si, phi_num_bins=36)
    cell = tc.Supercell.from_atoms(
        atoms_si,
        cell_dim_angstroms=(20.0, 20.0, 20.0),
        r_max=6.0, r_step=0.1, phi_num_bins=36, rng_seed=42,
    )
    cell.generate(
        shell,
        grain_size=None,
        num_steps=40,
        bond_weight=0.6, angle_weight=0.2, repulsion_weight=1.3,
        hard_core_scale=0.86, nonbond_push_scale=0.45,
        displacement_sigma=0.12,
        capture_trajectory=False,
        show_progress=False,
    )
    for key in ("loss", "bond_loss", "angle_loss", "repulsion_loss"):
        arr = np.asarray(cell.shell_relax_history.get(key, []))
        if arr.size == 0:
            continue
        assert np.all(np.isfinite(arr)), (
            f"{key} contains non-finite values: "
            f"min={float(np.min(arr))}, max={float(np.max(arr))}"
        )
