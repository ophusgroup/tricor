"""Tests for the virtual-species fix in orientation refinement.

When ``Supercell.generate(refine_orientations=True, ...)`` retiles
each grain into a rotated master block, the per-atom virtual-species
index must follow the grain's source (``species_offset``), NOT be
recomputed from atomic numbers — for sp²/sp³ carbon, all atoms
share atomic number 6 and ``searchsorted`` would tag every atom as
the first virtual species, producing all-sp² cells regardless of
the requested ``(w_graphite, w_diamond)`` weights.

Pre-fix: refined sp3_nc had 0 atoms tagged sp³_C (all 10887 atoms
collapsed to sp²_C).  This regression test pins the fix.
"""
from __future__ import annotations

import numpy as np
import pytest

import tricor as tc
from tricor.shells import CoordinationShellTarget


def _build_carbon_sp_cell(atoms_graphite, atoms_diamond, w_graphite, w_diamond):
    """Build a small carbon SP cell at the requested sp²/sp³ ratio
    via composite shell + grain_sources."""
    shell_sp2 = CoordinationShellTarget.from_atoms(
        atoms_graphite, phi_num_bins=36,
    )
    shell_sp3 = CoordinationShellTarget.from_atoms(
        atoms_diamond, phi_num_bins=36,
    )
    shell = CoordinationShellTarget.from_targets(
        {"sp2": shell_sp2, "sp3": shell_sp3}
    )
    cell = tc.Supercell.from_atoms(
        atoms_graphite,
        cell_dim_angstroms=(20.0, 20.0, 20.0),
        r_max=6.0, r_step=0.1, phi_num_bins=36, rng_seed=42,
    )
    cell.generate(
        shell,
        grain_size=8.0,
        grain_sources=[
            {"atoms": atoms_graphite, "species_offset": 0,
             "weight": w_graphite},
            {"atoms": atoms_diamond, "species_offset": 1,
             "weight": w_diamond},
        ],
        num_steps=20,
        bond_weight=2.5, angle_weight=1.0, repulsion_weight=2.0,
        hard_core_scale=0.92, nonbond_push_scale=0.85,
        displacement_sigma=0.02,
        refine_orientations=True,
        refine_orientations_kwargs=dict(
            amplitudes_deg=(15.0, 5.0),
            trials_per_amplitude_per_grain=8,
            max_rounds_per_amplitude=1,
            cost_function="pair_distance",
            score_cutoff_factor=1.5,
            time_budget_sec=60.0,
            rng_seed=2024,
            show_progress=False,
        ),
        capture_trajectory=False,
        show_progress=False,
    )
    return cell


def test_sp3_only_keeps_sp3_species(atoms_graphite, atoms_diamond):
    """100% diamond grains → every atom must end up tagged sp³_C
    (virtual species index 1) after orientation refinement.

    Pre-fix: every atom collapsed to sp²_C (index 0)."""
    cell = _build_carbon_sp_cell(atoms_graphite, atoms_diamond, 0.0, 1.0)
    sidx = np.asarray(cell._atom_shell_species_index)
    n_sp2 = int(np.sum(sidx == 0))
    n_sp3 = int(np.sum(sidx == 1))
    n_total = len(cell.atoms)
    # Allow up to 5% drift (boundary atoms can swap during refinement
    # if a different rotation is accepted) but the dominant species
    # must be the requested one.
    assert n_sp3 >= 0.95 * n_total, (
        f"Expected >=95% sp³_C, got {n_sp3}/{n_total} ({100*n_sp3/n_total:.0f}%)"
    )
    assert n_sp2 == 0 or n_sp2 < 0.05 * n_total


def test_sp2_only_keeps_sp2_species(atoms_graphite, atoms_diamond):
    """100% graphite grains → every atom must end up tagged sp²_C
    (virtual species index 0) after orientation refinement."""
    cell = _build_carbon_sp_cell(atoms_graphite, atoms_diamond, 1.0, 0.0)
    sidx = np.asarray(cell._atom_shell_species_index)
    n_sp2 = int(np.sum(sidx == 0))
    n_sp3 = int(np.sum(sidx == 1))
    n_total = len(cell.atoms)
    assert n_sp2 >= 0.95 * n_total, (
        f"Expected >=95% sp²_C, got {n_sp2}/{n_total} ({100*n_sp2/n_total:.0f}%)"
    )


def test_mixed_keeps_both_species(atoms_graphite, atoms_diamond):
    """50/50 mix → both virtual species must be present in
    appreciable amounts after refinement."""
    cell = _build_carbon_sp_cell(atoms_graphite, atoms_diamond, 0.5, 0.5)
    sidx = np.asarray(cell._atom_shell_species_index)
    n_sp2 = int(np.sum(sidx == 0))
    n_sp3 = int(np.sum(sidx == 1))
    n_total = len(cell.atoms)
    # Each species should be at least 15% of the cell — well above
    # the failure mode (one species at 0%).  The exact ratio depends
    # on per-atom densities of graphite vs diamond and on which
    # grains the rng picks.
    assert n_sp2 >= 0.15 * n_total
    assert n_sp3 >= 0.15 * n_total


def test_grain_master_carries_species_offset(atoms_graphite, atoms_diamond):
    """The internal ``_grain_masters`` list (built by
    ``_grain.py``) must carry ``species_offset`` per master so the
    refinement retile can recover virtual species without falling
    back to atomic-number searchsorted."""
    shell_sp2 = CoordinationShellTarget.from_atoms(
        atoms_graphite, phi_num_bins=36,
    )
    shell_sp3 = CoordinationShellTarget.from_atoms(
        atoms_diamond, phi_num_bins=36,
    )
    shell = CoordinationShellTarget.from_targets(
        {"sp2": shell_sp2, "sp3": shell_sp3}
    )
    cell = tc.Supercell.from_atoms(
        atoms_graphite,
        cell_dim_angstroms=(20.0, 20.0, 20.0),
        r_max=6.0, r_step=0.1, phi_num_bins=36, rng_seed=42,
    )
    cell.generate(
        shell,
        grain_size=8.0,
        grain_sources=[
            {"atoms": atoms_graphite, "species_offset": 0, "weight": 0.5},
            {"atoms": atoms_diamond, "species_offset": 1, "weight": 0.5},
        ],
        num_steps=5,
        bond_weight=2.5, angle_weight=1.0, repulsion_weight=2.0,
        capture_trajectory=False,
        show_progress=False,
    )
    masters = cell._grain_masters
    assert len(masters) == 2
    assert masters[0]["species_offset"] == 0
    assert masters[1]["species_offset"] == 1
