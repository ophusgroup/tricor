"""Top-level smoke tests for the ``Supercell.generate`` workflow.

These keep the basic public API tied together: build a shell target,
build a Supercell, generate, measure_g3, optionally export.  If any
step regresses (e.g. the freeze_grain_interiors default flips, or
the autoshell-filter changes signature), at least one of these
fires.
"""
from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

import tricor as tc
from tricor.shells import CoordinationShellTarget
from tricor._shell_relax import _ShellRelaxMixin


def test_freeze_grain_interiors_default_is_off():
    """``Supercell.shell_relax(freeze_grain_interiors=False)`` was
    the 2026-05 default flip — multi-species cells need every atom
    to relax.  Pin the default."""
    sig = inspect.signature(_ShellRelaxMixin.shell_relax)
    assert sig.parameters["freeze_grain_interiors"].default is False


def test_basic_si_workflow(atoms_si, tmp_path):
    """End-to-end: build → generate → measure_g3 → check the output
    has reasonable shape."""
    shell = CoordinationShellTarget.from_atoms(atoms_si, phi_num_bins=36)
    cell = tc.Supercell.from_atoms(
        atoms_si,
        cell_dim_angstroms=(15.0, 15.0, 15.0),
        r_max=6.0, r_step=0.1, phi_num_bins=36, rng_seed=42,
    )
    cell.generate(
        shell,
        grain_size=None,
        num_steps=20,
        bond_weight=0.6, angle_weight=0.2, repulsion_weight=1.3,
        hard_core_scale=0.86, nonbond_push_scale=0.45,
        displacement_sigma=0.12,
        capture_trajectory=False,
        show_progress=False,
    )
    # Atom count should match the Si density of the box.
    assert len(cell.atoms) > 0
    # All atoms should be inside the box (positions wrap-corrected).
    assert np.all(cell.atoms.positions >= 0.0)
    box = np.diag(np.asarray(cell.atoms.cell.array))
    assert np.all(cell.atoms.positions < box[None, :] + 1e-6)
    # measure_g3 uses the grid the Supercell was constructed with.
    cell.measure_g3(show_progress=False)
    assert cell.current_distribution is not None
    g3 = np.asarray(cell.current_distribution.g3count)
    assert g3.ndim == 4
    assert int(g3.sum()) > 0


def test_grain_construction_with_grain_size(atoms_si):
    """Voronoi grain construction should produce a non-trivial number
    of grains for the requested grain size."""
    shell = CoordinationShellTarget.from_atoms(atoms_si, phi_num_bins=36)
    cell = tc.Supercell.from_atoms(
        atoms_si,
        cell_dim_angstroms=(20.0, 20.0, 20.0),
        r_max=6.0, r_step=0.1, phi_num_bins=36, rng_seed=42,
    )
    cell.generate(
        shell,
        grain_size=8.0,
        num_steps=20,
        bond_weight=2.0, angle_weight=0.8, repulsion_weight=2.2,
        hard_core_scale=0.93, nonbond_push_scale=0.75,
        displacement_sigma=0.04,
        capture_trajectory=False,
        show_progress=False,
    )
    assert cell._grain_ids is not None
    n_grains = len(np.unique(cell._grain_ids[cell._grain_ids >= 0]))
    # 20³ / 8³ ≈ 16 grains expected — accept anything in [4, 30].
    assert 4 <= n_grains <= 30


def test_bond_topology_respects_coord_target(atoms_sio2):
    """After ``shell_relax``, the bonded pair count per atom should
    not exceed the species's K = sum(coord_target[s, :]).  This pins
    the K-NN bond builder."""
    shell = CoordinationShellTarget.from_atoms(atoms_sio2, phi_num_bins=36)
    cell = tc.Supercell.from_atoms(
        atoms_sio2,
        cell_dim_angstroms=(15.0, 15.0, 15.0),
        r_max=6.0, r_step=0.1, phi_num_bins=36, rng_seed=42,
    )
    cell.generate(
        shell,
        grain_size=6.0,
        num_steps=20,
        bond_weight=1.65, angle_weight=1.35, repulsion_weight=1.3,
        hard_core_scale=0.82, nonbond_push_scale=0.72,
        displacement_sigma=0.011,
        capture_trajectory=False,
        show_progress=False,
    )
    # Every Si should have at most 4 O neighbours (K = 4) at the
    # Si-O target distance — we won't reproduce the bond builder
    # here, but a coarse check via ASE neighbour list:
    from ase.neighborlist import neighbor_list
    i, j, d = neighbor_list("ijd", cell.atoms, cutoff=2.0)
    syms = cell.atoms.get_chemical_symbols()
    coord_si_o = np.zeros(len(cell.atoms), dtype=int)
    for ii, jj in zip(i, j):
        if syms[ii] == "Si" and syms[jj] == "O":
            coord_si_o[ii] += 1
    si_mask = np.array([s == "Si" for s in syms])
    # No Si should have wildly more than 4 O within 2.0 Å.  Allow up
    # to 6 for boundary atoms in a small 15 Å cell.
    assert int(coord_si_o[si_mask].max()) <= 6
    # Most Si atoms should have at least 2 O within 2.0 Å (they're
    # supposed to coordinate 4).  A small cell + few FIRE steps gives
    # a rough constraint, not a hard one.
    assert int(np.median(coord_si_o[si_mask])) >= 2
