"""Tests for ``CoordinationShellTarget.from_atoms`` auto-filter
(``auto_filter_lattice_artifacts``).

The auto-filter zeros out ``coordination_target`` for species pairs
whose ``pair_peak`` is not the smallest in either row or column —
catching second-shell "lattice artefact" pairs in multi-element
crystals (Si-Si in α-quartz, Sr-Sr in SrTiO₃, etc.) that would
otherwise install bond springs at distances much larger than the
actual chemical bond.

Without the auto-filter the FIRE relaxer puts geometrically
incompatible springs on the same atom (Si pulled toward 4 Si at 3.06 Å
AND 4 O at 1.61 Å in SiO₂) and fails to converge.  These tests pin
the filter's behaviour so future changes don't silently regress.
"""
from __future__ import annotations

import numpy as np
import pytest

from atomode.shells import CoordinationShellTarget


def test_single_species_unchanged_si(atoms_si):
    """Single-species crystals should never trigger the auto-filter
    (it activates only when ``num_species >= 2``).  Si-Si is the only
    pair and must remain a real bond regardless of the flag."""
    on = CoordinationShellTarget.from_atoms(
        atoms_si, auto_filter_lattice_artifacts=True,
    )
    off = CoordinationShellTarget.from_atoms(
        atoms_si, auto_filter_lattice_artifacts=False,
    )
    np.testing.assert_array_equal(on.coordination_target, off.coordination_target)
    # Si has 4 nearest neighbours.
    assert int(round(float(on.coordination_target[0, 0]))) == 4


def test_single_species_unchanged_cu(atoms_cu):
    """Cu FCC: 12 nearest neighbours.  Single species → auto-filter
    is a no-op."""
    shell = CoordinationShellTarget.from_atoms(atoms_cu)
    assert int(round(float(shell.coordination_target[0, 0]))) == 12


def test_sio2_filters_lattice_artifacts(atoms_sio2):
    """In α-quartz, only Si-O is a real chemical bond.  Si-Si
    (~3.06 Å) and O-O (~2.64 Å) are lattice separations through a
    bridging atom.  The auto-filter must zero them and keep Si-O."""
    shell = CoordinationShellTarget.from_atoms(
        atoms_sio2, auto_filter_lattice_artifacts=True,
    )
    species_to_idx = {
        sym: i for i, sym in enumerate(shell.species_labels)
    }
    si = species_to_idx["Si"]
    o = species_to_idx["O"]
    # Si-O is real
    assert shell.coordination_target[si, o] > 0
    assert shell.coordination_target[o, si] > 0
    # Si-Si and O-O are lattice artefacts and must be filtered out
    assert shell.coordination_target[si, si] == 0.0
    assert shell.coordination_target[o, o] == 0.0


def test_sio2_off_keeps_artifacts(atoms_sio2):
    """With ``auto_filter_lattice_artifacts=False`` the same SiO₂
    extraction keeps Si-Si and O-O coordination targets non-zero
    (pre-2026-05 behaviour)."""
    shell = CoordinationShellTarget.from_atoms(
        atoms_sio2, auto_filter_lattice_artifacts=False,
    )
    species_to_idx = {
        sym: i for i, sym in enumerate(shell.species_labels)
    }
    si = species_to_idx["Si"]
    o = species_to_idx["O"]
    # All four pairs (Si-O, O-Si, Si-Si, O-O) carry real coordination
    # counts when the filter is disabled.
    assert shell.coordination_target[si, o] > 0
    assert shell.coordination_target[si, si] > 0
    assert shell.coordination_target[o, o] > 0


def test_srtio3_keeps_real_bonds(atoms_srtio3):
    """SrTiO₃: Ti-O (1.96 Å) and Sr-O (2.77 Å) are both real bonds.
    Sr-Sr / Ti-Ti / O-O / Sr-Ti are lattice artefacts.  Auto-filter
    must keep the two real cross-species pairs and zero the rest."""
    shell = CoordinationShellTarget.from_atoms(
        atoms_srtio3, auto_filter_lattice_artifacts=True,
    )
    species_to_idx = {
        sym: i for i, sym in enumerate(shell.species_labels)
    }
    sr = species_to_idx["Sr"]
    ti = species_to_idx["Ti"]
    o = species_to_idx["O"]
    # Real bonds
    assert shell.coordination_target[ti, o] > 0
    assert shell.coordination_target[o, ti] > 0
    assert shell.coordination_target[sr, o] > 0
    assert shell.coordination_target[o, sr] > 0
    # Lattice artefacts
    assert shell.coordination_target[sr, sr] == 0.0
    assert shell.coordination_target[ti, ti] == 0.0
    assert shell.coordination_target[o, o] == 0.0
    assert shell.coordination_target[sr, ti] == 0.0
    assert shell.coordination_target[ti, sr] == 0.0


def test_default_is_filter_on():
    """``auto_filter_lattice_artifacts`` should default to ``True``.
    Verify by introspecting the signature."""
    import inspect

    sig = inspect.signature(CoordinationShellTarget.from_atoms)
    param = sig.parameters["auto_filter_lattice_artifacts"]
    assert param.default is True


def test_pair_peaks_not_clobbered_by_filter(atoms_sio2):
    """The auto-filter zeros ``coordination_target`` only.  The
    distance metadata (``pair_peak``, ``pair_inner``, ``pair_outer``,
    ``pair_hard_min``) is shared by the repulsion machinery and must
    survive untouched, otherwise hard-core repulsion against
    artefact pairs vanishes."""
    on = CoordinationShellTarget.from_atoms(
        atoms_sio2, auto_filter_lattice_artifacts=True,
    )
    off = CoordinationShellTarget.from_atoms(
        atoms_sio2, auto_filter_lattice_artifacts=False,
    )
    np.testing.assert_array_equal(on.pair_peak, off.pair_peak)
    np.testing.assert_array_equal(on.pair_inner, off.pair_inner)
    np.testing.assert_array_equal(on.pair_outer, off.pair_outer)
    np.testing.assert_array_equal(on.pair_hard_min, off.pair_hard_min)
