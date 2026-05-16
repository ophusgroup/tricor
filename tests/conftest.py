"""Shared test fixtures for tricor.

The fixtures here build small ASE reference crystals for the materials
the test suite exercises.  Keeping them out of individual test modules
avoids re-reading the docs ``structures/*.cif`` files (which the test
suite intentionally does not depend on — tests should run from a
fresh clone of just this repo).
"""
from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk


@pytest.fixture
def atoms_cu():
    """FCC copper, single species."""
    return bulk("Cu", "fcc", a=3.615)


@pytest.fixture
def atoms_si():
    """Diamond cubic silicon, single species."""
    return bulk("Si", "diamond", a=5.431)


@pytest.fixture
def atoms_sio2():
    """α-quartz SiO₂.  Hand-built primitive: 3 Si + 6 O.  We don't read
    the CIF here because the test suite shouldn't depend on docs/."""
    a = 4.9134
    c = 5.4052
    cell = np.array([
        [a, 0.0, 0.0],
        [-a / 2, a * np.sqrt(3) / 2, 0.0],
        [0.0, 0.0, c],
    ])
    # α-quartz P3₁21 fractional coordinates (rounded; exact wyckoff
    # values are not needed — the test only checks species counts /
    # auto-filter behaviour, not the precise reference geometry).
    si_frac = np.array([
        [0.4699, 0.0, 1.0 / 3.0],
        [0.0, 0.4699, 2.0 / 3.0],
        [0.5301, 0.5301, 0.0],
    ])
    o_frac = np.array([
        [0.4141, 0.2681, 0.2144],
        [0.7319, 0.1460, 0.5477],
        [0.8540, 0.5859, 0.8810],
        [0.2681, 0.4141, 0.7856],
        [0.1460, 0.7319, 0.4523],
        [0.5859, 0.8540, 0.1190],
    ])
    frac = np.vstack([si_frac, o_frac])
    numbers = [14] * 3 + [8] * 6
    return Atoms(
        numbers=numbers,
        scaled_positions=frac,
        cell=cell,
        pbc=True,
    )


@pytest.fixture
def atoms_srtio3():
    """Cubic SrTiO₃ perovskite (P m -3 m).  Sr at (0,0,0), Ti at the
    body centre, O on the cube faces."""
    a = 3.913
    cell = np.eye(3) * a
    frac = np.array([
        [0.0, 0.0, 0.0],   # Sr
        [0.5, 0.5, 0.5],   # Ti
        [0.5, 0.5, 0.0],   # O
        [0.5, 0.0, 0.5],   # O
        [0.0, 0.5, 0.5],   # O
    ])
    numbers = [38, 22, 8, 8, 8]  # Sr, Ti, O, O, O
    return Atoms(
        numbers=numbers,
        scaled_positions=frac,
        cell=cell,
        pbc=True,
    )


@pytest.fixture
def atoms_graphite():
    """Graphite C, P6₃/mmc."""
    a = 2.467
    c = 6.708
    cell = np.array([
        [a, 0.0, 0.0],
        [-a / 2, a * np.sqrt(3) / 2, 0.0],
        [0.0, 0.0, c],
    ])
    frac = np.array([
        [0.0, 0.0, 0.0],
        [1.0 / 3.0, 2.0 / 3.0, 0.0],
        [0.0, 0.0, 0.5],
        [2.0 / 3.0, 1.0 / 3.0, 0.5],
    ])
    return Atoms(
        numbers=[6] * 4,
        scaled_positions=frac,
        cell=cell,
        pbc=True,
    )


@pytest.fixture
def atoms_diamond():
    """Diamond C, Fd-3m."""
    return bulk("C", "diamond", a=3.567)
