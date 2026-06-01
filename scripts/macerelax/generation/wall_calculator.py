"""Composite calculator that wraps a base calculator (e.g. MACE) and adds
a steep repulsive wall for any atom-pair below a per-element-pair minimum
distance.

Use this to prevent a foundation MLIP from drifting into spurious low-energy
basins where atoms get pulled into near-overlap (e.g. the 50° Si-O-Si /
short Si-Si artifact MACE-MP/MPA shows on disordered SiO₂ — see
MACE_RELAX_PILOT.md §2c).

Wall functional form (purely repulsive, smooth, compact support):

    V_pair(r) = k * (r_min - r) ** exponent      for r < r_min
              = 0                                  for r >= r_min

Force on atom i from neighbour j (PBC-aware):

    F_i = +k * n * (r_min - r) ** (n - 1) * (-r_hat_i->j)

i.e. pushes i directly away from j when r < r_min, zero otherwise.

The wall adds an extra energy term to the trajectory dynamics. If you save
the trajectory and feed it to a learned model, the model will learn the
MACE+wall dynamics, not pure MACE. Document this in the training-data
provenance.
"""
from __future__ import annotations

import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.neighborlist import neighbor_list


class MinDistanceWallCalculator(Calculator):
    """Wrap a base calculator; add a per-pair-type repulsive wall below r_min.

    Vectorized: per-pair lookup uses a dense (Zmax+1, Zmax+1) r_min table,
    force scatter-add uses np.add.at. See MinDistanceWallCalculatorLoop for
    the original pure-Python reference implementation.
    """

    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        base_calc: Calculator,
        r_min_per_pair: dict,
        k: float = 1000.0,
        exponent: int = 4,
        **kwargs,
    ):
        """
        Parameters
        ----------
        base_calc
            Underlying ASE calculator (e.g. MACE). Energy / forces are
            obtained by calling its .calculate() on the same atoms.
        r_min_per_pair
            Dict mapping (Z_i, Z_j) -> minimum allowed pair distance in Å.
            Keys are normalized to sorted tuples internally, so providing
            either (8, 14) or (14, 8) for Si-O works.
        k
            Wall stiffness (eV / Å^exponent). 1000 is a reasonable default
            — gives ~10 eV/Å force at 0.1 Å violation with exponent=4.
        exponent
            Polynomial degree of the wall (≥ 1). 4 gives a steep but
            smooth wall; 2 is harmonic; 1 is linear ramp (discontinuous
            derivative, not recommended).
        """
        super().__init__(**kwargs)
        self.base_calc = base_calc
        self.r_min = {
            tuple(sorted((int(a), int(b)))): float(v)
            for (a, b), v in r_min_per_pair.items()
        }
        self.k_wall = float(k)
        self.exponent = int(exponent)
        if self.exponent < 1:
            raise ValueError("exponent must be >= 1")
        self.cutoff = max(self.r_min.values()) if self.r_min else 0.0
        # Symmetric dense lookup; zero = no wall for that pair.
        if self.r_min:
            zmax = max(max(key) for key in self.r_min)
            table = np.zeros((zmax + 1, zmax + 1), dtype=np.float64)
            for (a, b), v in self.r_min.items():
                table[a, b] = v
                table[b, a] = v
            self._r_min_table = table
            self._zmax = zmax
        else:
            self._r_min_table = None
            self._zmax = -1

    def calculate(
        self,
        atoms=None,
        properties=("energy", "forces"),
        system_changes=all_changes,
    ):
        Calculator.calculate(self, atoms, properties, system_changes)

        # 1. Base calculator: triggers a MACE forward pass when atoms changed.
        self.base_calc.calculate(self.atoms, properties, system_changes)
        E = float(self.base_calc.results["energy"])
        F = np.asarray(self.base_calc.results["forces"], dtype=np.float64).copy()

        # 2. Wall contribution (vectorized).
        E_wall = 0.0
        n_violations = 0
        max_penetration = 0.0
        if self.cutoff > 0.0 and self._r_min_table is not None:
            i_idx, j_idx, D_vec, dist = neighbor_list(
                "ijDd", self.atoms, self.cutoff, self_interaction=False,
            )
            if len(i_idx) > 0:
                z = self.atoms.numbers
                zi = z[i_idx]
                zj = z[j_idx]
                # Species outside the table get r_min=0 → no wall.
                in_table = (zi <= self._zmax) & (zj <= self._zmax)
                r_min_arr = np.zeros(len(i_idx), dtype=np.float64)
                if in_table.any():
                    r_min_arr[in_table] = self._r_min_table[
                        zi[in_table], zj[in_table]
                    ]
                pen = r_min_arr - dist
                viol = pen > 0
                if viol.any():
                    pen_v = pen[viol]
                    i_v = i_idx[viol]
                    D_v = D_vec[viol]
                    dist_v = dist[viol]
                    n_exp = self.exponent
                    kw = self.k_wall
                    V_arr = kw * pen_v ** n_exp
                    f_mag = kw * n_exp * (pen_v ** (n_exp - 1))
                    inv_r = 1.0 / np.maximum(dist_v, 1e-12)
                    # D_vec is r_j - r_i (i→j); -D_vec/r pushes i away from j.
                    force = -(f_mag * inv_r)[:, None] * D_v
                    np.add.at(F, i_v, force)
                    E_wall = 0.5 * float(V_arr.sum())  # undirected pairs counted twice
                    n_violations = int(viol.sum()) // 2
                    max_penetration = float(pen_v.max())

        self.results["energy"] = E + E_wall
        self.results["forces"] = F
        # Provenance for debugging — accessible as atoms.calc.results["wall_*"]
        self.results["wall_energy"] = E_wall
        self.results["wall_n_violations"] = n_violations
        self.results["wall_max_penetration"] = max_penetration


class MinDistanceWallCalculatorLoop(Calculator):
    """Original pure-Python loop implementation. Kept only as the parity
    reference for MinDistanceWallCalculator — do not use in production."""

    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        base_calc: Calculator,
        r_min_per_pair: dict,
        k: float = 1000.0,
        exponent: int = 4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_calc = base_calc
        self.r_min = {
            tuple(sorted((int(a), int(b)))): float(v)
            for (a, b), v in r_min_per_pair.items()
        }
        self.k_wall = float(k)
        self.exponent = int(exponent)
        if self.exponent < 1:
            raise ValueError("exponent must be >= 1")
        self.cutoff = max(self.r_min.values()) if self.r_min else 0.0

    def calculate(
        self,
        atoms=None,
        properties=("energy", "forces"),
        system_changes=all_changes,
    ):
        Calculator.calculate(self, atoms, properties, system_changes)

        self.base_calc.calculate(self.atoms, properties, system_changes)
        E = float(self.base_calc.results["energy"])
        F = np.asarray(self.base_calc.results["forces"], dtype=np.float64).copy()

        E_wall = 0.0
        n_violations = 0
        max_penetration = 0.0
        if self.cutoff > 0.0:
            i_idx, j_idx, D_vec, dist = neighbor_list(
                "ijDd", self.atoms, self.cutoff, self_interaction=False,
            )
            z = self.atoms.numbers
            n_exp = self.exponent
            for n_pair in range(len(i_idx)):
                i = int(i_idx[n_pair])
                j = int(j_idx[n_pair])
                key = tuple(sorted((int(z[i]), int(z[j]))))
                r_min = self.r_min.get(key)
                if r_min is None:
                    continue
                r = float(dist[n_pair])
                if r >= r_min:
                    continue
                pen = r_min - r
                V = self.k_wall * pen ** n_exp
                f_mag = self.k_wall * n_exp * (pen ** (n_exp - 1))
                F[i] -= f_mag * (D_vec[n_pair] / max(r, 1e-12))
                E_wall += V
                n_violations += 1
                if pen > max_penetration:
                    max_penetration = pen
            E_wall *= 0.5

        self.results["energy"] = E + E_wall
        self.results["forces"] = F
        self.results["wall_energy"] = E_wall
        self.results["wall_n_violations"] = n_violations // 2
        self.results["wall_max_penetration"] = max_penetration


def per_pair_min_from_atoms(atoms, margin: float = 0.0,
                              cutoff: float = 5.0) -> dict:
    """Compute observed min pair distance per (Z, Z) from the current atoms.

    Uses ASE's `neighbor_list` cell-list lookup at `cutoff` (default 5 Å)
    instead of an O(N²) all-pairs distance matrix — sub-second at
    N~10k atoms versus 5-15 seconds for the old implementation.

    Cutoff rationale: typical bond_relax-cleaned structures have all
    meaningful pair distances < 4 Å. 5 Å gives margin while keeping the
    neighbor list small. If for some reason no pair of a given (Z, Z)
    appears within cutoff (unusually low density / pair absent), that
    pair is omitted from the output — the wall is silent on it, which
    is correct: there's nothing to push apart.

    Subtracts ``margin`` (Å) from each observed min — set margin = 0 to
    use the observed minimum as the floor (recommended); set margin > 0
    if you want to allow small additional compression.
    """
    from ase.neighborlist import neighbor_list

    i_idx, j_idx, dist = neighbor_list(
        "ijd", atoms, cutoff, self_interaction=False,
    )
    z = np.asarray(atoms.numbers)
    species = sorted(set(int(zi) for zi in z))
    if len(i_idx) == 0:
        return {}
    z_i = z[i_idx]
    z_j = z[j_idx]
    out: dict = {}
    for a in species:
        for b in species:
            if b < a:
                continue
            # neighbor_list returns each pair in both directions; mask both.
            mask = ((z_i == a) & (z_j == b)) | ((z_i == b) & (z_j == a))
            if not mask.any():
                continue
            min_d = float(dist[mask].min())
            out[(a, b)] = max(0.0, min_d - float(margin))
    return out
