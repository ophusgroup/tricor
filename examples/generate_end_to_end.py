#!/usr/bin/env python3
"""Generate an amorphous SiO2 cell end to end, by either available route.

Two independent pathways produce a disordered structure, and both can be
followed by a MACE relaxation under a hard-core wall:

  voronoi   Supercell.generate(grain_size=...) tiles crystalline grains
            into an amorphous matrix and quenches the spring network.
            Geometry-driven: places atoms, then relaxes positions.

  www       tricor.crn builds a cation continuous random network by
            Wooten-Winer-Weaire bond switching under a Keating potential,
            then decorates it with bridging anions.  Topology-driven:
            rewires bonds, so it reaches the fully-coordinated low-strain
            networks geometric packing cannot.

Usage::

    python examples/generate_end_to_end.py voronoi
    python examples/generate_end_to_end.py www
    python examples/generate_end_to_end.py both --mace

``--mace`` adds the MACE + hard-core-wall relaxation and needs
``mace-torch`` installed (``pip install 'tricor[mace]'``); without it the
script still runs and reports the unrelaxed structure.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from ase.io import read, write

import tricor as tc

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
CIF = HERE / "cifs" / "mp-10851_SiO2.cif"

# wall_calculator lives with the generation scripts, not in the package
sys.path.insert(0, str(REPO / "scripts" / "macerelax" / "generation"))


def report(tag: str, atoms) -> None:
    """Minimum pair distance and mean nearest-neighbour distance."""
    d = atoms.get_all_distances(mic=True)
    np.fill_diagonal(d, np.inf)
    print(f"  {tag:<22} N={len(atoms):<5d} "
          f"min={d.min():.3f} A   mean_nn={np.sort(d, axis=1)[:, 0].mean():.3f} A")


def mace_relax(atoms, fmax: float, max_steps: int):
    """FIRE under MACE + a per-pair hard-core wall.  None if MACE absent."""
    try:
        from mace.calculators import mace_mp
    except ImportError:
        print("  [skip] mace-torch not installed — returning unrelaxed cell")
        return None
    from ase.optimize import FIRE
    from wall_calculator import (MinDistanceWallCalculator,
                                 per_pair_min_from_atoms)

    atoms = atoms.copy()
    # Floors measured from the structure itself, minus the default 0.1 A
    # margin, so the wall never fights a contact the seed already has.
    r_min = per_pair_min_from_atoms(atoms)
    atoms.calc = MinDistanceWallCalculator(
        base_calc=mace_mp(default_dtype="float32"), r_min_per_pair=r_min,
    )
    opt = FIRE(atoms, maxstep=0.1, logfile=None)
    opt.run(fmax=fmax, steps=max_steps)
    n = opt.get_number_of_steps()
    f = float(np.linalg.norm(atoms.get_forces(), axis=1).max())
    state = "converged" if f <= fmax else f"HIT THE {max_steps}-STEP CAP"
    print(f"  {n} steps, fmax={f:.4f} eV/A ({state}); "
          f"raise --max-steps to relax fully")
    return atoms


def run_voronoi(cell_dim, grain, use_mace, fmax, max_steps):
    """Voronoi grain-in-matrix packing + spring quench."""
    print("\nvoronoi — grains tiled into an amorphous matrix")
    seed = read(CIF)
    shell = tc.CoordinationShellTarget.from_atoms(seed)
    cell = tc.Supercell.from_atoms(
        seed, (cell_dim,) * 3, rng_seed=42, relative_density=0.92,
    )
    cell.generate(
        shell,
        grain_size=grain,
        crystalline_fraction=0.3,
        # relative_density then describes the AMORPHOUS matrix only and the
        # grains stay at full crystal density; the grains are also held
        # through the quench.
        protect_crystallites=True,
        num_steps=200,
        show_progress=False,
    )
    report("packed + quenched", cell.atoms)
    out = cell.atoms
    if use_mace:
        relaxed = mace_relax(out, fmax, max_steps)
        if relaxed is not None:
            report("after MACE", relaxed)
            out = relaxed
    return out


def run_www(reps, disorder, use_mace, fmax, max_steps):
    """WWW bond switching on the cation network, then anion decoration."""
    from tricor.crn import (Network, build_crn, calibrate_t_melt,
                            decorate_ax2)

    print("\nwww — Wooten-Winer-Weaire continuous random network")
    # Diamond-lattice Si: z=4 everywhere, and from_diamond defaults to
    # Keating(angle_mode="tetrahedral"), which is what the annealer
    # requires -- bond switching is blocked for mixed coordination.
    net = Network.from_diamond(reps)
    print(f"  diamond start: N={net.n}, "
          f"L={float(np.atleast_1d(net.L)[0]):.3f} A, "
          f"E/atom={net.energy_per_atom():.4f} eV")

    # t_max is an ENERGY and its useful range is a material property (a
    # silicon switch costs ~1.3 eV, so acceptance is 0% at 0.25 eV).  So
    # measure this network's own melting scale and drive the anneal with
    # the DIMENSIONLESS ratio, which is what transfers across chemistries.
    t_melt = calibrate_t_melt(net, rng=np.random.default_rng(0))
    t_max = disorder * t_melt
    print(f"  t_melt={t_melt:.3f} eV -> t_max={t_max:.3f} eV "
          f"(disorder={disorder})")

    ann, log = build_crn(net, t_max=t_max, rng=np.random.default_rng(1))
    st = ann.stats
    z = (ann.net.nbrs >= 0).sum(axis=1)
    print(f"  {st.accepted}/{st.attempted} switches accepted "
          f"({100 * st.accepted / max(st.attempted, 1):.1f}%), "
          f"E/atom={ann.net.energy_per_atom():.4f} eV, "
          f"z=4 for {100 * np.mean(z == 4):.1f}% of atoms")
    if st.accepted == 0:
        print("  WARNING: no switches accepted — still the crystal. "
              "Raise --disorder.")
    report("Si CRN (cation only)", ann.net.to_atoms())
    # decorate_ax2 rescales the Si sublattice so a bridging O on each
    # Si-Si bond gives d(Si-O)=1.61 A at 144 deg.  Expect a warning that
    # some bonds exceed 2*d_ax: a CRN at this disorder carries a ~5%
    # bond-length spread, so ~13% of bridges start out linear (180 deg).
    # That is intrinsic to the construction, not a failure -- it is what
    # the MACE relaxation below is for.  Annealing the Si network at the
    # silica Si...Si distance instead does not help; the relative spread,
    # and therefore the fraction affected, is unchanged.
    atoms = decorate_ax2(ann.net).to_atoms()      # add the bridging O
    report("decorated to SiO2", atoms)
    if use_mace:
        relaxed = mace_relax(atoms, fmax, max_steps)
        if relaxed is not None:
            report("after MACE", relaxed)
            atoms = relaxed
    return atoms


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("pathway", choices=("voronoi", "www", "both"))
    ap.add_argument("--mace", action="store_true",
                    help="also run the MACE + hard-core-wall relaxation")
    ap.add_argument("--cell-dim", type=float, default=24.0,
                    help="voronoi cell edge in A (default 24)")
    ap.add_argument("--grain", type=float, default=10.0,
                    help="voronoi grain diameter in A (default 10)")
    ap.add_argument("--www-reps", type=int, default=3,
                    help="diamond cell repeats for WWW; 8*reps^3 cations "
                         "(default 3 -> 216)")
    ap.add_argument("--disorder", type=float, default=1.0,
                    help="WWW dial: t_max as a multiple of the measured "
                         "t_melt.  <0.8 barely switches, >1.5 starts to "
                         "collapse geometry (default 1.0)")
    ap.add_argument("--max-steps", type=int, default=200,
                    help="FIRE step ceiling for the MACE relaxation "
                         "(default 200; raise it to reach fmax)")
    ap.add_argument("--fmax", type=float, default=0.05,
                    help="MACE relaxation force target, eV/A (default 0.05)")
    ap.add_argument("--out", type=Path, default=None,
                    help="write the final structure(s) to this .xyz")
    args = ap.parse_args()

    results = {}
    if args.pathway in ("voronoi", "both"):
        results["voronoi"] = run_voronoi(args.cell_dim, args.grain, args.mace, args.fmax, args.max_steps)
    if args.pathway in ("www", "both"):
        results["www"] = run_www(args.www_reps, args.disorder, args.mace, args.fmax, args.max_steps)

    if args.out is not None:
        for name, atoms in results.items():
            path = args.out.with_name(f"{args.out.stem}_{name}{args.out.suffix}")
            write(path, atoms)
            print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
