"""CRN build protocol: hot randomisation -> staged cooling -> quench.

Phase-2 lesson learned the hard way (see scratch/phase2_crn/tunability.json):
applying bond switches *unconditionally* from the crystal and then annealing
briefly reproduces the very pathology the Voronoi mosaic has -- 22%三-rings,
30 deg bond-angle spread.  Bond switching is only half the method; the network
must be equilibrated, or a topologically-perfect network is still a
geometrically terrible one.

The correct protocol, following Wooten-Winer-Weaire and the annealing schedule
of Barkema & Mousseau (PRB 62, 4985):

  1. RANDOMISE hot.  Metropolis at T_hot (~2 eV).  From a perfect crystal a
     single transposition costs ~1.3 eV, so acceptance is 0% at 0.25 eV and
     ~6% at 2 eV -- randomisation genuinely requires a hot bath, it is not an
     artefact.  The number of ACCEPTED switches per atom is the disorder dial.
  2. COOL in stages through T = 1.0, 0.5, 0.35, 0.25 eV, each with a fixed
     attempts-per-atom budget, so the network equilibrates as it orders.
  3. QUENCH with FIRE to the local minimum.

DISORDER IS NOT SET BY PARTIAL MELTING -- measured, 2026-08-18.
Cooling to T~0 anneals partial disorder back OUT.  At t_max/T_melt = 0.5 the
run accepted 226 switches and returned PERFECT DIAMOND (ring6 = 1.000,
angle spread 0.00, E/atom = 0).  That is correct physics, not a bug: for
Keating silicon the crystal is the unique global minimum, so any disorder that
is still individually reversible gets undone on the way down.

Consequence for the pipeline: a continuous "how melted" dial is not a robust
way to get intermediate order.  The robust dial is LOCKED CRYSTALLINE GRAINS,
which are topologically protected -- the MC can never switch a grain-internal
bond, so the grain cannot anneal away no matter how long the matrix is
annealed.  The intended design is therefore:

    matrix = always a fully-annealed CRN (deliberately NOT tunable)
    dial   = grain size and volume fraction (Nakhmanson et al.)

which maps directly onto tricor's existing `grain_size` and
`crystalline_fraction` conditioning fields -- the latter being a dead feature
today (always 1.0), which this would bring to life as a real, measured knob.


FULLY LOCAL, per Hemmann et al. (Adv. Funct. Mater. 2026), who drop the global
relaxation step of Barkema-Mousseau/Vink "to achieve evolution independent of
system size".  No relaxation in this module touches more than one cluster:

  * every attempted switch relaxes only a ~150-atom patch, remapped into a
    compact local index space (build_patch), so nothing is ever allocated or
    scattered over N;
  * the soft non-bonded repulsion that keeps Keating from letting unbonded
    atoms collapse runs INSIDE that patch (patch_nonbonded), not globally;
  * there are no periodic global quenches.  An earlier version quenched every
    200 accepted switches, which is O(N) per call -- at 500k atoms and ~1
    switch/atom that is ~2500 whole-system relaxations and would have dominated
    the entire build.

A single terminal global quench remains (final_global_quench), because it is
one O(N) pass amortised over the whole build rather than an O(N) term per move.
Set it False for a strictly local build.

ARCHITECTURAL BOUNDARY -- the WWW stage and the MACE stage MUST stay separate.
Nothing in this module, or in www.py, ever calls MACE.  Two independent reasons:

  1. Correctness.  Metropolis acceptance is only valid if every energy entering
     the accept/reject test comes from ONE potential.  Relaxing on the MACE PES
     inside the MC loop would mean proposals are generated under Keating and
     judged under MACE; detailed balance breaks and the sampled ensemble stops
     meaning anything.  The MC loop is Keating-only, always.
  2. They would undo each other.  WWW changes topology at fixed geometry; MACE
     changes geometry at fixed topology.  Interleaving re-relaxes a structure
     whose bonds are about to move again -- pure waste -- and, measured, it
     actively hurts: handing MACE a network with 1.6 A non-bonded contacts made
     it displace atoms far enough to drop geometric 4-coordination from 0.949
     to 0.920 (scratch/phase2_crn/www_plus_mace.json).

The correct composition is strictly sequential and one-way:

     WWW (topology + Keating/repulsion geometry)  ->  [finished network]
                                                  ->  MACE teacher (geometry)
                                                  ->  student / production

The soft non-bonded repulsion added to global_quench() exists precisely so this
module can hand over a geometrically clean network, leaving MACE with little to
do -- which is the regime where Phase-1 Control B showed MACE *preserves* a good
CRN (angle spread 12.69 -> 11.89 deg) rather than damaging it.

Triangles are forbidden throughout (`forbid_triangles`), matching the ~0.3%
3-ring content of real a-Si; without that ban the hot phase manufactures
strained triangles that the cooling schedule cannot undo.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, replace

import numpy as np

from .www import (Network, WWWAnnealer, bond_hops,
                  patch_energy_forces)


@dataclass
class Schedule:
    """Triangular temperature profile: heat to t_max, cool to ~0, quench.

    t_max is the DISORDER DIAL (Hemmann et al. AFM 2026 use T_max/T_melt the
    same way).  An earlier design used "accepted switches at a fixed hot T" as
    the dial and cooled only to 0.25 eV; that failed, because 0.25 eV is still
    hot enough to accept ~4% of moves, so the cooling stages accepted far more
    switches than the dial ever did and every dose converged to the same
    structure.  Cooling must reach a temperature where only energy-LOWERING
    switches survive, otherwise the anneal keeps randomising instead of
    repairing.
    """
    t_max: float = 2.0
    n_cool_stages: int = 5
    t_final: float = 0.02          # effectively T=0: downhill moves only
    attempts_per_atom_per_stage: float = 30.0
    final_global_quench: bool = True   # ONE terminal O(N) pass, not per-move

    def ladder(self) -> list:
        """Geometric cooling ladder from t_max down to t_final."""
        return list(np.geomspace(self.t_max, self.t_final,
                                 self.n_cool_stages + 1))[1:]


@dataclass
class BuildLog:
    events: list = field(default_factory=list)

    def add(self, **kw):
        self.events.append(kw)
        return kw


def build_crn(net: Network, t_max: float | None = None,
              sched: Schedule | None = None,
              frozen_atoms: np.ndarray | None = None,
              rng: np.random.Generator | None = None,
              on_event=None, forbid_triangles: bool = False,
              repair_first: bool = True,
              structured_quench: bool = True) -> tuple[WWWAnnealer, BuildLog]:
    """Heat to t_max, cool to ~0, quench.  t_max is the disorder dial.

    Returns the annealer (whose .stats carries the accepted-switch count, the
    *measured* disorder) and a per-phase log.
    """
    sched = sched or Schedule()
    if t_max is not None:
        sched = replace(sched, t_max=t_max)
    log = BuildLog()
    # forbid_triangles defaults OFF: the ban is chemistry-specific -- its
    # bipartite form forbids EDGE-SHARING, which is exactly how rutile TiO2 and
    # corundum Al2O3 are built, and it rejected 537 of 600 TiO2 proposals.  With
    # the T=0 structured quench in place a strained motif goes away because
    # removing it lowers the energy, not because it was forbidden.
    ann = WWWAnnealer(net, T=sched.t_max, frozen_atoms=frozen_atoms,
                      forbid_triangles=forbid_triangles, rng=rng)
    n_free = int((~ann.frozen).sum())
    if repair_first:
        # Barkema & Mousseau's first-quench repair of close-but-unbonded pairs.
        # The freshly built network really does carry the artefact they
        # describe -- 28-37 pairs inside the bond length on a 216-atom cell.
        ann.global_quench(iters=20000, fmax_stop=1e-6)
        n_fix = ann.repair_close_pairs()
        ev = log.add(phase="repair", rewires=int(n_fix),
                     e_per_atom=net.energy_per_atom())
        if on_event:
            on_event(ev)
    budget = int(sched.attempts_per_atom_per_stage * n_free)

    for phase, T in [("randomise", sched.t_max)] + \
                    [("cool", t) for t in sched.ladder()]:
        t0 = time.perf_counter()
        a0, t_0 = ann.stats.accepted, ann.stats.attempted
        ann.T = T
        while (ann.stats.attempted - t_0) < budget:
            ann.step()
        ev = log.add(phase=phase, T=round(T, 4),
                     accepted=ann.stats.accepted - a0,
                     attempted=ann.stats.attempted - t_0,
                     e_per_atom=net.energy_per_atom(),
                     seconds=round(time.perf_counter() - t0, 1))
        if on_event:
            on_event(ev)

    if sched.final_global_quench:
        ann.global_quench(iters=2000, fmax_stop=1e-3)
    if structured_quench:
        # The step Phase 1 identified as the missing one.  A geometric quench
        # relaxes COORDINATES at fixed topology and cannot remove a strained
        # motif; only accepting energy-lowering transpositions can, and it is
        # what takes the angle spread from ~29 deg to the published ~12-13 deg.
        n_sw = ann.quench_topology_exact()
        ev = log.add(phase="structured_quench", switches=int(n_sw),
                     exhausted=bool(ann.quench_exhausted),
                     e_per_atom=net.energy_per_atom())
        if on_event:
            on_event(ev)
    ev = log.add(phase="quench", e_per_atom=net.energy_per_atom(),
                 total_accepted=ann.stats.accepted,
                 switches_per_atom=ann.stats.accepted / max(1, n_free))
    if on_event:
        on_event(ev)
    return ann, log


def clone(net: Network) -> Network:
    # species MUST be carried: without it the copy defaults to all-one-element,
    # move="auto" then picks the WWW move, and calibrate_t_melt calibrates a
    # move the real run will never use (and de-bipartitises the probe copy).
    return Network(net.pos.copy(), net.nbrs.copy(), net.L, net.pot,
                   net.species.copy())


def calibrate_t_melt(net: Network, rng=None, target_accept: float = 0.02,
                     t_lo: float = 0.1, t_hi: float = 16.0,
                     n_attempts: int = 500, iters: int = 7,
                     frozen_atoms=None, verbose: bool = False) -> float:
    """Find the temperature at which switch acceptance reaches target_accept.

    WHY THIS EXISTS.  t_max is an ENERGY, so its useful range is set by what a
    bond switch costs, which is a material property: it scales with the Keating
    stiffness (alpha, beta) and the equilibrium bond length.  For silicon a
    switch costs ~1.3 eV, so acceptance is exactly 0% at 0.25 eV and only ~6%
    at 2 eV -- a t_max that randomises silicon would do nothing to a stiffer
    network and would vaporise a softer one.  Hand-tuning t_max per chemistry
    is a non-starter for a 509-composition corpus.

    So we do what Hemmann et al. (Adv. Funct. Mater. 2026) do: measure the
    material's own melting scale once, cheaply, and express the user-facing
    dial as the DIMENSIONLESS ratio t_max / t_melt.  That ratio is what
    transfers across chemistries; the eV value never has to be exposed.

    Bisection on log T, on a throwaway copy so the caller's network is never
    disturbed.  ~500 attempts per probe x 7 iterations is seconds.
    """
    rng = rng or np.random.default_rng(0)
    lo, hi = np.log(t_lo), np.log(t_hi)
    best = None
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        T = float(np.exp(mid))
        probe = WWWAnnealer(clone(net), T=T, frozen_atoms=frozen_atoms,
                            forbid_triangles=True,
                            rng=np.random.default_rng(rng.integers(1 << 30)))
        for _ in range(n_attempts):
            probe.step()
        acc = probe.stats.accepted / max(1, probe.stats.attempted)
        if verbose:
            print(f"    calibrate: T={T:.3f} accept={acc*100:.2f}%",
                  flush=True)
        best = T
        if acc < target_accept:
            lo = mid
        else:
            hi = mid
    return best

def calibrate_t_melt_hemmann(net: Network, rng=None, p_melt: float = 0.001,
                             n_sample: int = 4000, frozen_atoms=None,
                             quantile: float = 0.0, verbose: bool = False):
    """T_melt exactly as Hemmann et al. (Adv. Funct. Mater. 2600037) define it.

        "T_melt as the temperature at which the energetically lowest bond switch
         and relaxation are accepted with probability P_accept > P_melt := 0.1%"
        isolated from the Metropolis factor:  T_melt = (E_f - E_i)/ln(P_melt)

    Note what this presupposes: it is only meaningful on a network where every
    switch is UPHILL, i.e. after a T=0 structured quench.  On such a network the
    smallest uphill cost dE_min sets the scale, and

        T_melt = dE_min / (-ln p_melt)

    (the sign follows from P = exp(-dE/T), so T = dE / (-ln P); the paper's
    expression is the same quantity written with ln of a probability < 1).

    This is a DIFFERENT and much colder quantity than a bisection on global
    acceptance rate, which is what an earlier version of this module used.  That
    version targeted 2% acceptance over ALL proposals, giving a temperature deep
    into the melt regime -- which is why "t_max = T_melt" there melted networks
    into self-intersection rather than gently disordering them, and why the
    conclusion that the temperature profile is not a usable disorder dial was
    drawn from a test that never reproduced the published protocol.

    quantile > 0 uses that quantile of the uphill costs instead of the strict
    minimum, which is more robust to a single anomalously cheap switch on a
    small sample; quantile=0 is the paper's literal definition.

    IMPORTANT -- WHICH NETWORK TO CALIBRATE ON.  Hemmann compute this for the
    initial CRYSTALLINE lattice and use it "as a reference for rescaling
    temperatures".  It must not be computed on a disordered network: there the
    switch costs form a near-continuous distribution reaching down to zero, so
    the strict minimum shrinks with sample size and T_melt collapses.  Measured:
    quenched diamond gives 0.616 eV, while a quenched CRN of the same material
    gives 0.025 eV, which degenerates the whole triangular profile to a single
    stage.  Calibrate on the reference crystal, then apply the profile to
    whatever network is being annealed.
    """
    rng = rng or np.random.default_rng(0)
    probe = WWWAnnealer(clone(net), T=1e-9, frozen_atoms=frozen_atoms,
                        rng=np.random.default_rng(rng.integers(1 << 30)))
    costs = []
    for _ in range(n_sample):
        pick = (probe.propose_bipartite() if probe.move == "bipartite"
                else probe.propose())
        if pick is None:
            continue
        A, B, C, D = pick
        moving = bond_hops(probe.net, np.array([A, B, C, D]), probe.n_shell)
        g0, p0, b0, a0, _, nb0 = probe.patch_with_scenery(moving)
        # ka_ang MUST be passed here.  e_f comes back from _relax_local, which
        # scores with the valency-scaled angular stiffness; omitting it here
        # scored e_i under the FLAT stiffness, so with beta_by_valency=True the
        # switch cost e_f - e_i was a difference between two different energy
        # functionals and T_melt was calibrated on it.
        e_i, _ = patch_energy_forces(p0, b0, a0, probe.net.L, probe.net.pot,
                                     want_forces=False, nb_pairs=nb0,
                                     ka_ang=probe.ka_for(g0, a0),
                                     # off_ang matters for the same reason
                                     # ka_ang does: _relax_local scores e_f with
                                     # the per-species equilibrium angle, so
                                     # scoring e_i with the scalar offset makes
                                     # e_f - e_i a difference between two
                                     # different functionals -- and every move
                                     # then looks downhill, so the calibration
                                     # finds no uphill switch at all.
                                     off_ang=probe.off_for(g0, a0),
                                     r_rep=probe.r_rep, k_rep=probe.k_rep,
                                     r_core=probe.r_core, k_core=probe.k_core)
        (probe._apply_bipartite if probe.move == "bipartite"
         else probe._apply)(A, B, C, D)
        e_f, (movable, backup), _ = probe._relax_local(moving, None)
        probe.net.pos[movable] = backup
        (probe._revert_bipartite if probe.move == "bipartite"
         else probe._revert)(A, B, C, D)
        if e_f > e_i:
            costs.append(e_f - e_i)
    if not costs:
        raise RuntimeError("no uphill switches sampled; quench the network first")
    costs = np.sort(np.asarray(costs))
    de = float(costs[0] if quantile <= 0 else
               np.quantile(costs, quantile))
    t_melt = de / (-np.log(p_melt))
    if verbose:
        print(f"    {len(costs)} uphill switches sampled; dE min={costs[0]:.4f} "
              f"median={np.median(costs):.4f} -> T_melt={t_melt:.4f} eV")
    return t_melt


def triangular_profile(t_max: float, delta_t: float):
    """Hemmann's temperature profile: heat at +delta_t to t_max, cool at
    -delta_t to 0, then a T=0 quench.  Returns the temperature ladder."""
    up = list(np.arange(delta_t, t_max + 1e-12, delta_t))
    if not up or up[-1] < t_max:
        up.append(t_max)
    down = list(np.arange(t_max - delta_t, 0.0, -delta_t))
    return up + down
