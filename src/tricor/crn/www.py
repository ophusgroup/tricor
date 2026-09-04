"""Wooten-Winer-Weaire bond-switching generator for continuous random networks.

Phase 2 of the MRO investigation (see PHASE1_MRO_RESULTS.md).  Phase 1 showed
the Voronoi-crystallite packing cannot build a valid tetrahedral network at any
grain size, while the downstream machinery (bond_relax, MACE teacher) preserves
a good network intact.  So the fix belongs at the seeding stage, and this is it.

Why bond switching rather than better packing: a WWW transposition **preserves
coordination exactly by construction** -- every one of the four participating
atoms keeps its coordination number.  Starting from a perfect diamond network,
the topology is 100% four-coordinated at every step, forever.  That alone fixes
the single largest Phase-1 failure (33% four-coordinated).  Disorder is then
introduced purely as *topological* randomness, which is what a CRN is.

Method (Wooten, Winer & Weaire PRL 54, 1392 (1985), with the accelerations of
Barkema & Mousseau PRB 62, 4985 (2000) and Vink et al. PRB 64, 245214 (2001)):

  * Configuration = coordinates + an explicit 4-neighbour list.  Atoms interact
    ONLY through that list, so no neighbour search ever happens.
  * Energy = Keating (alpha = 2.965 eV/A^2, beta = 0.285*alpha, d = 2.35 A).
  * Move = bond transposition: for a bond A-B, pick C in N(A)\\{B} and
    D in N(B)\\{A}; break A-C and B-D, form A-D and B-C.
  * Acceptance = Metropolis on the *relaxed* energy.
  * Accelerations, all three of them:
      (1) LOCAL relaxation.  Only atoms within `n_shell` bond hops of the four
          switch atoms move; the energy terms touching them are gathered in
          O(cluster), never O(N).  Cost per attempted move is independent of
          system size.
      (2) A-PRIORI THRESHOLD with harmonic early rejection.  E_t is drawn
          BEFORE relaxing; the relaxation aborts as soon as E - c|F|^2 > E_t
          shows the threshold is unreachable.  ~99% of moves are rejected in a
          well-relaxed network, so this is where most of the speed comes from.
      (3) No per-accept global relaxation.  Global quenches are batched
          (`quench_every`) instead of run after every accepted move, which is
          the O(N) term that would otherwise dominate at large N.

Tunable disorder: the knob is the number of ACCEPTED switches per atom (report-
ed as `switches_per_atom`), equivalently the anneal temperature.  Starting from
the crystal and switching *outward* is far cheaper than the random-start deep
anneals of the WWW papers, which exist to erase crystalline memory -- the
opposite of what a tunable-disorder pipeline wants.

Locked grains (Nakhmanson et al. PRB 63, 235207 (2001)) are supported via
`frozen_bonds`: bonds inside a designated grain are never proposed for
switching, so the grain stays topologically crystalline while the surrounding
matrix randomises.  This is the discrete MRO dial and costs nothing at runtime.

Pure numpy; the hot kernels are written to be numba-jittable later if needed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Keating parameters for silicon (Barkema & Mousseau PRB 62, 4985).
ALPHA_SI = 2.965          # eV / A^2, bond stretching
BETA_FRAC = 0.285         # beta / alpha, bond bending
D_SI = 2.35               # A, equilibrium bond length

ANGLE_PAIRS = np.array([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
                       dtype=np.int64)


@dataclass(frozen=True)
class Keating:
    """Keating-type strain energy.

    angle_mode selects the bond-bending equilibrium:

      "tetrahedral"    cos(theta_eq) = -1/3, the classical Keating form.  Only
                       meaningful for Z = 4.
      "max_repulsion"  cos(theta_eq) = -1 for EVERY coordination number, i.e.
                       Hemmann et al. (Adv. Funct. Mater. 2026) eq. for
                       arbitrary Z: bonds simply repel each other and spread
                       over the sphere.  For Z = 4 this DERIVES 109.47 deg
                       rather than imposing it (verified).  Still the dataclass
                       default for backward compatibility, but see below.

    POLICY, and it overrides the "more general is better" reading above:
    WWWAnnealer now REFUSES any angle_mode except "tetrahedral" unless passed
    allow_multivalent=True.  Every non-tetrahedral mode -- max_repulsion,
    ideal_per_z and crystal alike -- collapses geometrically in the switching
    MC, because Keating constrains only bonded terms and the excluded-volume
    term that would hold unbonded atoms apart is unusable there (k_rep = 0
    gives no repulsion; k_rep > 0 miscounts scenery bonds as overlaps).  This
    is NOT specific to max_repulsion: the measured failure below used
    angle_mode="crystal" with crystal-measured theta_0, the best-configured
    non-tetrahedral option.  See the block comment in WWWAnnealer.__init__ for
    the numbers.  Mixed-coordination materials go through the MOSAIC pathway.

    Caveat, measured: the angular term alone is DEGENERATE for Z >= 6 -- an
    octahedron and a 17-degree-min-angle arrangement have identical energy.
    The degeneracy is lifted by the non-bonded repulsion (r_rep), whose cutoff
    must be the ideal neighbour-neighbour separation for that Z, i.e. the
    reference crystal's second-shell distance.  With that, Z = 3..12 all
    recover their ideal geometry.
    """
    alpha: float = ALPHA_SI
    beta_frac: float = BETA_FRAC
    d: float = D_SI
    angle_mode: str = "max_repulsion"
    beta_by_valency: bool = False

    @property
    def d2(self) -> float:
        return self.d * self.d

    @property
    def kb(self) -> float:
        """Bond-stretch prefactor, (3/16) * alpha / d^2."""
        return 3.0 / 16.0 * self.alpha / self.d2

    @property
    def ka(self) -> float:
        """Bond-bend prefactor, (3/8) * beta * alpha / d^2.

        Hemmann et al. (arXiv:2601.10333) write BOTH conventions with the same
        prefactor -- their Eq. in Sec. 2.1 is

            E = (3/16) sum (r_ij^2 - 1)^2 + (3/8) beta sum (r_ij.r_ik + 1/3)^2

        and Sec. 2.2 replaces only the constant inside the bracket, +1/3 -> +1,
        leaving (3/8) beta untouched.  Both therefore give ka/kb = 2*beta.

        A 2/3 rescale used to be applied here for max_repulsion, on the
        reasoning that the two conventions have different curvature about the
        same geometry so beta should denote a constant PHYSICAL stiffness.  That
        is defensible in isolation but it silently redefines beta relative to
        the paper: it made ka/kb = 4*beta/3, so our beta was 2/3 of theirs and
        no beta value could be compared with their swept range [0, 10].  Since
        beta is THEIR disorder knob, matching their definition matters more than
        making it convention-independent.
        """
        return 3.0 / 8.0 * (self.beta_frac * self.alpha) / self.d2

    def ka_for_z(self, z: np.ndarray) -> np.ndarray:
        """Per-angle bend prefactor, optionally scaled by the CENTRE's valency.

        Hemmann et al.: "For polyvalent networks, one could introduce a
        valency-dependent beta to accommodate the Z-dependence of energy."  The
        angular term sums over C(Z,2) pairs per atom, so at fixed beta a
        six-coordinate atom carries 15 pairs against a tetrahedral atom's 6.
        Measured at the ideal geometry the angular energy runs 0, 12, 42.7, 192
        for Z = 2, 3, 4, 6 -- a 4.5x jump from tetrahedral to octahedral, which
        inflates every dE around a high-Z atom and is why the T=0 quench found
        no downhill move at all for TiO2, Al2O3 and TiN.

        Scaling by C(4,2)/C(Z,2) makes the per-atom angular energy scale
        comparable across valency, normalised to tetrahedral:
            Z=2 -> 6.0,  Z=3 -> 2.0,  Z=4 -> 1.0,  Z=6 -> 0.4,  Z=8 -> 0.214
        """
        if not self.beta_by_valency:
            return np.full(len(z), self.ka)
        zc = np.maximum(z, 2)
        pairs = zc * (zc - 1) / 2.0
        return self.ka * (6.0 / pairs)

    @property
    def angle_offset(self) -> float:
        """Constant c in (r_ij . r_ik + c)^2, i.e. -d^2 cos(theta_eq).

        c = d^2/3   tetrahedral   (theta_eq = 109.47 deg, Keating / BM)
        c = d^2     max_repulsion (theta_eq = 180 deg, Hemmann, any Z)

        For "ideal_per_z" the offset is per-atom and comes from ideal_offset();
        this scalar is then only a fallback for code paths without valency.
        """
        if self.angle_mode == "tetrahedral":
            return self.d2 / 3.0
        if self.angle_mode in ("max_repulsion", "ideal_per_z", "crystal"):
            return self.d2
        raise ValueError(f"unknown angle_mode {self.angle_mode!r}")

    # Per-species equilibrium cos(theta), MEASURED from the reference crystal.
    # The general Keating form in the literature is
    #     (3 beta / 16 d_ij d_ik) (r_ij . r_ik - d_ij d_ik cos theta_0)^2
    # i.e. the offset is -d^2 cos(theta_0) with theta_0 the MATERIAL's own
    # equilibrium angle (Lepkowski & Gorczyca, arXiv:1002.0437, Eq. 1).  So a
    # per-structure equilibrium angle is the standard form, not an extension.
    # Taking theta_0 from the reference crystal removes the need to guess it
    # from a table of coordination numbers -- an earlier attempt did that and
    # invented a value for Z=5 with no justification.
    cos_theta0: tuple = ()          # (species, cos_theta0) pairs

    # cos of the ideal angle for each coordination, used ONLY as a fallback
    # when no crystal reference is supplied.
    IDEAL_COS = {1: -1.0, 2: -1.0, 3: -0.5, 4: -1.0 / 3.0,
                 5: -0.25, 6: 0.0}

    def species_offset(self, species: np.ndarray) -> np.ndarray:
        """Per-atom offset -d^2 cos(theta_0) from the crystal-measured angles."""
        table = dict(self.cos_theta0)
        return np.array([-self.d2 * table[int(sp)] for sp in species])

    def ideal_offset(self, z: np.ndarray) -> np.ndarray:
        """Per-atom offset -d^2 cos(theta_eq(Z)).

        Hemmann set theta_eq = 180 deg for EVERY valency, arguing that a
        4-coordinated vertex still relaxes to tetrahedral geometry through
        frustration.  Topologically it does, but the price is measurable: the
        term is minimised at r^2 cos(theta) = -d^2, so at tetrahedral geometry
        it wants r = sqrt(3) d, and a silicon network came out with <r> = 2.47 A
        against 2.357, atomic overlaps down to 0.48 A, and a MACE relaxation
        that then moved atoms 0.75 A repairing it.  With theta_eq set from the
        coordination instead, the same network keeps <r> = 2.35 A and MACE moves
        atoms 0.13 A.

        This reduces EXACTLY to their form at Z = 2 (where 180 deg is
        achievable) and to Keating at Z = 4, so it is a generalisation of both
        rather than a third convention.
        """
        zc = np.clip(np.asarray(z, dtype=np.int64), 1, 6)
        cos_eq = np.array([self.IDEAL_COS[int(v)] for v in zc])
        return -self.d2 * cos_eq


def mic(dv: np.ndarray, L: float) -> np.ndarray:
    """Minimum image for a cubic cell."""
    # L may be a scalar (cubic) or a length-3 array (orthorhombic); the
    # arithmetic broadcasts either way, which is what lets the reference
    # lattices read from CIFs -- tetragonal, hexagonal-turned-orthorhombic --
    # use the same code path as a cubic cell.
    return dv - L * np.round(dv / L)


# ── network container ────────────────────────────────────────────────────────


class Network:
    """Coordinates + a fixed-coordination neighbour list in a cubic cell.

    `nbrs` is (N, Zmax) padded with -1, so MIXED VALENCY is supported: SiO2 is
    Si with Z=4 and O with Z=2 in one network, Si3N4 is 4/3, TiO2 is 6/3.  A
    bond switch conserves every atom's Z exactly, so whatever coordination
    statistics the initial network carries are preserved for all time -- which
    is how Hemmann et al. control composition.

    `species` is an optional (N,) array of atomic numbers, used only for
    reporting and for writing ASE objects.
    """

    def __init__(self, pos: np.ndarray, nbrs: np.ndarray, L,
                 pot: Keating | None = None,
                 species: np.ndarray | None = None):
        self.pos = np.ascontiguousarray(pos, dtype=np.float64)
        self.nbrs = np.ascontiguousarray(nbrs, dtype=np.int64)
        # Orthorhombic cell as a length-3 array.  A scalar is accepted and
        # broadcast, so every cubic call site is unchanged; `mic`, cKDTree's
        # boxsize and coordinate scaling all take the vector form directly.
        Lv = np.asarray(L, dtype=np.float64).reshape(-1)
        self.L = np.repeat(Lv, 3) if Lv.size == 1 else Lv
        if self.L.size != 3 or not np.all(self.L > 0):
            raise ValueError(f"cell must be 3 positive lengths, got {L!r}")
        self.pot = pot or Keating()
        self.species = (np.full(len(self.pos), 14, dtype=np.int64)
                        if species is None
                        else np.asarray(species, dtype=np.int64))
        assert self.nbrs.ndim == 2 and len(self.nbrs) == len(self.pos)
        self.z = (self.nbrs >= 0).sum(axis=1).astype(np.int64)

    @property
    def volume(self) -> float:
        return float(np.prod(self.L))

    def suggest_r_rep(self, frac: float = 0.5) -> float:
        """Non-bonded repulsion cutoff derived from THIS network's own shells.

        Must be material-specific.  The cutoff has to sit above the bond length
        (so bonded pairs are not fighting the springs) and below the nearest
        genuine second-neighbour distance (so real second neighbours are not
        penalised).  Those two distances differ a lot between chemistries:

            Si   (d = 2.35): bonds 2.35, nearest non-bonded 3.84 -> r_rep 3.10
            AX2  (d = 1.61): bonds 1.61, nearest non-bonded 2.63 -> r_rep 2.12

        A single hard-coded value cannot serve both -- measured, r_rep = 3.0 on
        the AX2 network wrongly penalises 1296 legitimate second-neighbour
        pairs.

        frac stays at 0.5 (the midpoint) deliberately.  A verification pass
        suggested pushing it to ~1.0 so the cutoff reaches the ideal
        neighbour-neighbour separation, which is what breaks the angular
        degeneracy at Z >= 6 for an ISOLATED vertex.  But those two distances
        coincide in a tetrahedral network (2*d*sin(109.47/2) = 3.84 = the
        second shell), and in a DISORDERED network the second-neighbour
        distances are a broad distribution, not a delta -- a cutoff at 0.9 of
        the way lands inside its lower tail and penalises real second
        neighbours.  The midpoint prevents overlap, which is what this term is
        for, without fighting the second shell.  So derive it, exactly as T_melt is derived, and the user never
        sets it per chemistry.  In the pipeline this is available directly from
        tricor's CoordinationShellTarget (first and second shell of the
        reference crystal).
        """
        from scipy.spatial import cKDTree
        L = self.L
        pos = np.mod(self.pos, L)
        # np.mod(-1e-18, L) can return exactly L; clamp it back inside.
        pos = np.where(pos >= L, pos - L, pos)
        rmax = min(0.49 * float(np.min(L)), 6.0)
        pairs = cKDTree(pos, boxsize=L).query_pairs(rmax, output_type="ndarray")
        if len(pairs) == 0:
            return 1.3 * self.pot.d
        d = np.linalg.norm(mic(pos[pairs[:, 1]] - pos[pairs[:, 0]], L), axis=1)
        bonded = np.array([pairs[k, 1] in self.neighbours(int(pairs[k, 0]))
                           for k in range(len(pairs))], dtype=bool)
        if not bonded.any() or not (~bonded).any():
            return 1.3 * self.pot.d
        # Percentiles, not max/min.  On a RELAXED lattice they agree, but on a
        # freshly built disordered network the bond distribution has a long
        # tail and max/min put the cutoff at 4.19 A -- above the second shell,
        # so the repulsion fought the quench (E/atom 4.56 with it vs 2.61
        # without).
        d_bond = float(np.percentile(d[bonded], 95))
        d_second = float(np.percentile(d[~bonded], 5))
        if d_second <= d_bond:
            # Shells OVERLAP -- true of any freshly constructed, unrelaxed
            # network, where the "midpoint between shells" heuristic is
            # meaningless (it returned 4.03 A on a loop-expansion start, above
            # the second shell, so the repulsion fought the quench).  Fall back
            # to the potential's own length scale.
            return 1.30 * self.pot.d
        return d_bond + frac * (d_second - d_bond)

    def refresh_z(self) -> None:
        self.z = (self.nbrs >= 0).sum(axis=1).astype(np.int64)

    def neighbours(self, i: int) -> np.ndarray:
        row = self.nbrs[i]
        return row[row >= 0]

    @property
    def n(self) -> int:
        return len(self.pos)

    # -- construction -------------------------------------------------------

    @classmethod
    def from_diamond(cls, reps: int, a: float = 5.431,
                     pot: Keating | None = None) -> "Network":
        """Perfect diamond network with an explicit bond list.

        Built analytically rather than by neighbour search so the bond list is
        exact and the starting energy is identically zero (up to the mismatch
        between a*sqrt(3)/4 and the Keating d).
        """
        basis = np.array([
            [0.00, 0.00, 0.00], [0.00, 0.50, 0.50],
            [0.50, 0.00, 0.50], [0.50, 0.50, 0.00],
            [0.25, 0.25, 0.25], [0.25, 0.75, 0.75],
            [0.75, 0.25, 0.75], [0.75, 0.75, 0.25],
        ])
        cells = np.array([(i, j, k) for i in range(reps)
                          for j in range(reps) for k in range(reps)],
                         dtype=np.float64)
        pos = ((cells[:, None, :] + basis[None, :, :]).reshape(-1, 3)) * a
        L = a * reps
        pos = np.mod(pos, L)

        # Exact neighbour list: in diamond every atom has 4 neighbours at
        # a*sqrt(3)/4.  Find them by MIC distance once, at construction only.
        from scipy.spatial import cKDTree
        tree = cKDTree(np.mod(pos, L), boxsize=L)
        dist, idx = tree.query(np.mod(pos, L), k=5)
        nbrs = idx[:, 1:5].astype(np.int64)
        expect = a * np.sqrt(3) / 4
        if not np.allclose(dist[:, 1:5], expect, atol=1e-6):
            raise RuntimeError("diamond neighbour construction failed")
        # Z=4 throughout, so default to the tetrahedral convention: it is zero
        # at equilibrium and is what the Keating literature values (beta=0.285,
        # E/atom ~0.3 for a CRN) are defined against.  max_repulsion remains
        # available and is required for any other/mixed coordination.
        net = cls(pos, nbrs, L, pot or Keating(angle_mode="tetrahedral"))
        net._assert_symmetric()
        return net

    @classmethod
    def from_ax2(cls, reps: int, d_ax: float = 1.61, z_cation: int = 4,
                 cation: int = 14, anion: int = 8,
                 pot: Keating | None = None) -> "Network":
        """MIXED-VALENCY AX2 network (silica topology): A has Z=4, X has Z=2.

        Built by the classical decoration construction: take the diamond
        A-sublattice and put one X on every A-A bond.  Every A then has 4 X
        neighbours and every X has exactly 2 A neighbours -- beta-cristobalite
        topology, the crystalline parent of silica glass.

        This is the test that the engine really is valency-general: the
        neighbour list is ragged (4 and 2 in the same network), and because a
        WWW switch conserves each atom's coordination, the 4/2 split is
        preserved exactly for all time.
        """
        a = d_ax * 2.0 * 4.0 / np.sqrt(3.0)      # A-A = 2*d_ax; A-A = a*sqrt3/4
        base = cls.from_diamond(reps, a=a, pot=pot)
        A = base.n
        bonds = base.bond_array()
        nb_ax = len(bonds)
        L = base.L
        pos = np.empty((A + nb_ax, 3))
        pos[:A] = base.pos
        mid = base.pos[bonds[:, 0]] + 0.5 * mic(
            base.pos[bonds[:, 1]] - base.pos[bonds[:, 0]], L)
        pos[A:] = mid

        zmax = max(z_cation, 2)
        nbrs = np.full((A + nb_ax, zmax), -1, dtype=np.int64)
        fill = np.zeros(A + nb_ax, dtype=np.int64)
        for k, (i, j) in enumerate(bonds):
            x = A + k
            nbrs[i, fill[i]] = x; fill[i] += 1
            nbrs[j, fill[j]] = x; fill[j] += 1
            nbrs[x, 0] = i; nbrs[x, 1] = j
        species = np.concatenate([np.full(A, cation), np.full(nb_ax, anion)])
        net = cls(pos, nbrs, L, pot or Keating(d=d_ax), species)
        net._assert_symmetric()
        if not (np.all(net.z[:A] == z_cation) and np.all(net.z[A:] == 2)):
            raise RuntimeError("AX2 construction produced wrong coordination")
        return net

    @classmethod
    def from_zincblende(cls, reps: int, d_ab: float = 1.89,
                        cation: int = 14, anion: int = 6,
                        pot: Keating | None = None) -> "Network":
        """Equal-stoichiometry AB network (zinc blende / SiC topology).

        Both species are 4-coordinated and every bond is A-B, so this is the
        case bond DECORATION cannot produce (there is no bridging anion to
        place) and the plain WWW move destroys (it forms A-A and B-B).  It is
        the test case for the species-preserving swap.
        """
        a = d_ab * 4.0 / np.sqrt(3.0)
        base = cls.from_diamond(reps, a=a, pot=pot)
        # the diamond basis alternates sublattices in blocks of 4
        idx = np.arange(base.n)
        sub = (idx % 8) >= 4
        species = np.where(sub, anion, cation).astype(np.int64)
        net = cls(base.pos, base.nbrs, base.L,
                  pot or Keating(d=d_ab, angle_mode="max_repulsion"), species)
        b = net.bond_array()
        if not np.all(species[b[:, 0]] != species[b[:, 1]]):
            raise RuntimeError("zinc-blende construction is not bipartite")
        return net

    @classmethod
    def from_random_binary(cls, n_atoms: int, density: float = 0.0499,
                           z: int = 4, d: float = 2.35,
                           species_a: int = 31, species_b: int = 33,
                           allow_wrong_bonds: bool = False,
                           rng: np.random.Generator | None = None,
                           pot: Keating | None = None, relax: bool = True,
                           max_tries: int = 200) -> "Network":
        """RANDOM START for a binary CRN, per Mousseau & Barkema,
        J. Phys.: Condens. Matter 16 (2004) [cond-mat/0408705]:

            "Atoms are initially placed in the box at random, and labeled A and
             B, in equal proportions. Bonds are then assigned to pairs of
             differently labeled atoms, with a strong preference for near
             atoms, until the desired total coordination of four is reached."

        This is NOT the same thing as melting a crystal, which is what the
        earlier dose sweep in this module did.  A random start carries no
        crystalline memory at all, so it neither anneals back to the crystal at
        low disorder nor has to be driven through a self-intersecting molten
        state to lose its lattice.  Their construction yields a network with a
        bond-angle spread of ~35 deg, no long-range order, and no chemical or
        coordination defects, which is then annealed DOWN.

        Bonds are assigned greedily shortest-first over unlike pairs, which is
        the "strong preference for near atoms".  With allow_wrong_bonds=False
        (default) only A-B bonds are ever created, so the start is chemically
        perfect; the paper's chemical-defect studies switch that on.
        """
        rng = rng or np.random.default_rng(0)
        L = (n_atoms / density) ** (1.0 / 3.0)
        na = n_atoms // 2
        species = np.array([species_a] * na + [species_b] * (n_atoms - na),
                           dtype=np.int64)

        def _separated(rng):
            """Random points with a MINIMUM SEPARATION.

            Barkema & Mousseau place atoms "randomly at crystalline density"
            with a minimum separation of 2.3 A.  Skipping that and using plain
            uniform points leaves pairs almost coincident, and the bond list
            built from them is geometrically frustrated in a way no relaxation
            can undo (measured: 22-31 eV/atom and a 94 deg angle spread that a
            4000-step quench barely moved).
            """
            from scipy.spatial import cKDTree
            r_min = 0.98 * d
            q = rng.random((n_atoms, 3)) * L
            for _ in range(400):
                t = cKDTree(np.mod(q, L), boxsize=L)
                pr = t.query_pairs(r_min, output_type="ndarray")
                if len(pr) == 0:
                    break
                dv = mic(q[pr[:, 1]] - q[pr[:, 0]], L)
                dd = np.linalg.norm(dv, axis=1)
                push = 0.5 * (r_min - dd) / np.maximum(dd, 1e-9)
                delta = push[:, None] * dv
                np.add.at(q, pr[:, 1], delta)
                np.add.at(q, pr[:, 0], -delta)
                q = np.mod(q, L)
            return np.mod(q, L)

        # A 4-regular BIPARTITE graph is exactly z disjoint perfect matchings
        # between the A and B sublattices.  Building it that way guarantees
        # coordination exactly z on every atom and zero wrong bonds by
        # construction, which plain greedy shortest-first does not: greedy
        # exhausts the short pairs early, leaves ~4% of atoms under-coordinated
        # and drives the remaining bonds out to 3.3 A against a 2.35 A target.
        pos = _separated(rng)
        A = np.where(species == species_a)[0]
        B = np.where(species == species_b)[0]
        if len(A) != len(B):
            raise ValueError("random binary start needs equal A and B counts")
        nbrs = np.full((n_atoms, z), -1, dtype=np.int64)
        deg = np.zeros(n_atoms, dtype=np.int64)
        used: set = set()
        dmat = np.linalg.norm(
            mic(pos[A][:, None, :] - pos[B][None, :, :], L), axis=2)
        for _round in range(z):
            # greedy shortest-first perfect matching for this round, skipping
            # pairs already bonded in an earlier round
            cost = dmat.copy()
            for (ia, ib) in used:
                cost[ia, ib] = np.inf
            order = np.dstack(np.unravel_index(
                np.argsort(cost, axis=None), cost.shape))[0]
            takenA, takenB = set(), set()
            for ia, ib in order:
                ia, ib = int(ia), int(ib)
                if ia in takenA or ib in takenB:
                    continue
                if not np.isfinite(cost[ia, ib]):
                    continue
                takenA.add(ia); takenB.add(ib); used.add((ia, ib))
                u, v = int(A[ia]), int(B[ib])
                nbrs[u, deg[u]] = v; deg[u] += 1
                nbrs[v, deg[v]] = u; deg[v] += 1
                if len(takenA) == len(A):
                    break
        if not np.all(deg == z):
            raise RuntimeError(
                f"matching construction left coordination {np.bincount(deg)}")

        net = cls(pos, nbrs, L, pot or Keating(d=d, angle_mode="tetrahedral"),
                  species)
        net._assert_symmetric()
        if relax:
            # The raw assignment is geometrically random (angle spread ~87 deg,
            # E ~ 23 eV/atom).  Barkema & Mousseau quench the initial network
            # before quoting ~30-35 deg: "initial CRN configurations with a
            # bond-angular spread of around 35 degrees".  Quench here so the
            # constructor returns a usable starting network rather than a
            # random point cloud with a bond list attached.
            ann = WWWAnnealer(net, rng=rng)
            ann.global_quench(iters=4000, fmax_stop=1e-3)
        return net

    @classmethod
    def from_random_ax(cls, n_cation: int, z_cation: int, z_anion: int,
                       d: float, density: float,
                       species_cation: int, species_anion: int,
                       rng: np.random.Generator | None = None,
                       pot: Keating | None = None,
                       r_min_frac: float = 0.92,
                       max_tries: int = 60) -> "Network":
        """Random start for a general A_m X_n network with UNEQUAL valency.

        Generalises Mousseau & Barkema's binary construction ("bonds assigned to
        pairs of differently labeled atoms, with a strong preference for near
        atoms, until the desired total coordination is reached") to arbitrary
        (z_cation, z_anion).  Stoichiometry is fixed by bond conservation --
        every bond has one A end and one X end, so

            N_A * z_A = N_X * z_X

        which gives SiO2 (4,2) -> 1:2, Si3N4 (4,3) -> 3:4, TiO2 (6,3) -> 1:2,
        Al2O3 (6,4) -> 2:3, SiC (4,4) -> 1:1.  The network is bipartite by
        construction, so no homonuclear bond can ever exist, and the
        species-preserving swap plus the T=0 structured quench then apply
        unchanged.

        Bonds are filled shortest-first subject to both degree caps, in rounds
        of a rising cutoff so short bonds are exhausted before long ones -- the
        "strong preference for near atoms".
        """
        rng = rng or np.random.default_rng(0)
        from scipy.spatial import cKDTree
        if (n_cation * z_cation) % z_anion:
            raise ValueError(
                f"stoichiometry not integral: {n_cation}*{z_cation} "
                f"is not divisible by {z_anion}")
        n_anion = n_cation * z_cation // z_anion
        n_atoms = n_cation + n_anion
        L = (n_atoms / density) ** (1.0 / 3.0)
        species = np.concatenate([np.full(n_cation, species_cation),
                                  np.full(n_anion, species_anion)]).astype(np.int64)
        zt = np.concatenate([np.full(n_cation, z_cation),
                             np.full(n_anion, z_anion)]).astype(np.int64)
        zmax = int(max(z_cation, z_anion))
        r_min = r_min_frac * d

        best = None
        for _try in range(max_tries):
            pos = rng.random((n_atoms, 3)) * L
            for _ in range(400):
                t = cKDTree(np.mod(pos, L), boxsize=L)
                pr = t.query_pairs(r_min, output_type="ndarray")
                if len(pr) == 0:
                    break
                dv = mic(pos[pr[:, 1]] - pos[pr[:, 0]], L)
                dd = np.linalg.norm(dv, axis=1)
                push = 0.5 * (r_min - dd) / np.maximum(dd, 1e-9)
                delta = push[:, None] * dv
                np.add.at(pos, pr[:, 1], delta)
                np.add.at(pos, pr[:, 0], -delta)
                pos = np.mod(pos, L)
            pos = np.mod(pos, L)

            nbrs = np.full((n_atoms, zmax), -1, dtype=np.int64)
            deg = np.zeros(n_atoms, dtype=np.int64)
            tree = cKDTree(pos, boxsize=L)
            for r_cut in np.arange(1.15 * d, 3.2 * d, 0.12 * d):
                pairs = tree.query_pairs(float(r_cut), output_type="ndarray")
                if len(pairs) == 0:
                    continue
                cross = species[pairs[:, 0]] != species[pairs[:, 1]]
                pairs = pairs[cross]
                if len(pairs) == 0:
                    continue
                dv = mic(pos[pairs[:, 1]] - pos[pairs[:, 0]], L)
                order = np.argsort(np.linalg.norm(dv, axis=1))
                for k in order:
                    i, j = int(pairs[k, 0]), int(pairs[k, 1])
                    if deg[i] >= zt[i] or deg[j] >= zt[j]:
                        continue
                    if j in nbrs[i, :deg[i]]:
                        continue
                    nbrs[i, deg[i]] = j; deg[i] += 1
                    nbrs[j, deg[j]] = i; deg[j] += 1
                if np.all(deg == zt):
                    break
            short = int((zt - deg).clip(min=0).sum())
            if best is None or short < best[0]:
                best = (short, pos.copy(), nbrs.copy(), deg.copy())
            if short == 0:
                break

        short, pos, nbrs, deg = best
        short_greedy = short          # deficit left by the PUBLISHED step alone

        # AUGMENTING-PATH REPAIR.  Greedy fill leaves a few atoms short: at the
        # end, an under-coordinated atom's remaining in-range partners are all
        # saturated.  This is a degree-constrained bipartite subgraph (b-
        # matching), and the standard remedy is an augmenting path, not a
        # looser cutoff -- if a saturated neighbour X can hand one of its bonds
        # to some other atom that still has a free slot, the deficit moves and
        # can be filled.  Length-3 augmentation resolves essentially all of it.
        if short:
            from scipy.spatial import cKDTree
            tree = cKDTree(pos, boxsize=L)
            # A FIXED search radius, deliberately.  Letting it grow across
            # sweeps was tried and is worse: long bonds get committed early and
            # then block the augmenting paths that would have fixed the rest.
            # Measured, growing radius vs fixed: SiO2 6 -> 32 under-coordinated,
            # Si3N4 0 -> 10, and SiC bond length 1.99 -> 2.35 A against a 1.89 A
            # target.  More sweeps at a fixed radius is the thing that helps.
            r_rep_ = 3.0 * d
            for _sweep in range(10):
                deficit = [int(i) for i in np.where(deg < zt)[0]]
                if not deficit:
                    break
                progressed = False
                for A in deficit:
                    while deg[A] < zt[A]:
                        cand = [int(c) for c in
                                tree.query_ball_point(pos[A], r_rep_)
                                if species[c] != species[A] and c != A
                                and c not in nbrs[A, :deg[A]]]
                        cand.sort(key=lambda c: np.linalg.norm(
                            mic(pos[c] - pos[A], L)))
                        placed = False
                        # direct: a partner with a free slot
                        for X in cand:
                            if deg[X] < zt[X]:
                                nbrs[A, deg[A]] = X; deg[A] += 1
                                nbrs[X, deg[X]] = A; deg[X] += 1
                                placed = progressed = True
                                break
                        if placed:
                            continue
                        # augment: X is saturated, but one of X's partners Y
                        # can be re-homed to some Z that still has a slot
                        for X in cand:
                            for slot in range(int(deg[X])):
                                Y = int(nbrs[X, slot])
                                if Y == A:
                                    continue
                                alt = [int(c) for c in
                                       tree.query_ball_point(pos[Y], r_rep_)
                                       if species[c] != species[Y]
                                       and deg[c] < zt[c] and c != X
                                       and c not in nbrs[Y, :deg[Y]]]
                                if not alt:
                                    continue
                                Z = min(alt, key=lambda c: np.linalg.norm(
                                    mic(pos[c] - pos[Y], L)))
                                ys = int(np.where(nbrs[Y] == X)[0][0])
                                nbrs[Y, ys] = Z
                                nbrs[Z, deg[Z]] = Y; deg[Z] += 1
                                nbrs[X, slot] = A
                                nbrs[A, deg[A]] = X; deg[A] += 1
                                placed = progressed = True
                                break
                            if placed:
                                break
                        if not placed:
                            break
                if not progressed:
                    break
            # Final targeted pass: whatever deficit survives is spatially
            # separated (cations short in one region, anions in another, both
            # outside the search radius).  Connect those directly regardless of
            # distance.  This touches only a handful of bonds, whereas widening
            # the radius globally commits long bonds everywhere and blocks the
            # augmenting paths -- measured, that made SiO2 worse (6 -> 32) and
            # inflated SiC bonds 1.99 -> 2.35 A.  A few long bonds the quench
            # can contract beat leaving atoms under-coordinated.
            for _ in range(200):
                dA = [int(i) for i in np.where(deg < zt)[0]]
                if not dA:
                    break
                fixed_any = False
                for A in dA:
                    partners = [int(c) for c in np.where(deg < zt)[0]
                                if species[c] != species[A] and c != A
                                and c not in nbrs[A, :deg[A]]]
                    if not partners:
                        continue
                    X = min(partners, key=lambda c: np.linalg.norm(
                        mic(pos[c] - pos[A], L)))
                    nbrs[A, deg[A]] = X; deg[A] += 1
                    nbrs[X, deg[X]] = A; deg[X] += 1
                    fixed_any = True
                if not fixed_any:
                    break
            short = int((zt - deg).clip(min=0).sum())

        net = cls(pos, nbrs, L,
                  pot or Keating(d=d, angle_mode="max_repulsion"), species)
        net._assert_symmetric()
        net.target_z = zt
        net.n_under = short
        # How much of the final topology is owed to machinery that is NOT in
        # either paper: the deficit greedy assignment alone leaves, versus what
        # survives the augmenting-path repair and the final targeted pass.
        net.n_under_greedy = short_greedy
        return net

    @classmethod
    def from_loop(cls, n_atoms: int, density: float = 0.0499,
                  d: float = 2.35, binary: bool = False,
                  species_a: int = 14, species_b: int = 14,
                  r_min_frac: float = 0.979, r_c0: float = 3.0,
                  rng: np.random.Generator | None = None,
                  pot: Keating | None = None) -> "Network":
        """Barkema & Mousseau loop expansion with the LOOP kept explicitly.

        "a closed loop visiting every atom exactly twice; steps of the loop are
        the bonds".  The loop is held here as a doubly-linked cyclic list, and
        an insertion puts A between two loop-ADJACENT atoms B and C:

            ... B , C ...   ->   ... B , A , C ...

        so the step B-C is replaced by B-A and A-C.  Every atom appears exactly
        twice and therefore ends with four bonds, by construction.

        This differs from the adjacency-set version in a way that turns out to
        matter: there, ANY bonded pair (B,C) was a candidate and the B-C bond
        was deleted outright.  In the loop, B and C must be CONSECUTIVE, and
        only that one occurrence of the step is consumed -- if B and C are also
        adjacent elsewhere in the loop, that bond survives.  The set version is
        therefore not a faithful implementation of the same algorithm, and it
        produced 13-14% three-membered rings, which no relaxation can remove
        and which forced an ad-hoc 3-ring ban.
        """
        rng = rng or np.random.default_rng(0)
        from scipy.spatial import cKDTree
        L = (n_atoms / density) ** (1.0 / 3.0)
        r_min = r_min_frac * d

        pos = rng.random((n_atoms, 3)) * L
        for _ in range(600):
            t = cKDTree(np.mod(pos, L), boxsize=L)
            pr = t.query_pairs(r_min, output_type="ndarray")
            if len(pr) == 0:
                break
            dv = mic(pos[pr[:, 1]] - pos[pr[:, 0]], L)
            dd = np.linalg.norm(dv, axis=1)
            step = 0.5 * (r_min - dd) / np.maximum(dd, 1e-9)
            delta = step[:, None] * dv
            np.add.at(pos, pr[:, 1], delta)
            np.add.at(pos, pr[:, 0], -delta)
            pos = np.mod(pos, L)
        pos = np.mod(pos, L)

        if binary:
            sp = np.array([species_a] * (n_atoms // 2) +
                          [species_b] * (n_atoms - n_atoms // 2), dtype=np.int64)
            rng.shuffle(sp)
        else:
            sp = np.full(n_atoms, species_a, dtype=np.int64)

        # doubly-linked cyclic loop over node slots; 2 nodes per atom
        cap = 2 * n_atoms + 8
        nxt = np.full(cap, -1, dtype=np.int64)
        prv = np.full(cap, -1, dtype=np.int64)
        at = np.full(cap, -1, dtype=np.int64)
        occ = np.zeros(n_atoms, dtype=np.int64)
        n_node = 0
        adj = [set() for _ in range(n_atoms)]

        tree = cKDTree(pos, boxsize=L)
        r_c = r_c0

        def link(a, b):
            nxt[a] = b; prv[b] = a
            adj[int(at[a])].add(int(at[b])); adj[int(at[b])].add(int(at[a]))

        # seed: a 4-cycle of mutually nearby atoms (alternating if binary)
        seed = None
        for _ in range(20000):
            i = int(rng.integers(n_atoms))
            near = [int(c) for c in tree.query_ball_point(pos[i], r_c) if c != i]
            if binary:
                near = [c for c in near if sp[c] != sp[i]]
            if len(near) < 2:
                continue
            j = int(rng.choice(near))
            nk = [int(c) for c in tree.query_ball_point(pos[j], r_c)
                  if c not in (i, j) and (not binary or sp[c] != sp[j])]
            if not nk:
                continue
            k = int(rng.choice(nk))
            nl = [int(c) for c in tree.query_ball_point(pos[k], r_c)
                  if c not in (i, j, k) and (not binary or sp[c] != sp[k])
                  and np.linalg.norm(mic(pos[c] - pos[i], L)) < r_c
                  and (not binary or sp[c] != sp[i])]
            if not nl:
                continue
            seed = (i, j, k, int(rng.choice(nl)))
            break
        if seed is None:
            raise RuntimeError("could not seed the loop")
        for a in seed:
            at[n_node] = a; occ[a] += 1; n_node += 1
        for q in range(4):
            link(q, (q + 1) % 4)

        def insert(node_b, A, A2=None):
            """put A (and optionally A2) between node_b and its successor.

            Single insertion B-A-C is impossible in a bipartite network: B and C
            are loop-adjacent and therefore already unlike, so no single A can
            differ from both.  Binary networks insert a PAIR, B-A1-A2-C, with A1
            unlike B and A2 unlike A1, which preserves the alternation exactly
            and still leaves B and C untouched.
            """
            nonlocal n_node
            nb = int(nxt[node_b])
            B, C = int(at[node_b]), int(at[nb])
            adj[B].discard(C); adj[C].discard(B)
            r = n_node; n_node += 1
            at[r] = A; occ[A] += 1
            link(node_b, r)
            if A2 is not None:
                r2 = n_node; n_node += 1
                at[r2] = A2; occ[A2] += 1
                link(r, r2); link(r2, nb)
            else:
                link(r, nb)
            # B-C may still be a step elsewhere in the loop
            n_ = int(nxt[0]); cur = 0
            for _ in range(n_node):
                u, v = int(at[cur]), int(at[int(nxt[cur])])
                if (u == B and v == C) or (u == C and v == B):
                    adj[B].add(C); adj[C].add(B)
                    break
                cur = int(nxt[cur])

        while np.any(occ < 2):
            progressed = False
            cand_atoms = [int(a) for a in np.where(occ < 2)[0]]
            rng.shuffle(cand_atoms)
            for A in cand_atoms:
                near = set(int(c) for c in tree.query_ball_point(pos[A], r_c))
                partners = [int(c) for c in near
                            if c != A and occ[c] < 2 and sp[c] != sp[A]] \
                    if binary else []
                best, best_cost = None, None
                cur = 0
                for _ in range(n_node):
                    nb = int(nxt[cur])
                    B, C = int(at[cur]), int(at[nb])
                    if B == A or C == A:
                        cur = nb; continue
                    if not binary:
                        if B in near and C in near and B not in adj[A] \
                                and C not in adj[A]:
                            cost = (np.linalg.norm(mic(pos[B] - pos[A], L)) +
                                    np.linalg.norm(mic(pos[C] - pos[A], L)))
                            if best_cost is None or cost < best_cost:
                                best, best_cost = cur, cost
                    else:
                        # pair insertion B-A-A2-C: A unlike B, A2 unlike A
                        if B in near and sp[B] != sp[A] and B not in adj[A]:
                            for A2 in partners:
                                if A2 == A or occ[A2] >= 2 or sp[A2] == sp[A]:
                                    continue
                                if sp[C] == sp[A2] or C in adj[A2]:
                                    continue
                                if np.linalg.norm(mic(pos[A2] - pos[C], L)) > r_c:
                                    continue
                                cost = (np.linalg.norm(mic(pos[B] - pos[A], L)) +
                                        np.linalg.norm(mic(pos[A2] - pos[A], L)) +
                                        np.linalg.norm(mic(pos[C] - pos[A2], L)))
                                if best_cost is None or cost < best_cost:
                                    best, best_cost = (cur, A2), cost
                    cur = nb
                if best is not None:
                    if binary:
                        insert(best[0], A, best[1])
                    else:
                        insert(best, A)
                    progressed = True
                    break
            if not progressed:
                r_c += 0.25
                if r_c > 0.49 * float(np.min(L)):
                    raise RuntimeError(
                        f"loop stalled: {int((occ < 2).sum())} atoms with "
                        f"occ<2 at r_c={r_c:.2f}")

        nbrs = np.full((n_atoms, 4), -1, dtype=np.int64)
        for a in range(n_atoms):
            row = sorted(adj[a])
            if len(row) != 4:
                raise RuntimeError(f"atom {a} ended with {len(row)} bonds")
            nbrs[a, :4] = row
        net = cls(pos, nbrs, L, pot or Keating(d=d, angle_mode="tetrahedral"),
                  sp)
        net._assert_symmetric()
        return net

    @classmethod
    def from_loop_expansion(cls, n_atoms: int, density: float = 0.0499,
                            d: float = 2.35, z: int = 4,
                            binary: bool = False,
                            forbid_3rings: bool = False,
                            forbid_4rings: bool = False,
                            species_a: int = 14, species_b: int = 14,
                            r_min_frac: float = 0.979,
                            rng: np.random.Generator | None = None,
                            pot: Keating | None = None,
                            relax: bool = True) -> "Network":
        """Barkema & Mousseau loop-expansion random start (PRB 62, 4985).

        Verbatim rule:  "Three atoms A, B and C are selected, such that A is not
        four-fold coordinated and is within a distance of r_c from B and C but
        not bonded to either, while B and C are bonded. Next, the bond BC is
        replaced by bonds AB and AC, expanding the loop by one step."

        Each insertion raises A's coordination by two and leaves B and C
        unchanged, so every atom is inserted exactly twice -- this is what "a
        closed loop visiting every atom exactly twice" means, and it gives 2N
        bonds, i.e. fourfold coordination, by construction.  r_c starts near
        3 A and is "gradually increased until all atoms are four-fold
        coordinated", which is what keeps the bonds short: topology and
        geometry are grown TOGETHER, unlike imposing a random 4-regular graph on
        a fixed point set (which gave 3.6 A bonds and 37 eV/atom).

        binary=True switches to PAIR insertion, B-A1-A2-C.  The elemental rule
        cannot preserve bipartiteness: B and C are bonded and therefore already
        unlike, so no single A can differ from both.  Inserting two atoms, one
        of each species, keeps the alternation exact and still leaves B and C
        untouched.
        """
        rng = rng or np.random.default_rng(0)
        from scipy.spatial import cKDTree
        L = (n_atoms / density) ** (1.0 / 3.0)
        r_min = r_min_frac * d

        # atoms at crystalline density, no two closer than ~2.3 A
        pos = rng.random((n_atoms, 3)) * L
        for _ in range(600):
            t = cKDTree(np.mod(pos, L), boxsize=L)
            pr = t.query_pairs(r_min, output_type="ndarray")
            if len(pr) == 0:
                break
            dv = mic(pos[pr[:, 1]] - pos[pr[:, 0]], L)
            dd = np.linalg.norm(dv, axis=1)
            step = 0.5 * (r_min - dd) / np.maximum(dd, 1e-9)
            delta = step[:, None] * dv
            np.add.at(pos, pr[:, 1], delta)
            np.add.at(pos, pr[:, 0], -delta)
            pos = np.mod(pos, L)

        if binary:
            sp = np.array([species_a] * (n_atoms // 2) +
                          [species_b] * (n_atoms - n_atoms // 2), dtype=np.int64)
            rng.shuffle(sp)
        else:
            sp = np.full(n_atoms, species_a, dtype=np.int64)

        nbrs = np.full((n_atoms, z), -1, dtype=np.int64)
        deg = np.zeros(n_atoms, dtype=np.int64)
        bonds: set = set()

        def add(u, v):
            nbrs[u, deg[u]] = v; deg[u] += 1
            nbrs[v, deg[v]] = u; deg[v] += 1
            bonds.add((min(u, v), max(u, v)))

        def drop(u, v):
            for a_, b_ in ((u, v), (v, u)):
                row = nbrs[a_][nbrs[a_] >= 0]
                row = row[row != b_]
                nbrs[a_] = -1
                nbrs[a_, :len(row)] = row
                deg[a_] = len(row)
            bonds.discard((min(u, v), max(u, v)))

        tree = cKDTree(np.mod(pos, L), boxsize=L)
        r_c = 3.0

        # seed loop: a 4-cycle of mutually nearby atoms (alternating if binary)
        seed = None
        for _try in range(4000):
            i = int(rng.integers(n_atoms))
            cand = [int(c) for c in tree.query_ball_point(np.mod(pos[i], L), r_c)
                    if c != i]
            if binary:
                cand = [c for c in cand if sp[c] != sp[i]]
            if len(cand) < 2:
                continue
            j, l = (int(x) for x in rng.choice(cand, 2, replace=False))
            others = [int(c) for c in tree.query_ball_point(np.mod(pos[j], L), r_c)
                      if c not in (i, j, l)]
            if binary:
                others = [c for c in others if sp[c] == sp[i]]
            if not others:
                continue
            k = int(rng.choice(others))
            if binary and not (sp[j] != sp[k] and sp[k] != sp[l]):
                continue
            seed = (i, j, k, l)
            break
        if seed is None:
            raise RuntimeError("could not seed the initial loop")
        i, j, k, l = seed
        add(i, j); add(j, k); add(k, l); add(l, i)

        # r_c is raised only when NO under-coordinated atom admits an
        # insertion at the current cutoff -- "gradually increased UNTIL ALL
        # ATOMS ARE FOUR-FOLD COORDINATED".  An earlier version raised it after
        # one randomly-chosen atom failed 40 times, which is a different thing:
        # a stall for atom A says nothing about the other under-coordinated
        # atoms, so the cutoff grew prematurely and long bonds were spent
        # before the short ones were exhausted.
        queue: list[int] = []
        while np.any(deg < z):
            if not queue:
                queue = [int(i) for i in np.where(deg < z)[0]]
                rng.shuffle(queue)
                swept_without_progress = True
            A = queue.pop()
            near = [int(c) for c in tree.query_ball_point(np.mod(pos[A], L), r_c)
                    if c != A]
            nearset = set(near)
            done = False
            if not binary:
                # replace bond BC with AB + AC
                for (B, C) in sorted(
                        (bc for bc in bonds
                         if bc[0] in nearset and bc[1] in nearset),
                        key=lambda bc: np.linalg.norm(
                            mic(pos[bc[0]] - pos[A], L)) +
                        np.linalg.norm(mic(pos[bc[1]] - pos[A], L))):
                    if B in nbrs[A][:deg[A]] or C in nbrs[A][:deg[A]]:
                        continue
                    if deg[A] + 2 > z:
                        continue
                    # NO LONGER NEEDED, default off.  Measured: with the T=0
                    # structured quench in place, 3-rings fall from 12% to
                    # EXACTLY ZERO on their own in ~361 switches, because
                    # removing a 3-ring lowers the energy.  The ban was
                    # compensating for a missing algorithm step, which is
                    # precisely the kind of patch that would have made this
                    # fragile on other chemistries.  Kept only as a diagnostic.
                    # Historical note: forbid short rings AT CONSTRUCTION.  Nothing in the
                    # insertion rule prevents them, and unconstrained it yields
                    # 11.6% 3-rings and 17.3% 4-rings (published a-Si: ~0.3%
                    # and ~2%).  A 3-ring pins three angles near 60 deg, so the
                    # first quench then stalls at ~30 deg instead of reaching
                    # ~13 -- the strain is topological and no relaxation can
                    # remove it.
                    nA = set(int(x) for x in nbrs[A][:deg[A]])
                    nB = set(int(x) for x in nbrs[B][:deg[B]]) - {C}
                    nC = set(int(x) for x in nbrs[C][:deg[C]]) - {B}
                    if forbid_3rings and ((nA & nB) or ((nA | {B}) & nC)):
                        continue                      # would close a 3-ring
                    # 4-rings are ALLOWED by default: Barkema & Mousseau note
                    # "allowing 4-membered rings during quench opens many
                    # relaxation pathways; the few formed are removed at the
                    # end".  Measured, banning them at construction forces
                    # longer bonds (2.62 vs 2.46 A) and a worse quench
                    # (5.27 vs 3.15 eV/atom).  3-rings are always forbidden.
                    if forbid_4rings:
                        nnB = set()
                        for x in nB:
                            nnB |= set(int(y) for y in nbrs[x][:deg[x]])
                        nnC = set()
                        for x in nC:
                            nnC |= set(int(y) for y in nbrs[x][:deg[x]])
                        if (nA & nnB) or ((nA | {B}) & nnC):
                            continue                  # would close a 4-ring
                    drop(B, C); add(A, B); add(A, C)
                    done = True
                    break
            else:
                # PAIR insertion B-A1-A2-C keeps the species alternating
                partners = [c for c in near if sp[c] != sp[A] and deg[c] < z]
                for A2 in sorted(partners, key=lambda c: np.linalg.norm(
                        mic(pos[c] - pos[A], L))):
                    if deg[A] + 2 > z or deg[A2] + 2 > z:
                        continue
                    n2 = set(int(c) for c in
                             tree.query_ball_point(np.mod(pos[A2], L), r_c))
                    for (B, C) in bonds:
                        if A in (B, C) or A2 in (B, C):
                            continue
                        for (b_, c_) in ((B, C), (C, B)):
                            if sp[b_] == sp[A] or sp[c_] == sp[A2]:
                                continue
                            if b_ not in nearset or c_ not in n2:
                                continue
                            if b_ in nbrs[A][:deg[A]] or c_ in nbrs[A2][:deg[A2]]:
                                continue
                            drop(B, C); add(A, b_); add(A, A2); add(A2, c_)
                            done = True
                            break
                        if done:
                            break
                    if done:
                        break
            if done:
                swept_without_progress = False
            if not queue and swept_without_progress:
                # a full sweep over every under-coordinated atom achieved
                # nothing: only now is the cutoff genuinely exhausted
                r_c += 0.25
                if r_c > 0.49 * float(np.min(L)):
                    raise RuntimeError(
                        f"loop expansion stalled: {int((deg < z).sum())} "
                        f"atoms under-coordinated at r_c={r_c:.2f}")

        net = cls(pos, nbrs, L, pot or Keating(d=d, angle_mode="tetrahedral"),
                  sp)
        net._assert_symmetric()
        if relax:
            WWWAnnealer(net, rng=rng).global_quench(iters=4000, fmax_stop=1e-3)
        return net

    def _assert_symmetric(self) -> None:
        for i in range(self.n):
            for j in self.neighbours(i):
                if i not in self.neighbours(int(j)):
                    raise RuntimeError(f"asymmetric neighbour list at {i}-{j}")

    def save(self, path) -> None:
        np.savez(str(path), pos=self.pos, nbrs=self.nbrs, L=self.L,
                 species=self.species, alpha=self.pot.alpha,
                 beta_frac=self.pot.beta_frac, d=self.pot.d,
                 angle_mode=self.pot.angle_mode)

    @classmethod
    def load(cls, path) -> "Network":
        z = np.load(str(path))
        am = str(z["angle_mode"]) if "angle_mode" in z else "max_repulsion"
        pot = Keating(float(z["alpha"]), float(z["beta_frac"]), float(z["d"]),
                      am)
        sp = z["species"] if "species" in z else None
        # L is a length-3 cell now; float() on it raises.  Older files hold a
        # scalar, which the constructor broadcasts, so both load.
        return cls(z["pos"], z["nbrs"], np.asarray(z["L"]).reshape(-1), pot, sp)

    def to_atoms(self, symbol: str | None = None):
        from ase import Atoms
        numbers = (np.full(self.n, 14) if symbol == "Si"
                   else self.species)
        return Atoms(numbers=numbers, positions=np.mod(self.pos, self.L),
                     cell=list(self.L), pbc=True)

    # -- energy -------------------------------------------------------------

    def bond_array(self) -> np.ndarray:
        """Unique bonds as an (n_bond, 2) array with i < j (padding skipped)."""
        zmax = self.nbrs.shape[1]
        i = np.repeat(np.arange(self.n), zmax)
        j = self.nbrs.ravel()
        m = (j >= 0) & (i < j)
        return np.stack([i[m], j[m]], axis=1)

    def angle_array(self, centers: np.ndarray | None = None) -> np.ndarray:
        """Angle triplets (center, nbr_a, nbr_b), for arbitrary/mixed Z.

        Atoms are grouped by coordination number so each group can use its own
        C(z,2) pair table; padding never enters.
        """
        c = np.arange(self.n) if centers is None else np.asarray(centers)
        out = []
        zc = self.z[c]
        for zval in np.unique(zc):
            if zval < 2:
                continue
            sel = c[zc == zval]
            nb = self.nbrs[sel][:, :zval]
            iu = np.triu_indices(int(zval), 1)
            a = nb[:, iu[0]]
            b = nb[:, iu[1]]
            cc = np.repeat(sel[:, None], len(iu[0]), axis=1)
            out.append(np.stack([cc.ravel(), a.ravel(), b.ravel()], axis=1))
        if not out:
            return np.zeros((0, 3), dtype=np.int64)
        return np.concatenate(out, axis=0)

    def energy_forces(self, bonds: np.ndarray, angles: np.ndarray,
                      pos: np.ndarray | None = None,
                      want_forces: bool = True):
        """Keating energy (and forces) for the given bond/angle term sets."""
        p = self.pot
        pos = self.pos if pos is None else pos
        L = self.L
        F = np.zeros_like(pos) if want_forces else None

        rij = mic(pos[bonds[:, 1]] - pos[bonds[:, 0]], L)
        s = np.einsum("ij,ij->i", rij, rij)
        db = s - p.d2
        E = p.kb * float(db @ db)
        if want_forces:
            fb = (4.0 * p.kb * db)[:, None] * rij
            np.add.at(F, bonds[:, 1], -fb)
            np.add.at(F, bonds[:, 0], fb)

        ci, aj, bk = angles[:, 0], angles[:, 1], angles[:, 2]
        rA = mic(pos[aj] - pos[ci], L)
        rB = mic(pos[bk] - pos[ci], L)
        if p.angle_mode == "ideal_per_z":
            off = p.ideal_offset(self.z[angles[:, 0]])
        elif p.angle_mode == "crystal":
            off = p.species_offset(self.species[angles[:, 0]])
        else:
            off = p.angle_offset
        u = np.einsum("ij,ij->i", rA, rB) + off
        E += p.ka * float(u @ u)
        if want_forces:
            g = (2.0 * p.ka * u)[:, None]
            np.add.at(F, aj, -g * rB)
            np.add.at(F, bk, -g * rA)
            np.add.at(F, ci, g * (rA + rB))
        return (E, F) if want_forces else E

    def total_energy(self) -> float:
        return self.energy_forces(self.bond_array(), self.angle_array(),
                                  want_forces=False)

    def energy_per_atom(self) -> float:
        return self.total_energy() / self.n


# ── local cluster machinery ──────────────────────────────────────────────────


def bond_hops(net: Network, seeds: np.ndarray, n_shell: int) -> np.ndarray:
    """All atoms within n_shell bond hops of the seed atoms (seeds included)."""
    seen = set(int(s) for s in seeds)
    frontier = list(seen)
    for _ in range(n_shell):
        nxt = []
        for x in frontier:
            for y in net.neighbours(x):
                y = int(y)
                if y not in seen:
                    seen.add(y)
                    nxt.append(y)
        frontier = nxt
    return np.fromiter(seen, dtype=np.int64)


def cluster_terms(net: Network, moving: np.ndarray):
    """Bond and angle term sets that involve any moving atom.

    Bonds: every bond with at least one end moving.  Angles: every angle whose
    center OR either arm moves -- gathered via the centers that are within one
    hop of a moving atom, then filtered.  Returns (bonds, angles, frozen_mask)
    where frozen_mask marks cluster atoms that must NOT move (the boundary
    ring is included in the terms but held fixed).
    """
    mov = np.zeros(net.n, dtype=bool)
    mov[moving] = True

    # candidate centers: moving atoms + their neighbours
    nb_flat = net.nbrs[moving].ravel()
    cand = np.unique(np.concatenate([moving, nb_flat[nb_flat >= 0]]))
    ang = net.angle_array(cand)
    keep = mov[ang[:, 0]] | mov[ang[:, 1]] | mov[ang[:, 2]]
    ang = ang[keep]

    zmax = net.nbrs.shape[1]
    i = np.repeat(cand, zmax)
    j = net.nbrs[cand].ravel()
    ok = j >= 0
    i, j = i[ok], j[ok]
    # Dedupe via a single int64 key rather than np.unique(axis=0).  axis=0
    # unique builds a structured view and sorts rows, which profiled at 32% of
    # total WWW runtime across 7 unique() calls per attempt.  Encoding the
    # ordered pair as lo*n + hi makes it a 1D unique: measured 39.1 us -> 12.2
    # us (3.2x) on a representative ~45-atom patch, and the result is
    # identical because key order IS lexicographic order for non-negative ids.
    # NOT the same thing as Vink et al. PRB 64, 245214 sec. IV.B point 3, which
    # is a separate and still-unimplemented optimisation: they cache each bond's
    # GEOMETRY (three components + squared length) behind a time-stamp flag so
    # repeated references within one energy/force evaluation are retrieved from
    # memory rather than recomputed.  This only speeds up BUILDING the
    # deduplicated bond list.  Both target the same observation -- "in the
    # calculation of the cluster energy most bonds are encountered more than
    # once" -- but their memoisation is still available as an additional win.
    lo = np.minimum(i, j).astype(np.int64)
    hi = np.maximum(i, j).astype(np.int64)
    key = np.unique(lo * net.n + hi)
    b = np.stack([key // net.n, key % net.n], axis=1)
    b = b[mov[b[:, 0]] | mov[b[:, 1]]]
    return b, ang


# ── compact local patches (the O(1)-per-move machinery) ──────────────────────


def build_patch(net: Network, moving: np.ndarray,
                lookup: np.ndarray | None = None):
    """Remap a cluster into a compact local index space.

    The naive implementation evaluates cluster terms against full (N,3)
    position/force arrays.  That allocates and scatters over N on EVERY
    attempted move, which silently turns an O(1) move into an O(N) one and
    destroys the size-independence the method is chosen for.  Here the cluster
    is remapped to 0..m-1 (m ~ 150), so cost depends only on cluster size.

    Returns (gidx, pos_loc, bonds_loc, angles_loc, movable_loc):
      gidx        global atom indices, shape (m,)
      pos_loc     their positions, shape (m,3)
      bonds_loc   bond term set in local indices
      angles_loc  angle term set in local indices
      movable_loc local indices of atoms allowed to move
    """
    bonds_g, angles_g = cluster_terms(net, moving)
    gidx = np.unique(np.concatenate(
        [bonds_g.ravel(), angles_g.ravel(), moving]))
    # A fresh np.full(N) here would be O(N) per attempted move -- the very cost
    # this patch machinery exists to avoid.  Reuse a persistent scratch buffer
    # and clear only the m entries actually written.
    own = lookup is None
    if own:
        lookup = np.full(net.n, -1, dtype=np.int64)
    lookup[gidx] = np.arange(len(gidx))
    bonds_loc = lookup[bonds_g]
    angles_loc = lookup[angles_g]
    movable_loc = lookup[moving]
    if not own:
        lookup[gidx] = -1
    # Unwrap about the first moving atom: the cluster is bond-connected and
    # only a few shells across, so this makes it spatially contiguous.  That
    # lets a plain (non-periodic) KD-tree find non-bonded contacts inside the
    # patch, and leaves intra-patch differences already minimum-image.
    ref = net.pos[moving[0]]
    pos_loc = ref + mic(net.pos[gidx] - ref, net.L)
    return gidx, pos_loc, bonds_loc, angles_loc, movable_loc


def patch_nonbonded(pos: np.ndarray, bonds: np.ndarray, r_rep: float,
                    L: float | None = None, exclude_second: bool = False):
    """Non-bonded pairs inside an (unwrapped) patch, closer than r_rep.

    Keating constrains only bonded distances and angles, so unbonded atoms can
    collapse together at zero energy cost -- WWW networks built without this
    develop ~1.6 A contacts.  Doing the check inside the patch keeps it
    O(cluster); the equivalent global check would reintroduce an O(N) term per
    move, which is exactly what the local scheme exists to avoid.
    """
    from scipy.spatial import cKDTree
    m = len(pos)
    # The patch is unwrapped about its first moving atom, but in a small cell
    # it can still span more than L/2, and a NON-periodic tree then misses
    # real contacts across the seam (measured: 15-25% missed at L=16-22 A).
    if L is not None and np.ptp(pos, axis=0).max() > 0.5 * float(np.min(L)):
        # np.mod can return exactly L for a tiny negative input (-1e-17 % L
        # == L in float64), and cKDTree rejects coordinates equal to the box
        # size.  Only reachable with k_rep > 0, so this path went unexercised
        # while the non-bonded term defaulted to off.
        wrapped = np.minimum(np.mod(pos, L), L * (1.0 - 1e-12))
        tree = cKDTree(wrapped, boxsize=L)
    else:
        tree = cKDTree(pos)
    pairs = tree.query_pairs(r_rep, output_type="ndarray")
    if len(pairs) == 0:
        return pairs
    # Vectorised set-difference on encoded (i,j) keys; a Python-level `in`
    # test here costs more than the rest of the move put together.
    bsort = np.sort(bonds, axis=1)
    bond_key = bsort[:, 0].astype(np.int64) * m + bsort[:, 1]
    psort = np.sort(pairs, axis=1)
    pair_key = psort[:, 0].astype(np.int64) * m + psort[:, 1]
    keep = ~np.isin(pair_key, bond_key)
    if exclude_second and keep.any():
        # von Alfthan, Kuronen & Kaski PRB 68, 073203 (2003) sum the repulsion
        # over pairs that are "neither nearest nor second nearest neighbors"
        # (their released code, galfthan/bomc src/bondList.c:nonNnNgbrs, does
        # the same).  Second neighbours are the intra-polyhedral X-X contacts
        # -- O-O across a SiO4 tetrahedron sits at ~2.63 A, right on the 2.6 A
        # cutoff -- so including them would fight the bonded angular term
        # instead of preventing overlap.
        adj = [[] for _ in range(m)]
        for a, b in bsort:
            adj[a].append(b); adj[b].append(a)
        second = set()
        for c in range(m):
            nb = adj[c]
            for u in range(len(nb)):
                for v in range(u + 1, len(nb)):
                    second.add((min(nb[u], nb[v]), max(nb[u], nb[v])))
        if second:
            skey = np.fromiter((a * m + b for a, b in second),
                               dtype=np.int64, count=len(second))
            keep &= ~np.isin(pair_key, skey)
    return pairs[keep]


try:                                     # optional compiled inner loop
    from .kernel import (patch_ef as _patch_ef_jit,
                         relax_fire as _relax_fire_jit,
                         patch_ef_nb as _patch_ef_nb_jit,
                         relax_fire_nb as _relax_fire_nb_jit)
except Exception:                        # pragma: no cover
    # numba is a hard dependency of tricor, so this branch is a safety
    # net rather than a supported configuration; the pure-numpy paths
    # below stay correct, just slower.
    _patch_ef_jit = None
    _relax_fire_jit = None
    _patch_ef_nb_jit = None
    _relax_fire_nb_jit = None


def patch_energy_forces(pos: np.ndarray, bonds: np.ndarray,
                        angles: np.ndarray, L: float, pot: Keating,
                        want_forces: bool = True, ka_ang=None, off_ang=None,
                        nb_pairs: np.ndarray | None = None,
                        r_rep: float = 3.0, k_rep: float = 3.0,
                        r_core: float = 2.0, k_core: float = 30.0,
                        rep_form: str = "harmonic"):
    """Keating energy/forces on a compact patch.

    Uses np.bincount rather than np.add.at for the scatter-add: add.at takes an
    unbuffered slow path and dominates the runtime of the inner loop.
    """
    m = len(pos)
    # Compiled fast path for the PUBLISHED energy (no non-bonded terms, which
    # default off).  A 50-atom patch is microseconds of real arithmetic but
    # ~125 numpy calls of dispatch overhead; profiling showed this routine at
    # 70% of the whole run, 62 calls per attempted move.  Verified against the
    # numpy path to 2e-13 in energy and 5e-15 in force over 25 patches.
    # The excluded-volume variant is compiled too, so chemistries with Z >= 6
    # -- where the angular term is degenerate and the non-bonded term is
    # therefore mandatory -- are no longer locked out of the fast path.
    # Verified: identical to patch_ef bit-for-bit at k_rep = k_core = 0, and
    # to the numpy branch below at 6e-12 in E / 1e-14 in F with them on.
    _use_nb = not (k_rep == 0.0 and k_core == 0.0
                   and (nb_pairs is None or len(nb_pairs) == 0))
    if (_patch_ef_jit is not None
            and (not _use_nb or _patch_ef_nb_jit is not None)):
        na = len(angles)
        ka_a = (np.full(na, pot.ka) if ka_ang is None
                else np.ascontiguousarray(np.broadcast_to(ka_ang, (na,)),
                                          dtype=np.float64))
        off_a = (np.full(na, pot.angle_offset) if off_ang is None
                 else np.ascontiguousarray(np.broadcast_to(off_ang, (na,)),
                                           dtype=np.float64))
        Lv = np.ascontiguousarray(np.broadcast_to(
            np.asarray(L, dtype=np.float64).reshape(-1), (3,)))
        _pos_c = np.ascontiguousarray(pos, dtype=np.float64)
        _b_c = np.ascontiguousarray(bonds, dtype=np.int64)
        _a_c = np.ascontiguousarray(angles, dtype=np.int64)
        if _use_nb:
            _nb_c = np.ascontiguousarray(
                nb_pairs if nb_pairs is not None else np.zeros((0, 2)),
                dtype=np.int64).reshape(-1, 2)
            E, F = _patch_ef_nb_jit(_pos_c, _b_c, _a_c, Lv, float(pot.kb),
                                    ka_a, float(pot.d2), off_a,
                                    bool(want_forces), _nb_c, float(r_rep),
                                    float(k_rep), float(r_core),
                                    float(k_core), rep_form == "quartic")
        else:
            E, F = _patch_ef_jit(_pos_c, _b_c, _a_c, Lv, float(pot.kb), ka_a,
                                 float(pot.d2), off_a, bool(want_forces))
        return E, (F if want_forces else None)

    rij = mic(pos[bonds[:, 1]] - pos[bonds[:, 0]], L)
    s = np.einsum("ij,ij->i", rij, rij)
    db = s - pot.d2
    E = pot.kb * float(db @ db)

    # Keating's bond term (r^2 - d^2)^2 is FINITE at r = 0 -- collapsing a bond
    # to zero length costs only kb*d^4 ~ 3 eV.  That is affordable at the
    # temperatures needed to randomise the network, and it happens: hot runs
    # produced bonded pairs at 0.49 A.  Keating was only ever used near
    # equilibrium, where this never arises.  Add an explicit short-range core
    # so bonds cannot pass through each other at any temperature.
    core = None
    rb = np.sqrt(np.maximum(s, 1e-24))
    cmask = rb < r_core
    if cmask.any():
        dc = r_core - rb[cmask]
        E += k_core * float(dc @ dc)
        core = (cmask, dc, rb[cmask])

    ci, aj, bk = angles[:, 0], angles[:, 1], angles[:, 2]
    rA = mic(pos[aj] - pos[ci], L)
    rB = mic(pos[bk] - pos[ci], L)
    # Per-angle equilibrium offset when the angular term follows the CENTRAL
    # atom's coordination (angle_mode="ideal_per_z"); a scalar otherwise.
    off = pot.angle_offset if off_ang is None else off_ang
    u = np.einsum("ij,ij->i", rA, rB) + off
    kang = pot.ka if ka_ang is None else ka_ang
    E += float(np.sum(kang * u * u))

    nb_act = None
    if nb_pairs is not None and len(nb_pairs):
        dv = mic(pos[nb_pairs[:, 1]] - pos[nb_pairs[:, 0]], L)
        rr = np.linalg.norm(dv, axis=1)
        act = rr < r_rep
        if act.any():
            if rep_form == "quartic":
                # von Alfthan PRB 68, 073203 (2003) Eq. (3.17):
                #     R_ij = 1/2 B (r_r^2 - r_ij^2)^2   for r_ij <= r_r
                # B = 0.8 eV/A^-4, r_r = 2.6 A (thesis Table 3.3, and
                # repulsive_k / repulsive_r0 in the reference bomc code).
                # Smoother at the cutoff than the half-harmonic below: both
                # E and dE/dr vanish there, and dE/dr -> 0 quadratically.
                dr = r_rep * r_rep - rr[act] * rr[act]
                E += 0.5 * k_rep * float(dr @ dr)
            else:
                dr = r_rep - rr[act]
                E += k_rep * float(dr @ dr)
            nb_act = (nb_pairs[act], dv[act], rr[act], dr)

    if not want_forces:
        return E, None

    fb = (4.0 * pot.kb * db)[:, None] * rij
    g = (2.0 * kang * u)[:, None]
    fA, fB, fC = -g * rB, -g * rA, g * (rA + rB)
    idx_parts = [bonds[:, 1], bonds[:, 0], aj, bk, ci]
    val_parts = [-fb, fb, fA, fB, fC]
    if core is not None:
        cmask, dc, rc = core
        fc = (2.0 * k_core * dc / np.maximum(rc, 1e-9))[:, None] * rij[cmask]
        idx_parts += [bonds[cmask, 1], bonds[cmask, 0]]
        val_parts += [fc, -fc]
    if nb_act is not None:
        pr, dv, rr, dr = nb_act
        if rep_form == "quartic":
            # E = 1/2 B (r_r^2-r^2)^2  ->  F_j = -dE/dr_j = 2 B (r_r^2-r^2) dv
            fv = (2.0 * k_rep * dr)[:, None] * dv
        else:
            fv = (2.0 * k_rep * dr / np.maximum(rr, 1e-9))[:, None] * dv
        idx_parts += [pr[:, 1], pr[:, 0]]
        val_parts += [fv, -fv]
    idx = np.concatenate(idx_parts)
    val = np.concatenate(val_parts, axis=0)
    F = np.empty((m, 3))
    for k in range(3):
        F[:, k] = np.bincount(idx, weights=val[:, k], minlength=m)
    return E, F


def decorate_ax2(cation_net: "Network", d_ax: float = 1.61,
                 cation: int = 14, anion: int = 8,
                 axa_angle_deg: float = 144.0, fit: str = "mean",
                 pot: Keating | None = None) -> "Network":
    """Turn an ANNEALED cation CRN into an AX2 network by bond decoration.

    This is the correct route to a mixed-valency (bipartite) network, and it
    exists because the WWW transposition CANNOT be run on the decorated network
    directly: the move forms A-D and B-C, and in an A-X network B and C are
    both X while D is A, so the new bonds are A-A and X-X by construction
    (measured: 200/200 proposals produce homonuclear bonds).  Coordination
    survives; chemistry does not.

    So the topology is generated where the validated tetravalent engine works --
    on the A sublattice -- and the anions are placed afterwards, one per A-A
    bond.  The resulting network is bipartite by construction, every A has the
    A-sublattice's coordination and every X has exactly 2.  This is also the
    standard construction for a-SiO2 models.

    The A sublattice is rescaled so that placing X on each bond gives the
    requested A-X bond length at the requested A-X-A angle:
        d(A-A) = 2 * d_ax * sin(axa_angle/2)
    Silica: d_ax = 1.61 A and A-X-A ~ 144 deg give d(A-A) ~ 3.06 A.  The anion
    is placed off the A-A midpoint by the amount that angle requires, in a
    direction chosen per bond, so the initial network already has a physical
    bridging angle rather than a linear one.
    """
    if not (0.0 < axa_angle_deg < 180.0):
        raise ValueError(f"axa_angle_deg must be in (0,180), got {axa_angle_deg}")
    if d_ax <= 0.0:
        raise ValueError(f"d_ax must be positive, got {d_ax}")
    L0 = cation_net.L
    bonds = cation_net.bond_array()
    if len(bonds) == 0:
        raise ValueError("cation network has no bonds")
    pos0 = cation_net.pos
    dvec = mic(pos0[bonds[:, 1]] - pos0[bonds[:, 0]], L0)
    dl_now = np.linalg.norm(dvec, axis=1)
    theta = np.radians(axa_angle_deg)
    d_aa_target = 2.0 * d_ax * np.sin(theta / 2.0)
    # Scale on the MEAN bond and any bond above the mean by more than
    # 1/sin(theta/2) cannot carry an anion at d_ax from both ends -- it gets
    # dumped on the midpoint at exactly 180 deg.  On a real relaxed CRN that
    # was 13.9% of anions.  Scale on the longest bond instead so every anion is
    # placeable; the mean bridging angle then sits slightly below the target,
    # which is harmless because the angle relaxes anyway and a spread is
    # physical, whereas a spike of exactly-180 deg bridges is an artefact.
    # fit="mean" vs "max" is a genuine trade-off, measured on a relaxed
    # 512-atom cation CRN:
    #     mean -> Si-O 1.614+/-0.015, Si-O-Si 147.6+/-18.0, 13.9% pinned at
    #             exactly 180 deg (the un-placeable long bonds), O-O min 2.04
    #     max  -> Si-O 1.610+/-0.000, Si-O-Si 124.7+/-10.5, 0% at 180 deg,
    #             but O-O min 1.62 (the cell shrinks) and the mean bridge is
    #             ~20 deg below experiment
    # "mean" is the default because it is the one validated END TO END: through
    # the MACE handoff it yields Si-O-Si 145.4+/-14.4 against an experimental
    # 144+/-15, i.e. MACE removes the 180-deg spike.  "max" is offered for a
    # decoration that must be artefact-free before any relaxation.
    # The deeper cause of the trade-off: the cation network is relaxed at
    # SILICON's bond length, so its A-A spread is wider than a rigid A-X-A
    # linkage can absorb.  Relaxing the cation network at the silica A-A
    # distance (~3.06 A) would remove the tension at source.
    if fit == "max":
        scale = (2.0 * d_ax * 0.999) / float(dl_now.max())
    elif fit == "mean":
        scale = d_aa_target / float(dl_now.mean())
    else:
        raise ValueError(f"fit must be 'max' or 'mean', got {fit!r}")

    L = L0 * scale
    posA = pos0 * scale
    dvec = mic(posA[bonds[:, 1]] - posA[bonds[:, 0]], L)
    dlen = np.linalg.norm(dvec, axis=1)
    mid = posA[bonds[:, 0]] + 0.5 * dvec
    # offset needed to bend A-X-A to the target angle
    # A bond longer than 2*d_ax cannot carry an anion at distance d_ax from
    # both ends.  Clamping to zero silently dumped 13-44% of anions on the
    # midpoint with an exactly-180.000 deg bridge and an over-long A-X bond.
    # Surface it instead.
    too_long = dlen >= 2.0 * d_ax
    if too_long.any():
        import warnings
        warnings.warn(
            f"decorate_ax2: {int(too_long.sum())}/{len(dlen)} cation-cation "
            f"bonds exceed 2*d_ax ({2*d_ax:.3f} A); longest {dlen.max():.3f} A. "
            "Those anions sit on the midpoint with a 180 deg bridge. The "
            "cation network is probably not relaxed at this bond length.",
            RuntimeWarning, stacklevel=2)
    off = np.sqrt(np.maximum(d_ax**2 - (dlen / 2.0) ** 2, 0.0))

    # Direction of the off-midpoint displacement.  A random perpendicular is
    # not good enough: anions on adjacent bonds can be pushed toward each other
    # and give unphysical X-X contacts (measured 1.16 A, where silica is
    # ~2.6 A).  The anions that can actually collide are exactly those on bonds
    # SHARING AN ATOM with this one, which the bond list already gives -- so
    # choose greedily against that exact set rather than searching space (an
    # earlier spatial version rebuilt its tree only periodically and, working
    # from a stale neighbour set, made the contacts worse).
    u = dvec / np.maximum(dlen[:, None], 1e-12)
    helper = np.tile(np.array([0.0, 0.0, 1.0]), (len(u), 1))
    flip = np.abs(np.einsum("ij,ij->i", u, helper)) > 0.9
    helper[flip] = np.array([1.0, 0.0, 0.0])
    e1 = helper - (np.einsum("ij,ij->i", helper, u))[:, None] * u
    e1 /= np.maximum(np.linalg.norm(e1, axis=1, keepdims=True), 1e-12)
    e2 = np.cross(u, e1)

    bonds_of_atom: dict[int, list[int]] = {}
    for k, (a, b_) in enumerate(bonds):
        bonds_of_atom.setdefault(int(a), []).append(k)
        bonds_of_atom.setdefault(int(b_), []).append(k)

    n_cand = 16
    phis = np.linspace(0.0, 2.0 * np.pi, n_cand, endpoint=False)
    cos_p, sin_p = np.cos(phis)[:, None], np.sin(phis)[:, None]
    posX = np.zeros((len(bonds), 3))
    # Sweep repeatedly rather than once.  A single pass leaves the first bond
    # choosing blind (nothing placed yet) and every later choice frozen against
    # a partly-empty set; re-choosing against ALL neighbours converges the worst
    # anion-anion contact substantially (1.33 -> ~1.9 A).
    done = np.zeros(len(bonds), dtype=bool)
    n_sweep = 8
    for sweep in range(n_sweep):
        moved = 0.0
        for k in range(len(bonds)):
            cands = mid[k] + off[k] * (cos_p * e1[k] + sin_p * e2[k])
            nbr = [m for a in bonds[k] for m in bonds_of_atom[int(a)]
                   if m != k and done[m]]
            if nbr:
                dv = mic(cands[:, None, :] - posX[nbr][None, :, :], L)
                dmin_c = np.sqrt(np.einsum("ijk,ijk->ij", dv, dv)).min(axis=1)
                pick = int(np.argmax(dmin_c))
            else:
                pick = 0
            new = cands[pick]
            if done[k]:
                moved = max(moved, float(np.linalg.norm(new - posX[k])))
            posX[k] = new
            done[k] = True
        if sweep and moved < 1e-9:
            break

    nA = cation_net.n
    nX = len(bonds)
    zA = int(cation_net.z.max())
    zmax = max(zA, 2)
    nbrs = np.full((nA + nX, zmax), -1, dtype=np.int64)
    fill = np.zeros(nA + nX, dtype=np.int64)
    for k, (i, j) in enumerate(bonds):
        x = nA + k
        nbrs[i, fill[i]] = x; fill[i] += 1
        nbrs[j, fill[j]] = x; fill[j] += 1
        nbrs[x, 0] = i
        nbrs[x, 1] = j
    species = np.concatenate([np.full(nA, cation), np.full(nX, anion)])
    allpos = np.mod(np.concatenate([posA, posX], axis=0), L)
    allpos = np.where(allpos >= L, allpos - L, allpos)
    net = Network(allpos, nbrs, L,
                  pot or Keating(d=d_ax, angle_mode="max_repulsion"), species)
    net._assert_symmetric()
    if not np.all(net.z[nA:] == 2):
        raise RuntimeError("decoration gave an anion with Z != 2")
    if not np.array_equal(net.z[:nA], cation_net.z):
        raise RuntimeError("decoration changed cation coordination")
    return net


# ── spatial cell list (spatial locality, not topological) ────────────────────


class CellList:
    """Uniform grid over the box with incremental updates.

    WHY THIS IS NEEDED.  The patch used for relaxation is built from BOND hops,
    so it only ever sees atoms that are topologically nearby.  Two atoms that
    are far apart along the network but adjacent in SPACE never share a patch,
    and their overlap is invisible to the patch repulsion.  At low disorder this
    never happens; at ~1 accepted switch per atom the network folds onto itself
    and it happens catastrophically -- measured: 1429 non-bonded pairs below
    2.0 A, the closest 4+ bond-hops apart, energy running to 155 eV/atom.

    A cell list restores SPATIAL locality without reintroducing an O(N) term:
    lookups touch 27 cells regardless of system size, and only the ~100 atoms a
    move actually displaces need reinserting.
    """

    def __init__(self, net, r_cut: float):
        # Per-axis grid: the cell is orthorhombic in general, so the number of
        # divisions and the spacing differ by axis.
        self.L = np.asarray(net.L, dtype=np.float64).reshape(-1)
        self.nc = np.maximum(1, np.floor(self.L / r_cut).astype(np.int64))
        self.h = self.L / self.nc
        self.cells: dict[int, set] = {}
        self.of = np.full(net.n, -1, dtype=np.int64)
        self.rebuild(net)

    def _key(self, p) -> int:
        c = np.mod(np.floor(np.mod(p, self.L) / self.h).astype(np.int64),
                   self.nc)
        return int((c[0] * self.nc[1] + c[1]) * self.nc[2] + c[2])

    def rebuild(self, net) -> None:
        self.cells = {}
        pos = np.mod(net.pos, self.L)
        c = np.mod((pos / self.h).astype(np.int64), self.nc)
        keys = (c[:, 0] * self.nc[1] + c[:, 1]) * self.nc[2] + c[:, 2]
        self.of = keys.astype(np.int64)
        for i, k in enumerate(keys):
            self.cells.setdefault(int(k), set()).add(int(i))

    def move(self, idx: np.ndarray, new_pos: np.ndarray) -> None:
        for a, p in zip(np.atleast_1d(idx), np.atleast_2d(new_pos)):
            a = int(a)
            k = self._key(p)
            if k == self.of[a]:
                continue
            old = self.cells.get(int(self.of[a]))
            if old is not None:
                old.discard(a)
            self.cells.setdefault(k, set()).add(a)
            self.of[a] = k

    def candidates(self, idx: np.ndarray) -> np.ndarray:
        """Atoms in the cells covering the patch's bounding box, plus a margin.

        Querying 27 cells per patch atom costs ~2400 dict lookups per move and
        dominated the runtime.  The patch is spatially compact, so one bounding
        box over it covers the same atoms in far fewer lookups, and the cell
        keys are computed vectorised.
        """
        idx = np.atleast_1d(idx)
        ncx, ncy, ncz = (int(v) for v in self.nc)
        k = self.of[idx]
        cx = k // (ncy * ncz)
        cy = (k // ncz) % ncy
        cz = k % ncz
        # unwrap the box: a patch may straddle the periodic seam, so work in
        # offsets from the first atom rather than absolute cell indices.
        def span(c, nc):
            d = np.mod(c - c[0] + nc // 2, nc) - nc // 2
            return int(d.min()) - 1, int(d.max()) + 1, int(c[0])
        x0, x1, ax = span(cx, ncx)
        y0, y1, ay = span(cy, ncy)
        z0, z1, az = span(cz, ncz)
        gx = np.mod(np.arange(x0, x1 + 1) + ax, ncx)
        gy = np.mod(np.arange(y0, y1 + 1) + ay, ncy)
        gz = np.mod(np.arange(z0, z1 + 1) + az, ncz)
        keys = ((gx[:, None, None] * ncy + gy[None, :, None]) * ncz
                + gz[None, None, :]).ravel()
        out: list = []
        get = self.cells.get
        for kk in keys:
            c = get(int(kk))
            if c:
                out.extend(c)
        return np.unique(np.asarray(out, dtype=np.int64)) if out else \
            np.zeros(0, dtype=np.int64)


# ── the WWW move ─────────────────────────────────────────────────────────────


def _replace(nbrs: np.ndarray, atom: int, old: int, new: int) -> None:
    """Swap one entry of an atom's neighbour row; coordination is unchanged."""
    row = nbrs[atom]
    row[np.where(row == old)[0][0]] = new


@dataclass
class MCStats:
    attempted: int = 0
    accepted: int = 0
    early_rejected: int = 0
    relax_rejected: int = 0
    invalid: int = 0


class WWWAnnealer:
    """Metropolis WWW bond switching with local relaxation + early rejection.

    frozen_atoms: boolean mask; bonds with BOTH ends frozen are never proposed
    and frozen atoms never move during relaxation (Nakhmanson-style locked
    grains).  Coordinates of frozen atoms still enter energy terms, so grains
    strain-couple to the matrix elastically once `release_frozen` is called.
    """

    def __init__(self, net: Network, T: float = 0.25, n_shell: int = 3,
                 relax_iters: int = 60, relax_step: float = 0.12,
                 early_check_from: int = 6, c_f: float | None = None,
                 frozen_atoms: np.ndarray | None = None,
                 forbid_triangles: bool = False,
                 r_rep: float | None = None, k_rep: float = 0.0,
                 r_core: float | None = None, k_core: float = 0.0,
                 rep_form: str = "harmonic",
                 move: str = "auto",
                 e_wrong: float = 0.0,
                 move_weights: tuple = (0.5, 0.4, 0.1),
                 warmup_attempts: int = 400, cf_safety: float = 15.0,
                 allow_multivalent: bool = False,
                 rng: np.random.Generator | None = None):
        # ── MULTIVALENT / NON-TETRAHEDRAL PATHWAY IS BLOCKED ───────────────
        # Policy: this repo is MOSAIC-FIRST.  Bond switching is offered as an
        # OPTIONAL pathway, and only for the case where it is known to be the
        # better model: a uniformly 4-coordinated (octet, Z = 4) network such
        # as a-Si, a-Ge or a-C, where coordination is fixed by octet closure
        # and is therefore phase-invariant, so a move set that conserves
        # coordination exactly is the right one.
        #
        # WHY THE OTHER MODES ARE BLOCKED, measured on B2O3 (Z_B = 3, Z_O = 2)
        # with angle_mode="crystal" and crystal-measured theta_0 -- i.e. the
        # BEST-configured non-tetrahedral run available, not a strawman:
        #
        #     pre-anneal +EV polish     min_nonbonded 1.840 A   n<1.2A     0
        #     pre-anneal pure requench  min_nonbonded 0.154 A   n<1.2A    66
        #     after switching           min_nonbonded 0.280 A   n<1.2A   135
        #     quench_topology_exact:    0 switches, budget exhausted
        #
        # Topological coordination was conserved perfectly (z_B = 3, z_O = 2
        # throughout, as bond switching must) while the GEOMETRY collapsed; the
        # geometric CN inflated 3.00 -> 3.43 (and to 8.75 at 4x T_melt) purely
        # from overlapping atoms.  The low Keating energy that run reported
        # (0.0014 eV/atom) is an artifact of scoring a collapsed structure with
        # a bonded-only potential.
        #
        # The cause is structural, not a tuning error.  Keating constrains only
        # bonded distances and angles, so a non-tetrahedral network has nothing
        # holding unbonded atoms apart, and the excluded-volume term that would
        # supply it is unusable in the switching MC:
        #   * k_rep = 0 -> no excluded volume -> collapse (above);
        #   * k_rep > 0 -> patch_with_scenery() passes build_patch()'s
        #     `bonds_loc`, which omits scenery-scenery bonds, so real bonds get
        #     scored as non-bonded overlaps (r_rep always exceeds d).  See also
        #     the k_rep note below: with it on, the quench stops converging
        #     (final |F|max 6.6 vs 8.4e-7 for pure Keating).
        # A uniformly tetrahedral network escapes this because angle_mode
        # "tetrahedral" has a genuine angular minimum at 109.47 deg that keeps
        # neighbours apart geometrically -- which is why www_voronoi.py needs no
        # excluded-volume term and reaches coord 4.0000, min_nonbonded 1.136 A,
        # E/atom 0.548.
        #
        # For mixed-coordination materials use the MOSAIC pathway instead; it
        # is validated against experiment on five packing-ruled oxides.
        #
        # allow_multivalent=True overrides the block.  It exists so the path is
        # not permanently unreachable if the excluded-volume defect is ever
        # fixed -- not as a supported option.  Nothing in the production
        # pipeline sets it, and a structure produced with it set has not been
        # validated against anything.
        if net.pot.angle_mode != "tetrahedral" and not allow_multivalent:
            raise NotImplementedError(
                f"WWW bond switching is blocked for angle_mode="
                f"{net.pot.angle_mode!r}: the non-tetrahedral pathway collapses "
                f"geometrically (measured on B2O3: min non-bonded 0.154-0.280 A, "
                f"135 pairs < 1.2 A, while topological coordination stayed "
                f"exact). Keating holds only bonded terms, and the excluded-"
                f"volume term needed here is broken in the switching MC "
                f"(k_rep=0 collapses; k_rep>0 miscounts scenery bonds as "
                f"overlaps). Use the MOSAIC pathway for mixed-coordination "
                f"materials, or pass allow_multivalent=True to override for "
                f"diagnostic work.")

        # DEFAULT OFF, and for the same reason the construction-time 3-ring
        # ban was removed: with the T=0 structured quench in place, a strained
        # motif is removed because removing it LOWERS THE ENERGY, not because
        # it was forbidden.  Imposing it instead bakes in a chemistry-specific
        # assumption -- the bipartite form of this ban forbids EDGE-SHARING,
        # which is strained in tetrahedral silica but is exactly how rutile
        # TiO2 and corundum Al2O3 are built.  Measured, leaving it on rejected
        # 537 of 600 proposals for TiO2 and the quench could accept nothing at
        # all.  Available as a diagnostic, not as a default.
        self.forbid_triangles = forbid_triangles
        # k_rep and k_core default to ZERO, i.e. the plain Keating energy of
        # Barkema & Mousseau and of Hemmann et al.  They are NOT part of either
        # published method, and switching them on has two measured costs:
        #   * the quench stops converging -- final |F|max 6.6 with them versus
        #     8.4e-7 with pure Keating, so "quench" silently returns a
        #     non-stationary structure;
        #   * it minimises a different energy than the published protocols, so
        #     nothing measured against them is comparable.
        # They exist for ONE regime the published work never enters: melting a
        # crystal, where Keating's (r^2-d^2)^2 is finite at r=0 and bonds
        # collapse.  A random-start protocol annealed at 0.25 eV never gets
        # there.  So they are opt-in for the melt protocol rather than a
        # standing modification of the potential.
        self.r_rep = net.suggest_r_rep() if r_rep is None else r_rep
        self.k_rep = k_rep
        # "quartic" selects von Alfthan, Kuronen & Kaski PRB 68, 073203 (2003)
        # Eq. (3.17), the published non-bonded term for bond-switched SILICA,
        # with their second-neighbour exclusion.  Their B = 0.8 eV/A^-4 and
        # r_r = 2.6 A; the reference implementation (galfthan/bomc) ships this
        # ON for SiO2.  Bonded-only Keating stays the default because that IS
        # the published protocol for ELEMENTAL a-Si (Barkema & Mousseau PRB 62,
        # 4985) -- measured, our a-Si has zero non-bonded contacts inside
        # 2.6 A, so there is nothing there for a repulsion to do.
        self.rep_form = rep_form
        # short-range core: keep bonds from passing through each other, scaled
        # to the bond length rather than fixed at a Si-specific 2.0 A.
        self.r_core = (0.85 * net.pot.d) if r_core is None else r_core
        self.k_core = k_core
        self.net = net
        self.T = T                      # eV
        self.n_shell = n_shell
        self.relax_iters = relax_iters
        self.relax_step = relax_step    # A, per-iteration displacement cap
        self.early_check_from = early_check_from
        # E_relaxed ~ E - c_f |F|^2 (harmonic estimate); c_f is calibrated on
        # the fly from observed (E_drop / |F|^2) ratios if not given.
        self.c_f = c_f if c_f is not None else 0.02
        self._cf_fixed = c_f is not None
        self._cf_samples: list[float] = []
        # Early rejection is only sound if c_f OVER-estimates the energy drop
        # still available at the check iteration; under-estimating it rejects
        # moves a full relaxation would have accepted, which silently biases
        # the chain toward a colder ensemble.  Measured: an under-calibrated
        # c_f threw away ~80% of acceptable moves.  So: disable the shortcut
        # for a warmup, calibrate from what full relaxations actually do, and
        # keep a safety factor on top.
        self.warmup_attempts = warmup_attempts
        # Multiplier on the observed worst-case drop ratio.  It is not a
        # free knob: the sample it is estimated from is CENSORED, because a
        # move that gets aborted never contributes a measurement, so any bound
        # read off that sample underestimates the true worst case and the
        # estimator kills genuinely downhill moves.  Measured on a-Si at
        # T = 0.25, fraction of would-be ACCEPTS discarded: 39% at 1x, 1.7% at
        # 5x, 0% at 15x and beyond -- while aborts fall only 2013 -> 1911 and
        # runtime is flat at 23.1 s.  Being generous here is therefore free,
        # and being tight silently biases the walk.
        self.cf_safety = cf_safety
        # Accept EITHER a boolean mask of length n OR an index array, and say
        # so loudly if given something else.  Silently calling .astype(bool) on
        # an index array yields a wrong-length, almost-all-True mask that
        # freezes the wrong atoms with no error at all -- and an empty index
        # array yields a zero-length mask that only fails later, deep in
        # propose().
        if frozen_atoms is None:
            self.frozen = np.zeros(net.n, dtype=bool)
        else:
            fa = np.asarray(frozen_atoms)
            if fa.dtype == bool:
                if fa.shape != (net.n,):
                    raise ValueError(
                        f"frozen_atoms boolean mask has length {fa.shape}, "
                        f"expected ({net.n},)")
                self.frozen = fa.copy()
            else:
                if fa.size and (fa.max() >= net.n or fa.min() < 0):
                    raise ValueError("frozen_atoms index out of range")
                self.frozen = np.zeros(net.n, dtype=bool)
                self.frozen[fa.astype(np.int64)] = True
        self.rng = rng or np.random.default_rng(0)
        # "auto": use the species-preserving swap whenever the network is
        # bipartite, since the WWW transposition provably cannot be used there.
        if move == "auto":
            b = net.bond_array()
            hetero = bool(len(b)) and bool(
                np.all(net.species[b[:, 0]] != net.species[b[:, 1]]))
            move = "bipartite" if hetero else "www"
        self.move = move
        # Chemical-defect PENALTY, per homonuclear ("wrong") bond, in eV.
        # Mousseau & Barkema do not forbid wrong bonds -- they give them an
        # energy cost and study the resulting defect density, which is the
        # object of their paper.  Restricting binary runs to the
        # species-preserving swap (their move (b)) made "zero wrong bonds" true
        # by construction, which silently answers their research question
        # instead of measuring it, and rules out the chemical disorder that
        # real binary amorphous solids have.  e_wrong = 0 reproduces the
        # unpenalised limit; a large value approaches the hard ban.
        self.e_wrong = float(e_wrong)
        # Relative rates of moves (a) transposition, (b) neighbour exchange,
        # (d) species-identity swap, used when move == "mixed".
        self.move_weights = np.asarray(move_weights, dtype=float)
        self.move_weights /= self.move_weights.sum()
        # Which species may occupy the A1/A2 role of a bipartite swap.  In an
        # AX2 network a swap with an ANION in that role merely exchanges which
        # anion carries a given cation-cation link: the cation-sublattice edge
        # set is unchanged, so the move is null where it counts.  Restrict the
        # role to the species with the higher mean coordination (both, if tied,
        # as in zinc blende).
        self._role_mask = np.ones(net.n, dtype=bool)
        if move in ("bipartite", "mixed"):
            sp_u = np.unique(net.species)
            if len(sp_u) == 2:
                mz = [net.z[net.species == u].mean() for u in sp_u]
                if abs(mz[0] - mz[1]) > 1e-9:
                    keep = sp_u[int(np.argmax(mz))]
                    self._role_mask = net.species == keep
        self._role_idx = np.where(self._role_mask)[0]
        self._lookup = np.full(net.n, -1, dtype=np.int64)
        self._cells = CellList(net, self.r_rep) if self.k_rep > 0 else None
        self.stats = MCStats()
        # Early-rejection audit (off by default): see _relax_local.
        self._audit_early = False
        self._audit_abort_it = None
        self._audit = {"abort_reject": 0, "abort_accept": 0,
                       "keep_reject": 0, "keep_accept": 0}

    # -- move proposal ------------------------------------------------------

    def propose(self):
        """Pick a random A-B bond and C, D per WWW; reject invalid picks."""
        net = self.net
        rng = self.rng
        for _ in range(64):
            A = int(rng.integers(net.n))
            if net.z[A] < 2:
                continue
            B = int(net.nbrs[A][rng.integers(net.z[A])])
            if net.z[B] < 2:
                continue
            C = int(net.nbrs[A][rng.integers(net.z[A])])
            D = int(net.nbrs[B][rng.integers(net.z[B])])
            # validity: C != B, D != A, C != D, and no pre-existing A-D or
            # B-C bond (the move would otherwise duplicate a bond).
            if C == B or D == A or C == D:
                continue
            if D in net.neighbours(A) or C in net.neighbours(B):
                continue
            # Locked grains (Nakhmanson): the set of bonds internal to the
            # frozen region must be EXACTLY invariant.  The move deletes A-C
            # and B-D and creates A-D and B-C, so all four must be barred from
            # being frozen-frozen -- guarding only the deletions would still
            # let a switch CREATE a new intra-grain bond and silently alter
            # the grain's topology.
            fz = self.frozen
            if (fz[A] and fz[C]) or (fz[B] and fz[D]) or \
               (fz[A] and fz[D]) or (fz[B] and fz[C]):
                continue
            if self.forbid_triangles and self._makes_triangle(A, B, C, D):
                continue
            return A, B, C, D
        return None

    def _makes_triangle(self, A, B, C, D) -> bool:
        """Would the new A-D / B-C bonds close a 3-membered ring?

        Post-switch neighbourhoods are known analytically, so this is four
        set operations on 4-element sets -- no graph search.
        """
        net = self.net
        nA = (set(net.neighbours(A).tolist()) - {C}) | {D}
        nB = (set(net.neighbours(B).tolist()) - {D}) | {C}
        nC = (set(net.neighbours(C).tolist()) - {A}) | {B}
        nD = (set(net.neighbours(D).tolist()) - {B}) | {A}
        # a triangle on bond A-D <=> A and D share a neighbour afterwards
        if (nA - {D}) & (nD - {A}):
            return True
        if (nB - {C}) & (nC - {B}):
            return True
        return False

    def _apply(self, A, B, C, D):
        nb = self.net.nbrs
        _replace(nb, A, C, D)
        _replace(nb, B, D, C)
        _replace(nb, C, A, B)
        _replace(nb, D, B, A)

    def _revert(self, A, B, C, D):
        nb = self.net.nbrs
        _replace(nb, A, D, C)
        _replace(nb, B, C, D)
        _replace(nb, C, B, A)
        _replace(nb, D, A, B)

    def propose_bipartite(self):
        """Species-preserving swap for bipartite (A-X) networks.

        The WWW transposition forms A-D and B-C, which in a bipartite network
        are homonuclear by construction -- it cannot be used here at all.  This
        move instead takes two A-X bonds and exchanges their partners:

            A1-X1 and A2-X2   ->   A1-X2 and A2-X1

        Every one of the four atoms keeps its coordination AND its bonding
        partner species, so both the coordination statistics and the bipartite
        structure are exactly invariant.  A2 is drawn from A1's second shell so
        the four atoms stay topologically close and the move remains local.
        """
        net = self.net
        rng = self.rng
        sp = net.species
        for _ in range(64):
            A1 = int(self._role_idx[rng.integers(len(self._role_idx))])
            if net.z[A1] < 1:
                continue
            n1 = net.neighbours(A1)
            X1 = int(n1[rng.integers(len(n1))])
            # A2: a same-species atom two hops from A1, reached via some other
            # bridging anion, so the swap stays local.
            via = n1[n1 != X1]
            if len(via) == 0:
                continue
            Xb = int(via[rng.integers(len(via))])
            cands = net.neighbours(Xb)
            cands = cands[cands != A1]
            if len(cands) == 0:
                continue
            A2 = int(cands[rng.integers(len(cands))])
            if sp[A2] != sp[A1] or net.z[A2] < 1 or not self._role_mask[A2]:
                continue
            n2 = net.neighbours(A2)
            n2 = n2[n2 != X1]
            if len(n2) == 0:
                continue
            X2 = int(n2[rng.integers(len(n2))])
            if X2 == X1 or A2 == A1 or X2 == A1 or X1 == A2:
                continue
            # the new bonds must not already exist
            if X2 in net.neighbours(A1) or X1 in net.neighbours(A2):
                continue
            fz = self.frozen
            if (fz[A1] and fz[X1]) or (fz[A2] and fz[X2]) or \
               (fz[A1] and fz[X2]) or (fz[A2] and fz[X1]):
                continue
            # NOTE: forbid_triangles is VACUOUS here.  A bipartite graph has
            # no odd cycles at all, so a 3-ring can never form -- verified,
            # every full-network ring is even (4/6/8).  The physically strained
            # configuration in a ceramic is instead EDGE-SHARING: two cations
            # bridged by TWO anions, i.e. a 2-ring of the cation sublattice.
            # In silica these are rare and highly strained, and nothing was
            # barring them (23 present after a short run).  So the triangle ban
            # maps onto an edge-sharing ban here.
            if self.forbid_triangles and self._makes_edge_sharing(
                    A1, X1, A2, X2):
                continue
            return A1, X1, A2, X2
        return None

    def _makes_edge_sharing(self, A1, X1, A2, X2) -> bool:
        """Would the swap leave a cation bridged to another by TWO anions?

        Post-swap neighbourhoods are known analytically, so this is a small
        multiset count rather than a graph search.
        """
        net = self.net
        post = {A1: (set(net.neighbours(A1).tolist()) - {X1}) | {X2},
                A2: (set(net.neighbours(A2).tolist()) - {X2}) | {X1}}
        changed = {X1: (set(net.neighbours(X1).tolist()) - {A1}) | {A2},
                   X2: (set(net.neighbours(X2).tolist()) - {A2}) | {A1}}
        for A, anions in post.items():
            seen: dict[int, int] = {}
            for x in anions:
                partners = changed.get(int(x))
                if partners is None:
                    partners = set(net.neighbours(int(x)).tolist())
                for c in partners:
                    c = int(c)
                    if c == A:
                        continue
                    seen[c] = seen.get(c, 0) + 1
                    if seen[c] >= 2:
                        return True
        return False

    def _apply_bipartite(self, A1, X1, A2, X2):
        nb = self.net.nbrs
        _replace(nb, A1, X1, X2)
        _replace(nb, X2, A2, A1)
        _replace(nb, A2, X2, X1)
        _replace(nb, X1, A1, A2)

    def _revert_bipartite(self, A1, X1, A2, X2):
        nb = self.net.nbrs
        _replace(nb, A1, X2, X1)
        _replace(nb, X1, A2, A1)
        _replace(nb, A2, X1, X2)
        _replace(nb, X2, A1, A2)

    # -- local relaxation with early rejection ------------------------------

    def patch_with_scenery(self, moving: np.ndarray):
        """Patch plus its spatial scenery, and the non-bonded pair list.

        Both step() and _relax_local MUST score the identical term set.  When
        the scenery pairs were added only inside _relax_local, e_before and
        e_after came from different functionals -- an offset of +199 eV mean on
        diamond, which swamps T*ln(u) and drove the acceptance rate to exactly
        zero at every temperature.
        """
        net = self.net
        gidx, pos_loc, bonds_loc, angles_loc, movable_loc = build_patch(
            net, moving, self._lookup)
        nb_pairs = (patch_nonbonded(pos_loc, bonds_loc, self.r_rep, net.L,
                                    exclude_second=self.rep_form == "quartic")
                    if self.k_rep > 0 else None)
        if self.k_rep > 0 and self._cells is not None:
            cand = self._cells.candidates(gidx[movable_loc])
            extra = np.setdiff1d(cand, gidx)
            if len(extra):
                ref = net.pos[gidx[0]]
                pos_ext = ref + mic(net.pos[extra] - ref, net.L)
                base = len(pos_loc)
                pos_loc = np.concatenate([pos_loc, pos_ext], axis=0)
                gidx = np.concatenate([gidx, extra])
                dmat = mic(pos_loc[:base, None, :] - pos_loc[None, base:, :],
                           net.L)
                rr2 = np.einsum("ijk,ijk->ij", dmat, dmat)
                ii, jj = np.nonzero(rr2 < self.r_rep * self.r_rep)
                if len(ii):
                    ext = np.stack([ii, base + jj], axis=1)
                    nb_pairs = (ext if nb_pairs is None or len(nb_pairs) == 0
                                else np.concatenate([nb_pairs, ext], axis=0))
        return gidx, pos_loc, bonds_loc, angles_loc, movable_loc, nb_pairs

    def ka_for(self, gidx, angles_loc):
        """Per-angle bend constant, keyed on the CENTRE atom's valency."""
        if not self.net.pot.beta_by_valency:
            return None
        centres = gidx[angles_loc[:, 0]]
        return self.net.pot.ka_for_z(self.net.z[centres])

    def off_for(self, gidx, angles_loc):
        """Per-angle equilibrium offset, keyed on the CENTRE atom's valency.

        Returns None unless angle_mode is "ideal_per_z", in which case every
        angle is measured against the ideal angle for ITS centre's coordination
        rather than against one global equilibrium.
        """
        mode = self.net.pot.angle_mode
        if mode not in ("ideal_per_z", "crystal"):
            return None
        centres = gidx[angles_loc[:, 0]]
        if mode == "crystal":
            return self.net.pot.species_offset(self.net.species[centres])
        return self.net.pot.ideal_offset(self.net.z[centres])

    def patch_terms(self, gidx: np.ndarray, moving: np.ndarray):
        """Bond/angle term sets for a FIXED atom set, at the current topology.

        Used to re-derive only the topology-dependent part of a patch after a
        switch, leaving the atom set and the non-bonded pair list untouched so
        that e_before and e_after are strictly comparable.
        """
        bonds_g, angles_g = cluster_terms(self.net, moving)
        lookup = self._lookup
        lookup[gidx] = np.arange(len(gidx))
        keep_b = (lookup[bonds_g] >= 0).all(axis=1)
        keep_a = (lookup[angles_g] >= 0).all(axis=1)
        bonds_loc = lookup[bonds_g[keep_b]]
        angles_loc = lookup[angles_g[keep_a]]
        lookup[gidx] = -1
        return bonds_loc, angles_loc

    def _relax_local(self, moving: np.ndarray, e_target: float | None,
                     patch=None):
        """Damped steepest descent on the cluster.

        Returns (E_terms_final, positions_backup, aborted).  Positions are
        modified in place; the caller reverts with the backup on rejection.
        e_target=None disables early rejection (used at T=0 quench steps).
        """
        net = self.net
        if patch is None:
            (gidx, pos_loc, bonds_loc, angles_loc, movable_loc,
             nb_pairs) = self.patch_with_scenery(moving)
        else:
            gidx, pos_loc, _, _, movable_loc, nb_pairs = patch
            pos_loc = pos_loc.copy()
            bonds_loc, angles_loc = self.patch_terms(gidx, moving)
        free_loc = movable_loc[~self.frozen[gidx[movable_loc]]]
        movable = gidx[free_loc]
        backup = net.pos[movable].copy()
        E = None
        e0 = fsq0 = None
        e_chk = fsq_chk = None
        warm = self.stats.attempted < self.warmup_attempts
        # FIRE on the patch, not capped steepest descent.  Steepest descent
        # leaves each patch partially unrelaxed; with no global relaxation to
        # dissipate it (fully-local scheme), that residue accumulates over
        # thousands of accepted moves and the total energy runs away -- seen
        # directly as E/atom reaching 45 eV at high acceptance.
        v = np.zeros((len(free_loc), 3))
        dt, alpha, n_pos = 0.05, 0.10, 0
        DT_MAX, F_INC, F_DEC, F_ALPHA, N_MIN = 0.15, 1.1, 0.5, 0.99, 5

        # Fully compiled relaxation for the published energy.  Wiring only the
        # force evaluation into numba still left 62 Python round-trips per
        # attempted move, and that became the dominant cost once the
        # arithmetic was fast (7.93 -> 3.24 ms/attempt, Amdahl-capped).  The
        # c_f calibration samples are not collected on this path, so it is used
        # only once c_f is fixed -- during warm-up the numpy path still runs.
        _use_nb = not (self.k_rep == 0.0 and self.k_core == 0.0
                       and (nb_pairs is None or len(nb_pairs) == 0))
        if (_relax_fire_jit is not None and not warm and not self._audit_early
                and (not _use_nb or _relax_fire_nb_jit is not None)
                and len(free_loc)):
            na = len(angles_loc)
            ka_l = self.ka_for(gidx, angles_loc)
            off_l = self.off_for(gidx, angles_loc)
            ka_a = (np.full(na, net.pot.ka) if ka_l is None
                    else np.ascontiguousarray(np.broadcast_to(ka_l, (na,)),
                                              dtype=np.float64))
            off_a = (np.full(na, net.pot.angle_offset) if off_l is None
                     else np.ascontiguousarray(np.broadcast_to(off_l, (na,)),
                                               dtype=np.float64))
            Lv = np.ascontiguousarray(np.broadcast_to(
                np.asarray(net.L, dtype=np.float64).reshape(-1), (3,)))
            pl = np.ascontiguousarray(pos_loc, dtype=np.float64)
            _fire_args = (
                pl, np.ascontiguousarray(bonds_loc, dtype=np.int64),
                np.ascontiguousarray(angles_loc, dtype=np.int64),
                np.ascontiguousarray(free_loc, dtype=np.int64), Lv,
                float(net.pot.kb), ka_a, float(net.pot.d2), off_a,
                int(self.relax_iters),
                (-1.0e300 if e_target is None else float(e_target)),
                int(self.early_check_from), float(self.c_f),
                0.05, DT_MAX, 0.10, F_INC, F_DEC, F_ALPHA, N_MIN, 1e-3,
                float(self.relax_step))
            if _use_nb:
                _nb_c = np.ascontiguousarray(
                    nb_pairs if nb_pairs is not None else np.zeros((0, 2)),
                    dtype=np.int64).reshape(-1, 2)
                E, _it, aborted = _relax_fire_nb_jit(
                    *_fire_args, _nb_c, float(self.r_rep), float(self.k_rep),
                    float(self.r_core), float(self.k_core),
                    getattr(self, "rep_form", "harmonic") == "quartic")
            else:
                E, _it, aborted = _relax_fire_jit(*_fire_args)
            net.pos[movable] = pl[free_loc]
            if self._cells is not None:
                self._cells.move(movable, net.pos[movable])
            return E, (movable, backup), aborted

        for it in range(self.relax_iters):
            E, F = patch_energy_forces(pos_loc, bonds_loc, angles_loc,
                                       net.L, net.pot,
                                       ka_ang=self.ka_for(gidx, angles_loc),
                                       off_ang=self.off_for(gidx, angles_loc),
                                       nb_pairs=nb_pairs,
                                       r_rep=self.r_rep, k_rep=self.k_rep,
                                       rep_form=self.rep_form,
                                       r_core=self.r_core,
                                       k_core=self.k_core)
            f = F[free_loc]
            if it == 0:
                e0 = E
                fsq0 = float(np.einsum("ij,ij->", f, f)) if len(f) else 0.0
            fmax = float(np.abs(f).max()) if len(f) else 0.0
            if fmax < 1e-3:
                break
            if it == self.early_check_from:
                e_chk = E
                fsq_chk = float(np.einsum("ij,ij->", f, f))
            if (e_target is not None and not warm
                    and it >= self.early_check_from):
                est = E - self.c_f * float(np.einsum("ij,ij->", f, f))
                if est > e_target:
                    if self._audit_early:
                        # Diagnostic mode: note that the estimator WOULD have
                        # aborted here, but keep relaxing so the same move also
                        # gets its true verdict.  Comparing diverging runs
                        # cannot separate estimator bias from trajectory drift;
                        # scoring one move both ways can.
                        if self._audit_abort_it is None:
                            self._audit_abort_it = it
                    else:
                        net.pos[movable] = pos_loc[free_loc]
                        if self._cells is not None:
                            self._cells.move(movable, net.pos[movable])
                        return E, (movable, backup), True
            P = float(np.einsum("ij,ij->", f, v))
            if P > 0:
                nf = np.linalg.norm(f)
                v = ((1 - alpha) * v
                     + alpha * np.linalg.norm(v) * f / max(nf, 1e-12))
                n_pos += 1
                if n_pos > N_MIN:
                    dt = min(dt * F_INC, DT_MAX)
                    alpha *= F_ALPHA
            else:
                v[:] = 0.0
                dt *= F_DEC
                alpha = 0.10
                n_pos = 0
            v += dt * f
            dx = dt * v
            nrm = np.linalg.norm(dx, axis=1, keepdims=True)
            dx = np.where(nrm > self.relax_step,
                          dx * (self.relax_step / np.maximum(nrm, 1e-12)), dx)
            pos_loc[free_loc] += dx
        net.pos[movable] = pos_loc[free_loc]
        if self._cells is not None:
            self._cells.move(movable, net.pos[movable])
        E, _ = patch_energy_forces(pos_loc, bonds_loc, angles_loc,
                                   net.L, net.pot, want_forces=False,
                                   ka_ang=self.ka_for(gidx, angles_loc),
                                       off_ang=self.off_for(gidx, angles_loc),
                                   nb_pairs=nb_pairs, r_rep=self.r_rep,
                                   k_rep=self.k_rep, rep_form=self.rep_form,
                                   r_core=self.r_core,
                                   k_core=self.k_core)
        if (not self._cf_fixed and e_chk is not None and fsq_chk
                and e_chk - E > 1e-9):
            # ratio of the drop STILL AVAILABLE at the check iteration to the
            # force-squared there -- exactly the quantity the estimator needs
            # to bound from above.
            self._cf_samples.append((e_chk - E) / fsq_chk)
            if len(self._cf_samples) >= 40:
                # c_f must UPPER-bound the drop still available per |F|^2, so
                # that E - c_f|F|^2 lower-bounds the relaxed energy and an abort
                # means the move is confidently bad.  A percentile times a
                # safety constant does not bound anything: the tail beyond p99
                # is exactly what leaks through, and measured, cf_safety = 3
                # discarded 12-19% of the moves full relaxation would have
                # ACCEPTED.  The running maximum is the bound the estimator
                # actually needs, and it costs nothing -- aborts fall only from
                # 1949 to 1912 (all of them confidently bad), with identical
                # runtime, because the savings come from clearly-uphill moves
                # rather than from marginal ones.
                self._cf_samples = self._cf_samples[-2000:]
                self.c_f = self.cf_safety * float(max(self._cf_samples))
        return E, (movable, backup), False

    # -- one Metropolis step ------------------------------------------------

    def step(self) -> bool:
        kind = self.pick_kind()
        return self.try_move(self.propose_for(kind), kind)

    def pick_kind(self) -> str:
        """Which of Mousseau & Barkema's moves to attempt this step."""
        if self.move != "mixed":
            return self.move
        return ("www", "bipartite", "identity")[
            int(self.rng.choice(3, p=self.move_weights))]

    def propose_for(self, kind: str):
        if kind == "bipartite":
            return self.propose_bipartite()
        if kind == "identity":
            return self.propose_identity()
        return self.propose()

    def propose_identity(self):
        """Move (d): two atoms exchange species identity, topology fixed.

        Only defined between atoms of EQUAL coordination -- otherwise the swap
        would change each sublattice's coordination, which is a property of the
        chemistry rather than of the disorder.  In Mousseau & Barkema's A-B
        network both species are four-fold so the move is always available; in
        an AX2 network it never is, and the mixture renormalises accordingly.
        """
        net = self.net
        for _ in range(64):
            i = int(self.rng.integers(net.n))
            j = int(self.rng.integers(net.n))
            if i == j or net.species[i] == net.species[j]:
                continue
            if net.z[i] != net.z[j]:
                continue
            if self.frozen[i] or self.frozen[j]:
                continue
            return i, j
        return None

    def wrong_bond_delta(self, pick, kind: str) -> int:
        """Change in the number of homonuclear bonds produced by a move."""
        sp = self.net.species
        if kind == "identity":
            i, j = pick
            sp2 = sp.copy()
            sp2[i], sp2[j] = sp[j], sp[i]
            def around(sv, a):
                return int(np.sum(sv[self.net.neighbours(a)] == sv[a]))
            return (around(sp2, i) + around(sp2, j)
                    - around(sp, i) - around(sp, j))
        if kind == "bipartite":
            return 0          # species-preserving by construction
        A, B, C, D = (int(x) for x in pick)
        return (int(sp[A] == sp[D]) + int(sp[B] == sp[C])
                - int(sp[A] == sp[C]) - int(sp[B] == sp[D]))

    def try_move(self, pick, kind: str | None = None) -> bool:
        """Run the Metropolis test on an ALREADY-CHOSEN transposition.

        step() draws a move at random; the structured quench instead walks a
        fixed enumeration of every transposition, so the accept test lives here
        and both callers share exactly one code path.
        """
        self.stats.attempted += 1
        # "mixed" is a POLICY, not a move kind; step() resolves it via
        # pick_kind().  Falling through with kind="mixed" would silently take
        # the www branch, applying a transposition to a bipartite network.
        kind = self.move if kind is None else kind
        if kind == "mixed":
            kind = self.pick_kind()
        if pick is None or not self.is_valid(pick, kind):
            self.stats.invalid += 1
            return False
        if kind == "identity":
            return self._try_identity(pick)
        bipart = kind == "bipartite"
        A, B, C, D = pick

        moving = bond_hops(self.net, np.array([A, B, C, D]), self.n_shell)
        patch0 = self.patch_with_scenery(moving)
        g0, p0, b0, a0, _, nb0 = patch0
        e_before, _ = patch_energy_forces(p0, b0, a0, self.net.L,
                                          self.net.pot, want_forces=False,
                                          ka_ang=self.ka_for(g0, a0),
                                          off_ang=self.off_for(g0, a0),
                                          nb_pairs=nb0, r_rep=self.r_rep,
                                          k_rep=self.k_rep,
                                          rep_form=self.rep_form,
                                          r_core=self.r_core,
                                          k_core=self.k_core)
        # a-priori threshold (Barkema-Mousseau): draw BEFORE relaxing
        e_target = e_before - self.T * np.log(self.rng.random() + 1e-300)

        self._audit_abort_it = None
        (self._apply_bipartite if bipart else self._apply)(A, B, C, D)
        # SAME atom set and SAME non-bonded pair list as e_before; only the
        # bond/angle terms are re-derived for the new topology.
        e_after, (movable, backup), aborted = self._relax_local(
            moving, e_target, patch=patch0)

        e_after += self.e_wrong * self.wrong_bond_delta(pick, kind)
        if self._audit_early:
            would_abort = self._audit_abort_it is not None
            truly_ok = e_after <= e_target
            self._audit["abort_accept" if would_abort and truly_ok else
                        "abort_reject" if would_abort else
                        "keep_accept" if truly_ok else "keep_reject"] += 1
        if aborted or e_after > e_target:
            self.net.pos[movable] = backup
            if self._cells is not None:
                self._cells.move(movable, backup)
            (self._revert_bipartite if bipart else self._revert)(A, B, C, D)
            if aborted:
                self.stats.early_rejected += 1
            else:
                self.stats.relax_rejected += 1
            return False
        self.stats.accepted += 1
        return True

    # -- batched global quench ---------------------------------------------

    def nonbonded_pairs(self, r_rep: float):
        """Non-bonded pairs closer than r_rep, via one KD-tree query.

        The Keating potential constrains only BONDED distances and angles, so
        two atoms that are not bonded may sit arbitrarily close at zero energy
        cost.  WWW networks built this way develop ~1.6 A contacts, which is
        unphysical and forces the downstream MACE relaxation into large,
        network-damaging displacements.  A soft repulsion applied during the
        batched quench removes them at negligible cost.
        """
        from scipy.spatial import cKDTree
        L = self.net.L
        pos = np.mod(self.net.pos, L)
        pos = np.where(pos >= L, pos - L, pos)
        pairs = cKDTree(pos, boxsize=L).query_pairs(r_rep,
                                                    output_type="ndarray")
        if len(pairs) == 0:
            return pairs
        nb = self.net.nbrs
        keep = np.array([pairs[k, 1] not in nb[pairs[k, 0]]
                         for k in range(len(pairs))], dtype=bool)
        return pairs[keep]

    def _try_identity(self, pick) -> bool:
        """Accept/reject a species-identity swap (move (d)).

        Positions, bonds and coordinations are all unchanged, and the Keating
        terms depend on species only through the valency-dependent angular
        stiffness -- which cannot change here because the move is restricted to
        atoms of equal coordination.  The geometric energy is therefore exactly
        invariant and the Metropolis test reduces to the chemical-defect term,
        with no relaxation required.
        """
        i, j = int(pick[0]), int(pick[1])
        de = self.e_wrong * self.wrong_bond_delta((i, j), "identity")
        # A swap that changes nothing (de == 0, which is EVERY identity move
        # when e_wrong = 0) must not count as progress.  Accepting it clears the
        # quench's mark table, so `todo` never empties and exhaustion can never
        # be reached -- measured, 123/123 identity moves accepted with de
        # exactly 0 and quench_exhausted stuck False.  At T = 0 the descent
        # therefore requires a STRICT decrease; at finite T the Metropolis test
        # applies to uphill moves as before.
        if de >= 0.0:
            if self.T <= 0.0 or (de > 0.0 and
                                 self.rng.random() >= np.exp(-de / self.T)):
                self.stats.relax_rejected += 1
                return False
        sp = self.net.species
        sp[i], sp[j] = sp[j], sp[i]
        self.stats.accepted += 1
        return True

    # -- Barkema-Mousseau structured quench, exact stopping criterion ------

    def is_valid(self, pick, kind: str | None = None) -> bool:
        """Validity of a transposition, applying EXACTLY propose()'s rules.

        Factored out so the random walk and the systematic enumeration accept
        the same set of moves by construction rather than by inspection.
        """
        if pick is None:
            return False
        net = self.net
        fz = self.frozen
        kind = self.move if kind is None else kind
        if kind == "identity":
            i, j = int(pick[0]), int(pick[1])
            return (i != j and net.species[i] != net.species[j]
                    and net.z[i] == net.z[j]
                    and not fz[i] and not fz[j])
        A, B, C, D = (int(x) for x in pick)
        if kind == "bipartite":
            A1, X1, A2, X2 = A, B, C, D
            if A1 == A2 or X1 == X2:
                return False
            # With chemical defects allowed, the two role-A atoms may be bonded
            # to each other, so an atom can appear in both roles.  In a strictly
            # bipartite network that is impossible, which is why the swap could
            # assume it -- the exchange would otherwise create a self-bond.
            if X2 == A1 or X1 == A2:
                return False
            if net.species[A2] != net.species[A1] or not self._role_mask[A2]:
                return False
            if X1 not in net.neighbours(A1) or X2 not in net.neighbours(A2):
                return False
            if X2 in net.neighbours(A1) or X1 in net.neighbours(A2):
                return False
            if (fz[A1] and fz[X1]) or (fz[A2] and fz[X2]) or \
               (fz[A1] and fz[X2]) or (fz[A2] and fz[X1]):
                return False
            if self.forbid_triangles and self._makes_edge_sharing(
                    A1, X1, A2, X2):
                return False
            return True
        if C == B or D == A or C == D:
            return False
        if net.z[A] < 2 or net.z[B] < 2:
            return False
        if B not in net.neighbours(A):
            return False
        if C not in net.neighbours(A) or D not in net.neighbours(B):
            return False
        if D in net.neighbours(A) or C in net.neighbours(B):
            return False
        if (fz[A] and fz[C]) or (fz[B] and fz[D]) or \
           (fz[A] and fz[D]) or (fz[B] and fz[C]):
            return False
        if self.forbid_triangles and self._makes_triangle(A, B, C, D):
            return False
        return True

    def enumerate_moves(self) -> list:
        """Every transposition available in the current topology -- BM's "18N".

        For the tetravalent WWW move a transposition is (bond A-B, C a further
        neighbour of A, D a further neighbour of B), so the count is the sum
        over bonds of (z_A - 1)(z_B - 1).  With z = 4 and 2N bonds that is
        2N * 3 * 3 = 18N, reproducing their figure exactly, and the same
        expression generalises to any valency.
        """
        net = self.net
        out = []
        if self.move in ("bipartite", "mixed"):
            for A1 in self._role_idx:
                A1 = int(A1)
                n1 = net.neighbours(A1)
                for X1 in n1:
                    for Xb in n1:
                        if int(Xb) == int(X1):
                            continue
                        for A2 in net.neighbours(int(Xb)):
                            A2 = int(A2)
                            if A2 == A1 or not self._role_mask[A2]:
                                continue
                            for X2 in net.neighbours(A2):
                                if int(X2) == int(X1) or int(X2) == A1:
                                    continue
                                out.append(((A1, int(X1), A2, int(X2)),
                                            "bipartite"))
            if self.move == "bipartite":
                return out
        if self.move == "mixed":
            eq = np.where(self.net.species[:, None] != self.net.species[None, :])
            for i, j in zip(*eq):
                if i < j and net.z[i] == net.z[j]:
                    out.append(((int(i), int(j)), "identity"))
        for A, B in net.bond_array():
            A, B = int(A), int(B)
            for C in net.neighbours(A):
                if int(C) == B:
                    continue
                for D in net.neighbours(B):
                    if int(D) == A or int(D) == int(C):
                        continue
                    out.append(((A, B, int(C), int(D)), "www"))
        return out

    def quench_topology_exact(self, max_sweeps: int = 10_000,
                              attempt_budget: int | None = None,
                              drive: str = "random",
                              patience_per_atom: float = 25.0,
                              verify: bool = True,
                              log_every: int = 0) -> int:
        """T = 0 structured quench carrying Barkema-Mousseau's exact guarantee.

        Their criterion: at T = 0 a rejected transposition stays rejected until
        some other move is accepted, so one marks all 18N transpositions and
        stops when every one is marked.  That is what "no single bond switch
        lowers the energy" means, and it replaced a patience heuristic whose two
        invented constants traded the guarantee for a guess.

        The criterion says nothing about the ORDER in which moves are tried, and
        measurement says the order matters a great deal.  Driving the descent by
        the enumeration itself reaches a genuine but SHALLOWER minimum than
        driving it by random sampling -- from one starting network, 347 switches
        to E/atom 0.743 versus 444-545 switches to 0.50-0.60, reproduced across
        seeds.  Both endpoints are real: turning the random sampler loose on an
        enumeration-quenched network that had just reported exhausted found 0
        further switches, so the criterion is sound and the difference is purely
        which basin the descent lands in.

        So: drive with random sampling, which is cheap and finds deep minima,
        then VERIFY exhaustively with the estimator disabled and full relaxation
        on every one of the 18N moves.  If verification finds a switch, take it
        and drive again.  The loop ends only on a clean full sweep, so the
        guarantee is exactly BM's while the search order is the one that works.

        drive="sweep" restores enumeration-driven descent for comparison.
        attempt_budget bounds the work on large cells and leaves
        quench_exhausted False, so a bounded run reports that it is bounded
        rather than implying a guarantee it never established.
        """
        net = self.net
        T_save, self.T = self.T, 0.0
        ecf_save = self.early_check_from
        accepted = 0
        self.quench_exhausted = False
        stop_at = (None if attempt_budget is None
                   else self.stats.attempted + attempt_budget)
        try:
            for round_ in range(max_sweeps):
                # -- drive: cheap descent into a deep minimum ---------------
                if drive == "random":
                    patience = int(patience_per_atom * net.n)
                    since = 0
                    while since < patience:
                        if stop_at is not None and self.stats.attempted >= stop_at:
                            break
                        before = self.stats.accepted
                        self.step()
                        if self.stats.accepted > before:
                            accepted += 1
                            since = 0
                        else:
                            since += 1
                else:
                    moves = self.enumerate_moves()
                    for idx in self.rng.permutation(len(moves)):
                        if stop_at is not None and self.stats.attempted >= stop_at:
                            break
                        mv, kind = moves[int(idx)]
                        if self.try_move(mv, kind):
                            accepted += 1
                            break

                if stop_at is not None and self.stats.attempted >= stop_at:
                    break

                if not verify:
                    # Large cells only: one verification sweep is 18N full
                    # relaxations (~395k moves at 22k atoms), which is hours.
                    # Skipping it leaves quench_exhausted False, so the run
                    # reports that it never established the guarantee rather
                    # than implying it did.
                    break

                # -- verify: every transposition, estimator OFF -------------
                # The estimator is an efficiency device for the finite-T walk.
                # Here the whole claim is "no switch lowers the energy", so a
                # single discarded downhill move would turn the guarantee back
                # into a guess; verification pays full relaxation on every move.
                self.early_check_from = 1 << 30
                moves = self.enumerate_moves()
                found = False
                for idx in self.rng.permutation(len(moves)):
                    mv, kind = moves[int(idx)]
                    if self.try_move(mv, kind):
                        accepted += 1
                        found = True
                        break
                self.early_check_from = ecf_save
                if log_every and round_ % log_every == 0:
                    print(f"    round {round_:4d}  accepted={accepted}  "
                          f"verify_found={found}  E/at="
                          f"{net.energy_per_atom():.4f}", flush=True)
                if not found:
                    self.quench_exhausted = True
                    break
        finally:
            self.T = T_save
            self.early_check_from = ecf_save
        self.global_quench(iters=8000, fmax_stop=1e-6)
        return accepted

    def quench_topology(self, patience_per_atom: float = 60.0,
                        max_attempts_per_atom: float = 4000.0,
                        log_every: int = 0) -> int:
        """T=0 STRUCTURED QUENCH -- accept only switches that lower the energy,
        until no single bond switch can lower it further.

        Barkema & Mousseau: "at T=0 a rejected transposition stays rejected
        until some other move is accepted; mark all 18N possible transpositions
        and stop when all are marked", which "guarantees a state where no single
        bond switch lowers the energy".  Hemmann et al. end their temperature
        profile the same way: "Evolution occurs at T = 0 until no single bond
        switch can decrease the total strain energy."

        This is a TOPOLOGY-changing quench and it is a distinct step from the
        geometric relaxation in global_quench().  Omitting it is why a
        constructed network stalled around 30 deg with 12-15% three-membered
        rings: a pure geometric minimisation cannot remove a 3-ring, and a
        3-ring pins three angles near 60 deg.  No ad-hoc ring ban is needed once
        this runs -- the ring is removed because removing it lowers the energy.

        Convergence is by consecutive rejections rather than an explicit mark
        table: `patience_per_atom` * N failures in a row stands in for "all
        18N transpositions marked".
        """
        net = self.net
        T_save = self.T
        self.T = 0.0                       # accept only downhill moves
        patience = int(patience_per_atom * net.n)
        cap = int(max_attempts_per_atom * net.n)
        since_accept = 0
        accepted = 0
        start = self.stats.attempted
        try:
            while since_accept < patience and \
                    (self.stats.attempted - start) < cap:
                before = self.stats.accepted
                self.step()
                if self.stats.accepted > before:
                    accepted += 1
                    since_accept = 0
                    if log_every and accepted % log_every == 0:
                        self.global_quench(iters=600, fmax_stop=1e-4)
                        print(f"    quench_topology: {accepted} switches, "
                              f"E/atom={net.energy_per_atom():.3f}", flush=True)
                else:
                    since_accept += 1
        finally:
            self.T = T_save
        self.global_quench(iters=8000, fmax_stop=1e-6)
        return accepted

    def repair_close_pairs(self, r_close: float | None = None,
                           max_fix: int = 10000, gate: bool = False,
                           max_sweeps: int = 12, verbose: bool = False) -> int:
        """Rewire pairs that are close in space but not bonded.

        Barkema & Mousseau, on the first quench of a freshly constructed
        network: "sometimes a pair of atoms is closeby without being bonded; to
        eliminate such artefacts we replace a bond of each of these atoms by a
        bond between these atoms and another bond between their neighbors
        (conserving four-fold coordination)."

        Without this the first quench stalls in a local minimum at ~30 deg
        instead of reaching ~13 deg -- the geometry cannot improve because the
        TOPOLOGY is wrong, and only a rewiring can fix that.

        For P,Q close but unbonded: drop P-p and Q-q, add P-Q and p-q.  Every
        coordination is conserved.  Species are respected when the network is
        bipartite.

        gate=False is the published behaviour -- the repair is applied to every
        close pair, not only where it happens to lower the patch energy before
        relaxation.  The energy gate was added when the repair was destructive,
        but the destructiveness came from picking the FIRST valid neighbour pair
        rather than the best one; with the selection fixed, gating suppresses
        most of the step for no benefit (4-12 rewires gated vs 48-57 ungated,
        with the ungated run ending at LOWER energy, 2.03 vs 2.13 eV/atom).
        """
        from scipy.spatial import cKDTree
        net = self.net
        L = net.L
        r_close = r_close or 0.92 * net.pot.d
        fixed = 0
        for _sweep in range(max_sweeps):
            pos = np.mod(net.pos, L)
            pos = np.where(pos >= L, pos - L, pos)
            pairs = cKDTree(pos, boxsize=L).query_pairs(r_close,
                                                        output_type="ndarray")
            if len(pairs) == 0:
                break
            progressed = False
            for P, Q in pairs:
                P, Q = int(P), int(Q)
                if Q in net.neighbours(P):
                    continue
                fz = self.frozen
                if fz[P] and fz[Q]:
                    continue
                if self.move == "bipartite" and net.species[P] == net.species[Q]:
                    continue
                moving = bond_hops(net, np.array([P, Q]), self.n_shell)
                g0, s0, b0, a0, _, nb0 = self.patch_with_scenery(moving)
                kw = dict(nb_pairs=nb0, r_rep=self.r_rep, k_rep=self.k_rep,
                          rep_form=self.rep_form,
                          r_core=self.r_core, k_core=self.k_core)
                e_before, _ = patch_energy_forces(
                    s0, b0, a0, net.L, net.pot, want_forces=False, **kw)
                # WHICH bond of each atom to give up is not specified in the
                # source, and it is not a free choice: taking the first valid
                # neighbour pair can bond two atoms that are far apart, and the
                # strain that injects creates fresh close contacts, so the
                # repair cascades -- measured, 2058 rewires on 432 bonds with
                # the energy running 2.24 -> 5.62 eV/atom and the angle spread
                # getting WORSE (29.4 -> 33.2 deg).  Scoring every candidate and
                # keeping the best one is the only reading under which the step
                # does what it is for, namely remove an artefact rather than
                # introduce one.
                best = None
                for p_ in net.neighbours(P):
                    p_ = int(p_)
                    if p_ == Q:
                        continue
                    for q_ in net.neighbours(Q):
                        q_ = int(q_)
                        if q_ in (P, p_) or p_ in net.neighbours(q_):
                            continue
                        if self.move == "bipartite" and \
                                net.species[p_] == net.species[q_]:
                            continue
                        # Locked grains (Nakhmanson): the set of bonds internal
                        # to a frozen region must be EXACTLY invariant.  This
                        # rewiring DELETES P-p_ and Q-q_ and CREATES P-Q and
                        # p_-q_, so all four must be barred from being
                        # frozen-frozen -- guarding only the P/Q pair, as this
                        # did, still let the repair delete a bond inside a
                        # crystalline grain and silently alter its topology,
                        # which is the one thing the grain is there to preserve.
                        if (fz[P] and fz[p_]) or (fz[Q] and fz[q_]) or \
                           (fz[p_] and fz[q_]):
                            continue
                        _replace(net.nbrs, P, p_, Q)
                        _replace(net.nbrs, Q, q_, P)
                        _replace(net.nbrs, p_, P, q_)
                        _replace(net.nbrs, q_, Q, p_)
                        b1, a1 = self.patch_terms(g0, moving)
                        e_after, _ = patch_energy_forces(
                            s0, b1, a1, net.L, net.pot, want_forces=False, **kw)
                        _replace(net.nbrs, P, Q, p_)
                        _replace(net.nbrs, Q, P, q_)
                        _replace(net.nbrs, p_, q_, P)
                        _replace(net.nbrs, q_, p_, Q)
                        if best is None or e_after < best[2]:
                            best = (p_, q_, e_after)
                if best is None:
                    continue
                if gate and best[2] >= e_before:
                    continue
                p_, q_, _ = best
                _replace(net.nbrs, P, p_, Q)
                _replace(net.nbrs, Q, q_, P)
                _replace(net.nbrs, p_, P, q_)
                _replace(net.nbrs, q_, Q, p_)
                fixed += 1
                progressed = True
                if fixed >= max_fix:
                    break
            self.global_quench(iters=800, fmax_stop=1e-3)
            if verbose:
                print(f"      sweep {_sweep}: fixed={fixed} "
                      f"E/at={net.energy_per_atom():.3f}", flush=True)
            if not progressed or fixed >= max_fix:
                break
        return fixed

    def global_quench(self, iters: int = 600, fmax_stop: float = 2e-3,
                      r_rep: float | None = None, k_rep: float | None = None):
        """FIRE relaxation of ALL (non-frozen) atoms; O(N) but batched.

        FIRE rather than steepest descent: the Keating network has a wide
        curvature spread, and capped steepest descent stalls long before the
        true minimum, which would make a merely-unrelaxed network look
        genuinely strained.
        """
        net = self.net
        bonds = net.bond_array()
        angles = net.angle_array()
        # Must use the SAME derived cutoff as the MC, or the quench optimises
        # a different functional: r_rep=3.0 on AX2 penalises 1296 legitimate
        # second neighbours for 535 eV, larger than the whole Keating energy.
        r_rep = self.r_rep if r_rep is None else r_rep
        k_rep = self.k_rep if k_rep is None else k_rep
        free = np.where(~self.frozen)[0]
        nb_pairs = self.nonbonded_pairs(r_rep) if k_rep > 0 else None
        v = np.zeros((len(free), 3))
        dt, alpha, n_pos = 0.10, 0.10, 0
        DT_MAX, F_INC, F_DEC, F_ALPHA, N_MIN = 0.30, 1.1, 0.5, 0.99, 5
        E = None
        for it in range(iters):
            E, F = net.energy_forces(bonds, angles)
            if nb_pairs is not None:
                if it % 50 == 0:
                    nb_pairs = self.nonbonded_pairs(r_rep)
                if len(nb_pairs):
                    dv = mic(net.pos[nb_pairs[:, 1]] - net.pos[nb_pairs[:, 0]],
                             net.L)
                    r = np.linalg.norm(dv, axis=1)
                    act = r < r_rep
                    if act.any():
                        dr = r_rep - r[act]
                        E += k_rep * float(dr @ dr)
                        fr = 2.0 * k_rep * dr / np.maximum(r[act], 1e-9)
                        fv = fr[:, None] * dv[act]
                        np.add.at(F, nb_pairs[act, 1], fv)
                        np.add.at(F, nb_pairs[act, 0], -fv)
            f = F[free]
            fmax = float(np.abs(f).max())
            if fmax < fmax_stop:
                break
            P = float(np.einsum("ij,ij->", f, v))
            if P > 0:
                nf = np.linalg.norm(f)
                v = ((1 - alpha) * v
                     + alpha * np.linalg.norm(v) * f / max(nf, 1e-12))
                n_pos += 1
                if n_pos > N_MIN:
                    dt = min(dt * F_INC, DT_MAX)
                    alpha *= F_ALPHA
            else:
                v[:] = 0.0
                dt *= F_DEC
                alpha = 0.10
                n_pos = 0
            v += dt * f
            dx = dt * v
            norm = np.linalg.norm(dx, axis=1, keepdims=True)
            dx = np.where(norm > 0.20, dx * (0.20 / np.maximum(norm, 1e-12)),
                          dx)
            net.pos[free] += dx
        if self._cells is not None:
            self._cells.rebuild(net)
        return net.energy_forces(bonds, angles, want_forces=False)

    # -- driver -------------------------------------------------------------

    def run(self, n_accept_target: int, quench_every: int = 250,
            log_every: int = 500, max_attempts: int | None = None):
        """Anneal until n_accept_target switches are accepted."""
        t_att0, t_acc0 = self.stats.attempted, self.stats.accepted
        while (self.stats.accepted - t_acc0) < n_accept_target:
            if max_attempts is not None and \
               (self.stats.attempted - t_att0) >= max_attempts:
                break
            if self.step():
                acc = self.stats.accepted - t_acc0
                if quench_every and acc and acc % quench_every == 0:
                    self.global_quench()
            att = self.stats.attempted - t_att0
            if log_every and att and att % log_every == 0:
                s = self.stats
                yield dict(attempted=att, accepted=s.accepted - t_acc0,
                           early_rej=s.early_rejected,
                           e_per_atom=self.net.energy_per_atom())
        self.global_quench()

    # -- melt phase: unconditional switches (the disorder dial) -------------

    def melt(self, n_switches: int, relax_each: bool = True,
             log_every: int = 200):
        """Apply n_switches accepted-unconditionally random transpositions.

        This is the Wooten-Winer-Weaire randomisation phase and the pipeline's
        continuous disorder knob: switches/atom in [0, ~1+] interpolates from
        crystal to fully randomised topology.  Metropolis is skipped -- from a
        perfect crystal the first switches cost 2-3 eV each, so a thermal
        anneal at 0.25 eV would need ~1e5 attempts per accept just to leave
        the crystal.  Local relaxation still runs after each switch (without
        it, successive switches land on unrelaxed geometry and the network
        degrades).  Follow with polish() and/or global_quench().
        """
        quench_every = max(1, int(0.05 * self.net.n))  # ~0.05 sw/atom
        # melt() previously hardcoded the WWW move, so on a bipartite network
        # the pipeline's primary disorder dial silently created homonuclear
        # bonds (98/864 measured on a "silica" model) while coordination stayed
        # exactly 4/2, so no assertion fired. Dispatch like step() does.
        bipart = self.move == "bipartite"
        done = 0
        while done < n_switches:
            pick = self.propose_bipartite() if bipart else self.propose()
            if pick is None:
                continue
            A, B, C, D = pick
            (self._apply_bipartite if bipart else self._apply)(A, B, C, D)
            if relax_each:
                moving = bond_hops(self.net, np.array([A, B, C, D]),
                                   self.n_shell)
                self._relax_local(moving, e_target=None)
            done += 1
            if done % quench_every == 0:
                self.global_quench(iters=60)
            if log_every and done % log_every == 0:
                yield dict(switches=done,
                           e_per_atom=self.net.energy_per_atom())

    # -- T=0 polish: only energy-lowering switches --------------------------

    def polish(self, max_attempts: int, T: float = 0.03,
               log_every: int = 2000):
        """Low-temperature annealing pass after melt().

        Runs Metropolis at a small T (near-greedy) to remove the worst strain
        the melt phase left behind, without erasing the injected disorder.
        """
        T_save = self.T
        self.T = T
        att0, acc0 = self.stats.attempted, self.stats.accepted
        try:
            while (self.stats.attempted - att0) < max_attempts:
                self.step()
                att = self.stats.attempted - att0
                if log_every and att % log_every == 0:
                    yield dict(attempted=att,
                               accepted=self.stats.accepted - acc0,
                               e_per_atom=self.net.energy_per_atom())
        finally:
            self.T = T_save
        self.global_quench()
