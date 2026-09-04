"""Continuous random networks by Wooten-Winer-Weaire bond switching.

This is the topological route to an amorphous network, complementary to
the Voronoi grain packing in :mod:`tricor.supercell`.  Where the packing
path places atoms and relaxes geometry, WWW rewires *bonds* under a
Keating potential, so it reaches the fully-coordinated, low-strain
continuous random networks that geometric packing cannot.

Tetravalent cations only.  A binary AX2 network (SiO2, GeO2, ...) is
built as a cation-only CRN and then decorated with bridging anions --
see :func:`decorate_ax2`.

Typical end-to-end use::

    from tricor.crn import Network, build_crn, calibrate_t_melt, decorate_ax2

    net = Network.from_diamond(3)         # 216 tetravalent cations
    t_melt = calibrate_t_melt(net)        # this network's melting scale
    ann, log = build_crn(net, t_max=t_melt)
    sio2 = decorate_ax2(ann.net)          # add the bridging O
    atoms = sio2.to_atoms()               # -> ase.Atoms

``t_max`` is an ENERGY, so its useful range is a material property: a
silicon bond switch costs ~1.3 eV, and acceptance is exactly 0% at
0.25 eV.  Measure the network's own melting scale with
:func:`calibrate_t_melt` and drive the anneal with the dimensionless
ratio ``t_max / t_melt`` -- that ratio is what transfers across
chemistries.  Below ~0.8 the network stays crystalline; above ~1.5 the
geometry starts to collapse while topological coordination still reads
as perfect.

The inner energy/force loops use the numba kernels in
:mod:`tricor.crn.kernel` and fall back to pure numpy if numba is
unavailable.
"""

from .www import (
    CellList,
    Keating,
    MCStats,
    Network,
    WWWAnnealer,
    bond_hops,
    build_patch,
    cluster_terms,
    decorate_ax2,
    mic,
    patch_energy_forces,
    patch_nonbonded,
)
from .protocol import (
    BuildLog,
    Schedule,
    build_crn,
    calibrate_t_melt,
    calibrate_t_melt_hemmann,
    clone,
    triangular_profile,
)
__all__ = [
    # network + potential
    "Network", "Keating", "mic", "CellList",
    # annealing
    "WWWAnnealer", "MCStats", "Schedule", "BuildLog",
    "build_crn", "clone", "triangular_profile",
    "calibrate_t_melt", "calibrate_t_melt_hemmann",
    # patch energetics
    "bond_hops", "cluster_terms", "build_patch",
    "patch_energy_forces", "patch_nonbonded",
    # anion decoration
    "decorate_ax2",
]
