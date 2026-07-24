"""Build a stacking-fault-tetrahedron (SFT) atomic model.

Matrix: FCC Ni.  A regular tetrahedron with edges along <110> (the stair-rod
dislocation lines) and faces on {111} planes marks the SFT; atoms inside it are
relabelled Cr.  Edge length = 8 nearest-neighbour spacings.
"""
import numpy as np
from ase.build import bulk
from ase.io import write

a = 3.524                      # Ni fcc lattice constant (Angstrom)
EDGE = 8                       # SFT edge length in nearest-neighbour spacings
N = 10                         # supercell size (unit cells per side)

# --- FCC Ni supercell -------------------------------------------------------
ni = bulk("Ni", "fcc", a=a, cubic=True) * (N, N, N)
pos = ni.get_positions()

# --- SFT tetrahedron vertices (units of a/2, i.e. half-lattice) -------------
# Regular tetrahedron, all edges along <110>, all faces {111}.
V = np.array([[0, 0, 0],
              [EDGE, EDGE, 0],
              [EDGE, 0, EDGE],
              [0, EDGE, EDGE]], dtype=float) * (a / 2.0)

# Centre the tetrahedron in the supercell so it is fully embedded in Ni.
box_center = np.array([N * a / 2.0] * 3)
V += box_center - V.mean(axis=0)

# --- point-in-tetrahedron test ---------------------------------------------
def inside_tetra(p, verts, tol=1e-6):
    """True for points inside/on the tetrahedron defined by verts (4x3)."""
    def same_side(a_, b_, c_, d_, p_):
        n = np.cross(b_ - a_, c_ - a_)
        return (n @ (d_ - a_)) * (n @ (p_ - a_).T) >= -tol
    a_, b_, c_, d_ = verts
    return (same_side(a_, b_, c_, d_, p) &
            same_side(b_, c_, d_, a_, p) &
            same_side(c_, d_, a_, b_, p) &
            same_side(d_, a_, b_, c_, p))

mask = inside_tetra(pos, V)

symbols = np.array(ni.get_chemical_symbols())
symbols[mask] = "Cr"
ni.set_chemical_symbols(symbols.tolist())

print(f"total atoms : {len(ni)}")
print(f"Cr (in SFT) : {mask.sum()}")
print(f"Ni (matrix) : {(~mask).sum()}")
print("tetra vertices (Angstrom):")
for v in V:
    print("  ", np.round(v, 3))

write("scratch/sft_Ni_Cr.xyz", ni)
write("scratch/sft_Ni_Cr.cif", ni)
print("wrote scratch/sft_Ni_Cr.xyz and scratch/sft_Ni_Cr.cif")
