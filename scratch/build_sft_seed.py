"""SFT relaxation seed: FCC Ni matrix (perfect lattice) + displaced Cr tetrahedron.

  - Ni matrix atoms stay exactly on parent FCC lattice sites.
  - Cr = every atom inside OR on the four {111} faces of the tetrahedron.
  - Only the Cr atoms are displaced, by the SFT dislocation displacement field
    (Volterra solid-angle superposition of the four bounding faults):
        base  face : Frank    partial  b = a/3<111>
        3 side faces: Shockley partial  b = a/6<112>
    so the four intrinsic stacking faults and the six a/6<110> stair-rod edges
    are seeded.  Hand this to your relaxer to settle into the SFT.
"""
import numpy as np, itertools
from ase.build import bulk
from ase.io import write
from scipy.spatial import cKDTree

a = 3.524          # Ni fcc lattice constant (Angstrom)
EDGE = 8           # SFT edge in nearest-neighbour spacings
N = 14             # supercell size (cells/side)

# ---- <hkl> families -------------------------------------------------------
def family(base):
    s = set()
    for p in itertools.permutations(base):
        for sg in itertools.product([1, -1], repeat=3):
            s.add(tuple(sg[i] * p[i] for i in range(3)))
    return [np.array(v, float) for v in s]
U112, U110 = family((1, 1, 2)), family((1, 1, 0))
def is_in(v, fam): return any(np.allclose(v, w, atol=1e-6) for w in fam)

# ---- tetrahedron (Thompson: edges = a/2<110>) -----------------------------
P = (a / 2) * np.array([[0, 0, 0], [EDGE, EDGE, 0],
                        [EDGE, 0, EDGE], [0, EDGE, EDGE]], float)
G = P.mean(0)
face_idx = [(1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2)]
def outward_normal(tri):
    n = np.cross(tri[1] - tri[0], tri[2] - tri[0]); n /= np.linalg.norm(n)
    return n if n @ (tri.mean(0) - G) > 0 else -n
normals = [outward_normal(P[list(f)]) for f in face_idx]

# ---- Burgers vectors (Frank base + 3 Shockley, forced by stair-rod edges) --
bF = -(a / 3.0) * np.round(normals[0] * np.sqrt(3))
faults = [bF, None, None, None]
for k in range(1, 4):
    for b_sh in [(a / 6) * v for v in U112 if abs(v @ normals[k]) < 1e-9]:
        if is_in((bF - b_sh) / (a / 6), U110):
            faults[k] = b_sh; break

def solid_angle(tri, R):
    A, B, C = tri[0] - R, tri[1] - R, tri[2] - R
    la = np.linalg.norm(A, axis=1); lb = np.linalg.norm(B, axis=1); lc = np.linalg.norm(C, axis=1)
    tr = np.einsum('ij,ij->i', A, np.cross(B, C))
    dn = (la*lb*lc + np.einsum('ij,ij->i', A, B)*lc
          + np.einsum('ij,ij->i', A, C)*lb + np.einsum('ij,ij->i', B, C)*la)
    return 2 * np.arctan2(tr, dn)

# ---- build perfect Ni lattice, centre tetra -------------------------------
ni = bulk("Ni", "fcc", a=a, cubic=True) * (N, N, N)
Pc = P + (np.array([N * a / 2] * 3) - G)
R0 = ni.get_positions()

# ---- Cr = inside OR on the boundary of the tetrahedron --------------------
M = np.vstack([Pc[1]-Pc[0], Pc[2]-Pc[0], Pc[3]-Pc[0]]).T
bc = np.linalg.solve(M, (R0 - Pc[0]).T).T
b0 = 1 - bc.sum(1)
tol = 1e-6
cr = (bc[:, 0] >= -tol) & (bc[:, 1] >= -tol) & (bc[:, 2] >= -tol) & (b0 >= -tol)

# ---- displace ONLY the Cr atoms by the SFT field --------------------------
u = np.zeros_like(R0)
uC = np.zeros((cr.sum(), 3))
for k in range(4):
    tri = Pc[list(face_idx[k])]
    if np.cross(tri[1]-tri[0], tri[2]-tri[0]) @ normals[k] < 0:
        tri = tri[[0, 2, 1]]
    uC += np.outer(solid_angle(tri, R0[cr]) / (4*np.pi), faults[k])
u[cr] = uC
R = R0 + u                                  # Ni rows have u=0 -> stay on lattice
ni.set_positions(R)
sym = np.array(ni.get_chemical_symbols()); sym[cr] = "Cr"; ni.set_chemical_symbols(sym.tolist())

# ---- report / checks ------------------------------------------------------
print("Burgers vectors (a units):")
for k in range(4):
    hkl = np.round(normals[k]*np.sqrt(3)).astype(int)
    kind = "Frank " if k == 0 else "Shock."
    print(f"  face ({hkl[0]:2d}{hkl[1]:2d}{hkl[2]:2d}) {kind} b = a*{np.round(faults[k]/a,4)}")
for i, j in [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]:
    d = np.round((faults[i]-faults[j])/(a/6)).astype(int)
    assert is_in(d, U110)
print("all six edges = a/6<110> stair-rods: OK")

ni_moved = np.abs(u[~cr]).max()
tree = cKDTree(R)
dCr, _ = tree.query(R[cr], k=2)
print(f"\ntotal atoms {len(ni)} | Cr (SFT, displaced) {cr.sum()} | Ni (matrix, on lattice) {(~cr).sum()}")
print(f"max Ni-matrix displacement : {ni_moved:.2e} A  (must be 0)")
print(f"max Cr displacement        : {np.linalg.norm(u[cr],axis=1).max():.3f} A")
print(f"min NN distance at Cr atoms : {dCr[:,1].min():.3f} A  (seed; you will relax)")

write("scratch/sft_seed_Ni_Cr.xyz", ni)
write("scratch/sft_seed_Ni_Cr.cif", ni)
print("\nwrote scratch/sft_seed_Ni_Cr.xyz and .cif")
