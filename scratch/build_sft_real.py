"""Real (geometric) stacking-fault tetrahedron in FCC Ni.

An SFT is a regular tetrahedron with the four {111} planes as faces and the six
<110> edges as stair-rod dislocation lines.  It is a *vacancy* defect: one face
is a Frank fault (collapsed {111} layer, b = a/3<111>) and the three inclined
faces are Shockley intrinsic faults (b = a/6<112>).  Adjacent faces share
a/6<110> stair-rod dislocations at every edge (verified below).

The lattice is displaced by the Volterra solid-angle field
    u(r) = sum_k (b_k / 4pi) * Omega_k(r)
where Omega_k is the signed solid angle subtended by triangular face k.  This is
the standard *unrelaxed* geometric SFT: it carries genuine stacking faults on
all four faces and the full stair-rod network.  Dislocation cores remain
elastically strained (min NN ~1.9 A) as expected for an unrelaxed model; that
strain relaxes away in a low-stacking-fault-energy metal.

Atoms inside the tetrahedron are relabelled Cr (matrix = Ni).
"""
import numpy as np, itertools
from ase.build import bulk
from ase.io import write
from scipy.spatial import cKDTree

a = 3.524          # Ni fcc lattice constant (Angstrom)
EDGE = 8           # SFT edge length in nearest-neighbour spacings
N = 14             # supercell size (cells/side)
DELTA = 1.0        # Volterra cut offset (A) -> keeps cuts between {111} planes

# ---- <hkl> families -------------------------------------------------------
def family(base):
    s = set()
    for p in itertools.permutations(base):
        for sg in itertools.product([1, -1], repeat=3):
            s.add(tuple(sg[i] * p[i] for i in range(3)))
    return [np.array(v, float) for v in s]
U112, U110 = family((1, 1, 2)), family((1, 1, 0))
def is_in(v, fam):
    return any(np.allclose(v, w, atol=1e-6) for w in fam)

# ---- tetrahedron (Thompson: edges = a/2<110>) -----------------------------
P = (a / 2) * np.array([[0, 0, 0], [EDGE, EDGE, 0],
                        [EDGE, 0, EDGE], [0, EDGE, EDGE]], float)
G = P.mean(0)
face_idx = [(1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2)]   # opposite vertex k
def outward_normal(tri):
    n = np.cross(tri[1] - tri[0], tri[2] - tri[0]); n /= np.linalg.norm(n)
    return n if n @ (tri.mean(0) - G) > 0 else -n
normals = [outward_normal(P[list(f)]) for f in face_idx]

# ---- Burgers vectors: Frank base + 3 Shockley sides (forced by stair-rods) -
BASE = 0
bF = -(a / 3.0) * np.round(normals[BASE] * np.sqrt(3))     # a/3<111>, inward
faults = [bF, None, None, None]
for k in range(1, 4):
    for b_sh in [(a / 6) * v for v in U112 if abs(v @ normals[k]) < 1e-9]:
        if is_in((bF - b_sh) / (a / 6), U110):             # base-edge reaction
            faults[k] = b_sh
            break

# ---- solid-angle (Van Oosterom-Strackee) ----------------------------------
def solid_angle(tri, R):
    A, B, C = tri[0] - R, tri[1] - R, tri[2] - R
    la = np.linalg.norm(A, axis=1); lb = np.linalg.norm(B, axis=1); lc = np.linalg.norm(C, axis=1)
    tr = np.einsum('ij,ij->i', A, np.cross(B, C))
    dn = (la * lb * lc + np.einsum('ij,ij->i', A, B) * lc
          + np.einsum('ij,ij->i', A, C) * lb + np.einsum('ij,ij->i', B, C) * la)
    return 2 * np.arctan2(tr, dn)

# ---- build ----------------------------------------------------------------
ni = bulk("Ni", "fcc", a=a, cubic=True) * (N, N, N)
Pc = P + (np.array([N * a / 2] * 3) - G)      # centre tetra in the box
R0 = ni.get_positions()

u = np.zeros_like(R0)
for k in range(4):
    tri = Pc[list(face_idx[k])] + DELTA * normals[k]        # cut in interplane gap
    if np.cross(tri[1] - tri[0], tri[2] - tri[0]) @ normals[k] < 0:
        tri = tri[[0, 2, 1]]                                # RH normal -> outward
    u += np.outer(solid_angle(tri, R0) / (4 * np.pi), faults[k])
Rd = R0 + u
ni.set_positions(Rd)

# ---- label interior atoms Cr (tested on undisplaced lattice) ---------------
def inside(pts, verts, tol=1e-6):
    M = np.vstack([verts[1] - verts[0], verts[2] - verts[0], verts[3] - verts[0]]).T
    bc = np.linalg.solve(M, (pts - verts[0]).T).T
    b0 = 1 - bc.sum(1)
    return (bc[:, 0] >= -tol) & (bc[:, 1] >= -tol) & (bc[:, 2] >= -tol) & (b0 >= -tol)
mask = inside(R0, Pc)
sym = np.array(ni.get_chemical_symbols()); sym[mask] = "Cr"
ni.set_chemical_symbols(sym.tolist())

# ---- verification ---------------------------------------------------------
print("Burgers vectors (units of a):")
names = {0: "Frank  (base)", 1: "Shockley", 2: "Shockley", 3: "Shockley"}
for k in range(4):
    hkl = np.round(normals[k] * np.sqrt(3)).astype(int)
    print(f"  face {k} ({hkl[0]:2d}{hkl[1]:2d}{hkl[2]:2d}) {names[k]:14s}"
          f" b = a*({np.round(faults[k]/a,4)})  |b|={np.linalg.norm(faults[k]):.3f} A")
print("\nEdge dislocations (b_i - b_j):")
for i, j in [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]:
    d = (faults[i] - faults[j]) / (a / 6)
    print(f"  edge faces {i}-{j}: a/6*({np.round(d).astype(int)})  stair-rod a/6<110>? {is_in(np.round(d), U110)}")

dd, _ = cKDTree(Rd).query(Rd, k=2)
print(f"\ntotal atoms {len(ni)} | Cr (in SFT) {mask.sum()} | Ni {(~mask).sum()}")
print(f"SFT edge length  = {EDGE * a/np.sqrt(2):.2f} A")
print(f"max |u|          = {np.linalg.norm(u,axis=1).max():.3f} A")
print(f"min NN distance  = {dd[:,1].min():.3f} A  (perfect FCC {a/np.sqrt(2):.3f}); "
      f"{(dd[:,1]<2.2).sum()} core atoms <2.2 A")

write("scratch/sft_Ni_Cr_real.xyz", ni)
write("scratch/sft_Ni_Cr_real.cif", ni)
print("\nwrote scratch/sft_Ni_Cr_real.xyz and scratch/sft_Ni_Cr_real.cif")
