"""Build the Volterra SFT seed (correct Burgers topology) and relax with EMT.

Local (T=0) energy minimisation localises the smeared elastic strain into sharp
intrinsic stacking faults (HCP) on the four {111} faces + stair-rod dislocation
cores at the six edges, without crossing any barrier (so the SFT does not heal).
"""
import numpy as np, itertools
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.optimize import FIRE
from ase.constraints import FixAtoms
from ase.io import write

a = 3.524; EDGE = 8; N = 12; DELTA = 1.0

def fam(b):
    s = set()
    for p in itertools.permutations(b):
        for sg in itertools.product([1, -1], repeat=3):
            s.add(tuple(sg[i] * p[i] for i in range(3)))
    return [np.array(v, float) for v in s]
U112, U110 = fam((1, 1, 2)), fam((1, 1, 0))
def isin(v, f): return any(np.allclose(v, w, atol=1e-6) for w in f)

P = (a / 2) * np.array([[0, 0, 0], [EDGE, EDGE, 0], [EDGE, 0, EDGE], [0, EDGE, EDGE]], float)
G = P.mean(0)
fi = [(1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2)]
def onrm(t):
    n = np.cross(t[1] - t[0], t[2] - t[0]); n /= np.linalg.norm(n)
    return n if n @ (t.mean(0) - G) > 0 else -n
nm = [onrm(P[list(f)]) for f in fi]
nb = nm[0]; bF = -(a / 3.0) * np.round(nb * np.sqrt(3)); fa = [bF]
for k in range(1, 4):
    for b in [(a / 6) * v for v in U112 if abs(v @ nm[k]) < 1e-9]:
        if isin((bF - b) / (a / 6), U110): fa.append(b); break

def SA(t, R):
    A, B, C = t[0] - R, t[1] - R, t[2] - R
    la = np.linalg.norm(A, axis=1); lb = np.linalg.norm(B, axis=1); lc = np.linalg.norm(C, axis=1)
    tr = np.einsum('ij,ij->i', A, np.cross(B, C))
    dn = (la*lb*lc + np.einsum('ij,ij->i', A, B)*lc
          + np.einsum('ij,ij->i', A, C)*lb + np.einsum('ij,ij->i', B, C)*la)
    return 2 * np.arctan2(tr, dn)

ni = bulk("Ni", "fcc", a=a, cubic=True) * (N, N, N)
Pc = P + (np.array([N * a / 2] * 3) - G)
R0 = ni.get_positions()
u = np.zeros_like(R0)
for k in range(4):
    t = Pc[list(fi[k])] + DELTA * nm[k]
    if np.cross(t[1]-t[0], t[2]-t[0]) @ nm[k] < 0: t = t[[0, 2, 1]]
    u += np.outer(SA(t, R0) / (4*np.pi), fa[k])
ni.set_positions(R0 + u)

# tag interior atoms now (on the seed) so we can relabel after relaxation
def inside(pts, v, tol=1e-6):
    M = np.vstack([v[1]-v[0], v[2]-v[0], v[3]-v[0]]).T
    bc = np.linalg.solve(M, (pts - v[0]).T).T; b0 = 1 - bc.sum(1)
    return (bc[:, 0] >= -tol)&(bc[:, 1] >= -tol)&(bc[:, 2] >= -tol)&(b0 >= -tol)
interior = inside(R0, Pc)

# pin a thin outer shell so the box faces don't drift; relax everything else
c = np.array([N*a/2]*3)
shell = (np.abs(ni.get_positions() - c).max(1) > (N*a/2 - a))
ni.set_constraint(FixAtoms(mask=shell))
ni.calc = EMT()

print(f"atoms {len(ni)} | interior(Cr) {interior.sum()} | pinned shell {shell.sum()}")
opt = FIRE(ni, logfile="scratch/relax.log")
opt.run(fmax=0.08, steps=400)

np.save("scratch/relaxed_pos.npy", ni.get_positions())
np.save("scratch/interior_mask.npy", interior)
from ase.io import write as W
ni.set_constraint()
sym = np.array(ni.get_chemical_symbols()); sym[interior] = "Cr"; ni.set_chemical_symbols(sym.tolist())
W("scratch/sft_relaxed.xyz", ni)
W("scratch/sft_relaxed.cif", ni)
print("wrote scratch/sft_relaxed.xyz / .cif")
