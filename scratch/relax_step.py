"""Prism-step SFT seed (rigid full-Shockley shifts -> real HCP faults) then relax.

The sharp shifts create genuine intrinsic faults on the four {111} faces but also
overlaps at the Frank base and trailing artifacts. We delete overlapping atoms
(the physical vacancy content of the Frank loop) and relax with EMT so the
strain settles into clean HCP faults + stair-rod cores.
"""
import numpy as np, itertools
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.optimize import FIRE
from ase.constraints import FixAtoms
from ase.io import write
from scipy.spatial import cKDTree

a = 3.524; EDGE = 8; N = 12

def fam(b):
    s = set()
    for p in itertools.permutations(b):
        for sg in itertools.product([1, -1], repeat=3):
            s.add(tuple(sg[i] * p[i] for i in range(3)))
    return [np.array(v, float) for v in s]
U112, U110 = fam((1, 1, 2)), fam((1, 1, 0))
def isin(v, f): return any(np.allclose(v, w, atol=1e-6) for w in f)

P = (a/2)*np.array([[0,0,0],[EDGE,EDGE,0],[EDGE,0,EDGE],[0,EDGE,EDGE]], float); G = P.mean(0)
fi = [(1,2,3),(0,2,3),(0,1,3),(0,1,2)]
def onrm(t):
    n = np.cross(t[1]-t[0], t[2]-t[0]); n /= np.linalg.norm(n)
    return n if n @ (t.mean(0)-G) > 0 else -n
nm = [onrm(P[list(f)]) for f in fi]
nb = nm[0]; bF = -(a/3.0)*np.round(nb*np.sqrt(3)); fa = [bF]
for k in range(1, 4):
    for b in [(a/6)*v for v in U112 if abs(v @ nm[k]) < 1e-9]:
        if isin((bF-b)/(a/6), U110): fa.append(b); break

def in_tri(pts, tri, n):
    v0=tri[1]-tri[0]; v1=tri[2]-tri[0]; v2=pts-tri[0]
    d00=v0@v0; d01=v0@v1; d11=v1@v1; d20=v2@v0; d21=v2@v1; den=d00*d11-d01*d01
    u=(d11*d20-d01*d21)/den; v=(d00*d21-d01*d20)/den
    return (u>=-1e-9)&(v>=-1e-9)&(u+v<=1+1e-9)

ni = bulk("Ni","fcc",a=a,cubic=True)*(N,N,N)
Pc = P + (np.array([N*a/2]*3) - G); R0 = ni.get_positions()
u = np.zeros_like(R0)
for k in range(4):
    tri = Pc[list(fi[k])]; n = nm[k]
    s = (R0-tri[0]) @ n
    proj = R0 - np.outer(s, n)
    H = (s > 0.05) & in_tri(proj, tri, n)
    u[H] += fa[k]
R = R0 + u

# interior tag (pre-shift lattice)
def inside(pts, v, tol=1e-6):
    M = np.vstack([v[1]-v[0], v[2]-v[0], v[3]-v[0]]).T
    bc = np.linalg.solve(M, (pts-v[0]).T).T; b0 = 1-bc.sum(1)
    return (bc[:,0]>=-tol)&(bc[:,1]>=-tol)&(bc[:,2]>=-tol)&(b0>=-tol)
interior = inside(R0, Pc)

# delete one atom from every too-close pair (Frank vacancy content)
tree = cKDTree(R); pairs = tree.query_pairs(1.9, output_type='ndarray')
drop = set()
for i, j in pairs:
    if i not in drop and j not in drop:
        drop.add(j if interior[j] else i)   # prefer deleting interior (vacancy)
keep = np.array([i not in drop for i in range(len(R))])
R = R[keep]; interior = interior[keep]
at = bulk("Ni","fcc",a=a,cubic=True)*(N,N,N); at = at[keep]; at.set_positions(R)

c = np.array([N*a/2]*3)
shell = np.abs(R - c).max(1) > (N*a/2 - a)
at.set_constraint(FixAtoms(mask=shell)); at.calc = EMT()
print(f"atoms {len(at)} (deleted {len(drop)}) | interior {interior.sum()} | pinned {shell.sum()}")
dd0,_ = cKDTree(R).query(R,k=2); print("seed minNN", round(dd0[:,1].min(),3))
FIRE(at, logfile="scratch/relax_step.log").run(fmax=0.1, steps=500)
at.set_constraint()
sym = np.array(at.get_chemical_symbols()); sym[interior]="Cr"; at.set_chemical_symbols(sym.tolist())
write("scratch/sft_step_relaxed.xyz", at); write("scratch/sft_step_relaxed.cif", at)
print("wrote scratch/sft_step_relaxed.xyz / .cif")
