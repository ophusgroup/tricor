"""Common Neighbor Analysis (fixed-cutoff) to classify FCC/HCP/other.

FCC atom : 12 neighbours, every bond has CNA signature (4,2,1)
HCP atom : 6 bonds (4,2,1) + 6 bonds (4,2,2)  -> stacking faults
Other    : dislocation cores / surfaces
"""
import numpy as np, sys
from ase.io import read
from scipy.spatial import cKDTree
sys.setrecursionlimit(10000)

def longest_chain(adj):
    """Longest path length (number of bonds) in a small bond graph."""
    n = len(adj)
    best = 0
    def dfs(v, seen, length):
        nonlocal best
        best = max(best, length)
        for w in np.where(adj[v])[0]:
            if w not in seen:
                dfs(int(w), seen | {int(w)}, length + 1)
    for s in range(n):
        dfs(s, {s}, 0)
    return best

fn = sys.argv[1] if len(sys.argv) > 1 else "scratch/sft_Ni_Cr_real.xyz"
at = read(fn); R = at.get_positions(); sym = np.array(at.get_chemical_symbols())
a = 3.524; rc = 0.854 * a                       # midway 1st-2nd NN of fcc

tree = cKDTree(R)
nbrs = tree.query_ball_point(R, rc)
nbrs = [np.array([j for j in nb if j != i]) for i, nb in enumerate(nbrs)]

cls = np.zeros(len(R), int)                      # 1=fcc 2=hcp 0=other
for i in range(len(R)):
    ni = nbrs[i]
    if len(ni) != 12:
        continue
    seti = set(ni.tolist())
    sigs = []
    for j in ni:
        common = np.array([k for k in nbrs[j] if k in seti])
        ncn = len(common)
        # bond graph among the common neighbours
        if ncn:
            sub = R[common]
            dsub = np.linalg.norm(sub[:, None] - sub[None], axis=2)
            adj = (dsub < rc) & (dsub > 1e-3)
            nb = int(adj.sum() // 2)
            chain = longest_chain(adj)   # third CNA index
        else:
            nb, chain = 0, 0
        sigs.append((ncn, nb, chain))
    c421 = sum(1 for s in sigs if s == (4, 2, 1))
    c422 = sum(1 for s in sigs if s == (4, 2, 2))
    if c421 == 12:
        cls[i] = 1
    elif c421 == 6 and c422 == 6:
        cls[i] = 2

fcc = (cls == 1).sum(); hcp = (cls == 2).sum(); oth = (cls == 0).sum()
print(f"{fn}")
print(f"  FCC   : {fcc}")
print(f"  HCP   : {hcp}   <- stacking-fault atoms")
print(f"  other : {oth}   <- dislocation cores / surface")

# render HCP + other, projected along [1,-1,0]
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
e2 = np.array([1, 1, -2.]) / np.sqrt(6); e3 = np.array([1, 1, 1.]) / np.sqrt(3)
x = R @ e2; y = R @ e3
surf = np.array([len(nbrs[i]) < 11 for i in range(len(R))])  # box surface
fig, ax = plt.subplots(figsize=(7, 7))
sel = (cls == 1) & ~surf
ax.scatter(x[sel], y[sel], s=1, c="0.85")
sel = (cls == 2)
ax.scatter(x[sel], y[sel], s=14, c="tab:green", label=f"HCP (stacking fault) {hcp}")
sel = (cls == 0) & ~surf
ax.scatter(x[sel], y[sel], s=14, c="tab:red", label=f"other (disloc core) {(sel).sum()}")
ax.set_aspect("equal"); ax.legend(loc="upper right"); ax.set_title("SFT structure (CNA) proj [1,-1,0]")
plt.tight_layout(); plt.savefig("scratch/sft_cna.png", dpi=130)
print("wrote scratch/sft_cna.png")
