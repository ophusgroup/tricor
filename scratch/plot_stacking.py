"""2D <110>-projection of an fcc crystal showing ABC stacking + stacking faults.

Colours atoms by their true (111) in-plane registry (A/B/C sublattice), so the
matrix reads ...ABCABC... and any fault shows as a registry slip.  HCP (fault)
and disordered (dislocation-core) atoms are outlined.
"""
import sys, numpy as np
from ase.io import read
from scipy.spatial import cKDTree
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

fn   = sys.argv[1] if len(sys.argv) > 1 else "scratch/test_isf.xyz"
a    = float(sys.argv[2]) if len(sys.argv) > 2 else 3.61
slabT= float(sys.argv[3]) if len(sys.argv) > 3 else 1.0        # slab thickness (units of a/sqrt2)

at = read(fn); R = at.get_positions(); c = R.mean(0)
Yh = np.array([1, 1, 1.]) / np.sqrt(3)     # [111]  (stacking / vertical)
Xh = np.array([1, 1, -2.]) / np.sqrt(6)    # [11-2] (horizontal)
Zh = np.array([1, -1, 0.]) / np.sqrt(2)    # [1-10] (view direction)

# ---- true ABC in-plane registry -------------------------------------------
p1 = (a/2)*np.array([1, -1, 0.]); p2 = (a/2)*np.array([0, 1, -1.])
B = np.array([[p1@Xh, p2@Xh], [p1@Zh, p2@Zh]])
uv = np.c_[(R-c)@Xh, (R-c)@Zh]
ab = np.linalg.solve(B, uv.T).T                      # (alpha,beta) in p1,p2 basis
reg = np.round(3*((ab[:, 0] - np.floor(ab[:, 0])))) % 3   # 0/1/2 = A/B/C
reg = reg.astype(int)

# ---- CNA (FCC/HCP/other) to flag faults & cores ---------------------------
sys.setrecursionlimit(10000)
rc = 0.854*a
tree = cKDTree(R); nb = [np.array([j for j in x if j != i]) for i, x in enumerate(tree.query_ball_point(R, rc))]
def longest(adj):
    best = 0
    def dfs(v, seen, l):
        nonlocal best; best = max(best, l)
        for w in np.where(adj[v])[0]:
            if w not in seen: dfs(int(w), seen | {int(w)}, l+1)
    for s in range(len(adj)): dfs(s, {s}, 0)
    return best
cls = np.zeros(len(R), int)
for i in range(len(R)):
    ni = nb[i]
    if len(ni) != 12: continue
    si = set(ni.tolist()); c421 = c422 = 0
    for j in ni:
        com = np.array([k for k in nb[j] if k in si])
        if not len(com): continue
        sub = R[com]; d = np.linalg.norm(sub[:, None]-sub[None], axis=2)
        adj = (d < rc) & (d > 1e-3); nbnd = int(adj.sum()//2)
        if nbnd == 2:
            ch = longest(adj)
            if ch == 1: c421 += 1
            elif ch == 2: c422 += 1
    if c421 == 12: cls[i] = 1
    elif c421 == 6 and c422 == 6: cls[i] = 2
fcc = cls == 1; hcp = cls == 2; oth = cls == 0

# ---- thin slab along the view direction -----------------------------------
w = (R-c)@Zh
slab = np.abs(w) < slabT*(a/np.sqrt(2))*0.5
x = (R-c)@Xh; y = (R-c)@Yh

# ---- plot -----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 9))
cols = {0: "#4C72B0", 1: "#DD8452", 2: "#55A868"}    # A, B, C
lab = {0: "A", 1: "B", 2: "C"}
for k in (0, 1, 2):
    m = slab & fcc & (reg == k)
    ax.scatter(x[m], y[m], s=55, c=cols[k], edgecolors="none", label=f"{lab[k]} plane")
m = slab & hcp
ax.scatter(x[m], y[m], s=70, c="#C44E52", edgecolors="k", linewidths=0.8, label="HCP (stacking fault)")
m = slab & oth
ax.scatter(x[m], y[m], s=55, c="0.6", edgecolors="k", linewidths=0.5, label="disloc. core")
ax.set_xlabel(r"[11$\bar2$]  ($\AA$)"); ax.set_ylabel(r"[111]  ($\AA$)")
ax.set_title(f"{fn.split('/')[-1]}  —  [1$\\bar1$0] projection, ABC stacking")
ax.set_aspect("equal"); ax.legend(loc="upper right", framealpha=0.95)
plt.tight_layout(); out = "scratch/stacking_" + fn.split("/")[-1].replace(".xyz", "") + ".png"
plt.savefig(out, dpi=140); print("wrote", out, "| slab atoms", slab.sum(),
                                 "| HCP", (slab & hcp).sum(), "other", (slab & oth).sum())
