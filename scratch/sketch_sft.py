"""Idealised (UNRELAXED) SFT, <110> projection of a slice through it.

Perfect fcc.  A regular tetrahedron with base on (111) and edges along <110>.
The interior is displaced by intrinsic-fault Shockleys on the three inclined
{111} faces (the base carries the Frank/removed-plane fault), so the four faces
show a stacking-sequence change.  Pure geometry, no relaxation.
"""
import numpy as np, itertools
from ase.build import bulk
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

a = 3.524; EDGE = 10; M = 16
ni = bulk("Ni", "fcc", a=a, cubic=True) * (M, M, M)
R = ni.get_positions(); c0 = R.mean(0)

# tetrahedron: base = face (111), edges a/2<110>
P = (a/2)*np.array([[0,0,0],[EDGE,EDGE,0],[EDGE,0,EDGE],[0,EDGE,EDGE]], float)
G = P.mean(0); P = P + (c0 - G)
faces = [(1,2,3),(0,2,3),(0,1,3),(0,1,2)]
def onrm(t, cen):
    n = np.cross(t[1]-t[0], t[2]-t[0]); n/=np.linalg.norm(n)
    return n if n@(t.mean(0)-cen) > 0 else -n
nrm = [onrm(P[list(f)], P.mean(0)) for f in faces]

# barycentric interior test
Mmat = np.vstack([P[1]-P[0], P[2]-P[0], P[3]-P[0]]).T
bc = np.linalg.solve(Mmat, (R-P[0]).T).T; b0 = 1-bc.sum(1)
inside = (bc[:,0]>=-1e-6)&(bc[:,1]>=-1e-6)&(bc[:,2]>=-1e-6)&(b0>=-1e-6)

# displace interior by an intrinsic Shockley of the base (111): a/6[11-2]
b = (a/6)*np.array([1., 1., -2.])
disp = np.zeros_like(R); disp[inside] = b
R = R + disp
ni.set_positions(R)

# ---- projection basis: view along [1-10] ----------------------------------
Yh = np.array([1,1,1.])/np.sqrt(3)     # [111]  vertical
Xh = np.array([1,1,-2.])/np.sqrt(6)    # [11-2] horizontal
Zh = np.array([1,-1,0.])/np.sqrt(2)    # [1-10] view

# ---- ABC registry ---------------------------------------------------------
p1 = (a/2)*np.array([1,-1,0.]); p2 = (a/2)*np.array([0,1,-1.])
Bm = np.array([[p1@Xh, p2@Xh],[p1@Zh, p2@Zh]])
uv = np.c_[(R-c0)@Xh, (R-c0)@Zh]
ab = np.linalg.solve(Bm, uv.T).T
reg = (np.round(3*(ab[:,0]-np.floor(ab[:,0])))%3).astype(int)

# ---- thin slice through the SFT centre ------------------------------------
w = (R-c0)@Zh
slab = np.abs(w) < 0.5*(a/np.sqrt(2))          # one <110> column layer
x = (R-c0)@Xh; y = (R-c0)@Yh

R2 = 22.0
keep = slab & (np.abs(x) < R2) & (np.abs(y) < R2)
fig, ax = plt.subplots(figsize=(11,10))
col = {0:"#4C72B0",1:"#DD8452",2:"#55A868"}; nm = {0:"A",1:"B",2:"C"}
# faint horizontal guide lines at each (111) plane, coloured by registry
for yy in np.unique(np.round(y[keep],2)):
    row = keep & (np.abs(y-yy) < 0.2)
    if row.sum() < 2: continue
    k = int(np.bincount(reg[row]).argmax())
    ax.plot([-R2, R2], [yy, yy], color=col[k], lw=1, alpha=0.25, zorder=0)
for k in (0,1,2):
    m = keep & ~inside & (reg==k)
    ax.scatter(x[m], y[m], s=150, c=col[k], edgecolors="none", label=f"{nm[k]} plane", zorder=3)
for k in (0,1,2):
    m = keep & inside & (reg==k)
    ax.scatter(x[m], y[m], s=150, c=col[k], edgecolors="k", linewidths=1.8, zorder=4)
Pp = np.c_[(P-c0)@Xh, (P-c0)@Yh]
for i,j in itertools.combinations(range(4),2):
    ax.plot([Pp[i,0],Pp[j,0]],[Pp[i,1],Pp[j,1]], "k-", lw=1.5, alpha=0.6, zorder=2)
ax.scatter([],[],s=150,facecolors="w",edgecolors="k",linewidths=1.8,label="SFT interior")
# annotate the matrix stacking sequence on the far left
ys = np.sort(np.unique(np.round(y[keep & ~inside & (np.abs(x) > R2-8)], 2)))
ax.set_xlabel(r"[11$\bar2$]  ($\AA$)", fontsize=12); ax.set_ylabel(r"[111]  ($\AA$)  (stacking direction)", fontsize=12)
ax.set_title(r"Idealised (unrelaxed) SFT — [1$\bar1$0] projection, central slice"+"\n"
             r"matrix = perfect ...ABCABC...   ·   SFT interior displaced (black rings)", fontsize=12)
ax.set_xlim(-R2, R2); ax.set_ylim(-R2, R2); ax.set_aspect("equal")
ax.legend(loc="upper right", framealpha=0.97, fontsize=11)
plt.tight_layout(); plt.savefig("scratch/sft_sketch.png", dpi=140)
print("wrote scratch/sft_sketch.png | drawn", keep.sum(), "interior", (keep&inside).sum())
