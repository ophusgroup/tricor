"""Clean, gap-free idealised SFT, [1-10] projection.

Displacement u(r) = sum_k b_k * [r on interior side of face k], where b_k is the
in-plane Shockley a/6<112> of face k.  The four b_k SUM TO ZERO, so the interior
(inside all four faces) is unshifted -> perfect fcc.  Every jump across a face
plane is an in-plane Shockley, so atoms slide parallel to the plane: NO gaps, no
overlaps.  Slab is chosen on ORIGINAL positions so no atom drops out of view.
"""
import numpy as np, itertools
from ase.build import bulk
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

a = 3.524; EDGE = 10; M = 20
ni = bulk("Ni", "fcc", a=a, cubic=True) * (M, M, M)
R0 = ni.get_positions(); c0 = R0.mean(0)

P = (a/2)*np.array([[0,0,0],[EDGE,EDGE,0],[EDGE,0,EDGE],[0,EDGE,EDGE]], float)
P = P + (c0 - P.mean(0))
faces = [(1,2,3),(0,2,3),(0,1,3),(0,1,2)]; apexv = [0,1,2,3]
def onrm(t):
    n = np.cross(t[1]-t[0], t[2]-t[0]); n/=np.linalg.norm(n)
    return n if n@(t.mean(0)-P.mean(0)) > 0 else -n
nrm = [onrm(P[list(f)]) for f in faces]
def fam(b):
    s=set()
    for p in itertools.permutations(b):
        for sg in itertools.product([1,-1],repeat=3): s.add(tuple(sg[i]*p[i] for i in range(3)))
    return [np.array(v,float) for v in s]
U112 = fam((1,1,2))
shock = []
for k,f in enumerate(faces):
    n = nrm[k]; fc = P[list(f)].mean(0); ta = P[apexv[k]]-fc; ta = ta-(ta@n)*n
    shock.append(max([(a/6)*v for v in U112 if abs(v@n)<1e-9], key=lambda v: v@ta))
print("sum of Shockleys (should be ~0):", np.round(sum(shock),6))

# displacement: interior side of each face plane gets that face's Shockley
u = np.zeros_like(R0)
for k,f in enumerate(faces):
    s = (R0 - P[list(f)][0]) @ nrm[k]         # >0 outside, <0 inside
    u[s < 1e-9] += shock[k]
R = R0 + u

Xh=np.array([1,1,-2.])/np.sqrt(6); Yh=np.array([1,1,1.])/np.sqrt(3); Zh=np.array([1,-1,0.])/np.sqrt(2)
# registry from displaced position
p1=(a/2)*np.array([1,-1,0.]); p2=(a/2)*np.array([0,1,-1.])
Bm=np.array([[p1@Xh,p2@Xh],[p1@Zh,p2@Zh]])
uv=np.c_[(R-c0)@Xh,(R-c0)@Zh]; ab=np.linalg.solve(Bm,uv.T).T
reg=(np.round(3*(ab[:,0]-np.floor(ab[:,0])))%3).astype(int)

# interior flag (original position) for outlining
Mmat=np.vstack([P[1]-P[0],P[2]-P[0],P[3]-P[0]]).T
bc=np.linalg.solve(Mmat,(R0-P[0]).T).T; b0=1-bc.sum(1)
interior=(bc[:,0]>=-1e-6)&(bc[:,1]>=-1e-6)&(bc[:,2]>=-1e-6)&(b0>=-1e-6)

# slab selected on ORIGINAL depth so no atom disappears
w0=(R0-c0)@Zh; slab=np.abs(w0) < 0.5*(a/np.sqrt(2))
x=(R-c0)@Xh; y=(R-c0)@Yh
R2=24; keep=slab&(np.abs(x)<R2)&(np.abs(y)<R2)
col={0:"#4C72B0",1:"#DD8452",2:"#55A868"}; nm={0:"A",1:"B",2:"C"}
fig,ax=plt.subplots(figsize=(11,10))
for k in (0,1,2):
    m=keep&~interior&(reg==k); ax.scatter(x[m],y[m],s=140,c=col[k],edgecolors="none",label=f"{nm[k]} plane",zorder=3)
for k in (0,1,2):
    m=keep&interior&(reg==k); ax.scatter(x[m],y[m],s=140,c=col[k],edgecolors="k",linewidths=1.8,zorder=4)
Pp=np.c_[(P-c0)@Xh,(P-c0)@Yh]
for i,j in itertools.combinations(range(4),2):
    ax.plot([Pp[i,0],Pp[j,0]],[Pp[i,1],Pp[j,1]],"k-",lw=1.3,alpha=0.5,zorder=2)
ax.scatter([],[],s=140,facecolors="w",edgecolors="k",linewidths=1.8,label="SFT interior (perfect fcc)")
ax.set_xlabel(r"[11$\bar2$]  ($\AA$)",fontsize=12); ax.set_ylabel(r"[111]  ($\AA$)  (stacking)",fontsize=12)
ax.set_title(r"Idealised SFT (unrelaxed), [1$\bar1$0] projection — gap-free in-plane Shockleys",fontsize=12)
ax.set_xlim(-R2,R2); ax.set_ylim(-R2,R2); ax.set_aspect("equal"); ax.legend(loc="upper right",fontsize=11,framealpha=0.97)
plt.tight_layout(); plt.savefig("scratch/sft_sketch3.png",dpi=140)
print("wrote scratch/sft_sketch3.png | drawn", keep.sum())
