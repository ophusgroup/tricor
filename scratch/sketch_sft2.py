"""Idealised (unrelaxed) SFT, <110> projection — CORRECTED so the four faces are
equivalent intrinsic faults.  Interior stays perfect fcc; each {111} face carries
its OWN Shockley a/6<112> (pointing toward the apex), applied to the exterior
wedge of that face.  So the two edge-on faces come out mirror-equivalent.
"""
import numpy as np, itertools
from ase.build import bulk
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

a = 3.524; EDGE = 10; M = 18
ni = bulk("Ni", "fcc", a=a, cubic=True) * (M, M, M)
R0 = ni.get_positions(); c0 = R0.mean(0)

P = (a/2)*np.array([[0,0,0],[EDGE,EDGE,0],[EDGE,0,EDGE],[0,EDGE,EDGE]], float)
G = P.mean(0); P = P + (c0 - G)
faces = [(1,2,3),(0,2,3),(0,1,3),(0,1,2)]        # opposite vertex k
apexv = [0,1,2,3]
def onrm(t):
    n = np.cross(t[1]-t[0], t[2]-t[0]); n/=np.linalg.norm(n)
    return n if n@(t.mean(0)-P.mean(0)) > 0 else -n
nrm = [onrm(P[list(f)]) for f in faces]

# <112> family
def fam(b):
    s=set()
    for p in itertools.permutations(b):
        for sg in itertools.product([1,-1],repeat=3): s.add(tuple(sg[i]*p[i] for i in range(3)))
    return [np.array(v,float) for v in s]
U112 = fam((1,1,2))

# each face's intrinsic Shockley = in-plane <112> pointing toward the apex
shock = []
for k,f in enumerate(faces):
    n = nrm[k]; fc = P[list(f)].mean(0); toapex = P[apexv[k]] - fc
    toapex = toapex - (toapex@n)*n                       # in-plane part
    cand = [(a/6)*v for v in U112 if abs(v@n) < 1e-9]
    b = max(cand, key=lambda v: v@toapex)                # most toward apex
    shock.append(b)

def in_tri(pts, tri, n):
    v0=tri[1]-tri[0]; v1=tri[2]-tri[0]; v2=pts-tri[0]
    d00=v0@v0; d01=v0@v1; d11=v1@v1; d20=v2@v0; d21=v2@v1; den=d00*d11-d01*d01
    uu=(d11*d20-d01*d21)/den; vv=(d00*d21-d01*d20)/den
    return (uu>=-1e-9)&(vv>=-1e-9)&(uu+vv<=1+1e-9)

# interior = perfect fcc (unshifted).  Shift the exterior wedge of each face.
u = np.zeros_like(R0)
for k,f in enumerate(faces):
    tri = P[list(f)]; n = nrm[k]
    s = (R0-tri[0])@n; proj = R0 - np.outer(s,n)
    H = (s > 0.05) & in_tri(proj, tri, n)                # exterior side, within face
    u[H] += shock[k]
R = R0 + u

Mmat = np.vstack([P[1]-P[0],P[2]-P[0],P[3]-P[0]]).T
bc = np.linalg.solve(Mmat,(R0-P[0]).T).T; b0=1-bc.sum(1)
interior = (bc[:,0]>=-1e-6)&(bc[:,1]>=-1e-6)&(bc[:,2]>=-1e-6)&(b0>=-1e-6)

Xh=np.array([1,1,-2.])/np.sqrt(6); Yh=np.array([1,1,1.])/np.sqrt(3); Zh=np.array([1,-1,0.])/np.sqrt(2)
p1=(a/2)*np.array([1,-1,0.]); p2=(a/2)*np.array([0,1,-1.])
Bm=np.array([[p1@Xh,p2@Xh],[p1@Zh,p2@Zh]])
uv=np.c_[(R-c0)@Xh,(R-c0)@Zh]; ab=np.linalg.solve(Bm,uv.T).T
reg=(np.round(3*(ab[:,0]-np.floor(ab[:,0])))%3).astype(int)

w=(R-c0)@Zh; slab=np.abs(w)<0.5*(a/np.sqrt(2)); x=(R-c0)@Xh; y=(R-c0)@Yh
R2=22; keep=slab&(np.abs(x)<R2)&(np.abs(y)<R2)
col={0:"#4C72B0",1:"#DD8452",2:"#55A868"}; nm={0:"A",1:"B",2:"C"}
fig,ax=plt.subplots(figsize=(11,10))
for yy in np.unique(np.round(y[keep],2)):
    row=keep&(np.abs(y-yy)<0.2)
    if row.sum()<2: continue
    ax.plot([-R2,R2],[yy,yy],color=col[int(np.bincount(reg[row]).argmax())],lw=1,alpha=0.2,zorder=0)
for k in (0,1,2):
    m=keep&~interior&(reg==k); ax.scatter(x[m],y[m],s=150,c=col[k],edgecolors="none",label=f"{nm[k]} plane",zorder=3)
for k in (0,1,2):
    m=keep&interior&(reg==k); ax.scatter(x[m],y[m],s=150,c=col[k],edgecolors="k",linewidths=1.8,zorder=4)
Pp=np.c_[(P-c0)@Xh,(P-c0)@Yh]
for i,j in itertools.combinations(range(4),2):
    ax.plot([Pp[i,0],Pp[j,0]],[Pp[i,1],Pp[j,1]],"k-",lw=1.3,alpha=0.55,zorder=2)
ax.scatter([],[],s=150,facecolors="w",edgecolors="k",linewidths=1.8,label="SFT interior (perfect fcc)")
ax.set_xlabel(r"[11$\bar2$]  ($\AA$)",fontsize=12); ax.set_ylabel(r"[111]  ($\AA$)  (stacking)",fontsize=12)
ax.set_title(r"Idealised SFT — [1$\bar1$0] projection: interior perfect fcc, each face its own Shockley",fontsize=12)
ax.set_xlim(-R2,R2); ax.set_ylim(-R2,R2); ax.set_aspect("equal"); ax.legend(loc="upper right",fontsize=11,framealpha=0.97)
plt.tight_layout(); plt.savefig("scratch/sft_sketch2.png",dpi=140)
print("Shockleys (a units):",[tuple(np.round(b/a,3)) for b in shock])
print("wrote scratch/sft_sketch2.png")
