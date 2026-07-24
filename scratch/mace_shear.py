"""Best seed: build the geometric SFT by the four in-plane Shockley shears
(interior stays perfect fcc, faces get real intrinsic faults), remove the base
{111} plate for the vacancy content, then relax a few steps with MACE so the
faults confine to the tetrahedron.
"""
import sys, time, numpy as np, itertools
from ase.build import bulk
from ase.optimize import FIRE
from ase.io import write
from mace.calculators import mace_mp

a=3.52; EDGE=6; ncell=10
nsteps=int(sys.argv[1]) if len(sys.argv)>1 else 40
ni=bulk('Ni','fcc',a=a,cubic=True)*(ncell,ncell,ncell)
R0=ni.get_positions(); c0=R0.mean(0)
P=(a/2)*np.array([[0,0,0],[EDGE,EDGE,0],[EDGE,0,EDGE],[0,EDGE,EDGE]],float); P=P+(c0-P.mean(0))
faces=[(1,2,3),(0,2,3),(0,1,3),(0,1,2)]; apexv=[0,1,2,3]
def onrm(t):
    n=np.cross(t[1]-t[0],t[2]-t[0]); n/=np.linalg.norm(n); return n if n@(t.mean(0)-P.mean(0))>0 else -n
nrm=[onrm(P[list(f)]) for f in faces]
def fam(b):
    s=set()
    for p in itertools.permutations(b):
        for sg in itertools.product([1,-1],repeat=3): s.add(tuple(sg[i]*p[i] for i in range(3)))
    return [np.array(v,float) for v in s]
U112=fam((1,1,2))
shock=[]
for k,f in enumerate(faces):
    n=nrm[k]; fc=P[list(f)].mean(0); ta=P[apexv[k]]-fc; ta=ta-(ta@n)*n
    shock.append(max([(a/6)*v for v in U112 if abs(v@n)<1e-9], key=lambda v: v@ta))

# four in-plane Shockley shears (sum to zero -> interior perfect fcc)
u=np.zeros_like(R0)
for k,f in enumerate(faces):
    s=(R0-P[list(f)][0])@nrm[k]
    u[s<1e-9]+=shock[k]
R=R0+u; ni.set_positions(R)

# remove base {111} plate (vacancy content) : atoms on base plane inside triangle
nb=nrm[0]; base=P[list(faces[0])]
sN=(R0-base[0])@nb; onb=np.abs(sN)<0.3*a
# in base-plane 2D coords
e1=base[1]-base[0]; e1/=np.linalg.norm(e1); e2=np.cross(nb,e1)
def in_tri3(pts):
    tri2=np.array([[ (v-base[0])@e1,(v-base[0])@e2] for v in base])
    p2=np.c_[(pts-base[0])@e1,(pts-base[0])@e2]; ins=np.ones(len(pts),bool)
    for i in range(3):
        x0,y0=tri2[i]; x1,y1=tri2[(i+1)%3]
        ins&=((x1-x0)*(p2[:,1]-y0)-(y1-y0)*(p2[:,0]-x0))>=-1e-9
    return ins
remove=onb&in_tri3(R0)
print(f"shear seed built; remove {remove.sum()} base-plate atoms", flush=True)
del ni[remove]

ni.calc=mace_mp(model='small',device='cpu',default_dtype='float32')
t=time.time(); FIRE(ni,logfile='scratch/mace_shear.log').run(fmax=0.1,steps=nsteps)
print(f"relaxed {nsteps} steps in {time.time()-t:.0f}s", flush=True)
write('scratch/sft_shear.xyz',ni); print("wrote scratch/sft_shear.xyz", flush=True)
