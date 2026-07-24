"""Fast SFT seed: remove a triangular {111} vacancy plate, drop the tetrahedron
of atoms above it straight down by one {111} interplanar spacing (the Frank
collapse that closes the loop), then relax with MACE.
"""
import sys, time, numpy as np
from ase.build import bulk
from ase.optimize import FIRE
from ase.io import write
from mace.calculators import mace_mp

a=3.52; n=int(sys.argv[1]) if len(sys.argv)>1 else 10; ncell=int(sys.argv[4]) if len(sys.argv)>4 else 10
nsteps=int(sys.argv[2]) if len(sys.argv)>2 else 80
dropfac=float(sys.argv[3]) if len(sys.argv)>3 else 1.0

at=bulk('Ni','fcc',a=a,cubic=True)*(ncell,ncell,ncell)
R=at.get_positions(); c=R.mean(0)
Y=np.array([1,1,1.])/np.sqrt(3); X=np.array([1,1,-2.])/np.sqrt(6); Z=np.array([1,-1,0.])/np.sqrt(2)
sY=(R-c)@Y; plane=sY[np.argmin(np.abs(sY))]; on=np.abs(sY-plane)<0.3*a
uu=(R-c)@X; vv=(R-c)@Z
L=(n-1)*a/np.sqrt(2); rr=L/np.sqrt(3)
ang=np.deg2rad([90,210,330]); vx=rr*np.cos(ang); vy=rr*np.sin(ang)
def inside2d(px,py):
    ins=np.ones(len(px),bool)
    for i in range(3):
        x0,y0=vx[i],vy[i]; x1,y1=vx[(i+1)%3],vy[(i+1)%3]
        ins&=((x1-x0)*(py-y0)-(y1-y0)*(px-x0))>=-1e-9
    return ins
remove=on&inside2d(uu,vv)

# tetrahedron above the plate
base=np.array([c+plane*Y+vx[i]*X+vy[i]*Z for i in range(3)])
h=L*np.sqrt(2/3); apex=base.mean(0)+h*Y
verts=np.vstack([base,apex])
Mm=np.vstack([verts[1]-verts[0],verts[2]-verts[0],verts[3]-verts[0]]).T
bc=np.linalg.solve(Mm,(R-verts[0]).T).T; b0=1-bc.sum(1)
intet=(bc[:,0]>=-1e-6)&(bc[:,1]>=-1e-6)&(bc[:,2]>=-1e-6)&(b0>=-1e-6)&(~remove)

d111=a/np.sqrt(3)
R[intet]+= -Y*d111*dropfac          # drop the tetra straight down one layer
at.set_positions(R); del at[remove]
print(f"n={n} removed {remove.sum()} plate, dropped {intet.sum()} tetra atoms by {dropfac} d111", flush=True)

at.calc=mace_mp(model='small',device='cpu',default_dtype='float32')
t=time.time(); FIRE(at,logfile='scratch/mace_fast.log').run(fmax=0.1,steps=nsteps)
print(f"relaxed {nsteps} steps in {time.time()-t:.0f}s", flush=True)
write('scratch/sft_fast.xyz',at); print("wrote scratch/sft_fast.xyz", flush=True)
