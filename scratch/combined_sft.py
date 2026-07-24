import sys,time,numpy as np
from ase.build import bulk
from ase.optimize import FIRE
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units
from ase.io import write
from mace.calculators import mace_mp
a=3.52; n=int(sys.argv[1]); ncell=int(sys.argv[2])
at=bulk('Ni','fcc',a=a,cubic=True)*(ncell,ncell,ncell)
R=at.get_positions(); c=R.mean(0)
Y=np.array([1,1,1.])/np.sqrt(3);X=np.array([1,1,-2.])/np.sqrt(6);Z=np.array([1,-1,0.])/np.sqrt(2)
sY=(R-c)@Y; plane=sY[np.argmin(np.abs(sY))]; on=np.abs(sY-plane)<0.3*a
uu=(R-c)@X; vv=(R-c)@Z; L=(n-1)*a/np.sqrt(2); rr=L/np.sqrt(3)
ang=np.deg2rad([90,210,330]); vx=rr*np.cos(ang); vy=rr*np.sin(ang)
def in2(px,py):
    ins=np.ones(len(px),bool)
    for i in range(3):
        x0,y0=vx[i],vy[i]; x1,y1=vx[(i+1)%3],vy[(i+1)%3]
        ins&=((x1-x0)*(py-y0)-(y1-y0)*(px-x0))>=-1e-9
    return ins
remove=on&in2(uu,vv)
base=np.array([c+plane*Y+vx[i]*X+vy[i]*Z for i in range(3)]); h=L*np.sqrt(2/3); apex=base.mean(0)+h*Y
verts=np.vstack([base,apex]); Mm=np.vstack([verts[1]-verts[0],verts[2]-verts[0],verts[3]-verts[0]]).T
bc=np.linalg.solve(Mm,(R-verts[0]).T).T; b0=1-bc.sum(1)
intet=(bc[:,0]>=-1e-6)&(bc[:,1]>=-1e-6)&(bc[:,2]>=-1e-6)&(b0>=-1e-6)&(~remove)
R[intet]+=-Y*(a/np.sqrt(3)); at.set_positions(R); del at[remove]
print(f"n={n} removed {remove.sum()} dropped {intet.sum()}",flush=True)
at.calc=mace_mp(model='small',device='cpu',default_dtype='float32')
t=time.time(); FIRE(at,logfile='scratch/comb.log').run(fmax=0.1,steps=120)
MaxwellBoltzmannDistribution(at,temperature_K=600)
dyn=Langevin(at,3*units.fs,temperature_K=600,friction=0.01); dyn.run(250)
for T in np.linspace(600,10,5): dyn.set_temperature(temperature_K=float(T)); dyn.run(120)
FIRE(at,logfile='scratch/comb.log').run(fmax=0.03,steps=200)
print(f"done {time.time()-t:.0f}s",flush=True); write(f'scratch/sft_comb{n}.xyz',at)
