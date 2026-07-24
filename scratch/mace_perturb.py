import sys,time,numpy as np
from ase.build import bulk
from ase.optimize import FIRE
from ase.io import write
from mace.calculators import mace_mp
a=3.52; n=int(sys.argv[1]); ncell=10; nst=int(sys.argv[2])
at=bulk('Ni','fcc',a=a,cubic=True)*(ncell,ncell,ncell)
R=at.get_positions(); c=R.mean(0)
Y=np.array([1,1,1.])/np.sqrt(3);X=np.array([1,1,-2.])/np.sqrt(6);Z=np.array([1,-1,0.])/np.sqrt(2)
sY=(R-c)@Y; plane=sY[np.argmin(np.abs(sY))]; on=np.abs(sY-plane)<0.3*a
uu=(R-c)@X; vv=(R-c)@Z; L=(n-1)*a/np.sqrt(2); rr=L/np.sqrt(3)
ang=np.deg2rad([90,210,330]); vx=rr*np.cos(ang); vy=rr*np.sin(ang)
ins=np.ones(len(R),bool)
for i in range(3):
    x0,y0=vx[i],vy[i]; x1,y1=vx[(i+1)%3],vy[(i+1)%3]
    ins&=((x1-x0)*(vv-y0)-(y1-y0)*(uu-x0))>=-1e-9
remove=on&ins
# bias: drop the PRISM above the triangle by one layer to initiate closure
above=(sY>plane+0.1)&ins
R[above]+=-Y*(a/np.sqrt(3))
rng=np.random.RandomState(0); R+=rng.normal(0,0.12,R.shape)   # break metastability
at.set_positions(R); del at[remove]
print(f"n={n} removed {remove.sum()}, dropped prism {above.sum()}, perturbed",flush=True)
at.calc=mace_mp(model='small',device='cpu',default_dtype='float32')
t=time.time(); FIRE(at,logfile='scratch/mp.log').run(fmax=0.03,steps=nst)
print(f"relaxed {nst} in {time.time()-t:.0f}s",flush=True); write('scratch/sft_perturb.xyz',at)
