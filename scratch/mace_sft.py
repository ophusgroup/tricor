"""Real SFT in Ni via MACE-MP0 (paper's recipe: triangular {111} vacancy plate
-> energy-minimise -> collapses/dissociates into a stacking-fault tetrahedron).

Plate edge n atoms -> SFT base edge (n-1) nearest-neighbour spacings.
n=10 -> SFT side length 9.
"""
import sys, time, numpy as np
from ase.build import bulk
from ase.optimize import FIRE
from ase.io import write
from mace.calculators import mace_mp

el, a = "Ni", 3.52
n     = 10          # plate edge -> SFT side (n-1)=9
ncell = 10
dev   = "cpu"

at = bulk(el, "fcc", a=a, cubic=True) * (ncell, ncell, ncell)
R = at.get_positions(); c = R.mean(0)
Y = np.array([1,1,1.])/np.sqrt(3); X = np.array([1,1,-2.])/np.sqrt(6); Z = np.array([1,-1,0.])/np.sqrt(2)
sY = (R-c)@Y; plane = sY[np.argmin(np.abs(sY))]; on = np.abs(sY-plane) < 0.3*a
u = (R-c)@X; v = (R-c)@Z
L = (n-1)*a/np.sqrt(2); r = L/np.sqrt(3)
ang = np.deg2rad([90,210,330]); vx = r*np.cos(ang); vy = r*np.sin(ang)
ins = np.ones(len(R), bool)
for i in range(3):
    x0,y0 = vx[i],vy[i]; x1,y1 = vx[(i+1)%3],vy[(i+1)%3]
    ins &= ((x1-x0)*(v-y0)-(y1-y0)*(u-x0)) >= -1e-9
remove = on & ins
print(f"{el} {ncell}^3={len(at)} atoms | plate n={n} -> remove {remove.sum()} vacancies", flush=True)
del at[remove]

at.calc = mace_mp(model="small", device=dev, default_dtype="float32")
t = time.time()
opt = FIRE(at, logfile="scratch/mace_sft.log")
opt.run(fmax=0.08, steps=350)
print(f"relaxed in {time.time()-t:.0f}s, {opt.get_number_of_steps()} steps", flush=True)
write("scratch/sft_mace.xyz", at)
print("wrote scratch/sft_mace.xyz", flush=True)
