"""Form a clean SFT: remove an equilateral triangular {111} vacancy plate from a
perfect fcc crystal and energy-minimise (Silcox-Hirsch).  Low-SFE Cu gives a
perfect SFT for small plates.  Cubic supercell; orientation handled at plot time.
"""
import sys, numpy as np
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.optimize import FIRE
from ase.io import write

el   = sys.argv[1] if len(sys.argv) > 1 else "Cu"
n    = int(sys.argv[2]) if len(sys.argv) > 2 else 10          # atoms per plate edge
ncell= int(sys.argv[3]) if len(sys.argv) > 3 else 16
alat = {"Al": 4.05, "Cu": 3.61, "Ni": 3.52, "Au": 4.08, "Ag": 4.09}[el]

at = bulk(el, "fcc", a=alat, cubic=True) * (ncell, ncell, ncell)
R = at.get_positions(); c = R.mean(0)
Y = np.array([1, 1, 1.]) / np.sqrt(3)          # (111) normal
X = np.array([1, 1, -2.]) / np.sqrt(6)
Z = np.array([1, -1, 0.]) / np.sqrt(2)

sY = (R - c) @ Y
plane = sY[np.argmin(np.abs(sY))]
on = np.abs(sY - plane) < 0.3 * alat
u = (R - c) @ X; v = (R - c) @ Z                # in-plane coords of plate atoms
L = (n - 1) * alat / np.sqrt(2)                 # triangle edge, <110> aligned
r = L / np.sqrt(3)
ang = np.deg2rad([90, 210, 330]); vx = r*np.cos(ang); vy = r*np.sin(ang)
def inside(px, py):
    ins = np.ones(len(px), bool)
    for i in range(3):
        x0,y0 = vx[i],vy[i]; x1,y1 = vx[(i+1)%3],vy[(i+1)%3]
        # sign of cross product (edge -> point); triangle CCW
        ins &= ((x1-x0)*(py-y0) - (y1-y0)*(px-x0)) >= -1e-9
    return ins
remove = on & inside(u, v)
exp = n*(n+1)//2
print(f"{el} a={alat} box {ncell}^3 = {len(at)} atoms | plate n={n}: remove {remove.sum()} (expect ~{exp})")
del at[remove]

at.calc = EMT()
FIRE(at, logfile="scratch/make2.log").run(fmax=0.02, steps=800)
write(f"scratch/sft_{el}{n}.xyz", at)
print(f"wrote scratch/sft_{el}{n}.xyz ({len(at)} atoms)")
