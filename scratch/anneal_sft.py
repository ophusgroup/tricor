"""Form an SFT by MD annealing a triangular {111} vacancy plate (Silcox-Hirsch).
Heat -> hold -> slow quench -> minimise.  Low-SFE metal so the SFT is stable.
"""
import sys, numpy as np
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.optimize import FIRE
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units
from ase.io import write

el   = sys.argv[1] if len(sys.argv) > 1 else "Cu"
n    = int(sys.argv[2]) if len(sys.argv) > 2 else 12
ncell= int(sys.argv[3]) if len(sys.argv) > 3 else 13
Thold= float(sys.argv[4]) if len(sys.argv) > 4 else 800.0
alat = {"Al": 4.05, "Cu": 3.61, "Ni": 3.52, "Au": 4.08, "Ag": 4.09}[el]

at = bulk(el, "fcc", a=alat, cubic=True) * (ncell, ncell, ncell)
R = at.get_positions(); c = R.mean(0)
Y = np.array([1, 1, 1.]) / np.sqrt(3); X = np.array([1, 1, -2.]) / np.sqrt(6); Z = np.array([1, -1, 0.]) / np.sqrt(2)
sY = (R - c) @ Y; plane = sY[np.argmin(np.abs(sY))]; on = np.abs(sY - plane) < 0.3 * alat
u = (R - c) @ X; v = (R - c) @ Z
L = (n - 1) * alat / np.sqrt(2); r = L / np.sqrt(3)
ang = np.deg2rad([90, 210, 330]); vx = r * np.cos(ang); vy = r * np.sin(ang)
ins = np.ones(len(R), bool)
for i in range(3):
    x0, y0 = vx[i], vy[i]; x1, y1 = vx[(i+1) % 3], vy[(i+1) % 3]
    ins &= ((x1-x0)*(v-y0) - (y1-y0)*(u-x0)) >= -1e-9
remove = on & ins
print(f"{el} box {ncell}^3={len(at)} n={n} remove {remove.sum()} Thold={Thold}", flush=True)
del at[remove]
at.calc = EMT()

FIRE(at, logfile="scratch/anneal.log").run(fmax=0.1, steps=200)
MaxwellBoltzmannDistribution(at, temperature_K=Thold)
dyn = Langevin(at, 4*units.fs, temperature_K=Thold, friction=0.02)
dyn.run(4000)                                   # hold ~16 ps
for T in np.linspace(Thold, 10, 8):             # slow quench
    dyn.set_temperature(temperature_K=float(T)); dyn.run(500)
FIRE(at, logfile="scratch/anneal.log").run(fmax=0.03, steps=400)
write(f"scratch/sft_anneal_{el}{n}.xyz", at)
print(f"wrote scratch/sft_anneal_{el}{n}.xyz", flush=True)
