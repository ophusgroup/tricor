"""Form a real SFT physically: triangular Frank vacancy loop in Au (low SFE)
annealed with EMT molecular dynamics, then quenched.  In a low stacking-fault-
energy metal the Frank loop dissociates into a stacking-fault tetrahedron.
"""
import numpy as np
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.optimize import FIRE
from ase.md.langevin import Langevin
from ase import units
from ase.io import write
from scipy.spatial import cKDTree

el, a = "Au", 4.078
N = 12
side = int(__import__("sys").argv[1]) if len(__import__("sys").argv) > 1 else 10  # Frank loop edge (NN)

at = bulk(el, "fcc", a=a, cubic=True) * (N, N, N)
R = at.get_positions(); c = R.mean(0)
n = np.array([1, 1, 1.]) / np.sqrt(3)

# pick the (111) atomic plane nearest the centre
s = (R - c) @ n
plane_off = s[np.argmin(np.abs(s))]
on_plane = np.abs(s - plane_off) < 0.3

# triangle on that plane: 3 <110> in-plane directions from centre
e1 = np.array([1, -1, 0.]) / np.sqrt(2)
e2 = np.cross(n, e1)
P = R - np.outer((R - c) @ n - plane_off, n)      # project onto plane
xy = np.c_[(P - c) @ e1, (P - c) @ e2]
L = side * a / np.sqrt(2)
# equilateral triangle centred, circumradius ~ L/sqrt(3)
r = L / np.sqrt(3)
verts = np.array([[np.cos(t), np.sin(t)] for t in (np.pi/2, np.pi/2+2*np.pi/3, np.pi/2+4*np.pi/3)]) * r
def in_tri(pts, tri):
    d = []
    for i in range(3):
        A, B = tri[i], tri[(i+1) % 3]
        e = B - A; p = pts - A
        d.append(e[0]*p[:, 1] - e[1]*p[:, 0])
    d = np.array(d)
    return (d >= -1e-9).all(0) | (d <= 1e-9).all(0)
remove = on_plane & in_tri(xy, verts)
print(f"{el} atoms {len(at)}, removing Frank platelet of {remove.sum()} atoms (side {side})")
at = at[~remove]

at.calc = EMT()
# relax, then anneal, then quench
FIRE(at, logfile="scratch/frank.log").run(fmax=0.2, steps=200)
dyn = Langevin(at, 2*units.fs, temperature_K=700, friction=0.01)
dyn.run(3000)                                   # ~6 ps anneal
dyn.set_temperature(temperature_K=100); dyn.run(1500)
FIRE(at, logfile="scratch/frank.log").run(fmax=0.1, steps=300)

write("scratch/sft_au.xyz", at)
print("wrote scratch/sft_au.xyz")
