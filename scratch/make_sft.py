"""Form a real SFT the paper's way (Zhang et al., Scripta Mater. 136 (2017) 78):
remove an equilateral triangular vacancy plate on a (111) plane of a perfect fcc
crystal, then energy-minimise.  The plate collapses to a Frank loop and
dissociates (Silcox-Hirsch) into a stacking-fault tetrahedron: intrinsic faults
on the four {111} faces + six a/6<110> stair-rod edges.

Crystal oriented as in the paper:  X=[11-2]  Y=[111]  Z=[1-10].
"""
import sys, numpy as np
from ase.lattice.cubic import FaceCenteredCubic
from ase.calculators.emt import EMT
from ase.optimize import FIRE
from ase.io import write

el   = sys.argv[1] if len(sys.argv) > 1 else "Al"
n    = int(sys.argv[2]) if len(sys.argv) > 2 else 16      # atoms on plate edge
alat = {"Al": 4.05, "Cu": 3.61, "Ni": 3.52, "Au": 4.08, "Ag": 4.09}[el]

# oriented fcc block, X=[11-2] Y=[111] Z=[1-10]
dirs = [[1, 1, -2], [1, 1, 1], [1, -1, 0]]
size = {"Al": (7, 10, 7)}.get(el, (7, 10, 7))
at = FaceCenteredCubic(directions=dirs, symbol=el, latticeconstant=alat, size=size, pbc=True)
R = at.get_positions(); c = R.mean(0)
Y = np.array([1, 1, 1.]) / np.sqrt(3)              # (111) plane normal
X = np.array([1, 1, -2.]) / np.sqrt(6)
Z = np.array([1, -1, 0.]) / np.sqrt(2)

# pick the (111) atomic plane nearest the centre
sY = (R - c) @ Y
plane = sY[np.argmin(np.abs(sY))]
on = np.abs(sY - plane) < 0.3

# equilateral triangle in that plane, edge L = (n-1)*a/sqrt(2), <110> edges
L = (n - 1) * alat / np.sqrt(2)
xy = np.c_[(R - c) @ X, (R - c) @ Z]
r = L / np.sqrt(3)                                  # circumradius
ang = np.array([90, 210, 330]) * np.pi / 180
verts = np.c_[r * np.cos(ang), r * np.sin(ang)]
def inside(p, v):
    s = []
    for i in range(3):
        a, b = v[i], v[(i + 1) % 3]; e = b - a; q = p - a
        s.append(e[0] * q[:, 1] - e[1] * q[:, 0])
    s = np.array(s)
    return (s >= -1e-9).all(0) | (s <= 1e-9).all(0)
remove = on & inside(xy, verts)
print(f"{el} a={alat}  atoms {len(at)}  plate n={n} -> removing {remove.sum()} vacancies")
del at[remove]

at.calc = EMT()
FIRE(at, logfile="scratch/make.log").run(fmax=0.02, steps=600)
write(f"scratch/sft_{el}.xyz", at)
print(f"wrote scratch/sft_{el}.xyz  ({len(at)} atoms)")
