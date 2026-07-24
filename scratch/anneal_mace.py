import sys,time,numpy as np
from ase.io import read,write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.optimize import FIRE
from ase import units
from mace.calculators import mace_mp
at=read('scratch/sft_fast.xyz')
at.calc=mace_mp(model='small',device='cpu',default_dtype='float32')
T0=600.0
MaxwellBoltzmannDistribution(at,temperature_K=T0)
dyn=Langevin(at,3*units.fs,temperature_K=T0,friction=0.01)
t=time.time(); dyn.run(250)
for T in np.linspace(T0,10,5): dyn.set_temperature(temperature_K=float(T)); dyn.run(120)
FIRE(at,logfile='scratch/anneal9.log').run(fmax=0.03,steps=200)
print(f"anneal done {time.time()-t:.0f}s",flush=True); write('scratch/sft_anneal9.xyz',at)
