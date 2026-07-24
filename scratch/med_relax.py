import sys,time
from ase.io import read,write
from ase.optimize import FIRE
from mace.calculators import mace_mp
at=read('scratch/sft_anneal9.xyz'); at.calc=mace_mp(model='medium',device='cpu',default_dtype='float32')
t=time.time(); FIRE(at,logfile='scratch/med.log').run(fmax=0.03,steps=120)
print(f"medium relax {time.time()-t:.0f}s"); write('scratch/sft_med.xyz',at)
