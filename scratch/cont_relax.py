import sys,time
from ase.io import read,write
from ase.optimize import FIRE
from mace.calculators import mace_mp
fn=sys.argv[1]; nst=int(sys.argv[2])
at=read(fn); at.calc=mace_mp(model='small',device='cpu',default_dtype='float32')
t=time.time(); FIRE(at,logfile='scratch/cont.log').run(fmax=0.03,steps=nst)
print(f"relaxed {nst} more steps in {time.time()-t:.0f}s")
write(fn,at)
