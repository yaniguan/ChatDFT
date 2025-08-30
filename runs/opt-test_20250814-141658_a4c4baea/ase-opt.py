from ase.constraints import Hookean
from ase.build import surface
from ase import Atoms
from ase.build import molecule, add_adsorbate
from ase.optimize import BFGS
from ase.calculators.vasp import Vasp
import os
import subprocess
from ase.calculators.vasp import Vasp2
from ase.constraints import FixAtoms
from ase.visualize import view
from ase.lattice.hexagonal import *
import ase.io
import pickle


# Make a test slab
try:
    slab = ase.io.read('POSCAR')
except:
    try:
        slab = ase.io.read('POSCAR-adsorbate')
    except:
        slab = ase.io.read('CONTCAR')


calc_2 = Vasp(xc='pbe',
         encut=400,
         kpts=[5,5,1],
         ediff = 1.00e-06,
         ediffg = -0.01,
         algo='fast',
         ispin=1, #non spin polarized calculation
         ismear=2, #using the second order MP smearing
         sigma=0.1,
         nelm = 60,
         nsw = 1000,
         ibrion = 2,
         isif = 2,
         lwave = False,
         lcharg = False,
         npar = 4,
         lreal='A', #projection done in real space
         kgamma=True,
         ivdw = 4,
         istart=1, #if you want to restart put other values
         icharg=0)
slab.set_calculator(calc_2)
slab.get_potential_energy()
