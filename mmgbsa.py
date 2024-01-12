'''
    MM-GBSA calculation from pdb files using openmm and Amber.
'''

# libraries
import os
import argparse
import openmm.app as app
import openmm as mm
# import MDAnalysis as mda
import matplotlib.pyplot as plt
import pandas as pd

# from MDAnalysis.analysis import rms

from rdkit.Chem.FastSDMolSupplier import FastSDMolSupplier
from rdkit.Chem import MolToMolFile, MolFromMolFile, AddHs, MolToPDBFile, MolFromPDBFile

from parmed import load_file
from parmed.openmm import MdcrdReporter, StateDataReporter, RestartReporter

from openmm.unit import nanometer, picosecond, femtosecond, angstrom, kelvin, picoseconds, atmosphere, atmospheres

# ## argument parser
# parser = argparse.ArgumentParser(description='MM-GBSA calculation from protein and ligand files using openmm and Amber.')
# parser.add_argument('-p', '--protein', help='protein file in pdb format')
# parser.add_argument('-l', '--ligand', help='ligand file in sdf or mol2 format')
# parser.add_argument('-H', '--add_Hs', type=int, help='0 to not add hydrogens, 1 to add hydrogens only to the protein, 2 to add hydrogens only to the ligand, 3 to add hydrogens to both protein and ligand', default=0)
# # parser.add_argument('-o', '--output', help='output file name')

# # parse arguments
# args = parser.parse_args()

# ## Preparation of the input files
# # add hydrogens to the protein file using amber reduce
# if args.add_Hs == 1 or args.add_Hs == 3:
#     os.system('pdb4amber -i '+args.protein+' -o PROTEIN.pdb --dry --reduce')
# else:
#     os.system('pdb4amber -i '+args.protein+' -o PROTEIN.pdb --dry')

# # convert the ligand file to pdb format using rdkit
# add_Hs = (args.add_Hs == 2 or args.add_Hs == 3)
# if args.ligand.endswith('.sdf'):
#     mol = FastSDMolSupplier(args.ligand)[0]
#     if add_Hs:
#         mol = AddHs(mol, addCoords=True)
#     MolToPDBFile(mol, '_LIG.pdb')
# elif args.ligand.endswith('.mol2'):
#     mol = MolFromMolFile(args.ligand)
#     if add_Hs:
#         mol = AddHs(mol, addCoords=True)
#     MolToPDBFile(mol, '_LIG.pdb')
# elif args.ligand.endswith('.pdb'):
#     if add_Hs:
#         mol = MolFromPDBFile(args.ligand)
#         mol = AddHs(mol, addCoords=True)
#         MolToPDBFile(mol, '_LIG.pdb')
#     else:
#         os.system('cp '+args.ligand+' _LIG.pdb')
# else:
#     raise ValueError('Ligand file format not recognized.')
# # add residue name "LIG" for "UNL"
# os.system('sed -i \'s/UNL/LIG/g\' _LIG.pdb')
# # clean up the ligand file using pdb4amber
# os.system('pdb4amber -i _LIG.pdb -o _LIG_H.pdb --dry')
# # convert the ligand file to mol2 format using antechamber
# os.system('antechamber -i _LIG_H.pdb -fi pdb -o LIG.mol2 -fo mol2 -c bcc -s 0 -pf y')
# # convert the ligand file to frcmod format using parmchk2
# os.system('parmchk2 -i LIG.mol2 -f mol2 -o LIG.frcmod -s 2')
# # tleap preparation of protein, ligand and complex using leap.in
# # -> lig.prmtop, lig.inpcrd, rec.prmtop, rec.inpcrd, com.prmtop, com.inpcrd
# os.system('tleap -f leap.in')
# # remove temporary files
# os.system('rm _* sqm* PROTEIN_*')
# # assert False

## Input Preparation


## Equilibration of the complex
# This part runs a very short MD simulation to equilibrate the system.
# First, very short minimization is performed -- 1000 steps
# Second, heating is performed -- 50ps
# Last, constant pressure equilibration is performed -- 500ps

# load input files
inpcrd = app.AmberInpcrdFile('com_solvated.inpcrd')
prmtop = app.AmberPrmtopFile('com_solvated.prmtop')

# create system
system = prmtop.createSystem(
    nonbondedMethod=app.PME, # default sander is used in amber tutorial
    nonbondedCutoff=0.8*nanometer, # from Amber tutorial
    constraints=app.HBonds, # from Amber tutorial
    rigidWater=True, # from Amber tutorial
    ewaldErrorTolerance=0.0005,
)

# integrator
integrator = mm.LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)
# platform
# platform = mm.Platform.getPlatformByName('CUDA')
# setup simulation
simulation = app.Simulation(
    prmtop.topology,
    system,
    integrator,
    # platform,
)

# initial minimization -- 1000 steps
simulation.context.setPositions(inpcrd.positions)
simulation.minimizeEnergy()
simulation.step(1000)

# reporter for the equilibration
simulation.reporters.append(
    app.StateDataReporter(
        'equilibrated.log',
        1000,
        step=True,
        potentialEnergy=True,
        kineticEnergy=True,
        totalEnergy=True,
        temperature=True,
        volume=True,
        density=True,
        speed=True,
    )
)
simulation.reporters.append(
    MdcrdReporter(
        'equilibrated.mdcrd',
        1000,
        crds=True,
        vels=True,
        frcs=True,
    )
)

# heating -- increase temp by 2K every 0.2ps for 50ps
temp_ref = 300*kelvin
temp_i = 2*kelvin
simulation.context.setVelocitiesToTemperature(temp_i)
while temp_i <= temp_ref:
    simulation.step(100)
    temp_i += 2*kelvin
    integrator.setTemperature(temp_i)

# constant pressure equilibration -- 500ps
mdsteps = 250000
temp_ref = 300*kelvin
barostat = system.addForce(mm.MonteCarloBarostat(1*atmosphere, temp_ref, 1000))
simulation.context.reinitialize(True)
simulation.step(mdsteps)

# save the equilibrated system
simulation.saveState('equilibrated.xml')

## Plot total energy, temperature and RMSD
fig = plt.figure(figsize=(16, 8))

# # load equilibrated system and plot rmsd
# u = mda.Universe('com_solvated.prmtop', 'equilibrated.mdcrd')
# rmsd = rms.RMSD(u, u, select='all')
# rmsd.run()
# ax = fig.add_subplot(131)
# ax.plot(range(0, rmsd.n_frames*2, 2), rmsd.rmsd, label='RMSD')
# ax.set_xlabel('Time (ps)')
# ax.set_ylabel('RMSD (A)')
# ax.set_title('RMSD')

# load equilibrated log file and plot total energy and temperature
data = pd.read_csv('equilibrated.log')
ax = fig.add_subplot(132)
ax.plot(data['step']*0.002, data['totalEnergy'], label='Total Energy')
ax.set_xlabel('Time (ps)')
ax.set_ylabel('Energy (kJ/mol)')
ax.set_title('Total Energy')
ax.legend()

ax = fig.add_subplot(133)
ax.plot(data['step']*0.002, data['temperature'], label='Temperature')
ax.set_xlabel('Time (ps)')
ax.set_ylabel('Temperature (K)')
ax.set_title('Temperature')
ax.legend()

fig.savefig('equilibration.png')

assert False
## Production MD
# This part runs the production MD simulation.
# First, the equilibrated system is loaded.
# Second, the production MD is performed for 5ns.
# Last, the production MD is saved.

# load equilibrated system
simulation.context.reinitialize()
simulation.loadState('equilibrated.xml')

# reporter for the production MD
simulation.reporters.append(
    app.StateDataReporter(
        'production.log',
        1000,
        step=True,
        potentialEnergy=True,
        kineticEnergy=True,
        totalEnergy=True,
        temperature=True,
        volume=True,
        density=True,
        speed=True,
    )
)
simulation.reporters.append(
    MdcrdReporter(
        'production.mdcrd',
        1000,
        crds=True,
        vels=True,
        frcs=True,
    )
)

# production MD -- 5ns
mdsteps = 2500000
simulation.step(mdsteps)

# save the production MD system
simulation.saveState('production.xml')
simulation.saveCheckpoint('production.chk')

## MM-GBSA calculation
# TODO: add MM-GBSA calculation
