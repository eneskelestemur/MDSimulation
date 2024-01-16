'''
    This file contains the main function for the MD simulation of
    a protein-ligand complex.
'''

# libraries
import os
import time
import subprocess
import sys
import argparse
import openmm.app as app
import openmm as mm
import parmed as pmd

# import MDAnalysis as mda
import matplotlib.pyplot as plt
import pandas as pd

from MMPBSA_mods import API as MMPBSA_API

# from MDAnalysis.analysis import rms

from parmed.openmm import MdcrdReporter, StateDataReporter, RestartReporter

from openmm import unit
from openmmforcefields.generators import SystemGenerator
from openff.toolkit.topology import Molecule
from pdbfixer import PDBFixer



## Functions
def simulate_complex(protein_file, ligand_file, output_dir='tmp'):
    '''
        This function prepares and simulates the protein-ligand 
        complex for the simulation.
        
        Currently, this function will create many files that start with
        underscore (_) in the current directory. These files are temporary 
        files and can be deleted after the simulation is finished. In the later
        versions, these files will be created in a temporary directory and
        deleted automatically after the simulation.

        Args:
            protein_file (str): the path to the protein file in pdb format.
            ligand_file (str): the path to the ligand file in sdf or mol2 format.
                It will also work with pdb format, but it is not recommended.
            output_dir (str): the path to the output directory. This is where
                it stores all the prmtop files
    '''
    start_time = time.time()
    # params
    # TODO: parametrize hard-coded values for the simulation

    os.makedirs(output_dir, exist_ok=True)

    platform = mm.Platform.getPlatformByName('CUDA')

    # load the protein file
    print('Loading the protein file...', flush=True)
    protein = PDBFixer(protein_file)
    protein.findMissingResidues()
    protein.findMissingAtoms()
    protein.findNonstandardResidues()
    print('Missing residues:', protein.missingResidues, flush=True)
    print('Missing atoms:', protein.missingAtoms, flush=True)
    print('Nonstandard residues:', protein.nonstandardResidues, flush=True)
    print('Adding missing atoms...', flush=True)
    protein.addMissingAtoms()
    print('Removing heterogens including water...', flush=True)
    protein.removeHeterogens(False)
    print('Adding missing hydrogens...', flush=True)
    modeller = app.Modeller(protein.topology, protein.positions)
    modeller.addHydrogens(forcefield=app.ForceField('amber/protein.ff14SB.xml'), pH=7.0, platform=platform)
    print('Writing the fixed protein file...', flush=True)
    app.PDBFile.writeFile(modeller.topology, modeller.positions, open(f'{output_dir}/_protein.pdb', 'w'))
    print('Protein is loaded!', flush=True)

    # load the ligand file
    print('Loading the ligand file...', flush=True)
    if ligand_file.endswith('.sdf'):
        ligand = Molecule.from_file(ligand_file, file_format='sdf')
    elif ligand_file.endswith('.mol2'):
        ligand = Molecule.from_file(ligand_file, file_format='mol2')
    elif ligand_file.endswith('.pdb'):
        ligand = Molecule.from_file(ligand_file, file_format='pdb')
    else:
        raise ValueError('Ligand file format not recognized. It should be sdf, mol2 or pdb.')
    if isinstance(ligand, list):
        raise ValueError('Ligand file should contain only one molecule.')
    ligand.to_file(f'{output_dir}/_ligand.mol2', file_format='MOL')
    print('Ligand is loaded!', flush=True)

    # create system generator for simulation
    ff_kwargs = {
        'constraints': None, 
        'rigidWater': False,
        'removeCMMotion': False, 
    }
    system_generator = SystemGenerator(
        forcefields=['amber/protein.ff14SB.xml', 'amber/tip3p_standard.xml'],
        small_molecule_forcefield='gaff-2.11',
        molecules=ligand,
        forcefield_kwargs=ff_kwargs,
    )

    # create topology of the complex using modeller
    print('Creating topology of the complex...', flush=True)
    modeller = app.Modeller(protein.topology, protein.positions)
    ligand_topology = ligand.to_topology()
    modeller.addHydrogens(forcefield=app.ForceField('amber/protein.ff14SB.xml'), pH=7.0, platform=platform)
    modeller.add(ligand_topology.to_openmm(), ligand_topology.get_positions().to_openmm())
    app.PDBFile.writeFile(modeller.topology, modeller.positions, open(f'{output_dir}/_complex.pdb', 'w'))
    print('Topology of the complex is created!', flush=True)

    # solvate the complex
    print('Solvating the complex...', flush=True)
    modeller.addSolvent(
        system_generator.forcefield, 
        model='tip3p', 
        padding=1.0*unit.nanometer, 
        positiveIon='Na+',
        negativeIon='Cl-',
        ionicStrength=0.0*unit.molar,
        neutralize=True,
    )
    app.PDBFile.writeFile(modeller.topology, modeller.positions, open(f'{output_dir}/_complex_solvated.pdb', 'w'))
    print('Complex is solvated!', flush=True)

    # create system
    print('Creating the system for simulation...', flush=True)
    system = system_generator.create_system(modeller.topology, molecules=ligand)
    integrator = mm.LangevinMiddleIntegrator(300*unit.kelvin, 1.0/unit.picosecond, 0.002*unit.picoseconds)
    system.addForce(mm.MonteCarloBarostat(1.0*unit.atmosphere, 300*unit.kelvin))
    print('System is created!', flush=True)

    # create simulation
    print('Creating the simulation...', flush=True)
    simulation = app.Simulation(modeller.topology, system, integrator, platform=platform)
    context = simulation.context
    context.setPositions(modeller.positions)
    print('Simulation is created!', flush=True)

    # save the solvated complex as prmtop and inpcrd files
    struct = pmd.openmm.load_topology(modeller.topology, system, modeller.positions)
    amber_parm = pmd.amber.AmberParm.from_structure(struct)
    radii_update = pmd.tools.actions.changeRadii(amber_parm, 'mbondi2')
    radii_update.execute()
    amber_parm.write_parm(f'{output_dir}/_complex_solvated.prmtop')
    # struct.save(f'{output_dir}/_complex_solvated.prmtop', overwrite=True)
    print('Solvated complex is saved!', flush=True)

    # minimize the energy
    print('Minimizing the energy...', flush=True)
    simulation.minimizeEnergy()
    app.PDBFile.writeFile(simulation.topology,
                          context.getState(getPositions=True, enforcePeriodicBox=True).getPositions(), 
                          open(f'{output_dir}/_complex_solvated_minimized.pdb', 'w'))
    print('Energy is minimized!', flush=True)

    # equilibrate the system
    print('Equilibrating the system...', flush=True)
    simulation.context.setVelocitiesToTemperature(300*unit.kelvin)
    simulation.reporters.append(
        app.StateDataReporter(
            f'{output_dir}/_equilibration.log',
            100,
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
        app.DCDReporter(
            f'{output_dir}/_equilibration.dcd',
            100,
            enforcePeriodicBox=True,
        )
    )
    simulation.reporters.append(
        MdcrdReporter(
            f'{output_dir}/_equilibration.mdcrd',
            100,
            crds=True,
        )
    )
    simulation.step(1000)
    simulation.reporters.clear()
    print('System is equilibrated!', flush=True)

    # simulate the system
    print('Simulating the system...', flush=True)
    simulation.reporters.append(
        app.StateDataReporter(
            f'{output_dir}/_simulation.log',
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
        app.DCDReporter(
            f'{output_dir}/_simulation.dcd',
            1000,
            enforcePeriodicBox=True,
        )
    )
    simulation.reporters.append(
        MdcrdReporter(
            f'{output_dir}/_simulation.mdcrd',
            1000,
            crds=True,
        )
    )
    simulation.step(1000000)
    print(f'System is simulated for {0.002*simulation.currentStep/1000} ns!', flush=True)

    # save the final state
    print('Saving the final state...', flush=True)
    simulation.saveState(f'{output_dir}/_final_state.xml')
    print('Final state is saved!', flush=True)
    print(f'Total simulation time: {(time.time()-start_time)/60} minutes', flush=True)
    print('\n----------------------------------------\n\n', flush=True)

def calculate_mmgbsa(output_dir):
    '''
        This function calculates MMGBSA score for the protein-ligand complex.
        Complex should be simulated first.

        Args:
            output_dir (str): the location of the output directory.
    '''
    # prepare the amber topology files for the MMGBSA calculation
    print('Preparing the amber topology files for the MMP(G)BSA calculation...', flush=True)
    os.system(f'rm {output_dir}/_complex.prmtop {output_dir}/_protein.prmtop {output_dir}/_ligand.prmtop')
    subprocess.run(f'ante-MMPBSA.py \
              -p {output_dir}/_complex_solvated.prmtop \
              -c {output_dir}/_complex.prmtop \
              -l {output_dir}/_protein.prmtop \
              -r {output_dir}/_ligand.prmtop \
              -s ":WAT,HOH,NA,CL,CIO,CS,IB,K,LI,MG,RB" \
              -m ":UNK"', shell=True)
    # run the MMPBSA.py script on command line
    print('Calculating MMP(G)BSA for the complex...', flush=True)
    subprocess.run(f'MMPBSA.py -O -i mmgbsa.in -o {output_dir}/mmgbsa_results.dat \
                   -sp {output_dir}/_complex_solvated.prmtop \
                   -cp {output_dir}/_complex.prmtop \
                   -rp {output_dir}/_protein.prmtop \
                   -lp {output_dir}/_ligand.prmtop \
                   -y {output_dir}/*.mdcrd \
                   -prefix {output_dir}/_', shell=True)
    
    # clean up the large reference.frc file -- not sure what this file is for
    os.system('rm reference.frc')

def analyze_mmgbsa():
    '''
        This function analyzes the MMGBSA results using the MMPBSA.py.MPI module.
    '''
    # TODO: implement this function

def plot_simulation_log(log_file, data_to_plot: list):
    '''
        This function plots the data from simulation log file.

        Args:
            log_file (str): the path to the simulation log file.
            data_to_plot (list): the list of reported data to plot.
    '''
    # TODO: implement this function
    

# main function
if __name__ == '__main__':
    out_dir = 'tmp'
    # first simulate the complex. This will take a while.
    simulate_complex('1uom_A_rec.pdb', '1uom_pti_lig.sdf', out_dir)
    # now calculate MMGBSA from the simulation results. This stores everything to
    # {out_dir}/mmgbsa_results.dat. This is a plain text file that should be pretty easy
    # to parse. There are intermediate files that are created in the out_dir. These files
    # useful to debug the calculation, but they are not necessary to keep.
    calculate_mmgbsa(out_dir)
