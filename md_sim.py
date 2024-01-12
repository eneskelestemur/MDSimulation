'''
    This file contains the main function for the MD simulation of
    a protein-ligand complex.
'''

# libraries
import os
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
def simulate_complex(protein_file, ligand_file):
    '''
        This function prepares and simulates the protein-ligand 
        complex for the simulation.

        Args:
            protein_file (str): the path to the protein file in pdb format.
            ligand_file (str): the path to the ligand file in sdf or mol2 format.
                It will also work with pdb format, but it is not recommended.
    '''
    # params
    # TODO: parametrize hard-coded values for the simulation
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
    app.PDBFile.writeFile(modeller.topology, modeller.positions, open('_protein.pdb', 'w'))
    ##### amber files #####
    protein = app.PDBFile('_protein.pdb')
    system = app.ForceField('amber/protein.ff14SB.xml').createSystem(protein.topology)
    integrator = mm.LangevinMiddleIntegrator(300*unit.kelvin, 1.0/unit.picosecond, 0.002*unit.picoseconds)
    context = mm.Context(system, integrator, platform)
    context.setPositions(protein.positions)
    struct = pmd.openmm.load_topology(protein.topology, system, protein.positions)
    struct.save('_protein.prmtop', overwrite=True)
    struct.save('_protein.inpcrd', overwrite=True)
    ##### amber files #####
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
    ligand.to_file('_ligand.mol2', file_format='MOL')
    print('Ligand is loaded!', flush=True)

    # load the force fields using the system generator
    print('Setting up the force fields...', flush=True)
    ff_kwargs = {
        'constraints': app.HBonds, 
        'rigidWater': True, 
        'removeCMMotion': False,
    }
    system_generator = SystemGenerator(
        forcefields=['amber/protein.ff14SB.xml', 'amber/tip3p_standard.xml'],
        small_molecule_forcefield='gaff-2.11',
        molecules=ligand,
        forcefield_kwargs=ff_kwargs,
    )
    print('Force fields are set up!', flush=True)

    # create topology of the complex using modeller
    print('Creating topology of the complex...', flush=True)
    modeller = app.Modeller(protein.topology, protein.positions)
    ligand_topology = ligand.to_topology()
    modeller.addHydrogens(forcefield=app.ForceField('amber/protein.ff14SB.xml'), pH=7.0, platform=platform)
    modeller.add(ligand_topology.to_openmm(), ligand_topology.get_positions().to_openmm())
    app.PDBFile.writeFile(modeller.topology, modeller.positions, open('_complex.pdb', 'w'))
    ##### amber files #####
    # complex
    system = system_generator.create_system(modeller.topology, molecules=ligand)
    integrator = mm.LangevinMiddleIntegrator(300*unit.kelvin, 1.0/unit.picosecond, 0.002*unit.picoseconds)
    context = mm.Context(system, integrator, platform)
    context.setPositions(modeller.positions)
    struct = pmd.openmm.load_topology(modeller.topology, system, modeller.positions)
    struct.save('_complex.prmtop', overwrite=True)
    struct.save('_complex.inpcrd', overwrite=True)
    # ligand
    system = system_generator.create_system(ligand_topology)
    context = mm.Context(system, integrator, platform)
    context.setPositions(ligand_topology.get_positions())
    struct = pmd.openmm.load_topology(ligand_topology, system, ligand_topology.get_positions())
    struct.save('_ligand.prmtop', overwrite=True)
    struct.save('_ligand.inpcrd', overwrite=True)
    ##### amber files #####
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
    app.PDBFile.writeFile(modeller.topology, modeller.positions, open('_complex_solvated.pdb', 'w'))
    ##### amber files #####
    system = system_generator.create_system(modeller.topology, molecules=ligand)
    integrator = mm.LangevinMiddleIntegrator(300*unit.kelvin, 1.0/unit.picosecond, 0.002*unit.picoseconds)
    context = mm.Context(system, integrator, platform)
    context.setPositions(modeller.positions)
    struct = pmd.openmm.load_topology(modeller.topology, system, modeller.positions)
    struct.save('_complex_solvated.prmtop', overwrite=True)
    struct.save('_complex_solvated.inpcrd', overwrite=True)
    ##### amber files #####
    print('Complex is solvated!', flush=True)

    # create system
    print('Creating the system for simulation...', flush=True)
    system = system_generator.create_system(modeller.topology, molecules=ligand)
    integrator = mm.LangevinMiddleIntegrator(300*unit.kelvin, 1.0/unit.picosecond, 0.002*unit.picoseconds)
    system.addForce(mm.MonteCarloBarostat(1.0*unit.atmosphere, 300*unit.kelvin, 25))
    print('System is created!', flush=True)

    # create simulation
    print('Creating the simulation...', flush=True)
    simulation = app.Simulation(modeller.topology, system, integrator, platform=platform)
    context = simulation.context
    context.setPositions(modeller.positions)
    print('Simulation is created!', flush=True)

    # minimize the energy
    print('Minimizing the energy...', flush=True)
    simulation.minimizeEnergy()
    app.PDBFile.writeFile(simulation.topology, 
                          context.getState(getPositions=True, enforcePeriodicBox=True).getPositions(), 
                          open('_complex_solvated_minimized.pdb', 'w'))
    print('Energy is minimized!', flush=True)

    # equilibrate the system
    print('Equilibrating the system...', flush=True)
    simulation.context.setVelocitiesToTemperature(300*unit.kelvin)
    simulation.step(1000)
    print('System is equilibrated!', flush=True)

    # simulate the system
    print('Simulating the system...', flush=True)
    simulation.reporters.append(
        app.StateDataReporter(
            'simulation.log',
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
            'simulation.dcd',
            1000,
            enforcePeriodicBox=True,
        )
    )
    simulation.reporters.append(
        MdcrdReporter(
            'simulation.mdcrd',
            1000,
            crds=True,
        )
    )
    simulation.step(10000)
    print(f'System is simulated for {0.002*simulation.currentStep/1.0*unit.nanosecond}!', flush=True)

    # save the final state
    print('Saving the final state...', flush=True)
    simulation.saveState('_final_state.xml')
    print('Final state is saved!', flush=True)

def calculate_mmgbsa(mmgbsa_file_loc):
    '''
        This function calculates the MMGBSA for the protein-ligand complex.
        Complex should be simulated first.

        Args:
            mmgbsa_file_loc (str): the path to the MMPBSA.py script.
    '''
    # run the MMPBSA.py script on command line
    print('Calculating the MMGBSA for the complex...', flush=True)
    os.system(f'{mmgbsa_file_loc} -O -i mmgbsa.in -o mmgbsa_results.dat \
              -sp _complex_solvated.prmtop \
              -cp _complex.prmtop \
              -rp _protein.prmtop \
              -lp _ligand.prmtop \
              -y *.mdcrd ')
    
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
    simulate_complex('1uom_A_rec.pdb', '1uom_pti_lig.sdf')
    calculate_mmgbsa('MMPBSA.py')
