'''
    Simulation class for running MD simulations using OpenMM.
'''

# libraries
import os
import sys
import argparse
import openmm.app as app
import openmm as mm
import parmed as pmd

import MDAnalysis as mda
import matplotlib.pyplot as plt
import pandas as pd

from MMPBSA_mods import API as MMPBSA_API

from parmed.openmm import MdcrdReporter, StateDataReporter, RestartReporter

from openmm import unit
from openmmforcefields.generators import SystemGenerator
from openff.toolkit.topology import Molecule
from pdbfixer import PDBFixer


# Simulation class
class Simulation():
    '''
        Simulation class for running MD simulations using OpenMM.
    '''
    def __init__(self, protein_files, ligand_files=None, platform='CUDA'):
        '''
            Initialize the simulation class.

            Args:
                protein_files (list): the list of paths to the protein files in pdb format.
                ligand_files (list): the list of paths to the ligand files in sdf or mol2 format.
                    It will also work with pdb format, but it is not recommended.
                platform (str): the name of the platform to run the simulation on.
        '''
        
    def _load_protein(self, protein_file):
        '''
            Load the protein file and fix the missing residues, atoms, and nonstandard residues.

            Args:
                protein_file (str): the path to the protein file in pdb format.
        '''

    def _load_ligand(self, ligand_file):
        '''
            Load and process the ligand file. 

            Args:
                ligand_file (str): the path to the ligand file.
        '''
        
    def _prepare_simulation(
            self,
            proteins,
            ligands,
            integrator=mm.LangevinMiddleIntegrator(300*unit.kelvin, 1.0/unit.picosecond, 0.002*unit.picoseconds),
            additional_forces=[mm.MonteCarloBarostat(1.0*unit.atmosphere, 300*unit.kelvin, frequency=25)],
            forcefields=['amber14/protein.ff14SB.xml', 'amber14/tip3p.xml'],
            ligand_forcefield='gaff-2.11',
            solvate=True,
            solvent_kwargs={'model': 'tip3p', 'padding': 1.0*unit.nanometers},
            forcefield_kwargs={'constraints': None, 'rigidWater': False, 'removeCMMotion': False},
            nonperiodic_forcefield_kwargs={'nonbondedMethod' : app.NoCutoff},
            periodic_forcefield_kwargs={'nonbondedMethod' : app.PME},
            barostat=None,
    ):
        '''
            Prepare the simulation. This will return a 
            simulation object that is ready to run simulations.

            args:
                proteins (list): the list of protein objects.
                ligands (list): the list of ligand objects.
                integrator (openmm.Integrator): the integrator object.
                additional_forces (list): the list of additional forces to add to the system.
                forcefields (list): the list of forcefields to use for the protein and solvent.
                    First element must be the protein forcefield. 
                ligand_forcefield (str): the forcefield to use for the ligand.
                solvate (bool): whether to solvate the system or not.
                solvent_kwargs (dict): the keyword arguments for solvating the system.
                forcefield_kwargs (dict): the keyword arguments to use during the creation of the system.
                nonperiodic_forcefield_kwargs (dict): the keyword arguments for the nonperiodic topology.
                periodic_forcefield_kwargs (dict): the keyword arguments for the periodic topology.
                barostat (openmm.Force): the barostat to use for the simulation.

            returns:
                simulation (openmm.Simulation): the simulation object.
        '''

    def _save_amber_files(self, topology, system, positions, filename):
        '''
            Save the topology, system, and positions in amber format. 
            filename should not include the extension. This functions will
            automatically save prmtop and inpcrd files.

            Args:
                topology (openmm.Topology): the topology of the system.
                system (openmm.System): the system object.
                positions (openmm.Positions): the positions of the system.
                filename (str): the name of the output file.
        '''

    def _remove_tmp_files(self):
        '''
            Remove the temporary files.
        '''

    def run_simulation(
            self,
            num_steps=100000,
            minimize=True,
            equilibrate=True,
            reporters=[app.StateDataReporter(sys.stdout, 1000, potentialEnergy=True, temperature=True), app.DCDReporter('tmp/_trajectory.dcd', 1000)],
            integrator=mm.LangevinMiddleIntegrator(300*unit.kelvin, 1.0/unit.picosecond, 0.002*unit.picoseconds),
            additional_forces=[mm.MonteCarloBarostat(1.0*unit.atmosphere, 300*unit.kelvin, frequency=25)],
            forcefields=['amber14/protein.ff14SB.xml', 'amber14/tip3p.xml'],
            ligand_forcefield='gaff-2.11',
            solvate=True,
            solvent_kwargs={'model': 'tip3p', 'padding': 1.0*unit.nanometers},
            forcefield_kwargs={'constraints': None, 'rigidWater': False, 'removeCMMotion': False},
            nonperiodic_forcefield_kwargs={'nonbondedMethod' : app.NoCutoff},
            periodic_forcefield_kwargs={'nonbondedMethod' : app.PME},
            barostat=None,
            remove_tmp_files=False,
    ):
        '''
            Run the simulation.

            Args:
                num_steps (int): the number of steps to run the simulation for.
                minimize (bool): whether to minimize the energy or not, before the simulation.
                equilibrate (bool): whether to equilibrate the system or not, before the simulation.
                reporters (list): the list of reporters to use during the simulation.
                integrator (openmm.Integrator): the integrator object.
                additional_forces (list): the list of additional forces to add to the system.
                forcefields (list): the list of forcefields to use for the protein and solvent.
                    First element must be the protein forcefield. 
                ligand_forcefield (str): the forcefield to use for the ligand.
                solvate (bool): whether to solvate the system or not.
                solvent_kwargs (dict): the keyword arguments for solvating the system.
                forcefield_kwargs (dict): the keyword arguments to use during the creation of the system.
                nonperiodic_forcefield_kwargs (dict): the keyword arguments for the nonperiodic topology.
                periodic_forcefield_kwargs (dict): the keyword arguments for the periodic topology.
                barostat (openmm.Force): the barostat to use for the simulation.
                remove_tmp_files (bool): whether to remove the temporary files or not after the simulation.
        '''

