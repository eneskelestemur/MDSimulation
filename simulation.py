'''
    Simulation class for running MD simulations using OpenMM.
'''

# libraries
import os
import sys
import time
import argparse
import openmm.app as app
import openmm as mm
import parmed as pmd

import MDAnalysis as mda
import matplotlib.pyplot as plt
import pandas as pd

from MMPBSA_mods import API as MMPBSA_API

from parmed.openmm import MdcrdReporter

from openmm import unit
from openmmforcefields.generators import SystemGenerator
from openff.toolkit.topology import Molecule
from pdbfixer import PDBFixer


# Simulation class
class Simulation():
    '''
        Simulation class for running MD simulations using OpenMM.
    '''
    def __init__(self, protein_files, ligand_files=None, platform='CUDA', 
                 output_dir='MDSimulation', remove_tmp_files=False):
        '''
            Initialize the simulation class.

            Args:
                protein_files (list): the list of paths to the protein files in pdb format.
                    The loading of the protein files will be handled by the PDBFixer, so
                    the protein files can be in any format that PDBFixer can handle.
                ligand_files (list): the list of paths to the ligand files in sdf or mol2 format.
                    It will also work with pdb format, but it is not recommended.
                platform (str): the name of the platform to run the simulation on.
                output_dir (str): the path to the output directory. This directory will contain the
                    final output files and the tmp directory that will contain the temporary files.
                remove_tmp_files (bool): whether to remove the temporary files or not after the simulation.
                    These files will be saved in the tmp directory.
        '''
        self.protein_files = protein_files
        self.ligand_files = ligand_files
        self.platform = platform
        self.output_dir = output_dir
        self.remove_tmp_files = remove_tmp_files

        # create the tmp directory
        os.makedirs(f'{output_dir}/tmp', exist_ok=True)

        # load the protein and ligand files
        self.proteins = [self._load_protein(protein_file) for protein_file in self.protein_files]
        self.ligands = [self._load_ligand(ligand_file) for ligand_file in self.ligand_files]
        
    def _load_protein(self, protein_file):
        '''
            Load the protein file and fix the missing residues, atoms, and nonstandard residues.
            Used during initialization.

            Args:
                protein_file (str): the path to the protein file in pdb format.
        '''
        # load the protein file
        print('\nLoading the protein file...', flush=True)
        protein = PDBFixer(protein_file)
        protein.findMissingResidues()
        protein.findMissingAtoms()
        protein.findNonstandardResidues()
        print('Missing residues:', protein.missingResidues, flush=True)
        print('Missing atoms:', protein.missingAtoms, flush=True)
        print('Nonstandard residues:', protein.nonstandardResidues, flush=True)
        protein.addMissingAtoms()
        print('Missing atoms added!', flush=True)
        protein.removeHeterogens(False)
        print('Heterogens, including water, removed!', flush=True)
        modeller = app.Modeller(protein.topology, protein.positions)
        modeller.addHydrogens(forcefield=app.ForceField('amber/protein.ff14SB.xml'), pH=7.0, platform=self.platform)
        print('Missing hydrogens added!', flush=True)
        app.PDBFile.writeFile(modeller.topology, modeller.positions, open(f'{self.output_dir}/tmp/_protein.pdb', 'w'))
        print('Protein is saved in tmp/_protein.pdb!', flush=True)
        print('Protein is loaded!\n', flush=True)

    def _load_ligand(self, ligand_file):
        '''
            Load and process the ligand file. 
            Used during initialization.

            Args:
                ligand_file (str): the path to the ligand file.
        '''
        # load the ligand file
        print('\nLoading the ligand file...', flush=True)
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
        ligand.to_file(f'{self.output_dir}/tmp/_ligand.mol2', file_format='MOL')
        print('Ligand is saved in tmp/_ligand.mol2!', flush=True)
        print('Ligand is loaded!\n', flush=True)

    def _save_amber_files(self, topology, system, positions, gb_radii='mbondi2'):
        '''
            Save the topology, system, and positions in amber format. 
            filename should not include the extension. This functions will
            automatically save prmtop and inpcrd files.
            Used by the _prepare_simulation() method.

            Args:
                topology (openmm.Topology): the topology of the system.
                system (openmm.System): the system object.
                positions (openmm.Positions): the positions of the system.
                gb_radii (str): the radii to use for the Generalized Born model.
                    Allowed values: amber6, bondi, mbondi, mbondi2, mbondi3
        '''
        if gb_radii not in ['amber6', 'bondi', 'mbondi', 'mbondi2', 'mbondi3']:
            raise ValueError('gb_radii should be one of the following: amber6, bondi, mbondi, mbondi2, mbondi3')
        # save the solvated complex as prmtop and inpcrd files
        struct = pmd.openmm.load_topology(topology, system, positions)
        amber_parm = pmd.amber.AmberParm.from_structure(struct)
        radii_update = pmd.tools.actions.changeRadii(amber_parm, gb_radii)
        radii_update.execute()
        amber_parm.write_parm(f'{self.output_dir}/tmp/_complex_solvated.prmtop')
        print('Solvated complex is saved as prmtop!', flush=True)

    def _prepare_system_topology(
            self,
            proteins,
            forcefield,
            ligands=None,
            solvate=True,
            solvation_kwargs={'model': 'tip3p', 
                              'padding': 1.0*unit.nanometer, 
                              'positiveIon': 'Na+',
                              'negativeIon': 'Cl-',
                              'ionicStrength': 0.0*unit.molar,
                              'neutralize': True,}
    ):
        '''
            Prepare the system and topology of the complex.
            Used by the _prepare_simulation() method.

            Args:
                proteins (list): the list of protein objects.
                forcefield (str): the ForceField to use for determining van der Waals radii 
                    and atomic charges by modeller.AddSolvent() method. Ignored if solvate=False.
                ligands (list): the list of ligand objects.
                solvate (bool): whether to solvate the system or not.
                solvation_kwargs (dict): the keyword arguments for solvating the system.
                    refer to the documentation of openmm Modeller.addSolvent() for more information.

            Returns:
                modeller (openmm.Modeller): the modeller object that contains the topology 
                    and positions of the system.
        '''
        # use modeller to create topology of the system
        modeller = app.Modeller(proteins[0].topology, proteins[0].positions)
        for protein in proteins[1:]:
            modeller.add(protein.topology, protein.positions)
        if ligands is not None:
            for ligand in ligands:
                ligand_topology = ligand.to_topology()
                modeller.add(ligand_topology.to_openmm(), ligand_topology.get_positions().to_openmm())
        app.PDBFile.writeFile(modeller.topology, modeller.positions, open(f'{self.output_dir}/tmp/_complex.pdb', 'w'))
        print('Complex is saved in tmp/_complex.pdb!', flush=True)

        # solvate the complex
        if solvate:
            print('Solvating the complex...', flush=True)
            modeller.addSolvent(forcefield=forcefield, **solvation_kwargs)
            app.PDBFile.writeFile(modeller.topology, modeller.positions, open(f'{self.output_dir}/tmp/_complex_solvated.pdb', 'w'))
            print('Solvated Complex is saved in tmp/_complex_solvated.pdb!', flush=True)
            print('Complex is solvated!', flush=True)

        return modeller

    def remove_tmp_files(self):
        '''
            Remove the temporary files.
        '''
        # remove the tmp directory
        os.system(f'rm -rf {self.output_dir}/tmp')

    def _constrain_backbone(self, system: mm.System, positions, atoms, k):
        '''
            Constrain the protein.
            This function will add a new CustomExternalForce to the system, 
            thus the systen must be reinitialized after calling this function.

            Args:
                system (openmm.System): the system object.
                positions: the positions of the system. simulation.context.getState(getPositions=True).getPositions()
                atoms: the list of atoms to constrain. simulation.topology.atoms()
                k (float): the force constant to use for the constraint in kJ/mol.
        '''
        force = app.CustomExternalForce('k*periodicdistance(x, y, z, x0, y0, z0)^2')
        restraint_force = k * unit.kilojoules_per_mole / unit.angstroms**2
        force.addGlobalParameter('k', restraint_force)
        force.addPerParticleParameter('x0')
        force.addPerParticleParameter('y0')
        force.addPerParticleParameter('z0')
        for coords, atom in zip(positions, atoms):
            if atom.name in ['N', 'CA', 'C', 'O']:
                force.addParticle(int(atom.index), coords)
        system.addForce(force)
        
    def _prepare_simulation(
            self,
            proteins,
            ligands=None,
            integrator=mm.LangevinMiddleIntegrator(300*unit.kelvin, 1.0/unit.picosecond, 0.002*unit.picoseconds),
            additional_forces=[mm.MonteCarloBarostat(1.0*unit.atmosphere, 300*unit.kelvin, frequency=25)],
            forcefields=['amber14/protein.ff14SB.xml', 'amber14/tip3p.xml'],
            ligand_forcefield='gaff-2.11',
            solvate=True,
            solvent_kwargs={'model': 'tip3p', 'padding': 1.0*unit.nanometer, 'positiveIon': 'Na+',
                            'negativeIon': 'Cl-', 'ionicStrength': 0.0*unit.molar, 'neutralize': True,},
            forcefield_kwargs={'constraints': None, 'rigidWater': False, 'removeCMMotion': False},
            nonperiodic_forcefield_kwargs={'nonbondedMethod' : app.NoCutoff},
            periodic_forcefield_kwargs={'nonbondedMethod' : app.PME},
            gb_radii='mbondi2',
    ):
        '''
            Prepare the simulation. This will return a 
            simulation object that is ready to run simulations.
            Used by the run_simulation() method.

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
                    Including constraints and rigidWater may cause problems, avoid using them.
                nonperiodic_forcefield_kwargs (dict): the keyword arguments for the nonperiodic topology.
                periodic_forcefield_kwargs (dict): the keyword arguments for the periodic topology.
                gb_radii (str): the radii to use when saving the solvated complex prmtop file.
                    Allowed values: amber6, bondi, mbondi, mbondi2, mbondi3

            returns:
                simulation (openmm.Simulation): the simulation object.
        '''
        # create system generator for simulation
        system_generator = SystemGenerator(
            forcefields=forcefields,
            small_molecule_forcefield=ligand_forcefield,
            molecules=ligands,
            forcefield_kwargs=forcefield_kwargs,
            nonperiodic_forcefield_kwargs=nonperiodic_forcefield_kwargs,
            periodic_forcefield_kwargs=periodic_forcefield_kwargs,
        )

        # prepare the system and topology
        modeller = self._prepare_system_topology(
            proteins=proteins,
            ligands=ligands,
            forcefield=system_generator.forcefield,
            solvate=solvate,
            solvation_kwargs=solvent_kwargs,
        )

        # create system
        system = system_generator.create_system(modeller.topology, molecules=ligands)
        integrator = mm.LangevinMiddleIntegrator(300*unit.kelvin, 1.0/unit.picosecond, 0.002*unit.picoseconds)
        for force in additional_forces:
            system.addForce(force)

        # create simulation
        simulation = app.Simulation(modeller.topology, system, integrator, platform=self.platform)
        simulation.context.setPositions(modeller.positions)

        # save the solvated complex as prmtop files
        self._save_amber_files(modeller.topology, system, modeller.positions, gb_radii=gb_radii)

        return simulation
    
    def _minimize(self, simulation: app.Simulation, reporters: list, max_iterations: int=10000):
        '''
            Minimize the energy of the system.

            Args:
                simulation (openmm.Simulation): the simulation object.
                reporters (list): the list of reporters to use during the minimization.
                max_iterations (int): the maximum number of iterations to run the minimization for.

            Returns:
                simulation (openmm.Simulation): the simulation object.
        '''
        # minimize the energy
        [simulation.reporters.append(reporter) for reporter in reporters]
        simulation.minimizeEnergy(maxIterations=max_iterations)
        positions = simulation.context.getState(getPositions=True).getPositions()
        app.PDBFile.writeFile(simulation.topology, positions, open(f'{self.output_dir}/tmp/_minimized.pdb', 'w'))
        simulation.reporters.clear()
    
    def _equilibrate_nvt(self, simulation: app.Simulation, reporters: list):
        '''
            Equilibrate the system in the NVT ensemble for 50 ps.

            Args:
                simulation (openmm.Simulation): the simulation object.
                reporters (list): the list of reporters to use during the equilibration.
        '''
        # equilibrate in the NVT ensemble
        [simulation.reporters.append(reporter) for reporter in reporters]
        temperature = 1e-6
        simulation.context.setVelocitiesToTemperature(temperature*unit.kelvin)
        for _ in range(5000):
            simulation.step(5)
            temperature += 0.012*5
            simulation.integrator.setTemperature(temperature*unit.kelvin)
        positions = simulation.context.getState(getPositions=True).getPositions()
        app.PDBFile.writeFile(simulation.topology, positions, open(f'{self.output_dir}/tmp/_nvt_equilibrated.pdb', 'w'))
        simulation.reporters.clear()
    
    def _equilibrate_npt(self, simulation: app.Simulation, reporters: list):
        '''
            Equilibrate the system in the NPT ensemble for 50 ps.

            Args:
                simulation (openmm.Simulation): the simulation object.
                reporters (list): the list of reporters to use during the equilibration.
                num_steps (int): the number of steps to run the equilibration for.
        '''
        # equilibrate in the NPT ensemble
        [simulation.reporters.append(reporter) for reporter in reporters]
        simulation.system.addForce(mm.MonteCarloBarostat(1.0*unit.atmosphere, 300*unit.kelvin))
        simulation.context.reinitialize(True)
        simulation.step(25000)
        positions = simulation.context.getState(getPositions=True).getPositions()
        app.PDBFile.writeFile(simulation.topology, positions, open(f'{self.output_dir}/tmp/_npt_equilibrated.pdb', 'w'))
        simulation.reporters.clear()
        simulation.system.removeForce(simulation.system.getNumForces() - 1)
        simulation.context.reinitialize(True)
            
    def run_simulation(
            self,
            num_steps=1000000,
            minimize=True,
            nvt_equilibration=True,
            npt_equilibration=True,
            sim_reporters=[app.StateDataReporter(sys.stdout, 1000, potentialEnergy=True, temperature=True), app.DCDReporter('tmp/_sim_trajectory.dcd', 1000)],
            min_reporters=[app.StateDataReporter(sys.stdout, 1000, potentialEnergy=True, temperature=True), app.DCDReporter('tmp/_min_trajectory.dcd', 1000)],
            nvt_reporters=[app.StateDataReporter(sys.stdout, 1000, potentialEnergy=True, temperature=True), app.DCDReporter('tmp/_nvt_trajectory.dcd', 1000)],
            npt_reporters=[app.StateDataReporter(sys.stdout, 1000, potentialEnergy=True, temperature=True), app.DCDReporter('tmp/_npt_trajectory.dcd', 1000)],
            integrator=mm.LangevinMiddleIntegrator(300*unit.kelvin, 1.0/unit.picosecond, 0.002*unit.picoseconds),
            additional_forces=[mm.MonteCarloBarostat(1.0*unit.atmosphere, 300*unit.kelvin)],
            forcefields=['amber14/protein.ff14SB.xml', 'amber14/tip3p.xml'],
            ligand_forcefield='gaff-2.11',
            solvate=True,
            solvent_kwargs={'model': 'tip3p', 'padding': 1.0*unit.nanometers},
            forcefield_kwargs={'constraints': None, 'rigidWater': False, 'removeCMMotion': False},
            nonperiodic_forcefield_kwargs={'nonbondedMethod' : app.NoCutoff},
            periodic_forcefield_kwargs={'nonbondedMethod' : app.PME},
    ):
        '''
            Run the simulation.

            Args:
                num_steps (int): the number of steps to run the simulation for.
                minimize (bool): whether to minimize the energy or not, before the simulation.
                nvt_equilibration (bool): whether to run NVT equilibration or not.
                    Heats the system from 0 to 300 K over a period of 50 ps. 
                npt_equilibration (bool): whether to run NPT equilibration or not.
                    Equilibrates the system for 50 ps in the NPT ensemble.
                    The protein is constrained during the equilibration by 
                    the strength of 2 kcal/mol·Å2. 
                sim_reporters (list): the list of reporters to use during the main simulation.
                min_reporters (list): the list of reporters to use during the minimization.
                nvt_reporters (list): the list of reporters to use during the NVT equilibration.
                npt_reporters (list): the list of reporters to use during the NPT equilibration.
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
        '''
        # prepare the simulation
        simulation = self._prepare_simulation(
            proteins=self.proteins,
            ligands=self.ligands,
            integrator=integrator,
            additional_forces=additional_forces,
            forcefields=forcefields,
            ligand_forcefield=ligand_forcefield,
            solvate=solvate,
            solvent_kwargs=solvent_kwargs,
            forcefield_kwargs=forcefield_kwargs,
            nonperiodic_forcefield_kwargs=nonperiodic_forcefield_kwargs,
            periodic_forcefield_kwargs=periodic_forcefield_kwargs,
        )

        # minimize the energy
        if minimize:
            start_time = time.time()
            print('\nMinimizing the energy...', flush=True)
            self._minimize(simulation, min_reporters)
            print(f'Structure is minimized in {time.time() - start_time} seconds and saved in _minimized.pdb\n!', flush=True)

        if nvt_equilibration or npt_equilibration:
            # constrain the protein
            self._constrain_backbone(
                system=simulation.system,
                positions=simulation.context.getState(getPositions=True).getPositions(),
                atoms=simulation.topology.atoms(),
                k=2.0,
            )

        # nvt equilibration
        if nvt_equilibration:
            start_time = time.time()
            print('\nRunning NVT equilibration...', flush=True)
            self._equilibrate_nvt(simulation, nvt_reporters)
            print(f'NVT equilibration is finished in {time.time() - start_time} seconds and saved in _nvt_equilibrated.pdb\n!', flush=True)

        # npt equilibration
        if npt_equilibration:
            start_time = time.time()
            print('\nRunning NPT equilibration...', flush=True)
            self._equilibrate_npt(simulation, npt_reporters)
            print(f'NPT equilibration is finished in {time.time() - start_time} seconds and saved in _npt_equilibrated.pdb\n!', flush=True)
        
        if nvt_equilibration or npt_equilibration:
            # remove the constraints
            simulation.system.removeForce(simulation.system.getNumForces() - 1)
            simulation.context.reinitialize(True)

        # run the simulation
        start_time = time.time()
        print('\nRunning the simulation...', flush=True)
        [simulation.reporters.append(reporter) for reporter in sim_reporters]
        simulation.step(num_steps)
        print(f'System is simulated for {0.002*simulation.currentStep/1000} ns!', flush=True)

        # save the final state
        print('Saving the final state...', flush=True)
        simulation.saveState(f'{self.output_dir}/tmp/_final_state.xml')
        print('Final state is saved!', flush=True)
        print(f'Simulation time: {(time.time()-start_time)/60} minutes', flush=True)
        print('\n----------------------------------------\n\n', flush=True)
