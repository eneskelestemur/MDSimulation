'''
    Simulate protein-protein or protien-ligand complexes using simulation class.
'''

# import the required modules
import openmm.app as app
import openmm as mm
import parmed as pmd

from openmm import unit
from simulation import Simulation


## An example of simulating a protein-ligand complex using the Simulation class
# set output directory
output_dir = 'estrogen_ral_example'

# create the simulation object
complex_sim = Simulation(
    protein_files=[f'{output_dir}/Estrogen_Receptor.pdb'],
    ligand_files=[f'{output_dir}/Raloxifene.sdf'],
    platform='CPU',
    output_dir=output_dir,
    remove_tmp_files=False,
)
# run the simulation
complex_sim.run_simulation(
    num_steps=1000000,
    minimize=True,
    nvt_equilibration=True,
    npt_equilibration=True,
    sim_reporters=[app.StateDataReporter(f'{output_dir}/sim_data.log', 1000, step=True, potentialEnergy=True, totalEnergy=True, temperature=True, density=True), 
                    app.DCDReporter(f'{output_dir}/sim_trajectory.dcd', 1000), pmd.openmm.MdcrdReporter(f'{output_dir}/sim.mdcrd', 1000, crds=True)],
    equil_reporters=[app.StateDataReporter(f'{output_dir}/equil_data.log', 100, step=True, potentialEnergy=True, totalEnergy=True, temperature=True, density=True), 
                        app.DCDReporter(f'{output_dir}/equil_trajectory.dcd', 100), pmd.openmm.MdcrdReporter(f'{output_dir}/equil.mdcrd', 1000, crds=True)],
    integrator=mm.LangevinMiddleIntegrator(300*unit.kelvin, 1.0/unit.picosecond, 0.002*unit.picoseconds),
    additional_forces=[mm.MonteCarloBarostat(1.0*unit.atmosphere, 300*unit.kelvin)],
    forcefields=['amber14/protein.ff14SB.xml', 'amber14/tip3p.xml'],
    ligand_forcefield='gaff-2.11',
    solvate=True,
    solvent_kwargs={'model': 'tip3p', 'padding': 1.0*unit.nanometers},
    forcefield_kwargs={'constraints': None, 'rigidWater': False, 'removeCMMotion': False},
    nonperiodic_forcefield_kwargs={'nonbondedMethod' : app.NoCutoff},
    periodic_forcefield_kwargs={'nonbondedMethod' : app.PME},
)
# calculate the MMGBSA
complex_sim.calculate_mmgbsa()

# plots for inital equilibration
Simulation.plot_StateData(f'{output_dir}/equil_data.log',
                            ['Potential Energy (kJ/mole)', 'Total Energy (kJ/mole)', 'Temperature (K)', 'Density (g/mL)'],
                            save=True, show=False)
Simulation.plot_RMSD(f'{output_dir}/equil_trajectory.dcd', f'{output_dir}/simulated_complex.pdb',
                        labels=['Backbone', 'Protein', 'Ligand', r'$C_{\alpha}$'],
                        rmsd_kwargs=None, save=True)

# plots for main simulation
Simulation.plot_StateData(f'{output_dir}/sim_data.log',
                            ['Potential Energy (kJ/mole)', 'Total Energy (kJ/mole)', 'Temperature (K)', 'Density (g/mL)'],
                            save=True, show=False)
Simulation.plot_RMSD(f'{output_dir}/sim_trajectory.dcd', f'{output_dir}/simulated_complex.pdb',
                        labels=['Backbone', 'Protein', 'Ligand', r'$C_{\alpha}$'],
                        rmsd_kwargs=None, save=True)
Simulation.plot_RMSF(f'{output_dir}/sim_trajectory.dcd', 
                        f'{output_dir}/simulated_complex.pdb',
                        segid='A', save=True)
Simulation.plot_pairwise_rmsd(f'{output_dir}/sim_trajectory.dcd', 
                                f'{output_dir}/simulated_complex.pdb', 
                                select2align='backbone', select2calc='segid A', save=True)
Simulation.plot_pairwise_rmsd(f'{output_dir}/sim_trajectory.dcd', 
                                f'{output_dir}/simulated_complex.pdb', 
                                select2align='backbone', select2calc='segid B', save=True)
