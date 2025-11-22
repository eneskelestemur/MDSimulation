'''
    Simulate protein-protein or protien-ligand complexes using simulation class.
'''

# import the required modules
import openmm.app as app
import openmm as mm
import parmed as pmd

from openmm import unit
from legacy.simulation import Simulation


## An example of simulating a protein-ligand complex using the Simulation class
# set output directory
output_dir = 'sarsCov2_rdrp_ra2112_0'
protein_files = [
    f'{output_dir}/7ED5_chainA.pdb', 
    # f'{output_dir}/7ED5_chainB.pdb',
    # f'{output_dir}/7ED5_chainC.pdb', 
    # f'{output_dir}/7ED5_chainD.pdb',
]
ligand_files = [
    f'{output_dir}/ra2112_0.mol2'
]
rna_files = [
    f'{output_dir}/7ED5_chainI_rna.pdb', 
    f'{output_dir}/7ED5_chainJ_rna.pdb'
]

# create the simulation object
complex_sim = Simulation(
    protein_files=protein_files,
    ligand_files=ligand_files,
    rna_files=None,
    platform='CUDA', # 'CUDA' or 'CPU' or 'OpenCL'
    output_dir=output_dir,
    remove_tmp_files=False,
)

# run the simulation
complex_sim.run_simulation(
    num_steps=500000000,
    minimize=True,
    nvt_equilibration=True,
    npt_equilibration=True,
    sim_reporters=[app.StateDataReporter(f'{output_dir}/sim_data.log', 10000, step=True, potentialEnergy=True, totalEnergy=True, temperature=True, density=True), 
                   app.DCDReporter(f'{output_dir}/sim_trajectory.dcd', 100000)],
    equil_reporters=[app.StateDataReporter(f'{output_dir}/equil_data.log', 100, step=True, potentialEnergy=True, totalEnergy=True, temperature=True, density=True), 
                     app.DCDReporter(f'{output_dir}/equil_trajectory.dcd', 1000)],
    integrator=mm.LangevinMiddleIntegrator(300*unit.kelvin, 1.0/unit.picosecond, 0.002*unit.picoseconds),
    additional_forces=[mm.MonteCarloBarostat(1.0*unit.atmosphere, 300*unit.kelvin)],
    forcefields=['amber14/protein.ff14SB.xml', 'amber14/tip3p.xml', 'amber14/RNA.OL3.xml'],
    ligand_forcefield='gaff-2.11',
    solvate=True,
    solvent_kwargs={'model': 'tip3p', 'padding': 0.5*unit.nanometer, 'positiveIon': 'Na+',
                    'negativeIon': 'Cl-', 'ionicStrength': 0.025*unit.molar, 'neutralize': True,},
    forcefield_kwargs={'constraints': None, 'rigidWater': False, 'removeCMMotion': False},
    nonperiodic_forcefield_kwargs={'nonbondedMethod' : app.NoCutoff},
    periodic_forcefield_kwargs={'nonbondedMethod' : app.PME},
)
# calculate the MMGBSA
# complex_sim.calculate_mmgbsa()
