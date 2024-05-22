'''
    Simulate protein-protein or protien-ligand complexes using simulation class.
'''

from simulation import Simulation

## An example of simulating a protein-ligand complex using the Simulation class
# set output directory
output_dir = 'chikv_nsp2_helicase_rna'

# plots for inital equilibration
# Simulation.plot_StateData(f'{output_dir}/equil_data.log',
#                             ['Potential Energy (kJ/mole)', 'Total Energy (kJ/mole)', 'Temperature (K)', 'Density (g/mL)'],
#                             save=True, show=False)
# Simulation.plot_RMSD(f'{output_dir}/equil_trajectory.dcd', f'{output_dir}/minimized_complex.pdb',
#                         labels=['Backbone', 'Protein', 'RNA', r'$C_{\alpha}$'],
#                         rmsd_kwargs={'select': 'backbone', 'groupselections': ['segid A', 'segid B', 'name CA']}, 
#                         save=True)

# # plots for main simulation
# Simulation.plot_StateData(f'{output_dir}/sim_data.log',
#                             ['Potential Energy (kJ/mole)', 'Total Energy (kJ/mole)', 'Temperature (K)', 'Density (g/mL)'],
#                             save=True, show=False)
# Simulation.plot_RMSD(f'{output_dir}/sim_trajectory.dcd', f'{output_dir}/minimized_complex.pdb',
#                         labels=['Backbone', 'Protein', 'RNA', r'$C_{\alpha}$'],
#                         rmsd_kwargs={'select': 'backbone', 'groupselections': ['segid A', 'segid B', 'name CA']},
#                         save=True)
# Simulation.plot_RMSF(f'{output_dir}/sim_trajectory.dcd', 
#                      f'{output_dir}/minimized_complex.pdb',
#                      segid='A', save=True)
Simulation.plot_RMSF(f'{output_dir}/sim_trajectory.dcd',
                     f'{output_dir}/minimized_complex.pdb',
                     segid='B', save=True)
Simulation.plot_pairwise_rmsd(f'{output_dir}/sim_trajectory.dcd',
                              f'{output_dir}/minimized_complex.pdb', 
                              select2align='backbone', select2calc='segid A', save=True)
Simulation.plot_pairwise_rmsd(f'{output_dir}/sim_trajectory.dcd', 
                              f'{output_dir}/minimized_complex.pdb', 
                              select2align='backbone', select2calc='segid B', save=True)

