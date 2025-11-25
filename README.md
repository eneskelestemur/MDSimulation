# MDSim on OpenMM

Config-driven MD runner: prepare structures, solvate, run OpenMM, then analyze and visualize with MDAnalysis.

## Install
```bash
pip install -e .
```

## Run at the CLI
```bash
cp config_template.yaml my_run.yaml   # edit this file
mdsim run my_run.yaml                 # prep + simulation
mdsim analyze outputs/<run_name>      # analyses (RMSD/RMSF/pairwise/contacts/MMGBSA)
mdsim visualize outputs/<run_name>    # plots from analysis outputs
```
`mdsim analyze`/`visualize` read `outputs/<run_name>/config_resolved.yaml`.

## What the config controls
- **system**: input structures (protein/ligand/RNA), forcefields.
- **prep**: fix missing residues/atoms, protonate, strip heterogens.
- **solvation**: box padding, ions, water model.
- **simulation**: integrator/constraints/cutoffs, protocol (minimize/NVT/NPT/production), reporting/checkpoints.
- **analysis**: transformations (unwrap/center/wrap once, save transformed traj), RMSD/RMSF (multi-selection), pairwise RMSD, contacts, MMGBSA.
- **visualization**: figure DPI, RMSF aggregation per selection, pairwise heatmap tick density.
- Full schema: `config_template.yaml`.

## Outputs
```
outputs/<run_name>/
├── prep/                  # cleaned/solvated structures
├── sim/                   # trajectories, logs, states, final PDB
│   ├── sim_trajectory.dcd
│   ├── sim_trajectory_transformed.<ext>   # when transforms enabled
│   ├── sim_state.log, minimized_complex.pdb, final_state.xml, ...
├── analysis/              # CSVs: rmsd, rmsf, pairwise_rmsd, contacts, mmgbsa
├── visuals/               # PNG plots: state, rmsd, rmsf, pairwise_rmsd, contacts
├── cache/, tmp/           # optional caches/scratch
├── config_resolved.yaml   # resolved config used for the run
└── manifest.json          # key output paths
```

## Python API
```python
from pathlib import Path
from mdsim.config import load_run_config
from mdsim.workflow import MDWorkflow
from mdsim.analysis.workflow import AnalysisWorkflow

cfg = load_run_config(Path("my_run.yaml"))
wf = MDWorkflow(cfg, config_path=Path("my_run.yaml"))
result = wf.run()                 # prep + simulation

awf = AnalysisWorkflow(result.run_dir)
awf.run()                         # analyses (uses config_resolved.yaml)
```

## Notes
- Transformations: if enabled, we unwrap/center/wrap once, write `sim_trajectory_transformed.<ext>`, reload, and run all analyses on the transformed coords (no per-analysis transform overhead).
- Analyses use MDAnalysis selections (e.g., `backbone`, `resname UNK`, `protein and name CA`).
- RMSD/RMSF support multiple selections in one CSV/plot.

For selection syntax see MDAnalysis docs; for MMGBSA masks/parameters see Amber MMGBSA guidance.
