# MDSim: Modular MD runner on OpenMM

Single-system MD workflow that builds/cleans structures, optionally solvates, and runs a configurable OpenMM protocol from a YAML file.

## Quick start

```bash
pip install -e .
cp config_template.yaml my_run.yaml
# edit my_run.yaml
mdsim run my_run.yaml
```

## Config highlights (single system)

```yaml
run_name: "example"
output_root: "outputs"

system:
  name: "complex"
  protein_paths: ["inputs/protein.pdb"]
  ligand_paths: ["inputs/ligand.sdf"]
  rna_paths: []
  forcefields: ["amber14-all.xml", "amber14/tip3pfb.xml"]

prep:
  fix_missing_residues: true
  fix_missing_atoms: true
  add_hydrogens_pH: 7.0

solvation:
  enabled: true
  water_model: "tip3p"
  padding_nm: 1.0
  ionic_strength_m: 0.15

simulation:
  platform: "CUDA"
  temperature: 300.0
  friction: 1.0
  timestep_fs: 2.0
  protocol:
    minimize: { enabled: true, steps: 5000 }
    nvt:      { enabled: true, steps: 50000 }
    npt:      { enabled: true, steps: 100000 }
    production_steps: 1000000
  integrator: { type: "LangevinMiddle", seed: null }
  engine:
    constraints: "HBonds"
    cutoff_nm: 1.0
    switch_distance_nm: 0.9
    barostat_interval: 25
    ligand_forcefield: "gaff-2.11"
  reporting:
    equilibration: { interval: 1000 }
    production:    { interval: 1000 }
    checkpoint_interval: 10000

output:
  keep_tmp: false
  keep_cache: false
```

See `config_template.yaml`, `example_advanced.yaml`, and `estrogen_ral_ex/estrogen_ral.yaml` for full schemas and defaults.

## Output layout

```
outputs/<run_name>/
├── prep/                 # cleaned inputs, complex.pdb, complex_solvated.pdb
├── sim/                  # logs, trajectories, state XML, final PDB
├── cache/                # ligand template cache (removed if keep_cache: false)
├── tmp/                  # scratch (removed if keep_tmp: false)
├── config_resolved.yaml  # copy of input config
└── manifest.json         # key output paths
```
