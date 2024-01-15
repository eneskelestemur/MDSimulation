# MMGBSA/MMPBSA in OpenMM

Amber has utilities for running + analyzing MMGBSA simulations ([here](https://ambermd.org/tutorials/advanced/tutorial3/) is a tutorial), but they are annoying to use. The `md_sim.py` script here runs the simulation in OpenMM and analyses the results using amber's tools. If you run the script now, it will run MMGBSA on the example complex in this folder. As you can see at the bottom of the file, there are two functions you need to use:

```python
if __name__ == '__main__':
    out_dir = 'example_results'
    # first simulate the complex. This will take a while.
    simulate_complex('1uom_A_rec.pdb', '1uom_pti_lig.sdf', out_dir)
    # now calculate MMGBSA from the simulation results. This stores everything to
    # {out_dir}/mmgbsa_results.dat. This is a plan text file that should be pretty easy
    # to parse. 
    calculate_mmgbsa(out_dir)
```

## Logging into highgarden

I've set up accounts for y'all on highgarden. You can access them via: 
```bash
ssh {your_username}@152.2.40.219
```

If that command doesn't work, it may be because you're off the UNC network. In that case, you'll need to connect to the UNC VPN, ssh into longleaf, and ssh into highgarden via the longleaf connection. This is super annoying and I'm not sure why this is.

Once you're on, always cd to the shared `/tropsha` directory. Make sure to do **everything** in this folder! We have very little space on highgarden and `/tropsha` is the only folder pointing to an external hard drive. Putting even small files in your home directory (or anywhere else) could result in weird errors for you and everyone else using highgarden.

I've already cloned this repo in `/tropsha/MDSimulation`; feel free to use that. Note that both of you are using this same repo which is generally not advised; you can create a separate copy for each of you if you want instead.

## Using multiple GPUs in highgarden

Highgarden has 4 GPUs. If you run most programs (including `md_sim.py`), it will by default only use one GPU (`0`). The simplest way to use another GPU is to modify the `CUDA_VISIBLE_DEVICES` environment variable. For instance, running

```bash
CUDA_VISIBLE_DEVICES=2 python md_sim.py
```

will force it to use the third GPU (GPU `2`) by making that GPU the only one visible to the program. Running all the complexes in the set one one GPU will take too long, so make sure to use multiple! For instance, you could write a script to run MMGBSA on a single target in the benchmarking set, and then run the script three times simultaneously in three separate `tmux` sessions with different visible GPUs. Since there are 3 targets, this strategy will use 3 GPUs. I will leave it up to you to figure out how to use all 4, if you so desire.

## Running the example code

On highgarden, I've set up a conda environment for y'all to use (`mm`). You should be able to run the example code with the following:
```bash
conda activate mm
python md_sim.py
```

## Next steps

1. Make sure you can simulate the example complex in this repo. This might take a while! Once this works, write a function to analyze the resulting `.dat` files and get the $\Delta G$ on the complex.
2. Start by running this on just the `actives` for a target of your choice in the benchmarking set (I've posted the set in the slack). Plot the resulting delta Gs vs the experimental `pchembl_value` (-log Kd) of the active molecules. (You can find these `pchembl_values` in the `actives.csv` for the target). They should be negatively correlated! Check the correlation coefficient between the values.
3. Now that you've done this for the actives, try it for the random compounds for the same target. Now plot the distribution of MMGBSA delta Gs for both the actives and random. The active delta Gs should be lower! (We don't have experimental binding affinities for the random compounds, but they will be unlikely to bind).
4. Repeat for the other targets.
5. Get the Expected Enrichment Factors for all the targets using EEF code that I will provide shortly...
