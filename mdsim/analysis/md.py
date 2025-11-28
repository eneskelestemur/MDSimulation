from __future__ import annotations

"""
MD trajectory analyses (RMSD, RMSF, pairwise RMSD matrix/time series, contacts) using MDAnalysis.

All routines emit CSV outputs (no plotting). Selections use the MDAnalysis selection language.
"""

from pathlib import Path
from typing import Optional

import MDAnalysis as mda
import numpy as np
import pandas as pd
from MDAnalysis.analysis import rms
from MDAnalysis.analysis.diffusionmap import DistanceMatrix
from MDAnalysis.lib import distances

from ..logging_setup import get_logger

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #


def load_universe(topology: Path, trajectory: Path) -> mda.Universe:
    """
    Load an MDAnalysis Universe from a topology and trajectory.

    Args:
        topology: Path to topology (e.g., PDB).
        trajectory: Path to trajectory (e.g., DCD).

    Returns:
        MDAnalysis Universe with trajectory loaded.
    """
    return mda.Universe(str(topology), str(trajectory))


def load_reference(reference: Path) -> mda.Universe:
    """
    Load a reference structure as an MDAnalysis Universe.

    Args:
        reference: Path to reference PDB/structure.

    Returns:
        MDAnalysis Universe for the reference.
    """
    return mda.Universe(str(reference))


def _frame_indices(n_frames: int, stride: int, max_frames: Optional[int]) -> list[int]:
    """Return frame indices respecting stride and optional cap."""
    stride = max(1, stride)
    idxs = list(range(0, n_frames, stride))
    if max_frames is not None:
        idxs = idxs[: max_frames]
    return idxs


def _select_atoms_checked(universe: mda.Universe, selection: str) -> mda.core.groups.AtomGroup:
    """Select atoms and ensure the selection is non-empty."""
    ag = universe.select_atoms(selection)
    if len(ag) == 0:
        raise ValueError(f"Selection {selection} returned zero atoms.")
    return ag


# --------------------------------------------------------------------------- #
# Analyses
# --------------------------------------------------------------------------- #


def compute_rmsd(
    mobile: mda.Universe,
    reference: mda.Universe,
    target_sel: str,
    stride: int,
    max_frames: Optional[int],
    out_csv: Optional[Path],
    label: str,
) -> pd.DataFrame:
    """
    Compute RMSD against a reference for a target selection and optionally write CSV.

    Args:
        mobile: Universe containing trajectory to analyze.
        reference: Reference structure Universe.
        target_sel: Selection string to compute RMSD on.
        stride: Subsample frames every N steps.
        max_frames: Cap number of frames (after stride); None for all.
        out_csv: Output CSV path (if None, results are not written).
        label: Label stored alongside results.
    """
    stop = stride * max_frames if max_frames is not None else None
    calc = rms.RMSD(
        mobile,
        reference,
        select=target_sel,
        ref_frame=0,
        start=0,
        stop=stop,
        step=stride,
    )
    calc.run()
    arr = calc.rmsd  # columns: frame, time(ps), RMSD(Å)
    df = pd.DataFrame(
        {
            "frame": arr[:, 0].astype(int),
            "time_ns": arr[:, 1] / 1000.0,
            "rmsd_angstrom": arr[:, 2],
            "label": label,
        }
    )
    if out_csv is not None:
        df.to_csv(out_csv, index=False)
        logger.info("Wrote RMSD to %s", out_csv)
    return df


def compute_rmsf(
    mobile: mda.Universe,
    target_sel: str,
    stride: int,
    max_frames: Optional[int],
    out_csv: Optional[Path],
    label: str,
) -> pd.DataFrame:
    """
    Compute RMSF for a selection using MDAnalysis RMSF (optionally aligned) and optionally write CSV.

    Args:
        mobile: Universe containing trajectory to analyze.
        target_sel: Selection string to compute RMSF on.
        stride: Subsample frames every N steps.
        max_frames: Cap number of frames (after stride); None for all.
        out_csv: Output CSV path (if None, results are not written).
        label: Label stored alongside results.
    """
    # Work on a copy so later steps do not mutate the original Universe
    working = mobile.copy()
    atoms = _select_atoms_checked(working, target_sel)

    # Limit frames
    if max_frames is not None:
        stop = min(working.trajectory.n_frames, stride * max_frames)
    else:
        stop = None

    # Compute RMSF using MDAnalysis
    rmsf_res = rms.RMSF(atoms)
    rmsf_res.run(start=0, stop=stop, step=stride)
    rmsf_vals = rmsf_res.rmsf

    df = pd.DataFrame(
        {
            "atom_index": np.arange(len(atoms)),
            "rmsf_angstrom": rmsf_vals,
            "label": label,
            "residue_index": atoms.resindices,
            "residue_id": atoms.resnums,
            "residue_name": atoms.resnames,
        }
    )
    if out_csv is not None:
        df.to_csv(out_csv, index=False)
        logger.info("Wrote RMSF to %s", out_csv)
    return df


def compute_pairwise_rmsd(
    mobile: mda.Universe,
    selection: str,
    stride: int,
    max_frames: Optional[int],
    out_csv: Path,
) -> None:
    """
    Compute pairwise RMSD and write CSV.

    Outputs a symmetric RMSD matrix between frames for the given selection.
    Time is reported in ns.

    Args:
        mobile: Universe containing trajectory to analyze.
        selection: Selection string used for all frames.
        stride: Subsample frames every N steps.
        max_frames: Cap number of frames (after stride); None for all.
        out_csv: Output CSV path.
    """
    # Use MDAnalysis DistanceMatrix with RMSD metric; assumes any alignment handled upstream.
    total_frames = mobile.trajectory.n_frames
    stop = stride * max_frames if max_frames is not None else None
    dm = DistanceMatrix(mobile, select=selection, metric=rms.rmsd, start=0, stop=stop, step=stride)
    dm.run()
    matrix = dm.results.dist_matrix
    # Build labels based on the frames used
    frame_indices = _frame_indices(total_frames, stride, max_frames)
    times_ns = []
    for i in frame_indices:
        mobile.trajectory[i]
        times_ns.append(mobile.trajectory.time / 1000.0)
    labels = [f"time_{t:.3f}_ns" for t in times_ns]
    df = pd.DataFrame(matrix, columns=labels, index=labels)
    df.to_csv(out_csv, index=True)
    logger.info("Wrote pairwise RMSD matrix to %s", out_csv)


def compute_contacts(
    mobile: mda.Universe,
    selection1: str,
    selection2: str,
    cutoff_angstrom: float,
    stride: int,
    max_frames: Optional[int],
    per_residue: bool,
    out_csv: Path,
    label: str,
    selection1_name: Optional[str] = None,
    selection2_name: Optional[str] = None,
) -> None:
    """
    Compute contact counts/fractions between two selections.

    Args:
        mobile: Universe containing trajectory to analyze.
        selection1: MDAnalysis selection string for group 1.
        selection2: MDAnalysis selection string for group 2.
        cutoff_angstrom: Distance cutoff in angstroms.
        stride: Subsample frames every N steps.
        max_frames: Cap number of frames (after stride); None for all.
        per_residue: If True, count residue-residue contacts; else atom-atom.
        out_csv: Output CSV path.
        label: Label stored alongside results.
    """
    sel1 = _select_atoms_checked(mobile, selection1)
    sel2 = _select_atoms_checked(mobile, selection2)
    frame_indices = _frame_indices(mobile.trajectory.n_frames, stride, max_frames)
    counts: dict[tuple[int, int], int] = {}
    for i in frame_indices:
        mobile.trajectory[i]
        # Use capped_distance for efficiency; includes box for PBC-aware distances
        pairs, _ = distances.capped_distance(
            sel1.positions, sel2.positions, max_cutoff=cutoff_angstrom, box=mobile.trajectory.ts.dimensions
        )
        seen_keys = set()
        for idx1, idx2 in pairs:
            if per_residue:
                res1 = sel1.atoms[int(idx1)].resid
                res2 = sel2.atoms[int(idx2)].resid
                key = (res1, res2)
            else:
                a1 = sel1.atoms[int(idx1)].index
                a2 = sel2.atoms[int(idx2)].index
                key = (a1, a2)
            seen_keys.add(key)
        for key in seen_keys:
            counts[key] = counts.get(key, 0) + 1
    total = len(frame_indices)
    rows = []
    for key, val in counts.items():
        rows.append(
            {
                "id1": key[0],
                "id2": key[1],
                "count": val,
                "fraction": val / total if total else 0,
                "label": label,
                "selection1": selection1_name or selection1,
                "selection2": selection2_name or selection2,
            }
        )
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    logger.info("Wrote contacts to %s", out_csv)
