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
from MDAnalysis.analysis import align

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


def _align(mobile: mda.Universe, reference: mda.Universe, align_sel: Optional[str], *, self_align: bool = False) -> None:
    """Align mobile to reference (or itself) if an alignment selection is provided."""
    if not align_sel:
        return
    if self_align:
        align.alignto(mobile, mobile, select=align_sel, weights="mass")
    else:
        align.alignto(mobile, reference, select=align_sel, weights="mass")


# --------------------------------------------------------------------------- #
# Analyses
# --------------------------------------------------------------------------- #


def compute_rmsd(
    mobile: mda.Universe,
    reference: mda.Universe,
    align_sel: Optional[str],
    target_sel: str,
    stride: int,
    max_frames: Optional[int],
    out_csv: Path,
    label: str,
) -> None:
    """
    Compute RMSD against a reference for a target selection and write CSV.

    Args:
        mobile: Universe containing trajectory to analyze.
        reference: Reference structure Universe.
        align_sel: Selection string for alignment (optional).
        target_sel: Selection string to compute RMSD on.
        stride: Subsample frames every N steps.
        max_frames: Cap number of frames (after stride); None for all.
        out_csv: Output CSV path.
        label: Label stored alongside results.
    """
    mob_atoms = _select_atoms_checked(mobile, target_sel)
    ref_atoms = _select_atoms_checked(reference, target_sel)
    if len(mob_atoms) != len(ref_atoms):
        raise ValueError("RMSD target selections do not match in atom count.")

    frame_indices = _frame_indices(mobile.trajectory.n_frames, stride, max_frames)
    data = []
    for i in frame_indices:
        mobile.trajectory[i]
        _align(mobile, reference, align_sel, self_align=False)
        diff = mob_atoms.positions - ref_atoms.positions
        rmsd_val = np.sqrt((diff * diff).sum() / len(mob_atoms))
        data.append(
            {
                "frame": i,
                "time_ns": mobile.trajectory.time / 1000.0,
                "rmsd_angstrom": rmsd_val,
                "label": label,
            }
        )

    pd.DataFrame(data).to_csv(out_csv, index=False)
    logger.info("Wrote RMSD to %s", out_csv)


def compute_rmsf(
    mobile: mda.Universe,
    align_sel: Optional[str],
    target_sel: str,
    stride: int,
    max_frames: Optional[int],
    out_csv: Path,
    label: str,
) -> None:
    """
    Compute RMSF for a selection (optionally aligned each frame) and write CSV.

    Args:
        mobile: Universe containing trajectory to analyze.
        align_sel: Selection string for alignment (optional).
        target_sel: Selection string to compute RMSF on.
        stride: Subsample frames every N steps.
        max_frames: Cap number of frames (after stride); None for all.
        out_csv: Output CSV path.
        label: Label stored alongside results.
    """
    atoms = _select_atoms_checked(mobile, target_sel)
    frame_indices = _frame_indices(mobile.trajectory.n_frames, stride, max_frames)
    coords = []
    for i in frame_indices:
        mobile.trajectory[i]
        _align(mobile, mobile, align_sel, self_align=True)
        coords.append(atoms.positions.copy())
    if not coords:
        raise ValueError("No frames collected for RMSF.")
    coords = np.array(coords)
    mean = coords.mean(axis=0)
    diffs = coords - mean
    rmsf = np.sqrt((diffs * diffs).sum(axis=2).mean(axis=0))
    df = pd.DataFrame({"atom_index": np.arange(len(atoms)), "rmsf_angstrom": rmsf, "label": label})
    df.to_csv(out_csv, index=False)
    logger.info("Wrote RMSF to %s", out_csv)


def compute_pairwise_rmsd(
    mobile: mda.Universe,
    reference: mda.Universe,
    selection1: str,
    selection2: Optional[str],
    align_sel: Optional[str],
    stride: int,
    max_frames: Optional[int],
    out_csv: Path,
    label: str,
) -> None:
    """
    Compute pairwise RMSD and write CSV.

    If selection2 is provided, outputs a time-series RMSD between selection1 and selection2.
    If selection2 is None, outputs a symmetric RMSD matrix between frames for selection1.
    Time is reported in ns.

    Args:
        mobile: Universe containing trajectory to analyze.
        reference: Reference structure Universe (used for self alignment).
        selection1: First selection string.
        selection2: Second selection string (optional).
        align_sel: Selection string for alignment (optional).
        stride: Subsample frames every N steps.
        max_frames: Cap number of frames (after stride); None for all.
        out_csv: Output CSV path.
        label: Label stored alongside results.
    """
    if selection2:
        sel1 = _select_atoms_checked(mobile, selection1)
        sel2 = _select_atoms_checked(mobile, selection2)
        if len(sel1) != len(sel2):
            raise ValueError("Pairwise RMSD selections must have same atom count.")
        frame_indices = _frame_indices(mobile.trajectory.n_frames, stride, max_frames)
        data = []
        for i in frame_indices:
            mobile.trajectory[i]
            _align(mobile, mobile, align_sel, self_align=True)
            diff = sel1.positions - sel2.positions
            rmsd_val = np.sqrt((diff * diff).sum() / len(sel1))
            data.append(
                {
                    "frame": i,
                    "time_ns": mobile.trajectory.time / 1000.0,
                    "rmsd_angstrom": rmsd_val,
                    "label": label,
                }
            )
        pd.DataFrame(data).to_csv(out_csv, index=False)
        logger.info("Wrote pairwise RMSD (selection1 vs selection2) to %s", out_csv)
        return

    # Self RMSD matrix across frames
    sel = _select_atoms_checked(mobile, selection1)
    ref_sel = _select_atoms_checked(reference, selection1)
    if len(sel) != len(ref_sel):
        raise ValueError("Pairwise RMSD self requires same atom count in reference.")
    frame_indices = _frame_indices(mobile.trajectory.n_frames, stride, max_frames)
    coords = []
    times_ns = []
    for i in frame_indices:
        mobile.trajectory[i]
        _align(mobile, reference, align_sel, self_align=False)
        coords.append(sel.positions.copy())
        times_ns.append(mobile.trajectory.time / 1000.0)
    if not coords:
        raise ValueError("No frames collected for pairwise RMSD.")
    coords = np.array(coords)
    n = len(coords)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            diff = coords[i] - coords[j]
            rmsd_val = np.sqrt((diff * diff).sum() / len(sel))
            matrix[i, j] = rmsd_val
            matrix[j, i] = rmsd_val
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
    counts = {}
    for i in frame_indices:
        mobile.trajectory[i]
        dists = mda.lib.distances.distance_array(sel1.positions, sel2.positions)
        contacts = np.where(dists <= cutoff_angstrom)
        if per_residue:
            for idx1, idx2 in zip(*contacts):
                res1 = sel1.atoms[idx1].resid
                res2 = sel2.atoms[idx2].resid
                counts[(res1, res2)] = counts.get((res1, res2), 0) + 1
        else:
            for idx1, idx2 in zip(*contacts):
                a1 = sel1.atoms[idx1].index
                a2 = sel2.atoms[idx2].index
                counts[(a1, a2)] = counts.get((a1, a2), 0) + 1
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
            }
        )
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    logger.info("Wrote contacts to %s", out_csv)
