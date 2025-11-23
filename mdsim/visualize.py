"""
Visualization routines for mdsim analyses.

Current scope:
- State data plots (equilibration/production): potential energy, total energy, temperature, density vs time.
- RMSD, RMSF, pairwise RMSD, contacts (using analysis CSVs).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

from .logging_setup import get_logger

logger = get_logger(__name__)


def _load_state_log(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, delim_whitespace=True, comment="#")
    if "Time(ps)" in df.columns:
        df["time_ns"] = df["Time(ps)"] / 1000.0
    else:
        df["time_ns"] = df.index
    return df


def plot_state_data(log_path: Path, out_path: Path, title: str) -> None:
    """
    Plot state data (potential, total energy, temperature, density) vs time.

    Args:
        log_path: Path to StateDataReporter log (whitespace separated).
        out_path: Output image path (e.g., visuals/equil_state.png).
        title: Figure title.
    """
    df = _load_state_log(log_path)
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), constrained_layout=True)

    axes = axes.flatten()
    axes[0].plot(df["time_ns"], df["PotentialEnergy(kJ/mole)"])
    axes[0].set_ylabel("Potential Energy (kJ/mol)")
    axes[0].set_xlabel("Time (ns)")

    axes[1].plot(df["time_ns"], df["TotalEnergy(kJ/mole)"])
    axes[1].set_ylabel("Total Energy (kJ/mol)")
    axes[1].set_xlabel("Time (ns)")

    axes[2].plot(df["time_ns"], df["Temperature(K)"])
    axes[2].set_ylabel("Temperature (K)")
    axes[2].set_xlabel("Time (ns)")

    axes[3].plot(df["time_ns"], df["Density(g/mL)"])
    axes[3].set_ylabel("Density (g/mL)")
    axes[3].set_xlabel("Time (ns)")

    fig.suptitle(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    logger.info("Wrote state data plot to %s", out_path)


def plot_rmsd(csv_path: Path, out_path: Path, title: str) -> None:
    df = pd.read_csv(csv_path)
    if "time_ns" not in df.columns:
        logger.warning("RMSD CSV missing time_ns; skipping plot for %s", csv_path)
        return
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.lineplot(data=df, x="time_ns", y="rmsd_angstrom", hue="label", ax=ax)
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("RMSD (Å)")
    ax.set_title(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    logger.info("Wrote RMSD plot to %s", out_path)


def plot_rmsf(csv_path: Path, out_path: Path, title: str) -> None:
    df = pd.read_csv(csv_path)
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.lineplot(data=df, x="atom_index", y="rmsf_angstrom", hue="label", ax=ax)
    ax.set_xlabel("Atom index")
    ax.set_ylabel("RMSF (Å)")
    ax.set_title(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    logger.info("Wrote RMSF plot to %s", out_path)


def plot_pairwise_rmsd(csv_path: Path, out_path: Path, title: str) -> None:
    df = pd.read_csv(csv_path)
    if "rmsd_angstrom" in df.columns:
        # time-series case
        plot_rmsd(csv_path, out_path, title)
        return
    # matrix case
    mat = pd.read_csv(csv_path, index_col=0)
    data = mat.to_numpy(dtype=float)
    labels = mat.columns.tolist()
    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(data, xticklabels=labels, yticklabels=labels, cmap="mako", ax=ax, cbar_kws={"label": "RMSD (Å)"})
    ax.set_xlabel("Frame")
    ax.set_ylabel("Frame")
    ax.set_title(title)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote pairwise RMSD matrix plot to %s", out_path)


def plot_contacts(csv_path: Path, out_path: Path, title: str, top_n: int = 20, value: str = "fraction") -> None:
    df = pd.read_csv(csv_path)
    if value not in df.columns:
        logger.warning("Contacts CSV missing column %s; skipping plot for %s", value, csv_path)
        return
    df_sorted = df.sort_values(value, ascending=False).head(top_n)
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(
        data=df_sorted,
        x=value,
        y=df_sorted.apply(lambda r: f"{r['id1']}-{r['id2']}", axis=1),
        hue="label",
        ax=ax,
        dodge=False,
    )
    ax.set_xlabel(value.capitalize())
    ax.set_ylabel("Pair")
    ax.set_title(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote contacts plot to %s", out_path)
