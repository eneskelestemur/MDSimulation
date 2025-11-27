"""
Visualization routines for mdsim analyses.

Current scope:
- State data plots (equilibration/production): potential energy, total energy, temperature, density vs time.
- RMSD, RMSF, pairwise RMSD, contacts (using analysis CSVs).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

from .logging_setup import get_logger

logger = get_logger(__name__)


def _load_state_log(path: Path) -> pd.DataFrame:
    column_names: Optional[list[str]] = None
    with path.open() as handle:
        first_line = handle.readline()
        if first_line.lstrip().startswith("#"):
            header_text = first_line.lstrip()[1:].strip()
            try:
                column_names = next(csv.reader([header_text]))
            except Exception:  # pragma: no cover - defensive logging
                logger.warning("Failed to parse header for %s; using pandas defaults", path)

    read_kwargs = {"sep": ",", "comment": "#"}
    if column_names:
        # Header line is commented out; use the parsed names and treat all rows as data.
        read_kwargs.update(names=column_names, header=None)
    df = pd.read_csv(path, **read_kwargs)

    rename_map = {
        "Step": "step",
        "Time (ps)": "time_ps",
        "Time(ps)": "time_ps",
        "Potential Energy (kJ/mole)": "PotentialEnergy(kJ/mole)",
        "Total Energy (kJ/mole)": "TotalEnergy(kJ/mole)",
        "Temperature (K)": "Temperature(K)",
        "Density (g/mL)": "Density(g/mL)",
    }
    for src, dst in rename_map.items():
        if src in df.columns and dst not in df.columns:
            df[dst] = df[src]

    if "time_ps" in df.columns:
        df["time_ns"] = df["time_ps"] / 1000.0
    elif "Time(ps)" in df.columns:
        df["time_ns"] = df["Time(ps)"] / 1000.0
    else:
        df["time_ns"] = df.index
    return df


def plot_state_data(log_path: Path, out_path: Path, title: str, dpi: int = 300) -> None:
    """
    Plot state data (potential, total energy, temperature, density) vs time.

    Args:
        log_path: Path to StateDataReporter log (whitespace separated).
        out_path: Output image path (e.g., visuals/equil_state.png).
        title: Figure title.
    """
    df = _load_state_log(log_path)
    sns.set_theme(style="whitegrid", context="notebook")
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=False)

    axes = axes.flatten()
    axes[0].plot(df["time_ns"], df["PotentialEnergy(kJ/mole)"])
    axes[0].set_ylabel("Potential Energy (kJ/mol)")
    axes[0].set_xlabel("Time (ns)")
    axes[0].set_title("Potential Energy")

    axes[1].plot(df["time_ns"], df["TotalEnergy(kJ/mole)"])
    axes[1].set_ylabel("Total Energy (kJ/mol)")
    axes[1].set_xlabel("Time (ns)")
    axes[1].set_title("Total Energy")

    axes[2].plot(df["time_ns"], df["Temperature(K)"])
    axes[2].set_ylabel("Temperature (K)")
    axes[2].set_xlabel("Time (ns)")
    axes[2].set_title("Temperature")

    axes[3].plot(df["time_ns"], df["Density(g/mL)"])
    axes[3].set_ylabel("Density (g/mL)")
    axes[3].set_xlabel("Time (ns)")
    axes[3].set_title("Density")

    fig.suptitle(title)
    fig.tight_layout(pad=1.2)
    fig.subplots_adjust(top=0.90, wspace=0.25, hspace=0.30)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    logger.info("Wrote state data plot to %s", out_path)


def plot_rmsd(csv_path: Path, out_path: Path, title: str, dpi: int = 300) -> None:
    df_raw = pd.read_csv(csv_path)
    if "time_ns" not in df_raw.columns:
        logger.warning("RMSD CSV missing time_ns; skipping plot for %s", csv_path)
        return

    # Accept both wide (one column per selection) and long (label+rmsd_angstrom)
    value_cols = [c for c in df_raw.columns if c not in {"frame", "time_ns", "label"}]
    if "rmsd_angstrom" in value_cols:
        long_df = df_raw
    else:
        long_df = df_raw.melt(id_vars=["frame", "time_ns"], value_vars=value_cols, var_name="label", value_name="rmsd_angstrom")

    sns.set_theme(style="whitegrid", context="notebook")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.lineplot(data=long_df, x="time_ns", y="rmsd_angstrom", hue="label", ax=ax, linewidth=1.3)
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("RMSD (Å)")
    ax.set_title(title)
    ax.legend(title="Selection", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    logger.info("Wrote RMSD plot to %s", out_path)


def plot_rmsf(
    csv_path: Path,
    out_path: Path,
    title: str,
    aggregate_by_residue: bool = False,
    residue_col: str = "residue_index",
    aggregate_map: Optional[dict[str, bool]] = None,
    dpi: int = 300,
) -> None:
    df = pd.read_csv(csv_path)
    sns.set_theme(style="whitegrid", context="notebook")
    fig, ax = plt.subplots(figsize=(7, 4.5))

    aggregate_map = aggregate_map or {}
    frames = []
    for label, df_label in df.groupby("label"):
        use_residue = aggregate_map.get(label, aggregate_by_residue)
        plot_df = df_label.copy()
        x_col = "atom_index"
        x_label = "Atom index"
        if use_residue:
            if residue_col not in plot_df.columns:
                logger.warning("RMSF CSV missing %s; falling back to atom-level RMSF for %s (%s)", residue_col, csv_path, label)
            else:
                group_cols = [residue_col]
                if "residue_id" in plot_df.columns:
                    group_cols.append("residue_id")
                if "residue_name" in plot_df.columns:
                    group_cols.append("residue_name")
                plot_df = plot_df.groupby(group_cols, sort=True)["rmsf_angstrom"].mean().reset_index()
                plot_df["label"] = label
                x_col = "residue_id" if "residue_id" in plot_df.columns else residue_col
                x_label = "Residue"
                if "residue_name" in plot_df.columns:
                    plot_df["residue_label"] = plot_df.apply(
                        lambda r: f"{int(r[x_col])} {r['residue_name']}", axis=1
                    )
                logger.info("Aggregated RMSF by %s for selection '%s'", residue_col, label)
        plot_df["label"] = label
        plot_df["_x_col"] = x_col
        plot_df["_x_label"] = x_label
        frames.append(plot_df)

    if not frames:
        logger.warning("No RMSF data to plot for %s", csv_path)
        return

    plot_df = pd.concat(frames, ignore_index=True)
    # If different selections used different x_col labels, pick residue_id if any used residue aggregation.
    if any(plot_df["_x_col"] != "atom_index"):
        x_col = "residue_id" if "residue_id" in plot_df.columns else residue_col
        x_label = "Residue"
    else:
        x_col = "atom_index"
        x_label = "Atom index"

    # Use markers for selections with <20 points; otherwise lines only.
    labels = sorted(plot_df["label"].unique())
    palette = sns.color_palette(n_colors=len(labels))
    for color, label in zip(palette, labels):
        group = plot_df[plot_df["label"] == label]
        use_marker = len(group) < 50
        sns.lineplot(
            data=group,
            x=x_col,
            y="rmsf_angstrom",
            label=label,
            ax=ax,
            linewidth=1.3 if not use_marker else 1.2,
            marker="o" if use_marker else None,
            markersize=5 if use_marker else 0,
            color=color,
        )
    ax.legend(title="Selection", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    ax.set_xlabel(x_label)
    ax.set_ylabel("RMSF (Å)")
    ax.set_title(title)

    if x_col != "atom_index":
        unique_x = np.array(sorted(plot_df[x_col].unique()))
        max_ticks = 30
        if len(unique_x) <= max_ticks:
            ax.set_xticks(unique_x)
            if "residue_label" in plot_df.columns:
                labels = [plot_df.loc[plot_df[x_col] == x_val, "residue_label"].iloc[0] for x_val in unique_x]
                ax.set_xticklabels(labels, rotation=90)
        ax.tick_params(axis="x", labelsize=8)

    ax.set_xlabel("Residue/Atom")
    ax.legend(title="Selection", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    logger.info("Wrote RMSF plot to %s", out_path)


def plot_pairwise_rmsd(csv_path: Path, out_path: Path, title: str, max_tick_labels: int = 30, dpi: int = 300) -> None:
    df = pd.read_csv(csv_path)
    if "rmsd_angstrom" in df.columns:
        # time-series case
        plot_rmsd(csv_path, out_path, title)
        return
    # matrix case
    mat = pd.read_csv(csv_path, index_col=0)
    data = mat.to_numpy(dtype=float)
    labels = mat.columns.tolist()
    sns.set_theme(style="white", context="notebook")
    fig, ax = plt.subplots(figsize=(8, 6))

    # Hide dense tick labels and re-add a subset to avoid overlap
    tick_indices = np.linspace(0, len(labels) - 1, num=min(max_tick_labels, len(labels)), dtype=int)
    tick_labels = [labels[i] for i in tick_indices]

    sns.heatmap(
        data,
        xticklabels=False,
        yticklabels=False,
        cmap="mako",
        ax=ax,
        cbar_kws={"label": "RMSD (Å)"},
    )
    ax.set_xlabel("Frame")
    ax.set_ylabel("Frame")
    ax.set_title(title)
    ax.set_xticks(tick_indices + 0.5)
    ax.set_yticks(tick_indices + 0.5)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(tick_labels, rotation=0, fontsize=8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote pairwise RMSD matrix plot to %s", out_path)


def plot_contacts(csv_path: Path, out_path: Path, title: str, top_n: int = 20, value: str = "fraction", dpi: int = 300) -> None:
    df = pd.read_csv(csv_path)
    if value not in df.columns:
        logger.warning("Contacts CSV missing column %s; skipping plot for %s", value, csv_path)
        return
    sns.set_theme(style="whitegrid", context="notebook")
    unique1 = df["id1"].nunique()
    unique2 = df["id2"].nunique()

    if unique1 > 1 and unique2 > 1:
        pivot = df.pivot_table(index="id1", columns="id2", values=value, fill_value=0)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(pivot, cmap="mako", ax=ax, cbar_kws={"label": value.capitalize()})
        ax.set_xlabel("id2")
        ax.set_ylabel("id1")
        ax.set_title(title)
    else:
        df_sorted = df.sort_values(value, ascending=False).head(top_n)
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
        ax.legend().set_title("Selection")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote contacts plot to %s", out_path)
