import argparse
import logging
from pathlib import Path
from typing import List, Optional

from ..logging_setup import setup_logging


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="mdsim",
        description="Command-line interface for running OpenMM simulations with mdsim.",
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO).",
    )

    subparsers = parser.add_subparsers(
        title="subcommands",
        dest="command",
        required=True,
    )

    # -------------------- run --------------------
    run_parser = subparsers.add_parser(
        "run",
        help="Run a single simulation from a YAML configuration file.",
    )
    run_parser.add_argument(
        "config",
        type=str,
        help="Path to YAML configuration file.",
    )

    # -------------------- analyze --------------------
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Analyze simulation outputs (MMGBSA and future analyses).",
    )
    analyze_parser.add_argument(
        "run_dir",
        type=str,
        help="Path to completed run directory (contains config_resolved.yaml).",
    )

    # -------------------- visualize --------------------
    viz_parser = subparsers.add_parser(
        "visualize",
        help="Create plots from analysis outputs (state data, future visuals).",
    )
    viz_parser.add_argument(
        "run_dir",
        type=str,
        help="Path to completed run directory (contains analysis/ and sim/).",
    )

    return parser.parse_args(argv)


def _run_command(config_path: Path) -> int:
    """
    Implementation of `mdsim run <config.yaml>`.
    """
    # Import here so logging is configured first
    from ..config import load_run_config
    from ..workflow import MDWorkflow

    config_path = config_path.expanduser().resolve()

    run_cfg = load_run_config(config_path)
    workflow = MDWorkflow(run_cfg, config_path=config_path)

    result = workflow.run()

    # Short user-facing summary
    print(f"[mdsim] Run '{run_cfg.run_name}' finished.")
    print(f"  Output directory : {run_cfg.run_output_dir}")
    print(f"  Final PDB        : {result.final_pdb}")
    print(f"  Trajectory (DCD) : {result.trajectory_dcd}")
    print(f"  State data (LOG) : {result.state_data}")

    return 0


def _analyze_command(run_dir: Path) -> int:
    """
    Run post-simulation analyses for an existing run directory.
    """
    from ..analysis.workflow import AnalysisWorkflow

    run_dir = run_dir.expanduser().resolve()
    workflow = AnalysisWorkflow(run_dir)
    workflow.run()
    return 0


def _visualize_command(run_dir: Path) -> int:
    """
    Generate plots for a completed run (state data).
    """
    from ..visualize import plot_state_data, plot_rmsd, plot_rmsf, plot_pairwise_rmsd, plot_contacts
    from ..config import load_run_config, VisualizationConfig, RunConfig

    run_dir = run_dir.expanduser().resolve()
    visuals_dir = run_dir / "visuals"
    visuals_dir.mkdir(parents=True, exist_ok=True)

    viz_cfg: VisualizationConfig | None = None
    run_cfg: RunConfig | None = None
    cfg_path = run_dir / "config_resolved.yaml"
    if cfg_path.exists():
        try:
            run_cfg = load_run_config(cfg_path)
            viz_cfg = run_cfg.visualization
        except Exception as exc:  # pragma: no cover - defensive, CLI-facing
            print(f"[mdsim] Warning: failed to load visualization config from {cfg_path}: {exc}")
    else:
        print(f"[mdsim] Warning: config_resolved.yaml not found in {run_dir}; using visualization defaults.")

    rmsd_name = run_cfg.analysis.rmsd.name if run_cfg else "rmsd"
    rmsf_name = run_cfg.analysis.rmsf.name if run_cfg else "rmsf"
    pairwise_name = run_cfg.analysis.pairwise_rmsd.name if run_cfg else "pairwise_rmsd"
    contacts_name = run_cfg.analysis.contacts.name if run_cfg else "contacts"

    equil_log = run_dir / "sim" / "equil_state.log"
    prod_log = run_dir / "sim" / "sim_state.log"
    if equil_log.exists():
        plot_state_data(equil_log, visuals_dir / "equil_state.png", title="Equilibration", dpi=viz_cfg.figure_dpi if viz_cfg else 300)
    else:
        print(f"[mdsim] Equilibration log not found at {equil_log}; skipping.")
    if prod_log.exists():
        plot_state_data(prod_log, visuals_dir / "sim_state.png", title="Production", dpi=viz_cfg.figure_dpi if viz_cfg else 300)
    else:
        print(f"[mdsim] Production log not found at {prod_log}; skipping.")

    analysis_dir = run_dir / "analysis"
    # RMSD
    rmsd_csv = analysis_dir / f"{rmsd_name}.csv"
    if rmsd_csv.exists():
        plot_rmsd(
            rmsd_csv,
            visuals_dir / "rmsd.png",
            title="RMSD",
            dpi=viz_cfg.figure_dpi if viz_cfg else 300,
        )
    # RMSF
    rmsf_csv = analysis_dir / f"{rmsf_name}.csv"
    if rmsf_csv.exists():
        default_rmsf_agg = viz_cfg.rmsf.aggregate_by_residue if viz_cfg else False
        agg_map = None
        if viz_cfg:
            agg_map = {sel.name: sel.aggregate_by_residue for sel in viz_cfg.rmsf.selections}
        plot_rmsf(
            rmsf_csv,
            visuals_dir / "rmsf.png",
            title="RMSF",
            aggregate_by_residue=default_rmsf_agg,
            aggregate_map=agg_map,
            dpi=viz_cfg.figure_dpi if viz_cfg else 300,
        )
    # Pairwise RMSD
    pairwise_csv = analysis_dir / f"{pairwise_name}.csv"
    if pairwise_csv.exists():
        max_ticks = viz_cfg.pairwise_rmsd.max_tick_labels if viz_cfg else 30
        plot_pairwise_rmsd(
            pairwise_csv,
            visuals_dir / "pairwise_rmsd.png",
            title="Pairwise RMSD",
            max_tick_labels=max_ticks,
            dpi=viz_cfg.figure_dpi if viz_cfg else 300,
        )
    # Contacts
    contacts_csv = analysis_dir / f"{contacts_name}.csv"
    if contacts_csv.exists():
        plot_contacts(
            contacts_csv,
            visuals_dir / "contacts.png",
            title="Contacts",
            top_n=viz_cfg.contacts.top_n if viz_cfg else 20,
            value="fraction",
            dpi=viz_cfg.figure_dpi if viz_cfg else 300,
        )

    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)

    # Set up logging *before* importing modules that call get_logger
    level = getattr(logging, args.log_level.upper(), logging.INFO)
    setup_logging(level=level)

    if args.command == "run":
        return _run_command(Path(args.config))
    elif args.command == "analyze":
        return _analyze_command(Path(args.run_dir))
    elif args.command == "visualize":
        return _visualize_command(Path(args.run_dir))

    # Should not get here because subparsers are required
    raise RuntimeError(f"Unknown command: {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
