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

    # -------------------- analyze (stub for later) --------------------
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Analyze simulation outputs (structure/timeseries/MMGBSA).",
    )
    analyze_parser.add_argument(
        "config",
        type=str,
        help="Path to YAML configuration file used for the run.",
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


def _analyze_command(config_path: Path) -> int:
    """
    Placeholder for `mdsim analyze <config.yaml>`.

    For now, we just tell the user it's not implemented yet.
    Later this will call into mdsim.analysis.* using the same RunConfig.
    """
    config_path = config_path.expanduser().resolve()
    print(
        "[mdsim] 'analyze' is not implemented yet. "
        f"Config was: {config_path}"
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
        return _analyze_command(Path(args.config))

    # Should not get here because subparsers are required
    raise RuntimeError(f"Unknown command: {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
