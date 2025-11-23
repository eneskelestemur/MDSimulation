from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..config import RunConfig, load_run_config
from ..logging_setup import get_logger
from .mmgbsa import run_mmgbsa
from .md import (
    load_universe,
    load_reference,
    compute_rmsd,
    compute_rmsf,
    compute_pairwise_rmsd,
    compute_contacts,
)

logger = get_logger(__name__)


class AnalysisWorkflow:
    """
    Run post-simulation analyses for a completed run directory.
    """

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir).expanduser().resolve()
        cfg_path = self.run_dir / "config_resolved.yaml"
        if not cfg_path.exists():
            raise FileNotFoundError(f"config_resolved.yaml not found in {self.run_dir}")
        self.run_cfg: RunConfig = load_run_config(cfg_path)
        self.analysis_dir = self.run_dir / "analysis"
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Initialized AnalysisWorkflow for %s", self.run_dir)

    def run(self) -> None:
        """
        Execute configured analyses. Supports structural/timeseries analyses and MMGBSA.
        """
        self.analysis_dir.mkdir(parents=True, exist_ok=True)

        self._run_structural_analyses()

        if self.run_cfg.mmgbsa and self.run_cfg.mmgbsa.enabled:
            logger.info(
                "MMGBSA enabled; using config_resolved.yaml. Update analysis/mmgbsa section there to change parameters."
            )
            run_mmgbsa(self.run_dir, self.run_cfg)
        else:
            logger.info("MMGBSA is disabled; nothing to do.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_topology_and_trajectory(self) -> tuple[Path, Path]:
        sim_dir = self.run_dir / "sim"
        traj_candidates = [
            sim_dir / "sim_trajectory.dcd",
            sim_dir / "sim_trajectory.dcd.gz",
        ]
        traj = next((p for p in traj_candidates if p.exists()), None)
        if traj is None:
            raise FileNotFoundError(f"No trajectory (DCD) found in {sim_dir}")
        topology = sim_dir / "simulated_complex.pdb"
        if not topology.exists():
            raise FileNotFoundError(f"Topology file not found: {topology}")
        return topology, traj

    def _get_reference(self, ref_choice: str, ref_path: Optional[Path]) -> Path:
        sim_dir = self.run_dir / "sim"
        if ref_choice == "minimized":
            path = sim_dir / "minimized_complex.pdb"
        elif ref_choice == "final":
            path = sim_dir / "simulated_complex.pdb"
        elif ref_choice == "external":
            if not ref_path:
                raise ValueError("reference_path must be set when reference='external'")
            path = ref_path
        else:
            raise ValueError(f"Unknown reference choice '{ref_choice}'")
        if not path.exists():
            raise FileNotFoundError(f"Reference structure not found: {path}")
        return path

    def _run_structural_analyses(self) -> None:
        cfg = self.run_cfg.analysis
        if cfg is None:
            logger.info("No analysis section found; skipping structural analyses.")
            return

        topology, trajectory = self._get_topology_and_trajectory()
        mobile = load_universe(topology, trajectory)

        # RMSD
        if cfg.rmsd.enabled:
            ref_path = self._get_reference(cfg.rmsd.reference, cfg.rmsd.reference_path)
            reference = load_reference(ref_path)
            out_csv = self.analysis_dir / f"{cfg.rmsd.name}.csv"
            compute_rmsd(
                mobile=mobile,
                reference=reference,
                align_sel=cfg.rmsd.align_selection,
                target_sel=cfg.rmsd.target_selection,
                stride=cfg.rmsd.stride,
                max_frames=cfg.rmsd.max_frames,
                out_csv=out_csv,
                label=cfg.rmsd.name,
            )

        # RMSF
        if cfg.rmsf.enabled:
            out_csv = self.analysis_dir / f"{cfg.rmsf.name}.csv"
            compute_rmsf(
                mobile=mobile,
                align_sel=cfg.rmsf.selection,
                target_sel=cfg.rmsf.selection,
                stride=cfg.rmsf.stride,
                max_frames=cfg.rmsf.max_frames,
                out_csv=out_csv,
                label=cfg.rmsf.name,
            )

        # Pairwise RMSD
        if cfg.pairwise_rmsd.enabled:
            out_csv = self.analysis_dir / f"{cfg.pairwise_rmsd.name}.csv"
            compute_pairwise_rmsd(
                mobile=mobile,
                reference=reference,
                selection1=cfg.pairwise_rmsd.selection1,
                selection2=cfg.pairwise_rmsd.selection2,
                align_sel=cfg.pairwise_rmsd.align_selection,
                stride=cfg.pairwise_rmsd.stride,
                max_frames=cfg.pairwise_rmsd.max_frames,
                out_csv=out_csv,
                label=cfg.pairwise_rmsd.name,
            )

        # Contacts
        if cfg.contacts.enabled:
            out_csv = self.analysis_dir / f"{cfg.contacts.name}.csv"
            compute_contacts(
                mobile=mobile,
                selection1=cfg.contacts.selection1,
                selection2=cfg.contacts.selection2,
                cutoff_angstrom=cfg.contacts.cutoff_angstrom,
                stride=cfg.contacts.stride,
                max_frames=cfg.contacts.max_frames,
                per_residue=cfg.contacts.per_residue,
                out_csv=out_csv,
                label=cfg.contacts.name,
            )
