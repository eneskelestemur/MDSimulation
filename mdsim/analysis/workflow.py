from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import MDAnalysis.transformations as mda_trans
import MDAnalysis as mda

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
        reference_cache: dict[Path, object] = {}
        trans_cfg = cfg.transformations
        if trans_cfg and trans_cfg.enabled:
            transformations = []
            if trans_cfg.unwrap_selection:
                try:
                    ag = mobile.select_atoms(trans_cfg.unwrap_selection)
                    transformations.append(mda_trans.unwrap(ag))
                except Exception as exc:
                    logger.warning("Failed to set unwrap transformation (%s); continuing without unwrap", exc)
            if trans_cfg.center_selection:
                try:
                    ag = mobile.select_atoms(trans_cfg.center_selection)
                    transformations.append(mda_trans.center_in_box(ag, wrap=True))
                except Exception as exc:
                    logger.warning("Failed to set center_in_box transformation (%s); continuing without centering", exc)
            wrap_sel = None
            if trans_cfg.unwrap_selection and str(trans_cfg.unwrap_selection).lower() != "all":
                wrap_sel = f"not ({trans_cfg.unwrap_selection})"
            if wrap_sel:
                try:
                    transformations.append(mda_trans.wrap(mobile.select_atoms(wrap_sel)))
                except Exception as exc:
                    logger.warning("Failed to set wrap transformation (%s); continuing without final wrap", exc)
            if transformations:
                mobile.trajectory.add_transformations(*transformations)
                logger.info("Applied %d trajectory transformations for analysis", len(transformations))

            # Write transformed trajectory once, then reload without transformations
            sim_dir = self.run_dir / "sim"
            sim_dir.mkdir(parents=True, exist_ok=True)
            fmt = trans_cfg.save_format.lower()
            filename = trans_cfg.save_filename or f"sim_trajectory_transformed.{fmt}"
            out_path = sim_dir / filename
            stride = max(1, trans_cfg.save_interval)
            try:
                with mda.Writer(str(out_path), n_atoms=mobile.atoms.n_atoms) as writer:
                    for ts in mobile.trajectory[::stride]:
                        writer.write(mobile.atoms)
                logger.info("Wrote transformed trajectory to %s (stride=%d)", out_path, stride)
                # Reload transformed trajectory without transformations for analyses
                mobile = load_universe(topology, out_path)
            except Exception as exc:
                logger.warning("Failed to write or reload transformed trajectory (%s); continuing with original trajectory and transforms", exc)

        # RMSD
        if cfg.rmsd.enabled:
            ref_path = self._get_reference(cfg.rmsd.reference, cfg.rmsd.reference_path)
            reference = reference_cache.get(ref_path)
            if reference is None:
                reference = load_reference(ref_path)
                reference_cache[ref_path] = reference

            out_csv = self.analysis_dir / f"{cfg.rmsd.name}.csv"
            rmsd_frames = []
            for sel in cfg.rmsd.selections:
                df = compute_rmsd(
                    mobile=mobile,
                    reference=reference,
                    align_sel=cfg.rmsd.align_selection,
                    target_sel=sel.target_selection,
                    stride=sel.stride,
                    max_frames=sel.max_frames,
                    out_csv=None,
                    label=sel.name,
                )
                rmsd_frames.append(df.rename(columns={"rmsd_angstrom": sel.name}))
                logger.info("Computed RMSD for selection '%s'", sel.name)

            if rmsd_frames:
                base = rmsd_frames[0][["frame", "time_ns"]].copy()
                for df_sel in rmsd_frames:
                    sel_col = [c for c in df_sel.columns if c not in {"frame", "time_ns", "label"}]
                    if len(sel_col) != 1:
                        raise ValueError("RMSD selection DataFrame missing expected single RMSD column.")
                    base = base.merge(df_sel[["frame", "time_ns", sel_col[0]]], on=["frame", "time_ns"], how="outer")
                base.sort_values("frame").to_csv(out_csv, index=False)
                logger.info("Wrote combined RMSD (all selections) to %s", out_csv)

        # RMSF
        if cfg.rmsf.enabled:
            out_csv = self.analysis_dir / f"{cfg.rmsf.name}.csv"
            rmsf_frames = []
            for sel in cfg.rmsf.selections:
                df = compute_rmsf(
                    mobile=mobile,
                    align_sel=cfg.rmsf.align_selection or sel.target_selection,
                    target_sel=sel.target_selection,
                    stride=sel.stride,
                    max_frames=sel.max_frames,
                    out_csv=None,
                    label=sel.name,
                )
                rmsf_frames.append(df)
                logger.info("Computed RMSF for selection '%s'", sel.name)
            if rmsf_frames:
                pd.concat(rmsf_frames, ignore_index=True).to_csv(out_csv, index=False)
                logger.info("Wrote combined RMSF (all selections) to %s", out_csv)

        # Pairwise RMSD
        if cfg.pairwise_rmsd.enabled:
            ref_path = self._get_reference(cfg.rmsd.reference, cfg.rmsd.reference_path)
            reference_for_pairwise = reference_cache.get(ref_path)
            if reference_for_pairwise is None:
                reference_for_pairwise = load_reference(ref_path)
                reference_cache[ref_path] = reference_for_pairwise

            out_csv = self.analysis_dir / f"{cfg.pairwise_rmsd.name}.csv"
            compute_pairwise_rmsd(
                mobile=mobile,
                reference=reference_for_pairwise,
                selection=cfg.pairwise_rmsd.selection,
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
