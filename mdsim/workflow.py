'''
    Workflow orchestration for molecular dynamics simulations.
'''

from dataclasses import is_dataclass, fields
from pathlib import Path
from typing import Any, Dict, Optional

import json
import shutil
import yaml

from .builder import InputBuilder
from .config import RunConfig
from .engine import SimulationEngine, SimulationResult
from .logging_setup import get_logger

logger = get_logger(__name__)


class MDWorkflow:
    """
    Orchestrates building and running a system for a given RunConfig.
    """

    def __init__(self, run_cfg: RunConfig, config_path: Optional[Path] = None) -> None:
        self.run_cfg = run_cfg
        self.config_path = config_path

        # Output layout
        self.output_dir: Path = run_cfg.run_output_dir
        self.prep_dir = self.output_dir / "prep"
        self.sim_dir = self.output_dir / "sim"
        self.tmp_dir = self.output_dir / "tmp"
        self.cache_dir = self.output_dir / "cache"

        for d in (self.output_dir, self.prep_dir, self.sim_dir, self.tmp_dir, self.cache_dir):
            d.mkdir(parents=True, exist_ok=True)

        # One engine reused for this run
        self.engine = SimulationEngine(
            run_cfg.simulation,
            run_cfg.output,
            mmgbsa_enabled=bool(getattr(run_cfg, "mmgbsa", None) and getattr(run_cfg.mmgbsa, "enabled", False)),
        )

        logger.info(
            "Initialized MDWorkflow for run '%s' (output_dir=%s)",
            run_cfg.run_name,
            self.output_dir,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> SimulationResult:
        """
        Build and simulate the system defined in this RunConfig.

        Returns:
            SimulationResult: Results of the MD simulation.
        """
        spec = self.run_cfg.system
        logger.info(
            "Running system '%s' (solvate=%s) into %s",
            spec.name,
            self.run_cfg.solvation.enabled,
            self.output_dir,
        )

        # Build inputs
        builder = InputBuilder(
            spec,
            run_output_dir=self.output_dir,
            prep_cfg=self.run_cfg.prep,
            solvation_cfg=self.run_cfg.solvation,
            engine_cfg=self.run_cfg.simulation.engine,
            output_cfg=self.run_cfg.output,
        )
        built = builder.build()

        # Run MD
        result = self.engine.run(
            built,
            spec,
            sim_dir=self.sim_dir,
            tmp_dir=self.tmp_dir,
            cache_dir=self.cache_dir,
        )

        self._write_config_copy()
        self._write_manifest(built, result)

        # Clean up temp/cache unless configured to keep them
        self._cleanup_intermediate()

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cleanup_intermediate(self) -> None:
        """
        Remove tmp/ and cache/ under the run output dir unless config says keep.
        """
        keep_temp = bool(self.run_cfg.output.keep_tmp)
        keep_cache = bool(self.run_cfg.output.keep_cache)

        if not keep_temp:
            _remove_dir_if_exists(self.tmp_dir, label="temp")
        else:
            logger.info("Keeping temp directory for '%s': %s", self.run_cfg.system.name, self.tmp_dir)

        if not keep_cache:
            _remove_dir_if_exists(self.cache_dir, label="cache")
        else:
            logger.info("Keeping cache directory for '%s': %s", self.run_cfg.system.name, self.cache_dir)

    def _write_config_copy(self) -> None:
        """
        Save a resolved config copy into the run directory.
        """
        target = self.output_dir / self.run_cfg.output.config_copy
        if self.config_path and Path(self.config_path).exists():
            shutil.copy2(self.config_path, target)
            logger.info("Copied original config to %s", target)
            return

        # Fallback: dump dataclass-derived config
        with target.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(_serialize(self.run_cfg), fh, sort_keys=False)
        logger.info("Wrote resolved config to %s", target)

    def _write_manifest(self, built, result: SimulationResult) -> None:
        manifest_path = self.output_dir / self.run_cfg.output.manifest
        manifest: Dict[str, Any] = {
            "run_name": self.run_cfg.run_name,
            "output_dir": str(self.output_dir),
            "prep": {
                "complex_pdb": str(built.complex_pdb_path),
                "complex_solvated_pdb": str(built.solvated_pdb_path) if built.solvated_pdb_path else None,
            },
            "simulation": {
                "equil_state": str(result.equil_state_data) if result.equil_state_data else None,
                "equil_dcd": str(result.equil_trajectory_dcd) if result.equil_trajectory_dcd else None,
                "equil_mdcrd": str(result.equil_trajectory_mdcrd) if result.equil_trajectory_mdcrd else None,
                "state": str(result.state_data),
                "dcd": str(result.trajectory_dcd),
                "mdcrd": str(result.trajectory_mdcrd) if result.trajectory_mdcrd else None,
                "checkpoint": str(result.checkpoint) if result.checkpoint else None,
                "final_state_xml": str(result.final_state_xml),
                "final_pdb": str(result.final_pdb),
                "initial_state_xml": str(result.initial_state_xml),
                "minimized_pdb": str(self.sim_dir / self.run_cfg.output.minimized_pdb),
            },
            "config_copy": str(self.output_dir / self.run_cfg.output.config_copy),
        }
        with manifest_path.open("w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)
        logger.info("Wrote manifest to %s", manifest_path)


def _serialize(obj: Any) -> Any:
    if is_dataclass(obj):
        return {f.name: _serialize(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(v) for v in obj]
    return obj


def _remove_dir_if_exists(path: Path, label: str) -> None:
    if path.exists():
        logger.info("Removing %s directory: %s", label, path)
        shutil.rmtree(path, ignore_errors=True)
    else:
        logger.debug("No %s directory found at %s; nothing to remove.", label, path)
