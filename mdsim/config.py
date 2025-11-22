'''
    Configuration dataclasses and YAML loader for MD simulation runs.
'''

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Mapping

import yaml

from .logging_setup import get_logger

logger = get_logger(__name__)


# --- Core config blocks ------------------------------------------------------


@dataclass
class PrepConfig:
    fix_missing_residues: bool = True
    fix_missing_atoms: bool = True
    add_hydrogens_pH: float = 7.0
    remove_heterogens: bool = True
    ligand_allow_undefined_stereo: bool = True


@dataclass
class SolvationConfig:
    enabled: bool = True
    water_model: str = "tip3p"
    padding_nm: float = 1.0
    ionic_strength_m: float = 0.15
    positive_ion: str = "Na+"
    negative_ion: str = "Cl-"
    neutralize: bool = True
    solvent_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProtocolPhase:
    enabled: bool = True
    steps: int = 0


@dataclass
class MinimizeConfig:
    enabled: bool = True
    max_iterations: int = 0  # 0 → run until convergence
    tolerance_kj_per_mol_nm: float = 10.0


@dataclass
class ProtocolConfig:
    minimize: MinimizeConfig = field(default_factory=MinimizeConfig)
    nvt: ProtocolPhase = field(default_factory=lambda: ProtocolPhase(enabled=True, steps=50_000))
    npt: ProtocolPhase = field(default_factory=lambda: ProtocolPhase(enabled=True, steps=100_000))
    production_steps: int = 1_000_000
    nvt_start_temp_k: float = 5.0
    nvt_ramp_chunk_steps: int = 5


@dataclass
class IntegratorConfig:
    type: str = "LangevinMiddle"  # LangevinMiddle, Langevin, Verlet
    seed: Optional[int] = None


@dataclass
class EngineConfig:
    constraints: Optional[str] = None
    rigid_water: bool = False
    remove_cm_motion: bool = False
    cutoff_nm: float = 1.0
    switch_distance_nm: Optional[float] = None
    ewald_error_tolerance: Optional[float] = None
    dispersion_correction: Optional[bool] = None
    hydrogen_mass: Optional[float] = None  # enable HMR if set
    barostat_interval: Optional[int] = None
    ligand_forcefield: str = "gaff-2.11"


@dataclass
class ReporterGroup:
    state_file: str
    dcd_file: str
    mdcrd_file: str
    mdcrd_enabled: bool = False
    interval: int = 1_000


@dataclass
class ReportingConfig:
    equilibration: ReporterGroup = field(
        default_factory=lambda: ReporterGroup(
            state_file="equil_state.log",
            dcd_file="equil_trajectory.dcd",
            mdcrd_file="equil.mdcrd",
            mdcrd_enabled=False,
            interval=1_000,
        )
    )
    production: ReporterGroup = field(
        default_factory=lambda: ReporterGroup(
            state_file="sim_state.log",
            dcd_file="sim_trajectory.dcd",
            mdcrd_file="sim.mdcrd",
            mdcrd_enabled=False,
            interval=1_000,
        )
    )
    checkpoint_interval: Optional[int] = None
    checkpoint_file: str = "sim_checkpoint.chk"


@dataclass
class SimulationConfig:
    platform: str = "CUDA"
    temperature: float = 300.0  # K
    friction: float = 1.0  # 1/ps
    timestep_fs: float = 2.0  # fs

    protocol: ProtocolConfig = field(default_factory=ProtocolConfig)
    reporting: ReportingConfig = field(default_factory=ReportingConfig)
    integrator: IntegratorConfig = field(default_factory=IntegratorConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)


@dataclass
class OutputConfig:
    keep_tmp: bool = False
    keep_cache: bool = False
    final_pdb: str = "simulated_complex.pdb"
    final_state_xml: str = "final_state.xml"
    initial_state_xml: str = "initial_state.xml"
    minimized_pdb: str = "minimized_complex.pdb"
    config_copy: str = "config_resolved.yaml"
    manifest: str = "manifest.json"


@dataclass
class SystemSpec:
    name: str
    protein_paths: List[Path] = field(default_factory=list)
    ligand_paths: List[Path] = field(default_factory=list)
    rna_paths: List[Path] = field(default_factory=list)

    forcefields: List[str] = field(default_factory=list)


# --- Analysis-related config (unchanged, optional) ---------------------------


@dataclass
class RMSDConfig:
    enabled: bool = True
    reference: str = "last_frame"  # "initial", "last_frame", "external"
    align_selection: str = "protein"  # free-form, to be interpreted later
    reference_path: Optional[Path] = None


@dataclass
class RMSFConfig:
    enabled: bool = True
    selection: str = "protein"  # which atoms/residues


@dataclass
class PairwiseRMSDConfig:
    enabled: bool = False
    selection: str = "protein"
    max_frames: Optional[int] = None  # subsample if large


@dataclass
class TimeseriesConfig:
    enabled: bool = True
    columns: List[str] = field(default_factory=list)  # if empty, choose sensible defaults


@dataclass
class AnalysisConfig:
    rmsd: RMSDConfig = field(default_factory=RMSDConfig)
    rmsf: RMSFConfig = field(default_factory=RMSFConfig)
    pairwise_rmsd: PairwiseRMSDConfig = field(default_factory=PairwiseRMSDConfig)
    timeseries: TimeseriesConfig = field(default_factory=TimeseriesConfig)


# --- MMGBSA placeholder -----------------------------------------------------


@dataclass
class MMGBSAConfig:
    enabled: bool = False


# --- Top-level run config ----------------------------------------------------


@dataclass
class RunConfig:
    """
    Top-level config object for a run.

    This corresponds 1:1 to the main YAML structure.
    """

    run_name: str
    output_root: Path

    system: SystemSpec
    prep: PrepConfig
    solvation: SolvationConfig
    simulation: SimulationConfig
    output: OutputConfig

    analysis: Optional[AnalysisConfig] = None
    mmgbsa: Optional[MMGBSAConfig] = None

    @property
    def run_output_dir(self) -> Path:
        """Full output directory for this run."""
        return self.output_root / self.run_name


# --- YAML loader helpers -----------------------------------------------------


def _coerce_path(value: Optional[str], base_dir: Path) -> Optional[Path]:
    if value is None:
        return None
    p = Path(value)
    return p if p.is_absolute() else (base_dir / p)


def _load_prep_config(data: Mapping[str, Any] | None) -> PrepConfig:
    data = data or {}
    return PrepConfig(
        fix_missing_residues=bool(data.get("fix_missing_residues", True)),
        fix_missing_atoms=bool(data.get("fix_missing_atoms", True)),
        add_hydrogens_pH=float(data.get("add_hydrogens_pH", 7.0)),
        remove_heterogens=bool(data.get("remove_heterogens", True)),
        ligand_allow_undefined_stereo=bool(
            data.get("ligand_allow_undefined_stereo", True)
        ),
    )


def _load_solvation_config(data: Mapping[str, Any] | None) -> SolvationConfig:
    data = data or {}
    return SolvationConfig(
        enabled=bool(data.get("enabled", True)),
        water_model=str(data.get("water_model", "tip3p")),
        padding_nm=float(data.get("padding_nm", 1.0)),
        ionic_strength_m=float(data.get("ionic_strength_m", 0.15)),
        positive_ion=str(data.get("positive_ion", "Na+")),
        negative_ion=str(data.get("negative_ion", "Cl-")),
        neutralize=bool(data.get("neutralize", True)),
        solvent_kwargs=dict(data.get("solvent_kwargs", {}) or {}),
    )


def _load_protocol_phase(raw: Mapping[str, Any] | None, *, fallback_steps: int) -> ProtocolPhase:
    raw = raw or {}
    return ProtocolPhase(
        enabled=bool(raw.get("enabled", True)),
        steps=int(raw.get("steps", fallback_steps)),
    )


def _load_minimize_config(raw: Mapping[str, Any] | None) -> MinimizeConfig:
    raw = raw or {}
    return MinimizeConfig(
        enabled=bool(raw.get("enabled", True)),
        max_iterations=int(raw.get("max_iterations", 0)),
        tolerance_kj_per_mol_nm=float(raw.get("tolerance_kj_per_mol_nm", 10.0)),
    )


def _load_protocol_config(data: Mapping[str, Any] | None) -> ProtocolConfig:
    data = data or {}
    return ProtocolConfig(
        minimize=_load_minimize_config(data.get("minimize")),
        nvt=_load_protocol_phase(data.get("nvt"), fallback_steps=50_000),
        npt=_load_protocol_phase(data.get("npt"), fallback_steps=100_000),
        production_steps=int(data.get("production_steps", 1_000_000)),
        nvt_start_temp_k=float(data.get("nvt_start_temp_k", 5.0)),
        nvt_ramp_chunk_steps=int(data.get("nvt_ramp_chunk_steps", 5)),
    )


def _load_integrator_config(data: Mapping[str, Any] | None) -> IntegratorConfig:
    data = data or {}
    return IntegratorConfig(
        type=str(data.get("type", "LangevinMiddle")),
        seed=data.get("seed"),
    )


def _load_engine_config(data: Mapping[str, Any] | None) -> EngineConfig:
    data = data or {}
    switch_val = data.get("switch_distance_nm", None)
    return EngineConfig(
        constraints=data.get("constraints"),
        rigid_water=bool(data.get("rigid_water", False)),
        remove_cm_motion=bool(data.get("remove_cm_motion", False)),
        cutoff_nm=float(data.get("cutoff_nm", 1.0)),
        switch_distance_nm=float(switch_val) if switch_val is not None else None,
        ewald_error_tolerance=float(data.get("ewald_error_tolerance"))
        if data.get("ewald_error_tolerance") is not None
        else None,
        dispersion_correction=data.get("dispersion_correction", None),
        hydrogen_mass=float(data.get("hydrogen_mass"))
        if data.get("hydrogen_mass") is not None
        else None,
        barostat_interval=int(data.get("barostat_interval"))
        if data.get("barostat_interval") is not None
        else None,
        ligand_forcefield=str(data.get("ligand_forcefield", "gaff-2.11")),
    )


def _load_reporter_group(data: Mapping[str, Any] | None, defaults: ReporterGroup) -> ReporterGroup:
    data = data or {}
    return ReporterGroup(
        state_file=str(data.get("state_file", defaults.state_file)),
        dcd_file=str(data.get("dcd_file", defaults.dcd_file)),
        mdcrd_file=str(data.get("mdcrd_file", defaults.mdcrd_file)),
        mdcrd_enabled=bool(data.get("mdcrd_enabled", defaults.mdcrd_enabled)),
        interval=int(data.get("interval", defaults.interval)),
    )


def _load_reporting_config(data: Mapping[str, Any] | None) -> ReportingConfig:
    data = data or {}
    default_eq = ReportingConfig().equilibration
    default_prod = ReportingConfig().production
    return ReportingConfig(
        equilibration=_load_reporter_group(data.get("equilibration"), default_eq),
        production=_load_reporter_group(data.get("production"), default_prod),
        checkpoint_interval=int(data.get("checkpoint_interval"))
        if data.get("checkpoint_interval") is not None
        else None,
        checkpoint_file=str(data.get("checkpoint_file", "sim_checkpoint.chk")),
    )


def _load_simulation_config(data: Mapping[str, Any] | None) -> SimulationConfig:
    data = data or {}
    return SimulationConfig(
        platform=str(data.get("platform", "CUDA")),
        temperature=float(data.get("temperature", 300.0)),
        friction=float(data.get("friction", 1.0)),
        timestep_fs=float(data.get("timestep_fs", 2.0)),
        protocol=_load_protocol_config(data.get("protocol")),
        reporting=_load_reporting_config(data.get("reporting")),
        integrator=_load_integrator_config(data.get("integrator")),
        engine=_load_engine_config(data.get("engine")),
    )


def _load_output_config(data: Mapping[str, Any] | None) -> OutputConfig:
    data = data or {}
    return OutputConfig(
        keep_tmp=bool(data.get("keep_tmp", False)),
        keep_cache=bool(data.get("keep_cache", False)),
        final_pdb=str(data.get("final_pdb", "simulated_complex.pdb")),
        final_state_xml=str(data.get("final_state_xml", "final_state.xml")),
        initial_state_xml=str(data.get("initial_state_xml", "initial_state.xml")),
        minimized_pdb=str(data.get("minimized_pdb", "minimized_complex.pdb")),
        config_copy=str(data.get("config_copy", "config_resolved.yaml")),
        manifest=str(data.get("manifest", "manifest.json")),
    )


def _load_system(raw: Mapping[str, Any], base_dir: Path) -> SystemSpec:
    def _normalize_list(val: Any) -> List[str]:
        if val is None:
            return []
        if isinstance(val, (list, tuple)):
            return list(val)
        return [val]

    protein_paths_raw = _normalize_list(raw.get("protein_paths"))
    ligand_paths_raw = _normalize_list(raw.get("ligand_paths"))
    rna_paths_raw = _normalize_list(raw.get("rna_paths"))

    protein_paths = [
        _coerce_path(v, base_dir) for v in protein_paths_raw if v is not None
    ]
    ligand_paths = [
        _coerce_path(v, base_dir) for v in ligand_paths_raw if v is not None
    ]
    rna_paths = [
        _coerce_path(v, base_dir) for v in rna_paths_raw if v is not None
    ]

    return SystemSpec(
        name=str(raw["name"]),
        protein_paths=protein_paths,
        ligand_paths=ligand_paths,
        rna_paths=rna_paths,
        forcefields=list(raw.get("forcefields", []) or []),
    )


# Analysis loaders reused for compatibility
def _load_rmsd_config(data: Mapping[str, Any] | None, base_dir: Path) -> RMSDConfig:
    data = data or {}
    ref_path = data.get("reference_path")
    return RMSDConfig(
        enabled=bool(data.get("enabled", True)),
        reference=str(data.get("reference", "last_frame")),
        align_selection=str(data.get("align_selection", "protein")),
        reference_path=_coerce_path(ref_path, base_dir) if ref_path else None,
    )


def _load_rmsf_config(data: Mapping[str, Any] | None) -> RMSFConfig:
    data = data or {}
    return RMSFConfig(
        enabled=bool(data.get("enabled", True)),
        selection=str(data.get("selection", "protein")),
    )


def _load_pairwise_rmsd_config(data: Mapping[str, Any] | None) -> PairwiseRMSDConfig:
    data = data or {}
    max_frames = data.get("max_frames")
    return PairwiseRMSDConfig(
        enabled=bool(data.get("enabled", False)),
        selection=str(data.get("selection", "protein")),
        max_frames=int(max_frames) if max_frames is not None else None,
    )


def _load_timeseries_config(data: Mapping[str, Any] | None) -> TimeseriesConfig:
    data = data or {}
    return TimeseriesConfig(
        enabled=bool(data.get("enabled", True)),
        columns=list(data.get("columns", []) or []),
    )


def _load_analysis_config(data: Mapping[str, Any] | None, base_dir: Path) -> AnalysisConfig:
    data = data or {}
    return AnalysisConfig(
        rmsd=_load_rmsd_config(data.get("rmsd"), base_dir),
        rmsf=_load_rmsf_config(data.get("rmsf")),
        pairwise_rmsd=_load_pairwise_rmsd_config(data.get("pairwise_rmsd")),
        timeseries=_load_timeseries_config(data.get("timeseries")),
    )


def _load_mmgbsa_config(data: Mapping[str, Any] | None) -> MMGBSAConfig:
    data = data or {}
    return MMGBSAConfig(
        enabled=bool(data.get("enabled", False)),
    )


# --- YAML loader -------------------------------------------------------------


def load_run_config(path: Path | str) -> RunConfig:
    """
    Load a RunConfig from a YAML file.

    This is the single entrypoint config loader that the CLI and workflow
    should use.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        RunConfig: Loaded configuration object.
    """
    path = Path(path).expanduser().resolve()
    base_dir = path.parent

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    logger.info("Loaded YAML config from %s", path)

    run_name = str(raw.get("run_name", path.stem))
    output_root_raw = raw.get("output_root", "outputs")
    output_root = _coerce_path(output_root_raw, base_dir)

    if "system" not in raw:
        raise ValueError("Configuration must contain a 'system' section.")

    system = _load_system(raw.get("system"), base_dir)
    prep_cfg = _load_prep_config(raw.get("prep"))
    solvation_cfg = _load_solvation_config(raw.get("solvation"))
    sim_cfg = _load_simulation_config(raw.get("simulation"))
    output_cfg = _load_output_config(raw.get("output"))

    analysis_cfg = None
    if "analysis" in raw:
        analysis_cfg = _load_analysis_config(raw.get("analysis"), base_dir)

    mmgbsa_cfg = None
    if "mmgbsa" in raw:
        mmgbsa_cfg = _load_mmgbsa_config(raw.get("mmgbsa"))

    run_cfg = RunConfig(
        run_name=run_name,
        output_root=output_root,
        system=system,
        prep=prep_cfg,
        solvation=solvation_cfg,
        simulation=sim_cfg,
        output=output_cfg,
        analysis=analysis_cfg,
        mmgbsa=mmgbsa_cfg,
    )

    logger.info(
        "Configured run '%s' for system '%s' (output dir: %s)",
        run_cfg.run_name,
        run_cfg.system.name,
        run_cfg.run_output_dir,
    )
    return run_cfg
