from __future__ import annotations

"""
MMGBSA driver: prepares Amber prmtop/inpcrd files and runs MMPBSA.py using mdcrd trajectories.

Assumes AmberTools is available on PATH. All intermediates go to tmp/, inputs in cache/, results in analysis/.
"""

import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

import openmm.app as app
from openmm import unit
import parmed as pmd
from openff.toolkit.topology import Molecule
from openmmforcefields.generators.template_generators import (
    GAFFTemplateGenerator,
    SMIRNOFFTemplateGenerator,
    EspalomaTemplateGenerator,
)

from ..config import MMGBSAConfig, RunConfig, SystemSpec
from ..engine import SimulationEngine
from ..logging_setup import get_logger

logger = get_logger(__name__)


def run_mmgbsa(run_dir: Path, run_cfg: RunConfig) -> Path:
    """
    Run MMGBSA analysis for a completed simulation using AmberTools MMPBSA.py.

    Args:
        run_dir: Path to the completed run directory.
        run_cfg: Loaded RunConfig (ideally config_resolved.yaml from the run).

    Returns:
        Path to the results file (mmgbsa_results.dat) under analysis/.
    """
    mmgbsa_cfg = run_cfg.mmgbsa
    if not mmgbsa_cfg or not mmgbsa_cfg.enabled:
        raise ValueError("MMGBSA is not enabled in the provided configuration.")

    prep_dir = Path(run_dir) / "prep"
    sim_dir = Path(run_dir) / "sim"
    tmp_dir = Path(run_dir) / "tmp"
    cache_dir = Path(run_dir) / "cache"
    analysis_dir = Path(run_dir) / "analysis"

    tmp_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    complex_pdb = prep_dir / "complex.pdb"
    solvated_pdb = prep_dir / "complex_solvated.pdb"
    if not complex_pdb.exists() or not solvated_pdb.exists():
        raise FileNotFoundError("complex.pdb or complex_solvated.pdb not found in prep/.")

    mdcrd_files = sorted(sim_dir.glob("*.mdcrd"))
    if not mdcrd_files:
        raise FileNotFoundError("No mdcrd trajectory found in sim/. Enable mdcrd output or rerun simulation.")

    ligands = _load_ligands_from_prep(prep_dir)
    ff = _create_forcefield(run_cfg.system, run_cfg.simulation.engine, ligands, cache_dir / "ligand_templates.json")

    # Build prmtops/inpcrd under tmp/
    prmtops = _write_prmtops(
        complex_pdb=complex_pdb,
        solvated_pdb=solvated_pdb,
        forcefield=ff,
        engine_cfg=run_cfg.simulation.engine,
        tmp_dir=tmp_dir,
        mmgbsa_cfg=mmgbsa_cfg,
    )

    mmgbsa_in = cache_dir / "mmgbsa.in"
    results_path = analysis_dir / "mmgbsa_results.dat"
    _write_mmgbsa_input(mmgbsa_cfg, mmgbsa_in)

    _run_mmpbsa(
        mmgbsa_in=mmgbsa_in,
        results=results_path,
        prmtops=prmtops,
        mdcrd_files=mdcrd_files,
        tmp_dir=tmp_dir,
    )

    logger.info("MMGBSA finished. Results at %s", results_path)
    return results_path


def _load_ligands_from_prep(prep_dir: Path) -> List[Molecule]:
    """
    Load canonicalized ligands from prep/ as OpenFF Molecules.
    """
    lig_files = sorted(prep_dir.glob("ligand_canonical_*.sdf"))
    ligands: List[Molecule] = []
    for lf in lig_files:
        ligands.append(Molecule.from_file(str(lf), file_format="sdf", allow_undefined_stereo=True))
    return ligands


def _create_forcefield(
    spec: SystemSpec,
    engine_cfg,
    ligands: List[Molecule],
    cache_path: Path,
) -> app.ForceField:
    """
    Build a ForceField with ligand templates, using the same ligand FF as the engine.
    """
    if not spec.forcefields:
        raise ValueError("System has no forcefields defined; cannot build MMGBSA prmtops.")
    ff = app.ForceField(*spec.forcefields)
    if ligands:
        ligand_ff = engine_cfg.ligand_forcefield.lower()
        logger.info("Registering ligand template generator with FF=%s", engine_cfg.ligand_forcefield)
        if "gaff" in ligand_ff:
            generator = GAFFTemplateGenerator(
                molecules=ligands,
                forcefield=engine_cfg.ligand_forcefield,
                cache=str(cache_path),
            )
        elif "smirnoff" in ligand_ff:
            generator = SMIRNOFFTemplateGenerator(
                molecules=ligands,
                forcefield=engine_cfg.ligand_forcefield,
                cache=str(cache_path),
            )
        elif "espaloma" in ligand_ff:
            generator = EspalomaTemplateGenerator(
                molecules=ligands,
                forcefield=engine_cfg.ligand_forcefield,
                cache=str(cache_path),
                template_generator_kwargs={"charge_method": "nn"},
            )
        else:
            raise ValueError(f"Unrecognized ligand forcefield '{engine_cfg.ligand_forcefield}'")
        ff.registerTemplateGenerator(generator.generator)
    return ff


def _write_prmtops(
    complex_pdb: Path,
    solvated_pdb: Path,
    forcefield: app.ForceField,
    engine_cfg,
    tmp_dir: Path,
    mmgbsa_cfg: MMGBSAConfig,
) -> dict:
    """
    Create Amber prmtop/inpcrd files for solvated/unsolvated complex and receptor/ligand splits.
    """
    # Create systems
    constraints = SimulationEngine._map_constraints(engine_cfg.constraints)
    kwargs = {
        "constraints": constraints,
        "rigidWater": engine_cfg.rigid_water,
        "removeCMMotion": engine_cfg.remove_cm_motion,
    }
    periodic_kwargs = {
        "nonbondedMethod": app.PME,
    }
    cutoff_nm = float(engine_cfg.cutoff_nm)
    if cutoff_nm:
        periodic_kwargs["nonbondedCutoff"] = cutoff_nm * unit.nanometer
        if engine_cfg.switch_distance_nm:
            periodic_kwargs["switchDistance"] = float(engine_cfg.switch_distance_nm) * unit.nanometer
    if engine_cfg.ewald_error_tolerance is not None:
        periodic_kwargs["ewaldErrorTolerance"] = engine_cfg.ewald_error_tolerance
    if engine_cfg.dispersion_correction is not None:
        periodic_kwargs["useDispersionCorrection"] = bool(engine_cfg.dispersion_correction)
    if engine_cfg.hydrogen_mass is not None:
        kwargs["hydrogenMass"] = engine_cfg.hydrogen_mass * unit.amu

    solvated_pdb_obj = app.PDBFile(str(solvated_pdb))
    solvated_system = forcefield.createSystem(
        solvated_pdb_obj.topology,
        **kwargs,
        **periodic_kwargs,
    )
    complex_pdb_obj = app.PDBFile(str(complex_pdb))
    complex_system = forcefield.createSystem(
        complex_pdb_obj.topology,
        **kwargs,
        **periodic_kwargs,
    )

    prmtops = {}
    solvated_structure = pmd.openmm.load_topology(
        solvated_pdb_obj.topology, solvated_system, xyz=solvated_pdb_obj.positions
    )
    complex_structure = pmd.openmm.load_topology(
        complex_pdb_obj.topology, complex_system, xyz=complex_pdb_obj.positions
    )

    solvated_prmtop = tmp_dir / "_complex_solvated.prmtop"
    solvated_inpcrd = tmp_dir / "_complex_solvated.inpcrd"
    complex_prmtop = tmp_dir / "_complex.prmtop"
    complex_inpcrd = tmp_dir / "_complex.inpcrd"

    solvated_structure.save(str(solvated_prmtop), overwrite=True)
    solvated_structure.save(str(solvated_inpcrd), overwrite=True)
    complex_structure.save(str(complex_prmtop), overwrite=True)
    complex_structure.save(str(complex_inpcrd), overwrite=True)

    # Split receptor/ligand
    lig_mask, rec_mask = _determine_masks(complex_structure, mmgbsa_cfg)
    receptor_prmtop = tmp_dir / "_receptor.prmtop"
    ligand_prmtop = tmp_dir / "_ligand.prmtop"
    complex_structure[rec_mask].save(str(receptor_prmtop), overwrite=True)
    complex_structure[lig_mask].save(str(ligand_prmtop), overwrite=True)

    prmtops.update(
        {
            "solvated_prmtop": solvated_prmtop,
            "complex_prmtop": complex_prmtop,
            "receptor_prmtop": receptor_prmtop,
            "ligand_prmtop": ligand_prmtop,
        }
    )
    return prmtops


def _determine_masks(structure: pmd.Structure, cfg: MMGBSAConfig) -> Tuple[str, str]:
    """
    Choose receptor/ligand masks. If user provided both masks, use them; else auto-detect ligand by first non-protein residue.
    """
    if cfg.ligand_mask and cfg.receptor_mask:
        return cfg.ligand_mask, cfg.receptor_mask

    # Heuristic: choose first residue that is not water/ion and not a common protein residue
    protein_resnames = {
        "ALA",
        "CYS",
        "ASP",
        "GLU",
        "PHE",
        "GLY",
        "HIS",
        "ILE",
        "LYS",
        "LEU",
        "MET",
        "ASN",
        "PRO",
        "GLN",
        "ARG",
        "SER",
        "THR",
        "VAL",
        "TRP",
        "TYR",
    }
    ligand_resname: Optional[str] = None
    for res in structure.residues:
        name = res.name.strip()
        if name in {"WAT", "HOH", "NA", "CL", "CIO", "CS", "IB", "K", "LI", "MG", "RB"}:
            continue
        if name in protein_resnames:
            continue
        ligand_resname = name
        break

    if ligand_resname is None:
        raise ValueError(
            "Could not auto-detect ligand mask; please set mmgbsa.ligand_mask and receptor_mask in config_resolved."
        )

    lig_mask = f":{ligand_resname}"
    rec_mask = f"!{lig_mask}"
    return lig_mask, rec_mask


def _write_mmgbsa_input(cfg: MMGBSAConfig, path: Path) -> None:
    """Render a minimal mmgbsa.in file from config."""
    lines = []
    lines.append("&general")
    lines.append(f"  keep_files={cfg.keep_files}, debug_printlevel={cfg.debug_printlevel},")
    lines.append(f'  strip_mask="{cfg.strip_mask}",')
    lines.append(f"  startframe={cfg.startframe}, endframe={cfg.endframe}, interval={cfg.interval},")
    if cfg.decomposition:
        lines.append(f"  decomposition={cfg.decomposition},")
    lines.append("/")
    if cfg.gb_enabled:
        lines.append("&gb")
        lines.append(f"  igb={cfg.igb}, saltcon={cfg.saltcon},")
        lines.append("/")
    if cfg.pb_enabled:
        lines.append("&pb")
        lines.append(f"  istrng={cfg.istrng}, radiopt={cfg.radiopt}, inp={cfg.inp},")
        lines.append("/")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote mmgbsa input to %s", path)


def _run_mmpbsa(
    mmgbsa_in: Path,
    results: Path,
    prmtops: dict,
    mdcrd_files: List[Path],
    tmp_dir: Path,
) -> None:
    """
    Execute MMPBSA.py with prepared inputs. Requires AmberTools on PATH.
    """
    mdcrd_glob = " ".join(str(p) for p in mdcrd_files)
    cmd = [
        "MMPBSA.py",
        "-O",
        "-i",
        str(mmgbsa_in),
        "-o",
        str(results),
        "-sp",
        str(prmtops["solvated_prmtop"]),
        "-cp",
        str(prmtops["complex_prmtop"]),
        "-rp",
        str(prmtops["receptor_prmtop"]),
        "-lp",
        str(prmtops["ligand_prmtop"]),
    ]
    cmd.extend(["-y", mdcrd_glob])
    logger.info("Running MMPBSA.py")
    proc = subprocess.run(
        " ".join(cmd), shell=True, capture_output=True, text=True, cwd=str(tmp_dir)
    )
    if proc.returncode != 0:
        logger.error("MMPBSA.py failed:\nstdout:\n%s\nstderr:\n%s", proc.stdout, proc.stderr)
        raise RuntimeError("MMPBSA.py failed; see logs for details.")
