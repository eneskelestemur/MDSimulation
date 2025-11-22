'''
    Builder module to create OpenMM systems from specifications and configurations.
'''

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import openmm.app as app
from openmm import unit
from openff.toolkit.topology import Molecule
from openmmforcefields.generators.template_generators import (
    GAFFTemplateGenerator,
    SMIRNOFFTemplateGenerator,
    EspalomaTemplateGenerator,
)
from pdbfixer import PDBFixer
from rdkit import Chem

from .config import (
    EngineConfig,
    OutputConfig,
    PrepConfig,
    SolvationConfig,
    SystemSpec,
)
from .logging_setup import get_logger

logger = get_logger(__name__)


@dataclass
class BuiltSystem:
    """
    Container for geometry/topology built by :class:`InputBuilder`.

    The engine will typically take this and construct an openmm.System
    and openmm.Simulation from it.
    """

    name: str
    output_dir: Path

    topology: app.Topology
    positions: unit.Quantity  # (n_atoms, 3)

    complex_pdb_path: Path
    solvated_pdb_path: Optional[Path] = None

    ligands: List[Molecule] = field(default_factory=list)
    forcefield: Optional[app.ForceField] = None


class InputBuilder:
    """
    Build topology, positions and PDB files for a single system.

    Responsibilities:
        - Load and clean protein / ligand / RNA inputs.
        - Combine them into a single complex using app.Modeller.
        - Optionally solvate and ionize the complex.
        - Write intermediate PDB/SDF files into a per-system `tmp` directory.
        - Return a BuiltSystem with final topology/positions.

    Forcefield parameters are **not** created here; that's handled in the engine.
    We only need a ForceField instance to tell Modeller how to solvate.
    """

    def __init__(
        self,
        spec: SystemSpec,
        run_output_dir: Path,
        prep_cfg: PrepConfig,
        solvation_cfg: SolvationConfig,
        engine_cfg: EngineConfig,
        output_cfg: OutputConfig,
    ) -> None:
        """
        Initialize the InputBuilder.

        Args:
            spec: SystemSpec defining the system to build.
            run_output_dir: Base output directory for the entire run.
        """
        self.spec = spec
        self.prep_cfg = prep_cfg
        self.solvation_cfg = solvation_cfg
        self.engine_cfg = engine_cfg
        self.output_cfg = output_cfg

        self.output_dir = Path(run_output_dir)
        self.prep_dir = self.output_dir / "prep"
        self.tmp_dir = self.output_dir / "tmp"
        self.cache_dir = self.output_dir / "cache"

        # ensure directories exist
        for d in (self.output_dir, self.prep_dir, self.tmp_dir, self.cache_dir):
            d.mkdir(parents=True, exist_ok=True)

        # simple counters
        self.protein_counter = 0
        self.ligand_counter = 0
        self.rna_counter = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self) -> BuiltSystem:
        """
        Build the complex (and optionally the solvated complex).

        Args:
            solvate: Whether to solvate/ionize the complex using Modeller.addSolvent.

        Returns:
            BuiltSystem: Container with topology, positions, and PDB paths.
        """
        logger.info("Building system '%s' in %s", self.spec.name, self.output_dir)

        proteins: List[app.Modeller] = []
        ligands: List[Molecule] = []
        ligand_modellers: List[app.Modeller] = []
        rnas: List[app.Modeller] = []

        for protein_path in self.spec.protein_paths:
            proteins.append(self._load_protein(protein_path))

        for ligand_path in self.spec.ligand_paths:
            ligand_mol, ligand_mod = self._load_ligand(ligand_path)
            ligands.append(ligand_mol)
            ligand_modellers.append(ligand_mod)

        for rna_path in self.spec.rna_paths:
            rnas.append(self._load_rna(rna_path))

        if not proteins and not ligands and not rnas:
            raise ValueError(
                f"System '{self.spec.name}' has no protein, ligand, or RNA defined."
            )

        topology, positions, complex_pdb, solvated_pdb, forcefield = self._prepare_complex(
            proteins=proteins,
            ligand_modellers=ligand_modellers,
            rnas=rnas,
            ligands=ligands,
        )

        logger.info(
            "Built system '%s': %d atoms (solvated=%s)",
            self.spec.name,
            topology.getNumAtoms(),
            bool(solvated_pdb),
        )

        return BuiltSystem(
            name=self.spec.name,
            output_dir=self.output_dir,
            topology=topology,
            positions=positions,
            complex_pdb_path=complex_pdb,
            solvated_pdb_path=solvated_pdb,
            ligands=ligands,
            forcefield=forcefield,
        )

    # ------------------------------------------------------------------
    # Component loaders
    # ------------------------------------------------------------------

    def _load_protein(self, protein_file: Path | str) -> app.Modeller:
        """
        Load and clean a protein structure using PDBFixer.

        Steps:
            - findMissingResidues / findMissingAtoms / findNonstandardResidues
            - addMissingAtoms / addMissingHydrogens(pH 7.0)
            - remove heterogens (including waters)
            - write a cleaned PDB into tmp/

        Args:
            protein_file: Path to the input protein PDB file.

        Returns:
            app.PDBFile: Cleaned protein structure.
        """
        protein_file = Path(protein_file)
        logger.info("Loading protein from %s", protein_file)

        fixer = PDBFixer(str(protein_file))

        fix_missing_residues = self.prep_cfg.fix_missing_residues
        fix_missing_atoms = self.prep_cfg.fix_missing_atoms

        if fix_missing_residues:
            fixer.findMissingResidues()
        if fix_missing_atoms:
            fixer.findMissingAtoms()
        fixer.findNonstandardResidues()

        logger.debug("Protein missing residues: %s", getattr(fixer, "missingResidues", None))
        logger.debug("Protein missing atoms: %s", getattr(fixer, "missingAtoms", None))
        logger.debug("Protein nonstandard residues: %s", getattr(fixer, "nonstandardResidues", None))

        if fix_missing_atoms:
            fixer.addMissingAtoms()
        fixer.addMissingHydrogens(self.prep_cfg.add_hydrogens_pH)
        if self.prep_cfg.remove_heterogens:
            fixer.removeHeterogens(False)

        out_path = self.prep_dir / f"protein_clean_{self.protein_counter}.pdb"
        logger.info("Writing cleaned protein to %s", out_path)
        with out_path.open("w") as fh:
            app.PDBFile.writeFile(fixer.topology, fixer.positions, fh)

        modeller = app.Modeller(fixer.topology, fixer.positions)
        self.protein_counter += 1
        return modeller

    def _load_ligand(self, ligand_file: Path | str) -> Tuple[Molecule, app.Modeller]:
        """
        Load and standardize a ligand using OpenFF + RDKit.

        Args:
            ligand_file: Path to the input ligand file (.sdf, .mol2, or .pdb).

        Returns:
            Molecule: Standardized ligand molecule (OpenFF Molecule).
        """
        ligand_file = Path(ligand_file)
        logger.info("Loading ligand from %s", ligand_file)

        suffix = ligand_file.suffix.lower()
        if suffix == ".sdf":
            ligand = Molecule.from_file(
                str(ligand_file),
                file_format="sdf",
                allow_undefined_stereo=self.prep_cfg.ligand_allow_undefined_stereo,
            )
        elif suffix == ".mol2":
            mol = Chem.MolFromMol2File(str(ligand_file), removeHs=False)
            if mol is None:
                raise ValueError(f"RDKit failed to read mol2 ligand file: {ligand_file}")
            ligand = Molecule.from_rdkit(
                mol, allow_undefined_stereo=self.prep_cfg.ligand_allow_undefined_stereo
            )
        elif suffix == ".pdb":
            logger.warning(
                "Loading ligand from PDB is not ideal; positional/chemical information may be ambiguous."
            )
            mol = Chem.MolFromPDBFile(str(ligand_file), removeHs=False)
            if mol is None:
                raise ValueError(f"RDKit failed to read PDB ligand file: {ligand_file}")
            ligand = Molecule.from_rdkit(
                mol, allow_undefined_stereo=self.prep_cfg.ligand_allow_undefined_stereo
            )
        else:
            raise ValueError(
                f"Ligand file format not recognized for {ligand_file}. "
                "Expected .sdf, .mol2 or .pdb."
            )

        if isinstance(ligand, list):
            raise ValueError("Ligand file should contain exactly one molecule.")

        out_path = self.prep_dir / f"ligand_canonical_{self.ligand_counter}.sdf"
        logger.info("Writing standardized ligand to %s", out_path)
        ligand.to_file(str(out_path), file_format="sdf")

        topology = ligand.to_topology()
        modeller = app.Modeller(
            topology.to_openmm(),
            topology.get_positions().to_openmm(),
        )

        self.ligand_counter += 1
        return ligand, modeller

    def _load_rna(self, rna_file: Path | str) -> app.Modeller:
        """
        Load and clean an RNA structure using PDBFixer.
        
        Steps:
            - findMissingResidues / findMissingAtoms / findNonstandardResidues
            - addMissingAtoms / addMissingHydrogens(pH 7.0)
            - remove heterogens (including waters)
            - write a cleaned PDB into tmp/

        Args:
            rna_file: Path to the input RNA PDB file.

        Returns:
            app.PDBFile: Cleaned RNA structure.
        """
        rna_file = Path(rna_file)
        logger.info("Loading RNA from %s", rna_file)

        fixer = PDBFixer(str(rna_file))

        # For now we mirror the protein behaviour; can diverge later via extra options
        fixer.findMissingResidues()
        fixer.findMissingAtoms()
        fixer.findNonstandardResidues()

        logger.debug("RNA missing residues: %s", getattr(fixer, "missingResidues", None))
        logger.debug("RNA missing atoms: %s", getattr(fixer, "missingAtoms", None))
        logger.debug("RNA nonstandard residues: %s", getattr(fixer, "nonstandardResidues", None))

        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(self.prep_cfg.add_hydrogens_pH)
        if self.prep_cfg.remove_heterogens:
            fixer.removeHeterogens(False)

        out_path = self.prep_dir / f"rna_clean_{self.rna_counter}.pdb"
        logger.info("Writing cleaned RNA to %s", out_path)
        with out_path.open("w") as fh:
            app.PDBFile.writeFile(fixer.topology, fixer.positions, fh)

        modeller = app.Modeller(fixer.topology, fixer.positions)
        self.rna_counter += 1
        return modeller

    # ------------------------------------------------------------------
    # Complex assembly and solvation
    # ------------------------------------------------------------------

    def _prepare_complex(
        self,
        proteins: List[app.Modeller],
        ligand_modellers: List[app.Modeller],
        rnas: List[app.Modeller],
        ligands: List[Molecule],
    ) -> tuple[app.Topology, unit.Quantity, Path, Optional[Path], app.ForceField]:
        """
        Combine components into a complex and optionally solvate.

        Args:
            proteins: List of protein Modeller objects.
            ligand_modellers: List of ligand Modeller objects (positions + topology).
            rnas: List of RNA Modeller objects.
            ligands: List of ligand Molecule objects (for template generation).
        
        Returns:
            (topology: app.Topology  
            positions: unit.Quantity  
            complex_pdb_path: Path  
            solvated_pdb_path: Optional[Path])
        """
        # Initial modeller priority: protein > RNA > ligand
        if proteins:
            modeller = proteins[0]
            initial = ("protein", 0)
        elif rnas:
            modeller = rnas[0]
            initial = ("rna", 0)
        elif ligand_modellers:
            modeller = ligand_modellers[0]
            initial = ("ligand", 0)
        else:
            raise ValueError(
                f"System '{self.spec.name}' has no protein, ligand, or RNA defined."
            )

        # Add remaining components
        for idx, protein in enumerate(proteins):
            if ("protein", idx) == initial:
                continue
            modeller.add(protein.topology, protein.positions)

        for idx, rna in enumerate(rnas):
            if ("rna", idx) == initial:
                continue
            modeller.add(rna.topology, rna.positions)

        for idx, lig_mod in enumerate(ligand_modellers):
            if ("ligand", idx) == initial:
                continue
            modeller.add(lig_mod.topology, lig_mod.positions)

        # Write the unsolvated complex
        complex_pdb_path = self.prep_dir / "complex.pdb"
        logger.info("Writing complex (unsolvated) to %s", complex_pdb_path)
        with complex_pdb_path.open("w") as fh:
            app.PDBFile.writeFile(modeller.topology, modeller.positions, fh)

        solvated_pdb_path: Optional[Path] = None
        forcefield = self._create_forcefield(ligands)

        # Solvate the complex if requested
        if self.solvation_cfg.enabled:
            logger.info(
                "Solvating complex '%s' with model=%s, padding=%.3f nm, ionic_strength=%.3f M",
                self.spec.name,
                self.solvation_cfg.water_model,
                self.solvation_cfg.padding_nm,
                self.solvation_cfg.ionic_strength_m,
            )

            padding = self.solvation_cfg.padding_nm * unit.nanometer

            default_kwargs: Dict[str, Any] = {
                "model": self.solvation_cfg.water_model,
                "padding": padding,
                "positiveIon": self.solvation_cfg.positive_ion,
                "negativeIon": self.solvation_cfg.negative_ion,
                "ionicStrength": self.solvation_cfg.ionic_strength_m * unit.molar,
                "neutralize": self.solvation_cfg.neutralize,
            }
            # Allow per-system overrides via solvent_kwargs
            extra_kwargs = self.solvation_cfg.solvent_kwargs or {}
            default_kwargs.update(extra_kwargs)

            modeller.addSolvent(forcefield=forcefield, **default_kwargs)

            solvated_pdb_path = self.prep_dir / "complex_solvated.pdb"
            logger.info("Writing solvated complex to %s", solvated_pdb_path)
            with solvated_pdb_path.open("w") as fh:
                app.PDBFile.writeFile(modeller.topology, modeller.positions, fh)

        topology = modeller.topology
        positions = modeller.positions  # unit.Quantity

        return topology, positions, complex_pdb_path, solvated_pdb_path, forcefield

    # ------------------------------------------------------------------
    # ForceField helpers
    # ------------------------------------------------------------------

    def _create_forcefield(self, ligands: List[Molecule]) -> app.ForceField:
        if not self.spec.forcefields:
            raise ValueError(
                f"System '{self.spec.name}' requested parameterization but no forcefields were specified."
            )

        ff = app.ForceField(*self.spec.forcefields)

        if ligands:
            ligand_ff = self.engine_cfg.ligand_forcefield.lower()
            cache_fname = str(self.cache_dir / "ligand_templates.json")
            logger.info("Registering ligand template generator with FF=%s", self.engine_cfg.ligand_forcefield)
            if "gaff" in ligand_ff:
                generator = GAFFTemplateGenerator(
                    molecules=ligands,
                    forcefield=self.engine_cfg.ligand_forcefield,
                    cache=cache_fname,
                )
            elif "smirnoff" in ligand_ff:
                generator = SMIRNOFFTemplateGenerator(
                    molecules=ligands,
                    forcefield=self.engine_cfg.ligand_forcefield,
                    cache=cache_fname,
                )
            elif "espaloma" in ligand_ff:
                generator = EspalomaTemplateGenerator(
                    molecules=ligands,
                    forcefield=self.engine_cfg.ligand_forcefield,
                    cache=cache_fname,
                    template_generator_kwargs={"charge_method": "nn"},
                )
            else:
                raise ValueError(f"Unrecognized ligand forcefield '{self.engine_cfg.ligand_forcefield}'")

            ff.registerTemplateGenerator(generator.generator)

        return ff
