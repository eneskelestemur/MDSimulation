'''
    MD simulation engine: runs minimization, equilibration, and production.
'''

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

import time

import openmm as mm
import openmm.app as app
from openmm import unit
import parmed as pmd

from .builder import BuiltSystem
from .config import OutputConfig, SimulationConfig, SystemSpec
from .logging_setup import get_logger

logger = get_logger(__name__)


@dataclass
class SimulationResult:
    """
    File-based record of a completed simulation.

    This is what workflow/analysis code should consume.
    """

    name: str
    output_dir: Path

    # Equilibration output
    equil_state_data: Optional[Path]
    equil_trajectory_dcd: Optional[Path]
    equil_trajectory_mdcrd: Optional[Path]

    # Production output
    state_data: Path
    trajectory_dcd: Path
    trajectory_mdcrd: Path

    # Final state
    final_state_xml: Path
    final_pdb: Path

    initial_state_xml: Path
    checkpoint: Optional[Path]


class SimulationEngine:
    """
    Responsible for turning a BuiltSystem into a full MD run.

    It does NOT know about YAML; it only consumes dataclasses and
    returns a SimulationResult.
    """

    def __init__(
        self,
        cfg: SimulationConfig,
        output_cfg: OutputConfig,
        *,
        mmgbsa_enabled: bool = False,
    ) -> None:
        self.cfg = cfg
        self.output_cfg = output_cfg
        self.mmgbsa_enabled = mmgbsa_enabled
        self.platform = mm.Platform.getPlatformByName(cfg.platform)
        logger.info("Initialized SimulationEngine with platform=%s", cfg.platform)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        built: BuiltSystem,
        spec: SystemSpec,
        sim_dir: Path,
        tmp_dir: Path,
        cache_dir: Path,
    ) -> SimulationResult:
        """
        Run a full MD protocol: minimize → NVT → NPT → production.

        Args:
            built: BuiltSystem to simulate.
            spec: SystemSpec with forcefield info.
            sim_dir: Directory for simulation outputs.
            tmp_dir: Directory for temporary files.
            cache_dir: Directory for template/FF caches.

        Returns:
            SimulationResult: Record of the completed simulation.
        """
        solvate = built.solvated_pdb_path is not None
        n_steps = self.cfg.protocol.production_steps

        self.tmp_dir = Path(tmp_dir)
        self.cache_dir = Path(cache_dir)
        sim_dir = Path(sim_dir)
        sim_dir.mkdir(parents=True, exist_ok=True)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Starting MD for '%s' (solvated=%s, production_steps=%d)",
            built.name,
            bool(solvate),
            n_steps,
        )

        simulation = self._prepare_simulation(built, spec, solvate=solvate)

        # Core artifact paths
        equil_state = sim_dir / self.cfg.reporting.equilibration.state_file
        equil_dcd = sim_dir / self.cfg.reporting.equilibration.dcd_file
        equil_mdcrd = (
            sim_dir / self.cfg.reporting.equilibration.mdcrd_file
            if (self.cfg.reporting.equilibration.mdcrd_enabled or self.mmgbsa_enabled)
            else None
        )

        state_data = sim_dir / self.cfg.reporting.production.state_file
        traj_dcd = sim_dir / self.cfg.reporting.production.dcd_file
        traj_mdcrd = (
            sim_dir / self.cfg.reporting.production.mdcrd_file
            if (self.cfg.reporting.production.mdcrd_enabled or self.mmgbsa_enabled)
            else None
        )

        final_state_xml = sim_dir / self.output_cfg.final_state_xml
        final_pdb = sim_dir / self.output_cfg.final_pdb
        initial_state_xml = sim_dir / self.output_cfg.initial_state_xml
        minimized_pdb = sim_dir / self.output_cfg.minimized_pdb
        checkpoint_path: Optional[Path] = (
            sim_dir / self.cfg.reporting.checkpoint_file
            if self.cfg.reporting.checkpoint_interval
            else None
        )

        # Save initial state
        simulation.saveState(str(initial_state_xml))
        # Save starting positions as a placeholder minimized structure (overwrite after minimization)
        self._write_pdb(simulation, minimized_pdb)

        # Equilibration reporters (attach only if we will run NVT/NPT)
        equil_reporters: list[app.Reporter] = []
        if self.cfg.protocol.nvt.enabled or self.cfg.protocol.npt.enabled:
            equil_reporters = [
                app.StateDataReporter(
                    str(equil_state),
                    self.cfg.reporting.equilibration.interval,
                    step=True,
                    time=True,
                    potentialEnergy=True,
                    totalEnergy=True,
                    temperature=True,
                    density=True,
                    progress=False,
                ),
                app.DCDReporter(str(equil_dcd), self.cfg.reporting.equilibration.interval),
            ]
            if equil_mdcrd:
                equil_reporters.append(
                    pmd.openmm.MdcrdReporter(
                        str(equil_mdcrd),
                        self.cfg.reporting.equilibration.interval,
                        crds=True,
                    )
                )
            if checkpoint_path and self.cfg.reporting.checkpoint_interval:
                equil_reporters.append(
                    app.CheckpointReporter(
                        str(checkpoint_path), self.cfg.reporting.checkpoint_interval
                    )
                )

        # Production reporters
        sim_reporters = [
            app.StateDataReporter(
                str(state_data),
                self.cfg.reporting.production.interval,
                step=True,
                time=True,
                potentialEnergy=True,
                totalEnergy=True,
                temperature=True,
                density=True,
                progress=False,
            ),
            app.DCDReporter(str(traj_dcd), self.cfg.reporting.production.interval),
        ]
        if traj_mdcrd:
            sim_reporters.append(
                pmd.openmm.MdcrdReporter(
                    str(traj_mdcrd),
                    self.cfg.reporting.production.interval,
                    crds=True,
                )
            )
        if checkpoint_path and self.cfg.reporting.checkpoint_interval:
            sim_reporters.append(
                app.CheckpointReporter(
                    str(checkpoint_path), self.cfg.reporting.checkpoint_interval
                )
            )

        # Attach equil reporters
        for rep in equil_reporters:
            simulation.reporters.append(rep)

        # Minimization
        if self.cfg.protocol.minimize.enabled:
            logger.info(
                "Minimizing energy for '%s' (max %d iterations, tolerance=%.3f kJ/mol/nm)",
                built.name,
                self.cfg.protocol.minimize.max_iterations,
                self.cfg.protocol.minimize.tolerance_kj_per_mol_nm,
            )
            t0 = time.time()
            self._minimize(
                simulation,
                max_iterations=self.cfg.protocol.minimize.max_iterations,
                tolerance_kj_per_mol_nm=self.cfg.protocol.minimize.tolerance_kj_per_mol_nm,
            )
            self._write_pdb(simulation, minimized_pdb)
            logger.info(
                "Minimization for '%s' finished in %.2f s",
                built.name,
                time.time() - t0,
            )

        # NVT equilibration
        if self.cfg.protocol.nvt.enabled:
            logger.info(
                "Running NVT equilibration for '%s' (%d steps)",
                built.name,
                self.cfg.protocol.nvt.steps,
            )
            t0 = time.time()
            self._equilibrate_nvt(
                simulation,
                num_steps=self.cfg.protocol.nvt.steps,
                start_temp_k=self.cfg.protocol.nvt_start_temp_k,
                target_temp_k=self.cfg.temperature,
                ramp_chunk_steps=self.cfg.protocol.nvt_ramp_chunk_steps,
            )
            logger.info(
                "NVT equilibration for '%s' finished in %.2f s",
                built.name,
                time.time() - t0,
            )

        # NPT equilibration
        if self.cfg.protocol.npt.enabled:
            logger.info(
                "Running NPT equilibration for '%s' (%d steps)",
                built.name,
                self.cfg.protocol.npt.steps,
            )
            t0 = time.time()
            self._equilibrate_npt(
                simulation,
                num_steps=self.cfg.protocol.npt.steps,
                temperature_k=self.cfg.temperature,
            )
            logger.info(
                "NPT equilibration for '%s' finished in %.2f s",
                built.name,
                time.time() - t0,
            )

        # Remove equil reporters before production
        simulation.reporters.clear()

        # Attach production reporters
        for rep in sim_reporters:
            simulation.reporters.append(rep)

        # Production run
        logger.info("Running production for '%s' (%d steps)", built.name, n_steps)
        t0 = time.time()
        self._run_production(simulation, num_steps=n_steps)
        logger.info(
            "Production for '%s' finished in %.2f minutes (%.3f ns simulated)",
            built.name,
            (time.time() - t0) / 60.0,
            self.cfg.timestep_fs * simulation.currentStep / 1_000_000.0,
        )

        # Save final state & structure
        logger.info("Saving final state for '%s'", built.name)
        simulation.saveState(str(final_state_xml))
        positions = simulation.context.getState(
            getPositions=True,
            enforcePeriodicBox=True,
        ).getPositions()
        with final_pdb.open("w") as fh:
            app.PDBFile.writeFile(simulation.topology, positions, fh)

        return SimulationResult(
            name=built.name,
            output_dir=sim_dir,
            equil_state_data=equil_state,
            equil_trajectory_dcd=equil_dcd,
            equil_trajectory_mdcrd=equil_mdcrd,
            state_data=state_data,
            trajectory_dcd=traj_dcd,
            trajectory_mdcrd=traj_mdcrd,
            final_state_xml=final_state_xml,
            final_pdb=final_pdb,
            initial_state_xml=initial_state_xml,
            checkpoint=checkpoint_path,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_simulation(
        self,
        built: BuiltSystem,
        spec: SystemSpec,
        *,
        solvate: bool,
    ) -> app.Simulation:
        """
        Build an OpenMM Simulation from a BuiltSystem + SystemSpec.

        Args:
            built: BuiltSystem to simulate.
            spec: SystemSpec with forcefield info.
            solvate: Whether to solvate/ionize the complex using Modeller.addSolvent

        Returns:
            app.Simulation: Prepared OpenMM Simulation object.
        """
        logger.info(
            "Preparing Simulation for '%s' (solvated=%s, ligands=%d)",
            built.name,
            bool(solvate),
            len(built.ligands),
        )

        integrator = self._create_integrator()

        # Forcefield/system options
        eng = self.cfg.engine
        constraints = self._map_constraints(eng.constraints)

        forcefield_kwargs: Dict[str, Any] = {
            "constraints": constraints,
            "rigidWater": eng.rigid_water,
            "removeCMMotion": eng.remove_cm_motion,
        }

        cutoff_nm = float(eng.cutoff_nm)
        switch_nm = eng.switch_distance_nm

        # Non-periodic defaults
        nonperiodic_kwargs: Dict[str, Any] = {
            "nonbondedMethod": app.NoCutoff,
        }
        if cutoff_nm:
            nonperiodic_kwargs["nonbondedCutoff"] = cutoff_nm * unit.nanometer
            if switch_nm:
                nonperiodic_kwargs["switchDistance"] = switch_nm * unit.nanometer

        # Periodic defaults
        periodic_kwargs: Dict[str, Any] = {
            "nonbondedMethod": app.PME,
        }
        if cutoff_nm:
            periodic_kwargs["nonbondedCutoff"] = cutoff_nm * unit.nanometer
            if switch_nm:
                periodic_kwargs["switchDistance"] = switch_nm * unit.nanometer

        if eng.ewald_error_tolerance is not None:
            periodic_kwargs["ewaldErrorTolerance"] = eng.ewald_error_tolerance
        if eng.dispersion_correction is not None:
            periodic_kwargs["useDispersionCorrection"] = bool(eng.dispersion_correction)

        hydrogen_mass = (
            eng.hydrogen_mass * unit.amu if eng.hydrogen_mass is not None else None
        )
        if hydrogen_mass is not None:
            forcefield_kwargs["hydrogenMass"] = hydrogen_mass

        if not built.forcefield:
            raise ValueError(
                f"BuiltSystem for '{built.name}' does not contain a prepared ForceField."
            )

        if solvate:
            system = built.forcefield.createSystem(
                built.topology,
                **forcefield_kwargs,
                **periodic_kwargs,
            )
        else:
            system = built.forcefield.createSystem(
                built.topology,
                **forcefield_kwargs,
                **nonperiodic_kwargs,
            )

        # Optional barostat
        barostat_interval = eng.barostat_interval
        if solvate and barostat_interval:
            logger.info(
                "Adding MonteCarloBarostat to '%s' (interval=%d)",
                built.name,
                int(barostat_interval),
            )
            barostat = mm.MonteCarloBarostat(
                1.0 * unit.atmosphere,
                self.cfg.temperature * unit.kelvin,
                int(barostat_interval),
            )
            system.addForce(barostat)

        # Simulation object
        simulation = app.Simulation(
            built.topology,
            system,
            integrator,
            platform=self.platform,
        )
        simulation.context.setPositions(built.positions)

        return simulation

    def _create_integrator(self) -> mm.Integrator:
        integ_type = self.cfg.integrator.type.lower()
        timestep = self.cfg.timestep_fs * unit.femtoseconds
        temperature = self.cfg.temperature * unit.kelvin
        friction = self.cfg.friction / unit.picosecond

        if integ_type in {"langevinmiddle", "middle"}:
            integrator = mm.LangevinMiddleIntegrator(
                temperature,
                friction,
                timestep,
            )
        elif integ_type in {"langevin"}:
            integrator = mm.LangevinIntegrator(
                temperature,
                friction,
                timestep,
            )
        elif integ_type in {"verlet"}:
            integrator = mm.VerletIntegrator(timestep)
        else:
            raise ValueError(f"Unrecognized integrator type '{self.cfg.integrator.type}'")

        if self.cfg.integrator.seed is not None and hasattr(
            integrator, "setRandomNumberSeed"
        ):
            integrator.setRandomNumberSeed(int(self.cfg.integrator.seed))

        return integrator

    @staticmethod
    def _map_constraints(value: Optional[str]) -> Optional[Any]:
        """
        Map a string constraint spec to an OpenMM constraint enum.

        Args:
            value: String constraint spec from config.
        """
        if value is None:
            return None
        v = str(value).lower()
        if v in {"h-bonds", "hbonds"}:
            return app.HBonds
        if v in {"all-bonds", "allbonds"}:
            return app.AllBonds
        if v in {"h-angles", "hangles"}:
            return app.HAngles
        logger.warning("Unrecognized constraints value '%s'; using None.", value)
        return None

    # ------------------------------------------------------------------
    # Protocol pieces
    # ------------------------------------------------------------------

    @staticmethod
    def _minimize(
        simulation: app.Simulation,
        max_iterations: int = 0,
        tolerance_kj_per_mol_nm: float = 10.0,
    ) -> None:
        """
        Perform energy minimization.

        Args:
            simulation: OpenMM Simulation object.
            max_iterations: Maximum number of minimization iterations (0 → until convergence).
            tolerance_kj_per_mol_nm: Force tolerance (kJ/mol/nm).
        """
        tolerance = tolerance_kj_per_mol_nm * unit.kilojoules_per_mole / unit.nanometer
        simulation.minimizeEnergy(tolerance=tolerance, maxIterations=max_iterations)

    @staticmethod
    def _equilibrate_nvt(
        simulation: app.Simulation,
        num_steps: int,
        *,
        start_temp_k: float,
        target_temp_k: float,
        ramp_chunk_steps: int,
    ) -> None:
        """
        NVT equilibration with a temperature ramp from start_temp_k to target_temp_k.

        Args:
            simulation: OpenMM Simulation object.
            num_steps: Total MD steps to run.
            start_temp_k: Starting temperature for velocities/thermostat.
            target_temp_k: Final temperature.
            ramp_chunk_steps: MD steps per temperature increment (<=0 disables ramp).
        """
        # If ramp disabled or no steps, run at target temperature
        if num_steps <= 0 or ramp_chunk_steps <= 0:
            simulation.integrator.setTemperature(target_temp_k * unit.kelvin)
            simulation.context.setVelocitiesToTemperature(target_temp_k * unit.kelvin)
            simulation.step(max(num_steps, 0))
            return

        chunk = max(1, ramp_chunk_steps)
        n_chunks = max(1, (num_steps + chunk - 1) // chunk)
        temp = float(start_temp_k)
        delta = (target_temp_k - start_temp_k) / n_chunks

        simulation.integrator.setTemperature(temp * unit.kelvin)
        simulation.context.setVelocitiesToTemperature(temp * unit.kelvin)

        steps_remaining = num_steps
        for i in range(n_chunks):
            step_size = min(chunk, steps_remaining)
            if step_size <= 0:
                break
            simulation.step(step_size)
            steps_remaining -= step_size

            temp = temp + delta
            # Clamp to target at the end
            if i == n_chunks - 1 or steps_remaining <= 0:
                temp = target_temp_k
            simulation.integrator.setTemperature(temp * unit.kelvin)

    @staticmethod
    def _equilibrate_npt(
        simulation: app.Simulation,
        num_steps: int,
        *,
        temperature_k: float,
    ) -> None:
        """
        NPT equilibration using a MonteCarloBarostat.

        If a barostat is already present, we just run. If not, we temporarily
        add one for equilibration and then remove it.

        Args:
            simulation: OpenMM Simulation object.
            num_steps: Number of MD steps to run.
            temperature_k: Target temperature for the barostat.
        """
        system = simulation.system
        # Check if a barostat already exists
        has_barostat = any(
            isinstance(system.getForce(i), mm.MonteCarloBarostat)
            for i in range(system.getNumForces())
        )

        barostat_index: Optional[int] = None
        if not has_barostat:
            logger.info("Adding temporary barostat for NPT equilibration.")
            barostat = mm.MonteCarloBarostat(
                1.0 * unit.atmosphere,
                temperature_k * unit.kelvin,
            )
            barostat_index = system.addForce(barostat)
            simulation.context.reinitialize(True)

        simulation.step(num_steps)

        if barostat_index is not None:
            logger.info("Removing temporary barostat after NPT equilibration.")
            system.removeForce(barostat_index)
            simulation.context.reinitialize(True)

    @staticmethod
    def _run_production(simulation: app.Simulation, num_steps: int) -> None:
        """
        Run production MD simulation.

        Args:
            simulation: OpenMM Simulation object.
            num_steps: Number of MD steps to run.
        """
        simulation.step(num_steps)

    @staticmethod
    def _write_pdb(simulation: app.Simulation, path: Path) -> None:
        positions = simulation.context.getState(
            getPositions=True,
            enforcePeriodicBox=True,
        ).getPositions()
        with path.open("w") as fh:
            app.PDBFile.writeFile(simulation.topology, positions, fh)
