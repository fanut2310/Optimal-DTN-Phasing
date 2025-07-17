from __future__ import annotations
import logging, os, time
from pathlib import Path
import pandas as pd
import cea.config, cea.inputlocator
from cea.optimization_new.dynamic_dtn_optimization import ModifiedDemandsLocator
from cea.optimization_new.DTN_expansion_optimization import DTNExpansionOptimizer

# ──────────────────────────────────────────────────────────────────────────────
# utility logger
# ──────────────────────────────────────────────────────────────────────────────
def log() -> logging.Logger:
    lg = logging.getLogger("cea.dynamic_dtn_rerun")
    if not lg.handlers:
        fmt = "%(asctime)s | %(levelname)5s | %(message)s"
        logging.basicConfig(level=logging.INFO, format=fmt, datefmt="%H:%M:%S")
    return lg

# ──────────────────────────────────────────────────────────────────────────────
#  subclassed optimizer  (redirects all outputs + integer-safe GA)
# ──────────────────────────────────────────────────────────────────────────────
class DynamicDTNExpansionOptimizer(DTNExpansionOptimizer):
    def __init__(self, *args, dynamic_dtn_folder: str, **kwargs):
        self.logger = log()
        self._dynamic_folder = Path(dynamic_dtn_folder)
        super().__init__(*args, **kwargs)

    # --- output routing ------------------------------------------------------
    def _results_folder(self) -> Path:
        p = self._dynamic_folder / "dtn_optimization_results"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def save_all_evaluated_individuals(self, inds, _):
        return super().save_all_evaluated_individuals(inds, self._results_folder())

    def save_detailed_results(self, rows, _, return_df=False):
        if return_df:
            return super().save_detailed_results(rows, _, True)
        return super().save_detailed_results(rows, self._results_folder())

    def save_results(self, solution):
        # patch locator so parent writes to the dynamic folder
        orig = self.locator.get_dtn_expansion_optimization_results_folder
        self.locator.get_dtn_expansion_optimization_results_folder = \
            lambda: str(self._results_folder())
        try:
            return super().save_results(solution)
        finally:
            self.locator.get_dtn_expansion_optimization_results_folder = orig

    # --- phase-supply & LCA files into dynamic folder ------------------------
    def _patch_locator(self):
        phase_dir = self._results_folder() / "phase_supply_files"
        phase_dir.mkdir(exist_ok=True)
        lca_dir   = self._results_folder() / "lca_operation"
        lca_dir.mkdir(exist_ok=True)

        loc = self.locator
        self._orig_get_res   = loc.get_dtn_expansion_optimization_results_folder
        self._orig_get_lca   = loc.get_lca_operation
        loc.get_dtn_expansion_optimization_results_folder = lambda: str(self._results_folder())
        loc.get_lca_operation = lambda: str(lca_dir / "Total_LCA_operation.csv")

    def _unpatch_locator(self):
        loc = self.locator
        loc.get_dtn_expansion_optimization_results_folder = self._orig_get_res
        loc.get_lca_operation = self._orig_get_lca

    def calculate_district_emissions_new(self):
        self._patch_locator()
        try:
            return super().calculate_district_emissions_new()
        finally:
            self._unpatch_locator()

    def calculate_emissions_for_genome(self, *args, **kw):
        self._patch_locator()
        try:
            return super().calculate_emissions_for_genome(*args, **kw)
        finally:
            self._unpatch_locator()

    # --- GA: force integer genomes & safe mutation ---------------------------
    def _setup_genetic_algorithm(self):
        super()._setup_genetic_algorithm()

        def mutate_int(ind, indpb):
            from random import random, randint
            for i in range(len(ind)):
                if random() < indpb:
                    ind[i] = int(randint(1, self.num_phases))
            return ind,

        self.toolbox.unregister("mutate")
        self.toolbox.register("mutate", mutate_int, indpb=0.15)

    def _evaluate_individual(self, ind):
        # coerce to int to avoid "float interpreted as int" crash
        ind[:] = [int(round(g)) for g in ind]

        # Set the current individual for emissions calculation
        self.current_individual = ind

        try:
            # Call the parent method
            fitness = super()._evaluate_individual(ind)

            # Ensure fitness is returned as a tuple for DEAP
            if not isinstance(fitness, tuple):
                fitness = (fitness,)

            # For single-objective mode, ensure only one fitness value
            if not self.multi_objective_mode and len(fitness) > 1:
                # Use only the first objective (NPV)
                fitness = (fitness[0],)

            return fitness

        except Exception as e:
            self.logger.error(f"Error evaluating individual {ind}: {e}")
            # Return a penalty fitness value
            return (-1000000.0,)
        finally:
            # Clear the current individual
            self.current_individual = None

# ──────────────────────────────────────────────────────────────────────────────
#  main rerun driver
# ──────────────────────────────────────────────────────────────────────────────
class DynamicDTNRerun:
    """
    Dynamic DTN Rerun Optimization class.

    This class handles the rerunning of DTN optimization with modified demands.
    It generates updated metrics based on the modified demands to ensure
    accurate optimization results that reflect the changes in demand profiles.

    Important:
    ----------
    The 'testing_clusters' parameter is critical for accurate emissions calculation.
    If not specified, emissions calculation will include ALL buildings in the scenario,
    resulting in much higher emissions values than expected. This can also lead to
    the optimization not finding valid solutions that meet all constraints.

    Always specify the same testing_clusters that were used in the initial DTN optimization:
    Example: cea dynamic-dtn-rerun-optimization --scenario YOUR_SCENARIO --network-type DH --testing-clusters 1,2,3,4,7
    """
    def __init__(self, loc: cea.inputlocator.InputLocator, cfg: cea.config.Configuration, use_updated_metrics=True):
        """
        Initialize the DynamicDTNRerun class.

        Parameters:
        -----------
        loc : cea.inputlocator.InputLocator
            CEA InputLocator object
        cfg : cea.config.Configuration
            CEA Configuration object
        use_updated_metrics : bool, optional
            Whether to generate updated metrics based on modified demands (True)
            or use the original metrics from the initial DTN optimization (False).
            Default is True.
        """
        self.locator, self.cfg, self.lg = loc, cfg, log()
        self.ntype = cfg.dynamic_dtn_optimization.network_type
        self.dyn_folder = Path(loc.get_optimization_results_folder()) / "dynamic_dtn_optimization"
        self.use_updated_metrics = use_updated_metrics

        # Get testing clusters from config if available
        self.testing_clusters = None
        if hasattr(cfg.dtn_expansion_optimization, 'testing_clusters') and cfg.dtn_expansion_optimization.testing_clusters:
            # Convert string to list of integers if needed
            if isinstance(cfg.dtn_expansion_optimization.testing_clusters, str):
                self.testing_clusters = [int(c.strip()) for c in cfg.dtn_expansion_optimization.testing_clusters.split(',') if c.strip()]
            else:
                self.testing_clusters = cfg.dtn_expansion_optimization.testing_clusters

    # ---------- helpers ------------------------------------------------------
    def _original_metrics(self) -> pd.DataFrame:
        p = Path(self.locator.get_dtn_expansion_optimization_results_folder()) \
            / "phase_1" / "clusters_metrics.csv"
        if not p.exists():
            raise FileNotFoundError(f"metrics file not found: {p}")
        self.lg.info(f"Loading original metrics from: {p}")
        return pd.read_csv(p)

    def _generate_updated_metrics(self) -> pd.DataFrame:
        """
        Generate updated metrics based on modified demands.
        
        This method uses the PipeLayoutGenerator.generate_pipe_layouts() method directly
        to efficiently calculate metrics for all cluster combinations at once, avoiding
        the memory issues that can lead to stack overflow.
        
        Returns:
            pd.DataFrame: DataFrame with updated metrics for all cluster combinations
        """
        self.lg.info("Generating updated metrics with modified demands...")

        try:
            # Create a modified demands locator that points to the modified demand files
            mod_files = self._modified_demands()
            mod_loc = ModifiedDemandsLocator(self.locator, mod_files)
            
            # Import the PipeLayoutGenerator class
            from cea.optimization_new.DTN_expansion_optimization import PipeLayoutGenerator
            
            # Create a custom output folder for the PipeLayoutGenerator
            # This ensures the results are saved in the dynamic_dtn_optimization folder
            output_folder = self.dyn_folder / "updated_metrics"
            output_folder.mkdir(parents=True, exist_ok=True)
            
            # Create a PipeLayoutGenerator with the modified demands locator
            self.lg.info("Creating PipeLayoutGenerator with modified demands...")
            generator = PipeLayoutGenerator(
                locator=mod_loc,
                network_type=self.ntype,
                phase=1,
                testing_clusters=self.testing_clusters
            )
            
            # Override the output folder to save to our custom location
            generator.output_folder = output_folder
            
            # Use the generator's built-in method to calculate all metrics at once
            # This is the same method used in the original building clustering
            self.lg.info("Calculating metrics for all cluster combinations...")
            self.lg.info("This may take some time but is much more memory-efficient than the previous approach")
            updated_metrics = generator.generate_pipe_layouts()
            
            # The metrics are already saved to the output folder by generate_pipe_layouts()
            # Rename the file to make it clear these are updated metrics
            metrics_file = output_folder / "clusters_metrics.csv"
            updated_file = output_folder / "clusters_metrics_updated.csv"
            
            if metrics_file.exists():
                if updated_file.exists():
                    updated_file.unlink()  # Remove existing file if it exists
                metrics_file.rename(updated_file)
                self.lg.info(f"Renamed metrics file to {updated_file}")
            
            self.lg.info(f"Successfully generated updated metrics for {len(updated_metrics)} cluster combinations")
            return updated_metrics
            
        except Exception as e:
            self.lg.error(f"Error generating updated metrics: {str(e)}")
            self.lg.error("Falling back to original metrics...")
            
            # As a last resort, use the original metrics
            try:
                original_metrics = self._original_metrics()
                self.lg.warning("Using original metrics due to error in generating updated metrics")
                
                # Save a copy of the original metrics for reference
                output_dir = self.dyn_folder / "updated_metrics"
                output_dir.mkdir(parents=True, exist_ok=True)
                output_file = output_dir / "clusters_metrics_original.csv"
                original_metrics.to_csv(output_file, index=False)
                self.lg.info(f"Saved original metrics to {output_file}")
                
                return original_metrics
            except Exception as e2:
                self.lg.error(f"Error loading original metrics: {str(e2)}")
                raise RuntimeError(f"Failed to generate or load metrics: {str(e)} / {str(e2)}")

    def _modified_demands(self) -> dict[str, dict]:
        """
        Get a dictionary of all building demand files, with modified versions where available.

        This method creates a dictionary that maps building names to their demand files,
        using modified versions where available and original versions otherwise.

        Returns:
            dict: Dictionary mapping building names to their demand files
                  {building_name: {'original': original_file_path, 'modified': modified_file_path}}
        """
        mod_dir = self.dyn_folder / "modified_demands"
        if not mod_dir.exists():
            raise FileNotFoundError(f"Modified demand directory not found: {mod_dir}")

        # Get all buildings in the scenario
        total_demand = pd.read_csv(self.locator.get_total_demand())
        all_buildings = total_demand['name'].values

        # Start with modified buildings
        files = {f.stem: {'original': self.locator.get_demand_results_file(f.stem),
                          'modified': str(f)}
                 for f in mod_dir.glob("*.csv")}

        # Check for original demand files in the original_demands directory
        orig_dir = self.dyn_folder / "original_demands"
        if orig_dir.exists():
            self.lg.info(f"Found original demands directory: {orig_dir}")

            # Add all other buildings with original files from the original_demands directory
            modified_buildings = set(files.keys())
            for building in all_buildings:
                if building not in modified_buildings:
                    orig_file = orig_dir / f"{building}.csv"
                    if orig_file.exists():
                        files[building] = {'original': str(orig_file), 'modified': str(orig_file)}
                    else:
                        # Fall back to the standard location
                        original_file = self.locator.get_demand_results_file(building)
                        files[building] = {'original': original_file, 'modified': original_file}
        else:
            self.lg.warning(f"Original demands directory not found: {orig_dir}")
            self.lg.warning("Will use standard demand files for unmodified buildings")

            # Add all other buildings with original files from the standard location
            modified_buildings = set(files.keys())
            for building in all_buildings:
                if building not in modified_buildings:
                    original_file = self.locator.get_demand_results_file(building)
                    files[building] = {'original': original_file, 'modified': original_file}

        if not files:
            raise FileNotFoundError("No demand files found.")

        self.lg.info(f"Found {len(files)} demand files ({len(mod_dir.glob('*.csv'))} modified)")
        return files

    # ---------- core ---------------------------------------------------------
    def run(self):
        self.lg.info("=== Rerunning DTN optimisation with modified demands ===")
        start_time = time.time()

        # Check if testing_clusters is specified and warn if not
        if not self.testing_clusters:
            self.lg.warning("No testing_clusters specified! This will cause emissions calculation to include ALL buildings.")
            self.lg.warning("This may result in much higher emissions values and optimization may not find valid solutions.")
            self.lg.warning("Specify testing_clusters parameter with the same clusters used in the initial DTN optimization.")
            self.lg.warning("Example: --testing-clusters 1,2,3,4,7")
        else:
            self.lg.info(f"Using testing_clusters: {self.testing_clusters}")

        try:
            mod_files = self._modified_demands()
            new_cfg   = cea.config.Configuration()
            new_cfg.__setstate__(self.cfg.__getstate__())

            # single-objective → make list with one objective
            if not new_cfg.dtn_expansion_optimization.multi_objective_mode:
                new_cfg.dtn_expansion_optimization.multi_objective_functions = \
                    [new_cfg.dtn_expansion_optimization.objective_function]
                self.lg.info(f"Single-objective mode: Setting multi_objective_functions to [{new_cfg.dtn_expansion_optimization.objective_function}]")

            mod_loc = ModifiedDemandsLocator(self.locator, mod_files)

            # Load thermal network results
            tn_results_dir = self.dyn_folder / "thermal_network"
            if not tn_results_dir.exists():
                self.lg.warning(f"Thermal network results directory not found: {tn_results_dir}")
            else:
                self.lg.info(f"Found thermal network results in {tn_results_dir}")

            # Determine which metrics to use based on the use_updated_metrics parameter
            if self.use_updated_metrics:
                self.lg.info("Using updated metrics based on modified demands for more accurate optimization")
                self.lg.info("This ensures that the optimization reflects the changes in demand profiles")
                metrics_df = self._generate_updated_metrics()
            else:
                self.lg.info("Using original metrics from the initial DTN optimization")
                self.lg.info("Note: This may lead to suboptimal results if demands have changed significantly")
                metrics_df = self._original_metrics()

            optimizer = DynamicDTNExpansionOptimizer(
                locator               = mod_loc,
                network_type          = self.ntype,
                metrics_df            = metrics_df,
                dynamic_dtn_folder    = str(self.dyn_folder),
                num_phases            = new_cfg.dtn_expansion_optimization.num_phases,
                phase_durations       = self._parse(new_cfg.dtn_expansion_optimization.phase_durations, int),
                capex_budget_per_phase= self._parse(new_cfg.dtn_expansion_optimization.capex_budget_per_phase, float),
                total_expenditure_budget_per_phase = self._parse(new_cfg.dtn_expansion_optimization.total_expenditure_budget_per_phase, float),
                interest_rate         = new_cfg.dtn_expansion_optimization.interest_rate,
                cost_model            = new_cfg.dtn_expansion_optimization.cost_model,
                objective_function    = new_cfg.dtn_expansion_optimization.objective_function,
                diversity_factor      = new_cfg.dtn_expansion_optimization.diversity_factor,
                temperature_difference_dh = new_cfg.dtn_expansion_optimization.temperature_difference_dh,
                temperature_difference_dc = new_cfg.dtn_expansion_optimization.temperature_difference_dc,
                pressure_loss_pa_per_m = new_cfg.dtn_expansion_optimization.pressure_loss_pa_per_m,
                pump_operation_hours  = new_cfg.dtn_expansion_optimization.pump_operation_hours,
                pump_efficiency       = new_cfg.dtn_expansion_optimization.pump_efficiency,
                pump_load_factor      = new_cfg.dtn_expansion_optimization.pump_load_factor,
                pump_capex_a          = new_cfg.dtn_expansion_optimization.pump_capex_a,
                pump_capex_b          = new_cfg.dtn_expansion_optimization.pump_capex_b,
                cooling_cop           = new_cfg.dtn_expansion_optimization.cooling_cop,
                ghg_budget_per_phase  = self._parse(new_cfg.dtn_expansion_optimization.ghg_budget_per_phase, float),
                multi_objective_mode  = new_cfg.dtn_expansion_optimization.multi_objective_mode,
                multi_objective_functions = new_cfg.dtn_expansion_optimization.multi_objective_functions,
                testing_clusters      = self.testing_clusters
            )

            pop = new_cfg.dtn_expansion_optimization.population_size
            gen = new_cfg.dtn_expansion_optimization.num_generations
            self.lg.info(f"GA population {pop}, generations {gen}")
            results = optimizer.optimize(population_size=pop, num_generations=gen)

            # Check if optimization was successful
            if isinstance(results, dict) and 'genome' in results:
                self.lg.info(f"Optimization completed successfully with genome: {results['genome']}")
            elif isinstance(results, list) and len(results) > 0:
                # Handle case where results are returned as a list of individuals
                self.lg.info(f"Optimization completed successfully with {len(results)} valid solutions")
            elif hasattr(results, 'items') and any(
                    k for k in results.keys() if 'individual' in str(k).lower() or 'fitness' in str(k).lower()):
                # Handle case where results are in a different dictionary format
                self.lg.info(f"Optimization completed successfully with valid results")
            else:
                self.lg.warning("Optimization completed, but no valid genome found in results")

        except Exception as e:
            self.lg.error(f"Error during dynamic DTN rerun: {str(e)}")
            return {'error': str(e), 'status': 'failed'}
        finally:
            # Calculate elapsed time
            elapsed_time = time.time() - start_time
            hours, remainder = divmod(elapsed_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            time_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
            self.lg.info(f"Total runtime: {time_str}")

    # simple csv-list parser
    @staticmethod
    def _parse(s: str | None, typ=float):
        if not s or str(s).strip()=='':
            return None
        return [typ(x) for x in str(s).split(',')]

# ──────────────────────────────────────────────────────────────────────────────
def main(cfg: cea.config.Configuration):
    """
    Main entry point for the Dynamic DTN Rerun Optimization module.

    This function runs the Dynamic DTN Rerun Optimization with the specified configuration.
    It reoptimizes the DTN expansion with modified demands, generating updated metrics
    based on the modified demands to ensure accurate optimization results.

    Parameters:
    -----------
    cfg : cea.config.Configuration
        The configuration object containing all parameters for the optimization.

    Notes:
    ------
    By default, this function generates updated metrics based on the modified demands.
    To use the original metrics from the initial DTN optimization instead, set the
    'use_original_metrics' parameter to True in the command line:

    cea dynamic-dtn-rerun-optimization --scenario YOUR_SCENARIO --network-type DH --use-original-metrics

    Important:
    ----------
    The 'testing_clusters' parameter is critical for accurate emissions calculation.
    If not specified, emissions calculation will include ALL buildings in the scenario,
    resulting in much higher emissions values than expected. This can also lead to
    the optimization not finding valid solutions that meet all constraints.

    Always specify the same testing_clusters that were used in the initial DTN optimization:
    Example: cea dynamic-dtn-rerun-optimization --scenario YOUR_SCENARIO --network-type DH --testing-clusters 1,2,3,4,7

    Returns:
    --------
    dict
        The optimization results, including the optimal genome if found.
    """
    logger = log()
    logger.info("="*80)
    logger.info("Starting Dynamic DTN Rerun Optimization module")
    logger.info(f"Scenario: {cfg.scenario}")
    logger.info(f"Network type: {cfg.dynamic_dtn_optimization.network_type}")

    # Check if testing_clusters is specified in the configuration
    testing_clusters = None
    if hasattr(cfg.dtn_expansion_optimization, 'testing_clusters') and cfg.dtn_expansion_optimization.testing_clusters:
        if isinstance(cfg.dtn_expansion_optimization.testing_clusters, str):
            testing_clusters = [int(c.strip()) for c in cfg.dtn_expansion_optimization.testing_clusters.split(',') if c.strip()]
        else:
            testing_clusters = cfg.dtn_expansion_optimization.testing_clusters
        logger.info(f"Testing clusters specified: {testing_clusters}")
    else:
        logger.warning("="*80)
        logger.warning("WARNING: No testing_clusters specified in configuration!")
        logger.warning("This will cause emissions calculation to include ALL buildings in the scenario.")
        logger.warning("For accurate results, specify the same testing_clusters used in the initial DTN optimization.")
        logger.warning("Example: cea dynamic-dtn-rerun-optimization --scenario YOUR_SCENARIO --network-type DH --testing-clusters 1,2,3,4,7")
        logger.warning("="*80)

    logger.info("="*80)

    # Check if use_original_metrics is specified in the configuration
    use_original_metrics = False
    if hasattr(cfg.dynamic_dtn_optimization, 'use_original_metrics'):
        use_original_metrics = cfg.dynamic_dtn_optimization.use_original_metrics
        if use_original_metrics:
            logger.info("Using original metrics from the initial DTN optimization as specified in configuration")

    t0 = time.time()
    rerun = DynamicDTNRerun(
        cea.inputlocator.InputLocator(cfg.scenario), 
        cfg,
        use_updated_metrics=not use_original_metrics
    )
    results = rerun.run()

    # Calculate elapsed time
    elapsed_time = time.time() - t0
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    time_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"

    logger.info("="*80)

    # Check if optimization was successful
    if isinstance(results, dict) and 'genome' in results:
        logger.info(f"Dynamic DTN Rerun Optimization completed successfully in {time_str}")
        print("Dynamic DTN Rerun Optimization completed successfully.")
    elif isinstance(results, dict) and 'error' in results:
        logger.error(f"Dynamic DTN Rerun Optimization completed with errors in {time_str}")
        logger.error(f"Error: {results['error']}")
        print("Dynamic DTN Rerun Optimization completed with errors. See log for details.")
    else:
        logger.warning(f"Dynamic DTN Rerun Optimization completed with potential issues in {time_str}")
        print("Dynamic DTN Rerun Optimization completed with potential issues. See log for details.")

    logger.info(f"Results saved to: {Path(cea.inputlocator.InputLocator(cfg.scenario).get_optimization_results_folder()) / 'dynamic_dtn_optimization'}")
    logger.info("="*80)

    print(f"Total runtime: {time_str}")

    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Dynamic DTN Rerun Optimization")
    parser.add_argument('-s', '--scenario', help='Path to the scenario folder')
    parser.add_argument('-c', '--config', help='Path to the config file')
    parser.add_argument('--use-original-metrics', action='store_true', 
                        help='Use original metrics from the initial DTN optimization instead of generating updated metrics')
    parser.add_argument('--network-type', choices=['DH', 'DC'], 
                        help='Network type: DH (district heating) or DC (district cooling)')
    parser.add_argument('--testing-clusters', 
                        help='Comma-separated list of cluster IDs to include in the optimization')
    args = parser.parse_args()

    config = cea.config.Configuration(args.config)
    config.scenario = args.scenario

    # Set command line parameters in the configuration
    if args.network_type:
        config.dynamic_dtn_optimization.network_type = args.network_type

    if args.testing_clusters:
        config.dtn_expansion_optimization.testing_clusters = args.testing_clusters

    # Add use_original_metrics to the configuration
    if not hasattr(config.dynamic_dtn_optimization, 'use_original_metrics'):
        setattr(config.dynamic_dtn_optimization, 'use_original_metrics', False)

    if args.use_original_metrics:
        config.dynamic_dtn_optimization.use_original_metrics = True

    main(config)
