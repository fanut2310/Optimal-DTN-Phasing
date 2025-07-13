from __future__ import annotations

import os
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import time

import cea.config
import cea.inputlocator
from cea.optimization_new.DTN_expansion_optimization import DTNExpansionOptimizer
from cea.optimization_new.dynamic_dtn_optimization import ModifiedDemandsLocator


class DynamicDTNExpansionOptimizer(DTNExpansionOptimizer):
    """
    A subclass of DTNExpansionOptimizer that saves results to the dynamic DTN optimization folder
    instead of the original DTN expansion optimization folder.
    """

    def __init__(self, *args, **kwargs):
        # Extract dynamic_dtn_folder from kwargs if provided
        self.dynamic_dtn_folder = kwargs.pop('dynamic_dtn_folder', None)

        # Call the parent class constructor
        super().__init__(*args, **kwargs)

    def get_results_folder(self):
        """
        Return the folder where results should be saved.

        Returns:
            Path: Path to the results folder
        """
        if self.dynamic_dtn_folder:
            # Create the folder if it doesn't exist
            results_folder = Path(self.dynamic_dtn_folder) / "dtn_optimization_results"
            results_folder.mkdir(parents=True, exist_ok=True)
            return results_folder
        else:
            # Fall back to the original behavior
            return Path(self.locator.get_dtn_expansion_optimization_results_folder())

    def save_all_evaluated_individuals(self, all_individuals, output_dir):
        """Override to save to dynamic DTN folder"""
        # Ignore the provided output_dir and use our custom folder
        return super().save_all_evaluated_individuals(all_individuals, self.get_results_folder())

    def save_detailed_results(self, results, output_dir, return_df=False):
        """Override to save to dynamic DTN folder"""
        if return_df:
            # If we're just returning the DataFrame, use the original method
            return super().save_detailed_results(results, output_dir, return_df=True)
        else:
            # Otherwise, save to our custom folder
            return super().save_detailed_results(results, self.get_results_folder(), return_df=False)

    def save_results(self, solution):
        """Override to save to dynamic DTN folder"""
        # Create output directory
        output_dir = self.get_results_folder()
        output_dir.mkdir(parents=True, exist_ok=True)

        # Calculate district emissions using the new methodology
        district_emissions = self.calculate_district_emissions_new()

        # Rest of the method is the same as the original, but using our output_dir
        # For brevity, we'll call the parent method but modify the output_dir first
        # This is a bit of a hack, but it works
        original_get_dtn_expansion_optimization_results_folder = self.locator.get_dtn_expansion_optimization_results_folder
        self.locator.get_dtn_expansion_optimization_results_folder = lambda: str(output_dir)

        try:
            result = super().save_results(solution)
        finally:
            # Restore the original method
            self.locator.get_dtn_expansion_optimization_results_folder = original_get_dtn_expansion_optimization_results_folder

        return result

    def calculate_district_emissions_new(self):
        """Override to save phase supply files to dynamic DTN folder"""
        # Create a custom directory for phase-specific supply files
        phase_files_dir = self.get_results_folder() / "phase_supply_files"
        phase_files_dir.mkdir(parents=True, exist_ok=True)

        # We need to patch multiple locator methods to ensure all files are created in our custom directory
        original_get_dtn_expansion_optimization_results_folder = self.locator.get_dtn_expansion_optimization_results_folder
        original_get_lca_operation = self.locator.get_lca_operation

        # Create a custom get_lca_operation method that returns a path in our custom directory
        def custom_get_lca_operation():
            lca_dir = Path(self.dynamic_dtn_folder) / "lca_operation"
            lca_dir.mkdir(parents=True, exist_ok=True)
            return str(lca_dir / "Total_LCA_operation.csv")

        # Patch the locator methods
        self.locator.get_dtn_expansion_optimization_results_folder = lambda: str(phase_files_dir.parent)
        self.locator.get_lca_operation = custom_get_lca_operation

        try:
            # Run the parent method with our patched locator
            result = super().calculate_district_emissions_new()
        finally:
            # Restore the original methods
            self.locator.get_dtn_expansion_optimization_results_folder = original_get_dtn_expansion_optimization_results_folder
            self.locator.get_lca_operation = original_get_lca_operation

        return result

    def calculate_emissions_for_genome(self, cluster_phase_map):
        """Override to save phase supply files to dynamic DTN folder"""
        # Create a custom directory for phase-specific supply files
        phase_files_dir = self.get_results_folder() / "phase_supply_files"
        phase_files_dir.mkdir(parents=True, exist_ok=True)

        # We need to patch multiple locator methods to ensure all files are created in our custom directory
        original_get_dtn_expansion_optimization_results_folder = self.locator.get_dtn_expansion_optimization_results_folder
        original_get_lca_operation = self.locator.get_lca_operation

        # Create a custom get_lca_operation method that returns a path in our custom directory
        def custom_get_lca_operation():
            lca_dir = Path(self.dynamic_dtn_folder) / "lca_operation"
            lca_dir.mkdir(parents=True, exist_ok=True)
            return str(lca_dir / "Total_LCA_operation.csv")

        # Patch the locator methods
        self.locator.get_dtn_expansion_optimization_results_folder = lambda: str(phase_files_dir.parent)
        self.locator.get_lca_operation = custom_get_lca_operation

        try:
            # Run the parent method with our patched locator
            result = super().calculate_emissions_for_genome(cluster_phase_map)
        finally:
            # Restore the original methods
            self.locator.get_dtn_expansion_optimization_results_folder = original_get_dtn_expansion_optimization_results_folder
            self.locator.get_lca_operation = original_get_lca_operation

        return result

    def optimize(self, population_size=50, num_generations=30):
        """Override to add constraint violation tracking"""
        # Initialize counters for constraint violations
        self.capex_violations = 0
        self.expenditure_violations = 0
        self.ghg_violations = 0

        # Call the parent method
        result = super().optimize(population_size, num_generations)

        # Log summary of constraint violations
        log().info("=== Constraint Violation Summary ===")
        log().info(f"CAPEX budget violations: {self.capex_violations}")
        log().info(f"Total expenditure budget violations: {self.expenditure_violations}")
        log().info(f"GHG budget violations: {self.ghg_violations}")
        log().info("=================================")

        return result

    def _evaluate_individual(self, individual):
        """Override to track constraint violations"""
        # Call the parent method
        fitness = super()._evaluate_individual(individual)

        # Check if this individual violated any constraints
        if isinstance(fitness, tuple):
            # Multi-objective mode
            if fitness[0] == -1000 or fitness[1] == -1000000:
                # This individual violated a constraint
                self._track_constraint_violation(individual)
        else:
            # Single-objective mode
            if fitness == -1000 or fitness == -1000000:
                # This individual violated a constraint
                self._track_constraint_violation(individual)

        return fitness

    def _track_constraint_violation(self, individual):
        """Track which constraint was violated"""
        # Convert individual to cluster-phase mapping
        cluster_phase_map = {cluster: phase for cluster, phase in zip(self.all_clusters, individual)}

        # Calculate CAPEX for each phase
        phase_capex = [0] * self.num_phases
        phase_total_expenditure = [0] * self.num_phases

        # Group clusters by phase
        clusters_by_phase = {}
        for cluster, phase in cluster_phase_map.items():
            if phase > 0:  # Skip unconnected clusters (phase 0)
                if phase not in clusters_by_phase:
                    clusters_by_phase[phase] = []
                clusters_by_phase[phase].append(cluster)

        # Calculate costs for each phase
        for phase, clusters in clusters_by_phase.items():
            # Calculate CAPEX for this phase
            capex, _ = self._calculate_phase_capex(clusters, phase)
            phase_capex[phase-1] = capex

            # Calculate total expenditure for this phase
            total_expenditure = self._calculate_phase_total_expenditure(clusters, phase)
            phase_total_expenditure[phase-1] = total_expenditure

        # Check CAPEX budget constraints
        for phase in range(self.num_phases):
            if phase < len(phase_capex) and self.capex_budget_per_phase and phase_capex[phase] > self.capex_budget_per_phase[phase]:
                self.capex_violations += 1
                break

        # Check total expenditure budget constraints
        for phase in range(self.num_phases):
            if phase < len(phase_total_expenditure) and self.total_expenditure_budget_per_phase and phase_total_expenditure[phase] > self.total_expenditure_budget_per_phase[phase]:
                self.expenditure_violations += 1
                break

        # Check GHG budget constraints
        if self.ghg_budget_per_phase:
            ind_tuple = tuple(individual)
            if ind_tuple in self.emissions_cache:
                phase_emissions, _ = self.emissions_cache[ind_tuple]
                for phase in range(1, self.num_phases + 1):  # Start from 1, not 0
                    if phase <= self.num_phases and phase-1 < len(self.ghg_budget_per_phase) and self.ghg_budget_per_phase[phase-1] > 0:  # Use phase-1 for the budget index
                        if phase in phase_emissions and 'operation' in phase_emissions[phase]:
                            phase_ghg = phase_emissions[phase]['operation']  # No need for +1 here
                            if phase_ghg > self.ghg_budget_per_phase[phase-1]:
                                self.ghg_violations += 1
                                break

class DynamicDTNRerunOptimization:
    """
    Module for rerunning DTN optimization with modified demand files.
    This module implements step 6 of the dynamic DTN optimization process.
    """

    def _log_constraint_parameters(self, optimizer):
        """Log constraint parameters to help diagnose constraint violations"""
        self.logger.info("=== Constraint Parameters ===")

        # Log CAPEX budget constraints
        if optimizer.capex_budget_per_phase:
            self.logger.info(f"CAPEX budget per phase: {optimizer.capex_budget_per_phase}")
        else:
            self.logger.info("No CAPEX budget constraints specified")

        # Log total expenditure budget constraints
        if optimizer.total_expenditure_budget_per_phase:
            self.logger.info(f"Total expenditure budget per phase: {optimizer.total_expenditure_budget_per_phase}")
        else:
            self.logger.info("No total expenditure budget constraints specified")

        # Log GHG budget constraints
        if optimizer.ghg_budget_per_phase:
            self.logger.info(f"GHG budget per phase: {optimizer.ghg_budget_per_phase}")
        else:
            self.logger.info("No GHG budget constraints specified")

        self.logger.info("============================")

    def __init__(self, locator: cea.inputlocator.InputLocator, config: cea.config.Configuration):
        """Initialize the rerun optimization module."""
        self.locator = locator
        self.config = config
        self.network_type = config.dynamic_dtn_optimization.network_type
        self.logger = logging.getLogger(__name__)

        # Initialize variables that will be set later
        self.original_results = None
        self.modified_demand_files = {}
        self.tn_results_dir = None
        self.new_results = None

        # Get testing clusters from config if available
        self.testing_clusters = None
        if hasattr(config.dtn_expansion_optimization, 'testing_clusters') and config.dtn_expansion_optimization.testing_clusters:
            # Convert string to list of integers if needed
            if isinstance(config.dtn_expansion_optimization.testing_clusters, str):
                self.testing_clusters = [int(c.strip()) for c in config.dtn_expansion_optimization.testing_clusters.split(',') if c.strip()]
            else:
                self.testing_clusters = config.dtn_expansion_optimization.testing_clusters

    def load_original_results(self):
        """
        Load the original DTN optimization results.

        Returns:
            dict: Original optimization results
        """
        # Define path to results file
        dtn_results_folder = Path(self.locator.get_dtn_expansion_optimization_results_folder())
        results_file = dtn_results_folder / f"all_evaluated_individuals_{self.network_type}.csv"

        if not results_file.exists():
            self.logger.error(f"Original DTN optimization results file not found: {results_file}")
            raise FileNotFoundError(f"Original DTN optimization results file not found: {results_file}")

        self.logger.info(f"Loading original DTN expansion optimization results from {results_file}")

        # Load the results file
        results_df = pd.read_csv(results_file)

        # Also load the optimization settings file to get testing_clusters if available
        settings_file = dtn_results_folder / "optimization_settings.csv"
        if settings_file.exists():
            try:
                settings_df = pd.read_csv(settings_file)
                if 'testing_clusters' in settings_df.columns and not pd.isna(settings_df['testing_clusters'].iloc[0]):
                    testing_clusters_str = settings_df['testing_clusters'].iloc[0]
                    self.testing_clusters = [int(c.strip()) for c in testing_clusters_str.split(',') if c.strip()]
                    self.logger.info(f"Loaded testing clusters from optimization settings: {self.testing_clusters}")
            except Exception as e:
                self.logger.warning(f"Error loading testing clusters from optimization settings: {e}")

        # Find the best solution (highest fitness for the objective function)
        if 'fitness_NPV' in results_df.columns:
            # For NPV objective, higher is better
            best_row = results_df.loc[results_df['fitness_NPV'].idxmax()]
            self.logger.info(f"Found optimal solution with highest NPV: {best_row['fitness_NPV']}")
        elif 'fitness_ROI' in results_df.columns:
            # For ROI objective, higher is better
            best_row = results_df.loc[results_df['fitness_ROI'].idxmax()]
            self.logger.info(f"Found optimal solution with highest ROI: {best_row['fitness_ROI']}")
        else:
            # If neither NPV nor ROI is found, use the first row
            self.logger.warning("Could not find NPV or ROI fitness values in results. Using first solution.")
            best_row = results_df.iloc[0]

        # Extract the genome (connection sequence)
        genome_str = best_row['genome']
        self.logger.info(f"Extracting optimal genome from solution with ID: {best_row['individual_id']}")

        # Check if genome_str is already a list or needs conversion
        if isinstance(genome_str, list):
            genome = genome_str
        else:
            # Convert string representation to list
            genome = eval(genome_str)  # Convert string representation to list

        self.logger.info(f"Optimal genome extracted: {genome}")

        # Create a solution dictionary similar to what DTNExpansionOptimizer.optimize() would return
        solution = {
            'individual_id': best_row['individual_id'],
            'genome': genome,
        }

        # Add other metrics from the results file
        for col in results_df.columns:
            if col not in ['individual_id', 'genome']:
                solution[col] = best_row[col]

        # Parse phase clusters
        for phase in range(1, self.config.dtn_expansion_optimization.num_phases + 1):
            phase_clusters_col = f'phase_{phase}_clusters'
            if phase_clusters_col in results_df.columns:
                try:
                    solution[phase_clusters_col] = eval(best_row[phase_clusters_col])
                except:
                    self.logger.warning(f"Could not parse {phase_clusters_col} from results")

        self.original_results = solution
        return solution

    def load_modified_demand_files(self):
        """
        Load information about modified demand files.

        Returns:
            dict: Dictionary mapping building names to their modified demand files
        """
        # Define path to modified demand files
        modified_demand_dir = Path(self.locator.get_optimization_results_folder()) / "dynamic_dtn_optimization" / "modified_demands"

        if not modified_demand_dir.exists():
            self.logger.error(f"Modified demand directory not found: {modified_demand_dir}")
            raise FileNotFoundError(f"Modified demand directory not found: {modified_demand_dir}")

        self.logger.info(f"Loading modified demand files from {modified_demand_dir}")

        # Find all modified demand files
        modified_demand_files = {}
        for file_path in modified_demand_dir.glob("*.csv"):
            building_name = file_path.stem
            original_file = self.locator.get_demand_results_file(building_name)

            modified_demand_files[building_name] = {
                'original': original_file,
                'modified': str(file_path)
            }

        self.logger.info(f"Found {len(modified_demand_files)} modified demand files")
        self.modified_demand_files = modified_demand_files

        return modified_demand_files

    def load_thermal_network_results(self):
        """
        Load information about thermal network results.

        Returns:
            Path: Path to the thermal network results directory
        """
        # Define path to thermal network results
        tn_results_dir = Path(self.locator.get_optimization_results_folder()) / "dynamic_dtn_optimization" / "thermal_network"

        if not tn_results_dir.exists():
            self.logger.error(f"Thermal network results directory not found: {tn_results_dir}")
            raise FileNotFoundError(f"Thermal network results directory not found: {tn_results_dir}")

        self.logger.info(f"Loading thermal network results from {tn_results_dir}")
        self.tn_results_dir = tn_results_dir

        return tn_results_dir

    def _parse_list_param(self, param_str):
        """Parse a comma-separated string parameter into a list of values."""
        if not param_str or param_str.strip() == '':
            return None
        return [float(x) for x in param_str.split(',')]

    def _create_config_copy(self):
        """
        Create a new configuration object with the same settings as the existing one.

        Returns:
            A new Configuration object
        """
        # Get the state of the current config object
        config_state = self.config.__getstate__()

        # Create a new config object
        new_config = cea.config.Configuration()

        # Set the state of the new config object to match the current one
        new_config.__setstate__(config_state)

        return new_config

    def rerun_dtn_optimization(self):
        """
        Rerun the DTN optimization with the updated thermal network results.

        Returns:
            dict: New optimization results
        """
        self.logger.info("Rerunning DTN expansion optimization with modified demands")

        # Create a copy of the config for the new optimization
        new_config = self._create_config_copy()
        self.logger.info(f"Created configuration copy for DTN expansion optimization")

        # Create a custom locator that points to the modified demand files
        self.logger.info("Creating custom locator that points to modified demand files")
        modified_locator = ModifiedDemandsLocator(self.locator, self.modified_demand_files)

        # Define the dynamic DTN folder
        dynamic_dtn_folder = Path(self.locator.get_optimization_results_folder()) / "dynamic_dtn_optimization"

        # Run the new DTN optimization with the modified locator and custom output folder
        self.logger.info("Initializing DTN expansion optimizer with modified demands")
        optimizer = DynamicDTNExpansionOptimizer(
            locator=modified_locator,
            network_type=self.network_type,
            metrics_df=None,  # This will be loaded by the optimizer
            num_phases=new_config.dtn_expansion_optimization.num_phases,
            phase_durations=self._parse_list_param(new_config.dtn_expansion_optimization.phase_durations),
            capex_budget_per_phase=self._parse_list_param(new_config.dtn_expansion_optimization.capex_budget_per_phase),
            total_expenditure_budget_per_phase=self._parse_list_param(new_config.dtn_expansion_optimization.total_expenditure_budget_per_phase),
            interest_rate=new_config.dtn_expansion_optimization.interest_rate,
            cost_model=new_config.dtn_expansion_optimization.cost_model,
            objective_function=new_config.dtn_expansion_optimization.objective_function,
            diversity_factor=new_config.dtn_expansion_optimization.diversity_factor,
            temperature_difference_dh=new_config.dtn_expansion_optimization.temperature_difference_dh,
            temperature_difference_dc=new_config.dtn_expansion_optimization.temperature_difference_dc,
            pressure_loss_pa_per_m=new_config.dtn_expansion_optimization.pressure_loss_pa_per_m,
            pump_operation_hours=new_config.dtn_expansion_optimization.pump_operation_hours,
            pump_efficiency=new_config.dtn_expansion_optimization.pump_efficiency,
            pump_load_factor=new_config.dtn_expansion_optimization.pump_load_factor,
            pump_capex_a=new_config.dtn_expansion_optimization.pump_capex_a,
            pump_capex_b=new_config.dtn_expansion_optimization.pump_capex_b,
            cooling_cop=new_config.dtn_expansion_optimization.cooling_cop,
            ghg_budget_per_phase=self._parse_list_param(new_config.dtn_expansion_optimization.ghg_budget_per_phase),
            multi_objective_mode=new_config.dtn_expansion_optimization.multi_objective_mode,
            multi_objective_functions=new_config.dtn_expansion_optimization.multi_objective_functions,
            dynamic_dtn_folder=dynamic_dtn_folder  # Pass the custom folder
        )
        self.logger.info(f"DTN expansion optimizer initialized with {new_config.dtn_expansion_optimization.num_phases} phases")

        # Run the optimization
        population_size = new_config.dtn_expansion_optimization.population_size
        num_generations = new_config.dtn_expansion_optimization.num_generations
        self.logger.info(f"Starting DTN expansion optimization with population size {population_size} and {num_generations} generations")
        new_results = optimizer.optimize(population_size=population_size, num_generations=num_generations)
        self.logger.info("DTN expansion optimization completed successfully")

        # Log constraint parameters to help diagnose issues
        self._log_constraint_parameters(optimizer)

        # Save the new results
        self.new_optimizer = optimizer
        self.new_results = new_results

        # Check if new_results is a dictionary and has a 'genome' key
        if isinstance(new_results, dict) and 'genome' in new_results:
            self.logger.info(f"New optimization results saved with {len(new_results['genome'])} clusters")
        else:
            # Handle case where new_results doesn't have a 'genome' key
            self.logger.warning(f"Optimization completed, but no valid genome found in results. This may be due to constraint violations.")

            # Check for specific constraint violation patterns
            if isinstance(new_results, (int, float)) and new_results == -float('inf'):
                self.logger.error("Optimization failed due to constraint violations. All individuals violated constraints.")
                self.logger.error("This means none of the possible network configurations could meet all the specified constraints.")
                self.logger.error("This is a common issue when:")
                self.logger.error("1. The CAPEX budget is too low for the required network infrastructure")
                self.logger.error("2. The total expenditure budget is too low for the operational costs")
                self.logger.error("3. The GHG budget is too restrictive for the available supply systems")
                self.logger.info("Consider relaxing constraints such as CAPEX budget, total expenditure budget, or GHG budget.")

                # Add more specific guidance
                if optimizer.capex_budget_per_phase:
                    self.logger.info(f"Try increasing the CAPEX budget per phase (current: {optimizer.capex_budget_per_phase})")
                if optimizer.total_expenditure_budget_per_phase:
                    self.logger.info(f"Try increasing the total expenditure budget per phase (current: {optimizer.total_expenditure_budget_per_phase})")
                if optimizer.ghg_budget_per_phase:
                    self.logger.info(f"Try increasing the GHG budget per phase (current: {optimizer.ghg_budget_per_phase})")

                # Add specific guidance based on which constraints were most frequently violated
                if hasattr(optimizer, 'capex_violations') and optimizer.capex_violations > 0:
                    self.logger.error(f"CAPEX budget was violated {optimizer.capex_violations} times. Try increasing the CAPEX budget per phase.")
                if hasattr(optimizer, 'expenditure_violations') and optimizer.expenditure_violations > 0:
                    self.logger.error(f"Total expenditure budget was violated {optimizer.expenditure_violations} times. Try increasing the total expenditure budget per phase.")
                if hasattr(optimizer, 'ghg_violations') and optimizer.ghg_violations > 0:
                    self.logger.error(f"GHG budget was violated {optimizer.ghg_violations} times. Try increasing the GHG budget per phase or using lower-emission supply systems.")
            elif isinstance(new_results, dict):
                # Log available keys to help with debugging
                self.logger.info(f"Available keys in results: {list(new_results.keys())}")

                # Check for specific error messages or indicators in the results
                if 'error' in new_results:
                    self.logger.error(f"Optimization error: {new_results['error']}")
                elif 'constraint_violation' in new_results:
                    self.logger.error(f"Constraint violation: {new_results['constraint_violation']}")
            else:
                self.logger.info(f"Results type: {type(new_results)}")

            self.logger.info("You may need to adjust your constraints (e.g., increase CAPEX budget) or modify the demand reduction parameters.")

        return new_results

    def run(self):
        """
        Run the complete workflow for rerunning DTN optimization.

        Returns:
            dict: New optimization results
        """
        self.logger.info("Starting Dynamic DTN Rerun Optimization workflow")
        self.logger.info(f"Network type: {self.network_type}")

        # Step 1: Load original optimization results
        self.logger.info("STEP 1: Loading original DTN optimization results")
        self.load_original_results()

        # Check if original_results has a 'genome' key
        if isinstance(self.original_results, dict) and 'genome' in self.original_results:
            self.logger.info(f"Original optimization results loaded with {len(self.original_results['genome'])} clusters")
        else:
            self.logger.warning("Original optimization results do not contain a valid genome")
            if isinstance(self.original_results, dict):
                self.logger.info(f"Available keys in original results: {list(self.original_results.keys())}")
            else:
                self.logger.info(f"Original results type: {type(self.original_results)}")

        # Step 2: Load modified demand files
        self.logger.info("STEP 2: Loading modified demand files")
        self.load_modified_demand_files()

        # Step 3: Load thermal network results
        self.logger.info("STEP 3: Loading thermal network results")
        self.load_thermal_network_results()

        # Step 4: Rerun the DTN optimization
        self.logger.info("STEP 4: Rerunning the DTN optimization")
        try:
            new_results = self.rerun_dtn_optimization()

            # Check if optimization was successful
            if isinstance(new_results, dict) and 'genome' in new_results:
                self.logger.info("Dynamic DTN Rerun Optimization workflow completed successfully with valid results")
            else:
                self.logger.warning("Dynamic DTN Rerun Optimization workflow completed, but optimization may have encountered issues")
                self.logger.info("Check the logs above for more details on potential constraint violations or errors")
        except Exception as e:
            self.logger.error(f"Error during DTN optimization: {str(e)}")
            # Return a dictionary with error information
            new_results = {'error': str(e), 'status': 'failed'}

        self.logger.info(f"Results saved to: {Path(self.locator.get_optimization_results_folder()) / 'dynamic_dtn_optimization'}")

        return new_results

def log() -> logging.Logger:
    """Configure and return a logger for the dynamic DTN rerun optimization module."""
    lg = logging.getLogger("cea.dynamic_dtn_rerun_optimization")
    if not lg.handlers:
        # Configure logging to output to console
        logging.basicConfig(level=logging.INFO,
                           format="%(asctime)s | %(levelname)5s | %(message)s",
                           datefmt="%H:%M:%S")
    return lg

def main(config):
    """
    Run the Dynamic DTN Rerun Optimization module.

    Args:
        config: CEA Configuration object
    """
    # Set up logging
    logger = log()
    logger.info("="*80)
    logger.info("Starting Dynamic DTN Rerun Optimization module")
    logger.info(f"Scenario: {config.scenario}")
    logger.info(f"Network type: {config.dynamic_dtn_optimization.network_type}")
    logger.info("="*80)

    start_time = time.time()

    locator = cea.inputlocator.InputLocator(config.scenario)

    # Create and run the Dynamic DTN Rerun Optimizer
    logger.info("Initializing Dynamic DTN Rerun Optimizer")
    optimizer = DynamicDTNRerunOptimization(locator, config)

    # Run the optimization
    new_results = optimizer.run()

    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    time_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"

    logger.info("="*80)

    # Check if optimization was successful
    if isinstance(new_results, dict) and 'genome' in new_results:
        logger.info(f"Dynamic DTN Rerun Optimization completed successfully in {time_str}")
        print("Dynamic DTN Rerun Optimization completed successfully.")
    elif isinstance(new_results, dict) and 'error' in new_results:
        logger.error(f"Dynamic DTN Rerun Optimization completed with errors in {time_str}")
        logger.error(f"Error: {new_results['error']}")
        print("Dynamic DTN Rerun Optimization completed with errors. See log for details.")
    else:
        logger.warning(f"Dynamic DTN Rerun Optimization completed with potential issues in {time_str}")

        # Add more detailed warning
        if isinstance(new_results, (int, float)) and new_results == -float('inf'):
            logger.warning("All solutions violated constraints. Try relaxing constraints in the configuration.")
            logger.warning("Common solutions include:")
            logger.warning("1. Increase the CAPEX budget per phase")
            logger.warning("2. Increase the total expenditure budget per phase")
            logger.warning("3. Increase the GHG budget per phase or use lower-emission supply systems")
        else:
            logger.warning("No valid genome found in results. This may be due to constraint violations.")

        # Add specific guidance based on which constraints were most frequently violated
        if hasattr(optimizer, 'capex_violations') and optimizer.capex_violations > 0:
            logger.warning(f"CAPEX budget was violated {optimizer.capex_violations} times during optimization.")
        if hasattr(optimizer, 'expenditure_violations') and optimizer.expenditure_violations > 0:
            logger.warning(f"Total expenditure budget was violated {optimizer.expenditure_violations} times during optimization.")
        if hasattr(optimizer, 'ghg_violations') and optimizer.ghg_violations > 0:
            logger.warning(f"GHG budget was violated {optimizer.ghg_violations} times during optimization.")

        print("Dynamic DTN Rerun Optimization completed with potential issues. See log for details.")

    logger.info(f"Results saved to: {Path(locator.get_optimization_results_folder()) / 'dynamic_dtn_optimization'}")
    logger.info("="*80)

    print(f"Total runtime: {time_str}")

    return new_results

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Run Dynamic DTN Rerun Optimization")
    parser.add_argument('-s', '--scenario', help='Path to the scenario folder')
    parser.add_argument('-c', '--config', help='Path to the config file')
    args = parser.parse_args()

    config = cea.config.Configuration(args.config)
    config.scenario = args.scenario
    main(config)
