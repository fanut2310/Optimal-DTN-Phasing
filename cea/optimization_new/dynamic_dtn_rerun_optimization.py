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

class DynamicDTNRerunOptimization:
    """
    Module for rerunning DTN optimization with modified demand files.
    This module implements step 6 of the dynamic DTN optimization process.
    """

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

        # Run the new DTN optimization with the modified locator
        self.logger.info("Initializing DTN expansion optimizer with modified demands")
        optimizer = DTNExpansionOptimizer(
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
            multi_objective_functions=new_config.dtn_expansion_optimization.multi_objective_functions
        )
        self.logger.info(f"DTN expansion optimizer initialized with {new_config.dtn_expansion_optimization.num_phases} phases")

        # Run the optimization
        population_size = new_config.dtn_expansion_optimization.population_size
        num_generations = new_config.dtn_expansion_optimization.num_generations
        self.logger.info(f"Starting DTN expansion optimization with population size {population_size} and {num_generations} generations")
        new_results = optimizer.optimize(population_size=population_size, num_generations=num_generations)
        self.logger.info("DTN expansion optimization completed successfully")

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
                self.logger.info("Consider relaxing constraints such as CAPEX budget, total expenditure budget, or GHG budget.")
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
        logger.warning("No valid genome found in results. This may be due to constraint violations.")
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
