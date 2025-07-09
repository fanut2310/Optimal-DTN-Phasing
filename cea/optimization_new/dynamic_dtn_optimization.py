from __future__ import annotations

###############################################################################
# 1) STANDARD IMPORTS                                                        #
###############################################################################
import argparse
import io
import logging
import os
import shutil
import time
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union

import cea.config
import cea.inputlocator
from cea.optimization_new.DTN_expansion_optimization import DTNExpansionOptimizer
from cea.technologies.thermal_network.thermal_network import main as thermal_network_simulation_main
from cea.technologies.thermal_network_costs.thermal_network_costs_new import main as thermal_network_costs_main

__author__ = "Fan Ut Chang"
__copyright__ = "Copyright 2025, City Energy Analyst"
__license__ = "MIT"

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run Dynamic DTN Optimization")
    parser.add_argument('-s', '--scenario', help='Path to the scenario folder')
    parser.add_argument('-c', '--config', help='Path to the config file')
    return parser.parse_args()

class DynamicDTNOptimizer:
    """
    Dynamic DTN Optimization module that analyzes how strategic modifications to building demands
    affect the optimal connection sequence in district thermal networks.
    """

    def __init__(self, locator: cea.inputlocator.InputLocator, config: cea.config.Configuration):
        """
        Initialize the Dynamic DTN Optimizer.

        Args:
            locator: CEA InputLocator object
            config: CEA Configuration object
        """
        self.locator = locator
        self.config = config
        self.network_type = config.dynamic_dtn_optimization.network_type
        self.heating_reduction = config.dynamic_dtn_optimization.heating_demand_reduction / 100.0  # Convert percentage to fraction
        self.cooling_reduction = config.dynamic_dtn_optimization.cooling_demand_reduction / 100.0  # Convert percentage to fraction
        self.dhw_reduction = config.dynamic_dtn_optimization.dhw_demand_reduction / 100.0  # Convert percentage to fraction
        self.electricity_reduction = config.dynamic_dtn_optimization.electricity_demand_reduction / 100.0  # Convert percentage to fraction
        self.num_last_clusters = config.dynamic_dtn_optimization.num_last_clusters

        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing Dynamic DTN Optimization module")

    def check_dtn_optimization_results(self):
        """
        Check if the required DTN optimization results exist.

        Returns:
            bool: True if results exist, False otherwise
            str: Path to the results file if it exists, None otherwise
        """
        # Get the path to the DTN expansion optimization results folder
        results_folder = Path(self.locator.get_dtn_expansion_optimization_results_folder())

        # Check if the folder exists
        if not results_folder.exists():
            self.logger.error(f"DTN expansion optimization results folder not found: {results_folder}")
            return False, None

        # Check if the all_evaluated_individuals_{network_type}.csv file exists
        results_file = results_folder / f"all_evaluated_individuals_{self.network_type}.csv"
        if not results_file.exists():
            self.logger.error(f"DTN expansion optimization results file not found: {results_file}")
            return False, None

        return True, str(results_file)

    def load_original_optimization_results(self, results_file):
        """
        Load the original DTN expansion optimization results.

        Args:
            results_file: Path to the results file

        Returns:
            Original optimization results
        """
        self.logger.info(f"Loading original DTN expansion optimization results from {results_file}")

        # Load the results file
        results_df = pd.read_csv(results_file)

        # Find the best solution (highest fitness for the objective function)
        if 'fitness_NPV' in results_df.columns:
            # For NPV objective, higher is better
            best_row = results_df.loc[results_df['fitness_NPV'].idxmax()]
        elif 'fitness_ROI' in results_df.columns:
            # For ROI objective, higher is better
            best_row = results_df.loc[results_df['fitness_ROI'].idxmax()]
        else:
            # If neither NPV nor ROI is found, use the first row
            self.logger.warning("Could not find NPV or ROI fitness values in results. Using first solution.")
            best_row = results_df.iloc[0]

        # Extract the genome (connection sequence)
        genome_str = best_row['genome']
        genome = eval(genome_str)  # Convert string representation to list

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

    def identify_last_clusters(self, results):
        """
        Identify the clusters that are connected last in the original optimization.

        Args:
            results: Results from the original optimization

        Returns:
            List of cluster IDs that are connected last
        """
        self.logger.info(f"Identifying the last {self.num_last_clusters} clusters to be connected")

        # Get the genome (connection sequence)
        genome = results['genome']

        # Get all clusters that are connected (phase > 0)
        connected_clusters = []
        for i, phase in enumerate(genome):
            if phase > 0:  # Skip unconnected clusters (phase 0)
                connected_clusters.append((i, phase))  # (cluster_id, phase)

        # Sort by phase (descending) to get the last connected clusters
        connected_clusters.sort(key=lambda x: x[1], reverse=True)

        # Get the last N clusters
        last_clusters = [cluster_id for cluster_id, _ in connected_clusters[:self.num_last_clusters]]

        self.logger.info(f"Last clusters to be connected: {last_clusters}")
        return last_clusters

    def modify_building_demands(self, last_clusters):
        """
        Modify the demand of buildings in the last clusters.

        Args:
            last_clusters: List of cluster IDs to modify

        Returns:
            Dictionary mapping building names to their modified demand files
        """
        self.logger.info("Modifying building demands for retrofit scenario")

        # Load the cluster assignment file to get buildings in each cluster
        cluster_assignment_file_path = self.locator.get_dtn_cluster_assignment_file()
        self.logger.info(f"Cluster assignment file path: {cluster_assignment_file_path}")

        # Convert to Path object
        cluster_assignment_file = Path(cluster_assignment_file_path)

        # Check if the file exists
        if not cluster_assignment_file.exists():
            self.logger.error(f"Cluster assignment file not found: {cluster_assignment_file}")
            raise FileNotFoundError(f"Cluster assignment file not found: {cluster_assignment_file}")

        # Convert back to string with proper escaping
        cluster_assignment_file_str = str(cluster_assignment_file).replace('\\', '/')
        self.logger.info(f"Reading cluster assignment file from: {cluster_assignment_file_str}")

        try:
            # Read the file using the string representation of the path
            cluster_df = pd.read_csv(cluster_assignment_file_str)
        except Exception as e:
            self.logger.error(f"Error reading cluster assignment file: {e}")
            # Try alternative approach
            self.logger.info("Trying alternative approach to read the file...")
            try:
                with open(cluster_assignment_file_str, 'r') as f:
                    file_content = f.read()
                    self.logger.info(f"File content preview: {file_content[:100]}...")
                cluster_df = pd.read_csv(io.StringIO(file_content))
            except Exception as e2:
                self.logger.error(f"Alternative approach also failed: {e2}")
                raise

        # Get all buildings in the last clusters
        buildings_to_modify = []
        for cluster_id in last_clusters:
            cluster_buildings = cluster_df[cluster_df['cluster_id'] == cluster_id]['building_name'].tolist()
            buildings_to_modify.extend(cluster_buildings)

        self.logger.info(f"Buildings to modify: {buildings_to_modify}")

        # Create a directory for modified demand files
        modified_demand_dir = Path(self.locator.get_optimization_results_folder()) / "dynamic_dtn_optimization" / "modified_demands"
        modified_demand_dir.mkdir(parents=True, exist_ok=True)

        # Dictionary to store original and modified demand files
        modified_demand_files = {}

        # Modify the demand for each building
        for building in buildings_to_modify:
            # Load the original demand file
            original_demand_file_path = self.locator.get_demand_results_file(building)
            self.logger.info(f"Demand results file path for building {building}: {original_demand_file_path}")

            # Convert to Path object
            original_demand_file = Path(original_demand_file_path)

            # Check if the file exists
            if not original_demand_file.exists():
                self.logger.error(f"Demand results file not found for building {building}: {original_demand_file}")
                raise FileNotFoundError(f"Demand results file not found for building {building}: {original_demand_file}")

            # Convert back to string with proper escaping
            original_demand_file_str = str(original_demand_file).replace('\\', '/')
            self.logger.info(f"Reading demand results file from: {original_demand_file_str}")

            try:
                # Read the file using the string representation of the path
                demand_df = pd.read_csv(original_demand_file_str)
            except Exception as e:
                self.logger.error(f"Error reading demand results file for building {building}: {e}")
                # Try alternative approach
                self.logger.info("Trying alternative approach to read the file...")
                try:
                    with open(original_demand_file_str, 'r') as f:
                        file_content = f.read()
                    demand_df = pd.read_csv(io.StringIO(file_content))
                except Exception as e2:
                    self.logger.error(f"Alternative approach also failed: {e2}")
                    raise

            # Apply reductions to the demand
            if self.heating_reduction > 0:
                heating_columns = [col for col in demand_df.columns if 'QH' in col]
                for col in heating_columns:
                    demand_df[col] = demand_df[col] * (1 - self.heating_reduction)

            if self.cooling_reduction > 0:
                cooling_columns = [col for col in demand_df.columns if 'QC' in col]
                for col in cooling_columns:
                    demand_df[col] = demand_df[col] * (1 - self.cooling_reduction)

            if self.dhw_reduction > 0:
                dhw_columns = [col for col in demand_df.columns if 'QHW' in col]
                for col in dhw_columns:
                    demand_df[col] = demand_df[col] * (1 - self.dhw_reduction)

            if self.electricity_reduction > 0:
                elec_columns = [col for col in demand_df.columns if 'E' in col]
                for col in elec_columns:
                    demand_df[col] = demand_df[col] * (1 - self.electricity_reduction)

            # Save the modified demand file
            modified_demand_file = modified_demand_dir / f"{building}.csv"
            modified_demand_file_str = str(modified_demand_file).replace('\\', '/')
            self.logger.info(f"Saving modified demand file to: {modified_demand_file_str}")

            try:
                # Save the file using the string representation of the path
                demand_df.to_csv(modified_demand_file_str, index=False)
            except Exception as e:
                self.logger.error(f"Error saving modified demand file for building {building}: {e}")
                # Try alternative approach
                self.logger.info("Trying alternative approach to save the file...")
                try:
                    with open(modified_demand_file_str, 'w') as f:
                        demand_df.to_csv(f, index=False)
                except Exception as e2:
                    self.logger.error(f"Alternative approach also failed: {e2}")
                    raise

            # Store the mapping
            modified_demand_files[building] = {
                'original': original_demand_file_str,
                'modified': modified_demand_file_str
            }

        self.modified_demand_files = modified_demand_files
        return modified_demand_files

    def rerun_thermal_network_simulation(self):
        """
        Rerun the thermal network simulation with modified demands.
        """
        self.logger.info("Rerunning thermal network simulation with modified demands")

        # Create a copy of the config for the thermal network simulation
        tn_config = self.config.copy()
        tn_config.thermal_network.network_type = self.network_type

        # Temporarily replace the original demand files with modified ones
        self._replace_demand_files(use_modified=True)

        try:
            # Run the thermal network simulation
            thermal_network_simulation_main(tn_config)

            # Run the thermal network costs calculation
            tnc_config = self.config.copy()
            tnc_config.thermal_network_costs.network_type = self.network_type
            thermal_network_costs_main(tnc_config)
        finally:
            # Restore the original demand files
            self._replace_demand_files(use_modified=False)

    def rerun_dtn_optimization(self):
        """
        Rerun the DTN optimization with the updated thermal network results.

        Returns:
            New optimization results
        """
        self.logger.info("Rerunning DTN expansion optimization with modified demands")

        # Create a copy of the config for the new optimization
        new_config = self.config.copy()

        # Temporarily replace the original demand files with modified ones
        self._replace_demand_files(use_modified=True)

        try:
            # Run the new DTN optimization
            optimizer = DTNExpansionOptimizer(
                locator=self.locator,
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

            # Run the optimization
            population_size = new_config.dtn_expansion_optimization.population_size
            num_generations = new_config.dtn_expansion_optimization.num_generations
            new_results = optimizer.optimize(population_size=population_size, num_generations=num_generations)

            # Save the new results
            self.new_optimizer = optimizer
            self.new_results = new_results

            return new_results
        finally:
            # Restore the original demand files
            self._replace_demand_files(use_modified=False)

    def compare_results(self):
        """
        Compare the original and new optimization results to assess sensitivity.

        Returns:
            DataFrame with comparison results
        """
        self.logger.info("Comparing original and new optimization results")

        # Create a directory for comparison results
        comparison_dir = Path(self.locator.get_optimization_results_folder()) / "dynamic_dtn_optimization" / "comparison"
        comparison_dir.mkdir(parents=True, exist_ok=True)

        # Extract connection sequences
        original_sequence = self._extract_connection_sequence(self.original_results)
        new_sequence = self._extract_connection_sequence(self.new_results)

        # Compare the sequences
        sequence_changes = self._compare_sequences(original_sequence, new_sequence)

        # Compare objective values
        objective_changes = self._compare_objectives(self.original_results, self.new_results)

        # Create a summary DataFrame
        summary = {
            'parameter': ['Heating Demand Reduction', 'Cooling Demand Reduction', 'DHW Demand Reduction', 'Electricity Demand Reduction',
                         'Number of Last Clusters Modified', 'Sequence Changes', 'Objective Value Changes'],
            'value': [f"{self.heating_reduction * 100}%", f"{self.cooling_reduction * 100}%", 
                     f"{self.dhw_reduction * 100}%", f"{self.electricity_reduction * 100}%",
                     self.num_last_clusters, sequence_changes, objective_changes]
        }
        summary_df = pd.DataFrame(summary)

        # Save the summary
        summary_file = comparison_dir / "summary.csv"
        summary_df.to_csv(str(summary_file), index=False)

        # Create detailed comparison of connection sequences
        sequence_comparison = {
            'cluster_id': list(range(len(original_sequence['flat']))),
            'original_phase': original_sequence['flat'],
            'new_phase': new_sequence['flat']
        }
        sequence_df = pd.DataFrame(sequence_comparison)
        sequence_df['phase_change'] = sequence_df['new_phase'] - sequence_df['original_phase']

        # Save the sequence comparison
        sequence_file = comparison_dir / "sequence_comparison.csv"
        sequence_df.to_csv(str(sequence_file), index=False)

        # Create visualizations
        self._create_visualizations(sequence_df, comparison_dir)

        self.logger.info(f"Comparison results saved to {comparison_dir}")
        return summary_df

    def _parse_list_param(self, param_str):
        """Parse a comma-separated string parameter into a list of values."""
        if not param_str or param_str.strip() == '':
            return None
        return [float(x) for x in param_str.split(',')]

    def _replace_demand_files(self, use_modified=True):
        """
        Replace original demand files with modified ones or vice versa.

        Args:
            use_modified: If True, replace original with modified. If False, restore original.
        """
        for building, files in self.modified_demand_files.items():
            if use_modified:
                # Backup the original file
                backup_file = files['original'] + '.bak'
                self.logger.info(f"Creating backup of original file: {backup_file}")

                try:
                    if not os.path.exists(backup_file):
                        shutil.copy2(files['original'], backup_file)
                except Exception as e:
                    self.logger.error(f"Error creating backup file for building {building}: {e}")
                    continue

                # Replace with modified file
                self.logger.info(f"Replacing original file with modified file: {files['original']} <- {files['modified']}")
                try:
                    shutil.copy2(files['modified'], files['original'])
                except Exception as e:
                    self.logger.error(f"Error replacing original file with modified file for building {building}: {e}")
                    continue
            else:
                # Restore the original file
                backup_file = files['original'] + '.bak'
                self.logger.info(f"Restoring original file from backup: {files['original']} <- {backup_file}")

                try:
                    if os.path.exists(backup_file):
                        shutil.copy2(backup_file, files['original'])
                        os.remove(backup_file)
                except Exception as e:
                    self.logger.error(f"Error restoring original file from backup for building {building}: {e}")
                    continue

    def _extract_connection_sequence(self, results):
        """
        Extract the connection sequence from optimization results.

        Args:
            results: Optimization results

        Returns:
            Dictionary with connection sequence information
        """
        # Get the genome (connection sequence)
        genome = results['genome']

        # Extract the connection sequence by phase
        connection_sequence = []
        for phase in range(1, self.config.dtn_expansion_optimization.num_phases + 1):
            clusters_in_phase = []
            for i, cluster_phase in enumerate(genome):
                if cluster_phase == phase:
                    clusters_in_phase.append(i)
            connection_sequence.append(clusters_in_phase)

        return {
            'by_phase': connection_sequence,
            'flat': genome
        }

    def _compare_sequences(self, original_sequence, new_sequence):
        """
        Compare original and new connection sequences.

        Args:
            original_sequence: Original connection sequence
            new_sequence: New connection sequence

        Returns:
            String describing the changes
        """
        # Count how many clusters changed phases
        changes = 0
        for i, (orig, new) in enumerate(zip(original_sequence['flat'], new_sequence['flat'])):
            if orig != new:
                changes += 1

        return f"{changes} clusters changed phases ({changes/len(original_sequence['flat'])*100:.1f}%)"

    def _compare_objectives(self, original_results, new_results):
        """
        Compare objective values between original and new results.

        Args:
            original_results: Original optimization results
            new_results: New optimization results

        Returns:
            String describing the changes
        """
        changes = []

        # Check for NPV
        if 'fitness_NPV' in original_results and 'fitness_NPV' in new_results:
            original_npv = original_results['fitness_NPV']
            new_npv = new_results['fitness_NPV']
            percent_change = (new_npv - original_npv) / abs(original_npv) * 100
            changes.append(f"NPV: {percent_change:.2f}%")

        # Check for ROI
        if 'fitness_ROI' in original_results and 'fitness_ROI' in new_results:
            original_roi = original_results['fitness_ROI']
            new_roi = new_results['fitness_ROI']
            percent_change = (new_roi - original_roi) / abs(original_roi) * 100
            changes.append(f"ROI: {percent_change:.2f}%")

        # Check for emissions
        if 'fitness_emissions' in original_results and 'fitness_emissions' in new_results:
            original_emissions = original_results['fitness_emissions']
            new_emissions = new_results['fitness_emissions']
            percent_change = (new_emissions - original_emissions) / abs(original_emissions) * 100
            changes.append(f"Emissions: {percent_change:.2f}%")

        return ", ".join(changes) if changes else "No comparable objective values found"

    def _create_visualizations(self, sequence_df, output_dir):
        """
        Create visualizations of the results comparison.

        Args:
            sequence_df: DataFrame with sequence comparison
            output_dir: Directory to save visualizations
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            # Set style
            sns.set(style="whitegrid")

            # Create a heatmap of phase changes
            plt.figure(figsize=(12, 8))
            pivot_df = sequence_df.pivot_table(
                index='cluster_id', 
                values='phase_change',
                aggfunc='first'
            ).reset_index()
            pivot_df = pivot_df.sort_values('phase_change')

            # Plot the heatmap
            ax = sns.heatmap(
                pivot_df[['phase_change']].T, 
                cmap='RdBu_r',
                center=0,
                cbar_kws={'label': 'Phase Change (New - Original)'}
            )
            ax.set_xticklabels(pivot_df['cluster_id'])
            ax.set_title('Phase Changes by Cluster')
            ax.set_xlabel('Cluster ID')
            ax.set_ylabel('')

            # Save the figure
            plt.tight_layout()
            plt.savefig(str(Path(output_dir) / 'phase_changes_heatmap.png'), dpi=300)
            plt.close()

            # Create a bar chart of original vs new phases
            plt.figure(figsize=(12, 8))

            # Sort by original phase
            sorted_df = sequence_df.sort_values(['original_phase', 'cluster_id'])

            # Plot
            bar_width = 0.35
            x = np.arange(len(sorted_df))

            plt.bar(x - bar_width/2, sorted_df['original_phase'], bar_width, label='Original Phase')
            plt.bar(x + bar_width/2, sorted_df['new_phase'], bar_width, label='New Phase')

            plt.xlabel('Cluster ID')
            plt.ylabel('Phase')
            plt.title('Original vs New Connection Phases')
            plt.xticks(x, sorted_df['cluster_id'], rotation=90)
            plt.legend()

            # Save the figure
            plt.tight_layout()
            plt.savefig(str(Path(output_dir) / 'phase_comparison_bar.png'), dpi=300)
            plt.close()

        except ImportError:
            self.logger.warning("Matplotlib or seaborn not available. Skipping visualizations.")

    def run(self):
        """
        Run the complete Dynamic DTN Optimization workflow.

        Returns:
            Summary DataFrame with comparison results
        """
        self.logger.info("Starting Dynamic DTN Optimization workflow")

        # Step 1: Check if DTN optimization results exist
        results_exist, results_file = self.check_dtn_optimization_results()
        if not results_exist:
            error_msg = (
                "DTN expansion optimization results not found. Please run the DTN expansion optimization first "
                "using the 'dtn-expansion-optimization' module with the same network type."
            )
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        # Step 2: Load the original DTN optimization results
        original_results = self.load_original_optimization_results(results_file)

        # Step 3: Identify the last clusters to be connected
        last_clusters = self.identify_last_clusters(original_results)

        # Step 4: Modify the demand of buildings in the last clusters
        self.modify_building_demands(last_clusters)

        # Step 5: Rerun the thermal network simulation with modified demands
        self.rerun_thermal_network_simulation()

        # Step 6: Rerun the DTN optimization with updated thermal network results
        new_results = self.rerun_dtn_optimization()

        # Step 7: Compare the results to assess sensitivity
        summary = self.compare_results()

        self.logger.info("Dynamic DTN Optimization workflow completed")
        return summary

def main(config):
    """
    Run the Dynamic DTN Optimization module.

    Args:
        config: CEA Configuration object
    """
    locator = cea.inputlocator.InputLocator(config.scenario)

    # Create and run the Dynamic DTN Optimizer
    optimizer = DynamicDTNOptimizer(locator, config)
    summary = optimizer.run()

    print("Dynamic DTN Optimization completed successfully.")
    print(summary)

    return summary

if __name__ == '__main__':
    args = parse_args()
    config = cea.config.Configuration(args.config)
    config.scenario = args.scenario
    main(config)
