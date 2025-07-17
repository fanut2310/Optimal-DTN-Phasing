from __future__ import annotations

###############################################################################
# 1) STANDARD IMPORTS                                                        #
###############################################################################
import argparse
import io
import logging
import os
import shutil
import sys
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

###############################################################################
# 2) CUSTOM INPUTLOCATOR                                                     #
###############################################################################

class ModifiedDemandsLocator(cea.inputlocator.InputLocator):
    """A custom InputLocator that redirects demand file requests to modified versions."""

    def __init__(self, locator, modified_demand_files):
        """
        Initialize with the original locator and a mapping of modified files.

        Args:
            locator: The original InputLocator
            modified_demand_files: Dictionary mapping building names to modified file paths
        """
        super().__init__(locator.scenario)
        # Copy all attributes from the original locator
        self.__dict__.update(locator.__dict__)
        self.original_locator = locator
        self.modified_demand_files = modified_demand_files

    def get_demand_results_file(self, building, format='csv'):
        """
        Override to return the path to the modified demand file if available.

        Args:
            building: Building name
            format: File format (default: 'csv')

        Returns:
            Path to the modified demand file if available, otherwise the original path
        """
        if building in self.modified_demand_files:
            return self.modified_demand_files[building]['modified']
        else:
            return self.original_locator.get_demand_results_file(building, format)

__author__ = "Fan Ut Chang"
__copyright__ = "Copyright 2025, City Energy Analyst"
__license__ = "MIT"

###############################################################################
# 3) LOGGING                                                                 #
###############################################################################

def log() -> logging.Logger:
    """Configure and return a logger for the dynamic DTN optimization module."""
    lg = logging.getLogger("cea.dynamic_dtn_optimization")
    if not lg.handlers:
        # Configure logging to output to console
        logging.basicConfig(level=logging.INFO,
                           format="%(asctime)s | %(levelname)5s | %(message)s",
                           datefmt="%H:%M:%S")
    return lg

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

        # Get testing clusters from config if available
        self.testing_clusters = None
        if hasattr(config.dtn_expansion_optimization, 'testing_clusters') and config.dtn_expansion_optimization.testing_clusters:
            # Convert string to list of integers if needed
            if isinstance(config.dtn_expansion_optimization.testing_clusters, str):
                self.testing_clusters = [int(c.strip()) for c in config.dtn_expansion_optimization.testing_clusters.split(',') if c.strip()]
            else:
                self.testing_clusters = config.dtn_expansion_optimization.testing_clusters

        # Set up logging
        self.logger = log()
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
        self.logger.info(f"Checking for DTN expansion optimization results in: {results_folder}")

        # Check if the folder exists
        if not results_folder.exists():
            self.logger.error(f"DTN expansion optimization results folder not found: {results_folder}")
            return False, None

        # Check if the all_evaluated_individuals_{network_type}.csv file exists
        results_file = results_folder / f"all_evaluated_individuals_{self.network_type}.csv"
        if not results_file.exists():
            self.logger.error(f"DTN expansion optimization results file not found: {results_file}")
            return False, None

        self.logger.info(f"DTN expansion optimization results found: {results_file}")
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

        # Also load the optimization settings file to get testing_clusters if available
        results_folder = Path(self.locator.get_dtn_expansion_optimization_results_folder())
        settings_file = results_folder / "optimization_settings.csv"
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
        self.logger.info(f"Genome: {genome}")
        self.logger.info(f"Testing clusters: {self.testing_clusters}")

        # Get all clusters that are connected (phase > 0)
        connected_clusters = []
        for i, phase in enumerate(genome):
            if phase > 0:  # Skip unconnected clusters (phase 0)
                # Map genome index to actual cluster ID if testing_clusters is available
                if self.testing_clusters and i < len(self.testing_clusters):
                    cluster_id = self.testing_clusters[i]
                else:
                    cluster_id = i + 1  # Use 1-based indexing for cluster IDs
                connected_clusters.append((cluster_id, phase))  # (cluster_id, phase)
                self.logger.info(f"Cluster {cluster_id} is connected in phase {phase}")

        # Sort by phase (descending) to get the last connected clusters
        connected_clusters.sort(key=lambda x: x[1], reverse=True)
        self.logger.info(f"Connected clusters sorted by phase (descending): {connected_clusters}")

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

        # Use os.path.normpath to ensure the path is properly formatted for the current OS
        cluster_assignment_file_str = os.path.normpath(str(cluster_assignment_file))
        self.logger.info(f"Reading cluster assignment file from: {cluster_assignment_file_str}")

        try:
            # Read the file using the normalized path
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
            cluster_buildings = cluster_df[cluster_df['cluster'] == cluster_id]['name'].tolist()
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

            # Use os.path.normpath to ensure the path is properly formatted for the current OS
            original_demand_file_str = os.path.normpath(str(original_demand_file))
            self.logger.info(f"Reading demand results file from: {original_demand_file_str}")

            try:
                # Read the file using the normalized path
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
                self.logger.info(f"Applying {self.heating_reduction*100}% heating demand reduction to building {building}")
                # Space heating related columns
                heating_patterns = ['QH', 'Qhs', 'hs_', '_hs']
                heating_columns = []
                for pattern in heating_patterns:
                    pattern_columns = [col for col in demand_df.columns if pattern in col]
                    heating_columns.extend(pattern_columns)
                    self.logger.debug(f"Found {len(pattern_columns)} columns matching pattern '{pattern}'")

                # Exclude DHW columns if they were caught by the patterns
                heating_columns = [col for col in heating_columns if 'ww' not in col.lower() and 'hw' not in col.lower()]
                unique_heating_columns = set(heating_columns)
                self.logger.info(f"Identified {len(unique_heating_columns)} unique heating columns to modify: {', '.join(sorted(unique_heating_columns))}")

                # Apply reduction to all heating columns
                for col in set(heating_columns):  # Use set to remove duplicates
                    demand_df[col] = demand_df[col] * (1 - self.heating_reduction)

            if self.cooling_reduction > 0:
                self.logger.info(f"Applying {self.cooling_reduction*100}% cooling demand reduction to building {building}")
                # Space cooling related columns
                cooling_patterns = ['QC', 'Qc', 'cs_', '_cs', 'cdata', 'cre']
                cooling_columns = []
                for pattern in cooling_patterns:
                    pattern_columns = [col for col in demand_df.columns if pattern in col]
                    cooling_columns.extend(pattern_columns)
                    self.logger.debug(f"Found {len(pattern_columns)} columns matching pattern '{pattern}'")

                # Apply reduction to all cooling columns
                unique_cooling_columns = set(cooling_columns)
                self.logger.info(f"Identified {len(unique_cooling_columns)} unique cooling columns to modify: {', '.join(sorted(unique_cooling_columns))}")
                for col in set(cooling_columns):  # Use set to remove duplicates
                    demand_df[col] = demand_df[col] * (1 - self.cooling_reduction)

            if self.dhw_reduction > 0:
                self.logger.info(f"Applying {self.dhw_reduction*100}% DHW demand reduction to building {building}")
                # DHW related columns
                dhw_patterns = ['QHW', 'Qww', 'ww_', '_ww']
                dhw_columns = []
                for pattern in dhw_patterns:
                    pattern_columns = [col for col in demand_df.columns if pattern in col]
                    dhw_columns.extend(pattern_columns)
                    self.logger.debug(f"Found {len(pattern_columns)} columns matching pattern '{pattern}'")

                # Apply reduction to all DHW columns
                unique_dhw_columns = set(dhw_columns)
                self.logger.info(f"Identified {len(unique_dhw_columns)} unique DHW columns to modify: {', '.join(sorted(unique_dhw_columns))}")
                for col in set(dhw_columns):  # Use set to remove duplicates
                    demand_df[col] = demand_df[col] * (1 - self.dhw_reduction)

            if self.electricity_reduction > 0:
                self.logger.info(f"Applying {self.electricity_reduction*100}% electricity demand reduction to building {building}")
                # Electricity related columns
                elec_patterns = ['E_', 'Ea', 'Eve', 'GRID']
                elec_columns = []
                for pattern in elec_patterns:
                    pattern_columns = [col for col in demand_df.columns if pattern in col]
                    elec_columns.extend(pattern_columns)
                    self.logger.debug(f"Found {len(pattern_columns)} columns matching pattern '{pattern}'")

                # Exclude PV generation columns
                elec_columns = [col for col in elec_columns if 'PV' not in col]
                unique_elec_columns = set(elec_columns)
                self.logger.info(f"Identified {len(unique_elec_columns)} unique electricity columns to modify (excluding PV generation): {', '.join(sorted(unique_elec_columns))}")

                # Apply reduction to all electricity columns
                for col in set(elec_columns):  # Use set to remove duplicates
                    demand_df[col] = demand_df[col] * (1 - self.electricity_reduction)

            # Save the modified demand file
            modified_demand_file = modified_demand_dir / f"{building}.csv"
            modified_demand_file_str = os.path.normpath(str(modified_demand_file))
            self.logger.info(f"Saving modified demand file to: {modified_demand_file_str}")

            try:
                # Save the file using the normalized path
                demand_df.to_csv(modified_demand_file_str, index=False)
                self.logger.info(f"Successfully saved modified demand file for building {building}")
            except Exception as e:
                self.logger.error(f"Error saving modified demand file for building {building}: {e}")
                # Try alternative approach
                self.logger.info("Trying alternative approach to save the file...")
                try:
                    with open(modified_demand_file_str, 'w') as f:
                        demand_df.to_csv(f, index=False)
                    self.logger.info(f"Successfully saved modified demand file using alternative approach")
                except Exception as e2:
                    self.logger.error(f"Alternative approach also failed: {e2}")
                    raise

            # Store the mapping
            modified_demand_files[building] = {
                'original': original_demand_file_str,
                'modified': modified_demand_file_str
            }
            self.logger.debug(f"Added mapping for building {building} to modified_demand_files dictionary")

        self.modified_demand_files = modified_demand_files
        self.logger.info(f"Completed demand modifications for all {len(buildings_to_modify)} buildings in the last clusters")
        return modified_demand_files

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

    def _create_temp_scenario_with_modified_demands(self):
        """
        Create a temporary scenario with modified demand files.

        Returns:
            str: Path to the temporary scenario directory
        """
        self.logger.info("Creating temporary scenario with modified demand files")

        # Create a temporary directory
        temp_dir = Path(self.locator.get_optimization_results_folder()) / "dynamic_dtn_optimization" / "temp_scenario"
        temp_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Created temporary scenario directory: {temp_dir}")

        # Create necessary subdirectories
        temp_demand_dir = temp_dir / "outputs" / "data" / "demand"
        temp_demand_dir.mkdir(parents=True, exist_ok=True)

        # Copy necessary files from original scenario
        self._copy_scenario_files(str(self.config.scenario), str(temp_dir))

        # Copy all building demand files (modified and original)
        self._copy_all_building_demand_files(temp_demand_dir)

        # Generate updated Total_demand.csv and Total_demand_hourly.csv
        self._update_total_demand_files(temp_demand_dir)

        self.logger.info(f"Temporary scenario created at: {temp_dir}")
        return str(temp_dir)

    def _copy_scenario_files(self, source_scenario, target_scenario):
        """
        Copy necessary files from source scenario to target scenario.

        Args:
            source_scenario (str): Path to the source scenario
            target_scenario (str): Path to the target scenario
        """
        self.logger.info(f"Copying necessary files from {source_scenario} to {target_scenario}")

        # Create a source locator
        source_locator = cea.inputlocator.InputLocator(source_scenario)

        # Create target directories
        target_path = Path(target_scenario)
        (target_path / "inputs").mkdir(parents=True, exist_ok=True)
        (target_path / "inputs" / "building-geometry").mkdir(parents=True, exist_ok=True)
        (target_path / "inputs" / "networks").mkdir(parents=True, exist_ok=True)
        (target_path / "inputs" / "weather").mkdir(parents=True, exist_ok=True)
        (target_path / "outputs" / "data" / "demand").mkdir(parents=True, exist_ok=True)

        # Create thermal network directory structure
        thermal_network_dir = target_path / "outputs" / "data" / "thermal-network" / self.network_type
        thermal_network_dir.mkdir(parents=True, exist_ok=True)

        # Copy building geometry files
        zone_file = source_locator.get_zone_geometry()
        if os.path.exists(zone_file):
            shutil.copy2(zone_file, target_path / "inputs" / "building-geometry" / "zone.shp")
            # Copy related files (.dbf, .shx, etc.)
            for ext in ['.dbf', '.shx', '.prj', '.cpg']:
                related_file = zone_file.replace('.shp', ext)
                if os.path.exists(related_file):
                    shutil.copy2(related_file, target_path / "inputs" / "building-geometry" / f"zone{ext}")

        # Copy network files
        network_file = source_locator.get_network_layout_edges_shapefile(self.network_type)
        if os.path.exists(network_file):
            target_network_dir = target_path / "inputs" / "networks"
            shutil.copy2(network_file, target_network_dir / f"{self.network_type}.shp")
            # Copy related files (.dbf, .shx, etc.)
            for ext in ['.dbf', '.shx', '.prj', '.cpg']:
                related_file = network_file.replace('.shp', ext)
                if os.path.exists(related_file):
                    shutil.copy2(related_file, target_network_dir / f"{self.network_type}{ext}")

        # Copy Total_demand.csv file which is required for thermal network simulation
        total_demand_file = source_locator.get_total_demand()
        if os.path.exists(total_demand_file):
            self.logger.info(f"Copying Total_demand.csv file from {total_demand_file}")
            shutil.copy2(total_demand_file, target_path / "outputs" / "data" / "demand" / "Total_demand.csv")
        else:
            self.logger.warning(f"Total_demand.csv file not found at {total_demand_file}")

        # Copy weather.epw file which is required for thermal network simulation
        weather_file = source_locator.get_weather_file()
        if os.path.exists(weather_file):
            self.logger.info(f"Copying weather.epw file from {weather_file}")
            shutil.copy2(weather_file, target_path / "inputs" / "weather" / "weather.epw")
        else:
            self.logger.warning(f"Weather file not found at {weather_file}")

        # Create complete database directory structure
        (target_path / "inputs" / "database" / "COMPONENTS" / "CONVERSION").mkdir(parents=True, exist_ok=True)
        (target_path / "inputs" / "database" / "COMPONENTS" / "DISTRIBUTION").mkdir(parents=True, exist_ok=True)
        (target_path / "inputs" / "database" / "COMPONENTS" / "FEEDSTOCKS").mkdir(parents=True, exist_ok=True)
        (target_path / "inputs" / "database" / "COMPONENTS" / "FEEDSTOCKS" / "FEEDSTOCKS_LIBRARY").mkdir(parents=True, exist_ok=True)

        # Copy all CONVERSION files (including HEAT_EXCHANGERS.csv, HYDRAULICS.csv, VAPOR_COMPRESSION.csv, COOLING_TOWER.csv)
        conversion_folder = source_locator.get_db4_components_conversion_folder()
        if os.path.exists(conversion_folder):
            self.logger.info(f"Copying all conversion files from {conversion_folder}")
            for file in os.listdir(conversion_folder):
                if file.endswith('.csv'):
                    source_file = os.path.join(conversion_folder, file)
                    target_file = target_path / "inputs" / "database" / "COMPONENTS" / "CONVERSION" / file
                    shutil.copy2(source_file, target_file)
                    self.logger.info(f"Copied {file} from {source_file}")
        else:
            self.logger.warning(f"Conversion folder not found at {conversion_folder}")

        # Copy all DISTRIBUTION files (including THERMAL_GRID.csv)
        distribution_folder = source_locator.get_db4_components_distribution_folder()
        if os.path.exists(distribution_folder):
            self.logger.info(f"Copying all distribution files from {distribution_folder}")
            for file in os.listdir(distribution_folder):
                if file.endswith('.csv'):
                    source_file = os.path.join(distribution_folder, file)
                    target_file = target_path / "inputs" / "database" / "COMPONENTS" / "DISTRIBUTION" / file
                    shutil.copy2(source_file, target_file)
                    self.logger.info(f"Copied {file} from {source_file}")
        else:
            self.logger.warning(f"Distribution folder not found at {distribution_folder}")

        # Copy all FEEDSTOCKS files
        feedstocks_folder = source_locator.get_db4_components_feedstocks_folder()
        if os.path.exists(feedstocks_folder):
            self.logger.info(f"Copying all feedstocks files from {feedstocks_folder}")
            for file in os.listdir(feedstocks_folder):
                if file.endswith('.csv'):
                    source_file = os.path.join(feedstocks_folder, file)
                    target_file = target_path / "inputs" / "database" / "COMPONENTS" / "FEEDSTOCKS" / file
                    shutil.copy2(source_file, target_file)
                    self.logger.info(f"Copied {file} from {source_file}")
        else:
            self.logger.warning(f"Feedstocks folder not found at {feedstocks_folder}")

        # Copy all FEEDSTOCKS_LIBRARY files
        feedstocks_library_folder = source_locator.get_db4_components_feedstocks_library_folder()
        if os.path.exists(feedstocks_library_folder):
            self.logger.info(f"Copying all feedstocks library files from {feedstocks_library_folder}")
            for file in os.listdir(feedstocks_library_folder):
                if file.endswith('.csv'):
                    source_file = os.path.join(feedstocks_library_folder, file)
                    target_file = target_path / "inputs" / "database" / "COMPONENTS" / "FEEDSTOCKS" / "FEEDSTOCKS_LIBRARY" / file
                    shutil.copy2(source_file, target_file)
                    self.logger.info(f"Copied {file} from {source_file}")
        else:
            self.logger.warning(f"Feedstocks library folder not found at {feedstocks_library_folder}")

        # Copy thermal network files from the original scenario
        source_thermal_network_dir = Path(source_locator.get_thermal_network_folder()) / self.network_type
        if source_thermal_network_dir.exists():
            self.logger.info(f"Copying thermal network files from {source_thermal_network_dir}")

            # Copy edges.shp and related files
            edges_shp = source_thermal_network_dir / "edges.shp"
            if edges_shp.exists():
                self.logger.info(f"Copying edges.shp file from {edges_shp}")
                shutil.copy2(edges_shp, thermal_network_dir / "edges.shp")
                # Copy related files (.dbf, .shx, etc.)
                for ext in ['.dbf', '.shx', '.prj', '.cpg']:
                    related_file = str(edges_shp).replace('.shp', ext)
                    if os.path.exists(related_file):
                        shutil.copy2(related_file, thermal_network_dir / f"edges{ext}")
            else:
                self.logger.warning(f"edges.shp file not found at {edges_shp}")

            # Copy nodes.shp and related files
            nodes_shp = source_thermal_network_dir / "nodes.shp"
            if nodes_shp.exists():
                self.logger.info(f"Copying nodes.shp file from {nodes_shp}")
                shutil.copy2(nodes_shp, thermal_network_dir / "nodes.shp")
                # Copy related files (.dbf, .shx, etc.)
                for ext in ['.dbf', '.shx', '.prj', '.cpg']:
                    related_file = str(nodes_shp).replace('.shp', ext)
                    if os.path.exists(related_file):
                        shutil.copy2(related_file, thermal_network_dir / f"nodes{ext}")
            else:
                self.logger.warning(f"nodes.shp file not found at {nodes_shp}")

            # Copy other thermal network files (CSV files, etc.)
            for file in source_thermal_network_dir.glob("*.csv"):
                self.logger.info(f"Copying {file.name} file from {file}")
                shutil.copy2(file, thermal_network_dir / file.name)
        else:
            self.logger.warning(f"Thermal network directory not found at {source_thermal_network_dir}")

        self.logger.info("Necessary files copied to temporary scenario")

    def _copy_results_from_temp_scenario(self, temp_scenario_dir):
        """
        Copy thermal network results from temporary scenario to dynamic DTN folder.

        Args:
            temp_scenario_dir (str): Path to the temporary scenario directory
        """
        self.logger.info("Copying thermal network results from temporary scenario to dynamic DTN folder")

        # Create a directory for thermal network results
        tn_results_dir = Path(self.locator.get_optimization_results_folder()) / "dynamic_dtn_optimization" / "thermal_network"
        tn_results_dir.mkdir(parents=True, exist_ok=True)

        # Create a temporary locator
        temp_locator = cea.inputlocator.InputLocator(temp_scenario_dir)

        # Copy all files from Thermal Network Part 2 and Part 3

        # Part 3 (Costs) - 1 file
        # 1. Network costs file
        costs_file = temp_locator.get_network_layout_costs_file(self.network_type)
        if os.path.exists(costs_file):
            shutil.copy2(costs_file, tn_results_dir / f"{self.network_type}_costs.csv")
            self.logger.info(f"Copied network costs file: {os.path.basename(costs_file)}")
        else:
            self.logger.warning(f"Network costs file not found: {costs_file}")

        # Part 2 (Simulation) - 16 files
        # 2. Edge mass flow file
        edge_massflow_file = temp_locator.get_thermal_network_layout_massflow_edges_file(self.network_type, '')
        if os.path.exists(edge_massflow_file):
            shutil.copy2(edge_massflow_file, tn_results_dir / os.path.basename(edge_massflow_file))
            self.logger.info(f"Copied edge mass flow file: {os.path.basename(edge_massflow_file)}")
        else:
            self.logger.warning(f"Edge mass flow file not found: {edge_massflow_file}")

        # 3. Node mass flow file
        node_massflow_file = temp_locator.get_thermal_network_layout_massflow_nodes_file(self.network_type, '')
        if os.path.exists(node_massflow_file):
            shutil.copy2(node_massflow_file, tn_results_dir / os.path.basename(node_massflow_file))
            self.logger.info(f"Copied node mass flow file: {os.path.basename(node_massflow_file)}")
        else:
            self.logger.warning(f"Node mass flow file not found: {node_massflow_file}")

        # 4. Edge velocities file
        edge_velocity_file = temp_locator.get_thermal_network_velocity_edges_file(self.network_type, '')
        if os.path.exists(edge_velocity_file):
            shutil.copy2(edge_velocity_file, tn_results_dir / os.path.basename(edge_velocity_file))
            self.logger.info(f"Copied edge velocities file: {os.path.basename(edge_velocity_file)}")
        else:
            self.logger.warning(f"Edge velocities file not found: {edge_velocity_file}")

        # 5. Node pressures file
        node_pressure_file = temp_locator.get_network_pressure_at_nodes(self.network_type, '')
        if os.path.exists(node_pressure_file):
            shutil.copy2(node_pressure_file, tn_results_dir / os.path.basename(node_pressure_file))
            self.logger.info(f"Copied node pressures file: {os.path.basename(node_pressure_file)}")
        else:
            self.logger.warning(f"Node pressures file not found: {node_pressure_file}")

        # 6. Total pressure drops file
        pressure_loss_system_file = temp_locator.get_network_total_pressure_drop_file(self.network_type, '')
        if os.path.exists(pressure_loss_system_file):
            shutil.copy2(pressure_loss_system_file, tn_results_dir / os.path.basename(pressure_loss_system_file))
            self.logger.info(f"Copied total pressure drops file: {os.path.basename(pressure_loss_system_file)}")
        else:
            self.logger.warning(f"Total pressure drops file not found: {pressure_loss_system_file}")

        # 7. Pumping energy requirements file
        pumping_energy_file = temp_locator.get_network_energy_pumping_requirements_file(self.network_type, '')
        if os.path.exists(pumping_energy_file):
            shutil.copy2(pumping_energy_file, tn_results_dir / os.path.basename(pumping_energy_file))
            self.logger.info(f"Copied pumping energy requirements file: {os.path.basename(pumping_energy_file)}")
        else:
            self.logger.warning(f"Pumping energy requirements file not found: {pumping_energy_file}")

        # 8. Substation pressure losses file
        substation_ploss_file = temp_locator.get_thermal_network_substation_ploss_file(self.network_type, '')
        if os.path.exists(substation_ploss_file):
            shutil.copy2(substation_ploss_file, tn_results_dir / os.path.basename(substation_ploss_file))
            self.logger.info(f"Copied substation pressure losses file: {os.path.basename(substation_ploss_file)}")
        else:
            self.logger.warning(f"Substation pressure losses file not found: {substation_ploss_file}")

        # 9. Linear pressure drops in edges file
        linear_pressure_drop_file = temp_locator.get_network_linear_pressure_drop_edges(self.network_type, '')
        if os.path.exists(linear_pressure_drop_file):
            shutil.copy2(linear_pressure_drop_file, tn_results_dir / os.path.basename(linear_pressure_drop_file))
            self.logger.info(f"Copied linear pressure drops file: {os.path.basename(linear_pressure_drop_file)}")
        else:
            self.logger.warning(f"Linear pressure drops file not found: {linear_pressure_drop_file}")

        # 10. Total thermal losses file
        thermal_loss_system_file = temp_locator.get_network_total_thermal_loss_file(self.network_type, '')
        if os.path.exists(thermal_loss_system_file):
            shutil.copy2(thermal_loss_system_file, tn_results_dir / os.path.basename(thermal_loss_system_file))
            self.logger.info(f"Copied total thermal losses file: {os.path.basename(thermal_loss_system_file)}")
        else:
            self.logger.warning(f"Total thermal losses file not found: {thermal_loss_system_file}")

        # 11. Edge thermal losses file
        thermal_loss_edges_file = temp_locator.get_network_thermal_loss_edges_file(self.network_type, '')
        if os.path.exists(thermal_loss_edges_file):
            shutil.copy2(thermal_loss_edges_file, tn_results_dir / os.path.basename(thermal_loss_edges_file))
            self.logger.info(f"Copied edge thermal losses file: {os.path.basename(thermal_loss_edges_file)}")
        else:
            self.logger.warning(f"Edge thermal losses file not found: {thermal_loss_edges_file}")

        # 12. Linear thermal losses in edges file
        linear_thermal_loss_edges_file = temp_locator.get_network_linear_thermal_loss_edges_file(self.network_type, '')
        if os.path.exists(linear_thermal_loss_edges_file):
            shutil.copy2(linear_thermal_loss_edges_file, tn_results_dir / os.path.basename(linear_thermal_loss_edges_file))
            self.logger.info(f"Copied linear thermal losses file: {os.path.basename(linear_thermal_loss_edges_file)}")
        else:
            self.logger.warning(f"Linear thermal losses file not found: {linear_thermal_loss_edges_file}")

        # 13. Edge pressure losses file
        pressure_loss_edges_file = temp_locator.get_thermal_network_pressure_losses_edges_file(self.network_type, '')
        if os.path.exists(pressure_loss_edges_file):
            shutil.copy2(pressure_loss_edges_file, tn_results_dir / os.path.basename(pressure_loss_edges_file))
            self.logger.info(f"Copied edge pressure losses file: {os.path.basename(pressure_loss_edges_file)}")
        else:
            self.logger.warning(f"Edge pressure losses file not found: {pressure_loss_edges_file}")

        # 14. Plant heat requirements file
        plant_heat_req_file = temp_locator.get_thermal_network_plant_heat_requirement_file(self.network_type, '')
        if os.path.exists(plant_heat_req_file):
            shutil.copy2(plant_heat_req_file, tn_results_dir / os.path.basename(plant_heat_req_file))
            self.logger.info(f"Copied plant heat requirements file: {os.path.basename(plant_heat_req_file)}")
        else:
            self.logger.warning(f"Plant heat requirements file not found: {plant_heat_req_file}")

        # 15. Supply node temperatures file
        supply_temp_file = temp_locator.get_network_temperature_supply_nodes_file(self.network_type, '')
        if os.path.exists(supply_temp_file):
            shutil.copy2(supply_temp_file, tn_results_dir / os.path.basename(supply_temp_file))
            self.logger.info(f"Copied supply node temperatures file: {os.path.basename(supply_temp_file)}")
        else:
            self.logger.warning(f"Supply node temperatures file not found: {supply_temp_file}")

        # 16. Return node temperatures file
        return_temp_file = temp_locator.get_network_temperature_return_nodes_file(self.network_type, '')
        if os.path.exists(return_temp_file):
            shutil.copy2(return_temp_file, tn_results_dir / os.path.basename(return_temp_file))
            self.logger.info(f"Copied return node temperatures file: {os.path.basename(return_temp_file)}")
        else:
            self.logger.warning(f"Return node temperatures file not found: {return_temp_file}")

        # 17. Plant temperatures file
        plant_temp_file = temp_locator.get_network_temperature_plant(self.network_type, '')
        if os.path.exists(plant_temp_file):
            shutil.copy2(plant_temp_file, tn_results_dir / os.path.basename(plant_temp_file))
            self.logger.info(f"Copied plant temperatures file: {os.path.basename(plant_temp_file)}")
        else:
            self.logger.warning(f"Plant temperatures file not found: {plant_temp_file}")

        # Additional files needed for DTN expansion optimization
        # Edge list file
        edge_list_file = temp_locator.get_thermal_network_edge_list_file(self.network_type, '')
        if os.path.exists(edge_list_file):
            shutil.copy2(edge_list_file, tn_results_dir / os.path.basename(edge_list_file))
            self.logger.info(f"Copied edge list file: {os.path.basename(edge_list_file)}")
        else:
            self.logger.warning(f"Edge list file not found: {edge_list_file}")

        # Node types file
        node_types_file = temp_locator.get_thermal_network_node_types_csv_file(self.network_type, '')
        if os.path.exists(node_types_file):
            shutil.copy2(node_types_file, tn_results_dir / os.path.basename(node_types_file))
            self.logger.info(f"Copied node types file: {os.path.basename(node_types_file)}")
        else:
            self.logger.warning(f"Node types file not found: {node_types_file}")

        # Edge-node matrix file
        edge_node_file = temp_locator.get_thermal_network_edge_node_matrix_file(self.network_type, '')
        if os.path.exists(edge_node_file):
            shutil.copy2(edge_node_file, tn_results_dir / os.path.basename(edge_node_file))
            self.logger.info(f"Copied edge-node matrix file: {os.path.basename(edge_node_file)}")
        else:
            self.logger.warning(f"Edge-node matrix file not found: {edge_node_file}")

        self.logger.info(f"All thermal network results copied to {tn_results_dir}")
        self.tn_results_dir = tn_results_dir

    def _copy_all_building_demand_files(self, temp_demand_dir):
        """
        Copy all building demand files to the temporary scenario.

        Args:
            temp_demand_dir: Path to the demand directory in the temporary scenario
        """
        self.logger.info("Copying all building demand files to temporary scenario")

        # Get all buildings in the scenario
        total_demand = pd.read_csv(self.locator.get_total_demand())
        all_buildings = total_demand['name'].values

        # Copy modified demand files for buildings in the last cluster
        for building, files in self.modified_demand_files.items():
            self.logger.info(f"Copying modified demand file for building {building}")
            shutil.copy2(files['modified'], temp_demand_dir / f"{building}.csv")

        # Copy original demand files for all other buildings
        modified_buildings = set(self.modified_demand_files.keys())
        for building in all_buildings:
            if building not in modified_buildings:
                original_file = self.locator.get_demand_results_file(building)
                if os.path.exists(original_file):
                    self.logger.info(f"Copying original demand file for building {building}")
                    shutil.copy2(original_file, temp_demand_dir / f"{building}.csv")
                else:
                    self.logger.warning(f"Original demand file not found for building {building}")

    def _update_total_demand_files(self, temp_demand_dir):
        """
        Generate updated Total_demand.csv and Total_demand_hourly.csv based on all building demand files.

        This method ensures that the Total_demand.csv file contains all required columns,
        including QH_sys_MWhyr which is needed by the thermal network simulation.

        Args:
            temp_demand_dir: Path to the demand directory in the temporary scenario
        """
        self.logger.info("Generating updated Total_demand.csv and Total_demand_hourly.csv")

        # Get all building demand files in the temporary scenario
        building_files = list(temp_demand_dir.glob("*.csv"))
        building_names = [f.stem for f in building_files if
                          f.stem != "Total_demand" and f.stem != "Total_demand_hourly"]

        # Generate updated Total_demand.csv
        self.logger.info("Generating updated Total_demand.csv")

        # Create a DataFrame to store the yearly aggregated values
        total_demand_df = pd.DataFrame()

        # Process each building file
        for building in building_names:
            building_file = temp_demand_dir / f"{building}.csv"
            building_df = pd.read_csv(building_file)

            # Create a row for this building in the total_demand_df
            building_row = {'name': building}

            # Add metadata columns if they exist
            for col in ['Af_m2', 'Aroof_m2', 'GFA_m2', 'Aocc_m2', 'people0']:
                if col in building_df.columns:
                    building_row[col] = building_df[col].iloc[0]

            # Calculate yearly sums for hourly values and convert to MWh/yr
            # Look for columns ending with _kWh and convert to _MWhyr
            for col in building_df.columns:
                if col.endswith('_kWh'):
                    base_col = col[:-4]  # Remove _kWh suffix
                    yearly_sum = building_df[col].sum() / 1000  # Convert kWh to MWh
                    building_row[f"{base_col}_MWhyr"] = yearly_sum

                # Also include peak values (columns ending with 0_kW)
                elif col.endswith('0_kW'):
                    building_row[col] = building_df[col].iloc[0]

            # Add the row to the total demand DataFrame
            total_demand_df = pd.concat([total_demand_df, pd.DataFrame([building_row])], ignore_index=True)

        # Ensure QH_sys_MWhyr column exists (required by thermal network simulation)
        if 'QH_sys_MWhyr' not in total_demand_df.columns:
            # If QH_sys_kWh was not in the building files, try to calculate it from components
            if 'Qhs_sys_MWhyr' in total_demand_df.columns and 'Qww_sys_MWhyr' in total_demand_df.columns:
                total_demand_df['QH_sys_MWhyr'] = total_demand_df['Qhs_sys_MWhyr'] + total_demand_df['Qww_sys_MWhyr']
                self.logger.info("Created QH_sys_MWhyr column from Qhs_sys_MWhyr and Qww_sys_MWhyr")
            else:
                # If we can't calculate it, add a column with zeros (better than missing)
                total_demand_df['QH_sys_MWhyr'] = 0.0
                self.logger.warning("Could not calculate QH_sys_MWhyr, adding column with zeros")

        # Ensure QC_sys_MWhyr column exists (required by thermal network simulation)
        if 'QC_sys_MWhyr' not in total_demand_df.columns:
            # If QC_sys_kWh was not in the building files, try to calculate it from components
            if 'Qcs_sys_MWhyr' in total_demand_df.columns:
                # For cooling, we might also have data center and refrigeration cooling
                cooling_cols = ['Qcs_sys_MWhyr']
                if 'Qcdata_sys_MWhyr' in total_demand_df.columns:
                    cooling_cols.append('Qcdata_sys_MWhyr')
                if 'Qcre_sys_MWhyr' in total_demand_df.columns:
                    cooling_cols.append('Qcre_sys_MWhyr')

                total_demand_df['QC_sys_MWhyr'] = total_demand_df[cooling_cols].sum(axis=1)
                self.logger.info(f"Created QC_sys_MWhyr column from {cooling_cols}")
            else:
                # If we can't calculate it, add a column with zeros (better than missing)
                total_demand_df['QC_sys_MWhyr'] = 0.0
                self.logger.warning("Could not calculate QC_sys_MWhyr, adding column with zeros")

        # Save updated Total_demand.csv
        total_demand_df.to_csv(temp_demand_dir / "Total_demand.csv", index=False, float_format='%.3f', na_rep='nan')
        self.logger.info("Updated Total_demand.csv generated successfully")

        # Generate updated Total_demand_hourly.csv
        self.logger.info("Generating updated Total_demand_hourly.csv")

        # This approach follows the same logic as in cea.demand.demand_writers.YearlyDemandWriter.write_aggregate_hourly
        aggregated_hourly_results_df = pd.DataFrame()

        for i, building in enumerate(building_names):
            building_file = temp_demand_dir / f"{building}.csv"
            hourly_results_per_building = pd.read_csv(building_file).set_index('date')
            if i == 0:
                aggregated_hourly_results_df = hourly_results_per_building
            else:
                aggregated_hourly_results_df += hourly_results_per_building

        # Remove columns that shouldn't be summed
        if 'name' in aggregated_hourly_results_df.columns:
            aggregated_hourly_results_df = aggregated_hourly_results_df.drop(columns=['name'])
        if 'x_int' in aggregated_hourly_results_df.columns:
            aggregated_hourly_results_df = aggregated_hourly_results_df.drop(columns=['x_int'])

        # Save updated Total_demand_hourly.csv
        aggregated_hourly_results_df.to_csv(temp_demand_dir / "Total_demand_hourly.csv",
                                            index=True, float_format='%.3f', na_rep='nan')
        self.logger.info("Updated Total_demand_hourly.csv generated successfully")

    def _cleanup_temp_scenario(self, temp_scenario_dir):
        """
        Clean up the temporary scenario directory.
        This function has been modified to NOT delete any files.

        Args:
            temp_scenario_dir (str): Path to the temporary scenario directory
        """
        self.logger.info(f"Temporary scenario directory preserved (not cleaning up): {temp_scenario_dir}")
        # No cleanup is performed to preserve all files

    def _check_required_files(self, temp_scenario_dir, check_part="both"):
        """
        Check if all required files for thermal network simulation are present in the temporary scenario.

        Args:
            temp_scenario_dir (str): Path to the temporary scenario directory
            check_part (str): Which part to check for required files: "part2", "part3", or "both"

        Returns:
            bool: True if all required files are present, False otherwise
        """
        self.logger.info(f"Checking if all required files for thermal network simulation are present (checking {check_part})")

        # Create a temporary locator for the temporary scenario
        temp_locator = cea.inputlocator.InputLocator(temp_scenario_dir)

        # Initialize list of required files
        required_files = []

        # Add required files for Part 2 (thermal network simulation)
        if check_part == "part2" or check_part == "both":
            part2_files = [
                # Network layout files
                (temp_locator.get_network_layout_edges_shapefile(self.network_type), "Network layout edges shapefile"),
                (temp_locator.get_network_layout_nodes_shapefile(self.network_type), "Network layout nodes shapefile"),

                # Database files
                (temp_locator.get_database_components_distribution_thermal_grid(), "THERMAL_GRID.csv"),
                (temp_locator.get_db4_components_conversion_conversion_technology_csv('HEAT_EXCHANGERS'), "HEAT_EXCHANGERS.csv"),
                (temp_locator.get_db4_components_conversion_conversion_technology_csv('HYDRAULIC_PUMPS'), "HYDRAULIC_PUMPS.csv"),
                (temp_locator.get_db4_components_conversion_conversion_technology_csv('VAPOR_COMPRESSION_CHILLERS'), "VAPOR_COMPRESSION_CHILLERS.csv"),
                (temp_locator.get_db4_components_conversion_conversion_technology_csv('COOLING_TOWERS'), "COOLING_TOWERS.csv"),

                # Weather file
                (temp_locator.get_weather_file(), "Weather file"),

                # Demand files
                (temp_locator.get_total_demand(), "Total demand file"),

                # Thermal network files
                (os.path.join(temp_locator.get_thermal_network_folder(), self.network_type, "edges.shp"), "Thermal network edges shapefile"),
                (os.path.join(temp_locator.get_thermal_network_folder(), self.network_type, "nodes.shp"), "Thermal network nodes shapefile")
            ]
            required_files.extend(part2_files)

            # Check if all building demand files exist
            if os.path.exists(temp_locator.get_total_demand()):
                total_demand = pd.read_csv(temp_locator.get_total_demand())
                all_buildings = total_demand['name'].values
                for building in all_buildings:
                    building_file = temp_locator.get_demand_results_file(building)
                    required_files.append((building_file, f"Building demand file for {building}"))

        # Add required files for Part 3 (thermal network costs)
        if check_part == "part3" or check_part == "both":
            # These files are created by Part 2 and required by Part 3
            part3_files = [
                (temp_locator.get_thermal_network_edge_list_file(self.network_type, ''), "Thermal network edge list file"),
                (temp_locator.get_nominal_edge_mass_flow_csv_file(self.network_type, ''), "Nominal edge mass flow file"),
                (temp_locator.get_thermal_network_layout_massflow_edges_file(self.network_type, ''), "Edge mass flow file"),
                (temp_locator.get_thermal_network_layout_massflow_nodes_file(self.network_type, ''), "Node mass flow file"),
                (temp_locator.get_thermal_network_node_types_csv_file(self.network_type, ''), "Node types file"),
                (temp_locator.get_thermal_network_plant_heat_requirement_file(self.network_type, ''), "Plant heat requirement file")
            ]
            required_files.extend(part3_files)

        # Check if all required files exist
        missing_files = []
        for file_path, file_description in required_files:
            if not os.path.exists(file_path):
                missing_files.append((file_path, file_description))
                self.logger.warning(f"Required file not found: {file_path} ({file_description})")

        if missing_files:
            self.logger.error(f"Found {len(missing_files)} missing files required for thermal network simulation")
            return False
        else:
            self.logger.info("All required files for thermal network simulation are present")
            return True

    def rerun_thermal_network_simulation(self):
        """
        Rerun the thermal network simulation with modified demands.
        """
        self.logger.info("Rerunning thermal network simulation with modified demands")

        # Create a temporary scenario directory with modified demand files
        temp_scenario_dir = self._create_temp_scenario_with_modified_demands()

        # Check if all required files for Part 2 are present before starting the simulation
        self.logger.info("Checking required files for thermal network simulation (Part 2)")
        if not self._check_required_files(temp_scenario_dir, check_part="part2"):
            self.logger.error("Cannot proceed with thermal network simulation due to missing required files")
            self.logger.error("Please check the logs for details on missing files")
            return

        # Create configs that point to the temporary scenario
        tn_config = self._create_config_copy()
        tn_config.scenario = temp_scenario_dir
        tn_config.thermal_network.network_type = self.network_type

        # Run thermal network simulation on the temporary scenario
        self.logger.info("Starting thermal network simulation (Part 2)")
        self.logger.info("Detailed progress messages will be displayed in the console")
        thermal_network_simulation_main(tn_config)
        self.logger.info("Thermal network simulation (Part 2) completed successfully")

        # Check if all required files for Part 3 are present before starting the costs calculation
        self.logger.info("Checking required files for thermal network costs calculation (Part 3)")
        if not self._check_required_files(temp_scenario_dir, check_part="part3"):
            self.logger.error("Cannot proceed with thermal network costs calculation due to missing required files")
            self.logger.error("Please check the logs for details on missing files")
            return

        # Run thermal network costs calculation
        self.logger.info("Starting thermal network costs calculation (Part 3)")
        self.logger.info("Detailed progress messages will be displayed in the console")
        tnc_config = self._create_config_copy()
        tnc_config.scenario = temp_scenario_dir
        tnc_config.thermal_network_costs.network_type = self.network_type
        thermal_network_costs_main(tnc_config)
        self.logger.info("Thermal network costs calculation (Part 3) completed successfully")

        # Copy results back to the dynamic DTN folder
        self.logger.info("Copying thermal network results to dedicated folder")
        self._copy_results_from_temp_scenario(temp_scenario_dir)

        # Clean up temporary scenario
        self._cleanup_temp_scenario(temp_scenario_dir)

    def rerun_dtn_optimization(self):
        """
        Rerun the DTN optimization with the updated thermal network results.

        Returns:
            New optimization results
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
        self.logger.info(f"New optimization results saved with {len(new_results['genome'] if isinstance(new_results, dict) else new_results)} clusters")

        return new_results

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
        self.logger.info(f"Created comparison results directory: {comparison_dir}")

        # Extract connection sequences
        self.logger.info("Extracting connection sequences from original and new optimization results")
        original_sequence = self._extract_connection_sequence(self.original_results)
        new_sequence = self._extract_connection_sequence(self.new_results)
        self.logger.info(f"Original sequence has {len(original_sequence['flat'])} clusters")
        self.logger.info(f"New sequence has {len(new_sequence['flat'])} clusters")

        # Compare the sequences
        self.logger.info("Comparing connection sequences")
        sequence_changes = self._compare_sequences(original_sequence, new_sequence)
        self.logger.info(f"Sequence comparison result: {sequence_changes}")

        # Compare objective values
        self.logger.info("Comparing objective values")
        objective_changes = self._compare_objectives(self.original_results, self.new_results)
        self.logger.info(f"Objective comparison result: {objective_changes}")

        # Create a summary DataFrame
        self.logger.info("Creating summary of comparison results")
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
        self.logger.info(f"Saving summary to: {summary_file}")
        summary_df.to_csv(str(summary_file), index=False)

        # Create detailed comparison of connection sequences
        self.logger.info("Creating detailed comparison of connection sequences")
        sequence_comparison = {
            'cluster_id': list(range(len(original_sequence['flat']))),
            'original_phase': original_sequence['flat'],
            'new_phase': new_sequence['flat']
        }
        sequence_df = pd.DataFrame(sequence_comparison)
        sequence_df['phase_change'] = sequence_df['new_phase'] - sequence_df['original_phase']

        # Count clusters that changed phases
        changed_clusters = sequence_df[sequence_df['phase_change'] != 0]
        self.logger.info(f"Found {len(changed_clusters)} clusters that changed phases")

        # Save the sequence comparison
        sequence_file = comparison_dir / "sequence_comparison.csv"
        self.logger.info(f"Saving sequence comparison to: {sequence_file}")
        sequence_df.to_csv(str(sequence_file), index=False)

        # Create visualizations
        self.logger.info("Creating visualizations of comparison results")
        self._create_visualizations(sequence_df, comparison_dir)

        self.logger.info(f"Comparison results saved to {comparison_dir}")
        return summary_df

    def _parse_list_param(self, param_str):
        """Parse a comma-separated string parameter into a list of values."""
        if not param_str or param_str.strip() == '':
            return None
        return [float(x) for x in param_str.split(',')]

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
        Run the Dynamic DTN Optimization workflow up to thermal network simulation.

        This method stops after completing the thermal network simulation (Part 2 and 3)
        and does not proceed to rerun DTN optimization or results comparison.
        Those steps will be handled separately by dynamic_dtn_rerun_optimization.py
        and dynamic_dtn_result_comparison.py.

        Returns:
            None
        """
        self.logger.info("Starting Dynamic DTN Optimization workflow")
        self.logger.info(f"Network type: {self.network_type}")
        self.logger.info(f"Demand reduction parameters: Heating={self.heating_reduction*100}%, Cooling={self.cooling_reduction*100}%, DHW={self.dhw_reduction*100}%, Electricity={self.electricity_reduction*100}%")
        self.logger.info(f"Number of last clusters to modify: {self.num_last_clusters}")

        # Step 1: Check if DTN optimization results exist
        self.logger.info("STEP 1: Checking if DTN optimization results exist")
        results_exist, results_file = self.check_dtn_optimization_results()
        if not results_exist:
            error_msg = (
                "DTN expansion optimization results not found. Please run the DTN expansion optimization first "
                "using the 'dtn-expansion-optimization' module with the same network type."
            )
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        # Step 2: Load the original DTN optimization results
        self.logger.info("STEP 2: Loading original DTN optimization results")
        original_results = self.load_original_optimization_results(results_file)
        self.logger.info(f"Original optimization results loaded with {len(original_results['genome'])} clusters")

        # Step 3: Identify the last clusters to be connected
        self.logger.info("STEP 3: Identifying the last clusters to be connected")
        last_clusters = self.identify_last_clusters(original_results)
        self.logger.info(f"Identified {len(last_clusters)} clusters to modify: {last_clusters}")

        # Step 4: Modify the demand of buildings in the last clusters
        self.logger.info("STEP 4: Modifying the demand of buildings in the last clusters")
        self.modify_building_demands(last_clusters)

        # Step 5: Rerun the thermal network simulation with modified demands
        self.logger.info("STEP 5: Rerunning the thermal network simulation with modified demands")
        self.rerun_thermal_network_simulation()

        # Workflow stops here - DTN optimization rerunning and results comparison
        # will be handled separately by dynamic_dtn_rerun_optimization.py and dynamic_dtn_result_comparison.py
        self.logger.info("Dynamic DTN Optimization workflow completed successfully (up to thermal network simulation)")
        self.logger.info(f"Results saved to: {Path(self.locator.get_optimization_results_folder()) / 'dynamic_dtn_optimization'}")
        return None

def main(config):
    """
    Run the Dynamic DTN Optimization module.

    This function runs the Dynamic DTN Optimization workflow up to thermal network simulation
    (Part 2 and 3) and does not proceed to rerun DTN optimization or results comparison.
    Those steps will be handled separately by dynamic_dtn_rerun_optimization.py
    and dynamic_dtn_result_comparison.py.

    Args:
        config: CEA Configuration object
    """
    # Set up logging
    logger = log()
    logger.info("="*80)
    logger.info("Starting Dynamic DTN Optimization module")
    logger.info(f"Scenario: {config.scenario}")
    logger.info(f"Network type: {config.dynamic_dtn_optimization.network_type}")
    logger.info("="*80)

    start_time = time.time()

    locator = cea.inputlocator.InputLocator(config.scenario)

    # Create and run the Dynamic DTN Optimizer
    logger.info("Initializing Dynamic DTN Optimizer")
    optimizer = DynamicDTNOptimizer(locator, config)

    # Run the optimization up to thermal network simulation
    optimizer.run()

    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    time_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"

    logger.info("="*80)
    logger.info(f"Dynamic DTN Optimization completed successfully (up to thermal network simulation) in {time_str}")
    logger.info(f"Results saved to: {Path(locator.get_optimization_results_folder()) / 'dynamic_dtn_optimization'}")
    logger.info("="*80)

    print("Dynamic DTN Optimization completed successfully (up to thermal network simulation).")
    print(f"Total runtime: {time_str}")
    print("To continue with DTN optimization rerunning, use dynamic_dtn_rerun_optimization.py")
    print("To compare results, use dynamic_dtn_result_comparison.py")

    return None

if __name__ == '__main__':
    args = parse_args()
    config = cea.config.Configuration(args.config)
    config.scenario = args.scenario
    main(config)
