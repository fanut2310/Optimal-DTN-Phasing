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
from pathlib import Path
from typing import Dict, List, Optional, Union

import cea.config
import cea.inputlocator
from cea.technologies.thermal_network.thermal_network import main as thermal_network_simulation_main
from cea.technologies.thermal_network_costs.thermal_network_costs_new import main as thermal_network_costs_main


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
        # Retrofit reductions (fractions)
        self.heating_reduction = config.dynamic_dtn_optimization.heating_demand_reduction / 100.0
        self.cooling_reduction = config.dynamic_dtn_optimization.cooling_demand_reduction / 100.0
        self.dhw_reduction = config.dynamic_dtn_optimization.dhw_demand_reduction / 100.0
        self.electricity_reduction = config.dynamic_dtn_optimization.electricity_demand_reduction / 100.0

        # New options: early retrofit and last densification
        self.apply_early_retrofit = bool(getattr(config.dynamic_dtn_optimization, 'apply_early_retrofit', True))
        self.num_early_clusters_retrofit = int(getattr(config.dynamic_dtn_optimization, 'num_early_clusters_retrofit', 1))
        self.apply_last_densification = bool(getattr(config.dynamic_dtn_optimization, 'apply_last_densification', False))
        self.num_last_clusters_densify = int(getattr(config.dynamic_dtn_optimization, 'num_last_clusters_densify', 1))
        self.densification_pct = getattr(config.dynamic_dtn_optimization, 'densification_percent', 0) / 100.0

        # Deprecation warning for legacy parameter if present
        if hasattr(config.dynamic_dtn_optimization, 'num_last_clusters'):
            try:
                _ = config.dynamic_dtn_optimization.num_last_clusters
                # Log warning but ignore
                log().warning("Parameter 'dynamic-dtn-optimization:num-last-clusters' is deprecated and ignored. Use the new retrofit/densification options instead.")
            except Exception:
                pass

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

        # Parse phase_*_clusters columns without relying on config
        phase_cols = [c for c in results_df.columns if c.startswith('phase_') and c.endswith('_clusters')]
        # Sort by the integer after 'phase_' if possible
        try:
            phase_cols.sort(key=lambda c: int(c.split('_')[1]))
        except Exception:
            pass  # leave as-is if parsing fails

        parsed_any_phase = False
        for phase_col in phase_cols:
            try:
                solution[phase_col] = eval(best_row[phase_col])
                parsed_any_phase = True
            except Exception:
                self.logger.debug(f"Could not parse optional column {phase_col} from results")
        if not parsed_any_phase and phase_cols:
            self.logger.info("No phase_*_clusters metadata parsed from results (optional). Proceeding without them.")

        self.original_results = solution
        return solution

    def identify_early_clusters(self, results, k: int) -> List[int]:
        """Select EARLY clusters to retrofit: phases >= 2 (exclude phase 0 & 1), ascending phase order. Exclude cluster 0."""
        genome = results['genome']
        candidates = []
        for i, phase in enumerate(genome):
            if not isinstance(phase, (int, float)):
                continue
            # Map genome index to actual cluster ID if testing_clusters is available
            if self.testing_clusters and i < len(self.testing_clusters):
                cluster_id = self.testing_clusters[i]
            else:
                cluster_id = i + 1
            if cluster_id == 0:
                continue
            if phase >= 2:
                candidates.append((cluster_id, phase))
        # Earliest phases first
        candidates.sort(key=lambda x: x[1])
        k = max(0, int(k))
        selected = [c for c, _ in candidates[:k]]
        if len(selected) < k:
            self.logger.warning(f"Requested {k} early clusters for retrofit but only {len(selected)} available.")
        return selected

    def identify_last_clusters_for_densification(self, results, k: int) -> List[int]:
        """Select LAST clusters to densify: phases > 0 (connected), descending phase order. Exclude cluster 0."""
        genome = results['genome']
        candidates = []
        for i, phase in enumerate(genome):
            if not isinstance(phase, (int, float)):
                continue
            if self.testing_clusters and i < len(self.testing_clusters):
                cluster_id = self.testing_clusters[i]
            else:
                cluster_id = i + 1
            if cluster_id == 0:
                continue
            if phase > 0:
                candidates.append((cluster_id, phase))
        # Last phases first
        candidates.sort(key=lambda x: x[1], reverse=True)
        k = max(0, int(k))
        selected = [c for c, _ in candidates[:k]]
        if len(selected) < k:
            self.logger.warning(f"Requested {k} last clusters for densification but only {len(selected)} available.")
        return selected

    def modify_building_demands_and_built_form(self, retrofit_clusters: List[int], densify_clusters: List[int]):
        """
        Modify building demand time series and built form based on selected clusters:
        - Retrofit: reduce selected end-uses for early-phase clusters (exclude phases 0 & 1).
        - Densify: scale ALL *_kWh series and area-related metadata (e.g., GFA_m2, Af_m2, TFA_m2, floors_ag)
                   for last-phase clusters by (1 + densification_pct).
        Saves modified per-building CSVs and records mapping in self.modified_demand_files.
        """
        self.logger.info("Modifying building demands and built form for selected clusters")

        # Load the cluster assignment file to get buildings in each cluster
        cluster_assignment_file_path = self.locator.get_dtn_cluster_assignment_file()
        self.logger.info(f"Cluster assignment file path: {cluster_assignment_file_path}")
        cluster_assignment_file = Path(cluster_assignment_file_path)
        if not cluster_assignment_file.exists():
            self.logger.error(f"Cluster assignment file not found: {cluster_assignment_file}")
            raise FileNotFoundError(f"Cluster assignment file not found: {cluster_assignment_file}")

        cluster_df = pd.read_csv(os.path.normpath(str(cluster_assignment_file)))

        # Resolve buildings per action
        retrofit_buildings = set(cluster_df[cluster_df['cluster'].isin(retrofit_clusters)]['name'].tolist())
        densify_buildings = set(cluster_df[cluster_df['cluster'].isin(densify_clusters)]['name'].tolist())

        # Persist target building sets for later use (e.g., updating Total_demand.csv areas)
        self._retrofit_buildings = set(retrofit_buildings)
        self._densify_buildings = set(densify_buildings)

        self.logger.info(f"Buildings to RETROFIT: {len(retrofit_buildings)}")
        if retrofit_buildings:
            self.logger.info(f"RETROFIT targets ({len(retrofit_buildings)} buildings): {', '.join(sorted(retrofit_buildings))}")
            self.logger.info(f"Retrofit reductions: Heating={self.heating_reduction*100:.1f}%, Cooling={self.cooling_reduction*100:.1f}%, DHW={self.dhw_reduction*100:.1f}%, Electricity={self.electricity_reduction*100:.1f}%")
        self.logger.info(f"Buildings to DENSIFY: {len(densify_buildings)} (factor: {1.0 + self.densification_pct:.3f})")
        if densify_buildings:
            self.logger.info(f"DENSIFY targets ({len(densify_buildings)} buildings): {', '.join(sorted(densify_buildings))}")
            self.logger.info(f"Densification percent: {self.densification_pct*100:.1f}% (applies to all *_kWh and metadata columns Af_m2/Aroof_m2/GFA_m2/Aocc_m2/people0)")

        # Create output directory
        modified_demand_dir = Path(self.locator.get_optimization_results_folder()) / "dynamic_dtn_optimization" / "modified_demands"
        modified_demand_dir.mkdir(parents=True, exist_ok=True)

        modified_demand_files: Dict[str, Dict[str, str]] = {}

        # Precompute factors
        down_factors = {
            'heating': max(0.0, 1.0 - self.heating_reduction),
            'cooling': max(0.0, 1.0 - self.cooling_reduction),
            'dhw': max(0.0, 1.0 - self.dhw_reduction),
            'elec': max(0.0, 1.0 - self.electricity_reduction)
        }
        up_factor = (1.0 + self.densification_pct)

        # Iterate target buildings (union)
        target_buildings = sorted(retrofit_buildings.union(densify_buildings))
        if not target_buildings:
            self.logger.warning("No buildings selected for modification.")
        for building in target_buildings:
            original_demand_file_path = self.locator.get_demand_results_file(building)
            original_demand_file = Path(original_demand_file_path)
            if not original_demand_file.exists():
                self.logger.error(f"Demand results file not found for building {building}: {original_demand_file}")
                raise FileNotFoundError(f"Demand results file not found for building {building}: {original_demand_file}")

            demand_df = pd.read_csv(os.path.normpath(str(original_demand_file)))

            if building in retrofit_buildings:
                # Reduce heating
                if self.heating_reduction > 0:
                    for col in list(demand_df.columns):
                        if (col.endswith('_kWh') and ('QH' in col or 'Qhs' in col)) or ('hs_' in col or col.endswith('_hs')):
                            if 'ww' not in col.lower() and 'hw' not in col.lower():
                                demand_df[col] = demand_df[col] * down_factors['heating']
                # Reduce cooling
                if self.cooling_reduction > 0:
                    for col in list(demand_df.columns):
                        if col.endswith('_kWh') and (col.startswith('QC') or col.startswith('Qc') or 'cs_' in col or col.endswith('_cs') or 'cdata' in col or 'cre' in col):
                            demand_df[col] = demand_df[col] * down_factors['cooling']
                # Reduce DHW
                if self.dhw_reduction > 0:
                    for col in list(demand_df.columns):
                        if col.endswith('_kWh') and ('QHW' in col or 'Qww' in col or col.startswith('ww_') or col.endswith('_ww')):
                            demand_df[col] = demand_df[col] * down_factors['dhw']
                # Reduce electricity (excluding PV)
                if self.electricity_reduction > 0:
                    for col in list(demand_df.columns):
                        if ('E_' in col or col.startswith('Ea') or col.startswith('Eve') or 'GRID' in col) and col.endswith('_kWh') and 'PV' not in col:
                            demand_df[col] = demand_df[col] * down_factors['elec']
                action = 'retrofit'
            else:
                # Densify: scale all *_kWh time series and area metadata
                cols_scaled = 0
                for col in demand_df.columns:
                    if col.endswith('_kWh'):
                        demand_df[col] = demand_df[col] * up_factor
                        cols_scaled += 1
                # Scale selected metadata if present in per-building CSV (only Af_m2 & GFA_m2)
                for meta_col in ['GFA_m2', 'Af_m2']:
                    if meta_col in demand_df.columns:
                        try:
                            demand_df.loc[:, meta_col] = demand_df[meta_col] * up_factor
                        except Exception:
                            pass
                self.logger.debug(f"{building}: densified {cols_scaled} *_kWh columns and scaled Af/GFA by {up_factor:.3f}")
                action = 'densify'

            # Save the modified demand file
            modified_demand_file = modified_demand_dir / f"{building}.csv"
            demand_df.to_csv(os.path.normpath(str(modified_demand_file)), index=False)

            modified_demand_files[building] = {
                'original': os.path.normpath(str(original_demand_file)),
                'modified': os.path.normpath(str(modified_demand_file)),
                'action': action
            }

        # store for copying into temp scenario later
        self.modified_demand_files = modified_demand_files
        self.logger.info(f"Completed modifications: retrofit={len(retrofit_buildings)}, densify={len(densify_buildings)}")
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
        temp_dir = Path(self.locator.get_dynamic_dtn_optimization_temp_scenario_folder())
        temp_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Created temporary scenario directory: {temp_dir}")

        # Create necessary subdirectories
        temp_demand_dir = Path(self.locator.get_dynamic_dtn_optimization_temp_scenario_demand_folder())
        temp_demand_dir.mkdir(parents=True, exist_ok=True)

        # Copy necessary files from original scenario
        self._copy_scenario_files(str(self.config.scenario), str(temp_dir))

        # Copy all building demand files (modified and original)
        self._copy_all_building_demand_files(temp_demand_dir)

        # Generate updated Total_demand.csv and Total_demand_hourly.csv
        self._update_total_demand_files(temp_demand_dir)

        self.logger.info(f"Temporary scenario created at: {temp_dir}")
        return str(temp_dir)

    def get_temp_locator(self):
        """
        Get a locator for the temporary scenario.

        This method creates the temporary scenario if it doesn't exist,
        and returns a standard InputLocator instance for the temporary scenario.
        """
        self.logger.info("Getting locator for temporary scenario")

        temp_dir = Path(self.locator.get_dynamic_dtn_optimization_temp_scenario_folder())
        if not temp_dir.exists():
            self.logger.info("Temporary scenario doesn't exist, creating it")
            temp_scenario_dir = self._create_temp_scenario_with_modified_demands()
        else:
            self.logger.info(f"Using existing temporary scenario at: {temp_dir}")
            temp_scenario_dir = str(temp_dir)

        # Use a standard InputLocator for the temp scenario
        temp_locator = cea.inputlocator.InputLocator(temp_scenario_dir)
        self.logger.info(f"Created InputLocator for temp scenario: {temp_scenario_dir}")

        return temp_locator

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
        (target_path / "outputs" / "data" / "emissions").mkdir(parents=True, exist_ok=True)
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

            # Add metadata columns if they exist (restrict to five required)
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

        # Apply densification to ONLY the five metadata columns before first save (if any densified buildings)
        up_factor = 1.0 + getattr(self, 'densification_pct', 0.0)
        five_meta_cols = ['Af_m2', 'Aroof_m2', 'GFA_m2', 'Aocc_m2', 'people0']
        if up_factor != 1.0 and hasattr(self, '_densify_buildings') and len(self._densify_buildings) > 0:
            try:
                mask = total_demand_df['name'].isin(list(self._densify_buildings))
                for col in five_meta_cols:
                    if col in total_demand_df.columns:
                        total_demand_df.loc[mask, col] = total_demand_df.loc[mask, col] * up_factor
                if 'people0' in total_demand_df.columns:
                    total_demand_df.loc[mask, 'people0'] = total_demand_df.loc[mask, 'people0'].round()
                self.logger.info(f"Applied densification factor {up_factor:.3f} to metadata in Total_demand.csv before first save.")
            except Exception as e:
                self.logger.warning(f"Could not apply densification scaling to metadata before first save: {e}")
        # Reorder columns: name, five metadata, then PV_MWhyr (if present), then others
        front_cols = ['name'] + [c for c in five_meta_cols if c in total_demand_df.columns]
        other_cols = [c for c in total_demand_df.columns if c not in front_cols]
        if 'PV_MWhyr' in other_cols:
            other_cols = ['PV_MWhyr'] + [c for c in other_cols if c != 'PV_MWhyr']
        total_demand_df = total_demand_df[front_cols + other_cols]
        # Save updated Total_demand.csv
        total_demand_df.to_csv(temp_demand_dir / "Total_demand.csv", index=False, float_format='%.3f', na_rep='nan')
        self.logger.info("Updated Total_demand.csv generated successfully")

        # After saving the initial Total_demand.csv, check for missing metadata columns
        self.logger.info("Checking for missing metadata columns in Total_demand.csv")

        # Required metadata columns that should be present (only five)
        required_metadata_columns = ['Af_m2', 'Aroof_m2', 'GFA_m2', 'Aocc_m2', 'people0']

        # Read the saved Total_demand.csv to check for missing columns
        temp_total_demand = pd.read_csv(temp_demand_dir / "Total_demand.csv")
        missing_columns = [col for col in required_metadata_columns if col not in temp_total_demand.columns]

        if missing_columns:
            self.logger.info(f"Some metadata columns missing in temporary Total_demand.csv: {missing_columns}")
            
            # Try to get the missing columns from the original Total_demand.csv
            try:
                original_total_demand = pd.read_csv(self.locator.get_total_demand())
                self.logger.info(f"Reading original Total_demand.csv to retrieve missing columns")
                
                # Check if the original file has the missing columns
                available_columns = [col for col in missing_columns if col in original_total_demand.columns]
                
                if available_columns:
                    self.logger.info(f"Found {len(available_columns)} columns in original Total_demand.csv: {available_columns}")
                    
                    # Create a mapping from building names in original to temporary
                    building_mapping = {}
                    for building in temp_total_demand['name']:
                        if building in original_total_demand['name'].values:
                            building_mapping[building] = building
                    
                    # Add missing columns from original to temporary
                    for col in available_columns:
                        self.logger.info(f"Adding column {col} from original Total_demand.csv")
                        
                        # Create a dictionary to map building names to column values
                        col_values = {}
                        for building in original_total_demand['name']:
                            if building in building_mapping:
                                col_values[building] = original_total_demand.loc[
                                    original_total_demand['name'] == building, col].values[0]
                        
                        # Add the column to the temporary Total_demand.csv
                        temp_total_demand[col] = temp_total_demand['name'].map(col_values)
                        
                        # Fill NaN values with appropriate defaults
                        if col == 'GFA_m2' or col == 'Af_m2':
                            # If one exists but not the other, copy the value
                            if 'GFA_m2' in temp_total_demand.columns and 'Af_m2' in temp_total_demand.columns:
                                mask = temp_total_demand[col].isna()
                                if col == 'GFA_m2' and 'Af_m2' in temp_total_demand.columns:
                                    temp_total_demand.loc[mask, col] = temp_total_demand.loc[mask, 'Af_m2']
                                elif col == 'Af_m2' and 'GFA_m2' in temp_total_demand.columns:
                                    temp_total_demand.loc[mask, col] = temp_total_demand.loc[mask, 'GFA_m2']
                            # Default value if still NaN
                            temp_total_demand[col].fillna(1000.0, inplace=True)
                        elif col == 'Aroof_m2':
                            # Estimate roof area as a fraction of floor area if available
                            if 'GFA_m2' in temp_total_demand.columns:
                                mask = temp_total_demand[col].isna()
                                temp_total_demand.loc[mask, col] = temp_total_demand.loc[mask, 'GFA_m2'] / 5  # Assuming 5 floors on average
                            elif 'Af_m2' in temp_total_demand.columns:
                                mask = temp_total_demand[col].isna()
                                temp_total_demand.loc[mask, col] = temp_total_demand.loc[mask, 'Af_m2'] / 5
                            else:
                                temp_total_demand[col].fillna(200.0, inplace=True)
                        elif col == 'Aocc_m2':
                            # Estimate occupied area as a percentage of floor area if available
                            if 'GFA_m2' in temp_total_demand.columns:
                                mask = temp_total_demand[col].isna()
                                temp_total_demand.loc[mask, col] = temp_total_demand.loc[mask, 'GFA_m2'] * 0.8  # Assuming 80% of GFA is occupied
                            elif 'Af_m2' in temp_total_demand.columns:
                                mask = temp_total_demand[col].isna()
                                temp_total_demand.loc[mask, col] = temp_total_demand.loc[mask, 'Af_m2'] * 0.8
                            else:
                                temp_total_demand[col].fillna(800.0, inplace=True)
                        elif col == 'people0':
                            # Estimate people based on floor area if available
                            if 'GFA_m2' in temp_total_demand.columns:
                                mask = temp_total_demand[col].isna()
                                temp_total_demand.loc[mask, col] = (temp_total_demand.loc[mask, 'GFA_m2'] / 25).round()  # Assuming 25 m² per person
                            elif 'Af_m2' in temp_total_demand.columns:
                                mask = temp_total_demand[col].isna()
                                temp_total_demand.loc[mask, col] = (temp_total_demand.loc[mask, 'Af_m2'] / 25).round()
                            else:
                                temp_total_demand[col].fillna(40, inplace=True)
                
                # For any columns still missing, add with default values
                still_missing = [col for col in missing_columns if col not in temp_total_demand.columns]
                if still_missing:
                    self.logger.warning(f"Still missing columns after checking original Total_demand.csv: {still_missing}")
                    for col in still_missing:
                        if col in ['GFA_m2', 'Af_m2', 'TFA_m2']:
                            temp_total_demand[col] = 1000.0
                        elif col == 'Aroof_m2':
                            temp_total_demand[col] = 200.0
                        elif col == 'Aocc_m2':
                            temp_total_demand[col] = 800.0
                        elif col == 'people0':
                            temp_total_demand[col] = 40
                        elif col == 'floors_ag':
                            temp_total_demand[col] = 5
                
                # Apply densification scaling to area-related fields after backfilling (if any densified buildings)
                up_factor = 1.0 + getattr(self, 'densification_pct', 0.0)
                if up_factor != 1.0 and hasattr(self, '_densify_buildings') and len(self._densify_buildings) > 0:
                    try:
                        mask = temp_total_demand['name'].isin(list(self._densify_buildings))
                        five_meta_cols = ['Af_m2', 'Aroof_m2', 'GFA_m2', 'Aocc_m2', 'people0']
                        for col in five_meta_cols:
                            if col in temp_total_demand.columns:
                                temp_total_demand.loc[mask, col] = temp_total_demand.loc[mask, col] * up_factor
                        if 'people0' in temp_total_demand.columns:
                            temp_total_demand.loc[mask, 'people0'] = temp_total_demand.loc[mask, 'people0'].round()
                        self.logger.info(f"Applied densification factor {up_factor:.3f} to metadata fields after backfilling.")
                    except Exception as e:
                        self.logger.warning(f"Could not apply densification scaling to metadata after backfilling: {e}")
                # Reorder again to keep the five metadata columns as 2nd–6th
                front_cols = ['name'] + [c for c in ['Af_m2', 'Aroof_m2', 'GFA_m2', 'Aocc_m2', 'people0'] if c in temp_total_demand.columns]
                other_cols = [c for c in temp_total_demand.columns if c not in front_cols]
                temp_total_demand = temp_total_demand[front_cols + other_cols]
                # Save the updated Total_demand.csv
                temp_total_demand.to_csv(temp_demand_dir / "Total_demand.csv", index=False, float_format='%.3f', na_rep='nan')
                self.logger.info("Updated Total_demand.csv with metadata backfill and ordering")
                
            except Exception as e:
                self.logger.error(f"Error retrieving metadata columns from original Total_demand.csv: {e}")
                self.logger.warning("Adding default values for missing metadata columns")
                
                # Add default values for missing columns (only five)
                for col in missing_columns:
                    if col in ['GFA_m2', 'Af_m2']:
                        temp_total_demand[col] = 1000.0
                    elif col == 'Aroof_m2':
                        temp_total_demand[col] = 200.0
                    elif col == 'Aocc_m2':
                        temp_total_demand[col] = 800.0
                    elif col == 'people0':
                        temp_total_demand[col] = 40
                
                # Apply densification scaling to metadata before saving (if any densified buildings)
                up_factor = 1.0 + getattr(self, 'densification_pct', 0.0)
                if up_factor != 1.0 and hasattr(self, '_densify_buildings') and len(self._densify_buildings) > 0:
                    try:
                        mask = temp_total_demand['name'].isin(list(self._densify_buildings))
                        for col in ['Af_m2', 'Aroof_m2', 'GFA_m2', 'Aocc_m2', 'people0']:
                            if col in temp_total_demand.columns:
                                temp_total_demand.loc[mask, col] = temp_total_demand.loc[mask, col] * up_factor
                        if 'people0' in temp_total_demand.columns:
                            temp_total_demand.loc[mask, 'people0'] = temp_total_demand.loc[mask, 'people0'].round()
                        self.logger.info(f"Applied densification factor {up_factor:.3f} to metadata in exception path.")
                    except Exception as e2:
                        self.logger.warning(f"Could not apply densification scaling to metadata in exception path: {e2}")
                
                # Reorder columns and save
                front_cols = ['name'] + [c for c in ['Af_m2', 'Aroof_m2', 'GFA_m2', 'Aocc_m2', 'people0'] if c in temp_total_demand.columns]
                other_cols = [c for c in temp_total_demand.columns if c not in front_cols]
                temp_total_demand = temp_total_demand[front_cols + other_cols]
                temp_total_demand.to_csv(temp_demand_dir / "Total_demand.csv", index=False, float_format='%.3f', na_rep='nan')
                self.logger.info("Updated Total_demand.csv with default metadata and ordering")
        else:
            self.logger.info("All required metadata columns are present in Total_demand.csv")

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
        self.logger.info(f"Demand reduction parameters (retrofit): Heating={self.heating_reduction*100}%, Cooling={self.cooling_reduction*100}%, DHW={self.dhw_reduction*100}%, Electricity={self.electricity_reduction*100}%")
        self.logger.info(f"Options: early_retrofit={self.apply_early_retrofit} (k={self.num_early_clusters_retrofit}), last_densification={self.apply_last_densification} (k={self.num_last_clusters_densify}, pct={self.densification_pct*100}%)")

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

        # Step 3: Select target clusters for retrofit and/or densification
        self.logger.info("STEP 3: Selecting target clusters for retrofit and/or densification")
        retrofit_targets: List[int] = []
        densify_targets: List[int] = []
        if self.apply_early_retrofit and self.num_early_clusters_retrofit > 0:
            retrofit_targets = self.identify_early_clusters(original_results, self.num_early_clusters_retrofit)
            self.logger.info(f"Retrofit early-phase clusters (exclude phase 0 & 1): {retrofit_targets}")
        if self.apply_last_densification and self.num_last_clusters_densify > 0:
            densify_targets = self.identify_last_clusters_for_densification(original_results, self.num_last_clusters_densify)
            self.logger.info(f"Densify last-phase clusters: {densify_targets}")

        # Resolve overlap: prioritize retrofit
        overlap = sorted(set(retrofit_targets).intersection(densify_targets))
        if overlap:
            self.logger.warning(f"Clusters selected for both retrofit and densification: {overlap}. Prioritizing retrofit; removing from densification targets.")
            densify_targets = [c for c in densify_targets if c not in overlap]

        if not retrofit_targets and not densify_targets:
            self.logger.warning("No target clusters selected. Nothing will be modified. Aborting.")
            return None

        # Step 4: Modify the demand and built form of buildings in target clusters
        self.logger.info("STEP 4: Modifying demands and built form for selected clusters")
        self.modify_building_demands_and_built_form(retrofit_targets, densify_targets)

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
