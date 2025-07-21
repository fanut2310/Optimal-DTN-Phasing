from __future__ import annotations

###############################################################################
# 1) STANDARD IMPORTS                                                        #
###############################################################################
import argparse
import logging
import os
import shutil
import time
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
import itertools
import random

import cea.config
import cea.inputlocator
from cea.constants import HEAT_CAPACITY_OF_WATER_JPERKGK
from cea.analysis.lca.operation import lca_operation
from deap import base, tools, algorithms, creator

# Setup function for the creator based on optimization mode
def setup_creator(multi_objective=False, objective_function='NPV', multi_objective_functions=None):
    """
    Set up the creator based on optimization mode and selected objectives
    
    Parameters:
    -----------
    multi_objective : bool
        Whether to use multi-objective optimization
    objective_function : str
        The objective function to use for single-objective optimization ('NPV', 'ROI', or 'emissions')
    multi_objective_functions : list
        List of objectives to use for multi-objective optimization
    """
    # Clear any existing creator classes to avoid conflicts
    if hasattr(creator, "FitnessMax") or hasattr(creator, "FitnessMulti"):
        del creator.FitnessMax
        if hasattr(creator, "FitnessMulti"):
            del creator.FitnessMulti
    if hasattr(creator, "Individual"):
        del creator.Individual

    if multi_objective:
        # Set up weights for multi-objective optimization
        weights = []

        # If no objectives specified, use default (NPV/ROI and emissions)
        if not multi_objective_functions or len(multi_objective_functions) == 0:
            multi_objective_functions = ['NPV', 'emissions']

        # Limit to 3 objectives maximum
        multi_objective_functions = multi_objective_functions[:3]

        # Set weights based on objectives (maximize NPV/ROI, minimize emissions and total_capex)
        for obj in multi_objective_functions:
            # Convert to lowercase for case-insensitive comparison
            obj_lower = obj.lower() if isinstance(obj, str) else obj
            if obj_lower in ['npv', 'roi']:
                weights.append(1.0)  # Maximize NPV/ROI
            elif obj_lower in ['emissions', 'total_capex']:
                weights.append(-1.0)  # Minimize emissions and total_capex

        # Create fitness class with appropriate weights
        creator.create("FitnessMulti", base.Fitness, weights=tuple(weights))
        creator.create("Individual", list, fitness=creator.FitnessMulti)
    else:
        # For single-objective optimization
        if objective_function.lower() in ['npv', 'roi']:
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
            creator.create("Individual", list, fitness=creator.FitnessMax)
        else:  # For emissions
            creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
            creator.create("Individual", list, fitness=creator.FitnessMax)

__author__ = "Fan Ut Chang"
__copyright__ = "Copyright 2023, Architecture and Building Systems - ETH Zurich"
__credits__ = ["Jimeno A. Fonseca", "Shanshan Hsieh", "Reynold Mok", "Mathias Niffeler"]
__license__ = "MIT"
__version__ = "0.1"
__maintainer__ = "Architecture and Building Systems - ETH Zurich"
__email__ = "cea@arch.ethz.ch"
__status__ = "Production"

def log(*args):
    """Log a message to the standard logger."""
    print(*args)
    logging.info(*args)


class CustomDTNRerunOptimizer:
    """
    A completely new implementation for Dynamic DTN Rerun Optimization.
    
    This class directly uses the files in the temp_scenario folder without
    relying on complex redirection mechanisms or inheritance from other modules.
    """
    
    def __init__(self, locator: cea.inputlocator.InputLocator, config: cea.config.Configuration):
        """
        Initialize the optimizer with direct paths.
        
        Parameters:
        -----------
        locator : cea.inputlocator.InputLocator
            The locator for the scenario
        config : cea.config.Configuration
            The configuration object
        """
        self.locator = locator
        self.config = config
        self.scenario_path = Path(locator.scenario)
        self.network_type = config.dynamic_dtn_optimization.network_type
        
        # Parse testing clusters from config
        testing_clusters_str = config.dtn_expansion_optimization.testing_clusters
        if testing_clusters_str:
            self.testing_clusters = [int(c.strip()) for c in testing_clusters_str.split(',') if c.strip()]
        else:
            self.testing_clusters = []
        
        # Define direct paths to all required folders
        self.dtn_folder = self.scenario_path / "outputs" / "data" / "optimization" / "dynamic_dtn_optimization"
        self.temp_scenario = self.dtn_folder / "temp_scenario"
        self.temp_demand_dir = self.temp_scenario / "outputs" / "data" / "demand"
        self.temp_thermal_network_dir = self.temp_scenario / "outputs" / "data" / "thermal-network"
        
        # Original paths for reference files
        self.original_dtn_dir = self.scenario_path / "outputs" / "data" / "optimization" / "dtn_expansion"
        
        # Output paths
        self.output_dir = self.dtn_folder / "rerun_results"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Verify all required directories exist
        self._verify_directories()
        
        # Log initialization
        log(f"Initialized CustomDTNRerunOptimizer with:")
        log(f"  Scenario: {self.scenario_path}")
        log(f"  Network type: {self.network_type}")
        log(f"  Testing clusters: {self.testing_clusters}")
        log(f"  Temp scenario: {self.temp_scenario}")
        log(f"  Output directory: {self.output_dir}")
    
    def _verify_directories(self):
        """Verify that all required directories exist."""
        if not self.temp_scenario.exists():
            raise FileNotFoundError(f"Temp scenario directory not found: {self.temp_scenario}")
        
        if not self.temp_demand_dir.exists():
            raise FileNotFoundError(f"Temp demand directory not found: {self.temp_demand_dir}")
        
        if not self.original_dtn_dir.exists():
            raise FileNotFoundError(f"Original DTN directory not found: {self.original_dtn_dir}")
    
    def _create_temp_scenario_locator(self):
        """
        Create a custom InputLocator for the temp scenario.
        
        Returns:
        --------
        cea.inputlocator.InputLocator
            InputLocator for the temp scenario
        """
        # Create a new InputLocator for the temp scenario
        temp_locator = cea.inputlocator.InputLocator(scenario=str(self.temp_scenario))
        
        log(f"Created InputLocator for temp scenario: {self.temp_scenario}")
        
        return temp_locator
    
    def _prepare_optimization_files(self):
        """
        Prepare all files needed for optimization.
        
        This includes copying cluster files, building supply file, and database files
        to the temp scenario and ensuring the Total_demand.csv file is correctly used.
        
        Returns:
        --------
        pd.DataFrame
            The Total_demand.csv data
        """
        # Create directory for optimization files in temp scenario
        temp_opt_dir = self.temp_scenario / "outputs" / "data" / "optimization" / "dtn_expansion"
        temp_opt_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy cluster files from original scenario
        for file_name in ["cluster_edges.csv", "cluster_nodes.csv"]:
            src_file = self.original_dtn_dir / file_name
            dst_file = temp_opt_dir / file_name
            
            if src_file.exists():
                log(f"Copying {file_name} to temp scenario")
                shutil.copy2(src_file, dst_file)
            else:
                raise FileNotFoundError(f"{file_name} not found at {src_file}")
        
        # Create a custom InputLocator for the temp scenario
        self.temp_locator = self._create_temp_scenario_locator()
        
        # Copy building supply file
        self._copy_building_supply_file()
        
        # Copy database files
        self._copy_database_files()
        
        # Verify Total_demand.csv exists in temp scenario
        total_demand_path = self.temp_demand_dir / "Total_demand.csv"
        if not total_demand_path.exists():
            raise FileNotFoundError(f"Total_demand.csv not found in temp scenario: {total_demand_path}")
        
        log(f"Using Total_demand.csv from temp scenario: {total_demand_path}")
        
        # Load and verify Total_demand.csv
        total_demand = pd.read_csv(total_demand_path)
        
        # Verify required columns exist
        required_columns = ['name']
        if self.network_type == 'DH':
            required_columns.extend(['Qhs_sys_MWhyr', 'Qww_sys_MWhyr'])
        else:  # DC
            required_columns.extend(['Qcs_sys_MWhyr', 'Qcre_sys_MWhyr', 'Qcdata_sys_MWhyr'])
        
        # Check if GFA_m2 is missing but Af_m2 is present (they are equivalent)
        if 'GFA_m2' not in total_demand.columns and 'Af_m2' in total_demand.columns:
            log("Renaming 'Af_m2' column to 'GFA_m2' for compatibility")
            total_demand['GFA_m2'] = total_demand['Af_m2']
        
        # If GFA_m2 is still missing, try to get it from the original demand file
        if 'GFA_m2' not in total_demand.columns:
            log("GFA_m2 column missing from temp scenario demand file. Attempting to retrieve from original scenario.")
            try:
                original_demand = pd.read_csv(self.locator.get_total_demand())
                if 'GFA_m2' in original_demand.columns:
                    # Create a mapping of building name to GFA_m2
                    gfa_mapping = original_demand.set_index('name')['GFA_m2'].to_dict()
                    # Add GFA_m2 column to temp demand
                    total_demand['GFA_m2'] = total_demand['name'].map(gfa_mapping)
                    log("Successfully added GFA_m2 column from original scenario")
                elif 'Af_m2' in original_demand.columns:
                    # Try with Af_m2 which is equivalent
                    gfa_mapping = original_demand.set_index('name')['Af_m2'].to_dict()
                    total_demand['GFA_m2'] = total_demand['name'].map(gfa_mapping)
                    log("Successfully added GFA_m2 column from original scenario's Af_m2 column")
            except Exception as e:
                log(f"Error retrieving GFA_m2 from original scenario: {e}")

        # Save the updated demand file
        total_demand.to_csv(total_demand_path, index=False)
        log("Updated Total_demand.csv with required columns")
        
        missing_columns = [col for col in required_columns if col not in total_demand.columns]
        if missing_columns:
            raise ValueError(f"Total_demand.csv is missing required columns: {missing_columns}")
        
        return total_demand
        
    def _copy_building_supply_file(self):
        """
        Copy the building supply file from the original scenario to the temp scenario.
        """
        # Get the building supply file path
        src_file = Path(self.locator.get_building_supply())
        dst_file = self.temp_scenario / "inputs" / "building-properties" / "supply.csv"
        
        # Create the destination directory if it doesn't exist
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy the file
        if src_file.exists():
            log(f"Copying building supply file to temp scenario")
            shutil.copy2(src_file, dst_file)
        else:
            raise FileNotFoundError(f"Building supply file not found at {src_file}")
    
    def _copy_database_files(self):
        """
        Copy the database files from the original scenario to the temp scenario.
        """
        # Create the database directories in the temp scenario
        temp_db_dir = self.temp_scenario / "inputs" / "database"
        temp_db_dir.mkdir(parents=True, exist_ok=True)
        
        # Create the ASSEMBLIES/SUPPLY directory
        temp_supply_dir = temp_db_dir / "ASSEMBLIES" / "SUPPLY"
        temp_supply_dir.mkdir(parents=True, exist_ok=True)
        
        # Create the COMPONENTS/DISTRIBUTION directory
        temp_dist_dir = temp_db_dir / "COMPONENTS" / "DISTRIBUTION"
        temp_dist_dir.mkdir(parents=True, exist_ok=True)
        
        # Create the COMPONENTS/CONVERSION directory
        temp_conv_dir = temp_db_dir / "COMPONENTS" / "CONVERSION"
        temp_conv_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy the supply system files
        for file_name in ["SUPPLY_HEATING.csv", "SUPPLY_COOLING.csv", "SUPPLY_HOTWATER.csv", "SUPPLY_ELECTRICITY.csv"]:
            src_file = Path(self.locator.get_db4_assemblies_supply_folder()) / file_name
            dst_file = temp_supply_dir / file_name
            
            if src_file.exists():
                log(f"Copying {file_name} to temp scenario")
                shutil.copy2(src_file, dst_file)
            else:
                log(f"Warning: {file_name} not found at {src_file}")
        
        # Copy the piping cost data
        src_file = Path(self.locator.get_database_components_distribution_thermal_grid("THERMAL_GRID"))
        dst_file = temp_dist_dir / "THERMAL_GRID.csv"
        
        if src_file.exists():
            log(f"Copying THERMAL_GRID.csv to temp scenario")
            shutil.copy2(src_file, dst_file)
        else:
            log(f"Warning: THERMAL_GRID.csv not found at {src_file}")
        
        # Copy the heat exchanger cost parameters
        src_file = Path(self.locator.get_db4_components_conversion_conversion_technology_csv("HEAT_EXCHANGERS"))
        dst_file = temp_conv_dir / "HEAT_EXCHANGERS.csv"
        
        if src_file.exists():
            log(f"Copying HEAT_EXCHANGERS.csv to temp scenario")
            shutil.copy2(src_file, dst_file)
        else:
            log(f"Warning: HEAT_EXCHANGERS.csv not found at {src_file}")
    
    def _generate_cluster_metrics(self, total_demand):
        """
        Generate metrics for all cluster combinations.
        
        This is a simplified version of the PipeLayoutGenerator.generate_pipe_layouts method
        that directly uses the files in the temp scenario folder.
        
        Parameters:
        -----------
        total_demand : pd.DataFrame
            The Total_demand.csv data
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with metrics for all cluster combinations
        """
        # Load cluster files
        temp_opt_dir = self.temp_scenario / "outputs" / "data" / "optimization" / "dtn_expansion"
        self.cluster_edges = pd.read_csv(temp_opt_dir / "cluster_edges.csv")
        self.cluster_nodes = pd.read_csv(temp_opt_dir / "cluster_nodes.csv")
        
        # Load edge-node matrix
        edge_node_files = list(self.temp_thermal_network_dir.glob("edge_node_matrix*.csv"))
        if not edge_node_files:
            log("Warning: No edge_node_matrix file found. This is not critical but may affect some calculations.")
            edge_node_matrix = None
        else:
            edge_node_file = edge_node_files[0]
            edge_node_matrix = pd.read_csv(edge_node_file, index_col=0)
        
        # Get unique clusters (excluding 0 and -1)
        self.all_clusters = sorted([c for c in self.cluster_edges['cluster'].unique() if c > 0])
        
        # Filter to testing clusters if specified
        if self.testing_clusters:
            valid_clusters = [c for c in self.testing_clusters if c in self.all_clusters]
            if len(valid_clusters) != len(self.testing_clusters):
                missing = set(self.testing_clusters) - set(valid_clusters)
                log(f"Warning: Some testing clusters do not exist: {missing}")
            
            self.all_clusters = sorted(valid_clusters)
            log(f"Using {len(self.all_clusters)} testing clusters: {self.all_clusters}")
        else:
            log(f"Using all {len(self.all_clusters)} clusters")
        
        # Store total demand for later use
        self.total_demand = total_demand
        
        # Generate all possible combinations of clusters
        all_combinations = []
        for r in range(1, len(self.all_clusters) + 1):
            combinations = list(itertools.combinations(self.all_clusters, r))
            all_combinations.extend(combinations)
        
        log(f"Generated {len(all_combinations)} possible cluster combinations")
        
        # Calculate metrics for each combination
        all_metrics = []
        
        # Calculate metrics for cluster 0 alone first
        cluster0_metrics = self._calculate_metrics_for_cluster((0,), self.cluster_edges, self.cluster_nodes, total_demand)
        all_metrics.append(cluster0_metrics)
        log(f"Calculated metrics for cluster 0 (existing DTN)")
        
        # Calculate metrics for all other combinations
        for i, cluster_set in enumerate(all_combinations):
            if i % 100 == 0 and i > 0:
                log(f"Processed {i}/{len(all_combinations)} cluster combinations")
            
            metrics = self._calculate_metrics_for_cluster(cluster_set, self.cluster_edges, self.cluster_nodes, total_demand)
            all_metrics.append(metrics)
        
        # Convert to DataFrame
        metrics_df = pd.DataFrame(all_metrics)
        
        # Save to CSV
        output_file = self.output_dir / "clusters_metrics_updated.csv"
        metrics_df.to_csv(output_file, index=False)
        log(f"Saved metrics for {len(all_metrics)} cluster combinations to {output_file}")
        
        # Create a dictionary of metrics for each cluster combination
        self.cluster_metrics = {}
        for metrics in all_metrics:
            clusters_str = metrics['clusters']
            self.cluster_metrics[clusters_str] = metrics
        
        return metrics_df
        
    def _setup_genetic_algorithm(self):
        """Set up the genetic algorithm using DEAP."""
        # Create a toolbox
        self.toolbox = base.Toolbox()
        
        # Define genome representation: each gene is a phase number (1 to num_phases)
        # for each cluster
        self.toolbox.register("attr_phase", random.randint, 1, self.num_phases)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual,
                             self.toolbox.attr_phase, n=len(self.all_clusters))
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        
        # Register genetic operators
        self.toolbox.register("evaluate", self._evaluate_individual)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutUniformInt, low=1, up=self.num_phases, indpb=0.2)
        
        # Use different selection methods based on optimization mode
        if self.multi_objective_mode:
            # For multi-objective, we'll set up NSGA-II selection later
            pass
        else:
            self.toolbox.register("select", tools.selTournament, tournsize=3)
        
        # Register the map function
        self.toolbox.register("map", map)
        
    def _evaluate_individual(self, individual):
        """
        Evaluate the fitness of an individual.
        
        Parameters:
        -----------
        individual : list
            List of phase assignments for each cluster
        
        Returns:
        --------
        tuple
            Fitness value(s) based on the objective function
        """
        # Enforce integer genome
        individual[:] = [int(round(g)) for g in individual]
        
        # Convert individual to cluster-phase mapping
        cluster_phase_map = {cluster: phase for cluster, phase in zip(self.all_clusters, individual)}
        
        # Calculate total ROI and NPV across all phases
        total_roi = 0
        total_npv = 0
        
        # Calculate emissions using the detailed model
        ind_tuple = tuple(individual)
        if hasattr(self, 'emissions_cache') and ind_tuple in self.emissions_cache:
            # Use cached emissions results if available
            phase_emissions, has_non_district_scale = self.emissions_cache[ind_tuple]
            log(f"Using cached emissions for individual {ind_tuple}")
        else:
            # Calculate emissions and cache the results
            phase_emissions, has_non_district_scale = self.calculate_emissions_for_genome(cluster_phase_map)
            if not hasattr(self, 'emissions_cache'):
                self.emissions_cache = {}
            self.emissions_cache[ind_tuple] = (phase_emissions, has_non_district_scale)
            log(f"Calculated and cached emissions for individual {ind_tuple}")
        
        # Calculate weighted emissions across all phases
        weighted_emissions = 0
        for phase in range(1, self.num_phases + 1):
            phase_duration = self.phase_durations[phase-1]
            if phase in phase_emissions:
                weighted_emissions += phase_emissions[phase]['operation'] * phase_duration
        
        # Apply penalty if cluster 0 has non-DISTRICT scale systems
        if has_non_district_scale:
            log(f"Applying penalty for non-DISTRICT scale systems in cluster 0 for individual {ind_tuple}")
            total_roi = -1000
            total_npv = -1000000
            weighted_emissions = 1000000  # Penalize weighted emissions for multi-objective mode
        
        # Group clusters by phase
        clusters_by_phase = {}
        for cluster, phase in cluster_phase_map.items():
            if phase > 0:  # Skip unconnected clusters (phase 0)
                if phase not in clusters_by_phase:
                    clusters_by_phase[phase] = []
                clusters_by_phase[phase].append(cluster)
        
        # Calculate ROI, NPV, and costs for each phase
        for phase, clusters in clusters_by_phase.items():
            # Calculate ROI and NPV for this phase
            roi = self.calculate_roi(tuple(clusters), phase)
            npv = self.calculate_npv(tuple(clusters), phase)
            
            total_roi += roi
            total_npv += npv
        
        # Return fitness based on objective function
        if self.multi_objective_mode:
            fitness_values = []
            for obj in self.multi_objective_functions:
                if obj.lower() == 'npv':
                    fitness_values.append(total_npv)
                elif obj.lower() == 'roi':
                    fitness_values.append(total_roi)
                elif obj.lower() == 'emissions':
                    fitness_values.append(weighted_emissions)
            return tuple(fitness_values)
        else:
            if self.objective_function.lower() == 'npv':
                return (total_npv,)
            elif self.objective_function.lower() == 'roi':
                return (total_roi,)
            elif self.objective_function.lower() == 'emissions':
                return (weighted_emissions,)
            else:
                # Default to NPV
                return (total_npv,)
                
    def calculate_emissions_for_genome(self, cluster_phase_map):
        """
        Calculate emissions for a specific genome by creating phase-specific supply files
        and running the LCA operation module for each phase with its specific supply file.
        
        Parameters:
        -----------
        cluster_phase_map : dict
            Dictionary mapping cluster IDs to phases
        
        Returns:
        --------
        tuple
            (dict, bool) - Dictionary with emissions per phase and a flag indicating if cluster 0 has non-DISTRICT scale systems
        """
        # Initialize results dictionary - only track operation emissions
        phase_emissions = {phase: {'operation': 0}
                          for phase in range(1, self.num_phases + 1)}
        
        # Get original supply file
        supply_file = self.temp_locator.get_building_supply()
        original_supply_df = pd.read_csv(supply_file)
        
        # Get district supply systems from cluster 0 buildings
        cluster0_buildings = self._get_buildings_in_specific_cluster(0)
        if not cluster0_buildings:
            log(f"No buildings found in cluster 0 (existing DTN)")
            return phase_emissions, False
        
        # Get supply systems used by cluster 0 (existing DTN)
        district_supply_systems = original_supply_df[original_supply_df['name'].isin(cluster0_buildings)]
        
        # Extract district supply types
        district_heating_system = district_supply_systems['supply_type_hs'].iloc[0]
        district_cooling_system = district_supply_systems['supply_type_cs'].iloc[0]
        district_dhw_system = district_supply_systems['supply_type_dhw'].iloc[0]
        
        # Verify these are DISTRICT scale systems
        heating_df = pd.read_csv(self.temp_locator.get_database_assemblies_supply_heating())
        cooling_df = pd.read_csv(self.temp_locator.get_database_assemblies_supply_cooling())
        dhw_df = pd.read_csv(self.temp_locator.get_database_assemblies_supply_hot_water())
        
        # Check if district systems are actually DISTRICT scale
        is_district_heating = heating_df[heating_df['code'] == district_heating_system]['scale'].iloc[0] == 'DISTRICT'
        is_district_cooling = cooling_df[cooling_df['code'] == district_cooling_system]['scale'].iloc[0] == 'DISTRICT'
        is_district_dhw = dhw_df[dhw_df['code'] == district_dhw_system]['scale'].iloc[0] == 'DISTRICT'
        
        has_non_district_scale = not (is_district_heating and is_district_cooling and is_district_dhw)
        if has_non_district_scale:
            log(f"Cluster 0 buildings are not using DISTRICT scale systems. This will result in penalties for the optimization results.")
        
        # Create directory for phase-specific supply files
        phase_files_dir = self.output_dir / "phase_supply_files"
        phase_files_dir.mkdir(parents=True, exist_ok=True)
        
        # Create phase 0 supply file (original)
        phase0_supply_path = phase_files_dir / "phase0_supply.csv"
        original_supply_df.to_csv(phase0_supply_path, index=False)
        log(f"Created phase 0 supply file: {phase0_supply_path}")
        
        # Process each phase separately
        connected_clusters_by_phase = {}
        for phase in range(1, self.num_phases + 1):
            # Get clusters connected in this phase
            newly_connected_clusters = [cluster for cluster, p in cluster_phase_map.items() if p == phase]
            connected_clusters_by_phase[phase] = newly_connected_clusters
            
            # Make a copy of the original supply file for this phase
            phase_supply_df = original_supply_df.copy()
            
            # Get all clusters connected up to this phase
            all_connected_clusters = [0]  # Start with cluster 0
            for p in range(1, phase + 1):
                all_connected_clusters.extend(connected_clusters_by_phase.get(p, []))
            
            log(f"Phase {phase}: Connected clusters {all_connected_clusters}")
            
            # Get all buildings in connected clusters
            all_connected_buildings = []
            for cluster in all_connected_clusters:
                buildings = self._get_buildings_in_specific_cluster(cluster)
                all_connected_buildings.extend(buildings)
            
            # Update supply systems for connected buildings
            for building in all_connected_buildings:
                building_idx = phase_supply_df[phase_supply_df['name'] == building].index
                if len(building_idx) > 0:
                    phase_supply_df.loc[building_idx, 'supply_type_hs'] = district_heating_system
                    phase_supply_df.loc[building_idx, 'supply_type_cs'] = district_cooling_system
                    phase_supply_df.loc[building_idx, 'supply_type_dhw'] = district_dhw_system
            
            # Save the phase-specific supply file
            phase_supply_path = phase_files_dir / f"phase{phase}_supply.csv"
            phase_supply_df.to_csv(phase_supply_path, index=False)
            log(f"Created phase {phase} supply file: {phase_supply_path}")
            
            # Run LCA operation module with the phase-specific supply file
            lca_operation(self.temp_locator, custom_supply_path=str(phase_supply_path))
            
            # Load LCA results
            lca_operation_results = pd.read_csv(self.temp_locator.get_lca_operation())
            
            # Filter LCA results to only include buildings in testing clusters if specified
            if hasattr(self, 'testing_clusters') and self.testing_clusters:
                buildings_in_testing_clusters = []
                for cluster in self.testing_clusters:
                    buildings = self._get_buildings_in_specific_cluster(cluster)
                    buildings_in_testing_clusters.extend(buildings)
                lca_operation_results = lca_operation_results[lca_operation_results['name'].isin(buildings_in_testing_clusters)]
            
            # Calculate total emissions for all buildings in testing clusters
            phase_emissions[phase]['operation'] = lca_operation_results['GHG_sys_tonCO2'].sum()
        
        return phase_emissions, has_non_district_scale
        
    def _get_buildings_in_specific_cluster(self, cluster_id):
        """
        Get the buildings in a specific cluster.
        
        Parameters:
        -----------
        cluster_id : int
            Cluster ID
        
        Returns:
        --------
        list
            List of building names
        """
        buildings = self.cluster_nodes[
            (self.cluster_nodes['cluster'] == cluster_id) &
            (self.cluster_nodes['type'] == 'CONSUMER')
        ]['building'].tolist()
        
        return buildings
        
    def _get_buildings_in_clusters(self, cluster_set):
        """
        Get the buildings in a set of clusters.
        
        Parameters:
        -----------
        cluster_set : set
            Set of cluster IDs
        
        Returns:
        --------
        list
            List of building names
        """
        buildings = self.cluster_nodes[
            (self.cluster_nodes['cluster'].isin(cluster_set)) &
            (self.cluster_nodes['type'] == 'CONSUMER')
        ]['building'].unique().tolist()
        
        return buildings
        
    def _calculate_phase_capex(self, clusters, phase):
        """
        Calculate total CAPEX for a phase with interest rate adjustment.
        
        Parameters:
        -----------
        clusters : tuple or list
            Tuple or list of cluster IDs
        phase : int, optional
            Phase number (1-based). If provided, applies interest rate adjustment.
        
        Returns:
        --------
        float
            Total CAPEX for the phase with interest rate adjustment
        """
        # Get required pipes for the clusters
        required_pipes = self._get_required_pipes(set(clusters), self.cluster_edges)
        
        # Calculate pipe CAPEX
        pipe_capex = 0
        try:
            # Load pipe cost data
            piping_cost_data = pd.read_csv(self.temp_locator.get_database_components_distribution_thermal_grid('THERMAL_GRID'))
            
            # Merge with cost data
            cost_df = required_pipes.merge(piping_cost_data, on='pipe_DN')
            
            # Calculate cost for each pipe segment
            cost_df['pipe_cost'] = cost_df['Inv_USD2015perm'] * cost_df['length_m']
            
            # Sum up all pipe costs
            pipe_capex = cost_df['pipe_cost'].sum()
        except Exception as e:
            log(f"Error in detailed pipe CAPEX calculation: {e}. Using simplified calculation.")
            # Simplified pipe CAPEX calculation
            pipe_capex = required_pipes['length_m'].sum() * 1000  # $1000 per meter
        
        # Calculate HEX CAPEX
        buildings = self._get_buildings_in_clusters(set(clusters))
        hex_capex = 0
        try:
            # Load HEX cost parameters
            HEX_prices = pd.read_csv(self.temp_locator.get_db4_components_conversion_conversion_technology_csv('HEAT_EXCHANGERS'), index_col=0)
            a = HEX_prices['a']['District substation heat exchanger']
            b = HEX_prices['b']['District substation heat exchanger']
            c = HEX_prices['c']['District substation heat exchanger']
            d = HEX_prices['d']['District substation heat exchanger']
            e = HEX_prices['e']['District substation heat exchanger']
            
            # Get key for this cluster set
            key = '+'.join(map(str, sorted(clusters)))
            
            # Get metrics for this cluster set
            if key in self.cluster_metrics:
                metrics = self.cluster_metrics[key]
                
                # Get annual demand
                if self.network_type == 'DH':
                    annual_demand_mwh = metrics.get('total_annual_Qh_MWh', 0)
                else:
                    annual_demand_mwh = metrics.get('total_annual_Qc_MWh', 0)
                
                # Convert annual demand to peak demand (kW) using the diversity factor
                peak_demand_kw = annual_demand_mwh * 1000 / 2000 / 0.7  # Assuming 2000 equivalent full load hours and 0.7 diversity factor
                
                # Get temperature difference based on network type
                if self.network_type == 'DH':
                    temp_diff = 20  # Typical temperature difference for DH
                else:
                    temp_diff = 8   # Typical temperature difference for DC
                
                # Calculate mass flow rate (kg/s)
                mass_flow_rate = peak_demand_kw / (HEAT_CAPACITY_OF_WATER_JPERKGK * temp_diff / 1000)
                
                # Calculate HEX costs
                total_hex_capex = 0
                
                # Distribute mass flow rate among buildings proportionally to their number
                mass_flow_per_building = mass_flow_rate / len(buildings) if buildings else 0
                
                for _ in buildings:
                    # Split into several HEXs if flows are too high
                    if mass_flow_per_building <= 10:  # MAX_NODE_FLOW = 10 kg/s
                        mcp_sub = mass_flow_per_building * HEAT_CAPACITY_OF_WATER_JPERKGK
                        hex_capex = a + b * mcp_sub ** c + d * np.log(mcp_sub) + e * mcp_sub * np.log(mcp_sub)
                    else:
                        # We need to split into several HEXs
                        hex_capex = 0
                        number_of_HEXs = int(np.ceil(mass_flow_per_building / 10))
                        nodeflow_nom = mass_flow_per_building / number_of_HEXs
                        mcp_sub = nodeflow_nom * HEAT_CAPACITY_OF_WATER_JPERKGK
                        for i in range(number_of_HEXs):
                            hex_capex += (a + b * mcp_sub ** c + d * np.log(mcp_sub) + e * mcp_sub * np.log(mcp_sub))
                    
                    total_hex_capex += hex_capex
                
                hex_capex = total_hex_capex
            else:
                # Fallback if metrics not available
                hex_capex = len(buildings) * 10000  # $10,000 per building
        except Exception as e:
            log(f"Error in detailed HEX CAPEX calculation: {e}. Using simplified calculation.")
            # Simplified HEX CAPEX calculation
            hex_capex = len(buildings) * 10000  # $10,000 per building
        
        # Calculate pump CAPEX (simplified)
        pump_capex = required_pipes['length_m'].sum() * 100  # $100 per meter
        
        # Calculate cooling plant CAPEX (for DC only)
        cooling_plant_capex = 0
        if self.network_type == 'DC':
            cooling_plant_capex = required_pipes['length_m'].sum() * 200  # $200 per meter
        
        # Calculate total base CAPEX
        total_base_capex = pipe_capex + hex_capex + pump_capex + cooling_plant_capex
        
        # Apply interest rate adjustment based on phase
        discount_factor = 1.0
        if phase is not None and phase > 1:
            year_offset = 0
            for p in range(1, phase):
                year_offset += self.phase_durations[p-1]
            
            # Discount factor based on when the phase starts
            discount_factor = 1 / ((1 + self.interest_rate) ** year_offset)
        
        # Apply discount factor to get present value
        total_capex = total_base_capex * discount_factor
        
        return total_capex
        
    def calculate_roi(self, clusters, phase):
        """
        Calculate Discounted ROI for a cluster set in a specific phase.
        
        Parameters:
        -----------
        clusters : tuple or list
            Tuple or list of cluster IDs
        phase : int
            Phase number (1-based)
        
        Returns:
        --------
        float
            Discounted Return on Investment (ROI)
        """
        # For phase 0, return 0
        if phase == 0:
            return 0
        
        # Get key for this cluster set
        key = '+'.join(map(str, sorted(clusters)))
        
        # Get metrics for this cluster set
        if key not in self.cluster_metrics:
            return -float('inf')  # Invalid cluster set
        
        metrics = self.cluster_metrics[key]
        
        # Calculate CAPEX with interest rate adjustment
        capex = self._calculate_phase_capex(clusters, phase)
        
        # Calculate annual revenue (energy price * annual demand)
        if self.network_type == 'DH':
            annual_demand_kwh = metrics['total_annual_Qh_MWh'] * 1000  # Convert MWh to kWh
        else:
            annual_demand_kwh = metrics['total_annual_Qc_MWh'] * 1000  # Convert MWh to kWh
        
        annual_revenue = annual_demand_kwh * self.energy_price
        
        # Calculate annual O&M costs (typically 2-3% of CAPEX)
        annual_om_cost = 0.025 * capex
        
        # Net annual return
        net_annual_return = annual_revenue - annual_om_cost
        
        # Get the duration of this phase
        phase_duration = self.phase_durations[phase-1]
        
        # Calculate present value of net annual returns over the entire phase duration
        present_value_of_returns = 0
        for year in range(phase_duration):
            discount_factor = 1 / ((1 + self.interest_rate) ** (year + 1))
            present_value_of_returns += net_annual_return * discount_factor
        
        # Calculate Discounted ROI (present value of net annual returns / CAPEX)
        if capex > 0:
            roi = present_value_of_returns / capex
        else:
            roi = 0
        
        return roi
        
    def calculate_npv(self, clusters, phase, years=20):
        """
        Calculate Net Present Value for a cluster set in a specific phase.
        
        Parameters:
        -----------
        clusters : tuple or list
            Tuple or list of cluster IDs
        phase : int
            Phase number (1-based)
        years : int
            Number of years for NPV calculation
        
        Returns:
        --------
        float
            Net Present Value (NPV)
        """
        # Calculate CAPEX with interest rate adjustment
        capex = self._calculate_phase_capex(clusters, phase)
        
        # For phase 0, return negative CAPEX
        if phase == 0:
            return -capex
        
        # Get key for this cluster set
        key = '+'.join(map(str, sorted(clusters)))
        
        # Get metrics for this cluster set
        if key not in self.cluster_metrics:
            return -float('inf')  # Invalid cluster set
        
        metrics = self.cluster_metrics[key]
        
        # Calculate annual revenue and O&M costs
        if self.network_type == 'DH':
            annual_demand_kwh = metrics['total_annual_Qh_MWh'] * 1000  # Convert MWh to kWh
        else:
            annual_demand_kwh = metrics['total_annual_Qc_MWh'] * 1000  # Convert MWh to kWh
        
        annual_revenue = annual_demand_kwh * self.energy_price
        annual_om_cost = 0.025 * capex
        net_annual_return = annual_revenue - annual_om_cost
        
        # Calculate the year when this phase starts
        phase_start_year = 0
        for p in range(1, phase):
            phase_start_year += self.phase_durations[p-1]
        
        # Calculate NPV
        npv = -capex  # Initial investment (negative) at the start of the phase
        
        for year in range(years):
            # Only count returns for years after the phase starts
            if year >= phase_start_year:
                # Determine which phase this year belongs to
                current_phase = 1
                year_in_phases = year
                while current_phase <= self.num_phases:
                    if year_in_phases < self.phase_durations[current_phase-1]:
                        break
                    year_in_phases -= self.phase_durations[current_phase-1]
                    current_phase += 1
                
                # Only count returns if we're in or after the current phase
                if current_phase >= phase:
                    discount_factor = 1 / ((1 + self.interest_rate) ** (year + 1))
                    npv += net_annual_return * discount_factor
        
        return npv
        
    def run_optimization(self, population_size=50, num_generations=30):
        """
        Run the genetic algorithm to find the optimal phase assignment.
        
        Parameters:
        -----------
        population_size : int
            Size of the population
        num_generations : int
            Number of generations
        
        Returns:
        --------
        dict
            Dictionary with the optimal solution
        """
        log("=== Starting Genetic Algorithm Optimization ===")
        
        # Set up parameters for optimization with try-except blocks for each parameter
        try:
            self.num_phases = self.config.dtn_expansion_optimization.num_phases
        except AttributeError:
            self.num_phases = 3  # Default value
            log("Using default number of phases: 3")
        
        # Parse phase durations from config
        try:
            phase_durations_str = self.config.dtn_expansion_optimization.phase_durations
            if phase_durations_str:
                self.phase_durations = [int(d.strip()) for d in phase_durations_str.split(',') if d.strip()]
            else:
                self.phase_durations = [10] * self.num_phases
        except AttributeError:
            self.phase_durations = [10] * self.num_phases
            log(f"Using default phase durations: {self.phase_durations}")
        
        # Set interest rate
        try:
            self.interest_rate = self.config.dtn_expansion_optimization.interest_rate
        except AttributeError:
            self.interest_rate = 0.05  # Default value
            log("Using default interest rate: 0.05")
        
        # Set energy price
        try:
            self.energy_price = self.config.dtn_expansion_optimization.energy_price
        except AttributeError:
            self.energy_price = 0.1  # Default value
            log("Using default energy price: 0.1")
        
        # Set objective function
        try:
            self.objective_function = self.config.dtn_expansion_optimization.objective_function
        except AttributeError:
            self.objective_function = 'NPV'  # Default value
            log("Using default objective function: NPV")
        
        # Set multi-objective mode
        try:
            self.multi_objective_mode = self.config.dtn_expansion_optimization.multi_objective_mode
        except AttributeError:
            self.multi_objective_mode = False  # Default value
            log("Using default multi-objective mode: False")
        
        # Parse multi-objective functions from config
        try:
            multi_objective_functions_str = self.config.dtn_expansion_optimization.multi_objective_functions
            if multi_objective_functions_str:
                self.multi_objective_functions = [f.strip() for f in multi_objective_functions_str.split(',') if f.strip()]
            else:
                self.multi_objective_functions = ['NPV', 'emissions']
        except AttributeError:
            self.multi_objective_functions = ['NPV', 'emissions']  # Default value
            log("Using default multi-objective functions: NPV, emissions")
        
        log(f"Optimization parameters:")
        log(f"  Number of phases: {self.num_phases}")
        log(f"  Phase durations: {self.phase_durations}")
        log(f"  Interest rate: {self.interest_rate}")
        log(f"  Energy price: {self.energy_price}")
        log(f"  Objective function: {self.objective_function}")
        log(f"  Multi-objective mode: {self.multi_objective_mode}")
        log(f"  Multi-objective functions: {self.multi_objective_functions}")
        log(f"  Number of clusters: {len(self.all_clusters)}")
        
        # Set up the genetic algorithm
        setup_creator(self.multi_objective_mode, self.objective_function, self.multi_objective_functions)
        self._setup_genetic_algorithm()
        
        # Create initial population
        pop = self.toolbox.population(n=population_size)
        
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
        
        # Assign fitness values to individuals
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        # Track the best individual
        best_ind = tools.selBest(pop, 1)[0]
        
        log(f"Initial best individual: {best_ind}")
        log(f"Initial best fitness: {best_ind.fitness.values}")
        
        # Run the genetic algorithm
        for gen in range(num_generations):
            log(f"-- Generation {gen+1} --")
            
            # Select the next generation individuals
            offspring = self.toolbox.select(pop, len(pop))
            
            # Clone the selected individuals
            offspring = list(map(self.toolbox.clone, offspring))
            
            # Apply crossover and mutation
            for i in range(1, len(offspring), 2):
                if random.random() < 0.5:
                    self.toolbox.mate(offspring[i-1], offspring[i])
                    del offspring[i-1].fitness.values
                    del offspring[i].fitness.values
            
            for i in range(len(offspring)):
                if random.random() < 0.2:
                    self.toolbox.mutate(offspring[i])
                    del offspring[i].fitness.values
            
            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
            
            # Assign fitness values to individuals
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            # Replace the population with the offspring
            pop[:] = offspring
            
            # Update the best individual
            current_best = tools.selBest(pop, 1)[0]
            if current_best.fitness.values[0] > best_ind.fitness.values[0]:
                best_ind = self.toolbox.clone(current_best)
            
            log(f"Best fitness: {best_ind.fitness.values}")
        
        # Convert the best individual to a solution
        solution = {
            'individual': list(best_ind),
            'fitness': best_ind.fitness.values,
            'cluster_phase_map': {cluster: phase for cluster, phase in zip(self.all_clusters, best_ind)},
            'phases': {}
        }
        
        # Group clusters by phase
        for cluster, phase in solution['cluster_phase_map'].items():
            if phase not in solution['phases']:
                solution['phases'][phase] = []
            solution['phases'][phase].append(cluster)
        
        # Save the solution
        self._save_optimization_results(solution)
        
        log("=== Genetic Algorithm Optimization Completed ===")
        
        return solution
        
    def _save_optimization_results(self, solution):
        """
        Save the optimization results to files.
        
        Parameters:
        -----------
        solution : dict
            Dictionary with the optimization solution
        """
        # Create output directory
        results_dir = self.output_dir / "optimization_results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the solution as JSON
        import json
        with open(results_dir / "solution.json", 'w') as f:
            # Convert keys to strings for JSON serialization
            json_solution = {
                'individual': solution['individual'],
                'fitness': list(solution['fitness']),
                'cluster_phase_map': {str(k): v for k, v in solution['cluster_phase_map'].items()},
                'phases': {str(k): v for k, v in solution['phases'].items()}
            }
            json.dump(json_solution, f, indent=4)
        
        # Save the solution as CSV
        solution_df = pd.DataFrame({
            'cluster': list(solution['cluster_phase_map'].keys()),
            'phase': list(solution['cluster_phase_map'].values())
        })
        solution_df.to_csv(results_dir / "solution.csv", index=False)
        
        log(f"Optimization results saved to: {results_dir}")
    
    def _calculate_metrics_for_cluster(self, cluster_set, cluster_edges, cluster_nodes, total_demand):
        """
        Calculate metrics for a specific cluster set.
        
        This is a simplified version of the PipeLayoutGenerator.calculate_metrics method.
        
        Parameters:
        -----------
        cluster_set : tuple
            Tuple of cluster IDs
        cluster_edges : pd.DataFrame
            DataFrame of cluster edges
        cluster_nodes : pd.DataFrame
            DataFrame of cluster nodes
        total_demand : pd.DataFrame
            The Total_demand.csv data
            
        Returns:
        --------
        dict
            Dictionary of metrics for the cluster set
        """
        # Include cluster 0 (existing DTN) in the set
        clusters_to_connect = set(cluster_set).union({0})
        
        # Get required pipes
        required_pipes = self._get_required_pipes(clusters_to_connect, cluster_edges)
        
        # Calculate total pipe length
        total_pipe_length = required_pipes['length_m'].sum()
        
        # Get buildings in the clusters
        buildings_in_clusters = cluster_nodes[
            (cluster_nodes['cluster'].isin(clusters_to_connect)) &
            (cluster_nodes['type'] == 'CONSUMER')
        ]['building'].unique().tolist()
        
        # Calculate total annual demand
        if self.network_type == 'DH':
            # For district heating, use Qhs_sys_MWhyr + Qww_sys_MWhyr
            total_annual_demand = total_demand[
                total_demand['name'].isin(buildings_in_clusters)
            ]['Qhs_sys_MWhyr'].sum() + total_demand[
                total_demand['name'].isin(buildings_in_clusters)
            ]['Qww_sys_MWhyr'].sum()
            demand_type = 'Qh'
        else:
            # For district cooling, use Qcs_sys_MWhyr + Qcre_sys_MWhyr + Qcdata_sys_MWhyr
            total_annual_demand = total_demand[
                total_demand['name'].isin(buildings_in_clusters)
            ]['Qcs_sys_MWhyr'].sum() + total_demand[
                total_demand['name'].isin(buildings_in_clusters)
            ]['Qcre_sys_MWhyr'].sum() + total_demand[
                total_demand['name'].isin(buildings_in_clusters)
            ]['Qcdata_sys_MWhyr'].sum()
            demand_type = 'Qc'
        
        # Calculate linear heat density
        if total_pipe_length > 0:
            linear_heat_density = total_annual_demand / total_pipe_length * 1000  # MWh/km
        else:
            linear_heat_density = 0
        
        # Create metrics dictionary
        metrics = {
            'clusters': '+'.join(map(str, sorted(clusters_to_connect))),
            f'total_annual_{demand_type}_MWh': total_annual_demand,
            'total_pipe_length_m': total_pipe_length,
            f'linear_{demand_type}_density_MWh_per_km': linear_heat_density
        }
        
        return metrics
    
    def _get_required_pipes(self, clusters_to_connect, cluster_edges):
        """
        Get the required pipes for a set of clusters.
        
        This is a simplified version of the PipeLayoutGenerator.get_required_pipes_for_clusters method.
        
        Parameters:
        -----------
        clusters_to_connect : set
            Set of cluster IDs to connect
        cluster_edges : pd.DataFrame
            DataFrame of cluster edges
            
        Returns:
        --------
        pd.DataFrame
            DataFrame of required pipes
        """
        # Get edges that belong to the clusters in the set
        cluster_edges_subset = cluster_edges[cluster_edges['cluster'].isin(clusters_to_connect)]
        
        # Get main road edges (-1) that connect the clusters
        main_road_edges = cluster_edges[
            (cluster_edges['cluster'] == -1) &
            (cluster_edges['from_C'].isin(clusters_to_connect)) &
            (cluster_edges['to_C'].isin(clusters_to_connect))
        ]
        
        # Combine the edges
        required_pipes = pd.concat([cluster_edges_subset, main_road_edges], ignore_index=True)
        
        return required_pipes
    
    def run(self):
        """
        Run the optimization process.
        
        This method prepares all files, generates updated metrics,
        and runs the optimization algorithm.
        
        Returns:
        --------
        dict
            Dictionary with optimization results
        """
        log("=== Starting Dynamic DTN Rerun Optimization Part 2 ===")
        
        try:
            # Step 1: Prepare all files needed for optimization
            total_demand = self._prepare_optimization_files()
            
            # Step 2: Generate updated metrics
            log("Generating updated metrics with temp scenario demand files...")
            metrics_df = self._generate_cluster_metrics(total_demand)
            
            # Step 3: Compare with original metrics
            original_metrics_file = self.original_dtn_dir / "phase_1" / "clusters_metrics.csv"
            if original_metrics_file.exists():
                original_metrics = pd.read_csv(original_metrics_file)
                
                log("Comparing original and updated metrics:")
                if metrics_df.equals(original_metrics):
                    log("⚠️ Generated metrics are IDENTICAL to original metrics!")
                    log("This suggests that temp scenario demand files are not being used properly.")
                else:
                    log("✓ Generated metrics are DIFFERENT from original metrics")
                    log("This confirms that temp scenario demand files are being used correctly.")
                
                # Compare key columns
                if self.network_type == 'DH':
                    key_column = 'total_annual_Qh_MWh'
                else:
                    key_column = 'total_annual_Qc_MWh'
                    
                if key_column in metrics_df.columns and key_column in original_metrics.columns:
                    log(f"Detailed comparison of {key_column} column:")
                    
                    # Calculate differences
                    diff = metrics_df[key_column] - original_metrics[key_column]
                    
                    # Log summary statistics
                    log(f"  Max absolute difference: {diff.abs().max():.2f} MWh")
                    log(f"  Mean absolute difference: {diff.abs().mean():.2f} MWh")
                    log(f"  Sum of differences: {diff.sum():.2f} MWh")
                    log(f"  Percentage of values that changed: {(diff != 0).mean() * 100:.1f}%")
            
            # Step 4: Run the optimization algorithm
            log("Running genetic algorithm optimization with updated metrics...")
            # Try to get parameters from dtn-expansion-optimization section, or use defaults if not available
            try:
                population_size = self.config.dtn_expansion_optimization.population_size
            except AttributeError:
                population_size = 50  # Default value
                log("Using default population size: 50")
                
            try:
                num_generations = self.config.dtn_expansion_optimization.num_generations
            except AttributeError:
                num_generations = 30  # Default value
                log("Using default number of generations: 30")
                
            solution = self.run_optimization(
                population_size=population_size,
                num_generations=num_generations
            )
            
            log("=== Dynamic DTN Rerun Optimization Part 2 completed successfully ===")
            return {
                "status": "success",
                "metrics_df": metrics_df,
                "solution": solution,
                "output_dir": str(self.output_dir)
            }
            
        except Exception as e:
            import traceback
            log(f"Error during Dynamic DTN Rerun Optimization Part 2: {str(e)}")
            log(f"Traceback: {traceback.format_exc()}")
            
            return {
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc()
            }


def main(config):
    """
    Run the dynamic DTN optimization part 2 script.
    
    Parameters:
    -----------
    config : cea.config.Configuration
        The configuration object
    
    Returns:
    --------
    None
    """
    start = time.time()
    
    # Get the scenario and locator
    scenario = config.scenario
    locator = cea.inputlocator.InputLocator(scenario=scenario)
    
    # Create and run the optimizer
    optimizer = CustomDTNRerunOptimizer(locator, config)
    results = optimizer.run()
    
    # Log results
    if results["status"] == "success":
        log("Optimization completed successfully!")
        log(f"Results saved to: {results['output_dir']}")
    else:
        log(f"Optimization failed: {results['error']}")
    
    # Log execution time
    time_elapsed = time.time() - start
    log(f"Execution time: {time_elapsed:.2f} seconds")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the dynamic DTN optimization part 2 script.')
    parser.add_argument('-s', '--scenario', help='Path to the scenario folder')
    parser.add_argument('-c', '--config', help='Path to the config file')
    args = parser.parse_args()
    
    config = cea.config.Configuration()
    
    if args.scenario:
        config.scenario = args.scenario
    if args.config:
        config.load(args.config)
    
    main(config)