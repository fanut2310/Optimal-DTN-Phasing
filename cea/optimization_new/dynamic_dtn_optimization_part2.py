from __future__ import annotations

###############################################################################
# 1) STANDARD IMPORTS                                                        #
###############################################################################
import argparse
import logging
import os
import shutil
import time
import itertools
import random
import json
from pathlib import Path
from typing import Dict, Tuple, List, Set, Optional, Union

import geopandas as gpd
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from deap import base, tools, algorithms, creator

import cea.config
import cea.inputlocator
from cea.constants import HEAT_CAPACITY_OF_WATER_JPERKGK
from cea.analysis.lca.operation import lca_operation
from cea.optimization_new.dynamic_dtn_optimization import DynamicDTNOptimizer


# TempScenarioLocator class for redirecting file requests to the temporary scenario
class TempScenarioLocator(cea.inputlocator.InputLocator):
    """
    A locator that redirects file requests to the temporary scenario.
    
    This locator inherits from InputLocator and is initialized with the path to the 
    temporary scenario created by dynamic_dtn_optimization.py. It overrides methods
    to ensure that file requests are directed to the temporary scenario instead of
    the original scenario.
    """
    
    def __init__(self, original_locator, temp_scenario_path):
        """
        Initialize the locator with the path to the temporary scenario.
        
        Parameters:
        -----------
        original_locator : cea.inputlocator.InputLocator
            The original locator
        temp_scenario_path : str or Path
            Path to the temporary scenario
        """
        # Initialize with the temp scenario path
        super().__init__(str(temp_scenario_path))
        
        # Store the original locator for reference
        self.original_locator = original_locator
        
        # Copy attributes from original locator that might be needed
        self.__dict__.update({k: v for k, v in original_locator.__dict__.items() 
                             if k not in ['scenario', '_scenario', '_temp_directory']})
        
        # Clear any cache
        if hasattr(self, '_demand_cache'):
            self._demand_cache = {}
        
        log().info(f"Created TempScenarioLocator pointing to: {self.scenario}")
    
    def get_database_conversion_systems(self):
        """
        Get the path to the database conversion systems file in the temp scenario.
        
        Returns:
        --------
        str
            Path to the database conversion systems file
        """
        path = os.path.join(self.scenario, 'inputs', 'database', 'COMPONENTS', 'CONVERSION', 'CONVERSION_SYSTEMS.csv')
        
        if not os.path.exists(path):
            log().error(f"Database conversion systems file not found in temp scenario: {path}")
            raise FileNotFoundError(f"Database conversion systems file not found in temp scenario: {path}")
        
        return path
    
    def get_database_distribution_systems(self):
        """
        Get the path to the database distribution systems file in the temp scenario.
        
        Returns:
        --------
        str
            Path to the database distribution systems file
        """
        path = os.path.join(self.scenario, 'inputs', 'database', 'COMPONENTS', 'DISTRIBUTION', 'THERMAL_GRID.csv')
        
        if not os.path.exists(path):
            log().error(f"Database distribution systems file not found in temp scenario: {path}")
            raise FileNotFoundError(f"Database distribution systems file not found in temp scenario: {path}")
        
        return path
    
    def get_thermal_network_folder(self):
        """
        Get the path to the thermal network folder in the temp scenario.
        
        Returns:
        --------
        str
            Path to the thermal network folder
        """
        path = os.path.join(self.scenario, 'outputs', 'data', 'thermal-network')
        
        if not os.path.exists(path):
            log().error(f"Thermal network folder not found in temp scenario: {path}")
            raise FileNotFoundError(f"Thermal network folder not found in temp scenario: {path}")
        
        return path
    
    def get_total_demand(self):
        """
        Get the path to the total demand file in the temp scenario.
        
        Returns:
        --------
        str
            Path to the total demand file
        """
        path = os.path.join(self.scenario, 'outputs', 'data', 'demand', 'Total_demand.csv')
        
        if not os.path.exists(path):
            log().error(f"Total demand file not found in temp scenario: {path}")
            raise FileNotFoundError(f"Total demand file not found in temp scenario: {path}")
        
        return path
    
    def get_building_supply(self):
        """
        Get the path to the building supply file in the temp scenario.
        
        Returns:
        --------
        str
            Path to the building supply file
        """
        path = os.path.join(self.scenario, 'inputs', 'building-properties', 'supply.csv')
        
        if not os.path.exists(path):
            log().error(f"Building supply file not found in temp scenario: {path}")
            raise FileNotFoundError(f"Building supply file not found in temp scenario: {path}")
        
        return path
    
    def get_lca_operation(self):
        """
        Get the path to the LCA operation file in the temp scenario.
        
        Returns:
        --------
        str
            Path to the LCA operation file
        """
        path = os.path.join(self.scenario, 'outputs', 'data', 'emissions', 'Total_LCA_operation.csv')
        
        if not os.path.exists(path):
            log().error(f"LCA operation file not found in temp scenario: {path}")
            raise FileNotFoundError(f"LCA operation file not found in temp scenario: {path}")
        
        return path
    
    def get_demand_results_file(self, building_name):
        """
        Get the path to the demand results file for a specific building in the temp scenario.
        
        Parameters:
        -----------
        building_name : str
            Name of the building
            
        Returns:
        --------
        str
            Path to the demand results file
        """
        path = os.path.join(self.scenario, 'outputs', 'data', 'demand', f"{building_name}.csv")
        
        if not os.path.exists(path):
            log().error(f"Demand results file for building {building_name} not found in temp scenario: {path}")
            raise FileNotFoundError(f"Demand results file for building {building_name} not found in temp scenario: {path}")
        
        return path
    
    def get_database_supply_assemblies(self):
        """
        Get the path to the database supply assemblies file in the temp scenario.
        
        Returns:
        --------
        str
            Path to the database supply assemblies file
        """
        path = os.path.join(self.scenario, 'inputs', 'database', 'ASSEMBLIES', 'SUPPLY.xlsx')
        
        if not os.path.exists(path):
            log().error(f"Database supply assemblies file not found in temp scenario: {path}")
            raise FileNotFoundError(f"Database supply assemblies file not found in temp scenario: {path}")
        
        return path
        
    def get_dynamic_dtn_optimization_thermal_network_folder(self):
        """
        Override to return the path to the thermal network folder in the temp scenario.
        
        This override ensures that dynamic DTN-specific methods look for files in the correct
        location within the temporary scenario.
        
        Returns:
        --------
        str
            Path to the thermal network folder in the temp scenario
        """
        # Return the same path as get_thermal_network_folder()
        return self.get_thermal_network_folder()
        
    def get_dynamic_dtn_network_layout_costs_file(self, network_type, network_name=""):
        """
        Override to return the path to the network layout costs file in the temp scenario.
        
        This override ensures that the method returns the correct path to the cost file
        in the temporary scenario.
        
        Parameters:
        -----------
        network_type : str
            Type of the network (e.g., 'DH', 'DC')
        network_name : str, optional
            Name of the network
            
        Returns:
        --------
        str
            Path to the network layout costs file in the temp scenario
        """
        # Use get_thermal_network_folder() directly instead of get_dynamic_dtn_optimization_thermal_network_folder()
        file_name = f"{network_type}_costs.csv"
        return os.path.join(self.get_thermal_network_folder(), file_name)


# Setup function for the creator based on optimization mode
def setup_creator(multi_objective=False, objective_function='NPV', multi_objective_functions=None):
    """
    Set up the creator based on optimization mode and selected objectives
    
    Parameters:
    -----------
    multi_objective : bool
        Whether to use multi-objective optimization
    objective_function : str
        The objective function to use for single-objective optimization
    multi_objective_functions : list
        List of objective functions to use for multi-objective optimization
    """
    # Clear any existing creator attributes to avoid conflicts
    if hasattr(creator, 'FitnessMax'):
        del creator.FitnessMax
    if hasattr(creator, 'FitnessMin'):
        del creator.FitnessMin
    if hasattr(creator, 'FitnessMulti'):
        del creator.FitnessMulti
    if hasattr(creator, 'Individual'):
        del creator.Individual

    if multi_objective:
        # For multi-objective optimization, create a fitness class that handles multiple objectives
        creator.create("FitnessMulti", base.Fitness, weights=(1.0,) * len(multi_objective_functions))
        creator.create("Individual", list, fitness=creator.FitnessMulti)
    else:
        # For single-objective optimization, create a fitness class based on the objective function
        if objective_function.lower() == 'emissions':
            # For emissions, we want to minimize, so use negative weights
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
            creator.create("Individual", list, fitness=creator.FitnessMin)
        else:
            # For other objectives (NPV, ROI), we want to maximize
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
            creator.create("Individual", list, fitness=creator.FitnessMax)


def log():
    """
    Get the logger for this module.
    
    Returns:
    --------
    logging.Logger
        The logger for this module
    """
    return logging.getLogger("cea.optimization.dynamic_dtn_optimization_part2")


def _read_shp_force_2d(path: Path):
    """
    Read a shapefile and force 2D geometry.
    
    Parameters:
    -----------
    path : Path
        Path to the shapefile
        
    Returns:
    --------
    geopandas.GeoDataFrame
        The shapefile as a GeoDataFrame with 2D geometry
    """
    gdf = gpd.read_file(path)
    gdf['geometry'] = gdf['geometry'].apply(lambda geom: geom.buffer(0))
    return gdf


def parse_args():
    """
    Parse command line arguments.
    
    Returns:
    --------
    argparse.Namespace
        The parsed arguments
    """
    parser = argparse.ArgumentParser(description='Run the dynamic DTN optimization part 2 script.')
    parser.add_argument('-s', '--scenario', help='Path to the scenario folder')
    parser.add_argument('-c', '--config', help='Path to the config file')
    return parser.parse_args()


def check_thermal_network_prerequisites(locator, network_type, bypass_check=False):
    """
    Check if the thermal network prerequisites are met.
    
    Parameters:
    -----------
    locator : cea.inputlocator.InputLocator
        The locator for the scenario
    network_type : str
        The network type (DH or DC)
    bypass_check : bool
        Whether to bypass the check
        
    Returns:
    --------
    bool
        True if prerequisites are met, False otherwise
    """
    if bypass_check:
        return True
        
    # Check if the thermal network has been created
    thermal_network_folder = Path(locator.get_thermal_network_folder())
    if not thermal_network_folder.exists():
        log().error(f"Thermal network folder not found: {thermal_network_folder}")
        return False
        
    # Check if the network layout exists
    network_layout_file = thermal_network_folder / f"{network_type}_network_layout.shp"
    if not network_layout_file.exists():
        log().error(f"Network layout file not found: {network_layout_file}")
        return False
        
    # Check if the network edges file exists
    network_edges_file = thermal_network_folder / f"{network_type}_network_edges.shp"
    if not network_edges_file.exists():
        log().error(f"Network edges file not found: {network_edges_file}")
        return False
        
    # Check if the network nodes file exists
    network_nodes_file = thermal_network_folder / f"{network_type}_network_nodes.shp"
    if not network_nodes_file.exists():
        log().error(f"Network nodes file not found: {network_nodes_file}")
        return False
        
    return True


class DTNExpansionOptimizer:
    """
    Optimizer for district thermal network expansion.
    
    This class implements the genetic algorithm for optimizing the expansion of a district thermal network.
    """
    
    def __init__(self, locator: cea.inputlocator.InputLocator, network_type: str, metrics_df: pd.DataFrame,
                 num_phases: int = 3, phase_durations: Optional[List[int]] = None,
                 capex_budget_per_phase: Optional[List[float]] = None,
                 total_expenditure_budget_per_phase: Optional[List[float]] = None,
                 interest_rate: float = 0.05, cost_model: str = 'detailed',
                 objective_function: str = 'NPV', diversity_factor: float = 0.7,
                 temperature_difference_dh: float = 20, temperature_difference_dc: float = 10,
                 pressure_loss_pa_per_m: float = 200, pump_operation_hours: int = 4000,
                 pump_efficiency: float = 0.8, pump_load_factor: float = 0.5,
                 pump_capex_a: float = 1230, pump_capex_b: float = 0.65,
                 cooling_cop: float = 4.0, ghg_budget_per_phase: Optional[List[float]] = None,
                 multi_objective_mode: bool = False, multi_objective_functions: Optional[List[str]] = None,
                 testing_clusters=None):
        """
        Initialize the optimizer.
        
        Parameters:
        -----------
        locator : cea.inputlocator.InputLocator
            The locator for the scenario
        network_type : str
            The network type (DH or DC)
        metrics_df : pd.DataFrame
            DataFrame with metrics for each cluster
        num_phases : int
            Number of phases for the expansion
        phase_durations : list
            List of durations for each phase in years
        capex_budget_per_phase : list
            List of CAPEX budgets for each phase
        total_expenditure_budget_per_phase : list
            List of total expenditure budgets for each phase
        interest_rate : float
            Interest rate for financial calculations
        cost_model : str
            Cost model to use (detailed or simplified)
        objective_function : str
            Objective function to optimize (NPV, ROI, or emissions)
        diversity_factor : float
            Diversity factor for pipe sizing
        temperature_difference_dh : float
            Temperature difference for district heating
        temperature_difference_dc : float
            Temperature difference for district cooling
        pressure_loss_pa_per_m : float
            Pressure loss per meter of pipe
        pump_operation_hours : int
            Annual operation hours for pumps
        pump_efficiency : float
            Pump efficiency
        pump_load_factor : float
            Pump load factor
        pump_capex_a : float
            Pump CAPEX parameter a
        pump_capex_b : float
            Pump CAPEX parameter b
        cooling_cop : float
            Coefficient of performance for cooling
        ghg_budget_per_phase : list
            List of GHG budgets for each phase
        multi_objective_mode : bool
            Whether to use multi-objective optimization
        multi_objective_functions : list
            List of objective functions for multi-objective optimization
        testing_clusters : list
            List of cluster IDs to include in the optimization
        """
        self.locator = locator
        self.network_type = network_type
        self.metrics_df = metrics_df
        self.num_phases = num_phases
        self.phase_durations = phase_durations if phase_durations else [10] * num_phases
        self.capex_budget_per_phase = capex_budget_per_phase
        self.total_expenditure_budget_per_phase = total_expenditure_budget_per_phase
        self.interest_rate = interest_rate
        self.cost_model = cost_model
        self.objective_function = objective_function
        self.diversity_factor = diversity_factor
        self.temperature_difference_dh = temperature_difference_dh
        self.temperature_difference_dc = temperature_difference_dc
        self.pressure_loss_pa_per_m = pressure_loss_pa_per_m
        self.pump_operation_hours = pump_operation_hours
        self.pump_efficiency = pump_efficiency
        self.pump_load_factor = pump_load_factor
        self.pump_capex_a = pump_capex_a
        self.pump_capex_b = pump_capex_b
        self.cooling_cop = cooling_cop
        self.ghg_budget_per_phase = ghg_budget_per_phase
        self.multi_objective_mode = multi_objective_mode
        self.multi_objective_functions = multi_objective_functions
        self.testing_clusters = testing_clusters
        
        # Set up the energy price
        self.energy_price = self._get_energy_price()
        
        # Load cost data
        self._load_cost_data()
        
        # Load cluster data
        self._load_cluster_data()
        
        # Create a mapping of cluster metrics
        self._create_cluster_metrics_mapping()
        
        # Initialize the solution
        self.solution = None
        
        # Initialize the emissions cache
        self.emissions_cache = {}
        
        # Initialize the current individual
        self.current_individual = None
        
        # Set up the genetic algorithm
        self._setup_genetic_algorithm()
        
        # Log initialization
        log().info(f"Initialized DTNExpansionOptimizer with:")
        log().info(f"  Network type: {self.network_type}")
        log().info(f"  Number of phases: {self.num_phases}")
        log().info(f"  Phase durations: {self.phase_durations}")
        log().info(f"  Interest rate: {self.interest_rate}")
        log().info(f"  Cost model: {self.cost_model}")
        log().info(f"  Objective function: {self.objective_function}")
        log().info(f"  Multi-objective mode: {self.multi_objective_mode}")
        log().info(f"  Multi-objective functions: {self.multi_objective_functions}")
        log().info(f"  Testing clusters: {self.testing_clusters}")
    
    def _get_energy_price(self):
        """
        Read energy price from FEEDSTOCKS.xlsx based on network type.

        For DH: Uses the buy price from appropriate heating feedstock (default: NATURALGAS)
        For DC: Uses the buy price from GRID (electricity for cooling)

        Returns:
        --------
        float
            Energy price in USD/kWh
        
        Raises:
        -------
        ValueError
            If the energy price cannot be read from the feedstock data
        """
        # Determine which feedstock to use based on network type
        if self.network_type == 'DH':
            # For district heating, use NATURALGAS
            feedstock_name = 'NATURALGAS'
        else:
            # For district cooling, use GRID (electricity)
            feedstock_name = 'GRID'

        # Get the feedstock file path
        try:
            feedstock_file = self.locator.get_db4_components_feedstocks_feedstocks_csv(feedstocks=feedstock_name)
            log().info(f"Reading energy price from {feedstock_file}")
        except Exception as e:
            log().error(f"Could not find feedstock file for {feedstock_name}: {e}")
            raise ValueError(f"Could not find feedstock file for {feedstock_name}: {e}")

        # Read the feedstock data
        try:
            feedstock_data = pd.read_csv(feedstock_file)
        except Exception as e:
            log().error(f"Could not read feedstock data from {feedstock_file}: {e}")
            raise ValueError(f"Could not read feedstock data from {feedstock_file}: {e}")

        # Get the buy price column (Opex_var_buy_USD2015kWh)
        if 'Opex_var_buy_USD2015kWh' in feedstock_data.columns:
            # Calculate average price across all hours
            energy_price = feedstock_data['Opex_var_buy_USD2015kWh'].mean()
            log().info(f"Using energy price from {feedstock_name}: {energy_price:.4f} USD/kWh")
            return energy_price
        else:
            log().error(f"Column 'Opex_var_buy_USD2015kWh' not found in {feedstock_name} data.")
            raise ValueError(f"Column 'Opex_var_buy_USD2015kWh' not found in {feedstock_name} data.")
    
    def _load_cost_data(self):
        """Load cost data from dynamic DTN thermal network costs results."""
        try:
            # Load cost data from dynamic DTN thermal network costs results
            cost_file = Path(self.locator.get_dynamic_dtn_network_layout_costs_file(self.network_type))
            if not cost_file.exists():
                log().warning(f"Dynamic DTN cost file not found: {cost_file}. Falling back to original cost file.")
                cost_file = Path(self.locator.get_network_layout_costs_file(self.network_type))
                if not cost_file.exists():
                    raise FileNotFoundError(f"Cost file not found: {cost_file}. Please run Thermal Network Part 3 first.")

            log().info(f"Loading cost data from: {cost_file}")
            cost_data = pd.read_csv(cost_file)

            # Extract relevant cost components
            cost_components = {
                'capex_network_USD': cost_data['capex_network_USD'].iloc[0],
                'capex_pumps_USD': cost_data['capex_pumps_USD'].iloc[0],
                'capex_hex_USD': cost_data['capex_hex_USD'].iloc[0],
                'network_length_m': cost_data['network_length_m'].iloc[0],
            }

            # Calculate cost per meter of pipe (for simplified model)
            cost_components['cost_per_meter'] = cost_components['capex_network_USD'] / cost_components['network_length_m']

            # Load pipe cost data for detailed calculations
            try:
                # Load pipe cost data
                self.pipe_cost_df = pd.read_csv(self.locator.get_database_components_distribution_thermal_grid('THERMAL_GRID'))
                log().info(f"Loaded pipe cost data: {len(self.pipe_cost_df)} rows")
            except Exception as e:
                log().warning(f"Error loading pipe cost data: {e}")
                self.pipe_cost_df = None

            # Get pump cost data
            try:
                log().info("Trying to read pump cost data")
                # First try to get pump data from the temporary scenario
                pump_cost_file = self.locator.get_database_components_distribution_thermal_grid('PUMP')
                if os.path.exists(pump_cost_file):
                    self.pump_cost_df = pd.read_csv(pump_cost_file)
                    log().info(f"Loaded pump cost data: {len(self.pump_cost_df)} rows")
                else:
                    # If not found in temp scenario, use default pump cost data
                    log().info("Pump cost file not found in temporary scenario, using default pump cost data")
                    # Create default pump cost data
                    self.pump_cost_df = pd.DataFrame({
                        'code': ['PUMP'],
                        'InvC': [1000],
                        'InvC_unit': ['USD/kW'],
                        'maint': [0.05],
                        'maint_unit': ['% of InvC'],
                        'life_time': [20],
                        'life_time_unit': ['yr']
                    })
                    log().info("Created default pump cost data")
            except Exception as e:
                log().warning(f"Error reading pump cost data: {e}")
                log().warning("Creating default pump cost data")
                
                # Create default pump cost data
                self.pump_cost_df = pd.DataFrame({
                    'code': ['PUMP'],
                    'InvC': [1000],
                    'InvC_unit': ['USD/kW'],
                    'maint': [0.05],
                    'maint_unit': ['% of InvC'],
                    'life_time': [20],
                    'life_time_unit': ['yr']
                })
                log().info("Created default pump cost data")

            return cost_components
        except Exception as e:
            log().error(f"Error loading cost data: {e}")
            raise ValueError(f"Error loading cost data: {e}")
    
    def _load_cluster_data(self):
        """Load cluster data from the metrics file."""
        try:
            # Check if 'cluster' column exists in metrics_df
            if 'cluster' in self.metrics_df.columns:
                # Original approach
                log().info("Using 'cluster' column from metrics file")
                self.all_clusters = sorted(self.metrics_df['cluster'].unique())
            else:
                # Alternative approach using 'clusters' column (similar to original DTN module)
                log().info("'cluster' column not found, using 'clusters' column instead")
                unique_clusters = set()
                for cluster_str in self.metrics_df['clusters']:
                    # Skip cluster 0 (existing DTN)
                    if cluster_str == '0':
                        continue
                    # Split the cluster string (e.g., '1+3+4') into individual clusters
                    for c in cluster_str.split('+'):
                        if c != '0':  # Skip cluster 0
                            unique_clusters.add(int(c))
                
                # Sort the clusters to maintain the same order as before
                self.all_clusters = sorted(list(unique_clusters))
                log().info(f"Extracted clusters from 'clusters' column: {self.all_clusters}")
            
            # Filter clusters if testing_clusters is specified
            if self.testing_clusters:
                log().info(f"Filtering clusters to include only testing clusters: {self.testing_clusters}")
                self.all_clusters = [c for c in self.all_clusters if c in self.testing_clusters]
                log().info(f"Filtered clusters: {self.all_clusters}")
            
            # Get buildings in each cluster
            self.buildings_by_cluster = {}
            
            # Check if 'buildings' column exists in metrics_df
            if 'buildings' in self.metrics_df.columns:
                log().info("Using 'buildings' column from metrics file")
                for cluster in self.all_clusters:
                    cluster_rows = self.metrics_df[self.metrics_df['cluster'] == cluster]
                    if not cluster_rows.empty:
                        buildings = cluster_rows['buildings'].iloc[0]
                        if isinstance(buildings, str):
                            buildings = buildings.split(',')
                        self.buildings_by_cluster[cluster] = buildings
                        log().info(f"Cluster {cluster} has {len(buildings)} buildings")
            else:
                # Alternative approach: load from cluster_nodes.csv in the original DTN results
                log().info("'buildings' column not found, loading from cluster_nodes.csv")
                try:
                    # Use the original locator to get the cluster nodes file
                    original_locator = cea.inputlocator.InputLocator(self.locator.original_locator.scenario)
                    cluster_nodes_path = Path(original_locator.get_dtn_cluster_nodes_file())
                    if cluster_nodes_path.exists():
                        log().info(f"Loading cluster nodes from: {cluster_nodes_path}")
                        cluster_nodes = pd.read_csv(cluster_nodes_path)
                        for cluster in self.all_clusters:
                            buildings = cluster_nodes[cluster_nodes['cluster'] == cluster]['building'].tolist()
                            self.buildings_by_cluster[cluster] = buildings
                            log().info(f"Cluster {cluster} has {len(buildings)} buildings")
                    else:
                        log().warning(f"Cluster nodes file not found: {cluster_nodes_path}")
                except Exception as e:
                    log().warning(f"Error loading cluster nodes data: {e}")
            
            # Get all buildings in testing clusters
            if self.testing_clusters:
                self.buildings_in_testing_clusters = []
                for cluster in self.testing_clusters:
                    if cluster in self.buildings_by_cluster:
                        self.buildings_in_testing_clusters.extend(self.buildings_by_cluster[cluster])
                log().info(f"Total buildings in testing clusters: {len(self.buildings_in_testing_clusters)}")
            
            # Load total demand data
            self.total_demand = pd.read_csv(self.locator.get_total_demand())
            log().info(f"Loaded total demand data with {len(self.total_demand)} rows")
            
            # Load network data
            self._load_network_data()
            
        except Exception as e:
            log().error(f"Error loading cluster data: {e}")
            raise
    
    def _load_network_data(self):
        """Load network data from the thermal network files."""
        try:
            # Load network edges using the proper InputLocator method
            network_edges_file = self.locator.get_network_layout_edges_shapefile(self.network_type)
            log().info(f"Loading network edges from: {network_edges_file}")
            self.network_edges = _read_shp_force_2d(network_edges_file)
            
            # Load network nodes using the proper InputLocator method
            network_nodes_file = self.locator.get_network_layout_nodes_shapefile(self.network_type)
            log().info(f"Loading network nodes from: {network_nodes_file}")
            self.network_nodes = _read_shp_force_2d(network_nodes_file)
            
            # Create a mapping of edges to clusters
            self._create_edge_cluster_mapping()
            
        except Exception as e:
            log().error(f"Error loading network data: {e}")
            raise
    
    def _create_edge_cluster_mapping(self):
        """Create a mapping of edges to clusters."""
        try:
            # Create a mapping of edges to clusters
            self.edge_cluster_map = {}
            for cluster in self.all_clusters:
                # Get buildings in this cluster
                buildings = self.buildings_by_cluster[cluster]
                
                # Get edges connected to these buildings
                cluster_edges = self.network_edges[self.network_edges['name'].isin(buildings)]
                
                # Store edges for this cluster
                self.edge_cluster_map[cluster] = cluster_edges
                
            # Create a DataFrame with all cluster edges
            self.cluster_edges = pd.concat([self.edge_cluster_map[cluster] for cluster in self.all_clusters])
            
        except Exception as e:
            log().error(f"Error creating edge-cluster mapping: {e}")
            raise
    
    def _create_cluster_metrics_mapping(self):
        """Create a mapping of cluster metrics."""
        try:
            # Create a mapping of cluster metrics
            self.cluster_metrics = {}
            
            # Directly use the cluster nodes file which contains the 'cluster' column
            # Use the original locator to get the cluster nodes file
            original_locator = cea.inputlocator.InputLocator(self.locator.original_locator.scenario)
            cluster_nodes_path = Path(original_locator.get_dtn_cluster_nodes_file())
            if cluster_nodes_path.exists():
                log().info(f"Loading cluster nodes from: {cluster_nodes_path}")
                cluster_nodes_df = pd.read_csv(cluster_nodes_path)
                
                # Group by cluster and create metrics for each cluster
                for cluster in self.all_clusters:
                    # Get buildings in this cluster
                    buildings = cluster_nodes_df[cluster_nodes_df['cluster'] == cluster]['building'].tolist()
                    
                    # Create metrics for this cluster
                    self.cluster_metrics[cluster] = {
                        'cluster': cluster,
                        'buildings': buildings,
                        'num_buildings': len(buildings)
                    }
                    
                    # Add additional metrics if available in metrics_df
                    cluster_rows = self.metrics_df[self.metrics_df.get('cluster', -1) == cluster]
                    if not cluster_rows.empty:
                        # Update with metrics from metrics_df
                        self.cluster_metrics[cluster].update(cluster_rows.iloc[0].to_dict())
                
                log().info(f"Created metrics for {len(self.cluster_metrics)} clusters")
            else:
                log().warning(f"Cluster nodes file not found: {cluster_nodes_path}")
                raise FileNotFoundError(f"Cluster nodes file not found: {cluster_nodes_path}")
                
        except Exception as e:
            log().error(f"Error creating cluster metrics mapping: {e}")
            raise
    
    def _setup_genetic_algorithm(self):
        """Set up the genetic algorithm."""
        try:
            # Set up the creator
            setup_creator(multi_objective=self.multi_objective_mode, 
                         objective_function=self.objective_function,
                         multi_objective_functions=self.multi_objective_functions)
            
            # Create a toolbox
            self.toolbox = base.Toolbox()
            
            # Register attribute generator
            self.toolbox.register("attr_phase", random.randint, 0, self.num_phases)
            
            # Register individual creator
            self.toolbox.register("individual", tools.initRepeat, creator.Individual, 
                                 self.toolbox.attr_phase, n=len(self.all_clusters))
            
            # Register population creator
            self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
            
            # Register evaluation function
            self.toolbox.register("evaluate", self.evaluate_individual)
            
            # Register genetic operators
            self.toolbox.register("mate", tools.cxTwoPoint)
            self.toolbox.register("mutate", tools.mutUniformInt, low=0, up=self.num_phases, indpb=0.1)
            self.toolbox.register("select", tools.selTournament, tournsize=3)
            
        except Exception as e:
            log().error(f"Error setting up genetic algorithm: {e}")
            raise
            
    def _get_buildings_in_specific_cluster(self, cluster):
        """Get buildings in a specific cluster."""
        if cluster in self.buildings_by_cluster:
            return self.buildings_by_cluster[cluster]
        return []
        
    def _get_required_pipes(self, clusters, edges_df):
        """Get required pipes for a set of clusters."""
        # Create a list of buildings in the specified clusters
        buildings = []
        for cluster in clusters:
            buildings.extend(self._get_buildings_in_specific_cluster(cluster))
            
        # Filter edges to only include those connected to the buildings
        required_pipes = edges_df[edges_df['name'].isin(buildings)]
        
        return required_pipes
        
    def calculate_district_emissions_new(self):
        """
        Calculate district operation emissions using the new methodology.
        
        Workflow:
        1. Read from the generated genome to understand what cluster(s) it decides to connect at what phase
        2. Identify corresponding building IDs of each cluster
        3. Read from the Total_LCA_operation.csv, sum up the "GHG_sys_tonCO2" column for phase 0
        4. Read from the building-properties/supply.csv
        5. Make n copies of it if there are n phases in total
        6. Modify the supplied technologies for cooling, heating, dhw, of newly cluster(s) in phase 1 to those for cluster 0
        7. Run the LCA operation module again, repeat step 3 calculation for phase 1
        8. Repeat steps 6 to 7 for the following phases
        9. For the row of Total in DTN optimization result csv, copy from the last phase
        
        Returns:
        --------
        dict
            Dictionary with district operation emission and district operation emission per GFA for each phase
        """
        # Initialize results dictionary
        results = {}
        
        # Get original supply file
        supply_file = self.locator.get_building_supply()
        original_supply_df = pd.read_csv(supply_file)
        
        # Get district supply systems from cluster 0 buildings
        cluster0_buildings = self._get_buildings_in_specific_cluster(0)
        if not cluster0_buildings:
            log().warning("No buildings found in cluster 0 (existing DTN)")
            return {}
            
        # Get supply systems used by cluster 0 (existing DTN)
        district_supply_systems = original_supply_df[original_supply_df['name'].isin(cluster0_buildings)]
        
        # Extract district supply types
        district_heating_system = district_supply_systems['supply_type_hs'].iloc[0]
        district_cooling_system = district_supply_systems['supply_type_cs'].iloc[0]
        district_dhw_system = district_supply_systems['supply_type_dhw'].iloc[0]
        
        # Create directory for phase-specific supply files
        # Use the dedicated method to get the phase supply files folder
        phase_files_dir = Path(self.locator.get_dynamic_dtn_optimization_phase_supply_files_folder())
        phase_files_dir.mkdir(parents=True, exist_ok=True)
        
        # Create phase 0 supply file (original)
        phase0_supply_path = phase_files_dir / "phase0_supply.csv"
        original_supply_df.to_csv(phase0_supply_path, index=False)
        log().info(f"Created phase 0 supply file: {phase0_supply_path}")
        
        # Phase 0: Calculate emissions for the whole district with cluster 0 using district systems
        # and all other clusters using building-scale systems
        
        # Run LCA operation module with original supply file
        lca_operation(self.locator, custom_supply_path=str(phase0_supply_path))
        
        # Load LCA results
        lca_operation_results = pd.read_csv(self.locator.get_lca_operation())
        
        # Filter LCA results to only include buildings in testing clusters if specified
        if hasattr(self, 'testing_clusters') and self.testing_clusters:
            log().info(f"Filtering emissions to only include buildings in testing clusters: {self.testing_clusters}")
            lca_operation_results = lca_operation_results[lca_operation_results['name'].isin(self.buildings_in_testing_clusters)]
            log().info(f"Filtered to {len(lca_operation_results)} buildings for emissions calculation")
            
        # Calculate total GHG emissions and GFA for phase 0
        total_ghg = lca_operation_results['GHG_sys_tonCO2'].sum()
        total_gfa = lca_operation_results['GFA_m2'].sum()
        
        # Calculate emissions per GFA
        ghg_per_gfa = total_ghg / total_gfa if total_gfa > 0 else 0
        
        # Store results for phase 0
        results[0] = {
            'district_operation_emission [t CO2eq/yr]': total_ghg,
            'district_operation_emission_per_gfa [kg CO2eq/yr/m2]': ghg_per_gfa * 1000  # Convert to kg
        }
        
        # Get cluster assignments from current individual
        if hasattr(self, 'current_individual') and self.current_individual:
            # Use the current individual's phase assignments
            cluster_phase_map = {cluster: p for cluster, p in zip(self.all_clusters, self.current_individual)}
            clusters_by_phase = {}
            for phase in range(1, self.num_phases + 1):
                clusters_by_phase[phase] = [cluster for cluster, p in cluster_phase_map.items() if p == phase]
        elif self.solution and 'genome' in self.solution:
            # Use the solution's genome if available
            cluster_phase_map = {cluster: p for cluster, p in zip(self.all_clusters, self.solution['genome'])}
            clusters_by_phase = {}
            for phase in range(1, self.num_phases + 1):
                clusters_by_phase[phase] = [cluster for cluster, p in cluster_phase_map.items() if p == phase]
        else:
            # Fallback: distribute all_clusters evenly across phases
            log().warning("No current_individual or solution genome found, using fallback cluster distribution")
            clusters_by_phase = {}
            clusters_per_phase = max(1, len(self.all_clusters) // self.num_phases)
            for phase in range(1, self.num_phases + 1):
                start_idx = (phase - 1) * clusters_per_phase
                end_idx = min(phase * clusters_per_phase, len(self.all_clusters))
                if phase == self.num_phases:  # Ensure last phase gets any remaining clusters
                    end_idx = len(self.all_clusters)
                clusters_by_phase[phase] = self.all_clusters[start_idx:end_idx]
                
        # Log the cluster assignments for debugging
        for phase, clusters in clusters_by_phase.items():
            log().info(f"Phase {phase}: Connected clusters {clusters}")
            
        # Process each phase
        for phase in range(1, self.num_phases + 1):
            # Get clusters connected in this phase
            newly_connected_clusters = clusters_by_phase.get(phase, [])
            
            # Skip if no clusters are connected in this phase
            if not newly_connected_clusters:
                log().info(f"No clusters connected in phase {phase}, skipping")
                continue
                
            # Get all clusters connected up to this phase
            cumulative_clusters = set()
            for p in range(1, phase + 1):
                cumulative_clusters.update(clusters_by_phase.get(p, []))
                
            # Always include cluster 0 (existing DTN)
            cumulative_clusters.add(0)
            
            # Create a copy of the original supply DataFrame
            phase_supply_df = original_supply_df.copy()
            
            # Get buildings in the newly connected clusters
            newly_connected_buildings = []
            for cluster in newly_connected_clusters:
                newly_connected_buildings.extend(self._get_buildings_in_specific_cluster(cluster))
                
            # Update supply systems for newly connected buildings
            for building in newly_connected_buildings:
                # Find the row for this building
                building_idx = phase_supply_df[phase_supply_df['name'] == building].index
                
                # Skip if building not found
                if len(building_idx) == 0:
                    log().warning(f"Building {building} not found in supply file, skipping")
                    continue
                    
                # Update supply systems
                phase_supply_df.loc[building_idx, 'supply_type_hs'] = district_heating_system
                phase_supply_df.loc[building_idx, 'supply_type_cs'] = district_cooling_system
                phase_supply_df.loc[building_idx, 'supply_type_dhw'] = district_dhw_system
                
            # Save the phase supply file
            phase_supply_path = phase_files_dir / f"phase{phase}_supply.csv"
            phase_supply_df.to_csv(phase_supply_path, index=False)
            log().info(f"Created phase {phase} supply file: {phase_supply_path}")
            
            # Run LCA operation module with the phase supply file
            lca_operation(self.locator, custom_supply_path=str(phase_supply_path))
            
            # Load LCA results
            lca_operation_results = pd.read_csv(self.locator.get_lca_operation())
            
            # Filter LCA results to only include buildings in testing clusters if specified
            if hasattr(self, 'testing_clusters') and self.testing_clusters:
                lca_operation_results = lca_operation_results[lca_operation_results['name'].isin(self.buildings_in_testing_clusters)]
                
            # Calculate total GHG emissions and GFA for this phase
            total_ghg = lca_operation_results['GHG_sys_tonCO2'].sum()
            total_gfa = lca_operation_results['GFA_m2'].sum()
            
            # Calculate emissions per GFA
            ghg_per_gfa = total_ghg / total_gfa if total_gfa > 0 else 0
            
            # Store results for this phase
            results[phase] = {
                'district_operation_emission [t CO2eq/yr]': total_ghg,
                'district_operation_emission_per_gfa [kg CO2eq/yr/m2]': ghg_per_gfa * 1000  # Convert to kg
            }
            
            # Log emissions for this phase
            log().info(f"Phase {phase}: Emissions = {total_ghg:.2f} t CO2eq/yr")
            
        return results
        
    def evaluate_individual(self, individual):
        """
        Evaluate an individual.
        
        Parameters:
        -----------
        individual : list
            List of phase assignments for each cluster
            
        Returns:
        --------
        tuple
            Tuple with fitness value(s)
        """
        # Store the current individual for use in other methods
        self.current_individual = individual
        
        # Create a mapping of clusters to phases
        cluster_phase_map = {cluster: phase for cluster, phase in zip(self.all_clusters, individual)}
        
        # Group clusters by phase
        clusters_by_phase = {}
        for phase in range(1, self.num_phases + 1):
            clusters_by_phase[phase] = [cluster for cluster, p in cluster_phase_map.items() if p == phase]
            
        # Check if any clusters are assigned to phase 0 (not connected)
        unconnected_clusters = [cluster for cluster, phase in cluster_phase_map.items() if phase == 0]
        
        # Calculate emissions for this individual
        emissions_results = self.calculate_district_emissions_new()
        
        # Check if emissions results are valid
        if not emissions_results:
            log().warning("No emissions results calculated, returning penalty fitness")
            if self.objective_function.lower() == 'emissions':
                return (1e10,)  # Large penalty for emissions minimization
            else:
                return (-1e10,)  # Large penalty for NPV/ROI maximization
                
        # Get the final phase emissions
        final_phase = max(emissions_results.keys())
        final_emissions = emissions_results[final_phase]['district_operation_emission [t CO2eq/yr]']
        
        # Calculate CAPEX for each phase
        total_capex = 0
        for phase, clusters in clusters_by_phase.items():
            if not clusters:
                continue
                
            # Calculate CAPEX for this phase
            phase_capex, _ = self._calculate_phase_capex(clusters, phase)
            total_capex += phase_capex
            
            # Check if CAPEX budget is exceeded
            if self.capex_budget_per_phase and phase - 1 < len(self.capex_budget_per_phase):
                if phase_capex > self.capex_budget_per_phase[phase - 1]:
                    log().warning(f"Phase {phase} CAPEX ({phase_capex}) exceeds budget ({self.capex_budget_per_phase[phase - 1]})")
                    if self.objective_function.lower() == 'emissions':
                        return (1e10,)  # Large penalty for emissions minimization
                    else:
                        return (-1e10,)  # Large penalty for NPV/ROI maximization
                        
        # Calculate NPV for the entire project
        total_npv = 0
        for phase, clusters in clusters_by_phase.items():
            if not clusters:
                continue
                
            # Calculate NPV for this phase
            phase_npv = self.calculate_npv(tuple(clusters), phase)
            total_npv += phase_npv
            
        # Calculate ROI for the entire project
        if total_capex > 0:
            total_roi = total_npv / total_capex
        else:
            total_roi = 0
            
        # Return fitness based on objective function
        if self.objective_function.lower() == 'emissions':
            return (final_emissions,)
        elif self.objective_function.lower() == 'npv':
            return (total_npv,)
        elif self.objective_function.lower() == 'roi':
            return (total_roi,)
        else:
            log().warning(f"Unknown objective function: {self.objective_function}, using NPV")
            return (total_npv,)
            
    def _calculate_phase_capex(self, clusters, phase):
        """
        Calculate CAPEX for a phase.
        
        Parameters:
        -----------
        clusters : list
            List of clusters to connect in this phase
        phase : int
            Phase number
            
        Returns:
        --------
        tuple
            Tuple with total CAPEX and CAPEX breakdown
        """
        try:
            # Get required pipes for these clusters
            required_pipes = self._get_required_pipes(clusters, self.cluster_edges)
            
            # Calculate pipe CAPEX
            pipe_capex = self._calculate_pipe_capex(required_pipes)
            
            # Calculate pump CAPEX
            pump_capex = self._calculate_pump_capex(required_pipes)
            
            # Calculate total CAPEX
            total_capex = pipe_capex + pump_capex
            
            # Create CAPEX breakdown
            capex_breakdown = {
                'pipe_capex': pipe_capex,
                'pump_capex': pump_capex,
                'total_capex': total_capex
            }
            
            return total_capex, capex_breakdown
            
        except Exception as e:
            log().error(f"Error calculating CAPEX for phase {phase}: {e}")
            return 0, {}
            
    def _calculate_pipe_capex(self, pipes_df):
        """
        Calculate pipe CAPEX.
        
        Parameters:
        -----------
        pipes_df : pd.DataFrame
            DataFrame with pipes
            
        Returns:
        --------
        float
            Pipe CAPEX
        """
        try:
            # Calculate pipe CAPEX based on pipe diameter and length
            pipe_capex = 0
            
            # Iterate over pipes
            for _, pipe in pipes_df.iterrows():
                # Get pipe diameter and length
                pipe_diameter = pipe['Pipe_DN']
                pipe_length = pipe['length_m']
                
                # Find the cost for this pipe diameter
                pipe_cost_row = self.pipe_cost_df[self.pipe_cost_df['Pipe_DN'] == pipe_diameter]
                
                # Skip if pipe diameter not found
                if len(pipe_cost_row) == 0:
                    log().warning(f"Pipe diameter {pipe_diameter} not found in cost data, skipping")
                    continue
                    
                # Get pipe cost per meter
                pipe_cost_per_m = pipe_cost_row['InvC'].iloc[0]
                
                # Calculate pipe cost
                pipe_cost = pipe_cost_per_m * pipe_length
                
                # Add to total pipe CAPEX
                pipe_capex += pipe_cost
                
            return pipe_capex
            
        except Exception as e:
            log().error(f"Error calculating pipe CAPEX: {e}")
            return 0
            
    def _calculate_pump_capex(self, pipes_df):
        """
        Calculate pump CAPEX using the same approach as DTN expansion optimization.
        
        Parameters:
        -----------
        pipes_df : pd.DataFrame
            DataFrame with pipes
            
        Returns:
        --------
        float
            Pump CAPEX
        """
        try:
            # Calculate total pipe length
            total_pipe_length = pipes_df['length_m'].sum()
            
            # Skip if no pipes
            if total_pipe_length == 0:
                return 0
                
            # Get buildings connected to these pipes
            buildings = pipes_df['name'].unique()
            
            # Get temperature difference based on network type
            if self.network_type == 'DH':
                temp_diff = self.temperature_difference_dh
                # Calculate total heating demand
                annual_demand_mwh = self.total_demand[
                    self.total_demand['name'].isin(buildings)
                ]['Qhs_sys_MWhyr'].sum() + self.total_demand[
                    self.total_demand['name'].isin(buildings)
                ]['Qww_sys_MWhyr'].sum()
            else:
                temp_diff = self.temperature_difference_dc
                # Calculate total cooling demand
                annual_demand_mwh = self.total_demand[
                    self.total_demand['name'].isin(buildings)
                ]['Qcs_sys_MWhyr'].sum() + self.total_demand[
                    self.total_demand['name'].isin(buildings)
                ]['Qcre_sys_MWhyr'].sum() + self.total_demand[
                    self.total_demand['name'].isin(buildings)
                ]['Qcdata_sys_MWhyr'].sum()
            
            # Convert annual demand to peak demand (kW) using the diversity factor
            peak_demand_kw = annual_demand_mwh * 1000 / 2000 / self.diversity_factor  # Assuming 2000 equivalent full load hours
            
            # Calculate mass flow rate (kg/s)
            mass_flow_rate = peak_demand_kw / (HEAT_CAPACITY_OF_WATER_JPERKGK * temp_diff / 1000)
            
            # Calculate pressure loss
            pressure_loss = self.pressure_loss_pa_per_m * total_pipe_length
            
            # Calculate pump power (W)
            if mass_flow_rate > 0:
                pump_power = mass_flow_rate * pressure_loss / (self.pump_efficiency * 1000)  # W
            else:
                pump_power = 0
            
            # Calculate pump CAPEX based on pump power using configurable formula
            pump_capex = self.pump_capex_a * (pump_power / 1000) ** self.pump_capex_b  # USD
            
            # Also calculate annual pump electricity consumption for O&M cost
            self.annual_pump_electricity = pump_power * self.pump_operation_hours * self.pump_load_factor  # Wh
            
            return pump_capex
            
        except Exception as e:
            log().error(f"Error calculating pump CAPEX: {e}")
            return 0
            
    def _calculate_flow_rate_heating(self, pipes_df):
        """
        Calculate flow rate for heating.
        
        Parameters:
        -----------
        pipes_df : pd.DataFrame
            DataFrame with pipes
            
        Returns:
        --------
        float
            Flow rate in m3/s
        """
        try:
            # Get buildings connected to these pipes
            buildings = pipes_df['name'].unique()
            
            # Calculate total heating demand
            total_heating_demand = self.total_demand[
                self.total_demand['name'].isin(buildings)
            ]['Qhs_sys_MWhyr'].sum() + self.total_demand[
                self.total_demand['name'].isin(buildings)
            ]['Qww_sys_MWhyr'].sum()
            
            # Convert from MWh/yr to W
            total_heating_demand_w = total_heating_demand * 1e6 / 8760
            
            # Apply diversity factor
            peak_heating_demand_w = total_heating_demand_w / self.diversity_factor
            
            # Calculate flow rate
            flow_rate = peak_heating_demand_w / (HEAT_CAPACITY_OF_WATER_JPERKGK * self.temperature_difference_dh * 1000)  # m3/s
            
            return flow_rate
            
        except Exception as e:
            log().error(f"Error calculating flow rate for heating: {e}")
            return 0
            
    def _calculate_flow_rate_cooling(self, pipes_df):
        """
        Calculate flow rate for cooling.
        
        Parameters:
        -----------
        pipes_df : pd.DataFrame
            DataFrame with pipes
            
        Returns:
        --------
        float
            Flow rate in m3/s
        """
        try:
            # Get buildings connected to these pipes
            buildings = pipes_df['name'].unique()
            
            # Calculate total cooling demand
            total_cooling_demand = self.total_demand[
                self.total_demand['name'].isin(buildings)
            ]['Qcs_sys_MWhyr'].sum() + self.total_demand[
                self.total_demand['name'].isin(buildings)
            ]['Qcre_sys_MWhyr'].sum() + self.total_demand[
                self.total_demand['name'].isin(buildings)
            ]['Qcdata_sys_MWhyr'].sum()
            
            # Convert from MWh/yr to W
            total_cooling_demand_w = total_cooling_demand * 1e6 / 8760
            
            # Apply diversity factor
            peak_cooling_demand_w = total_cooling_demand_w / self.diversity_factor
            
            # Calculate flow rate
            flow_rate = peak_cooling_demand_w / (HEAT_CAPACITY_OF_WATER_JPERKGK * self.temperature_difference_dc * 1000)  # m3/s
            
            return flow_rate
            
        except Exception as e:
            log().error(f"Error calculating flow rate for cooling: {e}")
            return 0
            
    def calculate_npv(self, clusters, phase):
        """
        Calculate NPV for a set of clusters.
        
        Parameters:
        -----------
        clusters : tuple
            Tuple of clusters
        phase : int
            Phase number
            
        Returns:
        --------
        float
            NPV
        """
        try:
            # Convert clusters to list if it's a tuple
            clusters = list(clusters) if isinstance(clusters, tuple) else clusters
            
            # Skip if no clusters
            if not clusters:
                return 0
                
            # Calculate CAPEX
            capex, _ = self._calculate_phase_capex(clusters, phase)
            
            # Calculate annual revenue
            annual_revenue = self._calculate_annual_revenue(clusters)
            
            # Calculate annual O&M cost
            annual_om_cost = self._calculate_annual_om_cost(clusters)
            
            # Calculate annual cash flow
            annual_cash_flow = annual_revenue - annual_om_cost
            
            # Calculate NPV
            npv = -capex  # Initial investment
            
            # Get phase duration
            phase_duration = self.phase_durations[phase - 1] if phase - 1 < len(self.phase_durations) else 10
            
            # Calculate NPV for each year
            for year in range(1, phase_duration + 1):
                npv += annual_cash_flow / ((1 + self.interest_rate) ** year)
                
            return npv
            
        except Exception as e:
            log().error(f"Error calculating NPV for phase {phase}: {e}")
            return 0
            
    def calculate_roi(self, clusters, phase):
        """
        Calculate ROI for a set of clusters.
        
        Parameters:
        -----------
        clusters : tuple
            Tuple of clusters
        phase : int
            Phase number
            
        Returns:
        --------
        float
            ROI
        """
        try:
            # Convert clusters to list if it's a tuple
            clusters = list(clusters) if isinstance(clusters, tuple) else clusters
            
            # Skip if no clusters
            if not clusters:
                return 0
                
            # Calculate CAPEX
            capex, _ = self._calculate_phase_capex(clusters, phase)
            
            # Skip if CAPEX is zero
            if capex == 0:
                return 0
                
            # Calculate NPV
            npv = self.calculate_npv(clusters, phase)
            
            # Calculate ROI
            roi = npv / capex
            
            return roi
            
        except Exception as e:
            log().error(f"Error calculating ROI for phase {phase}: {e}")
            return 0
            
    def _calculate_annual_revenue(self, clusters):
        """
        Calculate annual revenue for a set of clusters.
        
        Parameters:
        -----------
        clusters : list
            List of clusters
            
        Returns:
        --------
        float
            Annual revenue
        """
        try:
            # Get buildings in these clusters
            buildings = []
            for cluster in clusters:
                buildings.extend(self._get_buildings_in_specific_cluster(cluster))
                
            # Calculate total demand
            if self.network_type == 'DH':
                # For district heating, use heating demand
                total_demand = self.total_demand[
                    self.total_demand['name'].isin(buildings)
                ]['Qhs_sys_MWhyr'].sum() + self.total_demand[
                    self.total_demand['name'].isin(buildings)
                ]['Qww_sys_MWhyr'].sum()
            else:
                # For district cooling, use cooling demand
                total_demand = self.total_demand[
                    self.total_demand['name'].isin(buildings)
                ]['Qcs_sys_MWhyr'].sum() + self.total_demand[
                    self.total_demand['name'].isin(buildings)
                ]['Qcre_sys_MWhyr'].sum() + self.total_demand[
                    self.total_demand['name'].isin(buildings)
                ]['Qcdata_sys_MWhyr'].sum()
                
            # Calculate annual revenue
            annual_revenue = total_demand * self.energy_price
            
            return annual_revenue
            
        except Exception as e:
            log().error(f"Error calculating annual revenue: {e}")
            return 0
            
    def _calculate_annual_om_cost(self, clusters):
        """
        Calculate annual O&M cost for a set of clusters.
        
        Parameters:
        -----------
        clusters : list
            List of clusters
            
        Returns:
        --------
        float
            Annual O&M cost
        """
        try:
            # Get required pipes for these clusters
            required_pipes = self._get_required_pipes(clusters, self.cluster_edges)
            
            # Calculate pipe O&M cost
            pipe_om_cost = self._calculate_pipe_om_cost(required_pipes)
            
            # Calculate pump O&M cost
            pump_om_cost = self._calculate_pump_om_cost(required_pipes)
            
            # Calculate total O&M cost
            total_om_cost = pipe_om_cost + pump_om_cost
            
            return total_om_cost
            
        except Exception as e:
            log().error(f"Error calculating annual O&M cost: {e}")
            return 0
            
    def _calculate_pipe_om_cost(self, pipes_df):
        """
        Calculate pipe O&M cost.
        
        Parameters:
        -----------
        pipes_df : pd.DataFrame
            DataFrame with pipes
            
        Returns:
        --------
        float
            Pipe O&M cost
        """
        try:
            # Calculate pipe O&M cost based on pipe CAPEX
            pipe_capex = self._calculate_pipe_capex(pipes_df)
            
            # Assume O&M cost is 1% of CAPEX
            pipe_om_cost = 0.01 * pipe_capex
            
            return pipe_om_cost
            
        except Exception as e:
            log().error(f"Error calculating pipe O&M cost: {e}")
            return 0
            
    def _calculate_pump_om_cost(self, pipes_df):
        """
        Calculate pump O&M cost based on electricity consumption.
        
        Parameters:
        -----------
        pipes_df : pd.DataFrame
            DataFrame with pipes
            
        Returns:
        --------
        float
            Pump O&M cost
        """
        try:
            # Calculate pump CAPEX first (this also calculates annual_pump_electricity)
            self._calculate_pump_capex(pipes_df)
            
            # Calculate pump O&M cost based on electricity consumption
            # Get energy price from feedstock data if available
            try:
                energy_price = self._get_energy_price()
            except:
                # Fallback to default price
                energy_price = 0.2  # USD/kWh
            
            # Convert Wh to kWh and calculate cost
            pump_om_cost = (self.annual_pump_electricity / 1000) * energy_price
            
            return pump_om_cost
            
        except Exception as e:
            log().error(f"Error calculating pump O&M cost: {e}")
            return 0
            
    def optimize(self, population_size=50, num_generations=30):
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
        dict or list
            Dictionary with the optimal solution (single-objective) or
            list of non-dominated solutions (multi-objective)
        """
        # Verify cluster 0 buildings have district supply systems
        self._verify_cluster0_supply_systems()
        
        # Create initial population
        pop = self.toolbox.population(n=population_size)
        
        # Add a list to track all evaluated individuals
        all_evaluated_individuals = []
        
        # Store the original evaluate function
        original_evaluate = self.toolbox.evaluate
        
        # Define a new evaluate function that tracks individuals
        def evaluate_and_track(individual):
            fitness = original_evaluate(individual)
            # Create a proper DEAP Individual copy using the creator
            if self.multi_objective_mode:
                ind_copy = creator.Individual(individual)
            else:
                ind_copy = creator.Individual(individual)
            ind_copy.fitness.values = fitness
            all_evaluated_individuals.append(ind_copy)
            return fitness
            
        # Replace the evaluate function in the toolbox
        self.toolbox.evaluate = evaluate_and_track
        
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
        
        # Assign fitness values to individuals
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
            
        if self.multi_objective_mode:
            # For multi-objective optimization, use NSGA-III
            
            # Determine number of objectives
            num_objectives = len(self.multi_objective_functions) if self.multi_objective_functions else 2
            num_objectives = min(num_objectives, 3)  # Limit to 3 objectives maximum
            
            # Reference point for NSGA-III (automatically determined)
            # Use different p values based on number of objectives to get reasonable number of reference points
            if num_objectives == 2:
                ref_points = tools.uniform_reference_points(num_objectives, p=12)
            elif num_objectives == 3:
                ref_points = tools.uniform_reference_points(num_objectives, p=6)
            else:
                ref_points = tools.uniform_reference_points(num_objectives, p=4)
                
            # Create the NSGA-III selection operator
            self.toolbox.register("select", tools.selNSGA3, ref_points=ref_points)
            
            # Track the Pareto front
            pareto = tools.ParetoFront()
            
            # Track statistics
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("avg", np.mean)
            stats.register("min", np.min)
            stats.register("max", np.max)
            
            # Use the custom evaluation function in the algorithm
            algorithms.eaMuPlusLambda(pop, self.toolbox, mu=population_size,
                                  lambda_=population_size,
                                  cxpb=0.5, mutpb=0.2,
                                  ngen=num_generations,
                                  stats=stats, halloffame=pareto, verbose=True)
                                  
            # Process the Pareto front solutions
            pareto_solutions = []
            for i, ind in enumerate(pareto):
                # Find the corresponding individual_id from all_evaluated_individuals
                individual_id = None
                ind_tuple = tuple(ind)
                
                # Search for this individual in all_evaluated_individuals to get its index
                for j, eval_ind in enumerate(all_evaluated_individuals):
                    if tuple(eval_ind) == ind_tuple:
                        individual_id = j
                        break
                        
                if individual_id is None:
                    log().warning(f"Could not find individual_id for Pareto solution {i}")
                    individual_id = f"unknown_{i}"
                    
                # Convert to cluster-phase mapping
                solution = {
                    'individual_id': individual_id,  # Use actual individual_id
                    'genome': list(ind),
                    'cluster_phase_map': {cluster: phase for cluster, phase in
                                         zip(self.all_clusters, ind) if phase > 0},
                    'phases': {}
                }
                
                # Add fitness values based on selected objectives
                objectives = self.multi_objective_functions
                if not objectives or len(objectives) == 0:
                    objectives = ['NPV', 'emissions']
                    
                # Limit to 3 objectives maximum
                objectives = objectives[:3]
                
                # Add fitness values to solution
                for i, obj in enumerate(objectives):
                    solution[f'fitness_{obj}'] = ind.fitness.values[i]
                    
                # Group clusters by phase
                for cluster, phase in solution['cluster_phase_map'].items():
                    if phase not in solution['phases']:
                        solution['phases'][phase] = []
                    solution['phases'][phase].append(cluster)
                    
                # Calculate metrics for each phase
                for phase, clusters in solution['phases'].items():
                    solution[f'phase_{phase}_roi'] = self.calculate_roi(tuple(clusters), phase)
                    solution[f'phase_{phase}_npv'] = self.calculate_npv(tuple(clusters), phase)
                    solution[f'phase_{phase}_capex'], _ = self._calculate_phase_capex(clusters, phase)
                    
                    # Get emissions for this phase
                    ind_tuple = tuple(ind)
                    if ind_tuple in self.emissions_cache:
                        phase_emissions, _ = self.emissions_cache[ind_tuple]
                        solution[f'phase_{phase}_emissions'] = phase_emissions[phase]['district_operation_emission [t CO2eq/yr]']
                        
                pareto_solutions.append(solution)
                
            # Create a scatter plot of all solutions tested during multi-objective optimization
            self.plot_multi_objective_results(all_evaluated_individuals, pareto, self.multi_objective_functions)
            
            # Save metrics for all evaluated individuals
            # Use the dedicated method to get the results folder
            self.save_all_evaluated_individuals(all_evaluated_individuals, 
                                               Path(self.locator.get_dynamic_dtn_optimization_results_folder()))
                                               
            return pareto_solutions
            
        else:
            # For single-objective optimization, use the original approach
            # Track the best individual
            hof = tools.HallOfFame(1)
            
            # Track statistics
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("avg", np.mean)
            stats.register("min", np.min)
            stats.register("max", np.max)
            
            # Use the custom evaluation function in the algorithm
            pop, logbook = algorithms.eaSimple(pop, self.toolbox, cxpb=0.5, mutpb=0.2,
                                              ngen=num_generations, stats=stats,
                                              halloffame=hof, verbose=True)
                                              
            # Get the best solution
            best_individual = hof[0]
            
            # Convert to cluster-phase mapping
            solution = {
                'genome': list(best_individual),
                'cluster_phase_map': {cluster: phase for cluster, phase in
                                     zip(self.all_clusters, best_individual) if phase > 0},
                'fitness': best_individual.fitness.values[0],
                'phases': {}
            }
            
            # Group clusters by phase
            for cluster, phase in solution['cluster_phase_map'].items():
                if phase not in solution['phases']:
                    solution['phases'][phase] = []
                solution['phases'][phase].append(cluster)
                
            # Calculate metrics for each phase
            for phase, clusters in solution['phases'].items():
                solution[f'phase_{phase}_roi'] = self.calculate_roi(tuple(clusters), phase)
                solution[f'phase_{phase}_npv'] = self.calculate_npv(tuple(clusters), phase)
                solution[f'phase_{phase}_capex'], _ = self._calculate_phase_capex(clusters, phase)
                
            # Save metrics for all evaluated individuals
            # Use the dedicated method to get the results folder
            self.save_all_evaluated_individuals(all_evaluated_individuals, 
                                               Path(self.locator.get_dynamic_dtn_optimization_results_folder()))
                                               
            return solution
            
    def _verify_cluster0_supply_systems(self):
        """Verify that cluster 0 buildings have district supply systems."""
        try:
            # Get buildings in cluster 0
            cluster0_buildings = self._get_buildings_in_specific_cluster(0)
            
            # Skip if no buildings in cluster 0
            if not cluster0_buildings:
                log().warning("No buildings found in cluster 0 (existing DTN)")
                return
                
            # Get supply systems for cluster 0 buildings
            supply_file = self.locator.get_building_supply()
            supply_df = pd.read_csv(supply_file)
            
            # Get supply systems database
            heating_file = self.locator.get_database_supply_assemblies()
            heating_df = pd.read_excel(heating_file, sheet_name='HEATING')
            cooling_df = pd.read_excel(heating_file, sheet_name='COOLING')
            dhw_df = pd.read_excel(heating_file, sheet_name='HOT_WATER')
            
            # Check if all cluster 0 buildings have district supply systems
            for _, row in supply_df[supply_df['name'].isin(cluster0_buildings)].iterrows():
                hs_code = row['supply_type_hs']
                cs_code = row['supply_type_cs']
                dhw_code = row['supply_type_dhw']
                
                hs_scale = heating_df[heating_df['code'] == hs_code]['scale'].iloc[0] if len(heating_df[heating_df['code'] == hs_code]) > 0 else 'UNKNOWN'
                cs_scale = cooling_df[cooling_df['code'] == cs_code]['scale'].iloc[0] if len(cooling_df[cooling_df['code'] == cs_code]) > 0 else 'UNKNOWN'
                dhw_scale = dhw_df[dhw_df['code'] == dhw_code]['scale'].iloc[0] if len(dhw_df[dhw_df['code'] == dhw_code]) > 0 else 'UNKNOWN'
                
                if hs_scale != 'DISTRICT' or cs_scale != 'DISTRICT' or dhw_scale != 'DISTRICT':
                    log().warning(f"Building {row['name']} in cluster 0 does not use DISTRICT scale systems: HS={hs_scale}, CS={cs_scale}, DHW={dhw_scale}. This will result in penalties for the optimization results.")
                    
        except Exception as e:
            log().error(f"Error verifying cluster 0 supply systems: {e}")
            
    def save_all_evaluated_individuals(self, all_individuals, output_dir):
        """
        Save metrics for all evaluated individuals during optimization to a CSV file.
        
        Parameters:
        -----------
        all_individuals : list
            List of all individuals evaluated during optimization
        output_dir : Path
            Output directory
            
        Returns:
        --------
        Path
            Path to the saved CSV file
        """
        try:
            # Create output directory
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create a list to store metrics for each individual
            metrics = []
            
            # Process each individual
            for i, ind in enumerate(all_individuals):
                # Create a dictionary with metrics for this individual
                ind_metrics = {
                    'individual_id': i,
                    'genome': str(list(ind)),
                }
                
                # Add fitness values
                if self.multi_objective_mode:
                    # For multi-objective optimization, add each objective
                    objectives = self.multi_objective_functions
                    if not objectives or len(objectives) == 0:
                        objectives = ['NPV', 'emissions']
                        
                    # Limit to 3 objectives maximum
                    objectives = objectives[:3]
                    
                    # Add fitness values
                    for j, obj in enumerate(objectives):
                        if j < len(ind.fitness.values):
                            ind_metrics[f'fitness_{obj}'] = ind.fitness.values[j]
                else:
                    # For single-objective optimization, add the fitness value
                    if self.objective_function.lower() == 'emissions':
                        ind_metrics['fitness_emissions'] = ind.fitness.values[0]
                    elif self.objective_function.lower() == 'npv':
                        ind_metrics['fitness_npv'] = ind.fitness.values[0]
                    elif self.objective_function.lower() == 'roi':
                        ind_metrics['fitness_roi'] = ind.fitness.values[0]
                    else:
                        ind_metrics['fitness'] = ind.fitness.values[0]
                        
                # Add to metrics list
                metrics.append(ind_metrics)
                
            # Convert to DataFrame
            metrics_df = pd.DataFrame(metrics)
            
            # Save to CSV
            metrics_file = output_dir / f"all_individuals_{self.network_type}.csv"
            metrics_df.to_csv(metrics_file, index=False)
            
            log().info(f"Saved metrics for {len(all_individuals)} individuals to {metrics_file}")
            
            return metrics_file
            
        except Exception as e:
            log().error(f"Error saving all evaluated individuals: {e}")
            return None
            
            
    def save_results(self, solution):
        """
        Save the optimization results to files.
        
        Parameters:
        -----------
        solution : dict
            Dictionary with the optimization solution
            
        Returns:
        --------
        dict
            Dictionary with paths to the saved files
        """
        try:
            # Use the dedicated method to get the results folder
            output_dir = Path(self.locator.get_dynamic_dtn_optimization_results_folder())
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save detailed results
            detailed_results_file = self._save_detailed_results(solution, output_dir)
            
            # Save summary results
            summary_results_file = self._save_summary_results(solution, output_dir)
            
            # Save optimization settings
            settings_file = self._save_optimization_settings(output_dir)
            
            # Save solution
            solution_file = output_dir / f"dtn_expansion_opt_solution_{self.network_type}.json"
            with open(solution_file, 'w') as f:
                # Convert solution to a JSON-serializable format
                json_solution = {k: v for k, v in solution.items() if k != 'phases'}
                json_solution['phases'] = {str(k): v for k, v in solution['phases'].items()}
                
                # Convert numpy values to Python types
                for k, v in json_solution.items():
                    if isinstance(v, np.ndarray):
                        json_solution[k] = v.tolist()
                    elif isinstance(v, np.integer):
                        json_solution[k] = int(v)
                    elif isinstance(v, np.floating):
                        json_solution[k] = float(v)
                        
                # Save as JSON
                json.dump(json_solution, f, indent=4)
                
            log().info(f"Saved solution to {solution_file}")
            
            return {
                'detailed_results': detailed_results_file,
                'summary_results': summary_results_file,
                'settings': settings_file,
                'solution': solution_file,
                'output_dir': output_dir
            }
            
        except Exception as e:
            log().error(f"Error saving results: {e}")
            return {}
            
    def _save_detailed_results(self, solution, output_dir):
        """
        Save detailed results to a CSV file.
        
        Parameters:
        -----------
        solution : dict
            Dictionary with the optimization solution
        output_dir : Path
            Output directory
            
        Returns:
        --------
        Path
            Path to the saved CSV file
        """
        try:
            # Create a list to store detailed results
            detailed_results = []
            
            # Determine the demand type based on network type
            demand_type = 'Qh' if self.network_type == 'DH' else 'Qc'
            
            # Process each phase
            for phase in sorted(solution['phases'].keys()):
                # Get clusters in this phase
                phase_clusters = solution['phases'][phase]
                
                # Create a row for each cluster in this phase
                for cluster in phase_clusters:
                    try:
                        # Get buildings in this cluster
                        buildings = self._get_buildings_in_specific_cluster(cluster)
                        
                        # Calculate demand for this cluster
                        if self.network_type == 'DH':
                            cluster_demand = self.total_demand[
                                self.total_demand['name'].isin(buildings)
                            ]['Qhs_sys_MWhyr'].sum() + self.total_demand[
                                self.total_demand['name'].isin(buildings)
                            ]['Qww_sys_MWhyr'].sum()
                        else:
                            cluster_demand = self.total_demand[
                                self.total_demand['name'].isin(buildings)
                            ]['Qcs_sys_MWhyr'].sum() + self.total_demand[
                                self.total_demand['name'].isin(buildings)
                            ]['Qcre_sys_MWhyr'].sum() + self.total_demand[
                                self.total_demand['name'].isin(buildings)
                            ]['Qcdata_sys_MWhyr'].sum()
                        
                        # Get required pipes for this cluster
                        required_pipes = self._get_required_pipes({cluster}, self.cluster_edges)
                        cluster_pipe_length = required_pipes['length_m'].sum()
                        
                        # Calculate linear heat density
                        if cluster_pipe_length > 0:
                            cluster_linear_heat_density = cluster_demand / cluster_pipe_length * 1000  # MWh/km
                        else:
                            cluster_linear_heat_density = 0
                        
                        # Create detailed result dictionary
                        detailed_result = {
                            'phase': phase,
                            'cluster': cluster,
                            'number_of_buildings': len(buildings),
                            'buildings': ','.join(buildings),
                            'pipe_length [m]': cluster_pipe_length,
                            f'annual_{demand_type} [MWh/yr]': cluster_demand,
                            f'linear_{demand_type}_density [MWh/km/yr]': cluster_linear_heat_density
                        }
                        
                        # Add to detailed results list
                        detailed_results.append(detailed_result)
                    except Exception as e:
                        log().warning(f"Error processing detailed results for cluster {cluster} in phase {phase}: {e}")
                        # Add a minimal result with error indication
                        detailed_results.append({
                            'phase': phase,
                            'cluster': cluster,
                            'number_of_buildings': 0,
                            'buildings': f"Error: {str(e)[:50]}...",
                            'pipe_length [m]': 0,
                            f'annual_{demand_type} [MWh/yr]': 0,
                            f'linear_{demand_type}_density [MWh/km/yr]': 0
                        })
            
            # Convert to DataFrame
            detailed_results_df = pd.DataFrame(detailed_results)
            
            # Format numeric columns to avoid #NAME? errors
            numeric_columns = [col for col in detailed_results_df.columns if any(x in col for x in 
                              ['pipe_length', 'annual', 'linear', 'number_of_buildings'])]
            
            for col in numeric_columns:
                # Convert to numeric, coerce errors to NaN, then fill NaN with 0
                detailed_results_df[col] = pd.to_numeric(detailed_results_df[col], errors='coerce').fillna(0)
                
                # Format with appropriate precision
                detailed_results_df[col] = detailed_results_df[col].round(2)
            
            # Save to CSV
            detailed_results_file = output_dir / f"dtn_expansion_opt_results_{self.network_type}_detailed.csv"
            detailed_results_df.to_csv(detailed_results_file, index=False)
            
            log().info(f"Detailed results saved to {detailed_results_file}")
            
            return detailed_results_file
            
        except Exception as e:
            log().error(f"Error saving detailed results: {e}")
            return None
            
    def _save_summary_results(self, solution, output_dir):
        """
        Save summary results to a CSV file.
        
        Parameters:
        -----------
        solution : dict
            Dictionary with the optimization solution
        output_dir : Path
            Output directory
            
        Returns:
        --------
        Path
            Path to the saved CSV file
        """
        try:
            # Create a list to store summary results
            results = []
            
            # Determine the demand type based on network type
            demand_type = 'Qh' if self.network_type == 'DH' else 'Qc'
            
            # Add a row for phase 0 (existing DTN)
            results.append({
                'phase': 0,
                'year': 'Year 0',
                'newly_connected_cluster(s)': 0,
                'cumulative_cluster(s)': 0,
                'number_of_newly_connected_buildings': len(self._get_buildings_in_specific_cluster(0)),
                'cumulative_number_of_buildings_connected': len(self._get_buildings_in_specific_cluster(0)),
                'capex_budget_per_phase [USD]': '-',
                'new_cluster(s)_capex [USD]': 0,
                'cumulative_capex_budget [USD]': '-',
                'cumulative_capex [USD]': 0,
                'total_expenditure_budget_per_phase [USD]': '-',
                'new_cluster(s)_total_expenditure [USD]': 0,
                'cumulative_total_expenditure [USD]': 0,
                'new_cluster(s)_revenue [USD]': 0,
                'cumulative_revenue [USD]': 0,
                'new_cluster(s)_om_cost [USD]': 0,
                'cumulative_om_cost [USD]': 0,
                'ghg_cap [t CO2eq/yr]': '-',
                'district_operation_emission [t CO2eq/yr]': 0,
                'district_operation_emission_per_gfa [kg CO2eq/yr/m2]': 0,
                'new_cluster(s)_discounted_roi [-]': 0,
                'overall_discounted_roi [-]': 0,
                'new_cluster(s)_npv [USD]': 0,
                'overall_npv [USD]': 0,
                'new_cluster(s)_pipe_length [m]': 0,
                'cumulative_pipe_length [m]': 0,
                'new_cluster(s)_annual_Qh [MWh/yr]': 0,
                'cumulative_annual_Qh [MWh/yr]': 0,
                'new_cluster(s)_linear_Qh_density [MWh/km/yr]': 0,
                'overall_linear_Qh_density [MWh/km/yr]': 0
            })
            
            # Initialize cumulative values
            total_capex = 0
            total_expenditure = 0
            total_revenue = 0
            total_om_cost = 0
            total_npv = 0
            total_pipe_length = 0
            total_demand = 0
            cumulative_buildings = len(self._get_buildings_in_specific_cluster(0))
            cumulative_clusters = [0]
            
            # Process each phase
            for phase in sorted(solution['phases'].keys()):
                # Get clusters in this phase
                newly_connected_clusters = solution['phases'][phase]
                
                # Get all clusters connected up to this phase
                cumulative_clusters.extend(newly_connected_clusters)
                
                # Get buildings in newly connected clusters
                newly_connected_buildings = []
                for cluster in newly_connected_clusters:
                    newly_connected_buildings.extend(self._get_buildings_in_specific_cluster(cluster))
                
                # Update cumulative buildings
                cumulative_buildings += len(newly_connected_buildings)
                
                # Get required pipes for newly connected clusters
                required_pipes = self._get_required_pipes(newly_connected_clusters, self.cluster_edges)
                new_pipe_length = required_pipes['length_m'].sum()
                total_pipe_length += new_pipe_length
                
                # Calculate demand for newly connected clusters
                if self.network_type == 'DH':
                    new_demand = self.total_demand[
                        self.total_demand['name'].isin(newly_connected_buildings)
                    ]['Qhs_sys_MWhyr'].sum() + self.total_demand[
                        self.total_demand['name'].isin(newly_connected_buildings)
                    ]['Qww_sys_MWhyr'].sum()
                else:
                    new_demand = self.total_demand[
                        self.total_demand['name'].isin(newly_connected_buildings)
                    ]['Qcs_sys_MWhyr'].sum() + self.total_demand[
                        self.total_demand['name'].isin(newly_connected_buildings)
                    ]['Qcre_sys_MWhyr'].sum() + self.total_demand[
                        self.total_demand['name'].isin(newly_connected_buildings)
                    ]['Qcdata_sys_MWhyr'].sum()
                
                # Update total demand
                total_demand += new_demand
                
                # Calculate linear heat density
                if new_pipe_length > 0:
                    new_linear_heat_density = new_demand / new_pipe_length * 1000  # MWh/km
                else:
                    new_linear_heat_density = 0
                    
                if total_pipe_length > 0:
                    overall_linear_heat_density = total_demand / total_pipe_length * 1000  # MWh/km
                else:
                    overall_linear_heat_density = 0
                
                # Calculate financial metrics with error handling
                try:
                    new_capex, _ = self._calculate_phase_capex(newly_connected_clusters, phase)
                    total_capex += new_capex
                except Exception as e:
                    log().warning(f"Error calculating CAPEX for phase {phase}: {e}")
                    new_capex = 0
                
                # Calculate NPV with error handling
                try:
                    new_npv = self.calculate_npv(tuple(newly_connected_clusters), phase)
                    total_npv += new_npv
                except Exception as e:
                    log().warning(f"Error calculating NPV for phase {phase}: {e}")
                    new_npv = 0
                
                # Calculate ROI with error handling
                try:
                    new_roi = self.calculate_roi(tuple(newly_connected_clusters), phase)
                    if new_roi == float('inf') or new_roi == -float('inf'):  # Handle special case
                        new_roi = 0
                except Exception as e:
                    log().warning(f"Error calculating ROI for phase {phase}: {e}")
                    new_roi = 0
                
                # Calculate overall ROI with error handling
                try:
                    if total_capex > 0:
                        overall_roi = total_npv / total_capex
                    else:
                        overall_roi = 0
                except Exception as e:
                    log().warning(f"Error calculating overall ROI for phase {phase}: {e}")
                    overall_roi = 0
                
                # Calculate annual revenue and O&M cost
                try:
                    new_revenue = self._calculate_annual_revenue(newly_connected_clusters)
                    total_revenue += new_revenue
                except Exception as e:
                    log().warning(f"Error calculating revenue for phase {phase}: {e}")
                    new_revenue = 0
                    
                try:
                    new_om_cost = self._calculate_annual_om_cost(newly_connected_clusters)
                    total_om_cost += new_om_cost
                except Exception as e:
                    log().warning(f"Error calculating O&M cost for phase {phase}: {e}")
                    new_om_cost = 0
                
                # Calculate total expenditure
                new_expenditure = new_capex + new_om_cost
                total_expenditure += new_expenditure
                
                # Get emissions for this phase
                try:
                    # Calculate emissions for this phase
                    emissions_results = self.calculate_district_emissions_new()
                    
                    # Get emissions for this phase
                    if phase in emissions_results:
                        district_operation_emission = emissions_results[phase]['district_operation_emission [t CO2eq/yr]']
                        district_operation_emission_per_gfa = emissions_results[phase]['district_operation_emission_per_gfa [kg CO2eq/yr/m2]']
                    else:
                        district_operation_emission = 0
                        district_operation_emission_per_gfa = 0
                except Exception as e:
                    log().warning(f"Error calculating emissions for phase {phase}: {e}")
                    district_operation_emission = 0
                    district_operation_emission_per_gfa = 0
                
                # Create result dictionary for this phase
                result = {
                    'phase': phase,
                    'year': f'Year {(phase - 1) * self.phase_durations[0]}',
                    'newly_connected_cluster(s)': ','.join(map(str, newly_connected_clusters)),
                    'cumulative_cluster(s)': ','.join(map(str, cumulative_clusters)),
                    'number_of_newly_connected_buildings': len(newly_connected_buildings),
                    'cumulative_number_of_buildings_connected': cumulative_buildings,
                    'capex_budget_per_phase [USD]': self.capex_budget_per_phase[phase - 1] if self.capex_budget_per_phase and phase - 1 < len(self.capex_budget_per_phase) else '-',
                    'new_cluster(s)_capex [USD]': new_capex,
                    'cumulative_capex_budget [USD]': sum(self.capex_budget_per_phase[:phase]) if self.capex_budget_per_phase and phase <= len(self.capex_budget_per_phase) else '-',
                    'cumulative_capex [USD]': total_capex,
                    'total_expenditure_budget_per_phase [USD]': self.total_expenditure_budget_per_phase[phase - 1] if self.total_expenditure_budget_per_phase and phase - 1 < len(self.total_expenditure_budget_per_phase) else '-',
                    'new_cluster(s)_total_expenditure [USD]': new_expenditure,
                    'cumulative_total_expenditure [USD]': total_expenditure,
                    'new_cluster(s)_revenue [USD]': new_revenue,
                    'cumulative_revenue [USD]': total_revenue,
                    'new_cluster(s)_om_cost [USD]': new_om_cost,
                    'cumulative_om_cost [USD]': total_om_cost,
                    'ghg_cap [t CO2eq/yr]': self.ghg_budget_per_phase[phase - 1] if self.ghg_budget_per_phase and phase - 1 < len(self.ghg_budget_per_phase) else '-',
                    'district_operation_emission [t CO2eq/yr]': district_operation_emission,
                    'district_operation_emission_per_gfa [kg CO2eq/yr/m2]': district_operation_emission_per_gfa,
                    'new_cluster(s)_discounted_roi [-]': new_roi,
                    'overall_discounted_roi [-]': overall_roi,
                    'new_cluster(s)_npv [USD]': new_npv,
                    'overall_npv [USD]': total_npv,
                    'new_cluster(s)_pipe_length [m]': new_pipe_length,
                    'cumulative_pipe_length [m]': total_pipe_length,
                    'new_cluster(s)_annual_Qh [MWh/yr]': new_demand,
                    'cumulative_annual_Qh [MWh/yr]': total_demand,
                    'new_cluster(s)_linear_Qh_density [MWh/km/yr]': new_linear_heat_density,
                    'overall_linear_Qh_density [MWh/km/yr]': overall_linear_heat_density
                }
                
                # Add to results list
                results.append(result)
            
            # Convert to DataFrame
            results_df = pd.DataFrame(results)
            
            # Format numeric columns to avoid #NAME? errors
            numeric_columns = [col for col in results_df.columns if any(x in col for x in 
                              ['capex', 'expenditure', 'revenue', 'cost', 'roi', 'npv', 'emission', 'pipe_length', 'annual', 'linear'])]
            
            for col in numeric_columns:
                # Convert to numeric, coerce errors to NaN, then fill NaN with 0
                results_df[col] = pd.to_numeric(results_df[col], errors='coerce').fillna(0)
                
                # Format with appropriate precision based on column type
                if 'roi' in col.lower():
                    # ROI values typically small decimals
                    results_df[col] = results_df[col].round(4)
                elif any(x in col.lower() for x in ['capex', 'expenditure', 'revenue', 'cost', 'npv']):
                    # Financial values typically large integers
                    results_df[col] = results_df[col].round(2)
                elif 'emission' in col.lower():
                    # Emissions typically need 2 decimal places
                    results_df[col] = results_df[col].round(2)
                else:
                    # Other numeric values with 2 decimal places
                    results_df[col] = results_df[col].round(2)
            
            # Save to CSV
            results_file = output_dir / f"dtn_expansion_opt_results_{self.network_type}.csv"
            results_df.to_csv(results_file, index=False)
            
            log().info(f"Summary results saved to {results_file}")
            
            return results_file
            
        except Exception as e:
            log().error(f"Error saving summary results: {e}")
            return None
            
    def _save_optimization_settings(self, output_dir):
        """
        Save optimization settings to a CSV or JSON file depending on optimization mode.
        
        Parameters:
        -----------
        output_dir : Path
            Output directory
            
        Returns:
        --------
        Path
            Path to the saved file
        """
        try:
            # Create a dictionary with optimization settings
            settings = {
                'network_type': self.network_type,
                'num_phases': self.num_phases,
                'phase_durations': self.phase_durations,
                'interest_rate': self.interest_rate,
                'cost_model': self.cost_model,
                'objective_function': self.objective_function,
                'diversity_factor': self.diversity_factor,
                'temperature_difference_dh': self.temperature_difference_dh,
                'temperature_difference_dc': self.temperature_difference_dc,
                'pressure_loss_pa_per_m': self.pressure_loss_pa_per_m,
                'pump_operation_hours': self.pump_operation_hours,
                'pump_efficiency': self.pump_efficiency,
                'pump_load_factor': self.pump_load_factor,
                'pump_capex_a': self.pump_capex_a,
                'pump_capex_b': self.pump_capex_b,
                'cooling_cop': self.cooling_cop,
                'multi_objective_mode': self.multi_objective_mode,
                'multi_objective_functions': self.multi_objective_functions,
                'testing_clusters': self.testing_clusters,
                'capex_budget_per_phase': self.capex_budget_per_phase,
                'total_expenditure_budget_per_phase': self.total_expenditure_budget_per_phase,
                'ghg_budget_per_phase': self.ghg_budget_per_phase
            }
            
            # Convert numpy values to Python types
            for k, v in settings.items():
                if isinstance(v, np.ndarray):
                    settings[k] = v.tolist()
                elif isinstance(v, np.integer):
                    settings[k] = int(v)
                elif isinstance(v, np.floating):
                    settings[k] = float(v)
            
            # For single-objective optimization, save as CSV
            if not self.multi_objective_mode:
                settings_file = output_dir / f"dtn_expansion_opt_settings_{self.network_type}.csv"
                pd.DataFrame([settings]).to_csv(settings_file, index=False)
                log().info(f"Optimization settings saved to {settings_file}")
            else:
                # For multi-objective optimization, save as JSON
                settings_file = output_dir / f"dtn_expansion_opt_settings_{self.network_type}.json"
                with open(settings_file, 'w') as f:
                    json.dump(settings, f, indent=4)
                log().info(f"Optimization settings saved to {settings_file}")
            
            return settings_file
            
        except Exception as e:
            log().error(f"Error saving optimization settings: {e}")
            return None


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
    
    # Get the network type
    network_type = config.dynamic_dtn_optimization.network_type
    
    # Check if the thermal network prerequisites are met
    if not check_thermal_network_prerequisites(locator, network_type, bypass_check=True):
        log().error("Thermal network prerequisites not met. Please run the thermal network script first.")
        return
    
    # Create a DynamicDTNOptimizer instance
    log().info("Creating DynamicDTNOptimizer instance")
    dynamic_optimizer = DynamicDTNOptimizer(locator, config)
    
    # Get a locator for the temporary scenario
    log().info("Getting locator for temporary scenario using DynamicDTNOptimizer")
    try:
        temp_locator = dynamic_optimizer.get_temp_locator()
    except Exception as e:
        log().error(f"Error getting temporary scenario locator: {e}")
        log().error("Please run dynamic_dtn_optimization.py first to create the temporary scenario.")
        return
    
    log().info(f"Successfully obtained temporary scenario locator")
    
    # Load the metrics DataFrame from the dynamic DTN optimization updated metrics file
    log().info("Loading metrics from dynamic DTN optimization updated metrics file")
    metrics_file = Path(locator.get_dynamic_dtn_optimization_updated_metrics_file())
    if not metrics_file.exists():
        log().error(f"Updated metrics file not found: {metrics_file}")
        log().error("Please run dynamic_dtn_optimization.py first to create the updated metrics file.")
        return
    
    log().info(f"Loading metrics from: {metrics_file}")
    metrics_df = pd.read_csv(metrics_file)
    
    # Get optimization parameters from config
    num_phases = config.dtn_expansion_optimization.num_phases
    
    # Parse phase durations from config
    phase_durations_str = config.dtn_expansion_optimization.phase_durations
    if phase_durations_str:
        phase_durations = [int(d.strip()) for d in phase_durations_str.split(',') if d.strip()]
    else:
        phase_durations = [10] * num_phases
        
    # Parse CAPEX budget per phase from config
    capex_budget_str = config.dtn_expansion_optimization.capex_budget_per_phase
    if capex_budget_str:
        capex_budget_per_phase = [float(b.strip()) for b in capex_budget_str.split(',') if b.strip()]
    else:
        capex_budget_per_phase = None
        
    # Parse total expenditure budget per phase from config
    total_expenditure_budget_str = config.dtn_expansion_optimization.total_expenditure_budget_per_phase
    if total_expenditure_budget_str:
        total_expenditure_budget_per_phase = [float(b.strip()) for b in total_expenditure_budget_str.split(',') if b.strip()]
    else:
        total_expenditure_budget_per_phase = None
        
    # Parse GHG budget per phase from config
    ghg_budget_str = config.dtn_expansion_optimization.ghg_budget_per_phase
    if ghg_budget_str:
        ghg_budget_per_phase = [float(b.strip()) for b in ghg_budget_str.split(',') if b.strip()]
    else:
        ghg_budget_per_phase = None
        
    # Parse testing clusters from config
    testing_clusters_str = config.dtn_expansion_optimization.testing_clusters
    if testing_clusters_str:
        testing_clusters = [int(c.strip()) for c in testing_clusters_str.split(',') if c.strip()]
    else:
        testing_clusters = None
    
    log().info("Creating optimizer with temporary scenario locator")
    
    # Create the optimizer with the temp locator
    optimizer = DTNExpansionOptimizer(
        locator=temp_locator,  # Use the temp locator instead of the original locator
        network_type=network_type,
        metrics_df=metrics_df,
        num_phases=num_phases,
        phase_durations=phase_durations,
        capex_budget_per_phase=capex_budget_per_phase,
        total_expenditure_budget_per_phase=total_expenditure_budget_per_phase,
        interest_rate=config.dtn_expansion_optimization.interest_rate,
        cost_model=config.dtn_expansion_optimization.cost_model,
        objective_function=config.dtn_expansion_optimization.objective_function,
        diversity_factor=config.dtn_expansion_optimization.diversity_factor,
        temperature_difference_dh=config.dtn_expansion_optimization.temperature_difference_dh,
        temperature_difference_dc=config.dtn_expansion_optimization.temperature_difference_dc,
        pressure_loss_pa_per_m=config.dtn_expansion_optimization.pressure_loss_pa_per_m,
        pump_operation_hours=config.dtn_expansion_optimization.pump_operation_hours,
        pump_efficiency=config.dtn_expansion_optimization.pump_efficiency,
        pump_load_factor=config.dtn_expansion_optimization.pump_load_factor,
        pump_capex_a=config.dtn_expansion_optimization.pump_capex_a,
        pump_capex_b=config.dtn_expansion_optimization.pump_capex_b,
        cooling_cop=config.dtn_expansion_optimization.cooling_cop,
        ghg_budget_per_phase=ghg_budget_per_phase,
        multi_objective_mode=config.dtn_expansion_optimization.multi_objective_mode,
        multi_objective_functions=config.dtn_expansion_optimization.multi_objective_functions if config.dtn_expansion_optimization.multi_objective_functions else None,
        testing_clusters=testing_clusters
    )
    
    # Run the optimization
    try:
        population_size = config.dtn_expansion_optimization.population_size
    except AttributeError:
        population_size = 50  # Default value
        log().info("Using default population size: 50")
        
    try:
        num_generations = config.dtn_expansion_optimization.num_generations
    except AttributeError:
        num_generations = 30  # Default value
        log().info("Using default number of generations: 30")
        
    solution = optimizer.optimize(population_size=population_size, num_generations=num_generations)
    
    # Save the results
    if isinstance(solution, list):
        # Multi-objective optimization - this branch can be removed if MOO is not supported
        log().warning("Multi-objective optimization is not supported for dynamic DTN optimization. Using first solution only.")
        if solution:
            log().info("Saving first solution from multi-objective results")
            result_files = optimizer.save_results(solution[0])
            log().info(f"Results saved to: {result_files['output_dir']}")
        else:
            log().error("No solutions found in multi-objective optimization")
    else:
        # Single-objective optimization
        log().info("Saving optimization results")
        result_files = optimizer.save_results(solution)
        log().info(f"Results saved to: {result_files['output_dir']}")
        
    # Log execution time
    time_elapsed = time.time() - start
    log().info(f"Execution time: {time_elapsed:.2f} seconds")


if __name__ == '__main__':
    args = parse_args()
    
    config = cea.config.Configuration()
    
    if args.scenario:
        config.scenario = args.scenario
    if args.config:
        config.load(args.config)
        
    main(config)