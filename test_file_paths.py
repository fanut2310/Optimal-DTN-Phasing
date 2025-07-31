"""
Test script to verify the file paths in dynamic DTN optimization part 2.
"""

import os
import sys
from pathlib import Path

import cea.config
import cea.inputlocator
from cea.optimization_new.dynamic_dtn_optimization import DynamicDTNOptimizer
from cea.optimization_new.dynamic_dtn_optimization_part2 import TempScenarioLocator

def main():
    """
    Test the file paths in dynamic DTN optimization part 2.
    """
    # Get the scenario and locator
    config = cea.config.Configuration()
    scenario = config.scenario
    locator = cea.inputlocator.InputLocator(scenario=scenario)
    
    # Get the network type
    network_type = config.dynamic_dtn_optimization.network_type
    
    # Create a DynamicDTNOptimizer instance
    print("Creating DynamicDTNOptimizer instance")
    dynamic_optimizer = DynamicDTNOptimizer(locator, config)
    
    # Get a locator for the temporary scenario
    print("Getting locator for temporary scenario using DynamicDTNOptimizer")
    try:
        temp_locator = dynamic_optimizer.get_temp_locator()
    except Exception as e:
        print(f"Error getting temporary scenario locator: {e}")
        print("Please run dynamic_dtn_optimization.py first to create the temporary scenario.")
        return
    
    print(f"Successfully obtained temporary scenario locator")
    
    # Test the file paths
    print("\nTesting file paths:")
    
    # 1. Test get_dynamic_dtn_network_layout_costs_file
    cost_file = temp_locator.get_dynamic_dtn_network_layout_costs_file(network_type)
    print(f"Cost file path: {cost_file}")
    print(f"Cost file exists: {os.path.exists(cost_file)}")
    
    # 2. Test get_dtn_cluster_nodes_file with original locator
    original_locator = cea.inputlocator.InputLocator(temp_locator.original_locator.scenario)
    cluster_nodes_file = original_locator.get_dtn_cluster_nodes_file()
    print(f"Cluster nodes file path: {cluster_nodes_file}")
    print(f"Cluster nodes file exists: {os.path.exists(cluster_nodes_file)}")
    
    # 3. Test get_thermal_network_folder
    thermal_network_folder = temp_locator.get_thermal_network_folder()
    print(f"Thermal network folder: {thermal_network_folder}")
    print(f"Thermal network folder exists: {os.path.exists(thermal_network_folder)}")
    
    # 4. Test get_network_layout_edges_shapefile
    network_edges_file = temp_locator.get_network_layout_edges_shapefile(network_type)
    print(f"Network edges file path: {network_edges_file}")
    print(f"Network edges file exists: {os.path.exists(network_edges_file)}")
    
    # 5. Test get_network_layout_nodes_shapefile
    network_nodes_file = temp_locator.get_network_layout_nodes_shapefile(network_type)
    print(f"Network nodes file path: {network_nodes_file}")
    print(f"Network nodes file exists: {os.path.exists(network_nodes_file)}")

if __name__ == "__main__":
    main()