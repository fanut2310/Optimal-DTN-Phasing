"""
Test script to verify the changes to cluster data handling in dynamic_dtn_optimization_part2.py.

This script tests the following changes:
1. Modified _load_cluster_data() method to directly use 'clusters' column
2. Modified _create_cluster_metrics_mapping() method to follow the approach in DTN_expansion_optimization.py
"""

import os
import sys
import logging
import pandas as pd
from pathlib import Path

import cea.config
import cea.inputlocator
from cea.optimization_new.dynamic_dtn_optimization import DynamicDTNOptimizer
from cea.optimization_new.dynamic_dtn_optimization_part2 import DTNExpansionOptimizer

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format="%(asctime)s | %(levelname)5s | %(message)s",
                   datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

def test_load_cluster_data():
    """Test the _load_cluster_data method."""
    # Get the scenario and locator
    config = cea.config.Configuration()
    scenario = config.scenario
    locator = cea.inputlocator.InputLocator(scenario=scenario)
    
    # Get the network type
    network_type = config.dynamic_dtn_optimization.network_type
    
    # Create a DynamicDTNOptimizer instance
    logger.info("Creating DynamicDTNOptimizer instance")
    dynamic_optimizer = DynamicDTNOptimizer(locator, config)
    
    # Get a locator for the temporary scenario
    logger.info("Getting locator for temporary scenario using DynamicDTNOptimizer")
    try:
        temp_locator = dynamic_optimizer.get_temp_locator()
    except Exception as e:
        logger.error(f"Error getting temporary scenario locator: {e}")
        logger.error("Please run dynamic_dtn_optimization.py first to create the temporary scenario.")
        return
    
    logger.info(f"Successfully obtained temporary scenario locator")
    
    # Load the metrics DataFrame from the dynamic DTN optimization updated metrics file
    logger.info("Loading metrics from dynamic DTN optimization updated metrics file")
    metrics_file = Path(locator.get_dynamic_dtn_optimization_updated_metrics_file())
    if not metrics_file.exists():
        logger.error(f"Updated metrics file not found: {metrics_file}")
        logger.error("Please run dynamic_dtn_optimization.py first to create the updated metrics file.")
        return
    
    logger.info(f"Loading metrics from: {metrics_file}")
    metrics_df = pd.read_csv(metrics_file)
    
    # Create the optimizer with the temp locator
    optimizer = DTNExpansionOptimizer(
        locator=temp_locator,
        network_type=network_type,
        metrics_df=metrics_df,
        num_phases=config.dtn_expansion_optimization.num_phases,
        phase_durations=[10] * config.dtn_expansion_optimization.num_phases,
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
    )
    
    # Test _load_cluster_data method
    logger.info("\nTesting _load_cluster_data method:")
    try:
        # Call _load_cluster_data method
        optimizer._load_cluster_data()
        
        # Check if all_clusters is populated
        logger.info(f"all_clusters: {optimizer.all_clusters}")
        
        # Check if buildings_by_cluster is populated
        logger.info(f"Number of clusters in buildings_by_cluster: {len(optimizer.buildings_by_cluster)}")
        for cluster, buildings in optimizer.buildings_by_cluster.items():
            logger.info(f"Cluster {cluster} has {len(buildings)} buildings")
        
        logger.info("_load_cluster_data method test passed")
    except Exception as e:
        logger.error(f"Error testing _load_cluster_data method: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")

def test_create_cluster_metrics_mapping():
    """Test the _create_cluster_metrics_mapping method."""
    # Get the scenario and locator
    config = cea.config.Configuration()
    scenario = config.scenario
    locator = cea.inputlocator.InputLocator(scenario=scenario)
    
    # Get the network type
    network_type = config.dynamic_dtn_optimization.network_type
    
    # Create a DynamicDTNOptimizer instance
    logger.info("Creating DynamicDTNOptimizer instance")
    dynamic_optimizer = DynamicDTNOptimizer(locator, config)
    
    # Get a locator for the temporary scenario
    logger.info("Getting locator for temporary scenario using DynamicDTNOptimizer")
    try:
        temp_locator = dynamic_optimizer.get_temp_locator()
    except Exception as e:
        logger.error(f"Error getting temporary scenario locator: {e}")
        logger.error("Please run dynamic_dtn_optimization.py first to create the temporary scenario.")
        return
    
    logger.info(f"Successfully obtained temporary scenario locator")
    
    # Load the metrics DataFrame from the dynamic DTN optimization updated metrics file
    logger.info("Loading metrics from dynamic DTN optimization updated metrics file")
    metrics_file = Path(locator.get_dynamic_dtn_optimization_updated_metrics_file())
    if not metrics_file.exists():
        logger.error(f"Updated metrics file not found: {metrics_file}")
        logger.error("Please run dynamic_dtn_optimization.py first to create the updated metrics file.")
        return
    
    logger.info(f"Loading metrics from: {metrics_file}")
    metrics_df = pd.read_csv(metrics_file)
    
    # Create the optimizer with the temp locator
    optimizer = DTNExpansionOptimizer(
        locator=temp_locator,
        network_type=network_type,
        metrics_df=metrics_df,
        num_phases=config.dtn_expansion_optimization.num_phases,
        phase_durations=[10] * config.dtn_expansion_optimization.num_phases,
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
    )
    
    # Test _create_cluster_metrics_mapping method
    logger.info("\nTesting _create_cluster_metrics_mapping method:")
    try:
        # Call _create_cluster_metrics_mapping method
        optimizer._create_cluster_metrics_mapping()
        
        # Check if cluster_metrics is populated
        logger.info(f"Number of entries in cluster_metrics: {len(optimizer.cluster_metrics)}")
        
        # Print a few sample entries
        for i, (key, metrics) in enumerate(optimizer.cluster_metrics.items()):
            if i < 3:  # Print only the first 3 entries
                logger.info(f"Cluster combination '{key}' has metrics: {list(metrics.keys())}")
        
        logger.info("_create_cluster_metrics_mapping method test passed")
    except Exception as e:
        logger.error(f"Error testing _create_cluster_metrics_mapping method: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")

def main():
    """Run the tests."""
    logger.info("Testing cluster data handling in dynamic_dtn_optimization_part2.py")
    
    # Test _load_cluster_data method
    test_load_cluster_data()
    
    # Test _create_cluster_metrics_mapping method
    test_create_cluster_metrics_mapping()
    
    logger.info("Tests completed")

if __name__ == "__main__":
    main()