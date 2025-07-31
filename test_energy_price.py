"""
Test script to verify the _get_energy_price method in dynamic_dtn_optimization_part2.py
"""

import os
import sys
import pandas as pd
from pathlib import Path

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cea.config
import cea.inputlocator
from cea.optimization_new.dynamic_dtn_optimization_part2 import DTNExpansionOptimizer

def main():
    """Test the _get_energy_price method"""
    # Create a configuration object
    config = cea.config.Configuration()
    
    # Get the scenario path from the configuration
    scenario_path = config.scenario
    
    # Create an InputLocator
    locator = cea.inputlocator.InputLocator(scenario=scenario_path)
    
    # Create a dummy metrics DataFrame
    metrics_df = pd.DataFrame({
        'clusters': ['1', '2', '3'],
        'total_demand_MWh': [100, 200, 300],
        'peak_demand_MW': [10, 20, 30],
        'network_length_m': [1000, 2000, 3000]
    })
    
    # Test for DH network
    print("Testing _get_energy_price for DH network")
    try:
        optimizer_dh = DTNExpansionOptimizer(
            locator=locator,
            network_type='DH',
            metrics_df=metrics_df
        )
        energy_price_dh = optimizer_dh._get_energy_price()
        print(f"Energy price for DH: {energy_price_dh}")
    except Exception as e:
        print(f"Error getting energy price for DH: {e}")
    
    # Test for DC network
    print("\nTesting _get_energy_price for DC network")
    try:
        optimizer_dc = DTNExpansionOptimizer(
            locator=locator,
            network_type='DC',
            metrics_df=metrics_df
        )
        energy_price_dc = optimizer_dc._get_energy_price()
        print(f"Energy price for DC: {energy_price_dc}")
    except Exception as e:
        print(f"Error getting energy price for DC: {e}")

if __name__ == "__main__":
    main()