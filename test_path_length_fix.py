"""
Test script to verify the path length fix in dynamic_dtn_optimization_part2.py
"""

import os
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cea.config
import cea.inputlocator
from cea.optimization_new.dynamic_dtn_optimization_part2 import TempScenarioLocator

def test_path_length_fix():
    """Test that the path length fix works correctly"""
    # Create a config object
    config = cea.config.Configuration()
    
    # Create an original locator
    original_locator = cea.inputlocator.InputLocator(config.scenario)
    
    # Create a temporary scenario path
    temp_scenario_path = os.path.join(
        original_locator.get_dynamic_dtn_optimization_folder(),
        'temp_scenario'
    )
    
    # Create a TempScenarioLocator
    temp_locator = TempScenarioLocator(original_locator, temp_scenario_path)
    
    # Test the fix for line 1387
    print("Testing fix for phase_files_dir path...")
    
    # Without the fix, this would create a nested path
    if hasattr(temp_locator, 'original_locator'):
        phase_files_dir = Path(temp_locator.original_locator.get_dynamic_dtn_optimization_temp_scenario_phase_supply_files_folder())
    else:
        phase_files_dir = Path(temp_locator.get_dynamic_dtn_optimization_temp_scenario_phase_supply_files_folder())
    
    print(f"Phase files directory path: {phase_files_dir}")
    print(f"Path length: {len(str(phase_files_dir))}")
    
    # Test the fix for line 1401/1486
    print("\nTesting fix for temp_demand_path...")
    
    # Without the fix, this would create a nested path
    if hasattr(temp_locator, 'original_locator'):
        temp_demand_path = temp_locator.original_locator.get_dynamic_dtn_optimization_temp_scenario_total_demand()
    else:
        temp_demand_path = temp_locator.get_dynamic_dtn_optimization_temp_scenario_total_demand()
    
    print(f"Temp demand path: {temp_demand_path}")
    print(f"Path length: {len(temp_demand_path)}")
    
    # Check if the paths are valid and not too long
    print("\nVerifying path lengths are within Windows limits...")
    if len(str(phase_files_dir)) > 260:
        print("ERROR: phase_files_dir path is still too long!")
    else:
        print("SUCCESS: phase_files_dir path is within Windows limits")
    
    if len(temp_demand_path) > 260:
        print("ERROR: temp_demand_path is still too long!")
    else:
        print("SUCCESS: temp_demand_path is within Windows limits")

if __name__ == "__main__":
    test_path_length_fix()