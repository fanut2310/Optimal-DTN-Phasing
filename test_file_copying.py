"""
Test script to verify that the file copying code in dynamic_dtn_optimization_part2.py works correctly.
This script simulates the file copying process without running the full optimization.
"""

import os
import shutil
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger()

def log():
    return logger

def test_file_copying():
    """Test the file copying code from dynamic_dtn_optimization_part2.py"""
    
    # Define test paths
    scenario_path = r"C:\Users\changf\OneDrive - ETH Zurich\CEA_projects\base_design\01_base_design_2025"
    temp_scenario_path = os.path.join(scenario_path, "outputs", "data", "optimization", "dynamic_dtn_optimization", "temp_scenario")
    rerun_results_path = os.path.join(scenario_path, "outputs", "data", "optimization", "dynamic_dtn_optimization", "rerun_results")
    
    # Create a mock temp_locator and locator with the necessary methods
    class MockLocator:
        def __init__(self, scenario):
            self.scenario = scenario
            
        def get_dtn_expansion_optimization_results_folder(self):
            if self.scenario == temp_scenario_path:
                return os.path.join(self.scenario, "outputs", "data", "optimization", "dtn_expansion")
            else:
                return os.path.join(self.scenario, "outputs", "data", "optimization", "dtn_expansion")
                
        def get_dynamic_dtn_optimization_results_folder(self):
            return os.path.join(scenario_path, "outputs", "data", "optimization", "dynamic_dtn_optimization", "rerun_results")
            
        def get_dynamic_dtn_optimization_updated_metrics_file(self):
            return os.path.join(scenario_path, "outputs", "data", "optimization", "dynamic_dtn_optimization", "updated_metrics", "clusters_metrics_updated.csv")
    
    # Create mock locators
    temp_locator = MockLocator(temp_scenario_path)
    locator = MockLocator(scenario_path)
    
    # Get paths
    temp_results_dir = Path(temp_locator.get_dtn_expansion_optimization_results_folder())
    rerun_results_dir = Path(locator.get_dynamic_dtn_optimization_results_folder())
    
    # Check if the paths exist
    log().info(f"Checking if temp_results_dir exists: {temp_results_dir}")
    if not temp_results_dir.exists():
        log().error(f"Temp results directory does not exist: {temp_results_dir}")
        return
    
    # Create rerun_results_dir if it doesn't exist
    rerun_opt_results_dir = rerun_results_dir / "optimization_results"
    rerun_opt_results_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for CSV files in temp_results_dir
    csv_files = list(temp_results_dir.glob("*.csv"))
    log().info(f"Found {len(csv_files)} CSV files in {temp_results_dir}")
    
    # Copy optimization results
    log().info(f"Copying results from {temp_results_dir} to {rerun_opt_results_dir}")
    for file in csv_files:
        target_file = rerun_opt_results_dir / file.name
        log().info(f"Would copy {file} to {target_file}")
        # Uncomment to actually copy the files
        # shutil.copy2(file, target_file)
        # log().info(f"Copied {file.name} to {target_file}")
    
    # Check for phase supply files
    temp_phase_supply_dir = temp_results_dir / "phase_supply_files"
    log().info(f"Checking if temp_phase_supply_dir exists: {temp_phase_supply_dir}")
    if temp_phase_supply_dir.exists():
        rerun_phase_supply_dir = rerun_results_dir / "phase_supply_files"
        rerun_phase_supply_dir.mkdir(parents=True, exist_ok=True)
        
        phase_supply_files = list(temp_phase_supply_dir.glob("*.csv"))
        log().info(f"Found {len(phase_supply_files)} phase supply files in {temp_phase_supply_dir}")
        
        for file in phase_supply_files:
            target_file = rerun_phase_supply_dir / file.name
            log().info(f"Would copy {file} to {target_file}")
            # Uncomment to actually copy the files
            # shutil.copy2(file, target_file)
            # log().info(f"Copied {file.name} to {target_file}")
    else:
        log().warning(f"Phase supply directory does not exist: {temp_phase_supply_dir}")
    
    # Check for updated metrics file
    metrics_file = Path(locator.get_dynamic_dtn_optimization_updated_metrics_file())
    log().info(f"Checking if metrics_file exists: {metrics_file}")
    if metrics_file.exists():
        target_file = rerun_results_dir / metrics_file.name
        log().info(f"Would copy {metrics_file} to {target_file}")
        # Uncomment to actually copy the files
        # shutil.copy2(metrics_file, target_file)
        # log().info(f"Copied {metrics_file.name} to {target_file}")
    else:
        log().warning(f"Metrics file does not exist: {metrics_file}")
    
    log().info("File copying test completed successfully")

if __name__ == "__main__":
    test_file_copying()