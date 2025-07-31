"""
Test script to verify the pump cost calculation in dynamic DTN optimization part 2.

This script compares the results of the original and modified pump cost calculation methods.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

import cea.config
import cea.inputlocator
from cea.optimization_new.dynamic_dtn_optimization import DynamicDTNOptimizer
from cea.optimization_new.dynamic_dtn_optimization_part2 import DTNExpansionOptimizer, TempScenarioLocator
from cea.constants import HEAT_CAPACITY_OF_WATER_JPERKGK

def original_calculate_pump_capex(self, pipes_df):
    """
    Original implementation of _calculate_pump_capex.
    
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
        # Calculate pump CAPEX based on pipe flow rate and pressure loss
        pump_capex = 0
        
        # Calculate total pipe length
        total_pipe_length = pipes_df['length_m'].sum()
        
        # Skip if no pipes
        if total_pipe_length == 0:
            return 0
            
        # Calculate total flow rate
        if self.network_type == 'DH':
            # For district heating, use heating demand
            total_flow_rate = original_calculate_flow_rate_heating(self, pipes_df)
        else:
            # For district cooling, use cooling demand
            total_flow_rate = original_calculate_flow_rate_cooling(self, pipes_df)
            
        # Calculate pressure loss
        pressure_loss = self.pressure_loss_pa_per_m * total_pipe_length
        
        # Calculate pump power
        pump_power = total_flow_rate * pressure_loss / (self.pump_efficiency * 1000)  # kW
        
        # Calculate pump CAPEX
        pump_capex = self.pump_capex_a * pump_power ** self.pump_capex_b
        
        return pump_capex
        
    except Exception as e:
        print(f"Error calculating pump CAPEX: {e}")
        return 0

def original_calculate_flow_rate_heating(self, pipes_df):
    """
    Original implementation of _calculate_flow_rate_heating.
    
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
        print(f"Error calculating flow rate for heating: {e}")
        return 0

def original_calculate_flow_rate_cooling(self, pipes_df):
    """
    Original implementation of _calculate_flow_rate_cooling.
    
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
        print(f"Error calculating flow rate for cooling: {e}")
        return 0

def original_calculate_pump_om_cost(self, pipes_df):
    """
    Original implementation of _calculate_pump_om_cost.
    
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
        # Calculate pump O&M cost based on pump power and operation hours
        # Calculate total pipe length
        total_pipe_length = pipes_df['length_m'].sum()
        
        # Skip if no pipes
        if total_pipe_length == 0:
            return 0
            
        # Calculate total flow rate
        if self.network_type == 'DH':
            # For district heating, use heating demand
            total_flow_rate = original_calculate_flow_rate_heating(self, pipes_df)
        else:
            # For district cooling, use cooling demand
            total_flow_rate = original_calculate_flow_rate_cooling(self, pipes_df)
            
        # Calculate pressure loss
        pressure_loss = self.pressure_loss_pa_per_m * total_pipe_length
        
        # Calculate pump power
        pump_power = total_flow_rate * pressure_loss / (self.pump_efficiency * 1000)  # kW
        
        # Calculate pump electricity consumption
        pump_electricity = pump_power * self.pump_operation_hours * self.pump_load_factor  # kWh
        
        # Calculate pump O&M cost
        # Assume electricity price is 0.2 USD/kWh
        pump_om_cost = pump_electricity * 0.2
        
        return pump_om_cost
        
    except Exception as e:
        print(f"Error calculating pump O&M cost: {e}")
        return 0

def main():
    """
    Test the pump cost calculation in dynamic DTN optimization part 2.
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
    
    # Load the metrics DataFrame from the dynamic DTN optimization updated metrics file
    print("Loading metrics from dynamic DTN optimization updated metrics file")
    metrics_file = Path(locator.get_dynamic_dtn_optimization_updated_metrics_file())
    if not metrics_file.exists():
        print(f"Updated metrics file not found: {metrics_file}")
        print("Please run dynamic_dtn_optimization.py first to create the updated metrics file.")
        return
    
    print(f"Loading metrics from: {metrics_file}")
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
    
    # Load network data
    optimizer._load_network_data()
    
    # Create a sample pipes DataFrame for testing
    # Use the first cluster's edges
    if len(optimizer.all_clusters) > 0:
        cluster = optimizer.all_clusters[0]
        pipes_df = optimizer.edge_cluster_map[cluster]
        
        # Test the original pump cost calculation
        print("\nTesting original pump cost calculation:")
        original_pump_capex = original_calculate_pump_capex(optimizer, pipes_df)
        original_pump_om_cost = original_calculate_pump_om_cost(optimizer, pipes_df)
        print(f"Original pump CAPEX: ${original_pump_capex:.2f}")
        print(f"Original pump O&M cost: ${original_pump_om_cost:.2f}")
        
        # Test the modified pump cost calculation
        print("\nTesting modified pump cost calculation:")
        modified_pump_capex = optimizer._calculate_pump_capex(pipes_df)
        modified_pump_om_cost = optimizer._calculate_pump_om_cost(pipes_df)
        print(f"Modified pump CAPEX: ${modified_pump_capex:.2f}")
        print(f"Modified pump O&M cost: ${modified_pump_om_cost:.2f}")
        
        # Compare the results
        print("\nComparison:")
        capex_diff = modified_pump_capex - original_pump_capex
        capex_diff_pct = (capex_diff / original_pump_capex) * 100 if original_pump_capex != 0 else 0
        print(f"CAPEX difference: ${capex_diff:.2f} ({capex_diff_pct:.2f}%)")
        
        om_diff = modified_pump_om_cost - original_pump_om_cost
        om_diff_pct = (om_diff / original_pump_om_cost) * 100 if original_pump_om_cost != 0 else 0
        print(f"O&M cost difference: ${om_diff:.2f} ({om_diff_pct:.2f}%)")
    else:
        print("No clusters found. Please run dynamic_dtn_optimization.py first.")

if __name__ == "__main__":
    main()