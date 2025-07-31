# Dynamic DTN Optimization Pump Cost Calculation Fix

## Issue Description

The pump cost calculation in the dynamic DTN optimization part 2 module was using a different approach than the DTN expansion optimization module. This could lead to inconsistent results between the two modules. The issue was identified in the following methods:

1. `_calculate_pump_capex`: This method was calculating pump CAPEX using a different approach for converting annual demand to peak demand and using different units for pump power.

2. `_calculate_pump_om_cost`: This method was calculating pump O&M cost using a fixed electricity price of 0.2 USD/kWh instead of getting the energy price from the feedstock data.

## Changes Made

### 1. Updated `_calculate_pump_capex` Method

The `_calculate_pump_capex` method was modified to match the approach used in the DTN expansion optimization module:

```python
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
```

Key changes:
1. Changed the approach for calculating peak demand from annual demand, now using 2000 equivalent full load hours instead of 8760 hours.
2. Calculated mass flow rate in kg/s instead of volumetric flow rate in m³/s.
3. Ensured consistent units in the pump power calculation (W instead of kW).
4. Added calculation of annual pump electricity consumption and stored it in `self.annual_pump_electricity` for use in the O&M cost calculation.

### 2. Updated `_calculate_pump_om_cost` Method

The `_calculate_pump_om_cost` method was modified to use the calculated electricity consumption from `_calculate_pump_capex` and to get the energy price from the feedstock data:

```python
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
```

Key changes:
1. Called `self._calculate_pump_capex(pipes_df)` at the beginning to calculate the pump CAPEX and the annual pump electricity consumption.
2. Used `self._get_energy_price()` to get the energy price from the feedstock data, with a fallback to the default value of 0.2 USD/kWh if the energy price cannot be obtained.
3. Converted the annual pump electricity consumption from Wh to kWh and calculated the pump O&M cost by multiplying by the energy price.

## Explanation of the Approach

### Peak Demand Calculation

The DTN expansion optimization module calculates peak demand from annual demand using 2000 equivalent full load hours and the diversity factor:

```python
peak_demand_kw = annual_demand_mwh * 1000 / 2000 / self.diversity_factor
```

This approach is more realistic than the previous approach in the dynamic DTN optimization part 2 module, which used 8760 hours (the total number of hours in a year):

```python
total_heating_demand_w = total_heating_demand * 1e6 / 8760
peak_heating_demand_w = total_heating_demand_w / self.diversity_factor
```

Using 2000 equivalent full load hours accounts for the fact that the peak demand is not constant throughout the year, but rather occurs during a limited number of hours.

### Mass Flow Rate Calculation

The DTN expansion optimization module calculates mass flow rate in kg/s based on peak demand:

```python
mass_flow_rate = peak_demand_kw / (HEAT_CAPACITY_OF_WATER_JPERKGK * temp_diff / 1000)
```

This is equivalent to the volumetric flow rate calculation in the dynamic DTN optimization part 2 module, assuming a water density of 1000 kg/m³:

```python
flow_rate = peak_heating_demand_w / (HEAT_CAPACITY_OF_WATER_JPERKGK * self.temperature_difference_dh * 1000)  # m3/s
```

However, using mass flow rate is more consistent with the units used in the pump power calculation.

### Pump Power Calculation

The DTN expansion optimization module calculates pump power in W based on mass flow rate and pressure loss:

```python
pump_power = mass_flow_rate * pressure_loss / (self.pump_efficiency * 1000)  # W
```

The dynamic DTN optimization part 2 module was calculating pump power in kW:

```python
pump_power = total_flow_rate * pressure_loss / (self.pump_efficiency * 1000)  # kW
```

Using W is more consistent with the units used in the pump CAPEX calculation.

### Pump CAPEX Calculation

The DTN expansion optimization module calculates pump CAPEX based on pump power in kW:

```python
pump_capex = self.pump_capex_a * (pump_power / 1000) ** self.pump_capex_b  # USD
```

The dynamic DTN optimization part 2 module was calculating pump CAPEX based on pump power in kW:

```python
pump_capex = self.pump_capex_a * pump_power ** self.pump_capex_b
```

However, since the pump power was already in kW, this was inconsistent with the units used in the DTN expansion optimization module.

### Pump O&M Cost Calculation

The DTN expansion optimization module calculates annual pump electricity consumption in Wh:

```python
annual_pump_electricity = pump_power * self.pump_operation_hours * self.pump_load_factor  # Wh
```

The dynamic DTN optimization part 2 module was calculating pump electricity consumption in kWh:

```python
pump_electricity = pump_power * self.pump_operation_hours * self.pump_load_factor  # kWh
```

Using Wh is more consistent with the units used in the pump power calculation.

## Expected Impact

The changes to the pump cost calculation methods in the dynamic DTN optimization part 2 module will ensure that the results are consistent with the DTN expansion optimization module. This will lead to more accurate and reliable results for the dynamic DTN optimization process.

The main differences in the results will be:

1. **Peak Demand**: The peak demand will be higher when using 2000 equivalent full load hours instead of 8760 hours, which will lead to higher pump power and pump costs.

2. **Energy Price**: The energy price will be obtained from the feedstock data instead of using a fixed value of 0.2 USD/kWh, which will lead to more accurate pump O&M costs.

These changes will ensure that the dynamic DTN optimization part 2 module uses the same approach for pump cost calculation as the DTN expansion optimization module, leading to more consistent and reliable results.

## Testing

A test script `test_pump_cost_calculation.py` has been created to verify the pump cost calculation and compare the results with the original implementation. This script:

1. Defines functions that implement the original pump cost calculation methods.
2. Sets up the dynamic DTN optimization environment.
3. Tests both the original and modified pump cost calculation methods and compares the results.

To run the test script:

```
python -m test_pump_cost_calculation
```

The script will print the results of both the original and modified pump cost calculation methods, as well as the differences between them. This will help verify that the changes don't break existing functionality and that the results are reasonable.