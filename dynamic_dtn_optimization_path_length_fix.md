# Dynamic DTN Optimization Path Length Fix

## Issue Description

The dynamic DTN optimization process was encountering a Windows path length limitation error:

```
FileNotFoundError: [WinError 206] The filename or extension is too long: 'C:\\Users\\changf\\OneDrive - ETH Zurich\\CEA_projects\\base_design\\base\\outputs\\data\\optimization\\dynamic_dtn_optimization\\temp_scenario\\outputs\\data\\optimization\\dynamic_dtn_optimization\\temp_scenario\\outputs\\data\\optimization\\dtn_expansion\\phase_supply_files'
```

This error occurs because Windows has a 260-character path length limit, and the nested path structure created during the dynamic DTN optimization process exceeds this limit.

## Root Cause Analysis

The issue was occurring in the `calculate_district_emissions_new()` method in `dynamic_dtn_optimization_part2.py`. When using a `TempScenarioLocator` instance, calling methods like `get_dynamic_dtn_optimization_temp_scenario_phase_supply_files_folder()` created nested paths that included "temp_scenario" multiple times.

The problematic lines were:

1. Line 1387: `phase_files_dir = Path(self.locator.get_dynamic_dtn_optimization_temp_scenario_phase_supply_files_folder())`
2. Line 1401: `temp_demand_path = locator.get_dynamic_dtn_optimization_temp_scenario_total_demand()`
3. Line 1486: `temp_demand_path = self.locator.get_dynamic_dtn_optimization_temp_scenario_total_demand()`

When these methods were called on a `TempScenarioLocator` instance, they created paths that included the temp_scenario path multiple times, exceeding Windows' 260-character path limit.

## Solution

The solution was to check if we're using a `TempScenarioLocator` and use the original_locator when available. This prevents the path nesting issue by using the original locator's methods to get the correct paths.

### Changes Made

1. Modified line 1387 to use original_locator when available:
```python
# Check if we're using a TempScenarioLocator and use original_locator if available
if hasattr(self.locator, 'original_locator'):
    # Use the original locator's method to get the correct path
    phase_files_dir = Path(self.locator.original_locator.get_dynamic_dtn_optimization_temp_scenario_phase_supply_files_folder())
else:
    # Use the current locator if it's not a TempScenarioLocator
    phase_files_dir = Path(self.locator.get_dynamic_dtn_optimization_temp_scenario_phase_supply_files_folder())
```

2. Fixed the unresolved reference to 'locator' in lines 1407-1408 and implemented the fix for line 1401 to use original_locator when available:
```python
# Check if we're using a TempScenarioLocator and use original_locator if available
if hasattr(self.locator, 'original_locator'):
    temp_demand_path = self.locator.original_locator.get_dynamic_dtn_optimization_temp_scenario_total_demand()
else:
    temp_demand_path = self.locator.get_dynamic_dtn_optimization_temp_scenario_total_demand()
```

3. Modified line 1486 to use original_locator when available:
```python
# Check if we're using a TempScenarioLocator and use original_locator if available
if hasattr(self.locator, 'original_locator'):
    temp_demand_path = self.locator.original_locator.get_dynamic_dtn_optimization_temp_scenario_total_demand()
else:
    temp_demand_path = self.locator.get_dynamic_dtn_optimization_temp_scenario_total_demand()
```

## Expected Outcome

These changes should prevent the path nesting issue by using the original locator's methods to get the correct paths. This will keep the paths within Windows' 260-character limit and resolve the FileNotFoundError.

## Long-term Recommendation

For a more robust long-term solution, it would be beneficial to modify the `TempScenarioLocator` class to properly handle these specific methods. This would prevent similar issues from occurring in other parts of the code that use the same locator methods.

For example, adding these methods to the `TempScenarioLocator` class:

```python
def get_dynamic_dtn_optimization_temp_scenario_phase_supply_files_folder(self):
    """
    Returns the folder containing the phase supply files in the temporary scenario
    """
    return os.path.join(self.scenario, 'outputs', 'data', 'optimization', 'dtn_expansion', 'phase_supply_files')

def get_dynamic_dtn_optimization_temp_scenario_total_demand(self, format='csv'):
    """
    Returns the path to the total demand file in the temporary scenario
    """
    return os.path.join(self.scenario, 'outputs', 'data', 'demand', f'Total_demand.{format}')
```

This would ensure that these methods return the correct paths without nesting when called on a `TempScenarioLocator` instance.