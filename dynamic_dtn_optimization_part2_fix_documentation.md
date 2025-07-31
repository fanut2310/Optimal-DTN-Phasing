# Dynamic DTN Optimization Part 2 File Path Fix

## Issue Description

When running the `dynamic_dtn_optimization_part2.py` script, the following error occurred:

```
16:22:29 | WARNING | Dynamic DTN cost file not found: C:\Users\changf\OneDrive - ETH Zurich\CEA_projects\base_design\01_base_design_2025\outputs\data\optimization\dynamic_dtn_optimization\temp_scenario\outputs\data\optimization\dynamic_dtn_optimization\thermal_network\DH_costs.csv. Falling back to original cost file.
```

The script was looking for thermal network files in the wrong location within the temporary scenario. Specifically:

1. The script was looking for files in:
   ```
   ..\temp_scenario\outputs\data\optimization\dynamic_dtn_optimization\thermal_network\DH_costs.csv
   ```

2. But the actual files were in:
   ```
   ..\temp_scenario\outputs\data\thermal-network\DH_costs.csv
   ```

## Root Cause

The issue was in how the `TempScenarioLocator` class handled paths to dynamic DTN optimization thermal network files. The class already had overrides for several methods to redirect file requests to the temporary scenario, but it was missing an override for `get_dynamic_dtn_optimization_thermal_network_folder()`.

When methods like `get_dynamic_dtn_network_layout_costs_file()` were called, they used `get_dynamic_dtn_optimization_thermal_network_folder()` to get the base directory, which returned a path to the original scenario's dynamic DTN optimization folder, not the temporary scenario's thermal network folder.

## Solution

The solution was to add an override for `get_dynamic_dtn_optimization_thermal_network_folder()` to the `TempScenarioLocator` class in `dynamic_dtn_optimization_part2.py`. The override returns the same path as `get_thermal_network_folder()`, which is the correct path to the thermal network folder in the temporary scenario.

```python
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
```

This ensures that when dynamic DTN-specific methods like `get_dynamic_dtn_network_layout_costs_file()` are called from a `TempScenarioLocator` instance, they'll look for files in the correct location.

## Testing

To test the changes:

1. Run `dynamic_dtn_optimization.py` first to create the temporary scenario:
   ```
   python -m cea.optimization_new.dynamic_dtn_optimization --scenario <path_to_scenario>
   ```

2. Then run `dynamic_dtn_optimization_part2.py`:
   ```
   python -m cea.optimization_new.dynamic_dtn_optimization_part2 --scenario <path_to_scenario>
   ```

The script should now correctly find the thermal network files in the temporary scenario and proceed with the optimization without the warning about missing files.

## Benefits of the Solution

1. **Minimal Changes**: The solution required adding just one method override to the `TempScenarioLocator` class, making it a minimal and focused fix.
2. **Maintainability**: By using the existing `get_thermal_network_folder()` method, the solution leverages code that's already been tested and works correctly.
3. **Robustness**: The solution ensures that all dynamic DTN-specific methods will look for files in the correct location, not just the specific method that was causing the error.
4. **Consistency**: The solution follows the same pattern as the other method overrides in the `TempScenarioLocator` class, maintaining consistency in the codebase.

## Alternative Approaches Considered

An alternative approach would have been to modify the dynamic DTN-specific methods in `inputlocator.py` to handle the case when they're called from a `TempScenarioLocator` instance. However, this would have required modifying multiple methods and would have been more invasive. The chosen solution is more focused and less likely to introduce new issues.