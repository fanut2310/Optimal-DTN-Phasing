# Fix for TempScenarioLocator in Dynamic DTN Rerun Optimization

## Issue Description

The Dynamic DTN Rerun Optimization module was failing with the following error:

```
Path returned by temp_locator.get_total_demand(): C:\Users\changf\OneDrive - ETH Zurich\CEA_projects\base_design\01_base_design_2025\outputs\data\demand\Total_demand.csv
temp_locator.get_total_demand() is not returning the path to the temp scenario's Total_demand.csv
Expected path to contain: C:\Users\changf\OneDrive - ETH Zurich\CEA_projects\base_design\01_base_design_2025\outputs\data\optimization\dynamic_dtn_optimization\temp_scenario\outputs\data\demand\Total_demand.csv
Actual path: C:\Users\changf\OneDrive - ETH Zurich\CEA_projects\base_design\01_base_design_2025\outputs\data\demand\Total_demand.csv
```

This error occurred because the `TempScenarioLocator` class was not correctly overriding the `get_demand_results_folder` method, which is used by the `get_total_demand` method to construct the path to the `Total_demand.csv` file.

## Root Cause Analysis

The issue was related to how the `TempScenarioLocator` class was implemented:

1. The `TempScenarioLocator` class correctly overrode the `get_total_demand` method, but it used `self.get_demand_results_folder()` to construct the path to the `Total_demand.csv` file.

2. Since the `get_demand_results_folder` method was not overridden, it was using the implementation from the parent class (`InputLocator`), which returns the path to the original demand folder, not the one in the temp scenario.

3. As a result, when `get_total_demand` was called, it was returning the path to the original `Total_demand.csv` file, not the one in the temp scenario folder.

## Solution Implemented

The solution was to add a `get_demand_results_folder` method to the `TempScenarioLocator` class that overrides the parent class's method to return the path to the demand folder in the temp scenario:

```python
def get_demand_results_folder(self):
    """
    Override to return the path to the demand folder in the temp scenario.
    
    This ensures that all methods that use get_demand_results_folder() will
    correctly point to the temp scenario's demand folder.
    
    Returns:
    --------
    str
        Path to the demand folder in the temp scenario
    """
    # Use the temp scenario path instead of the original scenario path
    return self._ensure_folder(self.scenario, 'outputs', 'data', 'demand')
```

This ensures that when `get_total_demand` calls `self.get_demand_results_folder()`, it gets the path to the demand folder in the temp scenario, not the original scenario.

## Verification

The script already includes comprehensive verification to ensure that the `get_total_demand` method returns the correct path:

1. In the `_generate_updated_metrics` method (lines 408-419):
   ```python
   # Verify that the temp_locator's get_total_demand method returns the correct path
   locator_total_demand_path = temp_locator.get_total_demand()
   self.lg.info(f"Path returned by temp_locator.get_total_demand(): {locator_total_demand_path}")
   
   # Check if the path is correct (should point to the temp scenario)
   if str(total_demand_path) not in str(locator_total_demand_path):
       self.lg.error(f"temp_locator.get_total_demand() is not returning the path to the temp scenario's Total_demand.csv")
       self.lg.error(f"Expected path to contain: {total_demand_path}")
       self.lg.error(f"Actual path: {locator_total_demand_path}")
       raise ValueError(f"temp_locator.get_total_demand() is not returning the correct path")
   
   self.lg.info(f"✓ temp_locator.get_total_demand() returns the correct path")
   ```

2. Similar verification is also performed in the `run` method (lines 625-636).

This verification code will confirm that our fix works correctly by checking if the path returned by `get_total_demand` contains the expected path to the temp scenario's `Total_demand.csv` file.

## How to Test

To test the fix, run the dynamic-dtn-rerun-optimization command with the same parameters as before:

```
C:\Users\changf\micromamba\envs\cea\python.exe D:\changf\CityEnergyAnalyst\cea\interfaces\cli\cli.py dynamic-dtn-rerun-optimization --scenario "C:\Users\changf\OneDrive - ETH Zurich\CEA_projects\base_design\01_base_design_2025" --network-type DH --testing-clusters 1,2,3,4,7
```

### Expected Behavior

1. The script should run without the "temp_locator.get_total_demand() is not returning the correct path" error.
2. The log should show messages like:
   - "Using Total_demand.csv from temp scenario: [path to temp scenario's Total_demand.csv]"
   - "✓ Total_demand.csv exists in temp scenario"
   - "✓ Total_demand.csv has all required columns"
   - "✓ temp_locator.get_total_demand() returns the correct path"
3. The optimization should complete successfully.
4. The updated metrics should reflect the modified demand files, which should result in different metrics than the original optimization.

### Troubleshooting

If you encounter any issues:

1. **Check the log messages**: Look for error messages related to the `get_total_demand` method or the `Total_demand.csv` file.
2. **Verify file paths**: Make sure the paths to the temp scenario folder and its subdirectories are correct.
3. **Check file existence**: Ensure that the `Total_demand.csv` file exists in the temp scenario folder.
4. **Compare metrics**: If the updated metrics are identical to the original metrics, it might indicate that the `Total_demand.csv` file in the temp scenario is identical to the original one, or that there's another issue with how the metrics are calculated.

## Conclusion

The fix ensures that the `TempScenarioLocator` class correctly overrides the `get_demand_results_folder` method, which is used by the `get_total_demand` method to construct the path to the `Total_demand.csv` file. This ensures that the `PipeLayoutGenerator` class uses the updated `Total_demand.csv` file from the temp scenario folder, which reflects the modified demand files.

This fix addresses the issue where the metrics were identical to the original metrics, allowing the optimization to properly reflect the changes in demand profiles.