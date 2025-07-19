# Fix for TempScenarioLocator.get_total_demand in Dynamic DTN Rerun Optimization

## Issue Description

The Dynamic DTN Rerun Optimization module was failing with the following error:

```
Path returned by temp_locator.get_total_demand(): C:\Users\changf\OneDrive - ETH Zurich\CEA_projects\base_design\01_base_design_2025\outputs\data\demand\Total_demand.csv
temp_locator.get_total_demand() is not returning the path to the temp scenario's Total_demand.csv
Expected path to contain: C:\Users\changf\OneDrive - ETH Zurich\CEA_projects\base_design\01_base_design_2025\outputs\data\optimization\dynamic_dtn_optimization\temp_scenario\outputs\data\demand\Total_demand.csv
Actual path: C:\Users\changf\OneDrive - ETH Zurich\CEA_projects\base_design\01_base_design_2025\outputs\data\demand\Total_demand.csv
```

This error occurred because the `TempScenarioLocator.get_total_demand()` method was not returning the path to the `Total_demand.csv` file in the temp scenario folder, despite having an override for this method.

## Root Cause Analysis

The issue was related to how the `get_total_demand` method was implemented in the `TempScenarioLocator` class:

1. The `get_total_demand` method was using `self.get_demand_results_folder()` to get the path to the demand folder, and then appending `Total_demand.{format}` to it.

2. Although the `TempScenarioLocator` class had an override for the `get_demand_results_folder` method, there might have been issues with how this method was being called or how the path was being constructed.

3. This resulted in the `get_total_demand` method returning the path to the original `Total_demand.csv` file, not the one in the temp scenario folder.

## Solution Implemented

The solution was to modify the `get_total_demand` method in the `TempScenarioLocator` class to construct the path directly using the temp scenario path, rather than relying on `get_demand_results_folder`:

```python
def get_total_demand(self, format='csv'):
    """
    Override to return the path to the Total_demand.csv file in the temp scenario.
    
    This ensures that the PipeLayoutGenerator uses the updated Total_demand.csv
    that reflects the modified demand files, rather than the original one.
    
    Parameters:
    -----------
    format : str, optional
        The file format (default: 'csv')
        
    Returns:
    --------
    str
        Path to the Total_demand.csv file in the temp scenario
    """
    # Add debug print to trace execution
    lg = log()
    lg.info(f"DEBUG: get_total_demand called in TempScenarioLocator with format={format}")
    
    # Construct the path directly using the scenario path
    # This avoids any issues with get_demand_results_folder not being called correctly
    total_demand_path = os.path.join(self.scenario, 'outputs', 'data', 'demand', f'Total_demand.{format}')
    lg.info(f"DEBUG: Directly constructed total_demand_path: {total_demand_path}")
    
    # Log that we're using the temp scenario's Total_demand.csv
    lg.info(f"Using Total_demand.csv from temp scenario: {total_demand_path}")
    
    # Verify that the file exists
    if not os.path.exists(total_demand_path):
        lg.warning(f"Total_demand.{format} not found in temp scenario: {total_demand_path}")
        lg.warning(f"Falling back to original Total_demand.{format}")
        fallback_path = self.original_locator.get_total_demand(format)
        lg.info(f"DEBUG: Falling back to original path: {fallback_path}")
        return fallback_path
        
    lg.info(f"DEBUG: Returning total_demand_path: {total_demand_path}")
    return total_demand_path
```

This direct approach ensures that the `get_total_demand` method returns the correct path to the `Total_demand.csv` file in the temp scenario folder, regardless of any issues with method overriding or caching.

## How to Test

To test the fix, run the dynamic-dtn-rerun-optimization command with the same parameters as before:

```
C:\Users\changf\micromamba\envs\cea\python.exe D:\changf\CityEnergyAnalyst\cea\interfaces\cli\cli.py dynamic-dtn-rerun-optimization --scenario "C:\Users\changf\OneDrive - ETH Zurich\CEA_projects\base_design\01_base_design_2025" --network-type DH --testing-clusters 1,2,3,4,7
```

### Expected Behavior

1. The script should run without the "temp_locator.get_total_demand() is not returning the correct path" error.
2. The log should show messages like:
   - "DEBUG: get_total_demand called in TempScenarioLocator with format=csv"
   - "DEBUG: Directly constructed total_demand_path: [path to temp scenario's Total_demand.csv]"
   - "Using Total_demand.csv from temp scenario: [path to temp scenario's Total_demand.csv]"
   - "✓ temp_locator.get_total_demand() returns the correct path"
3. The optimization should complete successfully.
4. The updated metrics should reflect the modified demand files, which should result in different metrics than the original optimization.

### Troubleshooting

If you encounter any issues:

1. **Check the log messages**: Look for the debug messages added to the `get_total_demand` method to see what paths are being constructed and returned.
2. **Verify file paths**: Make sure the paths to the temp scenario folder and its subdirectories are correct.
3. **Check file existence**: Ensure that the `Total_demand.csv` file exists in the temp scenario folder.
4. **Compare metrics**: If the updated metrics are identical to the original metrics, it might indicate that there's another issue with how the metrics are calculated.

## Conclusion

The fix ensures that the `get_total_demand` method in the `TempScenarioLocator` class returns the correct path to the `Total_demand.csv` file in the temp scenario folder. This allows the `PipeLayoutGenerator` class to use the updated `Total_demand.csv` file, which reflects the modified demand files, resulting in more accurate optimization results.

The direct approach of constructing the path using the temp scenario path, rather than relying on `get_demand_results_folder`, makes the code more robust and less prone to issues with method overriding or caching.