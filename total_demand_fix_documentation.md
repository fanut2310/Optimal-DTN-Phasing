# Total Demand Fix for Dynamic DTN Rerun Optimization

## Issue Description

The dynamic-dtn-rerun-optimization module was producing identical metrics to the original optimization, despite using modified demand files. This occurred because the `PipeLayoutGenerator` class was using the original `Total_demand.csv` file instead of the updated one in the temp scenario folder.

The error manifested as:
```
opt results obtained, but clusters_metrics_updated.csv still identical to that of the original clusters_metrics.csv, so in the rerun of DTN opt, the same genome value and same NPV are obtained.
```

## Root Cause Analysis

The issue was related to how the `PipeLayoutGenerator` class in `DTN_expansion_optimization.py` calculates metrics for cluster combinations:

1. In the `_load_inputs` method, it loads the `Total_demand.csv` file:
   ```python
   # Load total demand
   total_demand_path = Path(self.locator.get_total_demand())
   self.total_demand = pd.read_csv(total_demand_path)
   ```

2. In the `calculate_metrics` method, it uses this `total_demand` DataFrame to calculate the total annual demand:
   ```python
   # Calculate total annual demand
   if self.network_type == 'DH':
       # For district heating, use Qhs_sys_MWhyr + Qww_sys_MWhyr
       total_annual_demand = self.total_demand[
           self.total_demand['name'].isin(buildings_in_clusters)
       ]['Qhs_sys_MWhyr'].sum() + self.total_demand[
           self.total_demand['name'].isin(buildings_in_clusters)
       ]['Qww_sys_MWhyr'].sum()
       demand_type = 'Qh'
   ```

3. The `TempScenarioLocator` class was correctly pointing to the temp scenario folder, but it wasn't overriding the `get_total_demand` method. As a result, the `PipeLayoutGenerator` was still using the original `Total_demand.csv` file from the original scenario folder, not the updated one in the temp scenario folder.

## Solution Implemented

### 1. Override the `get_total_demand` Method in `TempScenarioLocator`

Added an override for the `get_total_demand` method in the `TempScenarioLocator` class to ensure it returns the path to the `Total_demand.csv` file in the temp scenario folder:

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
    # Get the path to the Total_demand.csv file in the temp scenario
    total_demand_path = os.path.join(self.get_demand_results_folder(), f'Total_demand.{format}')
    
    # Log that we're using the temp scenario's Total_demand.csv
    lg = log()
    lg.info(f"Using Total_demand.csv from temp scenario: {total_demand_path}")
    
    # Verify that the file exists
    if not os.path.exists(total_demand_path):
        lg.warning(f"Total_demand.{format} not found in temp scenario: {total_demand_path}")
        lg.warning(f"Falling back to original Total_demand.{format}")
        return self.original_locator.get_total_demand(format)
        
    return total_demand_path
```

### 2. Enhanced Verification in `_generate_updated_metrics` and `run` Methods

Added comprehensive verification to ensure the correct `Total_demand.csv` file is being used:

1. Verify that the file exists in the temp scenario folder
2. Load and verify that it has the required columns
3. Check that the `get_total_demand` method returns the correct path
4. Add detailed logging throughout the process

### 3. Added Detailed Metrics Comparison

Added code to compare the original and updated metrics in detail:

1. Check if the DataFrames are identical
2. Compare key columns based on the network type (DH or DC)
3. Calculate and log summary statistics about the differences
4. Find and log specific examples of the largest differences
5. Add fallback logic to compare other numeric columns if needed

## Benefits of the Changes

1. **Correct Metrics**: The `PipeLayoutGenerator` now uses the updated `Total_demand.csv` file from the temp scenario folder, which reflects the modified demand files.

2. **Comprehensive Verification**: The code now verifies that the correct files are being used at multiple points in the process.

3. **Detailed Logging**: The enhanced logging provides clear information about which files are being used and how the metrics have changed.

4. **Robust Error Handling**: The code includes fallback mechanisms and clear error messages if anything goes wrong.

## How to Test

Run the dynamic-dtn-rerun-optimization command with the same parameters:

```
C:\Users\changf\micromamba\envs\cea\python.exe D:\changf\CityEnergyAnalyst\cea\interfaces\cli\cli.py dynamic-dtn-rerun-optimization --scenario "C:\Users\changf\OneDrive - ETH Zurich\CEA_projects\base_design\01_base_design_2025" --network-type DH --testing-clusters 1,2,3,4,7
```

### Expected Behavior

1. The log should show messages confirming that the temp scenario's `Total_demand.csv` file is being used:
   - "Using Total_demand.csv from temp scenario: [path]"
   - "✓ Total_demand.csv exists in temp scenario"
   - "✓ Total_demand.csv has all required columns"
   - "✓ temp_locator.get_total_demand() returns the correct path"

2. The log should show that the metrics have changed:
   - "✓ Generated metrics are DIFFERENT from original metrics"
   - Details about the differences in key metrics

3. The optimization should complete with a potentially different optimal genome and NPV value.

## Troubleshooting

If issues persist, check the log for these specific messages:

1. **"⚠️ Generated metrics are IDENTICAL to original metrics!"**: This indicates that the temp scenario's `Total_demand.csv` file is not being used properly.

2. **"No differences found in the values despite DataFrames not being equal"**: This might indicate differences in other columns or metadata.

3. **"Could not compare [column] - not found in both DataFrames"**: This indicates that the expected column is missing from one of the DataFrames.

If you see any of these messages, check:

1. That the `Total_demand.csv` file in the temp scenario folder is actually different from the original one
2. That the `get_total_demand` method is being called correctly
3. That the `PipeLayoutGenerator` is using the `total_demand` DataFrame correctly

## Conclusion

The changes ensure that the dynamic-dtn-rerun-optimization module correctly uses the updated `Total_demand.csv` file from the temp scenario folder, which reflects the modified demand files. This fixes the issue where the metrics were identical to the original metrics, allowing the optimization to properly reflect the changes in demand profiles.