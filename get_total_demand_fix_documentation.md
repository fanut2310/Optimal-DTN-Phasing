# TempScenarioLocator get_total_demand() Fix Documentation

## Issue Description

The dynamic DTN optimization part 2 was not correctly calculating emissions based on the modified demand data. While the modified demand files were correctly created and placed in the temporary scenario folder, the emissions calculations were still using the original demand data from the original scenario.

## Root Cause

The issue was in how the `lca_operation` function was called in the `calculate_district_emissions_new` method of the `DTNExpansionOptimizer` class. The problem had two key aspects:

1. **Locator Path Handling**: When `lca_operation` was called with the temporary scenario locator, it correctly used the phase-specific supply files, but it was still reading the demand data from the original path.

2. **Total Demand File Access**: The `lca_operation` function in `cea/analysis/lca/operation.py` reads the demand data using:
   ```python
   # From cea/analysis/lca/operation.py
   # Where pd is pandas and locator is an instance of InputLocator
   demand = pd.read_csv(locator.get_total_demand())
   ```
   
   While the `TempScenarioLocator` class correctly redirected most file requests to the temporary scenario, it didn't explicitly override the `get_total_demand()` method. This meant it was using the inherited implementation from `InputLocator`, which pointed to the original scenario's total demand file, not the modified one.

## Solution

The fix adds a `get_total_demand()` method to the `TempScenarioLocator` class in `dynamic_dtn_optimization.py` to ensure it points to the temporary scenario's total demand file:

```python
# Added to TempScenarioLocator class in dynamic_dtn_optimization.py
# Where os is the os module imported at the top of the file
def get_total_demand(self, format='csv'):
    """
    Override to return the path to the total demand file in the temp scenario.
    
    Parameters:
    -----------
    format : str, optional
        Format of the file (default: 'csv')
        
    Returns:
    --------
    str
        Path to the total demand file in the temp scenario
    """
    return os.path.join(self.get_demand_results_folder(), f'Total_demand.{format}')
```

This ensures that when `lca_operation` calls `locator.get_total_demand()`, it gets the path to the modified total demand file in the temporary scenario, which contains the updated demand values for the modified cluster.

## Benefits of the Fix

This fix ensures that:

1. The `lca_operation` function reads the modified total demand data from the temporary scenario.
2. The emissions calculations reflect the changes in demand for the modified cluster.
3. The district operation emissions and emission per GFA values differ between the original DTN optimization and the dynamic DTN optimization part 2, correctly reflecting the impact of the modified demand.

## Expected Impact

After implementing this fix:

1. The district operation emissions and emission per GFA values should now differ between the original DTN optimization and the dynamic DTN optimization part 2.
2. The emissions calculations will correctly reflect the changes in demand for the modified cluster.
3. All parts of the dynamic DTN optimization process will consistently use the modified demand data, including the emissions calculations.

## Testing

The fix can be tested by:

1. Running the dynamic DTN optimization part 2 script with a modified demand for a cluster.
2. Comparing the district operation emissions and emission per GFA values between the original DTN optimization and the dynamic DTN optimization part 2.
3. Verifying that the emissions values differ, reflecting the impact of the modified demand.