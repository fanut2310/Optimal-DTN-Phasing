# Dynamic DTN Optimization Part 2 get_total_demand() Fix Documentation

## Issue Description

The dynamic DTN optimization part 2 was not correctly calculating emissions based on the modified demand data. While the modified demand files were correctly created and placed in the temporary scenario folder, the emissions calculations were still using the original demand data from the original scenario.

This resulted in the district operation emissions and emission per GFA values being the same between the original DTN optimization and the dynamic DTN optimization part 2, even though the demand for certain clusters (e.g., cluster 1) had been modified.

## Root Cause

The issue was in how the `lca_operation` function was called in the `calculate_district_emissions_new` method of the `DTNExpansionOptimizer` class in `dynamic_dtn_optimization_part2.py`. The problem had two key aspects:

1. **Method Signature Mismatch**: The `get_total_demand()` method in the `TempScenarioLocator` class in `dynamic_dtn_optimization_part2.py` didn't have the `format` parameter, which is present in the `InputLocator` class's implementation.

2. **Total Demand File Access**: The `lca_operation` function in `cea/analysis/lca/operation.py` reads the demand data using:
   ```python
   demand = pd.read_csv(locator.get_total_demand())
   ```
   
   When `lca_operation` calls `locator.get_total_demand()`, it expects to be able to use the default value of the `format` parameter, which is 'csv'. However, since the `get_total_demand()` method in `TempScenarioLocator` didn't have this parameter, it couldn't be called correctly, resulting in an error or using the original implementation from `InputLocator`.

## Solution

The fix modifies the `get_total_demand()` method in the `TempScenarioLocator` class in `dynamic_dtn_optimization_part2.py` to match the signature in `InputLocator`:

```python
def get_total_demand(self, format='csv'):
    """
    Get the path to the total demand file in the temp scenario.
    
    Parameters:
    -----------
    format : str, optional
        Format of the file (default: 'csv')
        
    Returns:
    --------
    str
        Path to the total demand file
    """
    path = os.path.join(self.scenario, 'outputs', 'data', 'demand', f'Total_demand.{format}')
    
    if not os.path.exists(path):
        log().error(f"Total demand file not found in temp scenario: {path}")
        raise FileNotFoundError(f"Total demand file not found in temp scenario: {path}")
    
    return path
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

## Implementation Details

The fix was implemented by:

1. Modifying the `get_total_demand()` method in the `TempScenarioLocator` class in `dynamic_dtn_optimization_part2.py` to add the `format` parameter with a default value of 'csv'.
2. Updating the docstring to document the new parameter.
3. Modifying the path construction to use the `format` parameter.

This change ensures that the method signature matches the one in `InputLocator`, allowing `lca_operation` to call it correctly.