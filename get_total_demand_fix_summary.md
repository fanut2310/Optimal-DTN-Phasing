# TempScenarioLocator get_total_demand() Fix Summary

## Issue

The dynamic DTN optimization part 2 was not correctly calculating emissions based on the modified demand data. While the modified demand files were correctly created and placed in the temporary scenario folder, the emissions calculations were still using the original demand data from the original scenario.

## Fix

Added a `get_total_demand()` method to the `TempScenarioLocator` class in `dynamic_dtn_optimization.py` to ensure it points to the temporary scenario's total demand file:

```python
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

## Impact

This fix ensures that:

1. The `lca_operation` function reads the modified total demand data from the temporary scenario.
2. The emissions calculations reflect the changes in demand for the modified cluster.
3. The district operation emissions and emission per GFA values differ between the original DTN optimization and the dynamic DTN optimization part 2, correctly reflecting the impact of the modified demand.

## Files Modified

- `cea/optimization_new/dynamic_dtn_optimization.py`: Added `get_total_demand()` method to `TempScenarioLocator` class.

## Documentation Created

- `get_total_demand_fix_documentation.md`: Detailed documentation of the issue, root cause, solution, and expected impact.
- `get_total_demand_fix_summary.md`: This summary file.