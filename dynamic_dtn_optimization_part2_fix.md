# Dynamic DTN Optimization Part 2 - Fix Summary

## Issues Addressed

1. **File Location Issues**: Script was looking for input files in original locations instead of the temporary scenario.
2. **Worksheet Not Found**: Error when trying to read 'ENERGY_PRICE' worksheet which doesn't exist.
3. **Column Not Found**: Error when trying to filter by 'type_mat' column which doesn't exist in some files.

## Solution Implemented

### 1. Created TempScenarioLocator Class

Created a class that inherits from `InputLocator` and redirects file requests to the temporary scenario:

```python
class TempScenarioLocator(cea.inputlocator.InputLocator):
    def __init__(self, original_locator, temp_scenario_path):
        super().__init__(str(temp_scenario_path))
        self.original_locator = original_locator
        # Copy necessary attributes
```

The class overrides methods to return paths in the temporary scenario with clear error messages.

### 2. Modified Main Function

Updated to:
- Check for the temp scenario folder
- Create a `TempScenarioLocator` instance
- Verify necessary files exist
- Pass the temp locator to the optimizer

```python
temp_scenario_path = Path(locator.get_dynamic_dtn_optimization_folder()) / "temp_scenario"
if not temp_scenario_path.exists():
    log().error(f"Temporary scenario folder not found: {temp_scenario_path}")
    return

temp_locator = TempScenarioLocator(locator, temp_scenario_path)

optimizer = DTNExpansionOptimizer(
    locator=temp_locator,  # Use temp locator instead of original
    # ... other parameters ...
)
```

### 3. Updated _get_energy_price Method

Modified to:
- Try 'ENERGY_PRICE' worksheet first
- Fall back to 'FEEDSTOCKS' worksheet if needed
- Map column names appropriately
- Provide clear error messages

### 4. Updated _load_cost_data Method

Modified to:
- Try 'THERMAL_GRID' worksheet first
- Try CSV format if needed
- Check for 'type_mat' column
- Try alternative column names if needed
- Create default pump data if necessary
- Provide clear error messages

## Benefits

1. **Robustness**: Handles edge cases gracefully with clear error messages
2. **Correctness**: Uses correct input files with modified demands
3. **Clarity**: More informative error messages
4. **Maintainability**: Better separation of concerns and error handling

## Testing Recommendations

1. Run `dynamic_dtn_optimization.py` first to create the temporary scenario
2. Run `dynamic_dtn_optimization_part2.py` and verify it accesses correct files
3. Check that it handles missing worksheets/columns gracefully
4. Verify results are correctly saved

## Usage Instructions

1. **Prerequisite**: Always run `dynamic_dtn_optimization.py` first to create the temporary scenario with modified demands
   ```
   cea dynamic-dtn-optimization --scenario YOUR_SCENARIO --network-type DH
   ```

2. **Run the updated script**: Use the same parameters as you would for the original script
   ```
   cea dynamic-dtn-optimization-part2 --scenario YOUR_SCENARIO --network-type DH
   ```

3. **Check the logs**: The script now provides detailed logging about which files it's accessing and any issues it encounters

4. **Review results**: Results will be saved in:
   ```
   outputs/data/optimization/dynamic_dtn_optimization/rerun_results/
   ```

## Troubleshooting

If you encounter errors:

1. **Missing temporary scenario**: Ensure `dynamic_dtn_optimization.py` completed successfully
2. **File format issues**: Check that the database files in the temporary scenario have the expected format
3. **Missing columns**: If you see errors about missing columns, check the structure of your database files
4. **Worksheet errors**: If you see errors about missing worksheets, check that your database files have the expected worksheets

## Future Improvements

1. Add more robust error handling for other potential issues
2. Implement a validation step to check the structure of all input files before starting the optimization
3. Add an option to regenerate the temporary scenario if it's missing or incomplete
4. Improve logging to provide more context about the optimization process