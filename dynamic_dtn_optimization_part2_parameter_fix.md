# Dynamic DTN Optimization Part 2 Parameter Fix

## Issue Description

The Dynamic DTN Rerun Optimization Part 2 script was failing with errors like:

```
Parameter not configured to work with this script: dtn-expansion-optimization:population-size
Parameter not configured to work with this script: dtn-expansion-optimization:num-phases
```

These errors occurred because the script was trying to access parameters from the `dtn-expansion-optimization` section of the configuration, but these parameters were not properly configured to work with the `dynamic_dtn_optimization_part2.py` script.

## Analysis

After examining the code in `dynamic_dtn_optimization_part2.py`, I found that the `run_optimization` method was accessing several parameters from the `dtn_expansion_optimization` section:

```python
# Set up parameters for optimization
self.num_phases = self.config.dtn_expansion_optimization.num_phases

# Parse phase durations from config
phase_durations_str = self.config.dtn_expansion_optimization.phase_durations

# Set interest rate
self.interest_rate = self.config.dtn_expansion_optimization.interest_rate

# Set energy price
self.energy_price = self.config.dtn_expansion_optimization.energy_price

# Set objective function
self.objective_function = self.config.dtn_expansion_optimization.objective_function

# Set multi-objective mode
self.multi_objective_mode = self.config.dtn_expansion_optimization.multi_objective_mode

# Parse multi-objective functions from config
multi_objective_functions_str = self.config.dtn_expansion_optimization.multi_objective_functions
```

Additionally, in the `run` method, it was trying to access:
- `population_size`
- `num_generations`

However, in the `scripts.yml` file, the `dynamic-dtn-optimization-part2` module was only configured to use a few parameters:

```yaml
parameters: [ 'general:scenario', 'general:multiprocessing', 'general:number-of-cpus-to-keep-free',
              'dynamic-dtn-optimization:network-type', 'dtn-expansion-optimization:testing-clusters' ]
```

This mismatch between the parameters being accessed in the code and the parameters configured in `scripts.yml` was causing the AttributeError exceptions.

## Solution

I updated the `scripts.yml` file to include all the necessary parameters from the `dtn-expansion-optimization` section for the `dynamic-dtn-optimization-part2` module:

```yaml
parameters: [ 'general:scenario', 'general:multiprocessing', 'general:number-of-cpus-to-keep-free',
              'dynamic-dtn-optimization:network-type', 'dtn-expansion-optimization:testing-clusters',
              'dtn-expansion-optimization:num-phases',
              'dtn-expansion-optimization:phase-durations',
              'dtn-expansion-optimization:interest-rate',
              'dtn-expansion-optimization:energy-price',
              'dtn-expansion-optimization:objective-function',
              'dtn-expansion-optimization:multi-objective-mode',
              'dtn-expansion-optimization:multi-objective-functions',
              'dtn-expansion-optimization:population-size',
              'dtn-expansion-optimization:num-generations' ]
```

This ensures that all the required parameters are properly configured to work with the module, which should resolve the AttributeError exceptions.

## Benefits of This Approach

1. **Comprehensive Solution**: By adding all the necessary parameters to the configuration, we ensure that all parts of the code can access the parameters they need.

2. **Consistency with Other Modules**: The `dynamic-dtn-optimization` module (Part 1) already uses these parameters, so it makes sense for the Part 2 module to use them as well.

3. **Maintainability**: This approach makes the code more maintainable by ensuring that the configuration accurately reflects the parameters used by the code.

4. **Robustness**: While the code already had some error handling for missing parameters (using try-except blocks), this solution is more robust because it ensures that the parameters are properly configured in the first place.

## Alternative Approaches Considered

1. **Add try-except blocks for all parameter accesses**: This would involve modifying the `run_optimization` method to handle the case when parameters are not available, similar to what's done in the `run` method. However, this would be more complex and would still require the parameters to be properly configured for optimal functionality.

2. **Use parameters from the dynamic-dtn-optimization section**: Another approach would be to modify the code to use parameters from the `dynamic-dtn-optimization` section instead of the `dtn-expansion-optimization` section. However, this would require more extensive code changes and might not be consistent with how the parameters are used in other parts of the codebase.

3. **Create a shared parameter section**: We could create a new section in the configuration file for parameters shared between different optimization modules. However, this would require more extensive changes to the configuration system.

## Conclusion

The implemented solution addresses the immediate issue while maintaining the intended behavior of the script. By adding the necessary parameters to the configuration, we ensure that the script can access all the parameters it needs without errors.