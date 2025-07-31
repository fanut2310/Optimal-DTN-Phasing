# Dynamic DTN Optimization Part 2 Parameter Fix

## Issue Description

When running the `dynamic-dtn-optimization-part2` script, the following error occurred:

```
AttributeError: Parameter not configured to work with this script: dynamic-dtn-optimization:heating-demand-reduction
```

This error occurred because the script was trying to access configuration parameters that were not registered for the script in the `scripts.yml` file.

## Root Cause

The `dynamic_dtn_optimization_part2.py` script creates an instance of the `DynamicDTNOptimizer` class from `dynamic_dtn_optimization.py`:

```python
# Create a DynamicDTNOptimizer instance
dynamic_optimizer = DynamicDTNOptimizer(locator, config)
```

The `DynamicDTNOptimizer` class initializes several parameters from the `dynamic_dtn_optimization` section of the configuration:

```python
self.network_type = config.dynamic_dtn_optimization.network_type
self.heating_reduction = config.dynamic_dtn_optimization.heating_demand_reduction / 100.0
self.cooling_reduction = config.dynamic_dtn_optimization.cooling_demand_reduction / 100.0
self.dhw_reduction = config.dynamic_dtn_optimization.dhw_demand_reduction / 100.0
self.electricity_reduction = config.dynamic_dtn_optimization.electricity_demand_reduction / 100.0
self.num_last_clusters = config.dynamic_dtn_optimization.num_last_clusters
```

However, only the `network-type` parameter was registered for the `dynamic-dtn-optimization-part2` script in the `scripts.yml` file, causing the error when trying to access the other parameters.

## Solution

The solution was to update the `scripts.yml` file to register all the missing parameters for the `dynamic-dtn-optimization-part2` script. The following parameters were added:

- `dynamic-dtn-optimization:heating-demand-reduction`
- `dynamic-dtn-optimization:cooling-demand-reduction`
- `dynamic-dtn-optimization:dhw-demand-reduction`
- `dynamic-dtn-optimization:electricity-demand-reduction`
- `dynamic-dtn-optimization:num-last-clusters`

These parameters were already registered for the `dynamic-dtn-optimization` script, so they just needed to be added to the `dynamic-dtn-optimization-part2` script as well.

## Changes Made

The `scripts.yml` file was updated to add the missing parameters to the `dynamic-dtn-optimization-part2` script:

```yaml
  - name: dynamic-dtn-optimization-part2
    label: "Dynamic DTN Optimization Part 2: Rerun Optimization"
    description: |
      Alternative implementation of Dynamic DTN Optimization Part 2 that directly uses files in the temp scenario folder
      without relying on complex redirection mechanisms or inheritance from other modules. This implementation
      avoids path handling issues that may occur with the standard implementation.
      
      **IMPORTANT**: This module requires results from Dynamic DTN Optimization Part 1.
      Please run Dynamic DTN Optimization Part 1 with the same network type before running this module.
    interfaces: [ cli, dashboard ]
    module: cea.optimization_new.dynamic_dtn_optimization_part2
    parameters: [ 'general:scenario', 'general:multiprocessing', 'general:number-of-cpus-to-keep-free',
                  'dynamic-dtn-optimization:network-type', 'dtn-expansion-optimization:testing-clusters',
                  'dynamic-dtn-optimization:heating-demand-reduction',
                  'dynamic-dtn-optimization:cooling-demand-reduction',
                  'dynamic-dtn-optimization:dhw-demand-reduction',
                  'dynamic-dtn-optimization:electricity-demand-reduction',
                  'dynamic-dtn-optimization:num-last-clusters',
                  'dtn-expansion-optimization:num-phases',
                  ... (other parameters) ... ]
```

## Testing

After making these changes, the `dynamic-dtn-optimization-part2` script should run without the AttributeError. The script will now be able to access all the required parameters from the `dynamic_dtn_optimization` section of the configuration.

## Recommendations for Future Development

When creating scripts that use classes or functions from other scripts, make sure to register all the configuration parameters that are used by those classes or functions in the `scripts.yml` file. This will prevent similar errors in the future.

It's also a good practice to check for other potentially missing parameters when fixing issues like this, to ensure that all required parameters are properly registered.