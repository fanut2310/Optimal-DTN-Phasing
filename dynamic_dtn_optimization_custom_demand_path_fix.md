# Dynamic DTN Optimization Custom Demand Path Fix

## Issue Description
The dynamic DTN optimization part 2 was not properly using the modified demand files from part 1, resulting in incorrect emissions calculations:
- Phase 3 emissions were not reflecting demand modifications made to cluster 1
- Phase 2 emissions incorrectly showed the same values as phase 3

## Root Cause
When `lca_operation` was called with a custom supply path, it was still using the original demand file instead of the modified demand file from the temporary scenario.

## Changes Made

### 1. Modified `operation.py`
- Added a new parameter `custom_demand_path` to the `lca_operation` function
- Updated the function to use this custom demand path when provided
- Updated the `main` function for backward compatibility

### 2. Updated `dynamic_dtn_optimization_part2.py`
- Modified both calls to `lca_operation` to include the new `custom_demand_path` parameter
- Used the temp scenario's total demand file path as the value for this parameter

## How the Fix Works
1. The `calculate_district_emissions_new` method now passes both the custom supply path and the custom demand path to `lca_operation`
2. The custom demand path points to the modified demand file in the temporary scenario
3. The `lca_operation` function uses this modified demand file when calculating emissions
4. This ensures phase 3 emissions correctly reflect demand modifications, while phase 2 emissions remain unaffected

## Testing
Verify that:
- Phase 3 emissions reflect the demand modifications made to cluster 1
- Phase 2 emissions are different from phase 3 emissions
- Phase 2 emissions match the original values from before the dynamic DTN optimization

## Backward Compatibility
- The `custom_demand_path` parameter is optional with a default value of `None`
- Existing code that calls `lca_operation` will continue to work without changes