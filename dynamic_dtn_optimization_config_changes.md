# Dynamic DTN Optimization Configuration Changes

## Issue Description

The Dynamic DTN Rerun Optimization Part 2 script was failing with errors like:

```
Parameter not configured to work with this script: dtn-expansion-optimization:population-size
Parameter not configured to work with this script: dtn-expansion-optimization:num-phases
```

These errors occurred because the script was trying to access parameters from the `dtn-expansion-optimization` section, but these parameters were not properly configured to work with the `dynamic_dtn_optimization_part2.py` script.

## Solution

I added the missing parameters to the `dynamic-dtn-optimization` section in the `default.config` file. The following parameters were added:

1. `num-phases` - Number of phases for the expansion
2. `phase-durations` - Comma-separated list of durations in years for each phase
3. `interest-rate` - Annual interest rate for NPV calculations
4. `energy-price` - Energy price per kWh for revenue calculations
5. `population-size` - Size of the population for the genetic algorithm
6. `num-generations` - Number of generations for the genetic algorithm
7. `objective-function` - Objective function to maximize or minimize

## Changes Made

```ini
# Parameters needed for dynamic DTN optimization part 2
num-phases = 3
num-phases.type = IntegerParameter
num-phases.help = Number of phases for the expansion

phase-durations = 3,3,3
phase-durations.type = StringParameter
phase-durations.help = Comma-separated list of durations in years for each phase (e.g., '3,3,3'). The length must match the number of phases.

interest-rate = 0.05
interest-rate.type = RealParameter
interest-rate.help = Annual interest rate for NPV calculations

energy-price = 0.1
energy-price.type = RealParameter
energy-price.help = Energy price per kWh for revenue calculations

population-size = 50
population-size.type = IntegerParameter
population-size.help = Size of the population for the genetic algorithm

num-generations = 30
num-generations.type = IntegerParameter
num-generations.help = Number of generations for the genetic algorithm

objective-function = NPV
objective-function.type = ChoiceParameter
objective-function.choices = NPV, ROI, emissions
objective-function.help = Objective function to maximize (NPV or ROI) or minimize (emissions)
```

## Why This Fixes the Issue

The dynamic_dtn_optimization_part2.py script was trying to access these parameters using `self.config.dtn_expansion_optimization.parameter_name`, which was causing AttributeError exceptions because the parameters were not configured to work with this script.

By adding these parameters to the `dynamic-dtn-optimization` section, they are now properly configured to work with the dynamic_dtn_optimization_part2.py script. The script can now access these parameters without errors.

## Alternative Approaches

There were a few alternative approaches that could have been taken:

1. **Modify the script**: We could have modified the script to use try-except blocks to handle the case when parameters are not available, falling back to default values. This approach was implemented in a previous fix but adding the parameters to the configuration file is a more permanent solution.

2. **Create a shared parameter section**: We could have created a new section in the configuration file for parameters shared between different optimization modules. However, this would require more extensive changes to the configuration system.

3. **Modify the configuration system**: We could have modified the configuration system to allow parameters to be accessed from different sections. This would be a more complex solution but would provide more flexibility.

The approach taken (adding the parameters to the dynamic-dtn-optimization section) is the most straightforward solution and doesn't require any changes to the script or the configuration system.