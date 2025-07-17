# Dynamic DTN Rerun Optimization - Implementation Changes

## Problem Summary

The `dynamic_dtn_rerun_optimization.py` script was experiencing a stack overflow error (exit code `0xC00000FD`) when attempting to generate updated metrics based on modified demand files. This error occurred because the original implementation was:

1. Processing each cluster combination individually in small batches
2. Making repeated calls to memory-intensive methods like `get_required_pipes_for_clusters()` and `calculate_metrics()`
3. Using inefficient row-by-row updates to a pandas DataFrame
4. Creating many temporary objects that could lead to excessive memory usage

Despite attempts to mitigate these issues with batch processing and garbage collection, the recursive nature of some of the methods still led to stack overflow.

## Solution Implemented

The solution replaces the inefficient implementation with a direct call to the `generate_pipe_layouts()` method from the `PipeLayoutGenerator` class in `DTN_expansion_optimization.py`. This method is already designed to efficiently calculate metrics for all cluster combinations at once.

### Key Changes:

1. **Direct Use of Proven Code**: Instead of reimplementing the logic, we now use the exact same code that successfully generated the original metrics.

2. **Efficient Processing**: The `generate_pipe_layouts()` method processes combinations one at a time and manages memory properly, without the recursive calls that were causing stack overflow.

3. **Consistent Results**: Using the same code ensures that the metrics are calculated consistently between the original and updated versions.

4. **Custom Output Location**: The output folder is overridden to save results in the dynamic_dtn_optimization/updated_metrics directory.

5. **Clear File Naming**: The output file is renamed from "clusters_metrics.csv" to "clusters_metrics_updated.csv" to make it clear these are updated metrics.

## How to Test the Changes

A batch file (`test_dynamic_dtn_rerun.bat`) has been created to test the modified script. To run the test:

1. Double-click the `test_dynamic_dtn_rerun.bat` file in the CityEnergyAnalyst directory
2. The script will execute the dynamic-dtn-rerun-optimization command with the same parameters you were using before
3. Monitor the console output for any errors or warnings

## Expected Results

When running the modified script, you should expect:

1. No stack overflow error (exit code `0xC00000FD`)
2. Successful generation of updated metrics based on modified demand files
3. Creation of a `clusters_metrics_updated.csv` file in the `outputs\data\optimization\dynamic_dtn_optimization\updated_metrics` directory
4. Successful completion of the DTN rerun optimization process

## Additional Notes

1. **Processing Time**: The script may still take some time to run, as it needs to calculate metrics for all possible cluster combinations. However, it should complete successfully without memory errors.

2. **Memory Usage**: The memory usage should be significantly lower than before, as the script now processes combinations one at a time without creating excessive temporary objects.

3. **Fallback Mechanism**: If any errors occur during the metrics generation, the script will fall back to using the original metrics, ensuring that the optimization can still proceed.

4. **Future Improvements**: For even better performance, consider implementing parallel processing for the metrics calculation, which could further reduce processing time.

## Technical Details of the Fix

The core of the fix is in the `_generate_updated_metrics()` method in `dynamic_dtn_rerun_optimization.py`. The key implementation changes are:

```python
# Create a PipeLayoutGenerator with the modified demands locator
generator = PipeLayoutGenerator(
    locator=mod_loc,
    network_type=self.ntype,
    phase=1,
    testing_clusters=self.testing_clusters
)

# Override the output folder to save to our custom location
generator.output_folder = output_folder

# Use the generator's built-in method to calculate all metrics at once
updated_metrics = generator.generate_pipe_layouts()
```

This approach leverages the existing, well-tested implementation in the `PipeLayoutGenerator` class, which is designed to handle large datasets efficiently without causing stack overflow.