# Dynamic DTN Optimization Temp Scenario Total Demand Fix

## Issue Description

The dynamic DTN optimization module was experiencing an issue where the `TempScenarioLocator` class in `dynamic_dtn_optimization_part2.py` was not correctly overriding the `get_total_demand()` method, resulting in the following warning:

```
14:02:19 | WARNING | WARNING: temp_locator is returning the same Total_demand.csv path as the original locator!
14:02:19 | WARNING | This suggests that the TempScenarioLocator is not correctly overriding get_total_demand()
```

This issue was preventing the dynamic DTN optimization module from correctly using the modified demand files in the temporary scenario, which is essential for generating updated metrics based on the modified demands.

## Solution Approach

Instead of relying on the `TempScenarioLocator` class to override the `get_total_demand()` method, we've implemented a more direct and reliable approach:

1. Added new functions to `inputlocator.py` that directly point to the Total_demand.csv and Total_demand_hourly.csv files in the temporary scenario:
   - `get_dynamic_dtn_optimization_temp_scenario_total_demand()`
   - `get_dynamic_dtn_optimization_temp_scenario_total_demand_hourly()`

2. Updated the `generate_updated_metrics()` function in `dynamic_dtn_optimization_part2.py` to use these new functions instead of relying on `TempScenarioLocator.get_total_demand()`.

3. Created a custom `CustomPipeLayoutGenerator` class that explicitly uses the temp scenario's total demand file when generating the updated metrics.

## Why This Approach Is Better

This approach offers several advantages over relying on the `TempScenarioLocator` class:

1. **Directness**: The new functions provide a direct way to access the total demand files in the temporary scenario, without relying on method overriding in a subclass.

2. **Reliability**: By bypassing the `TempScenarioLocator` class, we eliminate potential issues with inheritance and method overriding, which can be tricky to debug.

3. **Consistency**: All path functions are defined in one place (`inputlocator.py`), following the same pattern, which makes the code more consistent and easier to maintain.

4. **Maintainability**: If the folder structure changes in the future, you only need to update the functions in `inputlocator.py`, rather than finding all instances of manual path construction or method overriding.

5. **Clarity**: The code is more explicit about where it's getting the total demand files from, making it easier to understand and debug.

## Changes Made

### 1. Added New Functions to `inputlocator.py`

```python
def get_dynamic_dtn_optimization_temp_scenario_total_demand(self, format='csv'):
    """
    Returns the path to the total demand file in the temporary scenario for dynamic DTN optimization
    
    Parameters:
    -----------
    format : str, optional
        File format (default: 'csv')
        
    Returns:
    --------
    str
        Path to the total demand file in the temporary scenario
    """
    return os.path.join(self.get_dynamic_dtn_optimization_temp_scenario_demand_folder(), f'Total_demand.{format}')
    
def get_dynamic_dtn_optimization_temp_scenario_total_demand_hourly(self, format='csv'):
    """
    Returns the path to the hourly total demand file in the temporary scenario for dynamic DTN optimization
    
    Parameters:
    -----------
    format : str, optional
        File format (default: 'csv')
        
    Returns:
    --------
    str
        Path to the hourly total demand file in the temporary scenario
    """
    return os.path.join(self.get_dynamic_dtn_optimization_temp_scenario_demand_folder(), f'Total_demand_hourly.{format}')
```

### 2. Updated `generate_updated_metrics()` in `dynamic_dtn_optimization_part2.py`

```python
def generate_updated_metrics(locator, temp_locator, network_type, testing_clusters=None):
    """
    Generate updated metrics based on demand files in the temp scenario.
    """
    log().info("Generating updated metrics using temp scenario demand files...")

    # Use the new direct functions to get the paths to the total demand files
    original_total_demand = Path(locator.get_total_demand())
    temp_total_demand = Path(locator.get_dynamic_dtn_optimization_temp_scenario_total_demand())

    log().info(f"Original total demand path: {original_total_demand}")
    log().info(f"Temp scenario total demand path: {temp_total_demand}")

    if str(original_total_demand) == str(temp_total_demand):
        log().warning("WARNING: The paths to the original and temp scenario total demand files are the same!")
        log().warning("This suggests that the temp scenario may not have been properly created.")
    else:
        log().info("✓ Successfully identified distinct total demand files for original and temp scenarios")

    # Create directory for updated metrics
    output_dir = Path(locator.get_dynamic_dtn_optimization_updated_metrics_folder())
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create a custom PipeLayoutGenerator that uses the temp scenario's total demand file
    class CustomPipeLayoutGenerator(PipeLayoutGenerator):
        def _load_inputs(self):
            """Override to ensure we use the temp scenario's total demand file."""
            # Load cluster assignments
            cluster_edges_path = Path(self.locator.get_dtn_expansion_optimization_results_folder()) / "cluster_edges.csv"
            self.cluster_edges = pd.read_csv(cluster_edges_path)
            
            cluster_nodes_path = Path(self.locator.get_dtn_cluster_nodes_file())
            self.cluster_nodes = pd.read_csv(cluster_nodes_path)
            
            # Load edge-node matrix
            edge_node_path = Path(self.locator.get_thermal_network_edge_node_matrix_file(self.network_type))
            self.edge_node_matrix = pd.read_csv(edge_node_path, index_col=0)
            
            # CRITICAL CHANGE: Explicitly use the temp scenario's total demand file
            total_demand_path = temp_total_demand
            log().info(f"Loading total demand from temp scenario: {total_demand_path}")
            
            # Verify the file exists
            if not total_demand_path.exists():
                log().error(f"Total demand file not found in temp scenario: {total_demand_path}")
                raise FileNotFoundError(f"Total demand file not found in temp scenario: {total_demand_path}")
            
            self.total_demand = pd.read_csv(total_demand_path)
            
            # Rest of the method remains the same...
    
    log().info(
        f"Creating CustomPipeLayoutGenerator with network_type={network_type}, testing_clusters={testing_clusters}")
    generator = CustomPipeLayoutGenerator(
        locator=locator,  # Use the original locator with our custom override
        network_type=network_type,
        phase=1,
        testing_clusters=testing_clusters
    )

    # Generate the updated metrics
    log().info("Generating pipe layouts with temp scenario demand files...")
    metrics_df = generator.generate_pipe_layouts()

    # Save the updated metrics
    output_file = output_dir / "clusters_metrics_updated.csv"
    metrics_df.to_csv(output_file, index=False)
    log().info(f"Saved updated metrics to {output_file}")

    return metrics_df
```

## Expected Outcome

With these changes, the dynamic DTN optimization module should now correctly use the modified demand files in the temporary scenario when generating updated metrics. The warning about `temp_locator` returning the same Total_demand.csv path as the original locator should no longer appear.

This approach provides a more direct and reliable way to access the total demand files in the temporary scenario, making the code more maintainable and less prone to errors.