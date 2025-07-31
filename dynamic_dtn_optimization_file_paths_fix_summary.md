# Dynamic DTN Optimization File Paths Fix Summary

## Issue Description

The dynamic DTN optimization part 2 module was encountering errors when trying to access certain files:

```
14:32:22 | WARNING | Dynamic DTN cost file not found: C:\Users\changf\OneDrive - ETH Zurich\CEA_projects\base_design\01_base_design_2025\outputs\data\optimization\dynamic_dtn_optimization\temp_scenario\outputs\data\optimization\dynamic_dtn_optimization\thermal_network\DH_costs.csv. Falling back to original cost file.
14:32:22 |  INFO | Loading cost data from: C:\Users\changf\OneDrive - ETH Zurich\CEA_projects\base_design\01_base_design_2025\outputs\data\optimization\dynamic_dtn_optimization\temp_scenario\outputs\data\thermal-network\DH_costs.csv
14:32:22 |  INFO | Loaded pipe cost data: 27 rows
14:32:22 |  INFO | Trying to read pump cost data
14:32:22 |  INFO | Pump cost file not found in temporary scenario, using default pump cost data
14:32:22 |  INFO | Created default pump cost data
14:32:22 |  INFO | 'cluster' column not found, using 'clusters' column instead
14:32:22 |  INFO | Extracted clusters from 'clusters' column: [1, 2, 3, 4, 7]
14:32:22 |  INFO | Filtering clusters to include only testing clusters: [1, 2, 3, 4, 7]
14:32:22 |  INFO | Filtered clusters: [1, 2, 3, 4, 7]
14:32:22 |  INFO | 'buildings' column not found, loading from cluster_nodes.csv
14:32:22 |  INFO | Loading cluster nodes from: C:\Users\changf\OneDrive - ETH Zurich\CEA_projects\base_design\01_base_design_2025\outputs\data\optimization\dynamic_dtn_optimization\temp_scenario\outputs\data\optimization\dtn_expansion\cluster_nodes.csv
14:32:22 | ERROR | Error creating cluster metrics mapping: False
```

The main issues were:

1. The `get_dynamic_dtn_network_layout_costs_file` method in `TempScenarioLocator` was looking for the cost file in the wrong location:
   ```
   C:\Users\changf\OneDrive - ETH Zurich\CEA_projects\base_design\01_base_design_2025\outputs\data\optimization\dynamic_dtn_optimization\temp_scenario\outputs\data\optimization\dynamic_dtn_optimization\thermal_network\DH_costs.csv
   ```
   But the actual file was at:
   ```
   C:\Users\changf\OneDrive - ETH Zurich\CEA_projects\base_design\01_base_design_2025\outputs\data\optimization\dynamic_dtn_optimization\temp_scenario\outputs\data\thermal-network\DH_costs.csv
   ```

2. The `_load_cluster_data` method was using `self.locator.get_dtn_cluster_nodes_file()` to get the cluster nodes file, but it should have been using the original locator instead of the temporary locator.

## Changes Made

1. Added the `get_dynamic_dtn_network_layout_costs_file` method to the `TempScenarioLocator` class in `dynamic_dtn_optimization_part2.py`:

```python
def get_dynamic_dtn_network_layout_costs_file(self, network_type, network_name=""):
    """
    Override to return the path to the network layout costs file in the temp scenario.
    
    This override ensures that the method returns the correct path to the cost file
    in the temporary scenario.
    
    Parameters:
    -----------
    network_type : str
        Type of the network (e.g., 'DH', 'DC')
    network_name : str, optional
        Name of the network
        
    Returns:
    --------
    str
        Path to the network layout costs file in the temp scenario
    """
    # Use get_thermal_network_folder() directly instead of get_dynamic_dtn_optimization_thermal_network_folder()
    file_name = f"{network_type}_costs.csv"
    return os.path.join(self.get_thermal_network_folder(), file_name)
```

2. Modified the `_load_cluster_data` method in `dynamic_dtn_optimization_part2.py` to use the original locator for getting the cluster nodes file:

```python
# Alternative approach: load from cluster_nodes.csv in the original DTN results
log().info("'buildings' column not found, loading from cluster_nodes.csv")
try:
    # Use the original locator to get the cluster nodes file
    original_locator = cea.inputlocator.InputLocator(self.locator.original_locator.scenario)
    cluster_nodes_path = Path(original_locator.get_dtn_cluster_nodes_file())
    if cluster_nodes_path.exists():
        log().info(f"Loading cluster nodes from: {cluster_nodes_path}")
        cluster_nodes = pd.read_csv(cluster_nodes_path)
        for cluster in self.all_clusters:
            buildings = cluster_nodes[cluster_nodes['cluster'] == cluster]['building'].tolist()
            self.buildings_by_cluster[cluster] = buildings
            log().info(f"Cluster {cluster} has {len(buildings)} buildings")
    else:
        log().warning(f"Cluster nodes file not found: {cluster_nodes_path}")
except Exception as e:
    log().warning(f"Error loading cluster nodes data: {e}")
```

## Explanation

1. **Cost File Path Fix**: The `get_dynamic_dtn_network_layout_costs_file` method in `TempScenarioLocator` now uses `self.get_thermal_network_folder()` directly instead of `self.get_dynamic_dtn_optimization_thermal_network_folder()`. This ensures that it looks for the cost file in the correct location: `<temp_scenario>/outputs/data/thermal-network/DH_costs.csv`.

2. **Cluster Nodes File Path Fix**: The `_load_cluster_data` method now uses the original locator to get the cluster nodes file. This is necessary because the cluster nodes file is in the original scenario, not the temporary scenario. By creating a new `InputLocator` instance with the original scenario path, we bypass the redirection mechanism of `TempScenarioLocator` and get the correct path to the cluster nodes file in the main scenario folder.

## Expected Behavior

With these changes, the dynamic DTN optimization part 2 module should be able to correctly access all the input files it needs, whether they're in the temporary scenario or the original scenario. Specifically:

1. When `get_dynamic_dtn_network_layout_costs_file()` is called, it will look for the cost file in the `<temp_scenario>/outputs/data/thermal-network/` directory, not the `<temp_scenario>/outputs/data/optimization/dynamic_dtn_optimization/thermal_network/` directory.

2. When loading the cluster nodes file in `_load_cluster_data()`, it will use the original locator to get the path to the file in the original scenario, not the temporary scenario.

These changes should eliminate the warnings and errors related to file paths in the dynamic DTN optimization part 2 module.

## Testing

A test script `test_file_paths.py` has been created to verify that the file paths are correct. This script:

1. Creates a `DynamicDTNOptimizer` instance
2. Gets a locator for the temporary scenario
3. Tests various file paths to ensure they're correct:
   - `get_dynamic_dtn_network_layout_costs_file` - Should point to the thermal network folder in the temporary scenario
   - `get_dtn_cluster_nodes_file` with original locator - Should point to the cluster nodes file in the original scenario
   - `get_thermal_network_folder` - Should point to the thermal network folder in the temporary scenario
   - `get_network_layout_edges_shapefile` - Should point to the edges shapefile in the temporary scenario
   - `get_network_layout_nodes_shapefile` - Should point to the nodes shapefile in the temporary scenario

To run the test script:

```
python -m test_file_paths
```

The script will print the paths to the various files and whether they exist, which can be used to verify that the changes are working correctly.