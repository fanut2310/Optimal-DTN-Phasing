# Dynamic DTN Network Shapefile Fix

## Issue Description

When running the `dynamic_dtn_optimization_part2.py` script, the following error occurred:

```
18:24:02 | ERROR | C:\Users\changf\OneDrive - ETH Zurich\CEA_projects\base_design\01_base_design_2025\outputs\data\optimization\dynamic_dtn_optimization\temp_scenario\outputs\data\thermal-network\DH_network_edges.shp: No such file or directory
18:24:02 | ERROR | Error loading network data: C:\Users\changf\OneDrive - ETH Zurich\CEA_projects\base_design\01_base_design_2025\outputs\data\optimization\dynamic_dtn_optimization\temp_scenario\outputs\data\thermal-network\DH_network_edges.shp: No such file or directory
```

The script was looking for a file named `DH_network_edges.shp` in the thermal network folder, but this file doesn't exist. The actual shapefile is located at a different path, specifically in a subdirectory of the thermal network folder.

## Root Cause

The issue was in the `_load_network_data()` method of `dynamic_dtn_optimization_part2.py`, which was manually constructing file paths for the network edge and node shapefiles:

```python
# Get thermal network folder
thermal_network_folder = Path(self.locator.get_thermal_network_folder())

# Load network edges
network_edges_file = thermal_network_folder / f"{self.network_type}_network_edges.shp"
self.network_edges = _read_shp_force_2d(network_edges_file)

# Load network nodes
network_nodes_file = thermal_network_folder / f"{self.network_type}_network_nodes.shp"
self.network_nodes = _read_shp_force_2d(network_nodes_file)
```

This code was looking for files named `DH_network_edges.shp` and `DH_network_nodes.shp` directly in the thermal network folder. However, the actual shapefiles are located in a subdirectory and have different names:

```
<thermal_network_folder>\DH\edges.shp
<thermal_network_folder>\DH\nodes.shp
```

The original DTN expansion optimization module (`DTN_expansion_optimization.py`) uses the `get_network_layout_edges_shapefile()` and `get_network_layout_nodes_shapefile()` methods from the `InputLocator` class to get the correct file paths, but `dynamic_dtn_optimization_part2.py` was not using these methods.

## Solution

The solution was to modify the `_load_network_data()` method to use the proper InputLocator methods instead of manually constructing the file paths:

```python
def _load_network_data(self):
    """Load network data from the thermal network files."""
    try:
        # Load network edges using the proper InputLocator method
        network_edges_file = self.locator.get_network_layout_edges_shapefile(self.network_type)
        log().info(f"Loading network edges from: {network_edges_file}")
        self.network_edges = _read_shp_force_2d(network_edges_file)
        
        # Load network nodes using the proper InputLocator method
        network_nodes_file = self.locator.get_network_layout_nodes_shapefile(self.network_type)
        log().info(f"Loading network nodes from: {network_nodes_file}")
        self.network_nodes = _read_shp_force_2d(network_nodes_file)
        
        # Create a mapping of edges to clusters
        self._create_edge_cluster_mapping()
        
    except Exception as e:
        log().error(f"Error loading network data: {e}")
        raise
```

These changes ensure that the script uses the correct methods to get the file paths, which will point to the actual shapefile locations. The added logging statements will also help with debugging by showing the paths being used.

## Benefits of Using InputLocator Methods

Using the InputLocator methods instead of manually constructing file paths has several benefits:

1. **Consistency**: The code now uses the same approach as the original DTN expansion optimization module, ensuring consistent behavior.

2. **Maintainability**: If the file structure changes in the future, only the InputLocator methods need to be updated, not every place where file paths are constructed manually.

3. **Error Handling**: The InputLocator methods include error handling and validation, making the code more robust.

4. **Clarity**: The code is now more self-documenting, as the method names clearly indicate what files are being accessed.

## Testing

To test the changes, run the following commands:

1. First, run `dynamic_dtn_optimization.py` to create the temporary scenario:
   ```
   python -m cea.optimization_new.dynamic_dtn_optimization --scenario <path_to_scenario>
   ```

2. Then, run `dynamic_dtn_optimization_part2.py` to verify it can find all required files:
   ```
   python -m cea.optimization_new.dynamic_dtn_optimization_part2 --scenario <path_to_scenario>
   ```

The script should now run without errors related to missing shapefile files.

## Conclusion

This fix addresses the issue where `dynamic_dtn_optimization_part2.py` was looking for network edge and node shapefiles in the wrong location. By using the proper InputLocator methods, the script now correctly locates these files, making it more robust and consistent with the rest of the codebase.