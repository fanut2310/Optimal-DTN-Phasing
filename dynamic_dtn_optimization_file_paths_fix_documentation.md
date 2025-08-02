# Dynamic DTN Optimization File Paths Fix Documentation

## Issue Description

The dynamic DTN optimization part 2 module was encountering several issues when trying to access certain files:

1. **Cost File Path Issue (Q1)**: The module was looking for cost files in the wrong location:
   ```
   16:31:38 | WARNING | Dynamic DTN cost file not found: C:\Users\changf\OneDrive - ETH Zurich\CEA_projects\base_design\01_base_design_2025\outputs\data\optimization\dynamic_dtn_optimization\temp_scenario\outputs\data\optimization\dynamic_dtn_optimization\thermal_network\DH_costs.csv. Falling back to original cost file.
   ```
   The actual file was at:
   ```
   C:\Users\changf\OneDrive - ETH Zurich\CEA_projects\base_design\01_base_design_2025\outputs\data\optimization\dynamic_dtn_optimization\temp_scenario\outputs\data\thermal-network\DH_costs.csv
   ```

2. **Pump Cost File Warning (Q2)**: The module was showing warnings about pump cost files:
   ```
   16:31:38 |  INFO | Trying to read pump cost data
   16:31:38 |  INFO | Pump cost file not found in temporary scenario, using default pump cost data
   16:31:38 |  INFO | Created default pump cost data
   ```

3. **Cluster Metrics Error (Q3)**: The module was encountering errors when creating cluster metrics:
   ```
   16:31:38 | ERROR | Error creating cluster metrics mapping: False
   ```

## Changes Made

### 1. Cost File Path Fix (Q1)

Added the `get_thermal_network_folder()` and `get_dynamic_dtn_network_layout_costs_file()` methods to the `TempScenarioLocator` class in `dynamic_dtn_optimization.py`:

```python
def get_thermal_network_folder(self):
    """
    Get the path to the thermal network folder in the temp scenario.
    
    Returns:
    --------
    str
        Path to the thermal network folder
    """
    path = os.path.join(self.scenario, 'outputs', 'data', 'thermal-network')
    
    if not os.path.exists(path):
        logging.error(f"Thermal network folder not found in temp scenario: {path}")
        raise FileNotFoundError(f"Thermal network folder not found in temp scenario: {path}")
    
    return path
    
def get_dynamic_dtn_network_layout_costs_file(self, network_type, network_name=""):
    """
    Override to return the path to the network layout costs file in the temp scenario.
    
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
    # Use get_thermal_network_folder() directly
    file_name = f"{network_type}_costs.csv"
    return os.path.join(self.get_thermal_network_folder(), file_name)
```

This ensures that the method returns the correct path to the cost file in the temporary scenario.

### 2. Pump Cost File Warning (Q2)

No changes were made for this issue as it was determined to be expected behavior. The code is designed to try to load pump cost data from a file, and if the file doesn't exist, it creates default pump cost data. The warning message is just informational.

### 3. Cluster Metrics Error (Q3)

Enhanced the `_create_cluster_metrics_mapping()` method in `dynamic_dtn_optimization_part2.py` with better error handling and validation:

```python
def _create_cluster_metrics_mapping(self):
    """Create a mapping of cluster metrics."""
    try:
        # Create a mapping of cluster metrics
        self.cluster_metrics = {}
        
        # Directly use the cluster nodes file which contains the 'cluster' column
        # Use the original locator to get the cluster nodes file
        original_locator = cea.inputlocator.InputLocator(self.locator.original_locator.scenario)
        cluster_nodes_path = Path(original_locator.get_dtn_cluster_nodes_file())
        if cluster_nodes_path.exists():
            log().info(f"Loading cluster nodes from: {cluster_nodes_path}")
            cluster_nodes_df = pd.read_csv(cluster_nodes_path)
            
            # Verify that the 'cluster' column exists
            if 'cluster' not in cluster_nodes_df.columns:
                log().error(f"'cluster' column not found in cluster nodes file: {cluster_nodes_path}")
                raise ValueError(f"'cluster' column not found in cluster nodes file: {cluster_nodes_path}")
            
            # Group by cluster and create metrics for each cluster
            for cluster in self.all_clusters:
                # Get buildings in this cluster
                cluster_rows = cluster_nodes_df[cluster_nodes_df['cluster'] == cluster]
                if cluster_rows.empty:
                    log().warning(f"No buildings found for cluster {cluster} in cluster nodes file")
                    buildings = []
                else:
                    buildings = cluster_rows['building'].tolist()
                
                # Create metrics for this cluster
                self.cluster_metrics[cluster] = {
                    'cluster': cluster,
                    'buildings': buildings,
                    'num_buildings': len(buildings)
                }
                
                # Add additional metrics if available in metrics_df
                cluster_rows = self.metrics_df[self.metrics_df.get('cluster', -1) == cluster]
                if not cluster_rows.empty:
                    # Update with metrics from metrics_df
                    self.cluster_metrics[cluster].update(cluster_rows.iloc[0].to_dict())
            
            log().info(f"Created metrics for {len(self.cluster_metrics)} clusters")
        else:
            log().warning(f"Cluster nodes file not found: {cluster_nodes_path}")
            raise FileNotFoundError(f"Cluster nodes file not found: {cluster_nodes_path}")
            
    except Exception as e:
        log().error(f"Error creating cluster metrics mapping: {e}")
        # Print the full traceback for debugging
        import traceback
        log().error(f"Traceback: {traceback.format_exc()}")
        raise
```

The key improvements are:
1. Verifying that the 'cluster' column exists in the cluster nodes file
2. Handling the case when no buildings are found for a cluster
3. Adding more detailed error logging with a full traceback

## Expected Impact

### 1. Cost File Path Fix (Q1)

The dynamic DTN optimization part 2 module should now be able to find the cost files in the correct location. This will eliminate the warning message and ensure that the module uses the correct cost data.

### 2. Pump Cost File Warning (Q2)

The warning message will still appear, but it's just informational and doesn't indicate an error. The module will continue to use default pump cost data if the pump cost file doesn't exist.

### 3. Cluster Metrics Error (Q3)

The enhanced error handling and validation in the `_create_cluster_metrics_mapping()` method should help diagnose and resolve issues with the cluster nodes file. The method will now:
- Verify that the 'cluster' column exists in the cluster nodes file
- Handle the case when no buildings are found for a cluster
- Provide more detailed error logging with a full traceback

This should make it easier to identify and fix issues with the cluster nodes file.

## Testing

A test script `test_dynamic_dtn_file_paths.py` has been created to verify that the changes resolve the issues. The script tests that the `get_dynamic_dtn_network_layout_costs_file` method returns the correct path and notes that more comprehensive testing would be needed for the `_create_cluster_metrics_mapping` method.

To run the test script:

```
python -m test_dynamic_dtn_file_paths
```

## Conclusion

The changes made to the dynamic DTN optimization part 2 module should resolve the issues with file paths and improve error handling. The module should now be able to find the cost files in the correct location and provide more detailed error messages when issues occur with the cluster nodes file.