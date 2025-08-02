# Dynamic DTN Optimization Cluster Data Handling Fix

## Issue Description

The dynamic DTN optimization part 2 module was encountering errors when processing cluster data:

```
17:16:52 |  INFO | 'cluster' column not found, using 'clusters' column instead
17:16:52 |  INFO | Extracted clusters from 'clusters' column: [1, 2, 3, 4, 7]
...
17:16:53 |  INFO | Loading cluster nodes from: C:\Users\changf\OneDrive - ETH Zurich\CEA_projects\base_design\01_base_design_2025\outputs\data\optimization\dtn_expansion\cluster_nodes.csv
17:16:53 | ERROR | Error creating cluster metrics mapping: False
```

The main issues were:

1. The `_load_cluster_data()` method was checking for a 'cluster' column in the metrics file, but the updated metrics CSV actually uses a 'clusters' column instead.

2. The `_create_cluster_metrics_mapping()` method in `dynamic_dtn_optimization_part2.py` was trying to create a mapping from individual clusters to their metrics, while the original method in `DTN_expansion_optimization.py` creates a mapping from cluster combinations (like "1+2+3") to their metrics.

## Changes Made

### 1. Modified `_load_cluster_data()` Method

The `_load_cluster_data()` method was modified to directly use the 'clusters' column without checking for 'cluster' first:

```python
def _load_cluster_data(self):
    """Load cluster data from the metrics file."""
    try:
        # Directly use 'clusters' column (similar to original DTN module)
        log().info("Using 'clusters' column from metrics file")
        unique_clusters = set()
        for cluster_str in self.metrics_df['clusters']:
            # Skip cluster 0 (existing DTN)
            if cluster_str == '0':
                continue
            # Split the cluster string (e.g., '1+3+4') into individual clusters
            for c in cluster_str.split('+'):
                if c != '0':  # Skip cluster 0
                    unique_clusters.add(int(c))
        
        # Sort the clusters to maintain the same order as before
        self.all_clusters = sorted(list(unique_clusters))
        log().info(f"Extracted clusters from 'clusters' column: {self.all_clusters}")
        
        # Rest of the method remains the same...
```

### 2. Modified `_create_cluster_metrics_mapping()` Method

The `_create_cluster_metrics_mapping()` method was completely rewritten to follow the same approach as the original method in `DTN_expansion_optimization.py`:

```python
def _create_cluster_metrics_mapping(self):
    """Create a mapping of cluster metrics."""
    try:
        # Create a mapping of cluster metrics
        self.cluster_metrics = {}
        
        if self.metrics_df is not None:
            for _, row in self.metrics_df.iterrows():
                if 'clusters' in row:
                    key = row['clusters']
                    self.cluster_metrics[key] = row.to_dict()
                    
                    # also register an alias without the leading "0+"
                    if isinstance(key, str) and key.startswith('0+'):
                        alias = key[2:]  # e.g. "0+2+4" → "2+4"
                        self.cluster_metrics[alias] = row.to_dict()
                else:
                    log().warning("'clusters' column not found in metrics_df")
        else:
            log().warning("No metrics_df provided, cluster_metrics will be empty")
            
        log().info(f"Created metrics for {len(self.cluster_metrics)} clusters")
            
    except Exception as e:
        log().error(f"Error creating cluster metrics mapping: {e}")
        # Print the full traceback for debugging
        import traceback
        log().error(f"Traceback: {traceback.format_exc()}")
        raise
```

## Explanation of the Changes

### 1. `_load_cluster_data()` Method

The original method had two branches:
1. If 'cluster' column exists: Use it to get unique clusters
2. If 'cluster' column doesn't exist: Use 'clusters' column to extract individual clusters from cluster combinations

Since the updated metrics CSV uses a 'clusters' column instead of 'cluster', we modified the method to directly use the second branch (the 'clusters' column approach) without checking for the 'cluster' column first. This ensures that the method always uses the correct column.

### 2. `_create_cluster_metrics_mapping()` Method

The original method was trying to create a mapping from individual clusters to their metrics, using the cluster nodes file to get the buildings in each cluster. However, this approach was different from the one used in the original `DTN_expansion_optimization.py` module, which creates a mapping from cluster combinations to their metrics.

We completely rewrote the method to follow the same approach as the original module:
1. It iterates through each row in `metrics_df`
2. It uses the 'clusters' column value as the key for the mapping
3. It also registers an alias without the leading "0+" if the key starts with "0+"

This ensures that the method creates the correct mapping, which is used by other methods in the module.

## Expected Impact

### 1. `_load_cluster_data()` Method

The modified method will now correctly extract individual clusters from the 'clusters' column in the metrics file, without checking for the 'cluster' column first. This will eliminate the warning message about 'cluster' column not found and ensure that the method always uses the correct column.

### 2. `_create_cluster_metrics_mapping()` Method

The modified method will now correctly create a mapping from cluster combinations to their metrics, following the same approach as the original module. This will eliminate the error message about cluster metrics mapping and ensure that the method creates the correct mapping, which is used by other methods in the module.

## Testing

A test script `test_cluster_data_handling.py` has been created to verify the changes. This script:

1. Tests the `_load_cluster_data()` method to ensure it correctly loads cluster data using the 'clusters' column
2. Tests the `_create_cluster_metrics_mapping()` method to ensure it correctly creates a mapping from cluster combinations to their metrics

To run the test script:

```
python -m test_cluster_data_handling
```

The script will print information about the loaded clusters and the created metrics mapping, which can be used to verify that the changes are working correctly.

## Conclusion

The changes made to the cluster data handling in `dynamic_dtn_optimization_part2.py` ensure that the module correctly processes cluster data from the updated metrics file. This will eliminate the errors and warnings related to cluster data handling and ensure that the module works correctly with the updated metrics file format.