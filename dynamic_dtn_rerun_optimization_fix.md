# Dynamic DTN Rerun Optimization Fix

## Issue Description

The Dynamic DTN Rerun Optimization module was failing with the following error:

```
Error generating updated metrics: [Errno 2] No such file or directory: 'C:\\Users\\changf\\OneDrive - ETH Zurich\\CEA_projects\\base_design\\01_base_design_2025\\outputs\\data\\optimization\\dynamic_dtn_optimization\\temp_scenario\\outputs\\data\\optimization\\dtn_expansion\\cluster_edges.csv'
```

This error occurred because the `PipeLayoutGenerator` class was trying to load cluster assignment files from the temp scenario folder, but these files didn't exist there.

## Root Cause Analysis

The issue was related to how the `PipeLayoutGenerator` class loads input data:

1. The `PipeLayoutGenerator._load_inputs()` method tries to load cluster assignment files (`cluster_edges.csv` and `cluster_nodes.csv`) from the path returned by `self.locator.get_dtn_expansion_optimization_results_folder()`.

2. When using the `TempScenarioLocator`, this path points to the temp scenario folder, but the required files are actually in the original scenario's optimization folder.

3. The `dynamic_dtn_optimization.py` script only copies demand files and thermal network files to the temp scenario folder, but not the cluster assignment files.

## Solution Implemented

The solution was to modify the `_generate_updated_metrics` method in `dynamic_dtn_rerun_optimization.py` to copy the required cluster files from the original scenario to the temp scenario folder before creating the `PipeLayoutGenerator`. This ensures that all necessary files are available in the temp scenario folder.

### Changes Made

1. Added code to create the necessary directory structure in the temp scenario folder:
   ```python
   # Step 2: Create necessary directories in temp scenario for optimization files
   temp_opt_dir = temp_scenario_path / "outputs" / "data" / "optimization" / "dtn_expansion"
   temp_opt_dir.mkdir(parents=True, exist_ok=True)
   self.lg.info(f"Created directory for optimization files in temp scenario: {temp_opt_dir}")
   ```

2. Added code to copy the required cluster files from the original scenario to the temp scenario folder:
   ```python
   # Step 3: Copy required cluster files from original scenario to temp scenario
   original_opt_dir = Path(self.locator.get_dtn_expansion_optimization_results_folder())
   
   # Copy cluster_edges.csv
   cluster_edges_path = original_opt_dir / "cluster_edges.csv"
   if os.path.exists(cluster_edges_path):
       self.lg.info(f"Copying cluster_edges.csv to temp scenario")
       import shutil
       shutil.copy2(cluster_edges_path, temp_opt_dir / "cluster_edges.csv")
   else:
       raise FileNotFoundError(f"cluster_edges.csv not found at {cluster_edges_path}")
   
   # Copy cluster_nodes.csv
   cluster_nodes_path = original_opt_dir / "cluster_nodes.csv"
   if os.path.exists(cluster_nodes_path):
       self.lg.info(f"Copying cluster_nodes.csv to temp scenario")
       shutil.copy2(cluster_nodes_path, temp_opt_dir / "cluster_nodes.csv")
   else:
       raise FileNotFoundError(f"cluster_nodes.csv not found at {cluster_nodes_path}")
   ```

3. Updated the step numbering in the comments to reflect the new steps.

## How to Test

To test the solution, run the dynamic-dtn-rerun-optimization command with the same parameters:

```
C:\Users\changf\micromamba\envs\cea\python.exe D:\changf\CityEnergyAnalyst\cea\interfaces\cli\cli.py dynamic-dtn-rerun-optimization --scenario "C:\Users\changf\OneDrive - ETH Zurich\CEA_projects\base_design\01_base_design_2025" --network-type DH --testing-clusters 1,2,3,4,7
```

### Expected Behavior

1. The script should run without the "No such file or directory" error for cluster_edges.csv
2. The log should show messages like:
   - "Created directory for optimization files in temp scenario: [path]"
   - "Copying cluster_edges.csv to temp scenario"
   - "Copying cluster_nodes.csv to temp scenario"
3. The optimization should complete successfully
4. The updated metrics should be generated and saved to the expected location

## Troubleshooting

If you encounter any issues:

1. **Check if the cluster files exist in the original scenario**:
   - Verify that `cluster_edges.csv` and `cluster_nodes.csv` exist in the original scenario's optimization folder
   - If they don't exist, you may need to run the building-clustering module first

2. **Check file permissions**:
   - Ensure the script has write access to the temp scenario folder
   - Ensure the script has read access to the original scenario's optimization folder

3. **Check for import errors**:
   - If you see an error related to the `shutil` module, make sure it's properly imported
   - Add `import shutil` at the top of the file if needed

4. **Check for path issues**:
   - If you see path-related errors, check that the paths are correctly constructed
   - Use `os.path.exists()` to verify that files and directories exist before trying to access them

## Alternative Approach

If the current solution doesn't work, an alternative approach would be to modify the `TempScenarioLocator` class to redirect requests for specific files to the original scenario:

```python
class TempScenarioLocator(cea.inputlocator.InputLocator):
    """A simple InputLocator that points directly to the temp scenario folder."""

    def __init__(self, original_locator, temp_scenario_path):
        # Initialize with the temp scenario path
        super().__init__(str(temp_scenario_path))
        
        # Store the original locator for redirecting specific requests
        self.original_locator = original_locator
        
        # Copy attributes from original locator that might be needed
        self.__dict__.update({k: v for k, v in original_locator.__dict__.items() 
                             if k not in ['scenario', '_scenario', '_temp_directory']})
        
        # Clear any cache
        self._demand_cache = {}
        
        lg = log()
        lg.info(f"Created TempScenarioLocator pointing to: {self.scenario}")
    
    def get_dtn_expansion_optimization_results_folder(self):
        """Override to use the original locator for cluster files."""
        return self.original_locator.get_dtn_expansion_optimization_results_folder()
```

This approach would avoid the need to copy files, but it might be less explicit and could lead to confusion about which files are being used.

## Conclusion

The implemented solution ensures that all necessary files are available in the temp scenario folder, allowing the `PipeLayoutGenerator` to work correctly with the modified demand files. This approach is explicit, easy to understand, and avoids potential issues with path handling in the `InputLocator` class.