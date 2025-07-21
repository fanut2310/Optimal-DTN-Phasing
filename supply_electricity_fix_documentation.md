# Supply Electricity File Fix for Dynamic DTN Optimization Part 2

## Issue Description

The Dynamic DTN Rerun Optimization Part 2 script was failing with the following error:

```
Error during Dynamic DTN Rerun Optimization Part 2: [Errno 2] No such file or directory: 'C:\\Users\\changf\\OneDrive - ETH Zurich\\CEA_projects\\base_design\\01_base_design_2025\\outputs\\data\\optimization\\dynamic_dtn_optimization\\temp_scenario\\inputs\\database\\ASSEMBLIES\\SUPPLY\\SUPPLY_ELECTRICITY.csv'
```

This error occurred in the `calculate_emissions_for_genome` method when it called `lca_operation`, which tried to access the `SUPPLY_ELECTRICITY.csv` file. The file was missing because it wasn't being copied to the temporary scenario folder.

## Root Cause Analysis

The error occurred because the `_copy_database_files` method in `dynamic_dtn_optimization_part2.py` was copying several database files to the temporary scenario folder, but it wasn't copying the `SUPPLY_ELECTRICITY.csv` file:

```python
# Copy the supply system files
for file_name in ["SUPPLY_HEATING.csv", "SUPPLY_COOLING.csv", "SUPPLY_HOTWATER.csv"]:
    src_file = Path(self.locator.get_db4_assemblies_supply_folder()) / file_name
    dst_file = temp_supply_dir / file_name
    
    if src_file.exists():
        log(f"Copying {file_name} to temp scenario")
        shutil.copy2(src_file, dst_file)
    else:
        log(f"Warning: {file_name} not found at {src_file}")
```

The `lca_operation` function in `cea/analysis/lca/operation.py` requires the `SUPPLY_ELECTRICITY.csv` file to calculate emissions:

```python
factors_electricity = pd.read_csv(locator.get_database_assemblies_supply_electricity())
```

## Solution

The solution was to add `SUPPLY_ELECTRICITY.csv` to the list of supply system files that are copied:

```python
# Copy the supply system files
for file_name in ["SUPPLY_HEATING.csv", "SUPPLY_COOLING.csv", "SUPPLY_HOTWATER.csv", "SUPPLY_ELECTRICITY.csv"]:
    src_file = Path(self.locator.get_db4_assemblies_supply_folder()) / file_name
    dst_file = temp_supply_dir / file_name
    
    if src_file.exists():
        log(f"Copying {file_name} to temp scenario")
        shutil.copy2(src_file, dst_file)
    else:
        log(f"Warning: {file_name} not found at {src_file}")
```

## Why This Works

1. The `lca_operation` function requires the `SUPPLY_ELECTRICITY.csv` file to calculate emissions.
2. By copying this file to the temporary scenario folder, we ensure that all required files are available when the emissions calculation is performed.
3. This approach maintains the same structure as the original code, just adding one more file to the list of files to copy.

## Alternative Approaches Considered

1. **Copy All Files**: We could have modified the method to copy all files from the supply folder:

```python
# Copy all files from the supply folder
supply_folder = Path(self.locator.get_db4_assemblies_supply_folder())
for file_path in supply_folder.glob("*.csv"):
    dst_file = temp_supply_dir / file_path.name
    log(f"Copying {file_path.name} to temp scenario")
    shutil.copy2(file_path, dst_file)
```

This would ensure that all files are copied, but it might copy unnecessary files and could be less efficient.

2. **Add Error Handling**: We could have added error handling to the `calculate_emissions_for_genome` method to gracefully handle missing files:

```python
try:
    lca_operation(self.temp_locator, custom_supply_path=str(phase_supply_path))
except FileNotFoundError as e:
    log(f"Warning: {e}. Emissions calculation will be skipped.")
    # Set default values for emissions
    phase_emissions[phase]['operation'] = 0
```

This would prevent the script from crashing, but it wouldn't solve the underlying issue of missing files.

## Conclusion

The implemented solution addresses the immediate issue by ensuring that the `SUPPLY_ELECTRICITY.csv` file is copied to the temporary scenario folder. This allows the emissions calculation to proceed without errors, enabling the Dynamic DTN Rerun Optimization Part 2 module to complete successfully.