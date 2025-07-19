# Path Normalization Fix for Dynamic DTN Rerun Optimization

## Issue Description

The dynamic-dtn-rerun-optimization module was failing with the error:
```
Modified demand file not being used for B0018: C:\Users\changf\OneDrive - ETH Zurich\CEA_projects\base_design\01_base_design_2025\outputs\data\demand\B0018.csv
```

This occurred because the paths to the modified demand files were not being normalized consistently throughout the code, causing string comparisons to fail even when the paths referred to the same file.

## Root Cause

The issue was not related to case sensitivity (which was fixed previously) but to path normalization:

1. **Inconsistent path formats**: Paths were stored and returned in different formats (e.g., with different path separators or one being relative while the other was absolute).

2. **Direct string comparison**: The verification code was comparing paths directly as strings without normalizing them first.

3. **No debug information**: There was limited logging to help diagnose the exact nature of the path differences.

## Changes Made

### 1. Updated `ModifiedDemandsLocator.get_demand_results_file` in `dynamic_dtn_optimization.py`

```python
def get_demand_results_file(self, building, format='csv'):
    """
    Override to return the path to the modified demand file if available.
    """
    key = building
    if key in self.modified_demand_files:
        modified_path = self.modified_demand_files[key]['modified']
        # Normalize the path for consistent comparison
        modified_path = os.path.normpath(modified_path)
        # Verify the file exists
        if not os.path.exists(modified_path):
            raise FileNotFoundError(f"Modified demand file not found: {modified_path}")
        # Add debug print
        print(f"DEBUG: Using modified demand file for {building}: {modified_path}")
        return modified_path
    else:
        # Pass the original building name to maintain case consistency
        original_path = self.original_locator.get_demand_results_file(building, format)
        # Normalize the path for consistent comparison
        original_path = os.path.normpath(original_path)
        print(f"DEBUG: Using original demand file for {building}: {original_path}")
        return original_path
```

### 2. Updated verification section in `dynamic_dtn_rerun_optimization.py`

```python
# Verify that modified files are being used
self.lg.info("Verifying modified demand files are being used...")
for building, files in mod_files.items():
    # Add debug prints
    self.lg.info(f"Building: {building}, Modified path: {files['modified']}")
    test_path = mod_loc.get_demand_results_file(building)
    self.lg.info(f"Test path: {test_path}")
    
    # Normalize paths before comparison
    norm_test = os.path.normpath(test_path)
    norm_modified = os.path.normpath(files['modified'])
    
    if norm_test != norm_modified:
        self.lg.error(f"Modified demand file not being used for {building}: {test_path}")
        self.lg.info(f"Normalized paths - Test: {norm_test}, Modified: {norm_modified}")
        
        # Try absolute paths as a last resort
        abs_test = os.path.abspath(test_path)
        abs_modified = os.path.abspath(files['modified'])
        self.lg.info(f"Absolute paths - Test: {abs_test}, Modified: {abs_modified}")
        
        if abs_test == abs_modified:
            self.lg.info("Paths are equal after converting to absolute paths, continuing...")
        else:
            raise RuntimeError(f"Modified demand file not being used for {building}")
    else:
        self.lg.info(f"✓ Using modified demand file for {building}: {test_path}")
```

### 3. Updated `_modified_demands` method in `dynamic_dtn_rerun_optimization.py`

```python
def _modified_demands(self) -> dict[str, dict]:
    mod_dir = self.dyn_folder / "modified_demands"
    if not mod_dir.exists():
        raise FileNotFoundError(f"Modified demand directory not found: {mod_dir}")

    # Store the original file stem to use in the modified path
    files = {}
    for f in mod_dir.glob("*.csv"):
        key = f.stem  # Maintain original case for consistent file matching
        original_file = self.locator.get_demand_results_file(key)
        
        # Normalize paths to ensure consistent comparison
        original_path = os.path.normpath(original_file)
        modified_path = os.path.normpath(str(f))
        
        files[key] = {'original': original_path, 'modified': modified_path}
        self.lg.info(f"Added modified file mapping: {key} -> {modified_path}")

    if not files:
        raise FileNotFoundError("No modified demand files found.")
    self.lg.info(f"Found {len(files)} modified demand files")
    return files
```

## Benefits of the Changes

1. **Consistent path normalization**: All paths are now normalized using `os.path.normpath()` before being stored or compared.

2. **Enhanced debugging**: Added detailed logging to show the exact paths being compared, which helps diagnose any remaining issues.

3. **Fallback to absolute paths**: As a last resort, the code now tries comparing absolute paths if normalized paths don't match.

4. **Better error messages**: The error messages now include more information about the paths being compared.

## How to Test

Run the dynamic-dtn-rerun-optimization command with the same parameters as before:

```
C:\Users\changf\micromamba\envs\cea\python.exe D:\changf\CityEnergyAnalyst\cea\interfaces\cli\cli.py dynamic-dtn-rerun-optimization --scenario "C:\Users\changf\OneDrive - ETH Zurich\CEA_projects\base_design\01_base_design_2025" --network-type DH --testing-clusters 1,2,3,4,7
```

The command should now run without the "Modified demand file not being used" error.