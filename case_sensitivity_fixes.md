# Case Sensitivity Fixes for Dynamic DTN Rerun Optimization

## Problem Summary

The `dynamic_dtn_rerun_optimization.py` script was experiencing issues with case sensitivity when handling building demand files. Specifically:

1. The script was comparing file paths with case sensitivity
2. Windows file systems are case-insensitive, so "b0018.csv" and "B0018.csv" refer to the same file
3. However, Python string comparison is case-sensitive, causing the verification to fail with error messages like:
   ```
   09:29:27 | ERROR | Modified demand file not being used for B0018: C:\Users\changf\OneDrive - ETH Zurich\CEA_projects\base_design\01_base_design_2025\outputs\data\demand\b0018.csv
   ```

## Changes Made

### Modified the Verification Code in _generate_updated_metrics()

Updated the file path comparison to use `os.path.normcase()` for case-insensitive comparison:

```python
# Before
if test_path != files['modified']:
    self.lg.error(f"Modified demand file not being used for {display_name}: {test_path}")
    raise RuntimeError(f"Modified demand file not being used for {display_name}")

# After
if os.path.normcase(test_path) != os.path.normcase(files['modified']):
    self.lg.error(f"Modified demand file not being used for {display_name}: {test_path}")
    raise RuntimeError(f"Modified demand file not being used for {display_name}")
```

The `os.path.normcase()` function normalizes the case of a pathname according to the platform's rules. On Windows, it converts all characters to lowercase, making the comparison case-insensitive.

## Expected Impact

These changes ensure that:

1. **Case-Insensitive Comparison**: File paths are compared in a case-insensitive manner, which is appropriate for Windows file systems
2. **Correct Verification**: The verification process will correctly identify when modified demand files are being used, regardless of case differences
3. **Improved Error Messages**: If there are still issues with modified demand files not being used, the error messages will be more accurate

## Testing

A test batch file (`test_case_sensitivity.bat`) has been created to verify that the changes work as expected. To run the test:

1. Double-click the `test_case_sensitivity.bat` file in the CityEnergyAnalyst directory
2. The script will execute the dynamic-dtn-rerun-optimization command with the appropriate parameters
3. Monitor the console output for any errors related to case sensitivity
4. Verify that the script successfully completes without the previous error message about modified demand files not being used

## Additional Notes

- The changes maintain backward compatibility with existing code
- The fix is platform-aware, as `os.path.normcase()` behaves differently on different operating systems (e.g., it's a no-op on case-sensitive file systems like Linux)
- This approach is more robust than simply converting all paths to lowercase, as it respects the platform's file system rules