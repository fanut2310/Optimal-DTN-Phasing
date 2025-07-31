# Dynamic DTN Optimization Part 2 Energy Price Fix

## Issue Description

When running the `dynamic_dtn_optimization_part2.py` script, the following error occurred:

```
15:16:40 |  INFO | Looking for energy price in: C:\Users\changf\OneDrive - ETH Zurich\CEA_projects\base_design\01_base_design_2025\inputs\technology\components\CONVERSION.xlsx
15:16:40 |  INFO | Trying to read ENERGY_PRICE worksheet
15:16:40 | WARNING | Error reading ENERGY_PRICE worksheet: Worksheet named 'ENERGY_PRICE' not found
15:16:40 |  INFO | Trying to read FEEDSTOCKS worksheet instead
15:16:40 | ERROR | Error reading FEEDSTOCKS worksheet: Worksheet named 'FEEDSTOCKS' not found
15:16:40 | ERROR | Could not find energy price in database. Please check that the database contains either an ENERGY_PRICE or FEEDSTOCKS worksheet with the required columns.
```

The issue was that the `_get_energy_price` method in `dynamic_dtn_optimization_part2.py` was trying to read energy price information from Excel files that don't exist, and it had fallback mechanisms that needed to be removed.

## Root Cause

The `_get_energy_price` method in `dynamic_dtn_optimization_part2.py` was:
1. Using `locator.get_database_conversion_systems()` to get the path to a CONVERSION.xlsx file
2. Trying to read an ENERGY_PRICE worksheet from this file
3. If that failed, trying to read a FEEDSTOCKS worksheet from the same file
4. If both attempts failed, raising an error

However, these files and worksheets don't exist in the project structure. The original DTN expansion optimization script (`DTN_expansion_optimization.py`) uses a different approach to get energy prices:
1. It determines which feedstock to use based on network type (NATURALGAS for DH, GRID for DC)
2. It uses `locator.get_db4_components_feedstocks_feedstocks_csv()` to get the path to the feedstock CSV file
3. It reads the CSV file and extracts the energy price from the `Opex_var_buy_USD2015kWh` column
4. It falls back to default values if the file or column doesn't exist

## Solution

The solution was to modify the `_get_energy_price` method in `dynamic_dtn_optimization_part2.py` to:
1. Use the same approach as the original DTN expansion optimization script
2. Remove the fallback mechanisms
3. Show errors in the log and stop execution if files are not found

## Changes Made

The `_get_energy_price` method in `dynamic_dtn_optimization_part2.py` was replaced with the following implementation:

```python
def _get_energy_price(self):
    """
    Read energy price from FEEDSTOCKS.xlsx based on network type.

    For DH: Uses the buy price from appropriate heating feedstock (default: NATURALGAS)
    For DC: Uses the buy price from GRID (electricity for cooling)

    Returns:
    --------
    float
        Energy price in USD/kWh
    
    Raises:
    -------
    ValueError
        If the energy price cannot be read from the feedstock data
    """
    # Determine which feedstock to use based on network type
    if self.network_type == 'DH':
        # For district heating, use NATURALGAS
        feedstock_name = 'NATURALGAS'
    else:
        # For district cooling, use GRID (electricity)
        feedstock_name = 'GRID'

    # Get the feedstock file path
    try:
        feedstock_file = self.locator.get_db4_components_feedstocks_feedstocks_csv(feedstocks=feedstock_name)
        log().info(f"Reading energy price from {feedstock_file}")
    except Exception as e:
        log().error(f"Could not find feedstock file for {feedstock_name}: {e}")
        raise ValueError(f"Could not find feedstock file for {feedstock_name}: {e}")

    # Read the feedstock data
    try:
        feedstock_data = pd.read_csv(feedstock_file)
    except Exception as e:
        log().error(f"Could not read feedstock data from {feedstock_file}: {e}")
        raise ValueError(f"Could not read feedstock data from {feedstock_file}: {e}")

    # Get the buy price column (Opex_var_buy_USD2015kWh)
    if 'Opex_var_buy_USD2015kWh' in feedstock_data.columns:
        # Calculate average price across all hours
        energy_price = feedstock_data['Opex_var_buy_USD2015kWh'].mean()
        log().info(f"Using energy price from {feedstock_name}: {energy_price:.4f} USD/kWh")
        return energy_price
    else:
        log().error(f"Column 'Opex_var_buy_USD2015kWh' not found in {feedstock_name} data.")
        raise ValueError(f"Column 'Opex_var_buy_USD2015kWh' not found in {feedstock_name} data.")
```

Key changes:
1. Changed the method to use `locator.get_db4_components_feedstocks_feedstocks_csv()` instead of `locator.get_database_conversion_systems()`
2. Removed the fallback to default values if the file or column doesn't exist
3. Added proper error handling to show errors in the log and stop execution if files are not found
4. Updated the docstring to match the original DTN expansion optimization script

## Expected Behavior

With these changes, the `_get_energy_price` method in `dynamic_dtn_optimization_part2.py` will:
1. Try to read the energy price from the appropriate feedstock CSV file
2. If the file or column doesn't exist, it will log an error and raise an exception
3. The script will stop execution if the energy price cannot be determined

This behavior aligns with the requirements in the issue description:
- The method now uses the same approach as the original DTN expansion optimization script
- The fallback mechanisms have been removed
- Errors are shown in the log and execution stops if files are not found