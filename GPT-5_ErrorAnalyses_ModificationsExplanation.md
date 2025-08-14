Here’s a direct “symptom → fix” mapping for the five functions you edited, tied to the run settings you actually used.

Run settings that matter for these fixes
- Network type: DC
- Model: detailed (not simplified)
- Temperature control: VT, with automatic fallback to CT in solver internals
- Representative weeks: toggled (True for faster runs, False for full-year)
- Reuse prior sizing: toggled via load_max_edge_flowrate_from_previous_run (True/False)
- Multiprocessing: on/off (irrelevant to these five changes)

1) Error: downstream modules expect 8760 rows, but representative weeks produce 2016 rows
- When it happens:
  - use_representative_week_per_month = True (both during “pipe sizing” and the main thermal-hydraulic run)
  - Symptoms: plots, cost calculation, and map layers crash or produce wrong alignment because they read 8760-hour time series.
- Fixes you added:
  - extrapolate_datapoints_for_representative_weeks(...)
    - Repeats the 2016-hour data 4x and pads any remaining rows with row-wise means to reach 8760.
    - Used in save_all_results_to_csv for all time series and in calc_max_edge_flowrate for nominal node flows; also applied to nominal edge flows in thermal_network_main before writing CSVs.
    - Also replaces deprecated DataFrame.append with pd.concat to avoid pandas errors.
  - thermal_network_main(...)
    - When use_representative_week_per_month is True, it now extrapolates nominal edge flows to 8760 before saving. This ensures the costs and map layers can read consistent 8760-hour inputs even though the solver ran 2016 hours.

2) Error: empty/misaligned frames due to exhausted indices when slicing representative weeks
- When it happens:
  - use_representative_week_per_month = True (slicing first week of each month)
  - Symptoms: after one pass, arrays/dataframes come out empty or misaligned; follow-on code hits NaNs or index errors.
- Fix you added:
  - prepare_inputs_of_representative_weeks(...)
    - Materializes the month-week hour indices as a list once and applies the same slice consistently to T_ground_K, buildings_demands, t_target_supply_C, and t_target_supply_df.
    - Reindexes slices to a clean 0..2015. Prevents iterator exhaustion and alignment drift.

3) Error: legacy CSVs cause shape or column errors when reusing “nominal flows”
- When it happens:
  - load_max_edge_flowrate_from_previous_run = True (reusing a previous run’s CSVs)
  - Symptoms: a stray “Unnamed: 0” column (saved index) appears in nominal_edge_mass_flow.csv or nominal_node_mass_flow.csv and breaks vectorized operations or column alignment.
- Fixes you added:
  - load_max_edge_flowrate_from_previous_run(...)
  - load_node_flowrate_from_previous_run(...)
    - Both now strip “Unnamed: 0” if present before returning the DataFrame. This makes the bypass path robust across older result files.

4) Error: representative weeks run writes only 2016-row nominal edge mass flow (breaking later tools)
- When it happens:
  - use_representative_week_per_month = True and load_max_edge_flowrate_from_previous_run = False
  - Symptoms: the nominal edge mass-flow CSV ends up 2016 rows; cost model and map readers expect 8760 and fail.
- Fix you added:
  - thermal_network_main(...)
    - After calc_max_edge_flowrate, nominal edge flows are extrapolated to 8760 before writing when representative weeks is active. Keeps file contracts intact downstream.

5) Deprecation/compatibility warning: pandas append
- When it appears:
  - Any modern pandas version; previously used append in extrapolation step
- Fix you added:
  - extrapolate_datapoints_for_representative_weeks(...)
    - Uses pd.concat everywhere instead of append, removing deprecation warnings and preventing runtime errors on newer pandas.

Quick guide: which changes kick in under which run setting?
- Representative weeks = True:
  - prepare_inputs_of_representative_weeks is invoked (clean slicing and reindexing).
  - extrapolate_datapoints_for_representative_weeks is used in save_all_results_to_csv and calc_max_edge_flowrate, and thermal_network_main extrapolates nominal edge flows before writing. Result: all outputs are 8760 rows; downstream modules stop failing.
- Representative weeks = False:
  - None of the extrapolation/slicing paths run; no behavioral change (you get native 8760-hour results).
- Reuse prior sizing = True:
  - load_max_edge_flowrate_from_previous_run and load_node_flowrate_from_previous_run remove legacy “Unnamed: 0” safely and return clean DataFrames; avoids subtle shape mismatches when bypassing the heavy sizing step.