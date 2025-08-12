Short answer: you can set representative week = False to get unblocked quickly, but it’s not necessary once a small bug is fixed.

What’s happening
- The error is caused by an exhausted iterator in prepare_inputs_of_representative_weeks. Converting the chained ranges to a concrete list fixes it.

Your options
- Fast workaround (no code changes): set use_representative_week_per_month = False in your thermal network config and rerun. This will simulate all 8760 hours (slower but robust).
- Proper fix (keeps the 2016-hour speed-up): apply the tiny patch below.

Recommended minimal patch
```python
# Python
def prepare_inputs_of_representative_weeks(thermal_network):
    # Materialize the chained ranges once; iterators get exhausted if reused
    hours_idx = list(chain(
        range(0, 168), range(744, 912), range(1416, 1584), range(2160, 2328), range(2880, 3048),
        range(3624, 3792), range(4344, 4512), range(5088, 5256), range(5832, 6000), range(6522, 6690),
        range(7296, 7464), range(8016, 8184)
    ))

    # Cut out relevant parts of all arrays/dataframes
    thermal_network.T_ground_K = [
        value for index, value in enumerate(thermal_network.T_ground_K) if index in hours_idx
    ]
    for building in thermal_network.buildings_demands.keys():
        df = thermal_network.buildings_demands[building].iloc[hours_idx]
        df.index = range(0, 2016)
        thermal_network.buildings_demands[building] = df

    thermal_network.t_target_supply_C = thermal_network.t_target_supply_C.iloc[hours_idx]
    thermal_network.t_target_supply_C.index = range(0, 2016)

    thermal_network.t_target_supply_df = thermal_network.t_target_supply_df.iloc[hours_idx]
    thermal_network.t_target_supply_df.index = range(0, 2016)

    return np.nan
```


Which should I try first?
- If you don’t want to touch “TN part 2” right now, set use_representative_week_per_month = False and rerun to confirm everything else is fine.
- Then, when convenient, apply the patch to re-enable the representative weeks mode without errors.

Extra sanity checks (if issues persist after patch/workaround)
- Verify each building demand file has 8760 hourly rows.
- Ensure buildings_demands[building] isn’t empty before slicing (it should not be if Total_demand and per-building CSVs are consistent).

If you want, I can suggest the exact place to toggle the config value in your run setup.