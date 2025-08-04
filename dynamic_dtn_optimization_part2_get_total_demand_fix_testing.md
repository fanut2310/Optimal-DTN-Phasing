# Dynamic DTN Optimization Part 2 get_total_demand() Fix Testing

This document outlines the steps to test the fix for the emissions calculation issue in the dynamic DTN optimization part 2.

## Testing Steps

1. **Setup a Test Scenario**:
   - Choose a scenario with multiple buildings and clusters.
   - Ensure that the original DTN optimization has been run on this scenario.
   - Record the original district operation emissions and emission per GFA values for each phase.

2. **Modify Demand for a Cluster**:
   - Run the dynamic DTN optimization part 1 to modify the demand for a specific cluster (e.g., cluster 1).
   - Verify that the modified demand files are created in the temporary scenario folder.
   - Check that the Total_demand.csv file in the temporary scenario reflects the modified demand.

3. **Run Dynamic DTN Optimization Part 2**:
   - Run the dynamic DTN optimization part 2 with the fixed code.
   - This should use the modified demand data from the temporary scenario for the emissions calculations.

4. **Verify Results**:
   - Check the district operation emissions and emission per GFA values for each phase in the results of the dynamic DTN optimization part 2.
   - Compare these values with the original values recorded in step 1.
   - Verify that the values differ, reflecting the impact of the modified demand.

5. **Check Intermediate Files**:
   - Examine the phase-specific supply files created by the dynamic DTN optimization part 2.
   - Verify that the LCA operation results reflect the modified demand.

6. **Test with Different Clusters**:
   - Repeat the test with different clusters to ensure the fix works consistently.
   - Test with clusters that are connected in different phases to ensure the emissions calculations are correct for all phases.

## Expected Results

After implementing the fix, the following results are expected:

1. The district operation emissions and emission per GFA values should differ between the original DTN optimization and the dynamic DTN optimization part 2.
2. The difference should be consistent with the modifications made to the demand data.
3. The emissions calculations should correctly reflect the changes in demand for the modified cluster.
4. All parts of the dynamic DTN optimization process should consistently use the modified demand data, including the emissions calculations.

## Troubleshooting

If the test fails, check the following:

1. Verify that the `get_total_demand()` method in the `TempScenarioLocator` class has been correctly modified to include the `format` parameter.
2. Check that the Total_demand.csv file in the temporary scenario contains the modified demand data.
3. Examine the logs to see if there are any errors related to file paths or missing files.
4. Verify that the `lca_operation` function is being called with the correct parameters in the `calculate_district_emissions_new` method.

## Conclusion

This testing procedure will verify that the fix for the emissions calculation issue in the dynamic DTN optimization part 2 is working correctly. If all tests pass, the fix can be considered successful.