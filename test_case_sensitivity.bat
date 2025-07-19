@echo off
echo Testing Dynamic DTN Rerun Optimization with case sensitivity fixes...
echo.
echo Command: python D:/changf/CityEnergyAnalyst/cea/interfaces/cli/cli.py dynamic-dtn-rerun-optimization --scenario "C:\Users\changf\OneDrive - ETH Zurich\CEA_projects\base_design\01_base_design_2025" --network-type DH --testing-clusters 1,2,3,4,7
echo.
echo Running command...
C:\Users\changf\micromamba\envs\cea\python.exe D:/changf/CityEnergyAnalyst/cea/interfaces/cli/cli.py dynamic-dtn-rerun-optimization --scenario "C:\Users\changf\OneDrive - ETH Zurich\CEA_projects\base_design\01_base_design_2025" --network-type DH --testing-clusters 1,2,3,4,7
echo.
echo Test completed.
pause