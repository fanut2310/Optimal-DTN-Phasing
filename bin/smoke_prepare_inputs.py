import sys
import os

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    import pandas as pd
    import numpy as np
    from cea.technologies.thermal_network.thermal_network import prepare_inputs_of_representative_weeks
except Exception as e:
    print(f"SKIP: Could not import dependencies or target function: {e}")
    sys.exit(0)


class MockThermalNetwork:
    def __init__(self):
        # 8760 hours worth of dummy data
        self.T_ground_K = list(range(8760))
        self.buildings_demands = {
            'B01': pd.DataFrame({'Q_kW': np.arange(8760)})
        }
        self.t_target_supply_C = pd.Series(np.arange(8760), name='Ts_C')
        self.t_target_supply_df = pd.DataFrame({'Ts_C': np.arange(8760)})


def main():
    tn = MockThermalNetwork()
    prepare_inputs_of_representative_weeks(tn)

    # Assertions
    assert len(tn.T_ground_K) == 2016, f"Expected 2016 T_ground_K, got {len(tn.T_ground_K)}"
    assert 'B01' in tn.buildings_demands
    bd = tn.buildings_demands['B01']
    assert bd.shape[0] == 2016, f"Expected 2016 building demand rows, got {bd.shape[0]}"
    assert bd.index.start == 0 and bd.index.stop == 2016, f"Unexpected index range: {bd.index}"
    assert len(tn.t_target_supply_C) == 2016, f"Expected 2016 t_target_supply_C, got {len(tn.t_target_supply_C)}"
    assert tn.t_target_supply_C.index.start == 0 and tn.t_target_supply_C.index.stop == 2016
    assert tn.t_target_supply_df.shape[0] == 2016
    assert tn.t_target_supply_df.index.start == 0 and tn.t_target_supply_df.index.stop == 2016

    print("Smoke test passed: prepare_inputs_of_representative_weeks produces 2016-row timeseries and correct indices.")


if __name__ == '__main__':
    main()
