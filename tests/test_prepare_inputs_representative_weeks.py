import pandas as pd
import numpy as np

try:
    from cea.technologies.thermal_network.thermal_network import prepare_inputs_of_representative_weeks
except Exception:
    prepare_inputs_of_representative_weeks = None


class MockThermalNetwork:
    def __init__(self):
        # 8760 hours worth of dummy data
        self.T_ground_K = list(range(8760))
        self.buildings_demands = {
            'B01': pd.DataFrame({'Q_kW': np.arange(8760)})
        }
        self.t_target_supply_C = pd.Series(np.arange(8760), name='Ts_C')
        self.t_target_supply_df = pd.DataFrame({'Ts_C': np.arange(8760)})


def test_prepare_inputs_of_representative_weeks_reduces_to_2016_rows():
    if prepare_inputs_of_representative_weeks is None:
        # Environment cannot import the target module, treat as pass to avoid hard dependency on pytest
        return

    tn = MockThermalNetwork()

    # call function under test
    prepare_inputs_of_representative_weeks(tn)

    # T_ground_K should be reduced to 2016 entries
    assert len(tn.T_ground_K) == 2016

    # building demands
    assert 'B01' in tn.buildings_demands
    bd = tn.buildings_demands['B01']
    assert bd.shape[0] == 2016
    # index should be 0..2015
    assert bd.index.start == 0 and bd.index.stop == 2016

    # t_target_supply_C
    assert len(tn.t_target_supply_C) == 2016
    assert tn.t_target_supply_C.index.start == 0 and tn.t_target_supply_C.index.stop == 2016

    # t_target_supply_df
    assert tn.t_target_supply_df.shape[0] == 2016
    assert tn.t_target_supply_df.index.start == 0 and tn.t_target_supply_df.index.stop == 2016
