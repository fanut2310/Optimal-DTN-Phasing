"""
Optimization for Multi-phased District Thermal Network Expansion.
"""

__author__ = "Fan Ut Chang"
__copyright__ = "Copyright 2025, Architecture and Building Systems - ETH Zurich"
__credits__ = ["Fan Ut Chang"]
__license__ = "MIT"
__version__ = "0.1"
__maintainer__ = "Fan Ut Chang"
__email__ = "changf@ethz.ch"
__status__ = "Production"

import os
import cea.config
import cea.inputlocator
from cea.technologies.network_layout.main import layout_network, NetworkLayout
from cea.technologies.thermal_network.thermal_network import ThermalNetwork, thermal_network_main


def main(config):
    locator = cea.inputlocator.InputLocator(config.scenario)

    # Get network type from config (DH/DC)
    network_type = config.thermal_network.network_type  # Directly from thermal_network section

    # 1. Initialize NetworkLayout with config section
    network_layout = NetworkLayout(config.network_layout)
    network_layout.network_type = network_type  # Set from thermal_network section

    # 2. Run layout and simulation
    print("Running network layout...")
    layout_network(
        network_layout=network_layout,
        locator=locator,
        output_name_network="",
    )

    # Define thermal_network before using it
    network_name = ""
    thermal_network = ThermalNetwork(locator, network_name, config.thermal_network)
    thermal_network_main(locator, thermal_network)

    print("Thermal network parts 1 & 2 completed successfully.")



if __name__ == '__main__':
    import os
    from cea.config import Configuration
    config = Configuration()
    # Optional: allow overriding scenario via env var
    scenario_path = os.environ.get(
        'CEA_SCENARIO_PATH',
        r"C:\Users\User\OneDrive - ETH Zurich\CEA_projects\base_design\01_base_design_2025"
    )
    config.scenario = scenario_path
    # Ensure the thermal-network section exists
    if 'thermal-network' in config.sections:
        main(config)
    else:
        print("Error: 'thermal-network' section not found in configuration.")
        print("Available sections:", list(config.sections.keys()))
