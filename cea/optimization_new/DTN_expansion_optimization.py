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
import types
import cea.config
from cea.technologies.network_layout.main import layout_network, NetworkLayout
from cea.technologies.thermal_network.thermal_network import ThermalNetwork, thermal_network_main
import geopandas as gpd
from cea.inputlocator import InputLocator

# read and drop Z‐dimension
zone_gdf = gpd.read_file(zone_path)
zone_gdf['geometry'] = zone_gdf.geometry.apply(lambda geom: geom if geom.has_z is False
                                              else type(geom)([ (x, y) for x, y, *rest in geom.coords ]))
# overwrite with plain 2D
zone_gdf.to_file(zone_path, driver='ESRI Shapefile')

subprocess.check_call([cea_cli, "network-layout", "--scenario", config.scenario], cwd=config.scenario)

def main(config):
    locator = cea.inputlocator.InputLocator(config.scenario)

    # Override certain network layout parameters bypassing the configuration check.
    object.__setattr__(config.network_layout, "connected_buildings", [])
    object.__setattr__(config.network_layout, "consider_only_buildings_with_demand", False)

    # Ensure the thermal_network configuration exists with network_type.
    try:
        network_type = config.thermal_network.network_type
    except AttributeError:
        # Create a minimal dummy thermal network configuration with a default network type.
        thermal_network_config = types.SimpleNamespace()
        thermal_network_config.network_type = "DH"  # default value, change if needed
        object.__setattr__(config, "thermal_network", thermal_network_config)
        network_type = config.thermal_network.network_type

    # Initialize the network layout using the configuration.
    network_layout = NetworkLayout(config.network_layout)
    network_layout.network_type = network_type  # set network type from thermal_network configuration

    print("Running network layout...")
    layout_network(
        network_layout=network_layout,
        locator=locator,
        output_name_network="",
    )

    # Run the thermal network simulation (Parts 1 & 2)
    network_name = ""
    thermal_network = ThermalNetwork(locator, network_name, config.thermal_network)
    thermal_network_main(locator, thermal_network)

    print("Thermal network parts 1 & 2 completed successfully.")


if __name__ == '__main__':
    from cea.config import Configuration

    config = Configuration()

    # Allow scenario overriding via an environment variable.
    scenario_path = os.environ.get(
        'CEA_SCENARIO_PATH',
        r"C:\Users\changf\OneDrive - ETH Zurich\CEA_projects\base_design\01_base_design_2025"
    )
    config.scenario = scenario_path

    # Check for the presence of the 'thermal-network' section; if not present, we later
    # create a dummy configuration on the fly.
    if 'thermal-network' in config.sections:
        main(config)
    else:
        # If not found in the sections, you may either choose to create a dummy
        # configuration as above or show an error.
        print("Warning: 'thermal-network' section not found in configuration.")
        print("A default thermal network configuration will be used.")
        main(config)
