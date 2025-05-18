"""
Thermal network costs calculation for the "Thermal Network Part 2: simulation" module.
This module calculates the costs of a thermal network based on the results of the thermal_network.py simulation.
It is adapted from the original thermal_network_costs.py module to work with the data structures and objects
used in thermal_network.py.
"""

import numpy as np
import pandas as pd
import cea.config
import cea.inputlocator

from cea.optimization.prices import Prices as Prices
from cea.analysis.costs.equations import calc_capex_annualized
from cea.constants import HOURS_IN_YEAR
from cea.technologies.heat_exchangers import calc_Cinv_HEX_hisaka
import cea.technologies.pumps as pumps
import cea.technologies.chiller_vapor_compression as VCCModel
import cea.technologies.cooling_tower as CTModel
from cea.utilities import epwreader

__author__ = "Adapted from thermal_network_costs.py by Lennart Rogenhofer, Shanshan Hsieh"
__copyright__ = "Copyright 2015, Architecture and Building Systems - ETH Zurich"
__credits__ = ["Lennart Rogenhofer"]
__license__ = "MIT"
__version__ = "0.1"
__maintainer__ = "Daren Thomas"
__email__ = "cea@arch.ethz.ch"
__status__ = "Production"


class NetworkCostFeatures(object):
    """
    This class sets up features needed for cost calculations of the thermal network.
    It is a simplified version of the NetworkOptimizationFeatures class.
    """

    def __init__(self, network_type, network_name, locator):
        """
        Initialize the NetworkCostFeatures class.

        :param network_type: 'DH' for district heating or 'DC' for district cooling
        :param network_name: name of the network
        :param locator: InputLocator instance
        """
        self.network_type = network_type
        self.network_name = network_name
        self.locator = locator

        # Initialize attributes that will be calculated
        self.pipesCosts_USD = None
        self.DeltaP = None

        # Calculate pipe costs
        self.calculate_pipe_costs()

        # Calculate pressure loss
        self.calculate_pressure_loss()

    def calculate_pipe_costs(self):
        """
        Calculate the costs of the pipes in the network.
        """
        edges_file = pd.read_csv(self.locator.get_thermal_network_edge_list_file(self.network_type, self.network_name))
        piping_cost_data = pd.read_csv(self.locator.get_database_components_distribution_thermal_grid('THERMAL_GRID'))

        # Standardize column name in files
        merge_df = edges_file.rename(columns={'Pipe_DN': 'pipe_DN'}).merge(piping_cost_data, on='pipe_DN')
        merge_df['Inv_USD2015'] = merge_df['Inv_USD2015perm'] * merge_df['length_m']
        self.pipesCosts_USD = merge_df['Inv_USD2015'].sum()

    def calculate_pressure_loss(self):
        """
        Calculate the maximum pressure loss in the network.
        """
        pressure_loss_df = pd.read_csv(self.locator.get_network_energy_pumping_requirements_file(
            self.network_type, self.network_name))
        self.DeltaP = pressure_loss_df['pressure_loss_total_kW'].max() * 1000  # convert from kW to W


def calc_Capex_a_network_pipes(network_cost_features):
    """
    Calculate the annualized capital expenditure for network pipes.

    :param network_cost_features: NetworkCostFeatures instance
    :return: annualized capital expenditure for network pipes
    """
    InvC = network_cost_features.pipesCosts_USD

    # Assume lifetime of 25 years and 5 % IR
    Inv_IR = 5
    Inv_LT = 25
    Capex_a_netw = calc_capex_annualized(InvC, Inv_IR, Inv_LT)

    return Capex_a_netw


def calc_Ctot_network_pump(network_cost_features, locator):
    """
    Calculate the total pump investment and operational cost.

    :param network_cost_features: NetworkCostFeatures instance
    :param locator: InputLocator instance
    :return: annualized capital expenditure, fixed operational expenditure, and variable operational expenditure
    """
    network_type = network_cost_features.network_type
    network_name = network_cost_features.network_name

    # Read in node mass flows
    df = pd.read_csv(locator.get_nominal_edge_mass_flow_csv_file(network_type, network_name), index_col=0)
    mdotA_kgpers = np.array(df)
    mdotA_kgpers = np.nan_to_num(mdotA_kgpers)
    mdotnMax_kgpers = np.amax(mdotA_kgpers)  # find highest mass flow of all nodes at all timesteps

    # Read in total pressure loss in kW
    deltaP_df = pd.read_csv(locator.get_network_energy_pumping_requirements_file(network_type, network_name))
    deltaP_kW = deltaP_df['pressure_loss_total_kW'].sum()

    # Get electricity price
    from cea.technologies.supply_systems_database import SupplySystemsDatabase
    supply_systems = SupplySystemsDatabase(locator)
    prices = Prices(supply_systems)
    prices.ELEC_PRICE = np.mean(prices.ELEC_PRICE, dtype=np.float64)  # [USD/W]

    Opex_var = deltaP_kW * 1000 * prices.ELEC_PRICE

    # Get maximum pressure loss
    deltaPmax = network_cost_features.DeltaP

    # Get pumping energy and peak load
    peak_pump_power_W = pumps.calc_pump_power(mdotnMax_kgpers, deltaPmax)

    Capex_a_pump_USD, Opex_fixed_pump_USD, _ = pumps.calc_Cinv_pump(peak_pump_power_W, locator, 'PU1')

    return Capex_a_pump_USD, Opex_fixed_pump_USD, Opex_var


def calc_Ctot_cooling_plants(thermal_network, locator):
    """
    Calculate the costs of centralized cooling plants (chillers and cooling towers).

    :param thermal_network: ThermalNetwork instance
    :param locator: InputLocator instance
    :return: fixed operational expenditure, variable operational expenditure, annualized capital expenditure for chiller,
             and annualized capital expenditure for cooling tower
    """
    network_type = thermal_network.network_type
    network_name = thermal_network.network_name

    # Read in plant heat requirement
    plant_heat_hourly_kWh = pd.read_csv(
        locator.get_thermal_network_plant_heat_requirement_file(network_type, network_name))

    # Read in number of plants
    number_of_plants = len(plant_heat_hourly_kWh.columns)

    plant_heat_original_kWh = plant_heat_hourly_kWh.copy()
    plant_heat_peak_kW_list = plant_heat_hourly_kWh.abs().max(axis=0).values  # calculate peak demand
    plant_heat_sum_kWh_list = plant_heat_hourly_kWh.abs().sum().values  # calculate aggregated demand

    Opex_var_plant = 0.0
    Opex_fixed_plant = 0.0
    Capex_a_chiller = 0.0
    Capex_a_CT = 0.0

    # Get electricity price
    from cea.technologies.supply_systems_database import SupplySystemsDatabase
    supply_systems = SupplySystemsDatabase(locator)
    prices = Prices(supply_systems)
    prices.ELEC_PRICE = np.mean(prices.ELEC_PRICE, dtype=np.float64)  # [USD/W]

    # Get weather data
    weather_path = locator.get_weather_file()
    weather_data = epwreader.epw_reader(weather_path)[['year', 'drybulb_C', 'wetbulb_C']]

    # Calculate cost of chiller heat production and chiller capex and opex
    for plant_number in range(number_of_plants):  # iterate through all plants
        if number_of_plants > 1:
            plant_heat_peak_kW = plant_heat_peak_kW_list[plant_number]
        else:
            plant_heat_peak_kW = plant_heat_peak_kW_list[0]
        plant_heat_yearly_kWh = plant_heat_sum_kWh_list[plant_number]
        print('Annual plant heat production:', round(plant_heat_yearly_kWh, 0), '[kWh]')

        Capex_a_chiller_USD = 0.0
        Opex_fixed_chiller = 0.0
        Capex_a_CT_USD = 0.0
        Opex_fixed_CT = 0.0

        if plant_heat_peak_kW > 0:  # we have non 0 demand
            peak_demand_W = plant_heat_peak_kW * 1000  # convert to W
            print('Calculating cost of heat production at plant number: ', (plant_number + 1))

            # For simplicity, we'll use a constant COP for the chiller
            COP_plant = 4.0  # Typical value for a centralized chiller
            COP_chiller = 4.5  # Slightly higher than system COP

            # Calculate cost of producing cooling
            Opex_var_plant += abs(plant_heat_yearly_kWh) / COP_plant * 1000 * prices.ELEC_PRICE

            # Calculate equipment cost of chiller and cooling tower
            Capex_a_chiller_USD, Opex_fixed_chiller, _ = VCCModel.calc_Cinv_VCC(peak_demand_W, locator, 'CH1')
            Capex_a_CT_USD, Opex_fixed_CT, _ = CTModel.calc_Cinv_CT(peak_demand_W, locator, 'CT1')

        # Sum over all plants
        Capex_a_chiller += Capex_a_chiller_USD
        Capex_a_CT += Capex_a_CT_USD
        Opex_fixed_plant += Opex_fixed_chiller + Opex_fixed_CT

    return Opex_fixed_plant, Opex_var_plant, Capex_a_chiller, Capex_a_CT


def calc_Cinv_HEX(thermal_network, locator):
    """
    Calculate the investment cost of the substation heat exchanger.

    :param thermal_network: ThermalNetwork instance
    :param locator: InputLocator instance
    :return: annualized capital expenditure and fixed operational expenditure
    """
    # This is a simplified version that assumes all buildings are connected
    # and uses the maximum heat demand for HEX sizing

    # Get the maximum heat demand for each building
    if thermal_network.substations_HEX_specs is not None:
        Capex_a_hex, Opex_fixed_hex = calc_Cinv_HEX_hisaka(thermal_network)
    else:
        # If HEX specs are not available, use a simplified approach
        Capex_a_hex = 0.0
        Opex_fixed_hex = 0.0

        # Read in node types
        node_types_df = pd.read_csv(
            locator.get_thermal_network_node_types_csv_file(thermal_network.network_type, thermal_network.network_name))

        # Get buildings connected to the network
        buildings = node_types_df.loc[node_types_df['type'] == 'CONSUMER', 'building'].unique()

        for building in buildings:
            if building is not None and building != 'NONE':
                # Read in building demand
                building_demand = pd.read_csv(locator.get_demand_results_file(building))

                # Get maximum heat demand
                if thermal_network.network_type == 'DH':
                    max_demand_kW = building_demand['Qhs_sys_kWh'].abs().max()
                else:  # DC
                    max_demand_kW = building_demand['Qcs_sys_kWh'].abs().max()

                # Calculate HEX cost
                hex_cost = 500 * max_demand_kW  # Simplified cost function

                # Assume lifetime of 20 years and 5 % IR
                Inv_IR = 5
                Inv_LT = 20
                Capex_a_hex += calc_capex_annualized(hex_cost, Inv_IR, Inv_LT)
                Opex_fixed_hex += 0.05 * hex_cost  # Assume 5% of investment cost as annual O&M

    return Capex_a_hex, Opex_fixed_hex


def calc_network_size(thermal_network, locator):
    """
    Calculate the total network length and average pipe diameter.

    :param thermal_network: ThermalNetwork instance
    :param locator: InputLocator instance
    :return: total network length and average pipe diameter
    """
    network_type = thermal_network.network_type
    network_name = thermal_network.network_name

    network_info = pd.read_csv(
        locator.get_thermal_network_edge_list_file(network_type, network_name))
    length_m = network_info['length_m'].sum()
    average_diameter_m = network_info['D_int_m'].mean()

    return float(length_m), float(average_diameter_m)


def calculate_thermal_network_costs(thermal_network, config):
    """
    Calculate the costs of a thermal network.

    :param thermal_network: ThermalNetwork instance
    :param config: Configuration instance
    :return: None
    """
    try:
        print('Starting thermal network cost calculations...')

        # Initialize key variables
        locator = thermal_network.locator
        network_type = thermal_network.network_type
        network_name = thermal_network.network_name

        # Create NetworkCostFeatures instance
        network_cost_features = NetworkCostFeatures(network_type, network_name, locator)

        # Calculate network costs
        # Network pipes
        Capex_a_netw = calc_Capex_a_network_pipes(network_cost_features)

        # Network Pumps
        Capex_a_pump, Opex_fixed_pump, Opex_var_pump = calc_Ctot_network_pump(network_cost_features, locator)

        # Centralized plant
        Opex_fixed_plant, Opex_var_plant, Capex_a_chiller, Capex_a_CT = calc_Ctot_cooling_plants(thermal_network, locator)

        # Heat exchangers
        Capex_a_hex, Opex_fixed_hex = calc_Cinv_HEX(thermal_network, locator)

        # Calculate electricity consumption
        from cea.technologies.supply_systems_database import SupplySystemsDatabase
        supply_systems = SupplySystemsDatabase(locator)
        prices = Prices(supply_systems)
        prices.ELEC_PRICE = np.mean(prices.ELEC_PRICE, dtype=np.float64)  # [USD/W]
        el_price_per_Wh = prices.ELEC_PRICE
        el_MWh = (Opex_var_pump + Opex_var_plant) / el_price_per_Wh / 1e6

        # Calculate total costs
        Capex_a_total = Capex_a_netw + Capex_a_pump + Capex_a_chiller + Capex_a_CT + Capex_a_hex
        Opex_total = Opex_fixed_pump + Opex_var_pump + Opex_var_plant + Opex_fixed_plant + Opex_fixed_hex
        Costs_total = Capex_a_total + Opex_total

        # Calculate network size
        length_m, average_diameter_m = calc_network_size(thermal_network, locator)

        # Calculate annual demands
        total_demand = pd.read_csv(locator.get_total_demand())

        if network_type == 'DC':
            annual_demand_district_MWh = total_demand['Qcs_sys_MWhyr'].sum()
            annual_demand_building_scale_MWh = 0  # Assuming all buildings are connected
            annual_demand_network_MWh = annual_demand_district_MWh - annual_demand_building_scale_MWh
        else:  # DH
            annual_demand_district_MWh = total_demand['Qhs_sys_MWhyr'].sum()
            annual_demand_building_scale_MWh = 0  # Assuming all buildings are connected
            annual_demand_network_MWh = annual_demand_district_MWh - annual_demand_building_scale_MWh

        # Write outputs
        cost_output = {}
        cost_output['total_annual_cost'] = round(Costs_total, 2)
        cost_output['annual_opex'] = round(Opex_total, 2)
        cost_output['annual_capex'] = round(Capex_a_total, 2)
        cost_output['total_cost_per_MWh'] = round(Costs_total / annual_demand_district_MWh, 2)
        cost_output['opex_per_MWh'] = round(Opex_total / annual_demand_district_MWh, 2)
        cost_output['capex_per_MWh'] = round(Capex_a_total / annual_demand_district_MWh, 2)
        cost_output['annual_demand_district_MWh'] = round(annual_demand_district_MWh, 2)
        cost_output['annual_demand_building_scale_MWh'] = round(annual_demand_building_scale_MWh, 2)
        cost_output['annual_demand_network_MWh'] = round(annual_demand_network_MWh, 2)
        cost_output['opex_plant'] = round(Opex_fixed_plant + Opex_var_plant, 2)
        cost_output['opex_pump'] = round(Opex_fixed_pump + Opex_var_pump, 2)
        cost_output['opex_hex'] = round(Opex_fixed_hex, 2)
        cost_output['el_network_MWh'] = round(el_MWh, 2)
        cost_output['el_price'] = prices.ELEC_PRICE
        cost_output['capex_network'] = round(Capex_a_netw, 2)
        cost_output['capex_pumps'] = round(Capex_a_pump, 2)
        cost_output['capex_hex'] = round(Capex_a_hex, 2)
        cost_output['capex_chiller'] = round(Capex_a_chiller, 2)
        cost_output['capex_CT'] = round(Capex_a_CT, 2)
        cost_output['avg_diam_m'] = average_diameter_m
        cost_output['network_length_m'] = length_m

        cost_output = pd.DataFrame.from_dict(cost_output, orient='index').T
        cost_output.to_csv(locator.get_optimization_network_layout_costs_file(network_type))

        print('Thermal network cost calculations completed successfully.')

    except Exception as e:
        print(f'[Thermal-Network] Cost evaluation failed: {str(e)}')
