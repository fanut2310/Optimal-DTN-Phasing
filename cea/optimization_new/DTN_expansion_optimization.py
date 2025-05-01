"""
Optimization for Multi-phased District Thermal Network Expansion.
"""

__author__ = "Fan Ut Chang"
__copyright__ = "Copyright 2025"
__credits__ = ["Fan Ut Chang"]
__license__ = "MIT"
__version__ = "0.1"
__maintainer__ = "Fan Ut Chang"
__email__ = "changf@ethz.ch"
__status__ = "Production"

import os
import time
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import geopandas as gpd
import json
import pickle
from math import sqrt

from cea.optimization.constants import DH_CONVERSION_TECHNOLOGIES_WITH_SPACE_RESTRICTIONS, DH_CONVERSION_TECHNOLOGIES_WITH_SIZE_AGGREAGTION_NEEDED, DH_CONVERSION_TECHNOLOGIES_SHARE
from cea.utilities import dbf
from cea.utilities.standardize_coordinates import get_projected_coordinate_system, get_geographic_coordinate_system
from cea.optimization.distribution.network_optimization_features import NetworkOptimizationFeatures


from cea.technologies.thermal_network.thermal_network import ThermalNetwork, thermal_network_main
from cea.technologies.network_layout.main import layout_network, NetworkLayout



def main(config):
    """
    This script executes the thermal network expansion optimization sequence.
    First, it creates an MST for all buildings, then determines the optimal connection
    sequence of building clusters.

    :param config: cea configuration
    :return: None
    """
    # Extract relevant parameters from config
    scenario = config.scenario
    district_network_optimization = config.district_network_optimization
    network_type = district_network_optimization.network_type
    network_name = district_network_optimization.network_name

    print("Running thermal network layout (part 1)...")
    # Run thermal network layout
    thermal_network_layout_path = run_thermal_network_layout(config)

    print("Running thermal network simulation (part 2)...")
    # Run thermal network simulation
    thermal_network_results_path = run_thermal_network_simulation(config, thermal_network_layout_path)

    print("Getting building clusters...")
    # Get building clusters based on the network layout
    building_clusters = get_building_clusters(config, thermal_network_layout_path)

    print("Preparing phased optimization...")
    # Prepare for the phased optimization
    prepare_phased_optimization(config, building_clusters, thermal_network_results_path)

    # Create a folder to store the results of optimized networks for different phases
    result_folder = os.path.join(scenario, 'outputs', 'data', 'optimization', 'network', 'expansion')
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    phases_dict = {}
    phase_counter = 0
    for building_group in building_clusters:
        phase_counter += 1
        phases_dict[phase_counter] = building_group

    # Save the building phases assignment
    save_phase_assignment(result_folder, phases_dict)

    # Plot the network with phased buildings
    plot_network(config, thermal_network_layout_path, phases_dict)

    print("Done!")


def run_thermal_network_layout(config):
    """
    Run the thermal network layout for all buildings.
    This function sets up and runs the first part of the thermal network (layout)

    :param config: cea configuration
    :return: path to the thermal network layout results
    """
    # Extract relevant parameters from config
    scenario = config.scenario
    district_network_optimization = config.district_network_optimization
    network_type = district_network_optimization.network_type
    network_name = district_network_optimization.network_name

    # Set up paths
    building_path = os.path.join(scenario, 'inputs', 'building-geometry', 'zone.shp')
    district_shapefile_path = os.path.join(scenario, 'inputs', 'building-geometry', 'district.shp')
    streets_path = os.path.join(scenario, 'inputs', 'networks', 'streets.shp')

    network_layout_output_path = os.path.join(scenario, 'outputs', 'data', 'thermal-network', 'DH')

    # Create network layout parameters
    network_layout_parameters = NetworkLayout(network_type=network_type,
                                              network_name=network_name,
                                              connected_buildings=[],  # Connect all buildings
                                              disconnected_buildings=[],
                                              pipe_diameter=150,
                                              type_mat='T1',
                                              create_plant=True,
                                              allow_looped_networks=False,
                                              consider_only_buildings_with_demand=True)

    # Run the layout_network function
    layout_network(network_layout_parameters, building_path, district_shapefile_path, streets_path,
                   network_layout_output_path)

    # Return the path to the thermal network layout
    return network_layout_output_path


def get_node_cluster(edges_df, nodes_df):
    """
    Get the clusters of nodes in the network.

    :param edges_df: DataFrame with edges data
    :param nodes_df: DataFrame with nodes data
    :return: dictionary of node clusters
    """
    # Create a directed graph
    G = nx.Graph()

    # Add edges to the graph
    for i in range(len(edges_df)):
        start_node = edges_df.loc[i, 'start_node']
        end_node = edges_df.loc[i, 'end_node']
        G.add_edge(start_node, end_node)

    # Find connected components (clusters)
    clusters = list(nx.connected_components(G))

    # Create a dictionary to map node IDs to clusters
    node_to_cluster = {}
    for cluster_id, cluster in enumerate(clusters):
        for node in cluster:
            node_to_cluster[node] = cluster_id

    return node_to_cluster


def run_thermal_network_simulation(config, thermal_network_layout_path):
    """
    Run the thermal network simulation for the created layout.
    This function sets up and runs the second part of the thermal network (simulation)

    :param config: cea configuration
    :param thermal_network_layout_path: path to the thermal network layout results
    :return: path to the thermal network simulation results
    """
    # Extract relevant parameters from config
    scenario = config.scenario
    district_network_optimization = config.district_network_optimization
    network_type = district_network_optimization.network_type
    network_name = district_network_optimization.network_name

    # Set up paths
    thermal_network_path = os.path.join(scenario, 'outputs', 'data', 'thermal-network')

    # Create a thermal network object
    thermal_network = ThermalNetwork(locator=config.scenario, network_type=network_type, network_name=network_name)

    # Run the thermal network simulation
    thermal_network_main(config)

    return thermal_network_path


def get_building_clusters(config, thermal_network_layout_path):
    """
    Get clusters of buildings based on the thermal network layout.

    :param config: cea configuration
    :param thermal_network_layout_path: path to the thermal network layout results
    :return: list of building clusters
    """
    # Extract relevant parameters from config
    scenario = config.scenario
    district_network_optimization = config.district_network_optimization
    network_type = district_network_optimization.network_type
    network_name = district_network_optimization.network_name

    # Load the thermal network data
    edges_path = os.path.join(thermal_network_layout_path, network_type, network_name, 'edges.shp')
    nodes_path = os.path.join(thermal_network_layout_path, network_type, network_name, 'nodes.shp')

    edges_df = gpd.read_file(edges_path)
    nodes_df = gpd.read_file(nodes_path)

    # Get building nodes (excluding plant and junction nodes)
    building_nodes = nodes_df[nodes_df['Type'] == 'CONSUMER']

    # Get the node clusters
    node_clusters = get_node_cluster(edges_df, nodes_df)

    # Group buildings by their cluster
    building_clusters = []
    cluster_to_buildings = {}

    for index, row in building_nodes.iterrows():
        node_id = row['Name']
        building_name = row['Building']

        if node_id in node_clusters:
            cluster_id = node_clusters[node_id]
            if cluster_id not in cluster_to_buildings:
                cluster_to_buildings[cluster_id] = []
            cluster_to_buildings[cluster_id].append(building_name)

    # Convert dictionary to list of clusters
    for cluster_id, buildings in cluster_to_buildings.items():
        building_clusters.append(buildings)

    return building_clusters


def prepare_phased_optimization(config, building_clusters, thermal_network_results_path):
    """
    Prepare the data for the phased optimization of the thermal network.

    :param config: cea configuration
    :param building_clusters: list of building clusters
    :param thermal_network_results_path: path to the thermal network simulation results
    :return: None
    """
    # Extract relevant parameters from config
    scenario = config.scenario
    district_network_optimization = config.district_network_optimization
    network_type = district_network_optimization.network_type
    network_name = district_network_optimization.network_name

    # Create a folder to store the results
    result_folder = os.path.join(scenario, 'outputs', 'data', 'optimization', 'network', 'expansion')
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    # Calculate and save data for each phase
    for phase_num, buildings in enumerate(building_clusters, 1):
        # Create a subfolder for this phase
        phase_folder = os.path.join(result_folder, f'phase_{phase_num}')
        if not os.path.exists(phase_folder):
            os.makedirs(phase_folder)

        # Activate only the buildings for this phase
        activated_buildings = buildings

        # Generate the network layout and simulation for this phase
        phase_network_path = activate_phase_components(config, activated_buildings, phase_num, phase_folder)

        # Save the results for this phase
        save_phase_results(phase_folder, phase_network_path, activated_buildings, phase_num)


def activate_phase_components(config, buildings, phase_num, phase_folder):
    """
    Activate only the network components needed for the current phase.

    :param config: cea configuration
    :param buildings: list of buildings in the current phase
    :param phase_num: current phase number
    :param phase_folder: folder to save phase results
    :return: path to the network layout for this phase
    """
    # Extract relevant parameters from config
    scenario = config.scenario
    district_network_optimization = config.district_network_optimization
    network_type = district_network_optimization.network_type
    network_name = f"{district_network_optimization.network_name}_phase_{phase_num}"

    # Set up paths
    building_path = os.path.join(scenario, 'inputs', 'building-geometry', 'zone.shp')
    district_shapefile_path = os.path.join(scenario, 'inputs', 'building-geometry', 'district.shp')
    streets_path = os.path.join(scenario, 'inputs', 'networks', 'streets.shp')

    phase_network_layout_path = os.path.join(phase_folder, 'network_layout')
    if not os.path.exists(phase_network_layout_path):
        os.makedirs(phase_network_layout_path)

    # Create network layout parameters with only the buildings for this phase
    disconnected_buildings = []
    building_data = gpd.read_file(building_path)
    for building_name in building_data['Name']:
        if building_name not in buildings:
            disconnected_buildings.append(building_name)

    network_layout_parameters = NetworkLayout(network_type=network_type,
                                              network_name=network_name,
                                              connected_buildings=buildings,
                                              disconnected_buildings=disconnected_buildings,
                                              pipe_diameter=150,
                                              type_mat='T1',
                                              create_plant=True,
                                              allow_looped_networks=False,
                                              consider_only_buildings_with_demand=True)

    # Run the layout_network function for this phase
    layout_network(network_layout_parameters, building_path, district_shapefile_path, streets_path,
                   phase_network_layout_path)

    # Run thermal network simulation for this phase if needed
    # This can be commented out if only the layout is needed for the optimization
    phase_config = config.copy()
    phase_config.district_network_optimization.network_name = network_name
    thermal_network_main(phase_config)

    return phase_network_layout_path


def save_phase_assignment(result_folder, phases_dict):
    """
    Save the building-to-phase assignment to a JSON file.

    :param result_folder: folder to save results
    :param phases_dict: dictionary mapping phase numbers to lists of buildings
    :return: None
    """
    # Create a JSON file to store the phases
    phases_file = os.path.join(result_folder, 'building_phases.json')

    # Format the data for saving
    phases_data = {}
    for phase_num, buildings in phases_dict.items():
        phases_data[str(phase_num)] = buildings

    # Save the data
    with open(phases_file, 'w') as f:
        json.dump(phases_data, f, indent=4)

    # Create a CSV file for easier viewing
    phases_csv = os.path.join(result_folder, 'building_phases.csv')

    # Format data for CSV
    csv_data = []
    for phase_num, buildings in phases_dict.items():
        for building in buildings:
            csv_data.append({
                'Phase': phase_num,
                'Building': building
            })

    # Save CSV
    pd.DataFrame(csv_data).to_csv(phases_csv, index=False)


def save_phase_results(phase_folder, phase_network_path, buildings, phase_num):
    """
    Save the results for a specific phase.

    :param phase_folder: folder to save phase results
    :param phase_network_path: path to the network layout for this phase
    :param buildings: list of buildings in this phase
    :param phase_num: current phase number
    :return: None
    """
    # Save a list of buildings in this phase
    buildings_file = os.path.join(phase_folder, 'buildings.json')
    with open(buildings_file, 'w') as f:
        json.dump(buildings, f, indent=4)

    # Copy relevant network files to the phase folder if needed
    # This can be expanded to save more detailed results as needed

    # Create a summary file
    summary_file = os.path.join(phase_folder, 'summary.json')
    summary = {
        'phase_number': phase_num,
        'number_of_buildings': len(buildings),
        'buildings': buildings
    }

    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=4)


def plot_network(config, thermal_network_layout_path, phases_dict):
    """
    Plot the network layout with buildings color-coded by phase.

    :param config: cea configuration
    :param thermal_network_layout_path: path to the thermal network layout
    :param phases_dict: dictionary mapping phase numbers to lists of buildings
    :return: None
    """
    # Extract relevant parameters from config
    scenario = config.scenario
    district_network_optimization = config.district_network_optimization
    network_type = district_network_optimization.network_type
    network_name = district_network_optimization.network_name

    # Load the network data
    edges_path = os.path.join(thermal_network_layout_path, network_type, network_name, 'edges.shp')
    nodes_path = os.path.join(thermal_network_layout_path, network_type, network_name, 'nodes.shp')

    edges_df = gpd.read_file(edges_path)
    nodes_df = gpd.read_file(nodes_path)

    # Create a figure
    fig, ax = plt.subplots(figsize=(15, 15))

    # Plot edges (pipes)
    edges_df.plot(ax=ax, color='gray', linewidth=1)

    # Create a reverse mapping from building to phase
    building_to_phase = {}
    for phase_num, buildings in phases_dict.items():
        for building in buildings:
            building_to_phase[building] = phase_num

    # Plot nodes with colors based on phase
    consumer_nodes = nodes_df[nodes_df['Type'] == 'CONSUMER']

    # Create a colormap
    num_phases = len(phases_dict)
    cmap = plt.get_cmap('viridis', num_phases)

    # Plot each consumer node
    for idx, row in consumer_nodes.iterrows():
        building = row['Building']
        if building in building_to_phase:
            phase = int(building_to_phase[building])
            color = cmap(phase / num_phases)
            ax.scatter(row.geometry.x, row.geometry.y, c=[color], s=100, label=f'Phase {phase}')

    # Plot plant nodes
    plant_nodes = nodes_df[nodes_df['Type'] == 'PLANT']
    plant_nodes.plot(ax=ax, color='red', markersize=150, marker='*', label='Plant')

    # Plot junction nodes
    junction_nodes = nodes_df[nodes_df['Type'] == 'JUNCTION']
    junction_nodes.plot(ax=ax, color='black', markersize=30, marker='.', label='Junction')

    # Add legend, handling duplicates
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1, 1))

    # Set title and labels
    ax.set_title(f'District Heating Network Expansion Phases', fontsize=16)
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')

    # Save the plot
    result_folder = os.path.join(scenario, 'outputs', 'data', 'optimization', 'network', 'expansion')
    plt.savefig(os.path.join(result_folder, 'network_phases.png'), bbox_inches='tight', dpi=300)
    plt.close()


if __name__ == '__main__':
    from cea.config import Configuration

    config = Configuration()
    main(config)
