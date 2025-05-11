#!/usr/bin/env python

"""
Building Clustering Script for DTN Phased Optimization

This script clusters buildings based on annual heat demand (column "QH_sys_MWhyr" in Total_demand.csv)
and building proximity. *Cooling demand not yet considered and should be integrated later.

Outputs include:
1. A CSV file with building cluster assignments
2. A shapefile with individual building geometries and cluster assignments

The script handles mixed-use buildings and supports multi-stage district thermal network planning.
"""

__author__ = "Fan Ut Chang"
__copyright__ = "Copyright 2025, Architecture and Building Systems - ETH Zurich"
__credits__ = ["Fan Ut Chang"]
__license__ = "MIT"
__version__ = "0.1"
__maintainer__ = "Fan Ut Chang"
__email__ = "changf@ethz.ch"
__status__ = "Production"

# Standard libraries
import os
import sys
import warnings
from collections import Counter

# Data handling libraries
import pandas as pd
import numpy as np

# Geospatial data handling libraries
import geopandas as gpd
from shapely.geometry import Point, LineString
from scipy.spatial.distance import cdist, pdist, squareform

# Clustering libraries
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from hdbscan import HDBSCAN
from scipy.spatial import cKDTree

# Visualization library
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# CEA configuration and file locator
import cea.config
import cea.inputlocator

os.environ['OMP_NUM_THREADS'] = '1'

def get_existing_DTN_buildings_list(config):
    value = config.building_clustering.existing_dtn_buildings
    if isinstance(value, str):
        return [b.strip() for b in value.split(',') if b.strip()]
    elif isinstance(value, list):
        return value
    else:
        raise ValueError("Unexpected type for existing_dtn_buildings")

def merge_building_data(buildings_shp, demand_df):
    """
    Merge building spatial data (from zone shapefile) with demand data (from Total_demand.csv).
    - Calculates each building's centroid and extracts the x and y coordinates

    Parameters:
    -----------
    buildings_shp : GeoDataFrame
        Building geometries from the zone shapefile
    demand_df : DataFrame
        Building energy demand from Total_demand.csv

    Returns:
    --------
    GeoDataFrame
        A merged DataFrame containing building footprints, demand data, and centroid coordinates
    """
    # Convert name fields to string for consistent joining
    buildings_shp['name'] = buildings_shp['name'].astype(str)
    demand_df['name'] = demand_df['name'].astype(str)

    # Merge on building name
    merged_df = buildings_shp.merge(demand_df, on='name', how='inner')

    # Calculate centroid coordinates for each building
    centroids = merged_df.geometry.centroid
    merged_df['x'] = centroids.x
    merged_df['y'] = centroids.y

    return merged_df


def process_construction_year(buildings_df, year_weight=1.0):
    """
    Process construction year into meaningful categories based on building regulations periods.

    Parameters:
    -----------
    buildings_df : GeoDataFrame
        Building data with construction year information
    year_weight : float
        Weight to apply to construction year features (0-1)

    Returns:
    --------
    GeoDataFrame
        DataFrame with processed construction year features
    """
    df = buildings_df.copy()

    # Check if construction year column exists
    year_column = None
    potential_columns = ['year_built', 'construction_year', 'YEAR_BUILT', 'CONSTRUCTION_YEAR', 'YEAR', 'year']

    for col in potential_columns:
        if col in df.columns:
            year_column = col
            break

    if year_column is None:
        print("Warning: Construction year information not found. Skipping construction year processing.")
        return df

    # Define periods based on significant building code changes
    year_breaks = [1900, 1945, 1970, 1985, 2000, 2010, 2020]
    period_labels = ['pre1900', '1900_1945', '1946_1970', '1971_1985', '1986_2000', '2001_2010', 'post2010', 'future']

    # Create a categorical period column
    df['constr_period'] = pd.cut(df[year_column],
                                 bins=[-float('inf')] + year_breaks + [float('inf')],
                                 labels=period_labels,
                                 ordered=True)

    # Create binary columns for each period with the specified weight
    for period in period_labels:
        col_name = f"period_{period}"
        df[col_name] = (df['constr_period'] == period).astype(float) * year_weight

    return df


def process_use_type(buildings_df):
    """
    Process building use_types from CEA-4 use_type format with mixed-use buildings support.

    Parameters:
    -----------
    buildings_df : GeoDataFrame
        Building data with use_type1, use_type2, use_type3 columns

    Returns:
    --------
    GeoDataFrame
        DataFrame with processed use_type information
    """
    df = buildings_df.copy()

    # Check if use_type columns exist
    if 'use_type1' not in df.columns:
        print("Warning: Building use_type information not found. Skipping use_type processing.")
        return df

    # Create an use_type field based on use_type1
    df['use_type'] = df['use_type1']

    # Flag mixed-use buildings (those with a non-zero secondary use_type)
    if 'use_type2' in df.columns and 'use_type2r' in df.columns:
        mixed_use_mask = (df['use_type2'].notna()) & (df['use_type2r'] > 0)
        df['is_mixed_use'] = False
        df.loc[mixed_use_mask, 'is_mixed_use'] = True

    # Create numerical representations for each use_type
    # First collect all unique use_types
    use_types = set()
    for i in range(1, 4):
        type_col = f'use_type{i}'
        ratio_col = f'use_type{i}r'

        # Add to use_types if present with non-zero ratio
        if type_col in df.columns and ratio_col in df.columns:
            for use_type, ratio in zip(df[type_col], df[ratio_col]):
                if use_type and not pd.isna(use_type) and ratio > 0:
                    use_types.add(use_type)

    # Now create one-hot encoded columns for each use_type
    for use_type in use_types:
        # Truncate building type if too long for shapefile (max 10 chars)
        column_name = use_type[:10] if len(use_type) > 10 else use_type

        # Initialize with zero
        df[column_name] = 0.0

        # Fill in values from each use_type column
        for i in range(1, 4):
            type_col = f'use_type{i}'
            ratio_col = f'use_type{i}r'

            if type_col in df.columns and ratio_col in df.columns:
                # Where this use_type matches, set the ratio
                match_mask = (df[type_col] == use_type) & (df[ratio_col] > 0)

    return df


def ensure_use_type_diversity(df, min_use_types):
    """
    For each non-noise cluster, ensure it contains at least `min_use_types` distinct
    use_types. If not, attempt to swap in nearby buildings of missing use_types.
    """
    # Build spatial index once
    coords = df[['x', 'y']].values
    tree = cKDTree(coords)
    clusters = df['cluster'].values

    for cluster_id in sorted(set(clusters)):
        if cluster_id < 0:
            continue  # skip noise

        mask = clusters == cluster_id
        use_types = set(df.loc[mask, 'use_type'])
        if len(use_types) >= min_use_types:
            continue

        # Find nearby candidates from other clusters
        for idx in np.where(mask)[0]:
            # Query nearest neighbors
            dists, nbrs = tree.query(coords[idx], k=10)
            for dist, nbr in zip(dists[1:], nbrs[1:]):
                if clusters[nbr] != cluster_id:
                    nbr_use = df.at[nbr, 'use_type']
                    if nbr_use not in use_types:
                        # Swap this neighbor into current cluster
                        clusters[nbr] = cluster_id
                        use_types.add(nbr_use)
                        break
            if len(use_types) >= min_use_types:
                break

    df['cluster'] = clusters
    return df

def split_large_clusters(df, max_size=None, min_size=5):
    """
    Split large clusters into smaller ones using KMeans within the cluster.
    """
    # First, check if we need to define max_size
    if max_size is None:
        # Calculate average size and set max_size to 2x average
        avg_size = df[df['cluster'] >= 0]['cluster'].value_counts().mean()
        max_size = max(int(avg_size * 2), min_size * 2)
        print(f"Automatically determined max_size: {max_size} buildings")

    # Get cluster sizes
    cluster_sizes = df['cluster'].value_counts()
    print(f"Before splitting: {len(cluster_sizes)} clusters")
    print(f"Cluster sizes: {dict(cluster_sizes.sort_index())}")

    # Start numbering new clusters after the highest existing cluster number
    next_cluster_id = df['cluster'].max() + 1

    # Track if any splitting was done
    split_performed = False

    # Check each cluster
    for cluster_id, size in cluster_sizes.items():
        if cluster_id == -1:  # Skip noise points
            continue

        if size > max_size:
            # This cluster needs to be split
            print(f"Splitting cluster {cluster_id} with {size} buildings (exceeds max size of {max_size})")
            split_performed = True

            # Get the buildings in this cluster
            cluster_buildings = df[df['cluster'] == cluster_id]

            # Calculate how many subclusters we need
            n_subclusters = max(2, int(np.ceil(size / max_size)))
            print(f"  → Creating {n_subclusters} subclusters")

            # Use KMeans to split this cluster
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=n_subclusters, random_state=42, n_init=10)

            # Get coordinates for clustering
            X = cluster_buildings[['x', 'y']].values
            subclusters = kmeans.fit_predict(X)

            # Assign new cluster IDs
            for i in range(n_subclusters):
                # Get buildings that belong to this subcluster
                subcluster_idx = cluster_buildings.index[subclusters == i]

                if i == 0:
                    # Keep the first subcluster with original ID
                    print(f"  → Keeping original cluster {cluster_id} with {len(subcluster_idx)} buildings")
                else:
                    # Assign new IDs to other subclusters
                    print(f"  → Creating new cluster {next_cluster_id} with {len(subcluster_idx)} buildings")
                    df.loc[subcluster_idx, 'cluster'] = next_cluster_id
                    next_cluster_id += 1

    # Print summary after splitting
    final_cluster_sizes = df['cluster'].value_counts()
    print(f"After splitting: {len(final_cluster_sizes)} clusters")
    print(f"Final cluster sizes: {dict(final_cluster_sizes.sort_index())}")

    if not split_performed:
        print("No clusters exceeded size threshold. No splitting performed.")

    return df


def reassign_noise(df, max_distance=200):
    print("DEBUG: Inside reassign_noise function")
    print("DEBUG: Before reassignment, number of noise points (-1):", sum(df['cluster'] == -1))
    """
    Reassign HDBSCAN “noise” points (cluster = ‑1) to the nearest non-noise cluster
    if they lie within max_distance meters of any cluster member.
    """
    coords = df[['x', 'y']].values
    tree = cKDTree(coords)
    clusters = df['cluster'].values.copy()

    # Identify noise point indices
    noise_idxs = np.where(clusters == -1)[0]

    for idx in noise_idxs:
        # Query the nearest non-noise neighbor
        dists, nbrs = tree.query(coords[idx], k=len(coords))
        # Filter out self and all noise neighbors
        valid = [(d, n) for d, n in zip(dists[1:], nbrs[1:]) if clusters[n] != -1]
        if not valid:
            continue
        nearest_dist, nearest_idx = min(valid, key=lambda x: x[0])
        if nearest_dist <= max_distance:
            clusters[idx] = clusters[nearest_idx]

    df['cluster'] = clusters
    print("DEBUG: After reassignment in function, number of noise points (-1):", sum(df['cluster'] == -1))
    return df

# For spatial-only feature preparation
def prepare_features(merged_df, heat_col='QH_sys_MWhyr', spatial_weight=25.0,
                     use_type_weight=1.0, year_weight=1.0,
                     include_heat_demand=False, clustering_algorithm='hdbscan'):
    """
    Prepares the feature matrix for clustering with adjustable feature weighting.
    Handles both HDBSCAN (spatial-only) and K-means (multi-feature) approaches.
    """
    # For HDBSCAN: use only spatial features with increased weight
    if clustering_algorithm.lower() == 'hdbscan':
        numeric_features = merged_df[['x', 'y']].copy()
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(numeric_features)
        scaled_features[:, 0] *= spatial_weight
        scaled_features[:, 1] *= spatial_weight
        return scaled_features, scaler

    # For K-means: use all specified features
    else:
        # Keep as MWh instead of converting to kWh
        merged_df['heat_MWhyr'] = merged_df[heat_col]

        # Create initial numeric features array (with or without heat demand)
        if include_heat_demand:
            numeric_features = merged_df[['heat_MWhyr', 'x', 'y']].copy()
        else:
            numeric_features = merged_df[['x', 'y']].copy()

        # Normalize numeric features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(numeric_features)

        # Apply weights to spatial coordinates
        if include_heat_demand:
            scaled_features[:, 1] *= spatial_weight  # x coordinate
            scaled_features[:, 2] *= spatial_weight  # y coordinate
        else:
            scaled_features[:, 0] *= spatial_weight  # x coordinate
            scaled_features[:, 1] *= spatial_weight  # y coordinate

        # Add construction year features
        year_cols = [col for col in merged_df.columns if col.startswith('period_')]
        if year_cols:
            year_features = merged_df[year_cols].values
            if np.any(year_features):
                year_scaler = StandardScaler()
                scaled_year_features = year_scaler.fit_transform(year_features)
                scaled_features = np.hstack((scaled_features, scaled_year_features))

        # Add use_type features
        use_cols = [col for col in merged_df.columns if col not in
                    ['name', 'geometry', 'x', 'y', 'heat_MWhyr', 'use_type', 'is_mixed_use',
                     'use_type1', 'use_type2', 'use_type3', 'use_type1r', 'use_type2r', 'use_type3r',
                     'constr_period'] + year_cols
                    and col not in merged_df.columns[:20]]

        if use_cols:
            use_type_features = merged_df[use_cols].values
            if np.any(use_type_features):
                use_type_scaler = StandardScaler()
                scaled_use_type_features = use_type_scaler.fit_transform(use_type_features)
                scaled_use_type_features *= use_type_weight
                scaled_features = np.hstack((scaled_features, scaled_use_type_features))

        return scaled_features, scaler


# New function to ensure spatial coherence in clusters
def ensure_spatial_coherence(df, max_distance_threshold=100):
    """
    Post-process clusters to ensure buildings in the same cluster are spatially coherent.
    Buildings farther than max_distance_threshold from all other buildings in their cluster
    are reassigned as noise points.

    Parameters:
    -----------
    df : DataFrame
        DataFrame with cluster assignments and x, y coordinates
    max_distance_threshold : float
        Maximum allowed distance to nearest building in the same cluster (in map units)

    Returns:
    --------
    DataFrame
        Post-processed dataframe with updated cluster assignments
    """
    processed_df = df.copy()
    clusters = df['cluster'].unique()

    # Skip noise points (cluster -1)
    clusters = [c for c in clusters if c >= 0]

    for cluster in clusters:
        cluster_buildings = df[df['cluster'] == cluster]

        # Skip if only one building in cluster
        if len(cluster_buildings) <= 1:
            continue

        # Calculate pairwise distances between buildings in this cluster
        coords = cluster_buildings[['x', 'y']].values
        distances = cdist(coords, coords)

        # For each building, check if it's too far from all others
        for i, idx in enumerate(cluster_buildings.index):
            # Exclude self-distance by setting it to infinity
            dist_to_others = distances[i, :].copy()
            dist_to_others[i] = np.inf

            # Find distance to closest building in same cluster
            min_distance = np.min(dist_to_others)

            # If too isolated, mark as noise
            if min_distance > max_distance_threshold:
                processed_df.loc[idx, 'cluster'] = -1
                print(f"Building {cluster_buildings.iloc[i]['name']} removed from cluster {cluster} (too isolated)")

    return processed_df


def perform_clustering_kmeans(scaled_features, n_clusters, min_buildings_per_cluster=5):
    """
    Runs KMeans clustering with minimum cluster size constraint.

    Parameters:
    -----------
    scaled_features : ndarray
        The normalized feature array
    n_clusters : int
        Number of clusters to create
    min_buildings_per_cluster : int
        Minimum number of buildings per cluster

    Returns:
    --------
    labels : ndarray
        Cluster labels for each building
    model : KMeans
        The fitted KMeans model
    """
    # Ensure we don't try to create more clusters than data points
    n_clusters = min(n_clusters, len(scaled_features))

    print(f"Using KMeans Clustering with target of {n_clusters} clusters")

    # For KMeans, we can check minimum cluster sizes
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    model.fit(scaled_features)
    labels = model.labels_

    # Check if any cluster is too small (only if min_buildings_per_cluster > 1)
    if min_buildings_per_cluster > 1:
        unique_labels, counts = np.unique(labels, return_counts=True)
        too_small = counts < min_buildings_per_cluster

        # If some clusters are too small, try fewer clusters
        if np.any(too_small) and n_clusters > 2:
            print(f"Some clusters are too small. Reducing clusters from {n_clusters} to {n_clusters - 1}.")
            return perform_clustering_kmeans(scaled_features, n_clusters - 1, min_buildings_per_cluster)

    return labels, model


def perform_clustering_hdbscan(scaled_features, min_cluster_size=5, min_samples=None,
                               cluster_selection_epsilon=0.0, cluster_selection_method='leaf',
                               demand_values=None, max_demand_ratio=3.0):
    """
    Runs HDBSCAN clustering with optional demand-based post-processing.

    Parameters:
    -----------
    scaled_features : ndarray
        The normalized feature array
    min_cluster_size : int
        Minimum size of clusters (smaller clusters are considered noise)
    min_samples : int or None
        Number of samples in a neighborhood for a point to be a core point
    cluster_selection_epsilon : float
        Distance threshold for expanding clusters
    cluster_selection_method : str
        Method for selecting flat clusters from the hierarchy ('eom' or 'leaf')
    demand_values : array-like or None
        Heat demand values for each building for demand-based post-processing
    max_demand_ratio : float
        Maximum allowed ratio between highest and lowest cluster demand

    Returns:
    --------
    labels : ndarray
        Cluster labels for each building (-1 for noise points)
    clusterer : HDBSCAN
        The fitted HDBSCAN model
    """
    # Set min_samples to same as min_cluster_size if not provided
    if min_samples is None:
        min_samples = min_cluster_size

    print(f"Using HDBSCAN clustering with min_cluster_size={min_cluster_size}")

    # Create and fit HDBSCAN clusterer
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        metric='euclidean',
        cluster_selection_method=cluster_selection_method,
        prediction_data=True
    )

    # Perform clustering
    labels = clusterer.fit_predict(scaled_features)

    # Print summary
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    print(f"HDBSCAN found {n_clusters} clusters and {n_noise} noise points")

    # If no demand values provided, return the HDBSCAN results directly
    if demand_values is None:
        return labels, clusterer

    # Post-process to ensure demand balance between clusters
    labels = balance_cluster_demands(labels, demand_values, max_demand_ratio)

    return labels, clusterer


def balance_cluster_demands(labels, demand_values, max_ratio=3.0):
    """
    Post-processes clustering results to balance demands between clusters.

    Parameters:
    -----------
    labels : ndarray
        Cluster labels from HDBSCAN
    demand_values : array-like
        Heat demand values for each building
    max_ratio : float
        Maximum allowed ratio between highest and lowest cluster demand

    Returns:
    --------
    balanced_labels : ndarray
        Adjusted cluster labels with balanced demands
    """
    print(f"Before demand balancing: {len(set(labels)) - 1 if -1 in labels else len(set(labels))} clusters")

    # Convert to numpy arrays if not already
    labels = np.array(labels)
    demand_values = np.array(demand_values)

    # Get unique cluster labels (excluding -1 which is noise)
    unique_clusters = np.unique(labels)
    unique_clusters = unique_clusters[unique_clusters >= 0]

    if len(unique_clusters) <= 1:
        return labels  # No balancing needed with 0 or 1 cluster

    # Calculate total demand per cluster
    cluster_demands = {}
    for cluster in unique_clusters:
        cluster_demand = np.sum(demand_values[labels == cluster])
        cluster_demands[cluster] = cluster_demand

    # Check if demand ratio is within threshold
    max_demand = max(cluster_demands.values())
    min_demand = min(cluster_demands.values())

    if min_demand > 0 and (max_demand / min_demand) <= max_ratio:
        return labels  # Already balanced

    # If imbalanced, identify the high-demand buildings in the high-demand clusters
    high_demand_threshold = np.percentile(demand_values, 80)  # Top 20% buildings by demand

    balanced_labels = labels.copy()
    next_cluster_id = max(unique_clusters) + 1

    # Create new clusters for high-demand buildings from overloaded clusters
    for cluster in unique_clusters:
        # Skip clusters with demand below average
        if cluster_demands[cluster] < (sum(cluster_demands.values()) / len(cluster_demands)):
            continue

        # Find high-demand buildings in this cluster
        cluster_mask = (labels == cluster)
        high_demand_mask = (demand_values > high_demand_threshold) & cluster_mask

        # If significant number of high-demand buildings, create a new cluster
        if np.sum(high_demand_mask) >= min(3, np.sum(cluster_mask) // 3):
            balanced_labels[high_demand_mask] = next_cluster_id
            next_cluster_id += 1

    # Assign noise points (-1) to nearest cluster or their own clusters if high demand
    noise_mask = (labels == -1)
    if np.any(noise_mask):
        for i in np.where(noise_mask)[0]:
            if demand_values[i] > high_demand_threshold:
                # High-demand noise points get their own cluster
                balanced_labels[i] = next_cluster_id
                next_cluster_id += 1

    print(f"After demand balancing: {len(set(labels)) - 1 if -1 in labels else len(set(labels))} clusters")

    return balanced_labels

def spatial_majority_reassignment(df, n_neighbors=3):
    print("DEBUG: spatial_majority_reassignment function called with n_neighbors =", n_neighbors)
    """
    For each building, assign it to the majority cluster among its n nearest neighbors (excluding itself).
    """
    coords = df[['x', 'y']].values
    tree = cKDTree(coords)
    clusters = df['cluster'].values.copy()
    for i, (x, y) in enumerate(coords):
        dists, idxs = tree.query([x, y], k=n_neighbors+1)
        neighbor_clusters = clusters[idxs[1:]]  # Exclude self
        majority = pd.Series(neighbor_clusters).mode()
        if len(majority) > 0 and clusters[i] != majority[0]:
            clusters[i] = majority[0]
    df['cluster'] = clusters
    return df

def visualize_clusters(merged_df, show_interactive=False, save_path=None):
    """
    Visualizes the clustered buildings using a scatter plot with discrete colors.

    Parameters:
    -----------
    merged_df : DataFrame
        The dataframe with cluster assignments
    show_interactive : bool
        If True, shows an interactive plot (blocking call)
    save_path : str or None
        If provided, saves the plot to this path instead of displaying it
    """
    # Get unique cluster numbers
    clusters = sorted(merged_df['cluster'].unique())
    n_clusters = len(clusters)

    # Choose a discrete colormap based on number of clusters
    if n_clusters <= 10:
        # For small number of clusters, use a qualitative colormap
        colors = plt.cm.tab10.colors[:n_clusters]
    else:
        # For more clusters, use a larger discrete colormap
        colors = plt.cm.tab20.colors[:n_clusters]

    # Create a custom colormap
    cmap = ListedColormap(colors)

    # Create a scatter plot
    plt.figure(figsize=(12, 9))

    # Plot each cluster with a discrete color
    for i, cluster in enumerate(clusters):
        if cluster == -1:  # Noise points in HDBSCAN
            cluster_data = merged_df[merged_df['cluster'] == cluster]
            plt.scatter(cluster_data['x'], cluster_data['y'],
                        color='black', s=50, edgecolor='k', alpha=0.5,
                        label=f'Noise ({len(cluster_data)} bldgs)')
        else:
            cluster_data = merged_df[merged_df['cluster'] == cluster]
            plt.scatter(cluster_data['x'], cluster_data['y'],
                        color=colors[i % len(colors)], s=100, edgecolor='k', alpha=0.7,
                        label=f'Cluster {cluster} ({len(cluster_data)} bldgs)')

    # Add labels and title
    plt.xlabel('X Coordinate', fontsize=12)
    plt.ylabel('Y Coordinate', fontsize=12)
    plt.title('Building Clusters based on Location and use_types', fontsize=14)

    # Add a legend
    plt.legend(title='Clusters', loc='best', bbox_to_anchor=(1.05, 1), borderaxespad=0., fontsize=10)

    # Make layout tight to accommodate legend
    plt.tight_layout()

    # Either save the plot or show it interactively
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Plot saved to {save_path}")
    elif show_interactive:
        plt.show()  # Blocking call - waits for user to close window
    else:
        # Non-blocking - creates the figure but doesn't pause execution
        plt.draw()
        plt.pause(0.001)  # Small pause to render the figure
        plt.close()

def save_results(cluster_data, locator):
    """
    Save clustering results to the project outputs directory.
    Only saves essential columns for further analysis.

    Parameters:
    ----------
    cluster_data : DataFrame
        Data containing building IDs and their assigned clusters
    locator : InputLocator
        CEA InputLocator object

    Returns:
    -------
    tuple
        Paths to the saved CSV and shapefile
    """
    # Get scenario path from locator
    scenario_path = locator.scenario

    # Define the output directory
    output_dir = os.path.join(scenario_path, 'outputs', 'data', 'optimization', 'dtn_expansion')

    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Define output paths
    csv_path = os.path.join(output_dir, 'building_clusters.csv')
    shp_path = os.path.join(output_dir, 'building_clusters.shp')

    # Define core columns to keep with new order and renamed columns
    # Put 'use_type' right after 'heat_MWhyr' as requested
    core_columns = ['name', 'x', 'y', 'in_existing_DTN', 'cluster', 'heat_MWhyr', 'use_type']

    # Create a copy to avoid modifying the original dataframe
    save_data = cluster_data.copy()

    # Ensure all columns in core_columns exist
    for col in core_columns:
        if col not in save_data.columns and col == 'use_type' and 'dominant_use' in save_data.columns:
            save_data[col] = save_data['dominant_use']
        elif col not in save_data.columns and col == 'heat_MWhyr' and 'heat_kWhyr' in save_data.columns:
            save_data[col] = save_data['heat_kWhyr'] / 1000  # Convert kWh to MWh if needed
        elif col not in save_data.columns:
            save_data[col] = np.nan

    # Get geometry if it exists
    if isinstance(save_data, gpd.GeoDataFrame) and 'geometry' in save_data.columns:
        has_geometry = True
    else:
        has_geometry = False

    # Create the final dataframe with selected columns
    columns_to_keep = core_columns
    if has_geometry:
        columns_to_keep.append('geometry')

    # Create a trimmed dataframe with just the core columns
    trimmed_data = save_data[columns_to_keep].copy()

    # Save CSV (without geometry column if present)
    if has_geometry:
        trimmed_data.drop(columns=['geometry']).to_csv(csv_path, index=False)
    else:
        trimmed_data.to_csv(csv_path, index=False)

    # Save shapefile (only possible if geometry exists)
    if has_geometry:
        # Convert to GeoDataFrame if not already
        if not isinstance(trimmed_data, gpd.GeoDataFrame):
            trimmed_data = gpd.GeoDataFrame(trimmed_data, geometry='geometry')

        # Set CRS if available from original data
        if hasattr(cluster_data, 'crs'):
            trimmed_data.crs = cluster_data.crs

        # Save to shapefile
        trimmed_data.to_file(shp_path)

    print(f"Results saved to {output_dir}")
    return csv_path, shp_path

def cluster_buildings(buildings_shp, demand_df, locator,
                      existing_dtn_buildings=None,
                      clustering_algorithm='hdbscan',  # Changed default to HDBSCAN
                      extra_clusters=10,  # For K-means
                      min_cluster_size=5,  # For HDBSCAN
                      min_samples=None,  # For HDBSCAN
                      cluster_selection_epsilon=0.0,  # For HDBSCAN
                      cluster_selection_method='leaf',  # Changed from 'eom' to 'leaf' for better spatial coherence
                      network_type='DH',
                      spatial_weight=25.0,
                      use_type_weight=1.0,
                      year_weight=1.0,
                      use_construction_year=False,
                      noise_flag=True,
                      noise_reassign_distance=200,
                      ensure_min_use_types=True,
                      min_use_types_per_cluster=2,
                      min_buildings_per_cluster=1,
                      include_heat_demand=False,
                      max_demand_ratio=3.0,
                      max_distance_threshold=100,  # Maximum distance between buildings in same cluster
                      show_interactive_plot=False):
    """
    Main clustering function with support for K-means or HDBSCAN algorithms.

    Parameters:
    -----------
    buildings_shp : GeoDataFrame
        Building footprints from shapefile
    demand_df : DataFrame
        Building demand data
    locator : InputLocator
        CEA InputLocator object
    existing_dtn_buildings : list
        List of buildings to be considered in the existing district thermal network
    clustering_algorithm : str
        Clustering algorithm to use ('kmeans' or 'hdbscan')
    extra_clusters : int
        Number of clusters for K-means algorithm
    min_cluster_size : int
        Minimum size of clusters for HDBSCAN algorithm
    min_samples : int or None
        Number of samples in a neighborhood for HDBSCAN
    cluster_selection_epsilon : float
        Distance threshold for expanding clusters in HDBSCAN
    cluster_selection_method : str
        Method for selecting flat clusters ('eom' or 'leaf')
    network_type : str
        'DH' for district heating or 'DC' for district cooling
    spatial_weight : float
        Weight multiplier for spatial coordinates
    use_type_weight : float
        Weight multiplier for building use_types
    year_weight : float
        Weight multiplier for construction year
    use_construction_year : bool
        Whether to include construction year in clustering
    ensure_min_use_types : bool
        If true, enforces minimum number of use_types per cluster
    min_use_types_per_cluster : int
        Minimum number of use_types required in each cluster
    min_buildings_per_cluster : int
        Minimum number of buildings per cluster for K-means
    include_heat_demand : bool
        Whether to include heat demand as a clustering feature
    reassign_noise : bool
        Whether to reassign the noise to the best match of the other clusters
    noise_reassign_distance : float
        The max distance threshold for reassigning noise points
    max_demand_ratio : float
        Maximum ratio between highest and lowest cluster demand
    max_distance_threshold : float
        Maximum distance between buildings in the same cluster (for spatial coherence)
    show_interactive_plot : bool
        If True, shows interactive plot (blocks execution)
    """
    # Process use_types with diversity option
    buildings_shp = process_use_type(buildings_shp)

    # Process construction year if enabled
    if use_construction_year:
        buildings_shp = process_construction_year(buildings_shp, year_weight)

    # Merge spatial data and demand data
    merged_df = merge_building_data(buildings_shp, demand_df)

    # Store heat demand in MWh
    merged_df['heat_MWhyr'] = merged_df['QH_sys_MWhyr']

    # Identify buildings in existing DTN from config
    dtn_buildings = existing_dtn_buildings or []
    print(f"Using {len(dtn_buildings)} buildings from existing DTN: {dtn_buildings}")

    # Add a flag to mark DTN buildings
    merged_df['in_existing_DTN'] = merged_df['name'].isin(dtn_buildings)

    # Separate DTN and non-DTN buildings
    dtn_df = merged_df[merged_df['in_existing_DTN']].copy()
    non_dtn_df = merged_df[~merged_df['in_existing_DTN']].copy()

    # Extract heat demand for potential post-processing
    demand_values = non_dtn_df['heat_MWhyr'].values

    # For non-DTN buildings, prepare features and perform clustering
    # Perform clustering based on selected algorithm
    if clustering_algorithm.lower() == 'hdbscan':
        # Prepare features with clustering_algorithm parameter
        scaled_features, scaler = prepare_features(
            non_dtn_df,
            heat_col='QH_sys_MWhyr',
            spatial_weight=spatial_weight,
            use_type_weight=use_type_weight,
            year_weight=year_weight,
            include_heat_demand=include_heat_demand,
            clustering_algorithm='hdbscan'  # Explicitly set algorithm
        )

        # Perform HDBSCAN clustering
        labels, clusterer = perform_clustering_hdbscan(
            scaled_features,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=cluster_selection_epsilon,
            cluster_selection_method=cluster_selection_method
        )

        # Assign initial labels
        non_dtn_df['cluster'] = labels

        # Post-processing steps - FIXED LINE BELOW
        non_dtn_df = split_large_clusters(non_dtn_df,
                                          max_size=25)  # Changed parameter name from max_cluster_size to max_size
        non_dtn_df = ensure_spatial_coherence(non_dtn_df, max_distance_threshold=100)  # Remove spatial outliers
        non_dtn_df = reassign_noise(non_dtn_df, max_distance=200)  # Reassign nearby noise

        # Enforce use-type diversity after spatial processing
        if ensure_min_use_types:
            non_dtn_df = ensure_use_type_diversity(non_dtn_df, min_use_types_per_cluster)

        # Final cluster ID adjustment
        valid_clusters = non_dtn_df['cluster'].unique()
        cluster_map = {old: i + 1 for i, old in enumerate(valid_clusters)}
        cluster_map[-1] = -1
        non_dtn_df['cluster'] = non_dtn_df['cluster'].map(cluster_map)
        # ========== MODIFIED SECTION END ============

    else:
        # For K-means, use the standard approach with all features
        scaled_features, scaler = prepare_features(
            non_dtn_df,
            heat_col='QH_sys_MWhyr',
            spatial_weight=spatial_weight,
            use_type_weight=use_type_weight,
            year_weight=year_weight,
            include_heat_demand=include_heat_demand,
            clustering_algorithm='kmeans'
        )

        # Perform K-means clustering
        labels, clusterer = perform_clustering_kmeans(
            scaled_features,
            n_clusters=extra_clusters,
            min_buildings_per_cluster=min_buildings_per_cluster
        )

        # Assign cluster labels to non-DTN buildings
        non_dtn_df['cluster'] = labels + 1  # Shift labels so that DTN becomes cluster 0

    # Ensure all cluster labels are ≥ 1 (for non-DTN buildings)
    # This ensures DTN buildings can have cluster = 0
    if non_dtn_df['cluster'].min() < 1:
        # Get all unique cluster values excluding noise (-1)
        unique_clusters = sorted(list(set(non_dtn_df['cluster'].values)))
        if -1 in unique_clusters:
            unique_clusters.remove(-1)

        # Create a mapping from old to new cluster labels
        cluster_map = {old: i + 1 for i, old in enumerate(unique_clusters)}
        # Keep noise as -1
        cluster_map[-1] = -1

        # Apply the mapping
        non_dtn_df['cluster'] = non_dtn_df['cluster'].map(lambda x: cluster_map.get(x, x))

    # For DTN buildings, assign a cluster label of 0
    if not dtn_df.empty:
        dtn_df['cluster'] = 0
        final_df = pd.concat([dtn_df, non_dtn_df], ignore_index=True)
    else:
        final_df = non_dtn_df

    # Reassign noise buildings to the clusters that are within the noise_reassign_distance
    print("DEBUG: noise_flag value:", noise_flag)
    if noise_flag:
        print("DEBUG: Calling reassign_noise function with max_distance =", noise_reassign_distance)
        final_df = reassign_noise(final_df, max_distance=noise_reassign_distance)
        print("DEBUG: After reassign_noise, number of noise points (-1):", sum(final_df['cluster'] == -1))
    else:
        print("DEBUG: reassign_noise function NOT called because noise_flag is False")

    # Ensure minimum use_type diversity if requested
    if ensure_min_use_types:
        final_df = ensure_use_type_diversity(final_df, min_use_types_per_cluster)

    final_df = spatial_majority_reassignment(final_df, n_neighbors=3)

    # Save results
    out_csv, out_shp = save_results(final_df, locator)

    print(f"Results saved to:")
    print(f"  - Building clusters CSV: {out_csv}")
    print(f"  - Building clusters shapefile: {out_shp}")

    # Generate visualization for clusters
    plot_path = os.path.join(os.path.dirname(out_csv), 'building_clusters_plot.png')
    visualize_clusters(
        final_df,
        show_interactive=show_interactive_plot,
        save_path=plot_path
    )

    return final_df


def main(config):
    """
    Main function to run the building clustering.

    Args:
        config: Configuration object with all parameters
    """
    # Get parameters from config
    scenario = config.scenario

    # Algorithm selection
    clustering_algorithm = config.building_clustering.clustering_algorithm

    # K-means parameters
    extra_clusters = config.building_clustering.extra_clusters
    min_buildings_per_cluster = config.building_clustering.min_buildings_per_cluster

    # HDBSCAN parameters
    min_cluster_size = config.building_clustering.min_cluster_size
    min_samples = config.building_clustering.min_samples
    cluster_selection_epsilon = config.building_clustering.cluster_selection_epsilon
    cluster_selection_method = config.building_clustering.cluster_selection_method
    max_demand_ratio = config.building_clustering.max_demand_ratio

    # General parameters
    existing_dtn_buildings = config.building_clustering.existing_dtn_buildings
    network_type = config.building_clustering.network_type

    # Feature weighting
    spatial_weight = config.building_clustering.spatial_weight
    use_type_weight = config.building_clustering.use_type_weight
    use_construction_year = config.building_clustering.use_construction_year
    year_weight = config.building_clustering.year_weight
    include_heat_demand = config.building_clustering.include_heat_demand

    # Noise reassignment options
    noise_flag = True  # Force enable noise reassignment
    print("DEBUG: Forcing noise_flag to True")
    noise_reassign_distance = config.building_clustering.noise_reassign_distance

    # Diversity options
    ensure_min_use_types = config.building_clustering.ensure_min_use_types
    min_use_types_per_cluster = config.building_clustering.min_use_types_per_cluster

    # Determine if we're running from GUI or command line
    # In GUI mode, don't show interactive plots to avoid blocking
    show_interactive_plot = not hasattr(config, 'multiprocessing') or not config.multiprocessing

    # Setup locator and load data
    locator = cea.inputlocator.InputLocator(scenario=scenario)
    buildings_shp = gpd.read_file(locator.get_zone_geometry())
    demand_df = pd.read_csv(locator.get_total_demand())

    # Call the cluster_buildings function with parameters from config
    final_df = cluster_buildings(
        buildings_shp,
        demand_df,
        locator,
        existing_dtn_buildings=existing_dtn_buildings,        # use the local var
        clustering_algorithm=clustering_algorithm,
        extra_clusters=extra_clusters,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        cluster_selection_method=cluster_selection_method,
        network_type=network_type,
        spatial_weight=spatial_weight,
        use_type_weight=use_type_weight,
        use_construction_year=use_construction_year,
        ensure_min_use_types=ensure_min_use_types,
        min_use_types_per_cluster=min_use_types_per_cluster,
        min_buildings_per_cluster=min_buildings_per_cluster,
        include_heat_demand=include_heat_demand,
        max_demand_ratio=max_demand_ratio,
        noise_flag=noise_flag,                                  # fix typo here
        noise_reassign_distance=noise_reassign_distance,
        show_interactive_plot=show_interactive_plot
    )




if __name__ == "__main__":
    import os
    from cea.config import Configuration

    config = Configuration()

    TEST_BUILDINGS = [
        'B0000', 'B0001', 'B0002', 'B0003', 'B0019', 'B0020', 'B0021', 'B0022',
        'B0034', 'B0035', 'B0036', 'B0041', 'B0053', 'B0054', 'B0055', 'B0056',
        'B0074', 'B0075', 'B0076',
    ]

    if '--test' in sys.argv:
        config.building_clustering.existing_dtn_buildings = TEST_BUILDINGS

    # Try to use environment variable, fall back to hardcoded path
    scenario_path = os.environ.get('CEA_SCENARIO_PATH',
                                   r"C:\Users\changf\OneDrive - ETH Zurich\CEA_projects\base_design\01_base_design_2025")
    config.scenario = scenario_path

    # Check if the 'building-clustering' section exists in the configuration
    if 'building-clustering' in config.sections:
        main(config)
    else:
        print("Error: 'building-clustering' section not found in configuration.")
        print("Available sections:", list(config.sections.keys()))
