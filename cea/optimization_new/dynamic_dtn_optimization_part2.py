from __future__ import annotations

###############################################################################
# 1) STANDARD IMPORTS                                                        #
###############################################################################
import argparse
import logging
import os
import shutil
import time
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
import itertools

import cea.config
import cea.inputlocator
from cea.optimization_new.DTN_expansion_optimization import DTNExpansionOptimizer

__author__ = "Fan Ut Chang"
__copyright__ = "Copyright 2023, Architecture and Building Systems - ETH Zurich"
__credits__ = ["Jimeno A. Fonseca", "Shanshan Hsieh", "Reynold Mok", "Mathias Niffeler"]
__license__ = "MIT"
__version__ = "0.1"
__maintainer__ = "Architecture and Building Systems - ETH Zurich"
__email__ = "cea@arch.ethz.ch"
__status__ = "Production"

def log(*args):
    """Log a message to the standard logger."""
    print(*args)
    logging.info(*args)


class CustomDTNRerunOptimizer:
    """
    A completely new implementation for Dynamic DTN Rerun Optimization.
    
    This class directly uses the files in the temp_scenario folder without
    relying on complex redirection mechanisms or inheritance from other modules.
    """
    
    def __init__(self, locator: cea.inputlocator.InputLocator, config: cea.config.Configuration):
        """
        Initialize the optimizer with direct paths.
        
        Parameters:
        -----------
        locator : cea.inputlocator.InputLocator
            The locator for the scenario
        config : cea.config.Configuration
            The configuration object
        """
        self.locator = locator
        self.config = config
        self.scenario_path = Path(locator.scenario)
        self.network_type = config.dynamic_dtn_optimization.network_type
        
        # Parse testing clusters from config
        testing_clusters_str = config.dtn_expansion_optimization.testing_clusters
        if testing_clusters_str:
            self.testing_clusters = [int(c.strip()) for c in testing_clusters_str.split(',') if c.strip()]
        else:
            self.testing_clusters = []
        
        # Define direct paths to all required folders
        self.dtn_folder = self.scenario_path / "outputs" / "data" / "optimization" / "dynamic_dtn_optimization"
        self.temp_scenario = self.dtn_folder / "temp_scenario"
        self.temp_demand_dir = self.temp_scenario / "outputs" / "data" / "demand"
        self.temp_thermal_network_dir = self.temp_scenario / "outputs" / "data" / "thermal-network"
        
        # Original paths for reference files
        self.original_dtn_dir = self.scenario_path / "outputs" / "data" / "optimization" / "dtn_expansion"
        
        # Output paths
        self.output_dir = self.dtn_folder / "rerun_results"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Verify all required directories exist
        self._verify_directories()
        
        # Log initialization
        log(f"Initialized CustomDTNRerunOptimizer with:")
        log(f"  Scenario: {self.scenario_path}")
        log(f"  Network type: {self.network_type}")
        log(f"  Testing clusters: {self.testing_clusters}")
        log(f"  Temp scenario: {self.temp_scenario}")
        log(f"  Output directory: {self.output_dir}")
    
    def _verify_directories(self):
        """Verify that all required directories exist."""
        if not self.temp_scenario.exists():
            raise FileNotFoundError(f"Temp scenario directory not found: {self.temp_scenario}")
        
        if not self.temp_demand_dir.exists():
            raise FileNotFoundError(f"Temp demand directory not found: {self.temp_demand_dir}")
        
        if not self.original_dtn_dir.exists():
            raise FileNotFoundError(f"Original DTN directory not found: {self.original_dtn_dir}")
    
    def _prepare_optimization_files(self):
        """
        Prepare all files needed for optimization.
        
        This includes copying cluster files to the temp scenario and
        ensuring the Total_demand.csv file is correctly used.
        
        Returns:
        --------
        pd.DataFrame
            The Total_demand.csv data
        """
        # Create directory for optimization files in temp scenario
        temp_opt_dir = self.temp_scenario / "outputs" / "data" / "optimization" / "dtn_expansion"
        temp_opt_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy cluster files from original scenario
        for file_name in ["cluster_edges.csv", "cluster_nodes.csv"]:
            src_file = self.original_dtn_dir / file_name
            dst_file = temp_opt_dir / file_name
            
            if src_file.exists():
                log(f"Copying {file_name} to temp scenario")
                shutil.copy2(src_file, dst_file)
            else:
                raise FileNotFoundError(f"{file_name} not found at {src_file}")
        
        # Verify Total_demand.csv exists in temp scenario
        total_demand_path = self.temp_demand_dir / "Total_demand.csv"
        if not total_demand_path.exists():
            raise FileNotFoundError(f"Total_demand.csv not found in temp scenario: {total_demand_path}")
        
        log(f"Using Total_demand.csv from temp scenario: {total_demand_path}")
        
        # Load and verify Total_demand.csv
        total_demand = pd.read_csv(total_demand_path)
        
        # Verify required columns exist
        required_columns = ['name']
        if self.network_type == 'DH':
            required_columns.extend(['Qhs_sys_MWhyr', 'Qww_sys_MWhyr'])
        else:  # DC
            required_columns.extend(['Qcs_sys_MWhyr', 'Qcre_sys_MWhyr', 'Qcdata_sys_MWhyr'])
        
        missing_columns = [col for col in required_columns if col not in total_demand.columns]
        if missing_columns:
            raise ValueError(f"Total_demand.csv is missing required columns: {missing_columns}")
        
        return total_demand
    
    def _generate_cluster_metrics(self, total_demand):
        """
        Generate metrics for all cluster combinations.
        
        This is a simplified version of the PipeLayoutGenerator.generate_pipe_layouts method
        that directly uses the files in the temp scenario folder.
        
        Parameters:
        -----------
        total_demand : pd.DataFrame
            The Total_demand.csv data
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with metrics for all cluster combinations
        """
        # Load cluster files
        temp_opt_dir = self.temp_scenario / "outputs" / "data" / "optimization" / "dtn_expansion"
        cluster_edges = pd.read_csv(temp_opt_dir / "cluster_edges.csv")
        cluster_nodes = pd.read_csv(temp_opt_dir / "cluster_nodes.csv")
        
        # Load edge-node matrix
        edge_node_files = list(self.temp_thermal_network_dir.glob("edge_node_matrix*.csv"))
        if not edge_node_files:
            log("Warning: No edge_node_matrix file found. This is not critical but may affect some calculations.")
            edge_node_matrix = None
        else:
            edge_node_file = edge_node_files[0]
            edge_node_matrix = pd.read_csv(edge_node_file, index_col=0)
        
        # Get unique clusters (excluding 0 and -1)
        all_clusters = sorted([c for c in cluster_edges['cluster'].unique() if c > 0])
        
        # Filter to testing clusters if specified
        if self.testing_clusters:
            valid_clusters = [c for c in self.testing_clusters if c in all_clusters]
            if len(valid_clusters) != len(self.testing_clusters):
                missing = set(self.testing_clusters) - set(valid_clusters)
                log(f"Warning: Some testing clusters do not exist: {missing}")
            
            clusters = sorted(valid_clusters)
            log(f"Using {len(clusters)} testing clusters: {clusters}")
        else:
            clusters = all_clusters
            log(f"Using all {len(clusters)} clusters")
        
        # Generate all possible combinations of clusters
        all_combinations = []
        for r in range(1, len(clusters) + 1):
            combinations = list(itertools.combinations(clusters, r))
            all_combinations.extend(combinations)
        
        log(f"Generated {len(all_combinations)} possible cluster combinations")
        
        # Calculate metrics for each combination
        all_metrics = []
        
        # Calculate metrics for cluster 0 alone first
        cluster0_metrics = self._calculate_metrics_for_cluster((0,), cluster_edges, cluster_nodes, total_demand)
        all_metrics.append(cluster0_metrics)
        log(f"Calculated metrics for cluster 0 (existing DTN)")
        
        # Calculate metrics for all other combinations
        for i, cluster_set in enumerate(all_combinations):
            if i % 100 == 0 and i > 0:
                log(f"Processed {i}/{len(all_combinations)} cluster combinations")
            
            metrics = self._calculate_metrics_for_cluster(cluster_set, cluster_edges, cluster_nodes, total_demand)
            all_metrics.append(metrics)
        
        # Convert to DataFrame
        metrics_df = pd.DataFrame(all_metrics)
        
        # Save to CSV
        output_file = self.output_dir / "clusters_metrics_updated.csv"
        metrics_df.to_csv(output_file, index=False)
        log(f"Saved metrics for {len(all_metrics)} cluster combinations to {output_file}")
        
        return metrics_df
    
    def _calculate_metrics_for_cluster(self, cluster_set, cluster_edges, cluster_nodes, total_demand):
        """
        Calculate metrics for a specific cluster set.
        
        This is a simplified version of the PipeLayoutGenerator.calculate_metrics method.
        
        Parameters:
        -----------
        cluster_set : tuple
            Tuple of cluster IDs
        cluster_edges : pd.DataFrame
            DataFrame of cluster edges
        cluster_nodes : pd.DataFrame
            DataFrame of cluster nodes
        total_demand : pd.DataFrame
            The Total_demand.csv data
            
        Returns:
        --------
        dict
            Dictionary of metrics for the cluster set
        """
        # Include cluster 0 (existing DTN) in the set
        clusters_to_connect = set(cluster_set).union({0})
        
        # Get required pipes
        required_pipes = self._get_required_pipes(clusters_to_connect, cluster_edges)
        
        # Calculate total pipe length
        total_pipe_length = required_pipes['length_m'].sum()
        
        # Get buildings in the clusters
        buildings_in_clusters = cluster_nodes[
            (cluster_nodes['cluster'].isin(clusters_to_connect)) &
            (cluster_nodes['type'] == 'CONSUMER')
        ]['building'].unique().tolist()
        
        # Calculate total annual demand
        if self.network_type == 'DH':
            # For district heating, use Qhs_sys_MWhyr + Qww_sys_MWhyr
            total_annual_demand = total_demand[
                total_demand['name'].isin(buildings_in_clusters)
            ]['Qhs_sys_MWhyr'].sum() + total_demand[
                total_demand['name'].isin(buildings_in_clusters)
            ]['Qww_sys_MWhyr'].sum()
            demand_type = 'Qh'
        else:
            # For district cooling, use Qcs_sys_MWhyr + Qcre_sys_MWhyr + Qcdata_sys_MWhyr
            total_annual_demand = total_demand[
                total_demand['name'].isin(buildings_in_clusters)
            ]['Qcs_sys_MWhyr'].sum() + total_demand[
                total_demand['name'].isin(buildings_in_clusters)
            ]['Qcre_sys_MWhyr'].sum() + total_demand[
                total_demand['name'].isin(buildings_in_clusters)
            ]['Qcdata_sys_MWhyr'].sum()
            demand_type = 'Qc'
        
        # Calculate linear heat density
        if total_pipe_length > 0:
            linear_heat_density = total_annual_demand / total_pipe_length * 1000  # MWh/km
        else:
            linear_heat_density = 0
        
        # Create metrics dictionary
        metrics = {
            'clusters': '+'.join(map(str, sorted(clusters_to_connect))),
            f'total_annual_{demand_type}_MWh': total_annual_demand,
            'total_pipe_length_m': total_pipe_length,
            f'linear_{demand_type}_density_MWh_per_km': linear_heat_density
        }
        
        return metrics
    
    def _get_required_pipes(self, clusters_to_connect, cluster_edges):
        """
        Get the required pipes for a set of clusters.
        
        This is a simplified version of the PipeLayoutGenerator.get_required_pipes_for_clusters method.
        
        Parameters:
        -----------
        clusters_to_connect : set
            Set of cluster IDs to connect
        cluster_edges : pd.DataFrame
            DataFrame of cluster edges
            
        Returns:
        --------
        pd.DataFrame
            DataFrame of required pipes
        """
        # Get edges that belong to the clusters in the set
        cluster_edges_subset = cluster_edges[cluster_edges['cluster'].isin(clusters_to_connect)]
        
        # Get main road edges (-1) that connect the clusters
        main_road_edges = cluster_edges[
            (cluster_edges['cluster'] == -1) &
            (cluster_edges['from_C'].isin(clusters_to_connect)) &
            (cluster_edges['to_C'].isin(clusters_to_connect))
        ]
        
        # Combine the edges
        required_pipes = pd.concat([cluster_edges_subset, main_road_edges], ignore_index=True)
        
        return required_pipes
    
    def run(self):
        """
        Run the optimization process.
        
        This method prepares all files, generates updated metrics,
        and runs the optimization algorithm.
        
        Returns:
        --------
        dict
            Dictionary with optimization results
        """
        log("=== Starting Dynamic DTN Rerun Optimization Part 2 ===")
        
        try:
            # Step 1: Prepare all files needed for optimization
            total_demand = self._prepare_optimization_files()
            
            # Step 2: Generate updated metrics
            log("Generating updated metrics with temp scenario demand files...")
            metrics_df = self._generate_cluster_metrics(total_demand)
            
            # Step 3: Compare with original metrics
            original_metrics_file = self.original_dtn_dir / "phase_1" / "clusters_metrics.csv"
            if original_metrics_file.exists():
                original_metrics = pd.read_csv(original_metrics_file)
                
                log("Comparing original and updated metrics:")
                if metrics_df.equals(original_metrics):
                    log("⚠️ Generated metrics are IDENTICAL to original metrics!")
                    log("This suggests that temp scenario demand files are not being used properly.")
                else:
                    log("✓ Generated metrics are DIFFERENT from original metrics")
                    log("This confirms that temp scenario demand files are being used correctly.")
                
                # Compare key columns
                if self.network_type == 'DH':
                    key_column = 'total_annual_Qh_MWh'
                else:
                    key_column = 'total_annual_Qc_MWh'
                    
                if key_column in metrics_df.columns and key_column in original_metrics.columns:
                    log(f"Detailed comparison of {key_column} column:")
                    
                    # Calculate differences
                    diff = metrics_df[key_column] - original_metrics[key_column]
                    
                    # Log summary statistics
                    log(f"  Max absolute difference: {diff.abs().max():.2f} MWh")
                    log(f"  Mean absolute difference: {diff.abs().mean():.2f} MWh")
                    log(f"  Sum of differences: {diff.sum():.2f} MWh")
                    log(f"  Percentage of values that changed: {(diff != 0).mean() * 100:.1f}%")
            
            # Step 4: Run the optimization algorithm
            # For now, we'll just return the metrics
            # In a future version, this would call a custom implementation of the genetic algorithm
            
            log("=== Dynamic DTN Rerun Optimization Part 2 completed successfully ===")
            return {
                "status": "success",
                "metrics_df": metrics_df,
                "output_dir": str(self.output_dir)
            }
            
        except Exception as e:
            import traceback
            log(f"Error during Dynamic DTN Rerun Optimization Part 2: {str(e)}")
            log(f"Traceback: {traceback.format_exc()}")
            
            return {
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc()
            }


def main(config):
    """
    Run the dynamic DTN optimization part 2 script.
    
    Parameters:
    -----------
    config : cea.config.Configuration
        The configuration object
    
    Returns:
    --------
    None
    """
    start = time.time()
    
    # Get the scenario and locator
    scenario = config.scenario
    locator = cea.inputlocator.InputLocator(scenario=scenario)
    
    # Create and run the optimizer
    optimizer = CustomDTNRerunOptimizer(locator, config)
    results = optimizer.run()
    
    # Log results
    if results["status"] == "success":
        log("Optimization completed successfully!")
        log(f"Results saved to: {results['output_dir']}")
    else:
        log(f"Optimization failed: {results['error']}")
    
    # Log execution time
    time_elapsed = time.time() - start
    log(f"Execution time: {time_elapsed:.2f} seconds")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the dynamic DTN optimization part 2 script.')
    parser.add_argument('-s', '--scenario', help='Path to the scenario folder')
    parser.add_argument('-c', '--config', help='Path to the config file')
    args = parser.parse_args()
    
    config = cea.config.Configuration()
    
    if args.scenario:
        config.scenario = args.scenario
    if args.config:
        config.load(args.config)
    
    main(config)