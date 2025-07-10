from __future__ import annotations

import os
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import matplotlib.pyplot as plt
import seaborn as sns

import cea.config
import cea.inputlocator

class DynamicDTNResultComparison:
    """
    Module for comparing and visualizing results from Dynamic DTN Optimization.
    """
    
    def __init__(self, locator: cea.inputlocator.InputLocator, config: cea.config.Configuration):
        """Initialize the result comparison module."""
        self.locator = locator
        self.config = config
        self.network_type = config.dynamic_dtn_optimization.network_type
        self.logger = logging.getLogger(__name__)
    
    def load_results(self):
        """
        Load original and modified optimization results.
        
        Returns:
            tuple: (original_results, new_results) - dictionaries containing the optimization results
        """
        # Define paths to results files
        dtn_results_folder = Path(self.locator.get_dtn_expansion_optimization_results_folder())
        dynamic_dtn_folder = Path(self.locator.get_optimization_results_folder()) / "dynamic_dtn_optimization"
        
        # Load original results
        original_results_file = dtn_results_folder / f"all_evaluated_individuals_{self.network_type}.csv"
        if not original_results_file.exists():
            self.logger.error(f"Original DTN optimization results file not found: {original_results_file}")
            raise FileNotFoundError(f"Original DTN optimization results file not found: {original_results_file}")
        
        original_results_df = pd.read_csv(original_results_file)
        
        # Find the best solution in original results
        if 'fitness_NPV' in original_results_df.columns:
            best_original = original_results_df.loc[original_results_df['fitness_NPV'].idxmax()]
        elif 'fitness_ROI' in original_results_df.columns:
            best_original = original_results_df.loc[original_results_df['fitness_ROI'].idxmax()]
        else:
            self.logger.warning("Could not find NPV or ROI fitness values in original results. Using first solution.")
            best_original = original_results_df.iloc[0]
        
        # Load new results
        new_results_file = dynamic_dtn_folder / f"all_evaluated_individuals_{self.network_type}.csv"
        if not new_results_file.exists():
            self.logger.error(f"New DTN optimization results file not found: {new_results_file}")
            raise FileNotFoundError(f"New DTN optimization results file not found: {new_results_file}")
        
        new_results_df = pd.read_csv(new_results_file)
        
        # Find the best solution in new results
        if 'fitness_NPV' in new_results_df.columns:
            best_new = new_results_df.loc[new_results_df['fitness_NPV'].idxmax()]
        elif 'fitness_ROI' in new_results_df.columns:
            best_new = new_results_df.loc[new_results_df['fitness_ROI'].idxmax()]
        else:
            self.logger.warning("Could not find NPV or ROI fitness values in new results. Using first solution.")
            best_new = new_results_df.iloc[0]
        
        # Extract genomes
        original_genome = self._extract_genome(best_original['genome'])
        new_genome = self._extract_genome(best_new['genome'])
        
        # Create result dictionaries
        original_results = {
            'genome': original_genome,
            'individual_id': best_original['individual_id'],
            'metrics': best_original.to_dict()
        }
        
        new_results = {
            'genome': new_genome,
            'individual_id': best_new['individual_id'],
            'metrics': best_new.to_dict()
        }
        
        self.original_results = original_results
        self.new_results = new_results
        
        return original_results, new_results
    
    def _extract_genome(self, genome_str):
        """
        Extract genome from string representation.
        
        Args:
            genome_str: String representation of genome
            
        Returns:
            list: Genome as a list of integers
        """
        if isinstance(genome_str, list):
            return genome_str
        else:
            try:
                return eval(genome_str)
            except:
                self.logger.error(f"Could not parse genome: {genome_str}")
                return []
    
    def compare_results(self):
        """
        Compare original and modified optimization results to assess sensitivity.
        
        Returns:
            DataFrame: Summary of comparison results
        """
        self.logger.info("Comparing original and new optimization results")
        
        # Load results if not already loaded
        if not hasattr(self, 'original_results') or not hasattr(self, 'new_results'):
            self.load_results()
        
        # Create a directory for comparison results
        comparison_dir = Path(self.locator.get_optimization_results_folder()) / "dynamic_dtn_optimization" / "comparison"
        comparison_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract connection sequences
        original_sequence = self._extract_connection_sequence(self.original_results)
        new_sequence = self._extract_connection_sequence(self.new_results)
        
        # Compare the sequences
        sequence_changes = self._compare_sequences(original_sequence, new_sequence)
        
        # Compare objective values
        objective_changes = self._compare_objectives(self.original_results, self.new_results)
        
        # Get demand reduction values from config
        heating_reduction = self.config.dynamic_dtn_optimization.heating_demand_reduction
        cooling_reduction = self.config.dynamic_dtn_optimization.cooling_demand_reduction
        dhw_reduction = self.config.dynamic_dtn_optimization.dhw_demand_reduction
        electricity_reduction = self.config.dynamic_dtn_optimization.electricity_demand_reduction
        num_last_clusters = self.config.dynamic_dtn_optimization.num_last_clusters
        
        # Create a summary DataFrame
        summary = {
            'parameter': ['Heating Demand Reduction', 'Cooling Demand Reduction', 'DHW Demand Reduction', 'Electricity Demand Reduction',
                         'Number of Last Clusters Modified', 'Sequence Changes', 'Objective Value Changes'],
            'value': [f"{heating_reduction}%", f"{cooling_reduction}%", 
                     f"{dhw_reduction}%", f"{electricity_reduction}%",
                     num_last_clusters, sequence_changes, objective_changes]
        }
        summary_df = pd.DataFrame(summary)
        
        # Save the summary
        summary_file = comparison_dir / "summary.csv"
        summary_df.to_csv(str(summary_file), index=False)
        
        # Create detailed comparison of connection sequences
        sequence_comparison = {
            'cluster_id': list(range(len(original_sequence['flat']))),
            'original_phase': original_sequence['flat'],
            'new_phase': new_sequence['flat']
        }
        sequence_df = pd.DataFrame(sequence_comparison)
        sequence_df['phase_change'] = sequence_df['new_phase'] - sequence_df['original_phase']
        
        # Save the sequence comparison
        sequence_file = comparison_dir / "sequence_comparison.csv"
        sequence_df.to_csv(str(sequence_file), index=False)
        
        # Create visualizations
        self.create_visualizations(sequence_df, comparison_dir)
        
        self.logger.info(f"Comparison results saved to {comparison_dir}")
        return summary_df
    
    def _extract_connection_sequence(self, results):
        """
        Extract the connection sequence from optimization results.
        
        Args:
            results: Optimization results
            
        Returns:
            dict: Dictionary with connection sequence information
        """
        # Get the genome (connection sequence)
        genome = results['genome']
        
        # Extract the connection sequence by phase
        connection_sequence = []
        for phase in range(1, self.config.dtn_expansion_optimization.num_phases + 1):
            clusters_in_phase = []
            for i, cluster_phase in enumerate(genome):
                if cluster_phase == phase:
                    clusters_in_phase.append(i)
            connection_sequence.append(clusters_in_phase)
        
        return {
            'by_phase': connection_sequence,
            'flat': genome
        }
    
    def _compare_sequences(self, original_sequence, new_sequence):
        """
        Compare original and new connection sequences.
        
        Args:
            original_sequence: Original connection sequence
            new_sequence: New connection sequence
            
        Returns:
            str: Description of the changes
        """
        # Count how many clusters changed phases
        changes = 0
        for i, (orig, new) in enumerate(zip(original_sequence['flat'], new_sequence['flat'])):
            if orig != new:
                changes += 1
        
        return f"{changes} clusters changed phases ({changes/len(original_sequence['flat'])*100:.1f}%)"
    
    def _compare_objectives(self, original_results, new_results):
        """
        Compare objective values between original and new results.
        
        Args:
            original_results: Original optimization results
            new_results: New optimization results
            
        Returns:
            str: Description of the changes
        """
        changes = []
        
        # Check for NPV
        if 'fitness_NPV' in original_results['metrics'] and 'fitness_NPV' in new_results['metrics']:
            original_npv = original_results['metrics']['fitness_NPV']
            new_npv = new_results['metrics']['fitness_NPV']
            percent_change = (new_npv - original_npv) / abs(original_npv) * 100
            changes.append(f"NPV: {percent_change:.2f}%")
        
        # Check for ROI
        if 'fitness_ROI' in original_results['metrics'] and 'fitness_ROI' in new_results['metrics']:
            original_roi = original_results['metrics']['fitness_ROI']
            new_roi = new_results['metrics']['fitness_ROI']
            percent_change = (new_roi - original_roi) / abs(original_roi) * 100
            changes.append(f"ROI: {percent_change:.2f}%")
        
        # Check for emissions
        if 'fitness_emissions' in original_results['metrics'] and 'fitness_emissions' in new_results['metrics']:
            original_emissions = original_results['metrics']['fitness_emissions']
            new_emissions = new_results['metrics']['fitness_emissions']
            percent_change = (new_emissions - original_emissions) / abs(original_emissions) * 100
            changes.append(f"Emissions: {percent_change:.2f}%")
        
        return ", ".join(changes) if changes else "No comparable objective values found"
    
    def create_visualizations(self, sequence_df, output_dir):
        """
        Create visualizations of the results comparison.
        
        Args:
            sequence_df: DataFrame with sequence comparison
            output_dir: Directory to save visualizations
        """
        try:
            # Set style
            sns.set(style="whitegrid")
            
            # Create a heatmap of phase changes
            plt.figure(figsize=(12, 8))
            pivot_df = sequence_df.pivot_table(
                index='cluster_id', 
                values='phase_change',
                aggfunc='first'
            ).reset_index()
            pivot_df = pivot_df.sort_values('phase_change')
            
            # Plot the heatmap
            ax = sns.heatmap(
                pivot_df[['phase_change']].T, 
                cmap='RdBu_r',
                center=0,
                cbar_kws={'label': 'Phase Change (New - Original)'}
            )
            ax.set_xticklabels(pivot_df['cluster_id'])
            ax.set_title('Phase Changes by Cluster')
            ax.set_xlabel('Cluster ID')
            ax.set_ylabel('')
            
            # Save the figure
            plt.tight_layout()
            plt.savefig(str(Path(output_dir) / 'phase_changes_heatmap.png'), dpi=300)
            plt.close()
            
            # Create a bar chart of original vs new phases
            plt.figure(figsize=(12, 8))
            
            # Sort by original phase
            sorted_df = sequence_df.sort_values(['original_phase', 'cluster_id'])
            
            # Plot
            bar_width = 0.35
            x = np.arange(len(sorted_df))
            
            plt.bar(x - bar_width/2, sorted_df['original_phase'], bar_width, label='Original Phase')
            plt.bar(x + bar_width/2, sorted_df['new_phase'], bar_width, label='New Phase')
            
            plt.xlabel('Cluster ID')
            plt.ylabel('Phase')
            plt.title('Original vs New Connection Phases')
            plt.xticks(x, sorted_df['cluster_id'], rotation=90)
            plt.legend()
            
            # Save the figure
            plt.tight_layout()
            plt.savefig(str(Path(output_dir) / 'phase_comparison_bar.png'), dpi=300)
            plt.close()
            
        except ImportError:
            self.logger.warning("Matplotlib or seaborn not available. Skipping visualizations.")

def main(config):
    """
    Run the Dynamic DTN Result Comparison module.
    
    Args:
        config: CEA Configuration object
        
    Returns:
        Summary DataFrame with comparison results
    """
    locator = cea.inputlocator.InputLocator(config.scenario)
    
    # Create and run the comparison
    comparison = DynamicDTNResultComparison(locator, config)
    summary = comparison.compare_results()
    
    print("Dynamic DTN Result Comparison completed successfully.")
    print(summary)
    
    return summary

if __name__ == '__main__':
    main(cea.config.Configuration())