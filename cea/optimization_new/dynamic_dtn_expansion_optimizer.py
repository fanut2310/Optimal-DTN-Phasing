from __future__ import annotations

import os
from pathlib import Path
import logging

from cea.optimization_new.DTN_expansion_optimization import DTNExpansionOptimizer, log

class DynamicDTNExpansionOptimizer(DTNExpansionOptimizer):
    """
    A subclass of DTNExpansionOptimizer that saves results to the dynamic DTN optimization folder
    instead of the original DTN expansion optimization folder.
    """

    def __init__(self, *args, **kwargs):
        # Extract dynamic_dtn_folder from kwargs if provided
        self.dynamic_dtn_folder = kwargs.pop('dynamic_dtn_folder', None)

        # Call the parent class constructor
        super().__init__(*args, **kwargs)

    def get_results_folder(self):
        """
        Return the folder where results should be saved.

        Returns:
            Path: Path to the results folder
        """
        if self.dynamic_dtn_folder:
            # Create the folder if it doesn't exist
            results_folder = Path(self.dynamic_dtn_folder) / "dtn_optimization_results"
            results_folder.mkdir(parents=True, exist_ok=True)
            log().info(f"Saving DTN optimization results to dedicated folder: {results_folder}")
            return results_folder
        else:
            # Fall back to the original behavior
            default_folder = Path(self.locator.get_dtn_expansion_optimization_results_folder())
            log().warning(f"No dynamic_dtn_folder specified, falling back to default folder: {default_folder}")
            return default_folder

    def save_all_evaluated_individuals(self, all_individuals, output_dir):
        """Override to save to dynamic DTN folder"""
        # Ignore the provided output_dir and use our custom folder
        custom_folder = self.get_results_folder()
        log().info(f"Saving all evaluated individuals to dedicated folder: {custom_folder}")
        return super().save_all_evaluated_individuals(all_individuals, custom_folder)

    def save_detailed_results(self, results, output_dir, return_df=False):
        """Override to save to dynamic DTN folder"""
        if return_df:
            # If we're just returning the DataFrame, use the original method
            log().debug("Returning detailed results DataFrame without saving to file")
            return super().save_detailed_results(results, output_dir, return_df=True)
        else:
            # Otherwise, save to our custom folder
            custom_folder = self.get_results_folder()
            log().info(f"Saving detailed results to dedicated folder: {custom_folder}")
            return super().save_detailed_results(results, custom_folder, return_df=False)

    def save_results(self, solution):
        """Override to save to dynamic DTN folder"""
        # Create output directory
        output_dir = self.get_results_folder()
        output_dir.mkdir(parents=True, exist_ok=True)

        # For brevity, we'll call the parent method but modify the output_dir first
        # This is a bit of a hack, but it works
        original_get_dtn_expansion_optimization_results_folder = self.locator.get_dtn_expansion_optimization_results_folder
        self.locator.get_dtn_expansion_optimization_results_folder = lambda: str(output_dir)

        try:
            result = super().save_results(solution)
        finally:
            # Restore the original method
            self.locator.get_dtn_expansion_optimization_results_folder = original_get_dtn_expansion_optimization_results_folder

        return result

    def calculate_district_emissions_new(self):
        """Override to save phase supply files to dynamic DTN folder"""
        # Create a custom directory for phase-specific supply files
        phase_files_dir = self.get_results_folder() / "phase_supply_files"
        phase_files_dir.mkdir(parents=True, exist_ok=True)

        # Temporarily patch the locator to return our custom directory
        original_get_dtn_expansion_optimization_results_folder = self.locator.get_dtn_expansion_optimization_results_folder
        self.locator.get_dtn_expansion_optimization_results_folder = lambda: str(self.get_results_folder().parent)

        try:
            result = super().calculate_district_emissions_new()
        finally:
            # Restore the original method
            self.locator.get_dtn_expansion_optimization_results_folder = original_get_dtn_expansion_optimization_results_folder

        return result
