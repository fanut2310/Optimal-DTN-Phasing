from __future__ import annotations
import logging, os, time
from pathlib import Path
import pandas as pd
import cea.config, cea.inputlocator
from cea.optimization_new.DTN_expansion_optimization import DTNExpansionOptimizer

# ──────────────────────────────────────────────────────────────────────────────
# This file has been simplified to use a direct approach for accessing demand files
# in the temp scenario folder. Instead of using a complex redirection mechanism,
# we now use a simple locator that points directly to the temp scenario folder.
# This eliminates the need for path comparison and redirection logic, making the
# code more robust and easier to maintain.
#
# IMPORTANT UPDATE (2025-07-19):
# The TempScenarioLocator class has been enhanced to override the get_total_demand
# method to ensure it returns the path to the Total_demand.csv file in the temp
# scenario folder. This is critical because the PipeLayoutGenerator class uses this
# file to calculate metrics for cluster combinations. Without this override, the
# metrics would be identical to the original metrics, despite using modified demand
# files, because the total_demand DataFrame would still be loaded from the original
# scenario folder.
#
# Additional verification has been added to ensure the correct Total_demand.csv file
# is being used, and detailed metrics comparison has been added to help diagnose
# any remaining issues.
# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────
# custom locator for temp scenario folder (direct implementation)
# ──────────────────────────────────────────────────────────────────────────────
class TempScenarioLocator(cea.inputlocator.InputLocator):
    """
    A simple InputLocator that points directly to the temp scenario folder.
    
    This locator is used to access demand files and other resources in the temp scenario
    folder without the need for complex redirection logic. It inherits from InputLocator
    and is initialized with the path to the temp scenario folder, making all file paths
    relative to this folder.
    
    This approach is simpler and more robust than the previous approach of redirecting
    file requests from the original scenario to the modified demand files.
    """

    def __init__(self, original_locator, temp_scenario_path):
        """
        Initialize with the path to the temp scenario.
        
        Parameters:
        -----------
        original_locator : InputLocator
            The original locator to inherit attributes from
        temp_scenario_path : str or Path
            Path to the temp scenario folder
        """
        # Initialize with the temp scenario path
        super().__init__(str(temp_scenario_path))
        
        # Copy attributes from original locator that might be needed
        self.original_locator = original_locator
        self.__dict__.update({k: v for k, v in original_locator.__dict__.items() 
                             if k not in ['scenario', '_scenario', '_temp_directory']})
        
        # Clear any cache
        self._demand_cache = {}
        
        lg = log()
        lg.info(f"Created TempScenarioLocator pointing to: {self.scenario}")
        
    def get_demand_results_folder(self):
        """
        Override to return the path to the demand folder in the temp scenario.
        
        This ensures that all methods that use get_demand_results_folder() will
        correctly point to the temp scenario's demand folder.
        
        Returns:
        --------
        str
            Path to the demand folder in the temp scenario
        """
        # Add debug print to trace execution
        lg = log()
        lg.info(f"DEBUG: get_demand_results_folder called in TempScenarioLocator")
        lg.info(f"DEBUG: self.scenario = {self.scenario}")
        
        # Use the temp scenario path instead of the original scenario path
        result = self._ensure_folder(self.scenario, 'outputs', 'data', 'demand')
        lg.info(f"DEBUG: Returning demand folder path: {result}")
        return result
        
    def get_total_demand(self, format='csv'):
        """
        Override to return the path to the Total_demand.csv file in the temp scenario.
        
        This ensures that the PipeLayoutGenerator uses the updated Total_demand.csv
        that reflects the modified demand files, rather than the original one.
        
        Parameters:
        -----------
        format : str, optional
            The file format (default: 'csv')
            
        Returns:
        --------
        str
            Path to the Total_demand.csv file in the temp scenario
        """
        # Add debug print to trace execution
        lg = log()
        lg.info(f"DEBUG: get_total_demand called in TempScenarioLocator with format={format}")
        
        # Construct the path directly using the scenario path
        # This avoids any issues with get_demand_results_folder not being called correctly
        total_demand_path = os.path.join(self.scenario, 'outputs', 'data', 'demand', f'Total_demand.{format}')
        lg.info(f"DEBUG: Directly constructed total_demand_path: {total_demand_path}")
        
        # Log that we're using the temp scenario's Total_demand.csv
        lg.info(f"Using Total_demand.csv from temp scenario: {total_demand_path}")
        
        # Verify that the file exists
        if not os.path.exists(total_demand_path):
            lg.warning(f"Total_demand.{format} not found in temp scenario: {total_demand_path}")
            lg.warning(f"Falling back to original Total_demand.{format}")
            fallback_path = self.original_locator.get_total_demand(format)
            lg.info(f"DEBUG: Falling back to original path: {fallback_path}")
            return fallback_path
            
        lg.info(f"DEBUG: Returning total_demand_path: {total_demand_path}")
        return total_demand_path

# ──────────────────────────────────────────────────────────────────────────────
# utility logger
# ──────────────────────────────────────────────────────────────────────────────
def log() -> logging.Logger:
    lg = logging.getLogger("cea.dynamic_dtn_rerun")
    if not lg.handlers:
        fmt = "%(asctime)s | %(levelname)5s | %(message)s"
        logging.basicConfig(level=logging.INFO, format=fmt, datefmt="%H:%M:%S")
    return lg

# ──────────────────────────────────────────────────────────────────────────────
#  subclassed optimizer  (redirects all outputs + integer-safe GA)
# ──────────────────────────────────────────────────────────────────────────────
class DynamicDTNExpansionOptimizer(DTNExpansionOptimizer):
    def __init__(self, *args, dynamic_dtn_folder: str, **kwargs):
        self.logger = log()
        self._dynamic_folder = Path(dynamic_dtn_folder)
        super().__init__(*args, **kwargs)

    # --- output routing ------------------------------------------------------
    def _results_folder(self) -> Path:
        p = self._dynamic_folder / "dtn_optimization_results"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def save_all_evaluated_individuals(self, inds, _):
        return super().save_all_evaluated_individuals(inds, self._results_folder())

    def save_detailed_results(self, rows, _, return_df=False):
        if return_df:
            return super().save_detailed_results(rows, _, True)
        return super().save_detailed_results(rows, self._results_folder())

    def save_results(self, solution):
        # patch locator so parent writes to the dynamic folder
        orig = self.locator.get_dtn_expansion_optimization_results_folder
        self.locator.get_dtn_expansion_optimization_results_folder = \
            lambda: str(self._results_folder())
        try:
            return super().save_results(solution)
        finally:
            self.locator.get_dtn_expansion_optimization_results_folder = orig

    # --- phase-supply & LCA files into dynamic folder ------------------------
    def _patch_locator(self):
        phase_dir = self._results_folder() / "phase_supply_files"
        phase_dir.mkdir(exist_ok=True)
        lca_dir   = self._results_folder() / "lca_operation"
        lca_dir.mkdir(exist_ok=True)

        loc = self.locator
        self._orig_get_res   = loc.get_dtn_expansion_optimization_results_folder
        self._orig_get_lca   = loc.get_lca_operation
        loc.get_dtn_expansion_optimization_results_folder = lambda: str(self._results_folder())
        loc.get_lca_operation = lambda: str(lca_dir / "Total_LCA_operation.csv")

    def _unpatch_locator(self):
        loc = self.locator
        loc.get_dtn_expansion_optimization_results_folder = self._orig_get_res
        loc.get_lca_operation = self._orig_get_lca

    def calculate_district_emissions_new(self):
        self._patch_locator()
        try:
            return super().calculate_district_emissions_new()
        finally:
            self._unpatch_locator()

    def calculate_emissions_for_genome(self, *args, **kw):
        self._patch_locator()
        try:
            return super().calculate_emissions_for_genome(*args, **kw)
        finally:
            self._unpatch_locator()

    # --- GA: force integer genomes & safe mutation ---------------------------
    def _setup_genetic_algorithm(self):
        super()._setup_genetic_algorithm()

        def mutate_int(ind, indpb):
            from random import random, randint
            for i in range(len(ind)):
                if random() < indpb:
                    ind[i] = int(randint(1, self.num_phases))
            return ind,

        self.toolbox.unregister("mutate")
        self.toolbox.register("mutate", mutate_int, indpb=0.15)

    def _evaluate_individual(self, ind):
        # coerce to int to avoid "float interpreted as int" crash
        ind[:] = [int(round(g)) for g in ind]

        # Set the current individual for emissions calculation
        self.current_individual = ind

        try:
            # Call the parent method
            fitness = super()._evaluate_individual(ind)

            # Ensure fitness is returned as a tuple for DEAP
            if not isinstance(fitness, tuple):
                fitness = (fitness,)

            # For single-objective mode, ensure only one fitness value
            if not self.multi_objective_mode and len(fitness) > 1:
                # Use only the first objective (NPV)
                fitness = (fitness[0],)

            return fitness

        except Exception as e:
            self.logger.error(f"Error evaluating individual {ind}: {e}")
            # Return a penalty fitness value
            return (-1000000.0,)
        finally:
            # Clear the current individual
            self.current_individual = None

# ──────────────────────────────────────────────────────────────────────────────
#  main rerun driver
# ──────────────────────────────────────────────────────────────────────────────
class DynamicDTNRerun:
    """
    Dynamic DTN Rerun Optimization class.

    This class handles the rerunning of DTN optimization with modified demands.
    It generates updated metrics based on the modified demands to ensure
    accurate optimization results that reflect the changes in demand profiles.

    Important:
    ----------
    The 'testing_clusters' parameter is critical for accurate emissions calculation.
    If not specified, emissions calculation will include ALL buildings in the scenario,
    resulting in much higher emissions values than expected. This can also lead to
    the optimization not finding valid solutions that meet all constraints.

    Always specify the same testing_clusters that were used in the initial DTN optimization:
    Example: cea dynamic-dtn-rerun-optimization --scenario YOUR_SCENARIO --network-type DH --testing-clusters 1,2,3,4,7
    """
    def __init__(self, loc: cea.inputlocator.InputLocator, cfg: cea.config.Configuration, use_updated_metrics=True):
        """
        Initialize the DynamicDTNRerun class.

        Parameters:
        -----------
        loc : cea.inputlocator.InputLocator
            CEA InputLocator object
        cfg : cea.config.Configuration
            CEA Configuration object
        use_updated_metrics : bool, optional
            Whether to generate updated metrics based on modified demands (True)
            or use the original metrics from the initial DTN optimization (False).
            Default is True.
        """
        self.locator, self.cfg, self.lg = loc, cfg, log()
        self.ntype = cfg.dynamic_dtn_optimization.network_type
        self.dyn_folder = Path(loc.get_optimization_results_folder()) / "dynamic_dtn_optimization"
        self.use_updated_metrics = use_updated_metrics

        # Get testing clusters from config if available
        self.testing_clusters = None
        if hasattr(cfg.dtn_expansion_optimization, 'testing_clusters') and cfg.dtn_expansion_optimization.testing_clusters:
            # Convert string to list of integers if needed
            if isinstance(cfg.dtn_expansion_optimization.testing_clusters, str):
                self.testing_clusters = [int(c.strip()) for c in cfg.dtn_expansion_optimization.testing_clusters.split(',') if c.strip()]
            else:
                self.testing_clusters = cfg.dtn_expansion_optimization.testing_clusters

    # ---------- helpers ------------------------------------------------------
    def _original_metrics(self) -> pd.DataFrame:
        p = Path(self.locator.get_dtn_expansion_optimization_results_folder()) \
            / "phase_1" / "clusters_metrics.csv"
        if not p.exists():
            raise FileNotFoundError(f"metrics file not found: {p}")
        self.lg.info(f"Loading original metrics from: {p}")
        return pd.read_csv(p)

    def _generate_updated_metrics(self) -> pd.DataFrame:
        """
        Generate updated metrics based on demand files in the temp scenario.
        
        This method creates a TempScenarioLocator that points directly to the temp scenario
        folder, verifies that the necessary files exist, and then uses the PipeLayoutGenerator
        to generate updated metrics based on the demand files in the temp scenario.
        
        The direct approach eliminates the need for complex redirection logic, making the
        code more robust and easier to maintain.
        
        Returns:
        --------
        pd.DataFrame
            DataFrame containing updated metrics for all cluster combinations
        """
        self.lg.info("Generating updated metrics using temp scenario demand files...")

        try:
            # Step 1: Create a path to the temp scenario folder
            # This is where all the demand files (both modified and original) are stored
            temp_scenario_path = self.dyn_folder / "temp_scenario"
            if not temp_scenario_path.exists():
                raise FileNotFoundError(f"Temp scenario directory not found: {temp_scenario_path}")
            
            # Step 2: Create necessary directories in temp scenario for optimization files
            temp_opt_dir = temp_scenario_path / "outputs" / "data" / "optimization" / "dtn_expansion"
            temp_opt_dir.mkdir(parents=True, exist_ok=True)
            self.lg.info(f"Created directory for optimization files in temp scenario: {temp_opt_dir}")
            
            # Step 3: Copy required cluster files from original scenario to temp scenario
            original_opt_dir = Path(self.locator.get_dtn_expansion_optimization_results_folder())
            
            # Copy cluster_edges.csv
            cluster_edges_path = original_opt_dir / "cluster_edges.csv"
            if os.path.exists(cluster_edges_path):
                self.lg.info(f"Copying cluster_edges.csv to temp scenario")
                import shutil
                shutil.copy2(cluster_edges_path, temp_opt_dir / "cluster_edges.csv")
            else:
                raise FileNotFoundError(f"cluster_edges.csv not found at {cluster_edges_path}")
            
            # Copy cluster_nodes.csv
            cluster_nodes_path = original_opt_dir / "cluster_nodes.csv"
            if os.path.exists(cluster_nodes_path):
                self.lg.info(f"Copying cluster_nodes.csv to temp scenario")
                shutil.copy2(cluster_nodes_path, temp_opt_dir / "cluster_nodes.csv")
            else:
                raise FileNotFoundError(f"cluster_nodes.csv not found at {cluster_nodes_path}")
            
            # Step 4: Create a TempScenarioLocator that points directly to the temp scenario
            # This locator will be used to access all files in the temp scenario folder
            # It's a simpler approach than redirecting file requests from the original scenario
            self.lg.info(f"Creating locator for temp scenario: {temp_scenario_path}")
            temp_locator = TempScenarioLocator(self.locator, temp_scenario_path)
            
            # Step 5: Verify that the necessary directories and files exist
            # This ensures that the temp scenario folder has all the required data
            
            # Check demand directory
            demand_dir = temp_scenario_path / "outputs" / "data" / "demand"
            if not demand_dir.exists():
                raise FileNotFoundError(f"Demand directory not found in temp scenario: {demand_dir}")
            
            # Check thermal network directory (warning only, not critical)
            thermal_network_dir = temp_scenario_path / "outputs" / "data" / "thermal-network"
            if not thermal_network_dir.exists():
                self.lg.warning(f"Thermal network directory not found in temp scenario: {thermal_network_dir}")
            
            # Verify that the Total_demand.csv file exists in the temp scenario
            total_demand_path = demand_dir / "Total_demand.csv"
            self.lg.info(f"Checking Total_demand.csv in temp scenario: {total_demand_path}")
            
            if not os.path.exists(total_demand_path):
                self.lg.error(f"Total_demand.csv not found in temp scenario: {total_demand_path}")
                raise FileNotFoundError(f"Total_demand.csv not found in temp scenario: {total_demand_path}")
            
            self.lg.info(f"✓ Total_demand.csv exists in temp scenario")
            
            # Load and verify the Total_demand.csv file
            self.lg.info(f"Loading Total_demand.csv from temp scenario to verify contents")
            total_demand = pd.read_csv(total_demand_path)
            
            # Check if the Total_demand.csv file has the expected columns
            required_columns = ['name']
            if self.ntype == 'DH':
                required_columns.extend(['Qhs_sys_MWhyr', 'Qww_sys_MWhyr'])
            else:  # DC
                required_columns.extend(['Qcs_sys_MWhyr', 'Qcre_sys_MWhyr', 'Qcdata_sys_MWhyr'])
                
            missing_columns = [col for col in required_columns if col not in total_demand.columns]
            if missing_columns:
                self.lg.error(f"Total_demand.csv is missing required columns: {missing_columns}")
                raise ValueError(f"Total_demand.csv is missing required columns: {missing_columns}")
            
            self.lg.info(f"✓ Total_demand.csv has all required columns")
            
            # Check a sample of building demand files to ensure they exist
            sample_buildings = total_demand['name'].tolist()[:3]  # Check first 3 buildings
            self.lg.info(f"Verifying individual building demand files for: {sample_buildings}")
            
            for building in sample_buildings:
                file_path = demand_dir / f"{building}.csv"
                self.lg.info(f"Checking demand file for building {building}: {file_path}")
                
                if os.path.exists(file_path):
                    self.lg.info(f"  ✓ File exists for {building}")
                else:
                    self.lg.error(f"  ✗ File does not exist for {building}: {file_path}")
                    raise FileNotFoundError(f"Demand file not found for {building}: {file_path}")
            
            self.lg.info(f"All sample building files verified successfully")
            
            # Verify that the temp_locator's get_total_demand method returns the correct path
            locator_total_demand_path = temp_locator.get_total_demand()
            self.lg.info(f"Path returned by temp_locator.get_total_demand(): {locator_total_demand_path}")
            
            # Check if the path is correct (should point to the temp scenario)
            if str(total_demand_path) not in str(locator_total_demand_path):
                self.lg.error(f"temp_locator.get_total_demand() is not returning the path to the temp scenario's Total_demand.csv")
                self.lg.error(f"Expected path to contain: {total_demand_path}")
                self.lg.error(f"Actual path: {locator_total_demand_path}")
                raise ValueError(f"temp_locator.get_total_demand() is not returning the correct path")
            
            self.lg.info(f"✓ temp_locator.get_total_demand() returns the correct path")

            # Step 6: Generate new metrics using the PipeLayoutGenerator
            # This will use the temp_locator to access demand files in the temp scenario
            from cea.optimization_new.DTN_expansion_optimization import PipeLayoutGenerator

            self.lg.info(f"Creating PipeLayoutGenerator with network_type={self.ntype}, testing_clusters={self.testing_clusters}")
            generator = PipeLayoutGenerator(
                locator=temp_locator,  # Use the temp scenario locator
                network_type=self.ntype,
                phase=1,
                testing_clusters=self.testing_clusters
            )

            # Generate and return the updated metrics
            self.lg.info("Generating pipe layouts with temp scenario demand files...")
            metrics_df = generator.generate_pipe_layouts()

            # Verify the metrics have actually changed
            original_metrics = self._original_metrics()
            
            # Perform a detailed comparison of the metrics
            self.lg.info("Comparing original and updated metrics:")
            
            # Check if the DataFrames are identical
            if metrics_df.equals(original_metrics):
                self.lg.warning("⚠️ Generated metrics are IDENTICAL to original metrics!")
                self.lg.warning("This suggests that temp scenario demand files are not being used properly.")
            else:
                self.lg.info("✓ Generated metrics are DIFFERENT from original metrics")
                self.lg.info("This confirms that temp scenario demand files are being used correctly.")
            
            # Compare key columns regardless of whether the DataFrames are identical
            # This helps diagnose subtle issues even if the DataFrames appear different
            if self.ntype == 'DH':
                key_column = 'total_annual_Qh_MWh'
            else:
                key_column = 'total_annual_Qc_MWh'
                
            if key_column in metrics_df.columns and key_column in original_metrics.columns:
                self.lg.info(f"Detailed comparison of {key_column} column:")
                
                # Calculate differences
                diff = metrics_df[key_column] - original_metrics[key_column]
                
                # Log summary statistics
                self.lg.info(f"  Max absolute difference: {diff.abs().max():.2f} MWh")
                self.lg.info(f"  Mean absolute difference: {diff.abs().mean():.2f} MWh")
                self.lg.info(f"  Sum of differences: {diff.sum():.2f} MWh")
                self.lg.info(f"  Percentage of values that changed: {(diff != 0).mean() * 100:.1f}%")
                
                # Log a few specific examples of differences
                if (diff != 0).any():
                    # Find the indices with the largest differences
                    largest_diff_idx = diff.abs().nlargest(3).index
                    self.lg.info("  Examples of largest differences:")
                    
                    for idx in largest_diff_idx:
                        cluster = metrics_df.iloc[idx]['clusters']
                        original_val = original_metrics.iloc[idx][key_column]
                        updated_val = metrics_df.iloc[idx][key_column]
                        change_pct = ((updated_val - original_val) / original_val * 100) if original_val != 0 else float('inf')
                        
                        self.lg.info(f"    Cluster {cluster}: {original_val:.2f} → {updated_val:.2f} MWh ({change_pct:+.1f}%)")
                else:
                    self.lg.warning("  No differences found in the values despite DataFrames not being equal")
                    self.lg.warning("  This might indicate differences in other columns or metadata")
            else:
                self.lg.warning(f"Could not compare {key_column} column - not found in both DataFrames")
                
                # Try to find common columns to compare
                common_cols = set(metrics_df.columns) & set(original_metrics.columns)
                numeric_cols = [col for col in common_cols if 
                               pd.api.types.is_numeric_dtype(metrics_df[col]) and 
                               pd.api.types.is_numeric_dtype(original_metrics[col])]
                
                if numeric_cols:
                    self.lg.info(f"Comparing alternative numeric columns: {numeric_cols}")
                    for col in numeric_cols[:3]:  # Limit to first 3 columns
                        diff = metrics_df[col] - original_metrics[col]
                        self.lg.info(f"  Column {col}: Max diff = {diff.abs().max()}, Mean diff = {diff.abs().mean()}")

            self.lg.info(f"Generated updated metrics with {len(metrics_df)} cluster combinations")

            # Save a copy of the updated metrics for reference
            output_dir = self.dyn_folder / "updated_metrics"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / "clusters_metrics_updated.csv"
            metrics_df.to_csv(output_file, index=False)
            self.lg.info(f"Saved updated metrics to {output_file}")

            return metrics_df

        except Exception as e:
            self.lg.error(f"Error generating updated metrics: {str(e)}")
            import traceback
            self.lg.error(f"Traceback: {traceback.format_exc()}")
            raise RuntimeError(f"Failed to generate updated metrics: {str(e)}")


    # ---------- core ---------------------------------------------------------
    def run(self):
        """
        Run the dynamic DTN rerun optimization with temp scenario demand files.
        
        This method creates a TempScenarioLocator that points directly to the temp scenario
        folder, verifies that the necessary files exist, and then runs the optimization
        using the demand files in the temp scenario.
        
        The direct approach eliminates the need for complex redirection logic, making the
        code more robust and easier to maintain.
        
        Returns:
        --------
        dict
            Dictionary containing the optimization results
        """
        self.lg.info("=== Rerunning DTN optimisation with temp scenario demand files ===")
        start_time = time.time()

        # Check if testing_clusters is specified and warn if not
        if not self.testing_clusters:
            self.lg.warning("No testing_clusters specified! This will cause emissions calculation to include ALL buildings.")
            self.lg.warning("This may result in much higher emissions values and optimization may not find valid solutions.")
            self.lg.warning("Specify testing_clusters parameter with the same clusters used in the initial DTN optimization.")
            self.lg.warning("Example: --testing-clusters 1,2,3,4,7")
        else:
            self.lg.info(f"Using testing_clusters: {self.testing_clusters}")

        try:
            # Step 1: Create a path to the temp scenario folder
            # This is where all the demand files (both modified and original) are stored
            temp_scenario_path = self.dyn_folder / "temp_scenario"
            if not temp_scenario_path.exists():
                raise FileNotFoundError(f"Temp scenario directory not found: {temp_scenario_path}")
            
            # Step 2: Create a TempScenarioLocator that points directly to the temp scenario
            # This locator will be used to access all files in the temp scenario folder
            # It's a simpler approach than redirecting file requests from the original scenario
            self.lg.info(f"Creating locator for temp scenario: {temp_scenario_path}")
            temp_locator = TempScenarioLocator(self.locator, temp_scenario_path)
            
            # Step 3: Create a new configuration for the optimization
            # This copies the settings from the original configuration
            self.lg.info("Creating new configuration...")
            new_cfg = cea.config.Configuration()
            new_cfg.__setstate__(self.cfg.__getstate__())

            # Handle single-objective mode by converting the objective function to a list
            if not new_cfg.dtn_expansion_optimization.multi_objective_mode:
                new_cfg.dtn_expansion_optimization.multi_objective_functions = \
                    [new_cfg.dtn_expansion_optimization.objective_function]
                self.lg.info(f"Single-objective mode: Setting multi_objective_functions to [{new_cfg.dtn_expansion_optimization.objective_function}]")

            # Step 4: Verify that the necessary directories and files exist
            # This ensures that the temp scenario folder has all the required data
            
            # Check demand directory
            demand_dir = temp_scenario_path / "outputs" / "data" / "demand"
            if not demand_dir.exists():
                raise FileNotFoundError(f"Demand directory not found in temp scenario: {demand_dir}")
            
            # Verify that the Total_demand.csv file exists in the temp scenario
            total_demand_path = demand_dir / "Total_demand.csv"
            self.lg.info(f"Checking Total_demand.csv in temp scenario: {total_demand_path}")
            
            if not os.path.exists(total_demand_path):
                self.lg.error(f"Total_demand.csv not found in temp scenario: {total_demand_path}")
                raise FileNotFoundError(f"Total_demand.csv not found in temp scenario: {total_demand_path}")
            
            self.lg.info(f"✓ Total_demand.csv exists in temp scenario")
            
            # Load and verify the Total_demand.csv file
            self.lg.info(f"Loading Total_demand.csv from temp scenario to verify contents")
            total_demand = pd.read_csv(total_demand_path)
            
            # Check if the Total_demand.csv file has the expected columns
            required_columns = ['name']
            if self.ntype == 'DH':
                required_columns.extend(['Qhs_sys_MWhyr', 'Qww_sys_MWhyr'])
            else:  # DC
                required_columns.extend(['Qcs_sys_MWhyr', 'Qcre_sys_MWhyr', 'Qcdata_sys_MWhyr'])
                
            missing_columns = [col for col in required_columns if col not in total_demand.columns]
            if missing_columns:
                self.lg.error(f"Total_demand.csv is missing required columns: {missing_columns}")
                raise ValueError(f"Total_demand.csv is missing required columns: {missing_columns}")
            
            self.lg.info(f"✓ Total_demand.csv has all required columns")
            
            # Check a sample of building demand files to ensure they exist
            sample_buildings = total_demand['name'].tolist()[:3]  # Check first 3 buildings
            self.lg.info(f"Verifying individual building demand files for: {sample_buildings}")
            
            for building in sample_buildings:
                file_path = demand_dir / f"{building}.csv"
                self.lg.info(f"Checking demand file for building {building}: {file_path}")
                
                if os.path.exists(file_path):
                    self.lg.info(f"  ✓ File exists for {building}")
                else:
                    self.lg.error(f"  ✗ File does not exist for {building}: {file_path}")
                    raise FileNotFoundError(f"Demand file not found for {building}: {file_path}")
                    
            self.lg.info(f"All sample building files verified successfully")
            
            # Verify that the temp_locator's get_total_demand method returns the correct path
            locator_total_demand_path = temp_locator.get_total_demand()
            self.lg.info(f"Path returned by temp_locator.get_total_demand(): {locator_total_demand_path}")
            
            # Check if the path is correct (should point to the temp scenario)
            if str(total_demand_path) not in str(locator_total_demand_path):
                self.lg.error(f"temp_locator.get_total_demand() is not returning the path to the temp scenario's Total_demand.csv")
                self.lg.error(f"Expected path to contain: {total_demand_path}")
                self.lg.error(f"Actual path: {locator_total_demand_path}")
                raise ValueError(f"temp_locator.get_total_demand() is not returning the correct path")
            
            self.lg.info(f"✓ temp_locator.get_total_demand() returns the correct path")

            # Step 5: Check for thermal network results
            # These are needed for the optimization but not critical for this verification
            tn_results_dir = self.dyn_folder / "thermal_network"
            if not tn_results_dir.exists():
                self.lg.warning(f"Thermal network results directory not found: {tn_results_dir}")
            else:
                self.lg.info(f"Found thermal network results in {tn_results_dir}")

            # Step 6: Determine which metrics to use based on the use_updated_metrics parameter
            # We can either generate updated metrics based on the temp scenario demand files
            # or use the original metrics from the initial DTN optimization
            if self.use_updated_metrics:
                self.lg.info("Using updated metrics based on temp scenario demand files for more accurate optimization")
                self.lg.info("This ensures that the optimization reflects the changes in demand profiles")
                metrics_df = self._generate_updated_metrics()
            else:
                self.lg.info("Using original metrics from the initial DTN optimization")
                self.lg.info("Note: This may lead to suboptimal results if demands have changed significantly")
                metrics_df = self._original_metrics()

            # Step 7: Create and run the optimizer with the temp scenario locator
            # This ensures that all file requests during optimization use the temp scenario files
            optimizer = DynamicDTNExpansionOptimizer(
                locator               = temp_locator,  # Use the temp scenario locator
                network_type          = self.ntype,
                metrics_df            = metrics_df,
                dynamic_dtn_folder    = str(self.dyn_folder),
                num_phases            = new_cfg.dtn_expansion_optimization.num_phases,
                phase_durations       = self._parse(new_cfg.dtn_expansion_optimization.phase_durations, int),
                capex_budget_per_phase= self._parse(new_cfg.dtn_expansion_optimization.capex_budget_per_phase, float),
                total_expenditure_budget_per_phase = self._parse(new_cfg.dtn_expansion_optimization.total_expenditure_budget_per_phase, float),
                interest_rate         = new_cfg.dtn_expansion_optimization.interest_rate,
                cost_model            = new_cfg.dtn_expansion_optimization.cost_model,
                objective_function    = new_cfg.dtn_expansion_optimization.objective_function,
                diversity_factor      = new_cfg.dtn_expansion_optimization.diversity_factor,
                temperature_difference_dh = new_cfg.dtn_expansion_optimization.temperature_difference_dh,
                temperature_difference_dc = new_cfg.dtn_expansion_optimization.temperature_difference_dc,
                pressure_loss_pa_per_m = new_cfg.dtn_expansion_optimization.pressure_loss_pa_per_m,
                pump_operation_hours  = new_cfg.dtn_expansion_optimization.pump_operation_hours,
                pump_efficiency       = new_cfg.dtn_expansion_optimization.pump_efficiency,
                pump_load_factor      = new_cfg.dtn_expansion_optimization.pump_load_factor,
                pump_capex_a          = new_cfg.dtn_expansion_optimization.pump_capex_a,
                pump_capex_b          = new_cfg.dtn_expansion_optimization.pump_capex_b,
                cooling_cop           = new_cfg.dtn_expansion_optimization.cooling_cop,
                ghg_budget_per_phase  = self._parse(new_cfg.dtn_expansion_optimization.ghg_budget_per_phase, float),
                multi_objective_mode  = new_cfg.dtn_expansion_optimization.multi_objective_mode,
                multi_objective_functions = new_cfg.dtn_expansion_optimization.multi_objective_functions,
                testing_clusters      = self.testing_clusters
            )

            pop = new_cfg.dtn_expansion_optimization.population_size
            gen = new_cfg.dtn_expansion_optimization.num_generations
            self.lg.info(f"GA population {pop}, generations {gen}")
            results = optimizer.optimize(population_size=pop, num_generations=gen)

            # Check if optimization was successful
            if isinstance(results, dict) and 'genome' in results:
                self.lg.info(f"Optimization completed successfully with genome: {results['genome']}")
            else:
                self.lg.warning("Optimization completed, but no valid genome found in results")

            self.lg.info("=== Dynamic rerun completed ===")
            return results

        except Exception as e:
            self.lg.error(f"Error during dynamic DTN rerun: {str(e)}")
            return {'error': str(e), 'status': 'failed'}
        finally:
            # Calculate elapsed time
            elapsed_time = time.time() - start_time
            hours, remainder = divmod(elapsed_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            time_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
            self.lg.info(f"Total runtime: {time_str}")

    # simple csv-list parser
    @staticmethod
    def _parse(s: str | None, typ=float):
        if not s or str(s).strip()=='':
            return None
        return [typ(x) for x in str(s).split(',')]

# ──────────────────────────────────────────────────────────────────────────────
def main(cfg: cea.config.Configuration):
    """
    Main entry point for the Dynamic DTN Rerun Optimization module.

    This function runs the Dynamic DTN Rerun Optimization with the specified configuration.
    It reoptimizes the DTN expansion with modified demands, generating updated metrics
    based on the modified demands to ensure accurate optimization results.

    Parameters:
    -----------
    cfg : cea.config.Configuration
        The configuration object containing all parameters for the optimization.

    Notes:
    ------
    By default, this function generates updated metrics based on the modified demands.
    To use the original metrics from the initial DTN optimization instead, set the
    'use_original_metrics' parameter to True in the command line:

    cea dynamic-dtn-rerun-optimization --scenario YOUR_SCENARIO --network-type DH --use-original-metrics

    Important:
    ----------
    The 'testing_clusters' parameter is critical for accurate emissions calculation.
    If not specified, emissions calculation will include ALL buildings in the scenario,
    resulting in much higher emissions values than expected. This can also lead to
    the optimization not finding valid solutions that meet all constraints.

    Always specify the same testing_clusters that were used in the initial DTN optimization:
    Example: cea dynamic-dtn-rerun-optimization --scenario YOUR_SCENARIO --network-type DH --testing-clusters 1,2,3,4,7

    Returns:
    --------
    dict
        The optimization results, including the optimal genome if found.
    """
    logger = log()
    logger.info("="*80)
    logger.info("Starting Dynamic DTN Rerun Optimization module")
    logger.info(f"Scenario: {cfg.scenario}")
    logger.info(f"Network type: {cfg.dynamic_dtn_optimization.network_type}")

    # Check if testing_clusters is specified in the configuration
    testing_clusters = None
    if hasattr(cfg.dtn_expansion_optimization, 'testing_clusters') and cfg.dtn_expansion_optimization.testing_clusters:
        if isinstance(cfg.dtn_expansion_optimization.testing_clusters, str):
            testing_clusters = [int(c.strip()) for c in cfg.dtn_expansion_optimization.testing_clusters.split(',') if c.strip()]
        else:
            testing_clusters = cfg.dtn_expansion_optimization.testing_clusters
        logger.info(f"Testing clusters specified: {testing_clusters}")
    else:
        logger.warning("="*80)
        logger.warning("WARNING: No testing_clusters specified in configuration!")
        logger.warning("This will cause emissions calculation to include ALL buildings in the scenario.")
        logger.warning("For accurate results, specify the same testing_clusters used in the initial DTN optimization.")
        logger.warning("Example: cea dynamic-dtn-rerun-optimization --scenario YOUR_SCENARIO --network-type DH --testing-clusters 1,2,3,4,7")
        logger.warning("="*80)

    logger.info("="*80)

    # Check if use_original_metrics is specified in the configuration
    use_original_metrics = False
    if hasattr(cfg.dynamic_dtn_optimization, 'use_original_metrics'):
        use_original_metrics = cfg.dynamic_dtn_optimization.use_original_metrics
        if use_original_metrics:
            logger.info("Using original metrics from the initial DTN optimization as specified in configuration")

    t0 = time.time()
    rerun = DynamicDTNRerun(
        cea.inputlocator.InputLocator(cfg.scenario), 
        cfg,
        use_updated_metrics=not use_original_metrics
    )
    results = rerun.run()

    # Calculate elapsed time
    elapsed_time = time.time() - t0
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    time_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"

    logger.info("="*80)

    # Check if optimization was successful
    if isinstance(results, dict) and 'genome' in results:
        logger.info(f"Dynamic DTN Rerun Optimization completed successfully in {time_str}")
        print("Dynamic DTN Rerun Optimization completed successfully.")
    elif isinstance(results, dict) and 'error' in results:
        logger.error(f"Dynamic DTN Rerun Optimization completed with errors in {time_str}")
        logger.error(f"Error: {results['error']}")
        print("Dynamic DTN Rerun Optimization completed with errors. See log for details.")
    else:
        logger.warning(f"Dynamic DTN Rerun Optimization completed with potential issues in {time_str}")
        print("Dynamic DTN Rerun Optimization completed with potential issues. See log for details.")

    logger.info(f"Results saved to: {Path(cea.inputlocator.InputLocator(cfg.scenario).get_optimization_results_folder()) / 'dynamic_dtn_optimization'}")
    logger.info("="*80)

    print(f"Total runtime: {time_str}")

    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Dynamic DTN Rerun Optimization")
    parser.add_argument('-s', '--scenario', help='Path to the scenario folder')
    parser.add_argument('-c', '--config', help='Path to the config file')
    parser.add_argument('--use-original-metrics', action='store_true', 
                        help='Use original metrics from the initial DTN optimization instead of generating updated metrics')
    parser.add_argument('--network-type', choices=['DH', 'DC'], 
                        help='Network type: DH (district heating) or DC (district cooling)')
    parser.add_argument('--testing-clusters', 
                        help='Comma-separated list of cluster IDs to include in the optimization')
    args = parser.parse_args()

    config = cea.config.Configuration(args.config)
    config.scenario = args.scenario

    # Set command line parameters in the configuration
    if args.network_type:
        config.dynamic_dtn_optimization.network_type = args.network_type

    if args.testing_clusters:
        config.dtn_expansion_optimization.testing_clusters = args.testing_clusters

    # Add use_original_metrics to the configuration
    if not hasattr(config.dynamic_dtn_optimization, 'use_original_metrics'):
        setattr(config.dynamic_dtn_optimization, 'use_original_metrics', False)

    if args.use_original_metrics:
        config.dynamic_dtn_optimization.use_original_metrics = True

    main(config)
