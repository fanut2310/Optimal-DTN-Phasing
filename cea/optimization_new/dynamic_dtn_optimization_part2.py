from __future__ import annotations

###############################################################################
# 1) STANDARD IMPORTS                                                        #
###############################################################################
import argparse
import logging
import os
import shutil
import time
import itertools
import random
import json
from pathlib import Path
from typing import Dict, Tuple, List, Set, Optional, Union
import hashlib

import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from deap import base, tools, algorithms, creator

import cea.config
import cea.inputlocator
from cea.constants import HEAT_CAPACITY_OF_WATER_JPERKGK
from cea.analysis.lca.operation import lca_operation
from cea.analysis.costs.equations import calc_capex_annualized, calc_opex_annualized


# Note: The TempScenarioLocator class has been removed and replaced with direct path methods.
# Instead of using a separate locator class to redirect file requests to the temporary scenario,
# we now use direct path methods from the InputLocator class to access files in the temporary scenario.
# This approach is more explicit, easier to debug, and prevents nested path problems.


# Setup function for the creator based on optimization mode
def setup_creator(multi_objective=False, objective_function='NPV', multi_objective_functions=None):
    """
    Set up the creator based on optimization mode and selected objectives

    Parameters:
    -----------
    multi_objective : bool
        Whether to use multi-objective optimization
    objective_function : str
        The objective function to use for single-objective optimization
    multi_objective_functions : list
        List of objective functions to use for multi-objective optimization
    """
    # Clear any existing creator attributes to avoid conflicts
    if hasattr(creator, 'FitnessMax'):
        del creator.FitnessMax
    if hasattr(creator, 'FitnessMin'):
        del creator.FitnessMin
    if hasattr(creator, 'FitnessMulti'):
        del creator.FitnessMulti
    if hasattr(creator, 'Individual'):
        del creator.Individual

    if multi_objective:
        # Build weights per objective: +1 for benefits (NPV/ROI), -1 for costs (emissions/total_capex)
        if not multi_objective_functions or len(multi_objective_functions) == 0:
            multi_objective_functions = ['NPV', 'emissions']
        multi_objective_functions = multi_objective_functions[:3]

        weights = []
        for obj in multi_objective_functions:
            obj_lower = obj.lower() if isinstance(obj, str) else obj
            if obj_lower in ('npv', 'roi'):
                weights.append(1.0)
            elif obj_lower in ('emissions', 'total_capex'):
                weights.append(-1.0)
            else:
                weights.append(1.0)

        creator.create("FitnessMulti", base.Fitness, weights=tuple(weights))
        creator.create("Individual", list, fitness=creator.FitnessMulti)
    else:
        # Single objective: minimize for emissions, maximize otherwise
        if isinstance(objective_function, str) and objective_function.lower() == 'emissions':
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
            creator.create("Individual", list, fitness=creator.FitnessMin)
        else:
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
            creator.create("Individual", list, fitness=creator.FitnessMax)

def log():
    """
    Get the logger for this module and ensure it can print to console even if the
    environment didn't pre-configure handlers (fail-safe for CLI runs).

    Returns:
    --------
    logging.Logger
        The logger for this module
    """
    lg = logging.getLogger("cea.optimization.dynamic_dtn_optimization_part2")
    # Attach a console handler once if none present (prevents silent logs)
    if not lg.handlers:
        try:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s | %(levelname)5s | %(message)s", datefmt="%H:%M:%S")
            handler.setFormatter(formatter)
            lg.addHandler(handler)
        except Exception:
            pass
    # Prevent duplicate prints via root / 'cea' handlers
    lg.propagate = False
    return lg

# I/O logging helpers for fail-fast path validation in temp scenario
from pathlib import Path as _PathForIO

def _assert_under_temp(path: _PathForIO, temp_root: _PathForIO, what: str):
    """Ensure a given path is located under the dynamic temp scenario root."""
    p = _PathForIO(path)
    tr = _PathForIO(temp_root)
    if tr not in p.parents and p != tr:
        raise AssertionError(f"{what} is not under temp scenario! Got: {p}, temp root: {tr}")


def _log_io(action: str, path: _PathForIO, level: str = "debug", extra: Optional[dict] = None):
    msg = f"[IO] {action}: {path}"
    if extra:
        tail = " ".join([f"{k}={v}" for k, v in extra.items()])
        msg = f"{msg} | {tail}"
    if level.lower() == "debug":
        log().debug(msg)
    elif level.lower() == "warning":
        log().warning(msg)
    elif level.lower() == "error":
        log().error(msg)
    else:
        log().info(msg)


def _read_shp_force_2d(path: Path):
    """
    Read a shapefile and force 2D geometry.

    Parameters:
    -----------
    path : Path
        Path to the shapefile

    Returns:
    --------
    geopandas.GeoDataFrame
        The shapefile as a GeoDataFrame with 2D geometry
    """
    gdf = gpd.read_file(path)
    gdf['geometry'] = gdf['geometry'].apply(lambda geom: geom.buffer(0))
    return gdf


def parse_args():
    """
    Parse command line arguments.

    Returns:
    --------
    argparse.Namespace
        The parsed arguments
    """
    parser = argparse.ArgumentParser(description='Run the dynamic DTN optimization part 2 script.')
    parser.add_argument('-s', '--scenario', help='Path to the scenario folder')
    parser.add_argument('-c', '--config', help='Path to the config file')
    return parser.parse_args()


def check_thermal_network_prerequisites(locator, network_type, bypass_check=False):
    """
    Check if the thermal network prerequisites are met.

    Parameters:
    -----------
    locator : cea.inputlocator.InputLocator
        The locator for the scenario
    network_type : str
        The network type (DH or DC)
    bypass_check : bool
        Whether to bypass the check

    Returns:
    --------
    bool
        True if prerequisites are met, False otherwise
    """
    if bypass_check:
        return True

    # Check if the thermal network has been created
    thermal_network_folder = Path(locator.get_thermal_network_folder())
    if not thermal_network_folder.exists():
        log().error(f"Thermal network folder not found: {thermal_network_folder}")
        return False

    # Check if the network layout exists
    network_layout_file = thermal_network_folder / f"{network_type}_network_layout.shp"
    if not network_layout_file.exists():
        log().error(f"Network layout file not found: {network_layout_file}")
        return False

    # Check if the network edges file exists
    network_edges_file = thermal_network_folder / f"{network_type}_network_edges.shp"
    if not network_edges_file.exists():
        log().error(f"Network edges file not found: {network_edges_file}")
        return False

    # Check if the network nodes file exists
    network_nodes_file = thermal_network_folder / f"{network_type}_network_nodes.shp"
    if not network_nodes_file.exists():
        log().error(f"Network nodes file not found: {network_nodes_file}")
        return False

    return True


class DTNExpansionOptimizer:
    """
    Optimize the phased expansion of district thermal networks using genetic algorithm.

    This class assigns clusters to different phases to maximize overall ROI or NPV while
    respecting budget constraints for each phase. It includes detailed cost calculations for:

    - Pipe costs based on actual pipe diameters and lengths
    - Heat exchanger costs based on building loads and temperature differences
    - Pump costs based on mass flow rates and pressure losses
    - Cooling plant costs (chillers and cooling towers) for DC networks
    """

    def __init__(self, locator: cea.inputlocator.InputLocator, network_type: str,
                 metrics_df: pd.DataFrame, num_phases: int = 3,
                 phase_durations: Optional[List[int]] = None,
                 capex_budget_per_phase: Optional[List[float]] = None,
                 total_expenditure_budget_per_phase: Optional[List[float]] = None,
                 interest_rate: float = 0.05,
                 cost_model: str = 'detailed', objective_function: str = 'NPV',
                 diversity_factor: float = 0.7,
                 temperature_difference_dh: float = 20, temperature_difference_dc: float = 10,
                 pressure_loss_pa_per_m: float = 200, pump_operation_hours: int = 4000,
                 pump_efficiency: float = 0.8, pump_load_factor: float = 0.5,
                 pump_capex_a: float = 1230, pump_capex_b: float = 0.65,
                 cooling_cop: float = 4.0, ghg_budget_per_phase: Optional[List[float]] = None,
                 multi_objective_mode: bool = False, multi_objective_functions: Optional[List[str]] = None,
                 testing_clusters=None,
                 # New parameters for Q2/Q3
                 capex_per_phase_softening_pct: float = 0.20,
                 total_exp_per_phase_softening_pct: float = 0.20,
                 enforce_final_cumulative_capex: bool = True,
                 final_cumulative_capex_budget_total: Optional[float] = None,
                 enforce_final_cumulative_total_exp: bool = True,
                 final_cumulative_total_exp_budget_total: Optional[float] = None,
                 lock_committed_early_phases: bool = True,
                 locked_phase_indices: Optional[List[int]] = None,
                 lock_reference_genome: str = 'baseline-dtn-opt',
                 custom_locked_genome: Optional[List[int]] = None,
                 # Soft per-phase ceiling (always hard-penalized on exceedance)
                 enforce_per_phase_soft_ceiling: bool = True,
                 # GA hyperparameters
                 ga_crossover_probability: float = 0.5,
                 ga_mutation_probability: float = 0.2,
                 ga_tournament_size: int = 3,
                 ga_mutation_indpb: float = 0.4,
                 ga_random_seed: int = 0):
        """
        Initialize the DTN expansion optimizer.

        Parameters:
        -----------
        locator : cea.inputlocator.InputLocator
            CEA InputLocator object
        network_type : str
            'DH' for district heating or 'DC' for district cooling
        metrics_df : pd.DataFrame
            DataFrame with metrics for all cluster combinations
        num_phases : int
            Number of phases for the expansion
        phase_durations : list, optional
            Duration in years for each phase (e.g., [3,3,3]). Must match the number of phases.
        capex_budget_per_phase : list, optional
            CAPEX budget available for each phase (in USD). Only considers capital expenditure.
        total_expenditure_budget_per_phase : list, optional
            Total expenditure budget (CAPEX + OPEX) for each phase (in USD). Considers both capital and operational expenditures.
        Note: Energy price is now automatically read from FEEDSTOCKS.xlsx
            (NATURALGAS for DH, GRID for DC)
        interest_rate : float
            Annual interest rate for NPV calculations
        cost_model : str
            Method for calculating pipe costs (only 'detailed')
        objective_function : str
            Objective function to maximize ('NPV', 'ROI') or minimize ('emissions')
        diversity_factor : float
            Typical diversity factor for district energy systems (used to convert annual demand to peak demand)
        temperature_difference_dh : float
            Temperature difference in Kelvin for district heating networks
        temperature_difference_dc : float
            Temperature difference in Kelvin for district cooling networks
        pressure_loss_pa_per_m : float
            Pressure loss in Pa/m for pipe sizing
        pump_operation_hours : int
            Annual operation hours for pumps
        pump_efficiency : float
            Pump efficiency (fraction between 0 and 1)
        pump_load_factor : float
            Average load factor of the pump during operation hours (fraction between 0 and 1)
        pump_capex_a : float
            Coefficient 'a' in pump CAPEX formula: a * (pump_power / 1000) ^ b
        pump_capex_b : float
            Exponent 'b' in pump CAPEX formula: a * (pump_power / 1000) ^ b
        cooling_cop : float
            Coefficient of Performance (COP) for cooling plants
        ghg_budget_per_phase : list, optional
            GHG emission budgets for each phase (in tonCO2)
        multi_objective_mode : bool, optional
            If True, uses multi-objective optimization with selected objectives
        multi_objective_functions : list, optional
            List of objectives to use for multi-objective optimization. Options are 'NPV', 'Discounted_ROI', 'emissions', and 'total_capex'
        testing_clusters : list, optional
            List of cluster IDs to include in the optimization (if None, all clusters are included)
        """
        self.locator = locator
        self.network_type = network_type
        self.metrics_df = metrics_df
        self.num_phases = num_phases

        # Set default phase durations if not provided
        self.phase_durations = phase_durations or [3] * num_phases

        if cost_model and cost_model.lower() != 'detailed':
            log().warning(f"Ignoring cost_model='{cost_model}'. Only 'detailed' is supported now.")
        self.cost_model = 'detailed'

        # Validate that the length of phase_durations matches num_phases
        if len(self.phase_durations) != self.num_phases:
            raise ValueError(
                f"Length of phase_durations ({len(self.phase_durations)}) must match num_phases ({self.num_phases})")

        self.capex_budget_per_phase = capex_budget_per_phase or [float('inf')] * num_phases
        self.total_expenditure_budget_per_phase = total_expenditure_budget_per_phase or [float('inf')] * num_phases
        self.energy_price = self._get_energy_price()
        self.interest_rate = interest_rate
        self.cost_model = cost_model
        self.objective_function = objective_function
        self.diversity_factor = diversity_factor
        self.temperature_difference_dh = temperature_difference_dh
        self.temperature_difference_dc = temperature_difference_dc
        self.pressure_loss_pa_per_m = pressure_loss_pa_per_m
        self.pump_operation_hours = pump_operation_hours
        self.pump_efficiency = pump_efficiency
        self.pump_load_factor = pump_load_factor
        self.pump_capex_a = pump_capex_a
        self.pump_capex_b = pump_capex_b
        self.cooling_cop = cooling_cop
        self.ghg_budget_per_phase = ghg_budget_per_phase
        self.multi_objective_mode = multi_objective_mode
        self.multi_objective_functions = multi_objective_functions
        self.testing_clusters = testing_clusters

        # ---------------- New Q2/Q3 parameters ----------------
        self.capex_per_phase_softening_pct = float(capex_per_phase_softening_pct if capex_per_phase_softening_pct is not None else 0.20)
        self.total_exp_per_phase_softening_pct = float(total_exp_per_phase_softening_pct if total_exp_per_phase_softening_pct is not None else 0.20)
        self.enforce_final_cumulative_capex = bool(enforce_final_cumulative_capex)
        self.enforce_final_cumulative_total_exp = bool(enforce_final_cumulative_total_exp)
        # Derive final cumulative totals if not provided (0 or None => auto-sum of original budgets)
        try:
            self.final_cumulative_capex_budget_total = float(final_cumulative_capex_budget_total) if final_cumulative_capex_budget_total not in (None, "", []) else None
        except Exception:
            self.final_cumulative_capex_budget_total = None
        try:
            self.final_cumulative_total_exp_budget_total = float(final_cumulative_total_exp_budget_total) if final_cumulative_total_exp_budget_total not in (None, "", []) else None
        except Exception:
            self.final_cumulative_total_exp_budget_total = None
        # Auto-compute totals from per-phase arrays if needed and available
        if self.final_cumulative_capex_budget_total in (None, 0) and self.capex_budget_per_phase and all(np.isfinite(x) for x in self.capex_budget_per_phase if x is not None):
            try:
                self.final_cumulative_capex_budget_total = float(sum([x for x in self.capex_budget_per_phase if x not in (None, float('inf'))]))
            except Exception:
                self.final_cumulative_capex_budget_total = None
        if self.final_cumulative_total_exp_budget_total in (None, 0) and self.total_expenditure_budget_per_phase and all(np.isfinite(x) for x in self.total_expenditure_budget_per_phase if x is not None):
            try:
                self.final_cumulative_total_exp_budget_total = float(sum([x for x in self.total_expenditure_budget_per_phase if x not in (None, float('inf'))]))
            except Exception:
                self.final_cumulative_total_exp_budget_total = None
        # Compute soft per-phase arrays for reporting (not enforced)
        try:
            self.capex_soft_per_phase = [ (1.0 + self.capex_per_phase_softening_pct) * b if (b is not None and np.isfinite(b)) else b for b in (self.capex_budget_per_phase or []) ]
        except Exception:
            self.capex_soft_per_phase = self.capex_budget_per_phase
        try:
            self.total_exp_soft_per_phase = [ (1.0 + self.total_exp_per_phase_softening_pct) * b if (b is not None and np.isfinite(b)) else b for b in (self.total_expenditure_budget_per_phase or []) ]
        except Exception:
            self.total_exp_soft_per_phase = self.total_expenditure_budget_per_phase
        # Locking parameters (build map later after clusters are loaded)
        self.lock_committed_early_phases = bool(lock_committed_early_phases)
        self.lock_reference_genome = str(lock_reference_genome) if lock_reference_genome is not None else 'baseline-dtn-opt'
        self._locked_phase_indices_cfg = list(locked_phase_indices) if locked_phase_indices else [1]
        self._custom_locked_genome_cfg = list(custom_locked_genome) if custom_locked_genome else []
        self.locked_clusters_phase: Dict[int, int] = {}
        # Per-phase ceiling: always treated as hard if enabled
        self.enforce_per_phase_soft_ceiling = bool(enforce_per_phase_soft_ceiling)
        # ------------------------------------------------------

        # ---------------- GA hyperparameters ----------------
        try:
            self.ga_crossover_probability = float(ga_crossover_probability if ga_crossover_probability is not None else 0.5)
            self.ga_mutation_probability = float(ga_mutation_probability if ga_mutation_probability is not None else 0.2)
            self.ga_tournament_size = int(ga_tournament_size if ga_tournament_size is not None else 3)
            self.ga_mutation_indpb = float(ga_mutation_indpb if ga_mutation_indpb is not None else 0.4)
            self.ga_random_seed = int(ga_random_seed if ga_random_seed is not None else 0)
        except Exception:
            self.ga_crossover_probability = 0.5
            self.ga_mutation_probability = 0.2
            self.ga_tournament_size = 3
            self.ga_mutation_indpb = 0.4
            self.ga_random_seed = 0
        # Apply RNG seed if provided (>0)
        try:
            if int(self.ga_random_seed) > 0:
                random.seed(int(self.ga_random_seed))
                np.random.seed(int(self.ga_random_seed))
                log().info(f"GA RNG seeded with {self.ga_random_seed}")
        except Exception:
            pass
        # ------------------------------------------------------

        # Set up the creator based on optimization mode and selected objectives
        setup_creator(multi_objective_mode, objective_function, multi_objective_functions)

        # Load cost data from TN part 3 results
        self.cost_data = self._load_cost_data()

        # Load cluster data
        self._load_cluster_data()

        # Build lock map now that clusters are known
        self._build_lock_map()

        # Create a mapping from cluster combinations to their metrics
        self.cluster_metrics = self._create_cluster_metrics_mapping()

        # Create a cache for emissions results
        self.emissions_cache = {}

        # Emissions computation uses LCA Operation only (COP-based path removed)
        try:
            log().info("COP-based emissions path removed; LCA Operation is the sole emissions method.")
        except Exception:
            pass

        # Initialize DEAP toolbox
        self.toolbox = base.Toolbox()
        self._setup_genetic_algorithm()

    def _build_lock_map(self):
        """Build mapping of cluster->locked phase from configured reference genome after clusters are known."""
        self.locked_clusters_phase = {}
        if not getattr(self, 'lock_committed_early_phases', False):
            return
        locked_phases = set(self._locked_phase_indices_cfg or [1])
        genome_source = (self.lock_reference_genome or 'baseline-dtn-opt').lower()
        baseline_genome: List[int] = []
        try:
            if genome_source == 'custom' and self._custom_locked_genome_cfg:
                baseline_genome = [int(x) for x in self._custom_locked_genome_cfg]
            else:
                baseline_csv = Path(self.locator.get_dtn_expansion_optimization_results_folder()) / f"all_evaluated_individuals_{self.network_type}.csv"
                if baseline_csv.exists():
                    df = pd.read_csv(baseline_csv)
                    if 'genome' in df.columns:
                        if 'fitness_NPV' in df.columns:
                            best_row = df.loc[df['fitness_NPV'].idxmax()]
                        elif 'fitness_ROI' in df.columns:
                            best_row = df.loc[df['fitness_ROI'].idxmax()]
                        else:
                            best_row = df.iloc[0]
                        g = best_row['genome']
                        if isinstance(g, list):
                            baseline_genome = [int(x) for x in g]
                        else:
                            try:
                                baseline_genome = [int(x) for x in str(g).replace('[','').replace(']','').split(',') if str(x).strip() != '']
                            except Exception:
                                baseline_genome = []
                else:
                    log().warning(f"Baseline optimization results not found at {baseline_csv}; locking disabled.")
        except Exception as e:
            log().warning(f"Failed to build lock map from reference genome: {e}")
            baseline_genome = []
        if baseline_genome and len(baseline_genome) == len(self.all_clusters):
            for cluster, phase in zip(self.all_clusters, baseline_genome):
                if int(phase) in locked_phases:
                    self.locked_clusters_phase[int(cluster)] = int(phase)
        elif baseline_genome:
            log().warning("Reference genome length mismatch; locking disabled.")

    def _get_energy_price(self):
        """
        Read energy price from FEEDSTOCKS.xlsx based on network type.

        For DH: Uses the buy price from appropriate heating feedstock (default: NATURALGAS)
        For DC: Uses the buy price from GRID (electricity for cooling)

        Returns:
        --------
        float
            Energy price in USD/kWh
        """
        try:
            # Determine which feedstock to use based on network type
            if self.network_type == 'DH':
                # For district heating, try to use NATURALGAS first
                feedstock_name = 'NATURALGAS'
            else:
                # For district cooling, use GRID (electricity)
                feedstock_name = 'GRID'

            # Try to load the feedstock data
            feedstock_file = self.locator.get_db4_components_feedstocks_feedstocks_csv(feedstocks=feedstock_name)
            log().debug(f"Reading energy price from {feedstock_file}")

            feedstock_data = pd.read_csv(feedstock_file)

            # Get the buy price column (Opex_var_buy_USD2015kWh)
            if 'Opex_var_buy_USD2015kWh' in feedstock_data.columns:
                # Calculate average price across all hours
                energy_price = feedstock_data['Opex_var_buy_USD2015kWh'].mean()
                log().debug(f"Using energy price from {feedstock_name}: {energy_price:.4f} USD/kWh")
                return energy_price
            else:
                log().warning(
                    f"Column 'Opex_var_buy_USD2015kWh' not found in {feedstock_name} data. Using default value.")
        except Exception as e:
            log().warning(f"Could not read energy price from feedstock data: {e}")

        # Fallback to default values
        if self.network_type == 'DH':
            default_price = 0.08  # Default price for heating (USD/kWh)
        else:
            default_price = 0.12  # Default price for cooling (USD/kWh)

        log().warning(f"Using default energy price for {self.network_type}: {default_price} USD/kWh")
        return default_price

    def _load_cost_data(self):
        """Load cost data from thermal network costs results."""
        cost_file = Path(self.locator.get_dynamic_dtn_network_layout_costs_file(self.network_type))
        if not cost_file.exists():
            raise FileNotFoundError(f"Cost file not found: {cost_file}. Please run Thermal Network Part 3 first.")

        cost_data = pd.read_csv(cost_file)

        # Extract relevant cost components
        cost_components = {
            'capex_network_USD': cost_data['capex_network_USD'].iloc[0],
            'capex_pumps_USD': cost_data['capex_pumps_USD'].iloc[0],
            'capex_hex_USD': cost_data['capex_hex_USD'].iloc[0],
            'network_length_m': cost_data['network_length_m'].iloc[0],
        }

        return cost_components

    def _load_cluster_data(self):
        """Load cluster data from cluster_edges.csv and cluster_nodes.csv."""
        # Load cluster assignments with fail-fast checks
        temp_root = Path(self.locator.get_dynamic_dtn_optimization_temp_scenario_folder())

        # Prefer baseline DTN mapping for consistency (authoritative cluster mapping)
        baseline_edges_path = Path(self.locator.get_dtn_expansion_optimization_results_folder()) / "cluster_edges.csv"
        baseline_nodes_path = Path(self.locator.get_dtn_cluster_nodes_file())

        if baseline_edges_path.exists():
            _log_io("READ cluster_edges (baseline)", baseline_edges_path)
            self.cluster_edges = pd.read_csv(baseline_edges_path)
        else:
            cluster_edges_path = Path(self.locator.get_dynamic_dtn_optimization_temp_scenario_dtn_expansion_folder()) / "cluster_edges.csv"
            _log_io("READ cluster_edges (temp)", cluster_edges_path)
            _assert_under_temp(cluster_edges_path, temp_root, "cluster_edges")
            if not cluster_edges_path.exists():
                raise FileNotFoundError(f"Missing required temp file: cluster_edges.csv at {cluster_edges_path}")
            self.cluster_edges = pd.read_csv(cluster_edges_path)

        if baseline_nodes_path.exists():
            _log_io("READ cluster_nodes (baseline)", baseline_nodes_path)
            self.cluster_nodes = pd.read_csv(baseline_nodes_path)
        else:
            cluster_nodes_path = Path(self.locator.get_dynamic_dtn_optimization_temp_scenario_dtn_expansion_folder()) / "cluster_nodes.csv"
            _log_io("READ cluster_nodes (temp)", cluster_nodes_path)
            _assert_under_temp(cluster_nodes_path, temp_root, "cluster_nodes")
            if not cluster_nodes_path.exists():
                raise FileNotFoundError(f"Missing required temp file: cluster_nodes.csv at {cluster_nodes_path}")
            self.cluster_nodes = pd.read_csv(cluster_nodes_path)

        # Build authoritative building universe U from baseline mapping (CONSUMER only if available)
        try:
            nodes_df = self.cluster_nodes.copy()
            if 'type' in nodes_df.columns:
                nodes_df = nodes_df[nodes_df['type'] == 'CONSUMER']
            self._universe_U_names = sorted(nodes_df['building'].astype(str).unique().tolist())
        except Exception:
            self._universe_U_names = []

        # Load total demand
        total_demand_path = Path(self.locator.get_dynamic_dtn_optimization_temp_scenario_total_demand())
        _log_io("READ temp Total_demand", total_demand_path)
        _assert_under_temp(total_demand_path, temp_root, "temp Total_demand")
        if not total_demand_path.exists():
            raise FileNotFoundError(f"Temp Total_demand.csv missing at {total_demand_path}")
        self.total_demand = pd.read_csv(total_demand_path)

        # If testing_clusters is specified, use only those clusters
        if self.testing_clusters:
            log().info(f"Filtering to include only testing clusters: {self.testing_clusters}")
            self.all_clusters = sorted(self.testing_clusters)
        elif self.metrics_df is not None:
            # Extract unique clusters from the metrics DataFrame
            unique_clusters = set()
            for cluster_str in self.metrics_df['clusters']:
                # Skip cluster 0 (existing DTN)
                if cluster_str == '0':
                    continue
                # Split the cluster string (e.g., '1+3+4') into individual clusters
                for c in cluster_str.split('+'):
                    if c != '0':  # Skip cluster 0
                        unique_clusters.add(int(c))

            # Sort the clusters to maintain the same order as before
            self.all_clusters = sorted(list(unique_clusters))
        else:
            # Fallback: Extract unique clusters from cluster_nodes
            log().info("No metrics_df provided, extracting clusters from cluster_nodes")
            all_clusters = sorted([c for c in self.cluster_nodes['cluster'].unique() if c > 0])
            self.all_clusters = all_clusters

        log().info(f"Using {len(self.all_clusters)} clusters: {self.all_clusters}")

        # Get total number of buildings
        self.total_buildings = len(self.total_demand['name'].unique())

        # Get buildings in testing clusters (including cluster 0)
        if self.testing_clusters:
            self.buildings_in_testing_clusters = []
            # Always include cluster 0 (existing DTN)
            cluster0_buildings = self._get_buildings_in_specific_cluster(0)
            self.buildings_in_testing_clusters.extend(cluster0_buildings)

            # Add buildings from testing clusters
            for cluster in self.testing_clusters:
                buildings = self._get_buildings_in_specific_cluster(cluster)
                self.buildings_in_testing_clusters.extend(buildings)

            # Remove duplicates
            self.buildings_in_testing_clusters = list(set(self.buildings_in_testing_clusters))
            log().info(
                f"Filtered to {len(self.buildings_in_testing_clusters)} buildings out of {self.total_buildings} total buildings")

    def _create_cluster_metrics_mapping(self):
        """Create a mapping from cluster combinations to their metrics."""
        cluster_metrics = {}

        if self.metrics_df is not None:
            for _, row in self.metrics_df.iterrows():
                key = row['clusters']
                cluster_metrics[key] = row.to_dict()

                # also register an alias without the leading "0+"
                if key.startswith('0+'):
                    alias = key[2:]  # e.g. "0+2+4" → "2+4"
                    cluster_metrics[alias] = row.to_dict()
        else:
            log().warning("No metrics_df provided, cluster_metrics will be empty")

        return cluster_metrics

    def get_required_pipes_for_clusters(self, cluster_set):
        """
        Determine the minimum required pipes for a set of clusters.

        Parameters:
        -----------
        cluster_set : tuple or list
            Tuple or list of cluster IDs to connect

        Returns:
        --------
        pd.DataFrame
            DataFrame of required pipes
        """
        # Include cluster 0 (existing DTN) in the set
        clusters_to_connect = set(cluster_set).union({0})

        # Ensure cluster, from_C, and to_C columns are numeric for comparison
        try:
            # Make a copy to avoid SettingWithCopyWarning
            cluster_edges_df = self.cluster_edges.copy()
            # Convert columns to numeric, errors='coerce' will convert non-numeric values to NaN
            for col in ['cluster', 'from_C', 'to_C']:
                if col in cluster_edges_df.columns:
                    cluster_edges_df[col] = pd.to_numeric(cluster_edges_df[col], errors='coerce')
                    # Fill NaN with a value that won't match our filters
                    cluster_edges_df[col] = cluster_edges_df[col].fillna(-999)
        except Exception as e:
            log().warning(f"Error converting columns to numeric: {e}. Using original dataframe.")
            cluster_edges_df = self.cluster_edges

        # Get edges that belong to the clusters in the set
        cluster_edges = cluster_edges_df[cluster_edges_df['cluster'].isin(clusters_to_connect)]

        # Get main road edges (-1) that connect the clusters
        main_road_edges = cluster_edges_df[
            (cluster_edges_df['cluster'] == -1) &
            (cluster_edges_df['from_C'].isin(clusters_to_connect)) &
            (cluster_edges_df['to_C'].isin(clusters_to_connect))
            ]

        # Combine the edges with error handling
        try:
            # First, try to concatenate with default settings
            required_pipes = pd.concat([cluster_edges, main_road_edges])
        except Exception as e:
            log().warning(f"Error during dataframe concatenation: {e}")

            # If that fails, try with more explicit settings
            try:
                # Reset index to avoid index-related issues
                cluster_edges_reset = cluster_edges.reset_index(drop=True)
                main_road_edges_reset = main_road_edges.reset_index(drop=True)

                # Try concatenation with ignore_index=True
                required_pipes = pd.concat([cluster_edges_reset, main_road_edges_reset], ignore_index=True)
            except Exception as e2:
                log().error(f"Failed to concatenate dataframes even with reset_index: {e2}")

                # As a last resort, if one of the dataframes is empty, return the other
                if len(cluster_edges) == 0:
                    required_pipes = main_road_edges
                elif len(main_road_edges) == 0:
                    required_pipes = cluster_edges
                else:
                    # If both have data but can't be concatenated, try to create a new dataframe with common columns
                    common_columns = set(cluster_edges.columns).intersection(set(main_road_edges.columns))
                    if common_columns:
                        log().warning(f"Using only common columns for concatenation: {common_columns}")
                        required_pipes = pd.concat([
                            cluster_edges[list(common_columns)],
                            main_road_edges[list(common_columns)]
                        ], ignore_index=True)
                    else:
                        # If no solution works, raise an error
                        raise ValueError("Cannot concatenate dataframes - no common columns found")

        return required_pipes

    def _get_buildings_in_clusters(self, cluster_set):
        """
        Get all buildings in the specified clusters.

        Parameters:
        -----------
        cluster_set : tuple or list
            Tuple or list of cluster IDs

        Returns:
        --------
        list
            List of building names
        """
        # Include cluster 0 (existing DTN) in the set
        clusters_to_connect = set(cluster_set).union({0})

        # Get buildings in the clusters
        df_nodes = self.cluster_nodes
        if 'type' in df_nodes.columns:
            mask = (df_nodes['cluster'].isin(clusters_to_connect)) & (df_nodes['type'] == 'CONSUMER')
        else:
            mask = (df_nodes['cluster'].isin(clusters_to_connect))
        buildings = df_nodes.loc[mask, 'building'].tolist()

        return buildings

    def _get_buildings_in_specific_cluster(self, cluster_id):
        """
        Get buildings in a specific cluster only.

        Parameters:
        -----------
        cluster_id : int
            Cluster ID

        Returns:
        --------
        list
            List of building names in the specific cluster
        """
        # Get buildings in the specific cluster
        buildings = self.cluster_nodes[
            (self.cluster_nodes['cluster'] == cluster_id) &
            (self.cluster_nodes['type'] == 'CONSUMER')
            ]['building'].tolist()

        # Add debug logging
        log().debug(f"Cluster {cluster_id} consists of buildings: {buildings}")

        return buildings

    def calculate_detailed_capex(self, cluster_set):
        """
        Calculate CAPEX using detailed pipe diameter information.

        Parameters:
        -----------
        cluster_set : tuple or list
            Tuple or list of cluster IDs

        Returns:
        --------
        float
            Total pipe CAPEX
        """
        # Get edges for this cluster set
        required_pipes = self.get_required_pipes_for_clusters(cluster_set)

        try:
            # Load pipe cost data
            piping_cost_data = pd.read_csv(
                self.locator.get_database_components_distribution_thermal_grid('THERMAL_GRID'))

            # Merge with cost data
            cost_df = required_pipes.merge(piping_cost_data, on='pipe_DN')

            # Calculate cost for each pipe segment
            cost_df['pipe_cost'] = cost_df['Inv_USD2015perm'] * cost_df['length_m']

            # Sum up all pipe costs
            total_pipe_cost = cost_df['pipe_cost'].sum()

            return total_pipe_cost
        except Exception as e:
            raise RuntimeError(f"Detailed CAPEX calculation failed: {e}") from e

    def calculate_pump_costs(self, cluster_set):
        """
        Calculate pump costs using the detailed model with configurable parameters.

        Parameters:
        -----------
        cluster_set : tuple or list
            Tuple or list of cluster IDs

        Returns:
        --------
        tuple
            (pump_capex, annual_pump_electricity) - Total pump CAPEX and annual electricity consumption
        """
        # Get required pipes for this set of clusters
        required_pipes = self.get_required_pipes_for_clusters(cluster_set)

        # Calculate total pipe length
        total_pipe_length = required_pipes['length_m'].sum()

        # Get temperature difference based on network type
        if self.network_type == 'DH':
            temp_diff = self.temperature_difference_dh
        else:
            temp_diff = self.temperature_difference_dc

        # Calculate mass flow rate (kg/s) based on thermal demand
        key = '+'.join(map(str, sorted(cluster_set)))
        if key in self.cluster_metrics:
            metrics = self.cluster_metrics[key]
            if self.network_type == 'DH':
                annual_demand_mwh = metrics.get('total_annual_Qh_MWh', 0)
            else:
                annual_demand_mwh = metrics.get('total_annual_Qc_MWh', 0)

            # Convert annual demand to peak demand (kW) using the diversity factor
            peak_demand_kw = annual_demand_mwh * 1000 / 2000 / self.diversity_factor  # Assuming 2000 equivalent full load hours

            # Calculate mass flow rate (kg/s)
            mass_flow_rate = peak_demand_kw / (HEAT_CAPACITY_OF_WATER_JPERKGK * temp_diff / 1000)
        else:
            # Fallback if metrics not available
            mass_flow_rate = 0

        # Calculate pressure loss
        pressure_loss = self.pressure_loss_pa_per_m * total_pipe_length

        # Calculate pump power (W)
        if mass_flow_rate > 0:
            pump_power = mass_flow_rate * pressure_loss / (self.pump_efficiency * 1000)  # W
        else:
            pump_power = 0

        # Calculate pump CAPEX based on pump power using configurable formula
        pump_capex = self.pump_capex_a * (pump_power / 1000) ** self.pump_capex_b  # USD

        # Calculate annual pump electricity consumption
        annual_pump_electricity = pump_power * self.pump_operation_hours * self.pump_load_factor  # Wh (pump load factor for operation hours)

        return pump_capex, annual_pump_electricity

    def calculate_cooling_plant_costs(self, cluster_set):
        """
        Calculate cooling plant costs for a cluster set (only for DC networks).

        Parameters:
        -----------
        cluster_set : tuple or list
            Tuple or list of cluster IDs

        Returns:
        --------
        tuple
            (cooling_plant_capex, annual_cooling_plant_electricity) - Total cooling plant CAPEX and annual electricity consumption
        """
        if self.network_type != 'DC':
            return 0, 0  # Only applicable for DC networks

        # Get metrics for this cluster set
        key = '+'.join(map(str, sorted(cluster_set)))
        if key not in self.cluster_metrics:
            return 0, 0  # No metrics available

        metrics = self.cluster_metrics[key]

        # Get annual cooling demand
        annual_demand_mwh = metrics.get('total_annual_Qc_MWh', 0)
        annual_demand_kwh = annual_demand_mwh * 1000  # Convert to kWh

        # Convert annual demand to peak demand (kW) using the diversity factor
        peak_demand_kw = annual_demand_kwh / 2000 / self.diversity_factor  # Assuming 2000 equivalent full load hours
        peak_demand_w = peak_demand_kw * 1000  # Convert to W

        # Calculate cooling plant CAPEX based on typical costs for chillers and cooling towers
        if peak_demand_w > 0:
            # Chiller costs (USD)
            chiller_capex = 750 * (peak_demand_w / 1000) ** 0.85

            # Cooling tower costs (USD)
            ct_capex = 310 * (peak_demand_w / 1000) ** 0.65

            # Total cooling plant CAPEX
            cooling_plant_capex = chiller_capex + ct_capex

            # Calculate annual electricity consumption
            annual_cooling_plant_electricity = annual_demand_kwh / self.cooling_cop  # kWh
        else:
            cooling_plant_capex = 0
            annual_cooling_plant_electricity = 0

        return cooling_plant_capex, annual_cooling_plant_electricity

    def calculate_hex_costs(self, cluster_set):
        """
        Calculate heat exchanger costs for a cluster set using a detailed approach.

        Parameters:
        -----------
        cluster_set : tuple or list
            Tuple or list of cluster IDs

        Returns:
        --------
        float
            Total heat exchanger CAPEX
        """
        # Get buildings in the clusters
        buildings = self._get_buildings_in_clusters(cluster_set)

        if not buildings:
            return 0

        # Load HEX cost parameters from database
        try:
            HEX_prices = pd.read_csv(
                self.locator.get_db4_components_conversion_conversion_technology_csv('HEAT_EXCHANGERS'),
                index_col=0
            )
            a = HEX_prices['a']['District substation heat exchanger']
            b = HEX_prices['b']['District substation heat exchanger']
            c = HEX_prices['c']['District substation heat exchanger']
            d = HEX_prices['d']['District substation heat exchanger']
            e = HEX_prices['e']['District substation heat exchanger']
            Inv_IR = HEX_prices['IR_%']['District substation heat exchanger']
            Inv_LT = HEX_prices['LT_yr']['District substation heat exchanger']
        except Exception as ex:
            raise RuntimeError(f"Detailed HEX costs calculation failed: {ex}") from ex

        # Get metrics for this cluster set
        key = '+'.join(map(str, sorted(cluster_set)))
        if key not in self.cluster_metrics:
            # Fallback to simplified approach if metrics not available
            return len(buildings) * (self.cost_data['capex_hex_USD'] / self.total_buildings)

        metrics = self.cluster_metrics[key]

        # Get annual demand
        if self.network_type == 'DH':
            annual_demand_mwh = metrics.get('total_annual_Qh_MWh', 0)
        else:
            annual_demand_mwh = metrics.get('total_annual_Qc_MWh', 0)

        # Convert annual demand to peak demand (kW) using the diversity factor
        peak_demand_kw = annual_demand_mwh * 1000 / 2000 / self.diversity_factor  # Assuming 2000 equivalent full load hours

        # Get temperature difference based on network type
        if self.network_type == 'DH':
            temp_diff = self.temperature_difference_dh
        else:
            temp_diff = self.temperature_difference_dc

        # Calculate mass flow rate (kg/s)
        from cea.constants import HEAT_CAPACITY_OF_WATER_JPERKGK
        from cea.technologies.constants import MAX_NODE_FLOW

        mass_flow_rate = peak_demand_kw / (HEAT_CAPACITY_OF_WATER_JPERKGK * temp_diff / 1000)

        # Calculate HEX costs
        total_hex_capex = 0

        # Distribute mass flow rate among buildings proportionally to their number
        # This is a simplification; in a real implementation, you would use building-specific loads
        mass_flow_per_building = mass_flow_rate / len(buildings)

        for _ in buildings:
            # Split into several HEXs if flows are too high
            if mass_flow_per_building <= MAX_NODE_FLOW:
                mcp_sub = mass_flow_per_building * HEAT_CAPACITY_OF_WATER_JPERKGK
                hex_capex = a + b * mcp_sub ** c + d * np.log(mcp_sub) + e * mcp_sub * np.log(mcp_sub)
            else:
                # We need to split into several HEXs
                hex_capex = 0
                number_of_HEXs = int(np.ceil(mass_flow_per_building / MAX_NODE_FLOW))
                nodeflow_nom = mass_flow_per_building / number_of_HEXs
                mcp_sub = nodeflow_nom * HEAT_CAPACITY_OF_WATER_JPERKGK
                for i in range(number_of_HEXs):
                    hex_capex += (a + b * mcp_sub ** c + d * np.log(mcp_sub) + e * mcp_sub * np.log(mcp_sub))

            total_hex_capex += hex_capex

        return total_hex_capex

    def _calculate_phase_capex(self, cluster_set, phase=None):
        """
        Calculate total CAPEX for a phase with interest rate adjustment.

        Parameters:
        -----------
        cluster_set : tuple or list
            Tuple or list of cluster IDs
        phase : int, optional
            Phase number (1-based). If provided, applies interest rate adjustment.

        Returns:
        --------
        tuple
            (float, dict) - Total CAPEX for the phase with interest rate adjustment and a dictionary with component costs
        """
        # Calculate base CAPEX
        pipe_capex = self.calculate_detailed_capex(cluster_set)
        hex_capex = self.calculate_hex_costs(cluster_set)
        pump_capex, _ = self.calculate_pump_costs(cluster_set)
        cooling_plant_capex = 0
        if self.network_type == 'DC':
            cooling_plant_capex, _ = self.calculate_cooling_plant_costs(cluster_set)

        # Calculate total base CAPEX
        total_base_capex = pipe_capex + hex_capex + pump_capex + cooling_plant_capex

        # Apply interest rate adjustment based on phase
        discount_factor = 1.0
        if phase is not None:
            year_offset = 0
            for p in range(1, phase):
                year_offset += self.phase_durations[p - 1]

            # Discount factor based on when the phase starts
            discount_factor = 1 / ((1 + self.interest_rate) ** year_offset)

        # Apply discount factor to get present value
        total_capex = total_base_capex * discount_factor

        return total_capex, {
            'pipe_capex': pipe_capex * discount_factor,
            'hex_capex': hex_capex * discount_factor,
            'pump_capex': pump_capex * discount_factor,
            'cooling_plant_capex': cooling_plant_capex * discount_factor,
            'discount_factor': discount_factor
        }

    def _calculate_phase_total_expenditure(self, cluster_set, phase):
        """
        Calculate total expenditure (CAPEX + OPEX) for a phase.

        Parameters:
        -----------
        cluster_set : tuple or list
            Tuple or list of cluster IDs
        phase : int
            Phase number (1-based)

        Returns:
        --------
        float
            Total expenditure for the phase (CAPEX + OPEX across all years)
        """
        # --- phase-0 has no duration: only the (discounted) CAPEX ------------
        if phase == 0:
            capex, _ = self._calculate_phase_capex(cluster_set, phase=None)
            return capex
        # ---------------------------------------------------------------------

        # Calculate CAPEX with interest rate adjustment
        capex, _ = self._calculate_phase_capex(cluster_set, phase)

        # Get metrics for this cluster set
        key = '+'.join(map(str, sorted(cluster_set)))
        if key not in self.cluster_metrics:
            return capex  # If no metrics, return just CAPEX

        metrics = self.cluster_metrics[key]

        # Calculate annual revenue (energy price * annual demand)
        if self.network_type == 'DH':
            annual_demand_kwh = metrics['total_annual_Qh_MWh'] * 1000  # Convert MWh to kWh
        else:
            annual_demand_kwh = metrics['total_annual_Qc_MWh'] * 1000  # Convert MWh to kWh

        # Calculate annual O&M costs (typically 2-3% of CAPEX)
        annual_om_cost = 0.025 * capex

        # Get the duration of this phase
        phase_duration = self.phase_durations[phase - 1]

        # Calculate present value of OPEX for all years in the phase
        opex_present_value = 0
        for year in range(phase_duration):
            discount_factor = 1 / ((1 + self.interest_rate) ** (year + 1))
            opex_present_value += annual_om_cost * discount_factor

        # Total expenditure is CAPEX (happens once) plus OPEX (recurring) in present value
        total_expenditure = capex + opex_present_value

        return total_expenditure

    def calculate_roi(self, cluster_set, phase):
        """
        Calculate Discounted ROI for a cluster set in a specific phase.

        Discounted ROI = present value of net annual returns over phase duration / CAPEX

        This calculation fully discounts all future cash flows to account for the time value of money.

        Parameters:
        -----------
        cluster_set : tuple or list
            Tuple or list of cluster IDs
        phase : int
            Phase number (1-based)

        Returns:
        --------
        float
            Discounted Return on Investment (ROI)
        """
        # For phase 0, return 0 as per requirements
        if phase == 0:
            return 0

        # Get metrics for this cluster set
        key = '+'.join(map(str, sorted(cluster_set)))
        if key not in self.cluster_metrics:
            return -float('inf')  # Invalid cluster set

        metrics = self.cluster_metrics[key]

        # Calculate CAPEX with interest rate adjustment
        capex, _ = self._calculate_phase_capex(cluster_set, phase)

        # Calculate annual revenue (energy price * annual demand)
        if self.network_type == 'DH':
            annual_demand_kwh = metrics['total_annual_Qh_MWh'] * 1000  # Convert MWh to kWh
        else:
            annual_demand_kwh = metrics['total_annual_Qc_MWh'] * 1000  # Convert MWh to kWh

        annual_revenue = annual_demand_kwh * self.energy_price

        # Calculate annual O&M costs (typically 2-3% of CAPEX)
        annual_om_cost = 0.025 * capex

        # Net annual return
        net_annual_return = annual_revenue - annual_om_cost

        # Get the duration of this phase
        phase_duration = self.phase_durations[phase - 1]

        # Calculate present value of net annual returns over the entire phase duration
        present_value_of_returns = 0
        for year in range(phase_duration):
            discount_factor = 1 / ((1 + self.interest_rate) ** (year + 1))
            present_value_of_returns += net_annual_return * discount_factor

        # Calculate Discounted ROI (present value of net annual returns / CAPEX)
        if capex > 0:
            roi = present_value_of_returns / capex
        else:
            roi = 0

        return roi

    def calculate_npv(self, cluster_set, phase, years=20):
        """
        Calculate Net Present Value for a cluster set in a specific phase.

        Parameters:
        -----------
        cluster_set : tuple or list
            Tuple or list of cluster IDs
        phase : int
            Phase number (1-based)
        years : int
            Number of years for NPV calculation

        Returns:
        --------
        float
            Net Present Value (NPV)
        """
        # Calculate CAPEX with interest rate adjustment
        capex, _ = self._calculate_phase_capex(cluster_set, phase)

        # For phase 0, return negative CAPEX as per requirements
        if phase == 0:
            return -capex

        # Get metrics for this cluster set
        key = '+'.join(map(str, sorted(cluster_set)))
        if key not in self.cluster_metrics:
            return -float('inf')  # Invalid cluster set

        metrics = self.cluster_metrics[key]

        # Calculate annual revenue and O&M costs
        if self.network_type == 'DH':
            annual_demand_kwh = metrics['total_annual_Qh_MWh'] * 1000  # Convert MWh to kWh
        else:
            annual_demand_kwh = metrics['total_annual_Qc_MWh'] * 1000  # Convert MWh to kWh

        annual_revenue = annual_demand_kwh * self.energy_price
        annual_om_cost = 0.025 * capex
        net_annual_return = annual_revenue - annual_om_cost

        # Calculate the year when this phase starts
        phase_start_year = 0
        for p in range(1, phase):
            phase_start_year += self.phase_durations[p - 1]

        # Calculate NPV
        npv = -capex  # Initial investment (negative) at the start of the phase

        for year in range(years):
            # Only count returns for years after the phase starts
            if year >= phase_start_year:
                # Determine which phase this year belongs to
                current_phase = 1
                year_in_phases = year
                while current_phase <= self.num_phases:
                    if year_in_phases < self.phase_durations[current_phase - 1]:
                        break
                    year_in_phases -= self.phase_durations[current_phase - 1]
                    current_phase += 1

                # Only count returns if we're in or after the current phase
                if current_phase >= phase:
                    discount_factor = 1 / ((1 + self.interest_rate) ** (year + 1))
                    npv += net_annual_return * discount_factor

        return npv

    class TemporarySupplyFile:
        """Context manager for temporarily modifying the supply.csv file."""

        def __init__(self, locator, modified_supply_df):
            self.locator = locator
            self.modified_supply_df = modified_supply_df
            self.supply_file = locator.get_building_supply()
            # Use a unique identifier in the backup filename to avoid conflicts
            self.backup_file = os.path.join(locator.get_temporary_folder(), f'backup_supply_{id(self)}.csv')

        def __enter__(self):
            """Save backup and replace with modified file."""
            # Create backup of original file
            shutil.copy(self.supply_file, self.backup_file)

            # Replace with modified file
            self.modified_supply_df.to_csv(self.supply_file, index=False)
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            """Restore original file."""
            if os.path.exists(self.backup_file):
                shutil.copy(self.backup_file, self.supply_file)
                os.remove(self.backup_file)

    def _cache_emissions_for_individual(self, individual, emissions, has_non_district_scale):
        """Store emissions results and non-district scale flag for an individual in the cache"""
        key = tuple(individual) if not isinstance(individual, tuple) else individual
        self.emissions_cache[key] = (emissions, has_non_district_scale)

    def _compute_phase_emissions_from_cop(self, phase_supply_path: Union[str, Path], scope_buildings: Optional[Set[str]] = None, pump_electricity_kwh: float = 0.0, custom_demand_path: Optional[Union[str, Path]] = None) -> Tuple[float, float]:
        """
        Compute phase operational emissions from baseline thermal needs and phase-specific supply COP/efficiency without
        re-running Demand. Adds network pump electricity. Fails fast on any missing inputs.

        Returns (total_ton_co2, kg_co2_per_m2_per_yr)
        """
        # Resolve paths
        phase_supply_path = str(phase_supply_path)
        if not os.path.exists(phase_supply_path):
            log().error(f"Phase supply file not found: {phase_supply_path}")
            raise FileNotFoundError(f"Phase supply file not found: {phase_supply_path}")

        demand_path = str(custom_demand_path) if custom_demand_path else self.locator.get_dynamic_dtn_optimization_temp_scenario_total_demand()
        if not os.path.exists(demand_path):
            log().error(f"Total demand file not found: {demand_path}")
            raise FileNotFoundError(f"Total demand file not found: {demand_path}")

        # Load datasets
        demand = pd.read_csv(demand_path)
        supply = pd.read_csv(phase_supply_path)

        # Required demand columns
        req_cols = ['name', 'GFA_m2', 'QC_sys_MWhyr', 'Qhs_sys_MWhyr', 'Qww_sys_MWhyr', 'GRID_MWhyr', 'PV_MWhyr']
        missing = [c for c in req_cols if c not in demand.columns]
        if missing:
            log().error(f"Missing required demand columns: {missing} in {demand_path}")
            raise KeyError(f"Missing required demand columns: {missing}")

        # Required supply columns
        sup_req = ['name', 'supply_type_cs', 'supply_type_hs', 'supply_type_dhw', 'supply_type_el']
        sup_missing = [c for c in sup_req if c not in supply.columns]
        if sup_missing:
            log().error(f"Missing required supply columns: {sup_missing} in {phase_supply_path}")
            raise KeyError(f"Missing required supply columns: {sup_missing}")

        # Load assemblies (COP / efficiency and feedstock)
        cooling_db = pd.read_csv(self.locator.get_database_assemblies_supply_cooling())
        heating_db = pd.read_csv(self.locator.get_database_assemblies_supply_heating())
        dhw_db = pd.read_csv(self.locator.get_database_assemblies_supply_hot_water())
        el_db = pd.read_csv(self.locator.get_database_assemblies_supply_electricity())
        for db, nm in [(cooling_db, 'cooling'), (heating_db, 'heating'), (dhw_db, 'hot water'), (el_db, 'electricity')]:
            for col in ['code', 'feedstock', 'scale', 'efficiency']:
                if col not in db.columns:
                    log().error(f"Assemblies {nm} database missing column '{col}'")
                    raise KeyError(f"Assemblies {nm} database missing column '{col}'")

        # Load feedstock emission factors (kgCO2/MJ)
        from cea.datamanagement.format_helper.cea4_verify_db import get_csv_filenames
        factors_resources = {}
        list_feedstocks = get_csv_filenames(self.locator.get_db4_components_feedstocks_library_folder())
        for feedstock in list_feedstocks:
            factors_resources[feedstock] = pd.read_csv(self.locator.get_db4_components_feedstocks_feedstocks_csv(feedstocks=feedstock))
        if not factors_resources:
            log().error("No feedstock emission factors found in database.")
            raise RuntimeError("No feedstock emission factors found.")
        ef_simple = pd.concat([
            pd.DataFrame([(k, v['GHG_kgCO2MJ'].mean()) for k, v in factors_resources.items() if k != 'ENERGY_CARRIERS'],
                         columns=['code', 'GHG_kgCO2MJ']),
            pd.DataFrame([{'code': 'NONE', 'GHG_kgCO2MJ': 0.0}])
        ], ignore_index=True)

        def ef_kg_per_kwh(feedstock_code: str) -> float:
            row = ef_simple.loc[ef_simple['code'] == str(feedstock_code)]
            if row.empty or pd.isna(row['GHG_kgCO2MJ'].iloc[0]):
                log().error(f"Missing emission factor for feedstock '{feedstock_code}'")
                raise KeyError(f"Missing emission factor for feedstock '{feedstock_code}'")
            return float(row['GHG_kgCO2MJ'].iloc[0]) * 3.6

        # Merge supply with assemblies
        sup = supply[['name', 'supply_type_cs', 'supply_type_hs', 'supply_type_dhw', 'supply_type_el']].copy()
        sup = sup.merge(cooling_db[['code', 'feedstock', 'scale', 'efficiency']].rename(
            columns={'code': 'supply_type_cs', 'feedstock': 'feedstock_cs', 'scale': 'scale_cs', 'efficiency': 'eff_cs'}),
            on='supply_type_cs', how='left')
        sup = sup.merge(heating_db[['code', 'feedstock', 'scale', 'efficiency']].rename(
            columns={'code': 'supply_type_hs', 'feedstock': 'feedstock_hs', 'scale': 'scale_hs', 'efficiency': 'eff_hs'}),
            on='supply_type_hs', how='left')
        sup = sup.merge(dhw_db[['code', 'feedstock', 'scale', 'efficiency']].rename(
            columns={'code': 'supply_type_dhw', 'feedstock': 'feedstock_dhw', 'scale': 'scale_dhw', 'efficiency': 'eff_dhw'}),
            on='supply_type_dhw', how='left')

        # Validate COPs/effectiveness are present
        if sup[['eff_cs', 'eff_hs', 'eff_dhw']].isna().any().any():
            missing_rows = sup[sup[['eff_cs', 'eff_hs', 'eff_dhw']].isna().any(axis=1)]
            log().error(f"Missing efficiency/COP for some supply codes: {missing_rows.to_dict(orient='records')}")
            raise KeyError("Missing efficiency/COP for some supply codes")

        # Merge with demand
        df = demand.merge(sup, on='name', how='inner')
        if scope_buildings:
            df = df[df['name'].isin(scope_buildings)].copy()
        if df.empty:
            log().error("No buildings found in scope after merging demand and supply.")
            raise RuntimeError("Empty scope for emissions computation.")

        # Thermal needs (ensure non-negative)
        df['QC_sys_MWhyr'] = df['QC_sys_MWhyr'].abs().fillna(0.0)
        df['Qhs_sys_MWhyr'] = df['Qhs_sys_MWhyr'].abs().fillna(0.0)
        df['Qww_sys_MWhyr'] = df['Qww_sys_MWhyr'].abs().fillna(0.0)

        # Cooling emissions (generalized by feedstock)
        def emis_from_thermal(q_mwh, eff, feedstock):
            fs = str(feedstock).upper() if pd.notna(feedstock) else 'NONE'

            # Legitimate no-system case: zero emissions only if thermal need is zero
            if fs == 'NONE':
                if (q_mwh or 0) > 1e-9:
                    log().error(f"Non-zero thermal need ({q_mwh} MWh) with supply feedstock NONE. Inconsistent demand/supply configuration.")
                    raise ValueError("Thermal need present with NONE supply.")
                return 0.0

            # For any real feedstock, efficiency must be positive
            if eff is None or eff <= 0:
                log().error(f"Non-positive efficiency detected for feedstock {fs}")
                raise ValueError("Non-positive efficiency")

            if fs == 'GRID':
                e_kwh = (q_mwh * 1000.0) / eff
                return (e_kwh * ef_kg_per_kwh('GRID')) / 1000.0
            else:
                mwh_final = q_mwh / eff
                return (mwh_final * ef_kg_per_kwh(fs)) / 1000.0

        df['Emis_cool_t'] = [emis_from_thermal(q, e, fs) for q, e, fs in zip(df['QC_sys_MWhyr'], df['eff_cs'], df['feedstock_cs'])]
        df['Emis_hs_t'] = [emis_from_thermal(q, e, fs) for q, e, fs in zip(df['Qhs_sys_MWhyr'], df['eff_hs'], df['feedstock_hs'])]
        df['Emis_dhw_t'] = [emis_from_thermal(q, e, fs) for q, e, fs in zip(df['Qww_sys_MWhyr'], df['eff_dhw'], df['feedstock_dhw'])]

        # Electricity end-uses (non-HVAC)
        ef_grid = ef_kg_per_kwh('GRID')

        # PV EF fallback logic:
        # 1) Use PV if present; 2) Else use SOLAR if present; 3) Else assume 0 for PV operation
        if 'PV' in ef_simple['code'].values:
            ef_pv = ef_kg_per_kwh('PV')
        elif 'SOLAR' in ef_simple['code'].values:
            log().debug("PV feedstock EF not found; using SOLAR EF for PV.")
            ef_pv = ef_kg_per_kwh('SOLAR')
        else:
            log().warning("PV (and SOLAR) feedstock EF not found; assuming 0 kgCO2/kWh for PV operational emissions.")
            ef_pv = 0.0

        df['Emis_el_t'] = (df['GRID_MWhyr'] * ef_grid + df['PV_MWhyr'] * ef_pv) / 1000.0

        # Sum per building
        df['Emis_total_t'] = df['Emis_cool_t'] + df['Emis_hs_t'] + df['Emis_dhw_t'] + df['Emis_el_t']

        # Pumps
        if pump_electricity_kwh < 0:
            log().error("Pump electricity cannot be negative.")
            raise ValueError("Negative pump electricity")
        emis_pump_t = (pump_electricity_kwh * ef_grid) / 1000.0

        # Aggregate
        total_no_pump = float(df['Emis_total_t'].sum())
        total_t = total_no_pump + emis_pump_t
        gfa = float(df['GFA_m2'].sum())
        kg_per_m2_yr = (total_t * 1000.0 / gfa) if gfa > 0 else 0.0

        # Build breakdown and EF mapping for diagnostics
        emis_breakdown = {
            'emis_cooling_t': float(df['Emis_cool_t'].sum()),
            'emis_heating_t': float(df['Emis_hs_t'].sum()),
            'emis_dhw_t': float(df['Emis_dhw_t'].sum()),
            'emis_electricity_t': float(df['Emis_el_t'].sum()),
            'emis_pump_t': float(emis_pump_t),
            'emis_total_t': float(total_t),
            'ef_grid_kg_per_kwh': float(ef_grid),
        }
        # PV or SOLAR EF if available
        try:
            if 'PV' in ef_simple['code'].values:
                emis_breakdown['ef_pv_kg_per_kwh'] = float(ef_kg_per_kwh('PV'))
            elif 'SOLAR' in ef_simple['code'].values:
                emis_breakdown['ef_pv_kg_per_kwh'] = float(ef_kg_per_kwh('SOLAR'))
        except Exception:
            pass
        # Add EF mapping for feedstocks used this phase (hs/dhw/cs)
        used_feedstocks = set(str(x).upper() for x in pd.concat([
            df['feedstock_cs'], df['feedstock_hs'], df['feedstock_dhw']
        ]).dropna().unique().tolist())
        feedstock_efs = {}
        for fs in sorted(used_feedstocks):
            try:
                feedstock_efs[fs] = float(ef_kg_per_kwh(fs))
            except Exception:
                continue
        try:
            import json as _json
            emis_breakdown['feedstock_efs_json'] = _json.dumps(feedstock_efs)
        except Exception:
            emis_breakdown['feedstock_efs_json'] = str(feedstock_efs)

        log().debug(f"Emissions breakdown: COOL={emis_breakdown['emis_cooling_t']:.2f} t, "
                    f"HS={emis_breakdown['emis_heating_t']:.2f} t, DHW={emis_breakdown['emis_dhw_t']:.2f} t, "
                    f"EL={emis_breakdown['emis_electricity_t']:.2f} t, PUMP={emis_breakdown['emis_pump_t']:.2f} t, "
                    f"TOTAL={emis_breakdown['emis_total_t']:.2f} t")

        return total_t, kg_per_m2_yr, emis_breakdown

    def _run_demand_for_scenario(self, scenario_path: str):
        """Run CEA Demand for a given scenario path (used for temp scenario)."""
        try:
            import cea.config as _cfg
            from cea.demand import demand_main as _demand_main
            cfg = _cfg.Configuration()
            cfg.scenario = scenario_path
            try:
                cfg.general.debug = True if hasattr(cfg, 'general') and hasattr(cfg.general, 'debug') and getattr(logging.getLogger(), 'level', logging.INFO) <= logging.DEBUG else cfg.general.debug
            except Exception:
                pass
            _demand_main.main(cfg)
        except Exception as e:
            log().error(f"Demand run failed for scenario {scenario_path}: {e}")
            raise

    @staticmethod
    def _signature_from_supply_df(df: pd.DataFrame) -> str:
        """Create a stable SHA1 signature from the supply dataframe content relevant to energy carriers."""
        try:
            cols = [c for c in ['name', 'supply_type_hs', 'supply_type_cs', 'supply_type_dhw', 'supply_type_el'] if c in df.columns]
            sub = df[cols].copy()
            sub['__name_norm__'] = sub['name'].astype(str).str.strip().str.upper()
            sub = sub.sort_values('__name_norm__')
            sub = sub.fillna('')
            s = '\n'.join(f"{r['__name_norm__']}|{r.get('supply_type_hs','')}|{r.get('supply_type_cs','')}|{r.get('supply_type_dhw','')}|{r.get('supply_type_el','')}" for _, r in sub.iterrows())
            return hashlib.sha1(s.encode('utf-8')).hexdigest()[:12]
        except Exception:
            return f"x{random.randint(0, 999999):06d}"

    class _TemporarySupplyFileForScenario:
        """Context manager to swap supply.csv for a specific locator's scenario and restore afterwards."""
        def __init__(self, locator_obj: cea.inputlocator.InputLocator, modified_supply_df: pd.DataFrame):
            self.locator = locator_obj
            self.modified_supply_df = modified_supply_df
            self.supply_file = locator_obj.get_building_supply()
            base, ext = os.path.splitext(self.supply_file)
            self.backup_file = f"{base}.__bak_{int(time.time())}_{random.randint(0,99999)}{ext}"
        def __enter__(self):
            shutil.copy2(self.supply_file, self.backup_file)
            self.modified_supply_df.to_csv(self.supply_file, index=False)
            return self
        def __exit__(self, exc_type, exc_val, exc_tb):
            try:
                if os.path.exists(self.backup_file):
                    shutil.copy2(self.backup_file, self.supply_file)
                    os.remove(self.backup_file)
            except Exception:
                pass

    def calculate_district_emissions_new(self):
        """
        Calculate district operation emissions using the new methodology.

        Workflow:
        1. Read from the generated genome to understand what cluster(s) it decides to connect at what phase
        2. Identify corresponding building IDs of each cluster
        3. Read from the Total_LCA_operation.csv, sum up the "GHG_sys_tonCO2" column for phase 0
        4. Read from the building-properties/supply.csv
        5. Make n copies of it if there are n phases in total
        6. Modify the supplied technologies for cooling, heating, dhw, of newly cluster(s) in phase 1 to those for cluster 0
        7. Run the LCA operation module again, repeat step 3 calculation for phase 1
        8. Repeat steps 6 to 7 for the following phases
        9. For the row of Total in DTN optimization result csv, copy from the last phase

        Returns:
        --------
        dict
            Dictionary with district operation emission and district operation emission per GFA for each phase
        """

        # Initialize results dictionary
        results = {}

        # Get original supply file
        supply_file = self.locator.get_building_supply()
        original_supply_df = pd.read_csv(supply_file)

        # Get district supply systems from cluster 0 buildings
        cluster0_buildings = self._get_buildings_in_specific_cluster(0)
        if not cluster0_buildings:
            log().warning("No buildings found in cluster 0 (existing DTN)")
            return {}

        # Get supply systems used by cluster 0 (existing DTN)
        district_supply_systems = original_supply_df[original_supply_df['name'].isin(cluster0_buildings)]

        # Extract district supply types
        district_heating_system = district_supply_systems['supply_type_hs'].iloc[0]
        district_cooling_system = district_supply_systems['supply_type_cs'].iloc[0]
        district_dhw_system = district_supply_systems['supply_type_dhw'].iloc[0]

        # Create directory for phase-specific supply files
        phase_files_dir = Path(self.locator.get_dynamic_dtn_optimization_temp_scenario_phase_supply_files_folder())
        temp_root = Path(self.locator.get_dynamic_dtn_optimization_temp_scenario_folder())
        _assert_under_temp(phase_files_dir, temp_root, "phase supply folder")
        _log_io("ENSURE DIR phase supply", phase_files_dir)
        phase_files_dir.mkdir(parents=True, exist_ok=True)
        log().debug(f"Creating phase supply files directory at: {phase_files_dir}")
        _log_io("PHASE SUPPLY ROOT", phase_files_dir)

        # Create phase 0 supply file (original)
        phase0_supply_path = phase_files_dir / "phase0_supply.csv"
        _assert_under_temp(phase0_supply_path, temp_root, "phase0 supply file")
        _log_io("WRITE phase0_supply", phase0_supply_path)
        original_supply_df.to_csv(phase0_supply_path, index=False)
        log().debug(f"Created phase 0 supply file at: {phase0_supply_path}")

        # Prepare per-phase summary collection
        summary_rows = []

        # Load demand once for GFA denominators (temp scenario); restrict denominators to the authoritative universe U
        _demand_df = pd.read_csv(self.locator.get_dynamic_dtn_optimization_temp_scenario_total_demand())
        _universe = list(getattr(self, '_universe_U_names', []) or [])
        if _universe:
            _total_gfa_const = float(_demand_df.loc[_demand_df['name'].isin(_universe), 'GFA_m2'].sum())
        else:
            _total_gfa_const = float(_demand_df['GFA_m2'].sum())
        _name_to_gfa = dict(zip(_demand_df['name'], _demand_df['GFA_m2']))

        # Create directory for per-phase LCA outputs under temp scenario dtn_expansion
        phase_lca_dir = Path(self.locator.get_dynamic_dtn_optimization_temp_scenario_dtn_expansion_folder()) / "phase_LCA_operation"
        phase_lca_dir.mkdir(parents=True, exist_ok=True)
        # Create directory for per-phase Demand snapshots under temp scenario dtn_expansion
        phase_demand_dir = Path(self.locator.get_dynamic_dtn_optimization_temp_scenario_dtn_expansion_folder()) / "phase_demand"
        phase_demand_dir.mkdir(parents=True, exist_ok=True)

        # Phase 0 emissions: always use LCA operation with temp demand (COP-based path removed)
        if False:
            # District-wide scope: ALL buildings
            _universe_list = list(getattr(self, '_universe_U_names', []) or [])
            total_ghg, _, breakdown = self._compute_phase_emissions_from_cop(phase0_supply_path, scope_buildings=_universe_list if _universe_list else None, pump_electricity_kwh=0.0,
                                                                             custom_demand_path=self.locator.get_dynamic_dtn_optimization_temp_scenario_total_demand())
            # Collect breakdown row for phase 0
            row0 = {'phase': 0, 'cumulative_clusters': '0'}
            row0.update(breakdown)
            breakdown_rows.append(row0)
            # In COP mode we don't compute connected-only explicitly here; fall back to total for intensity
            connected_ghg = total_ghg
        else:
            from cea.analysis.lca.operation import lca_operation
            temp_demand_path = self.locator.get_dynamic_dtn_optimization_temp_scenario_total_demand()
            log().info("Preparing Phase 0 demand/LCA with caching:")
            log().info(f"  - Supply path: {phase0_supply_path}")
            log().info(f"  - Baseline temp demand path: {temp_demand_path}")
            # Use a temp scenario locator so outputs go under temp_scenario
            temp_locator = cea.inputlocator.InputLocator(self.locator.get_dynamic_dtn_optimization_temp_scenario_folder())
            _assert_under_temp(phase0_supply_path, temp_root, "LCA supply path (phase 0)")
            _assert_under_temp(_PathForIO(temp_demand_path), temp_root, "LCA demand path (phase 0)")
            # Phase 0 caching
            sig0 = self._signature_from_supply_df(original_supply_df)
            phase0_demand_cache = phase_demand_dir / f"phase0__{sig0}.csv"
            phase0_lca_cache = phase_lca_dir / f"phase0__{sig0}__Total_LCA_operation.csv"
            if phase0_demand_cache.exists():
                log().info(f"[Demand][Cache HIT] Phase 0 demand: {phase0_demand_cache}")
            else:
                shutil.copy2(temp_demand_path, phase0_demand_cache)
                log().info(f"[Demand][Cache MISS] Saved Phase 0 demand snapshot to {phase0_demand_cache}")
            # LCA cache
            if phase0_lca_cache.exists():
                log().info(f"[LCA][Cache HIT] Phase 0 LCA: {phase0_lca_cache}")
                lca_operation_results = pd.read_csv(phase0_lca_cache)
            else:
                _log_io("CALL LCA phase0", _PathForIO(temp_locator.scenario), extra={"supply": str(phase0_supply_path), "demand": str(phase0_demand_cache)})
                lca_operation(temp_locator, custom_supply_path=str(phase0_supply_path), custom_demand_path=str(phase0_demand_cache))
                lca_operation_results = pd.read_csv(temp_locator.get_lca_operation())
                try:
                    lca_operation_results.to_csv(phase0_lca_cache, index=False)
                    log().info(f"[LCA][Cache MISS] Saved Phase 0 LCA to {phase0_lca_cache}")
                except Exception:
                    pass
            log().info(f"Loaded LCA results with {len(lca_operation_results)} buildings")
            # Determine name column and build normalized series
            name_col = 'name' if 'name' in lca_operation_results.columns else ('Name' if 'Name' in lca_operation_results.columns else None)
            if not name_col:
                raise ValueError("LCA file missing building name column (expected 'name' or 'Name')")
            lca_names_norm = lca_operation_results[name_col].astype(str).str.strip().str.upper()
            # Universe mask (baseline CONSUMER buildings only) if available
            universe_raw = list(getattr(self, '_universe_U_names', []) or [])
            universe_norm = {str(b).strip().upper() for b in universe_raw}
            mask_total = lca_names_norm.isin(universe_norm) if universe_norm else pd.Series(True, index=lca_operation_results.index)
            # Total GHG limited to U
            total_ghg = float(lca_operation_results.loc[mask_total, 'GHG_sys_tonCO2'].sum())
            # Connected-only emissions for Phase 0 (cluster 0) with normalized name matching
            conn_set_raw = set(self._get_buildings_in_specific_cluster(0))
            conn_set_norm = {str(b).strip().upper() for b in conn_set_raw}
            mask_conn = lca_names_norm.isin(conn_set_norm)
            connected_ghg = float(lca_operation_results.loc[mask_conn, 'GHG_sys_tonCO2'].sum())
            # Diagnostics for Phase 0 matching
            try:
                diag_dir = Path(self.locator.get_dynamic_dtn_optimization_temp_scenario_phase_supply_files_folder())
                _assert_under_temp(diag_dir, temp_root, "phase supply folder (diag)")
                expected_names = sorted(conn_set_norm)
                matched_names = sorted(set(lca_names_norm[mask_conn].tolist()))
                pd.DataFrame({'expected_conn_name': expected_names,
                              'matched': [n in matched_names for n in expected_names]}).to_csv(
                    diag_dir / 'phase0_conn_name_match.csv', index=False)
                pd.DataFrame([{
                    'phase': 0,
                    'connected_expected_count': len(expected_names),
                    'connected_matched_in_lca': int(mask_conn.sum()),
                    'connected_ghg_t': connected_ghg,
                    'district_total_ghg_t': total_ghg
                }]).to_csv(diag_dir / 'phase0_conn_match_summary.csv', index=False)
                log().info(f"[Diag] Phase 0 name-match diagnostics written to {diag_dir}")
            except Exception as e:
                log().warning(f"Could not write Phase 0 name-match diagnostics: {e}")
            # Save phase 0 LCA results
            try:
                phase0_lca_csv = Path(self.locator.get_dynamic_dtn_optimization_temp_scenario_dtn_expansion_folder()) / "phase_LCA_operation" / "phase0_Total_LCA_operation.csv"
                lca_operation_results.to_csv(phase0_lca_csv, index=False)
                log().debug(f"Saved phase 0 LCA results to {phase0_lca_csv}")
            except Exception:
                pass
            # No hybrid replacement for Phase 0; preserve raw LCA totals for comparability

        # Compute GFA denominators for phase 0
        _connected_buildings_p0 = set(self._get_buildings_in_specific_cluster(0))
        if 'GFA_m2' in lca_operation_results.columns:
            try:
                name_col_gfa = 'name' if 'name' in lca_operation_results.columns else ('Name' if 'Name' in lca_operation_results.columns else None)
                mask_conn_gfa = lca_operation_results[name_col_gfa].astype(str).isin(_connected_buildings_p0) if name_col_gfa else pd.Series(False, index=lca_operation_results.index)
                connected_gfa_lca = float(lca_operation_results.loc[mask_conn_gfa, 'GFA_m2'].sum())
                total_gfa_lca = float(lca_operation_results['GFA_m2'].sum())
            except Exception:
                connected_gfa_lca = float(sum(_name_to_gfa.get(b, 0.0) for b in _connected_buildings_p0))
                total_gfa_lca = _total_gfa_const
        else:
            connected_gfa_lca = float(sum(_name_to_gfa.get(b, 0.0) for b in _connected_buildings_p0))
            total_gfa_lca = _total_gfa_const
        per_connected = (connected_ghg * 1000.0 / connected_gfa_lca) if connected_gfa_lca > 0 else 0.0
        per_total = (total_ghg * 1000.0 / total_gfa_lca) if total_gfa_lca > 0 else 0.0

        # Store results for phase 0
        results[0] = {
            'district_operation_emission [t CO2eq/yr]': total_ghg,
            'connected_clusters_operation_emissions [t CO2eq/yr]': connected_ghg,
            'district_operation_emission_per_connected_gfa [kg CO2eq/yr/m2]': per_connected,
            'district_operation_emission_per_total_gfa [kg CO2eq/yr/m2]': per_total
        }
        # Append to summary rows
        try:
            summary_rows.append({
                'phase': 0,
                'cumulative_clusters': '0',
                'district_operation_emission_tonCO2': total_ghg,
                'connected_clusters_operation_emissions_tonCO2': connected_ghg,
                'per_total_gfa_kgCO2_per_m2yr': per_total,
                'per_connected_gfa_kgCO2_per_m2yr': per_connected
            })
        except Exception:
            pass

        # Get cluster assignments from the final best genome if available; else fall back to current individual
        if hasattr(self, 'solution') and self.solution and 'genome' in self.solution and self.solution['genome']:
            cluster_phase_map = {cluster: p for cluster, p in zip(self.all_clusters, self.solution['genome'])}
            clusters_by_phase = {}
            for phase in range(1, self.num_phases + 1):
                clusters_by_phase[phase] = [cluster for cluster, p in cluster_phase_map.items() if p == phase]
        elif hasattr(self, 'current_individual') and self.current_individual:
            cluster_phase_map = {cluster: p for cluster, p in zip(self.all_clusters, self.current_individual)}
            clusters_by_phase = {}
            for phase in range(1, self.num_phases + 1):
                clusters_by_phase[phase] = [cluster for cluster, p in cluster_phase_map.items() if p == phase]
        else:
            # Fallback: distribute all_clusters evenly across phases
            log().warning("No final solution genome or current_individual found, using fallback cluster distribution")
            clusters_by_phase = {}
            clusters_per_phase = max(1, len(self.all_clusters) // self.num_phases)
            for phase in range(1, self.num_phases + 1):
                start_idx = (phase - 1) * clusters_per_phase
                end_idx = min(phase * clusters_per_phase, len(self.all_clusters))
                if phase == self.num_phases:  # Ensure last phase gets any remaining clusters
                    end_idx = len(self.all_clusters)
                clusters_by_phase[phase] = self.all_clusters[start_idx:end_idx]

        # Log the cluster assignments for debugging
        for phase, clusters in clusters_by_phase.items():
            log().info(f"Phase {phase}: Connecting clusters {clusters}")

        # Process each phase
        for phase in range(1, self.num_phases + 1):
            # Make a copy of the original supply file for this phase
            phase_supply_df = original_supply_df.copy()

            # Get all clusters connected up to this phase
            connected_clusters = [0]  # Start with cluster 0
            for p in range(1, phase + 1):
                connected_clusters.extend(clusters_by_phase.get(p, []))

            # Get all buildings in connected clusters
            connected_buildings = []
            for cluster in connected_clusters:
                buildings = self._get_buildings_in_specific_cluster(cluster)
                connected_buildings.extend(buildings)

            # Update supply systems for connected buildings (normalize names for robust matching)
            phase_supply_df['__name_norm__'] = phase_supply_df['name'].astype(str).str.strip().str.upper()
            connected_norm = {str(b).strip().upper() for b in connected_buildings}
            mask_conn = phase_supply_df['__name_norm__'].isin(connected_norm)
            phase_supply_df.loc[mask_conn, 'supply_type_hs'] = district_heating_system
            phase_supply_df.loc[mask_conn, 'supply_type_cs'] = district_cooling_system
            phase_supply_df.loc[mask_conn, 'supply_type_dhw'] = district_dhw_system
            # Clean helper column
            phase_supply_df.drop(columns='__name_norm__', inplace=True, errors='ignore')

            # Save the phase-specific supply file
            phase_supply_path = phase_files_dir / f"phase{phase}_supply.csv"
            _assert_under_temp(phase_supply_path, temp_root, "phase supply file")
            _log_io("WRITE phase_supply", phase_supply_path, extra={"phase": phase})
            phase_supply_df.to_csv(phase_supply_path, index=False)
            log().debug(f"Created phase {phase} supply file: {phase_supply_path}")

            # Demand + LCA caching per phase in temp scenario
            from cea.analysis.lca.operation import lca_operation
            temp_locator = cea.inputlocator.InputLocator(self.locator.get_dynamic_dtn_optimization_temp_scenario_folder())
            temp_demand_path = self.locator.get_dynamic_dtn_optimization_temp_scenario_total_demand()
            _assert_under_temp(phase_supply_path, temp_root, "LCA supply path (phase)")
            _assert_under_temp(_PathForIO(temp_demand_path), temp_root, "LCA demand path (phase)")
            sig = self._signature_from_supply_df(phase_supply_df)
            phase_demand_cache = phase_demand_dir / f"phase{phase}__{sig}.csv"
            phase_lca_cache = phase_lca_dir / f"phase{phase}__{sig}__Total_LCA_operation.csv"

            # Demand cache: if miss, swap supply in temp scenario and run Demand
            if phase_demand_cache.exists():
                log().info(f"[Demand][Cache HIT] Phase {phase} demand (temp): {phase_demand_cache}")
            else:
                log().info(f"[Demand][Cache MISS] Phase {phase}: swapping temp supply & re-running Demand")
                with self._TemporarySupplyFileForScenario(temp_locator, phase_supply_df):
                    self._run_demand_for_scenario(temp_locator.scenario)
                    try:
                        td_tmp = pd.read_csv(temp_demand_path)
                        diag = {
                            'DH_hs_MWhyr': float(td_tmp.get('DH_hs_MWhyr', pd.Series([0])).sum()),
                            'NG_hs_MWhyr': float(td_tmp.get('NG_hs_MWhyr', pd.Series([0])).sum()),
                            'DC_cs_MWhyr': float(td_tmp.get('DC_cs_MWhyr', pd.Series([0])).sum()),
                            'GRID_MWhyr': float(td_tmp.get('GRID_MWhyr', pd.Series([0])).sum()),
                            'PV_MWhyr': float(td_tmp.get('PV_MWhyr', pd.Series([0])).sum()),
                        }
                        log().info(f"[Demand][Phase {phase} temp] Carriers summary: {diag}")
                    except Exception:
                        pass
                    shutil.copy2(temp_demand_path, phase_demand_cache)
                    log().info(f"[Demand] Saved Phase {phase} temp demand snapshot to {phase_demand_cache}")

            # LCA: cache load or run against cached demand
            if phase_lca_cache.exists():
                log().info(f"[LCA][Cache HIT] Phase {phase} LCA (temp): {phase_lca_cache}")
                lca_operation_results = pd.read_csv(phase_lca_cache)
            else:
                _log_io("CALL LCA phase", _PathForIO(temp_locator.scenario), extra={"phase": phase, "supply": str(phase_supply_path), "demand": str(phase_demand_cache)})
                lca_operation(temp_locator, custom_supply_path=str(phase_supply_path), custom_demand_path=str(phase_demand_cache))
                lca_results_path = temp_locator.get_lca_operation()
                _assert_under_temp(_PathForIO(lca_results_path), temp_root, "LCA results (phase)")
                _log_io("READ LCA results phase", _PathForIO(lca_results_path), extra={"phase": phase})
                lca_operation_results = pd.read_csv(lca_results_path)
                try:
                    lca_operation_results.to_csv(phase_lca_cache, index=False)
                    log().info(f"[LCA][Cache MISS] Saved Phase {phase} LCA to {phase_lca_cache}")
                except Exception:
                    pass
            # Also save generic per-phase LCA CSV for inspection
            try:
                (Path(self.locator.get_dynamic_dtn_optimization_temp_scenario_dtn_expansion_folder()) / "phase_LCA_operation").mkdir(parents=True, exist_ok=True)
                phase_lca_csv = Path(self.locator.get_dynamic_dtn_optimization_temp_scenario_dtn_expansion_folder()) / "phase_LCA_operation" / f"phase{phase}_Total_LCA_operation.csv"
                lca_operation_results.to_csv(phase_lca_csv, index=False)
                log().debug(f"Saved phase {phase} LCA results to {phase_lca_csv}")
            except Exception:
                pass
            name_col = 'name' if 'name' in lca_operation_results.columns else ('Name' if 'Name' in lca_operation_results.columns else None)
            if not name_col:
                raise ValueError("LCA file missing building name column (expected 'name' or 'Name')")
            # Normalize names and build universe mask
            lca_names_norm = lca_operation_results[name_col].astype(str).str.strip().str.upper()
            universe_raw = list(getattr(self, '_universe_U_names', []) or [])
            universe_norm = {str(b).strip().upper() for b in universe_raw}
            mask_total = lca_names_norm.isin(universe_norm) if universe_norm else pd.Series(True, index=lca_operation_results.index)
            # Total GHG limited to U
            total_ghg = float(lca_operation_results.loc[mask_total, 'GHG_sys_tonCO2'].sum())
            # Connected-only emissions for this phase
            conn_set_raw = set(connected_buildings)
            conn_set_norm = {str(b).strip().upper() for b in conn_set_raw}
            mask_conn = lca_names_norm.isin(conn_set_norm)
            connected_ghg = float(lca_operation_results.loc[mask_conn, 'GHG_sys_tonCO2'].sum())

            # Compute GFA denominators for this phase (prefer LCA per-building GFA for consistency)
            if 'GFA_m2' in lca_operation_results.columns:
                try:
                    connected_gfa_lca = float(lca_operation_results.loc[mask_conn, 'GFA_m2'].sum())
                    total_gfa_lca = float(lca_operation_results.loc[mask_total, 'GFA_m2'].sum())
                except Exception:
                    connected_gfa_lca = float(sum(_name_to_gfa.get(b, 0.0) for b in set(connected_buildings)))
                    total_gfa_lca = _total_gfa_const
            else:
                connected_gfa_lca = float(sum(_name_to_gfa.get(b, 0.0) for b in set(connected_buildings)))
                total_gfa_lca = _total_gfa_const
            per_connected = (connected_ghg * 1000.0 / connected_gfa_lca) if connected_gfa_lca > 0 else 0.0
            per_total = (total_ghg * 1000.0 / total_gfa_lca) if total_gfa_lca > 0 else 0.0

            # Store results for this phase (drop legacy per_gfa column)
            results[phase] = {
                'district_operation_emission [t CO2eq/yr]': total_ghg,
                'connected_clusters_operation_emissions [t CO2eq/yr]': connected_ghg,
                'district_operation_emission_per_connected_gfa [kg CO2eq/yr/m2]': per_connected,
                'district_operation_emission_per_total_gfa [kg CO2eq/yr/m2]': per_total
            }

        # Write per-phase summary CSV
        try:
            if summary_rows:
                summary_df = pd.DataFrame(summary_rows)
                summary_file = Path(self.locator.get_dynamic_dtn_optimization_temp_scenario_dtn_expansion_folder()) / "emissions_by_phase_summary.csv"
                _assert_under_temp(summary_file, temp_root, "emissions summary file")
                _log_io("WRITE emissions summary", summary_file)
                summary_df.to_csv(summary_file, index=False)
                log().debug(f"Saved emissions-by-phase summary to {summary_file}")
        except Exception as e:
            log().warning(f"Could not write emissions summary CSV: {e}")

        return results

    def _get_clusters_connected_in_phase(self, phase):
        """
        Get the clusters connected in a specific phase.

        Parameters:
        -----------
        phase : int
            Phase number

        Returns:
        --------
        list
            List of cluster IDs connected in the specified phase
        """
        # This method depends on how the optimization results are stored
        # For single-objective optimization, we can get the clusters from the solution
        if hasattr(self, 'solution') and not self.multi_objective_mode:
            if phase in self.solution['phases']:
                return self.solution['phases'][phase]

        # For multi-objective optimization or if solution is not available,
        # we need to check if we're in the middle of an optimization
        # In that case, we can use the current individual being evaluated
        if hasattr(self, 'current_individual'):
            # Convert individual to cluster-phase mapping
            cluster_phase_map = {cluster: p for cluster, p in zip(self.all_clusters, self.current_individual)}
            # Get clusters connected in this phase
            return [cluster for cluster, p in cluster_phase_map.items() if p == phase]

        # If we can't determine the clusters, return an empty list
        return []

    def plot_multi_objective_results(self, all_individuals, pareto_front, objectives):
        """
        Create a scatter plot of all solutions tested during multi-objective optimization.

        Parameters:
        -----------
        all_individuals : list
            List of all individuals tested during optimization
        pareto_front : list
            List of Pareto-optimal individuals
        objectives : list
            List of objective functions used

        Returns:
        --------
        Path
            Path to the saved plot file
        """
        # Create output directory
        output_dir = Path(self.locator.get_dynamic_dtn_optimization_temp_scenario_dtn_expansion_folder())
        output_dir.mkdir(parents=True, exist_ok=True)

        # Extract fitness values for all individuals
        all_fitness = []
        for ind in all_individuals:
            if hasattr(ind, 'fitness') and hasattr(ind.fitness, 'values') and len(ind.fitness.values) > 0:
                all_fitness.append(ind.fitness.values)

        # Extract fitness values for Pareto front
        pareto_fitness = [ind.fitness.values for ind in pareto_front]

        # Create figure
        fig = plt.figure(figsize=(10, 8))

        # Determine plot type based on number of objectives
        if len(objectives) == 2:
            # 2D scatter plot
            ax = fig.add_subplot(111)

            # Plot all solutions
            x_all = [fit[0] for fit in all_fitness]
            y_all = [fit[1] for fit in all_fitness]
            ax.scatter(x_all, y_all, color='lightgray', alpha=0.5, label='All Solutions')

            # Plot Pareto front
            x_pareto = [fit[0] for fit in pareto_fitness]
            y_pareto = [fit[1] for fit in pareto_fitness]
            pareto_scatter = ax.scatter(x_pareto, y_pareto, color='red', s=100, label='Pareto Solutions')

            # Add solution IDs for Pareto solutions
            for i, (x, y) in enumerate(zip(x_pareto, y_pareto)):
                # Get the individual_id from the solution if available
                if i < len(pareto_front):
                    # Find the individual_id for this Pareto solution
                    pareto_ind = pareto_front[i]
                    ind_tuple = tuple(pareto_ind)

                    # Search in all_individuals to find matching individual_id
                    label_id = i  # Default fallback
                    for j, eval_ind in enumerate(all_individuals):
                        if hasattr(eval_ind, 'fitness') and hasattr(eval_ind.fitness, 'values'):
                            if tuple(eval_ind) == ind_tuple:
                                label_id = j
                                break

                    ax.annotate(f"ID {label_id}", (x, y), xytext=(5, 5), textcoords='offset points')
                else:
                    ax.annotate(f"ID {i}", (x, y), xytext=(5, 5), textcoords='offset points')

            # Set axis labels with units
            if objectives[0] == 'NPV':
                ax.set_xlabel('NPV [USD]')
            elif objectives[0] == 'ROI':
                ax.set_xlabel('ROI [-]')
            elif objectives[0] == 'emissions':
                ax.set_xlabel('Emissions [t CO2eq]')
            elif objectives[0] == 'total_capex':
                ax.set_xlabel('Total CAPEX [USD]')

            if objectives[1] == 'NPV':
                ax.set_ylabel('NPV [USD]')
            elif objectives[1] == 'ROI':
                ax.set_ylabel('ROI [-]')
            elif objectives[1] == 'emissions':
                ax.set_ylabel('Emissions [t CO2eq]')
            elif objectives[1] == 'total_capex':
                ax.set_ylabel('Total CAPEX [USD]')

        elif len(objectives) == 3:
            # 3D scatter plot
            ax = fig.add_subplot(111, projection='3d')

            # Plot all solutions
            x_all = [fit[0] for fit in all_fitness]
            y_all = [fit[1] for fit in all_fitness]
            z_all = [fit[2] for fit in all_fitness]
            ax.scatter(x_all, y_all, z_all, color='lightgray', alpha=0.5, label='All Solutions')

            # Plot Pareto front
            x_pareto = [fit[0] for fit in pareto_fitness]
            y_pareto = [fit[1] for fit in pareto_fitness]
            z_pareto = [fit[2] for fit in pareto_fitness]
            pareto_scatter = ax.scatter(x_pareto, y_pareto, z_pareto, color='red', s=100, label='Pareto Solutions')

            # Add solution IDs for Pareto solutions
            for i, (x, y, z) in enumerate(zip(x_pareto, y_pareto, z_pareto)):
                ax.text(x, y, z, str(i), size=8)

            # Set axis labels with units
            if objectives[0] == 'NPV':
                ax.set_xlabel('NPV [USD]')
            elif objectives[0] == 'ROI':
                ax.set_xlabel('ROI [-]')
            elif objectives[0] == 'emissions':
                ax.set_xlabel('Emissions [t CO2eq]')
            elif objectives[0] == 'total_capex':
                ax.set_xlabel('Total CAPEX [USD]')

            if objectives[1] == 'NPV':
                ax.set_ylabel('NPV [USD]')
            elif objectives[1] == 'ROI':
                ax.set_ylabel('ROI [-]')
            elif objectives[1] == 'emissions':
                ax.set_ylabel('Emissions [t CO2eq]')
            elif objectives[1] == 'total_capex':
                ax.set_ylabel('Total CAPEX [USD]')

            if objectives[2] == 'NPV':
                ax.set_zlabel('NPV [USD]')
            elif objectives[2] == 'ROI':
                ax.set_zlabel('ROI [-]')
            elif objectives[2] == 'emissions':
                ax.set_zlabel('Emissions [t CO2eq]')
            elif objectives[2] == 'total_capex':
                ax.set_zlabel('Total CAPEX [USD]')

        # Add title and legend with bigger, bold font
        plt.title(f'Multi-Objective Optimization Results - {self.network_type}',
                  fontsize=14, fontweight='bold')
        plt.legend(loc='best')

        # Save figure with appropriate name based on number of objectives
        if len(objectives) == 3:
            # For 3D plots, use "3d" in the filename
            plot_file_3d = output_dir / f"dtn_expansion_opt_pareto_plot_3d_{self.network_type}.png"
            plt.savefig(plot_file_3d, dpi=300, bbox_inches='tight')
            plt.close()

            # Also create a 2D version with the 3rd objective as color/size as a secondary visualization
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111)

            # Plot all solutions
            scatter_all = ax.scatter(x_all, y_all, c=z_all, cmap='viridis', alpha=0.5, s=50)

            # Plot Pareto front
            scatter_pareto = ax.scatter(x_pareto, y_pareto, c=z_pareto, cmap='viridis',
                                        edgecolors='red', linewidths=2, s=100)

            # Add solution IDs for Pareto solutions
            for i, (x, y) in enumerate(zip(x_pareto, y_pareto)):
                # Get the individual_id from the solution if available
                if i < len(pareto_front):
                    # Find the individual_id for this Pareto solution
                    pareto_ind = pareto_front[i]
                    ind_tuple = tuple(pareto_ind)

                    # Search in all_individuals to find matching individual_id
                    label_id = i  # Default fallback
                    for j, eval_ind in enumerate(all_individuals):
                        if hasattr(eval_ind, 'fitness') and hasattr(eval_ind.fitness, 'values'):
                            if tuple(eval_ind) == ind_tuple:
                                label_id = j
                                break

                    ax.annotate(f"ID {label_id}", (x, y), xytext=(5, 5), textcoords='offset points')
                else:
                    ax.annotate(f"ID {i}", (x, y), xytext=(5, 5), textcoords='offset points')

            # Set axis labels with units
            if objectives[0] == 'NPV':
                ax.set_xlabel('NPV [USD]')
            elif objectives[0] == 'ROI':
                ax.set_xlabel('ROI [-]')
            elif objectives[0] == 'emissions':
                ax.set_xlabel('Emissions [t CO2eq]')
            elif objectives[0] == 'total_capex':
                ax.set_xlabel('Total CAPEX [USD]')

            if objectives[1] == 'NPV':
                ax.set_ylabel('NPV [USD]')
            elif objectives[1] == 'ROI':
                ax.set_ylabel('ROI [-]')
            elif objectives[1] == 'emissions':
                ax.set_ylabel('Emissions [t CO2eq]')
            elif objectives[1] == 'total_capex':
                ax.set_ylabel('Total CAPEX [USD]')

            # Add colorbar
            cbar = plt.colorbar(scatter_all)
            if objectives[2] == 'NPV':
                cbar.set_label('NPV [USD]')
            elif objectives[2] == 'ROI':
                cbar.set_label('ROI [-]')
            elif objectives[2] == 'emissions':
                cbar.set_label('Emissions [t CO2eq]')
            elif objectives[2] == 'total_capex':
                cbar.set_label('Total CAPEX [USD]')

            # Add title and legend with bigger, bold font
            plt.title(f'Multi-Objective Optimization Results - {self.network_type}\n(Color: {objectives[2]})',
                      fontsize=14, fontweight='bold')

            # Save figure
            plot_file_2d = output_dir / f"dtn_expansion_opt_pareto_plot_2d_{self.network_type}.png"
            plt.savefig(plot_file_2d, dpi=300, bbox_inches='tight')
            plt.close()

            # Return the 3D plot file path for 3 objectives
            return plot_file_3d
        else:
            # For 2D plots, use the generic filename
            plot_file = output_dir / f"dtn_expansion_opt_pareto_plot_{self.network_type}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            return plot_file

    def calculate_ghg_emissions(self, cluster_set):
        """
        Calculate GHG emissions for a cluster set.
        This method is kept for backward compatibility but now uses the new approach.

        Special case: When cluster_set is (0,), calculate emissions for the whole district
        with cluster 0 using district systems and all other clusters using building-scale systems.

        Parameters:
        -----------
        cluster_set : tuple or list
            Tuple or list of cluster IDs

        Returns:
        --------
        float
            Total GHG emissions in tonCO2
        """
        # Special case for phase 0: calculate emissions for the whole district
        if cluster_set == (0,):
            # For phase 0, we can use calculate_district_emissions_new directly
            # It will calculate emissions for the whole district with cluster 0 using district systems
            # and all other clusters using building-scale systems
            district_emissions = self.calculate_district_emissions_new()

            # Return the operational emissions for phase 0
            if 0 in district_emissions:
                return district_emissions[0]['district_operation_emission [t CO2eq/yr]']
            else:
                return 0
        else:
            # For other cluster sets, we need to create a temporary individual
            # that assigns the specified clusters to phase 1
            temp_individual = [0] * len(self.all_clusters)

            # Map cluster IDs to indices in self.all_clusters
            cluster_indices = {cluster: i for i, cluster in enumerate(self.all_clusters)}

            # Set phase 1 for the specified clusters
            for cluster in cluster_set:
                if cluster in cluster_indices:
                    temp_individual[cluster_indices[cluster]] = 1

            # Set current_individual for use in calculate_district_emissions_new
            self.current_individual = temp_individual

            # Calculate emissions
            district_emissions = self.calculate_district_emissions_new()

            # Return the operational emissions for phase 1
            if 1 in district_emissions:
                return district_emissions[1]['district_operation_emission [t CO2eq/yr]']
            else:
                return 0

    def _setup_genetic_algorithm(self):
        """Set up the genetic algorithm using DEAP."""
        # Define genome representation: each gene is a phase number (1 to num_phases)
        # for each cluster (all clusters will be connected)
        self.toolbox.register("attr_phase", random.randint, 1, self.num_phases)

        # Define a custom individual creation function that ensures diversity
        def custom_individual():
            # Create individuals with more diverse phase assignments
            ind = []
            for _ in range(len(self.all_clusters)):
                # Distribute phases more evenly
                phase = random.randint(1, self.num_phases)
                ind.append(phase)
            ind = creator.Individual(ind)
            # Apply locks if configured
            if getattr(self, 'lock_committed_early_phases', False) and getattr(self, 'locked_clusters_phase', None):
                cluster_to_index = {c: i for i, c in enumerate(self.all_clusters)}
                for c, p in self.locked_clusters_phase.items():
                    idx = cluster_to_index.get(c, None)
                    if idx is not None:
                        ind[idx] = int(p)
            return ind

        # Register the custom individual creation function
        self.toolbox.register("individual", custom_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        # Create a custom population initialization function
        def custom_population(n):
            pop = []

            def _apply_locks(ind):
                try:
                    if getattr(self, 'lock_committed_early_phases', False) and getattr(self, 'locked_clusters_phase', None):
                        cluster_to_index = {c: i for i, c in enumerate(self.all_clusters)}
                        for c, p in self.locked_clusters_phase.items():
                            idx = cluster_to_index.get(c, None)
                            if idx is not None:
                                ind[idx] = int(p)
                except Exception:
                    pass
                return ind

            # Add some individuals with all clusters in the last phase
            last_phase_ind = creator.Individual([self.num_phases] * len(self.all_clusters))
            last_phase_ind = _apply_locks(last_phase_ind)
            pop.append(last_phase_ind)

            # Add some individuals with clusters evenly distributed across phases
            for i in range(min(n // 4, 5)):
                even_dist_ind = creator.Individual([])
                for j in range(len(self.all_clusters)):
                    phase = (j % self.num_phases) + 1
                    even_dist_ind.append(phase)
                even_dist_ind = _apply_locks(even_dist_ind)
                pop.append(even_dist_ind)

            # Add some individuals with progressive phase assignments
            prog_ind = creator.Individual([])
            clusters_per_phase = max(1, len(self.all_clusters) // self.num_phases)
            for j in range(len(self.all_clusters)):
                phase = min(j // clusters_per_phase + 1, self.num_phases)
                prog_ind.append(phase)
            prog_ind = _apply_locks(prog_ind)
            pop.append(prog_ind)

            # Fill the rest with random individuals
            while len(pop) < n:
                ind = self.toolbox.individual()
                ind = _apply_locks(ind)
                pop.append(ind)

            return pop

        # Override the population creation function
        self.toolbox.register("population", custom_population)

        # Define a repair function to enforce genome validity and locks (no per-phase budget moves)
        def repair_individual(individual):
            """
            Repair function: keep genes within [1, num_phases] and re-apply locking of committed phases.
            Per-phase budget moves are intentionally disabled (Q2: only final cumulative budgets are enforced).
            """
            # Clamp genes to valid phase range (robust cast)
            for i, g in enumerate(individual):
                try:
                    gg = int(round(g))
                except Exception:
                    gg = 1
                if gg < 1:
                    gg = 1
                elif gg > self.num_phases:
                    gg = self.num_phases
                individual[i] = gg
            # Re-apply locks
            if getattr(self, 'lock_committed_early_phases', False) and getattr(self, 'locked_clusters_phase', None):
                cluster_to_index = {c: i for i, c in enumerate(self.all_clusters)}
                for c, p in self.locked_clusters_phase.items():
                    idx = cluster_to_index.get(c, None)
                    if idx is not None:
                        individual[idx] = int(p)
            return individual

        # Register genetic operators
        self.toolbox.register("evaluate", self._evaluate_individual)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutUniformInt, low=1, up=self.num_phases,
                              indpb=getattr(self, 'ga_mutation_indpb', 0.4))  # configurable mutation rate
        self.toolbox.register("select", tools.selTournament, tournsize=getattr(self, 'ga_tournament_size', 3))

        # Define a repair decorator that wraps the genetic operators
        def repair_decorator(func):
            def wrapper(*args, **kwargs):
                result = func(*args, **kwargs)

                # If the underlying operator returns a tuple (DEAP convention)
                if isinstance(result, tuple):
                    # Mutation typically returns a single-individual tuple
                    if len(result) == 1:
                        ind = result[0]
                        repaired = repair_individual(ind)
                        ind[:] = repaired
                        return (ind,)
                    # Crossover typically returns a pair of individuals
                    elif len(result) == 2:
                        ind1, ind2 = result
                        rep1 = repair_individual(ind1); ind1[:] = rep1
                        rep2 = repair_individual(ind2); ind2[:] = rep2
                        return (ind1, ind2)
                    else:
                        # Fallback: repair all and return a tuple of same length
                        repaired_inds = []
                        for ind in result:
                            rep = repair_individual(ind); ind[:] = rep
                            repaired_inds.append(ind)
                        return tuple(repaired_inds)

                # If a list is returned (rare for DEAP mate/mutate, but safe to support)
                if isinstance(result, list):
                    for i, ind in enumerate(result):
                        rep = repair_individual(ind); ind[:] = rep
                        result[i] = ind
                    return result

                # Unexpected type: return as-is
                return result

            return wrapper

        # Register the repair function normally (for direct use if needed)
        self.toolbox.register("repair", repair_individual)

        # Apply the decorator to mate and mutate
        self.toolbox.decorate("mate", repair_decorator)
        self.toolbox.decorate("mutate", repair_decorator)

        # Register the map function (use the built-in map function)
        self.toolbox.register("map", map)

    def _evaluate_individual(self, individual):
        """
        Evaluate the fitness of an individual using the new emissions calculation approach.

        Parameters:
        -----------
        individual : list
            List of phase assignments for each cluster

        Returns:
        --------
        tuple
            Fitness value(s) - single objective (NPV or ROI) or multi-objective (NPV/ROI and emissions)
        """
        # NEW: enforce integer genome
        individual[:] = [int(round(g)) for g in individual]

        # Set current individual for use in emissions calculation
        self.current_individual = individual

        # Convert individual to cluster-phase mapping
        cluster_phase_map = {cluster: phase for cluster, phase in zip(self.all_clusters, individual)}

        # Calculate total ROI and NPV across all phases
        total_roi = 0
        total_npv = 0

        # Check if budget constraints are satisfied (EARLY FEASIBILITY CHECK before Demand/LCA)
        phase_capex = [0] * self.num_phases
        phase_total_expenditure = [0] * self.num_phases
        # Early feasibility short-circuit: if any phase violates budgets, skip emissions calculation
        try:
            # Group clusters by phase
            _clusters_by_phase_early = {}
            for c, p in cluster_phase_map.items():
                if int(p) > 0:
                    _clusters_by_phase_early.setdefault(int(p), []).append(c)
            violated = False
            violated_phase = None
            violated_msg = None
            for ph in range(1, self.num_phases + 1):
                clusters = _clusters_by_phase_early.get(ph, [])
                if clusters:
                    capex_val, _ = self._calculate_phase_capex(clusters, ph)
                    if self.capex_budget_per_phase and ph-1 < len(self.capex_budget_per_phase) and capex_val > self.capex_budget_per_phase[ph-1]:
                        violated = True
                        violated_phase = ph
                        violated_msg = f"CAPEX {capex_val} > {self.capex_budget_per_phase[ph-1]}"
                        break
                    total_exp_val = self._calculate_phase_total_expenditure(clusters, ph)
                    if self.total_expenditure_budget_per_phase and ph-1 < len(self.total_expenditure_budget_per_phase) and total_exp_val > self.total_expenditure_budget_per_phase[ph-1]:
                        violated = True
                        violated_phase = ph
                        violated_msg = f"TotalExp {total_exp_val} > {self.total_expenditure_budget_per_phase[ph-1]}"
                        break
            if violated:
                ind_tuple = tuple(individual)
                try:
                    log().info(f"[Skip Emissions] Infeasible individual {ind_tuple} at phase {violated_phase}: {violated_msg}")
                except Exception:
                    pass
                if self.multi_objective_mode:
                    objectives = self.multi_objective_functions or ['NPV', 'emissions']
                    objectives = objectives[:3]
                    pen = []
                    for obj in objectives:
                        o = obj.lower() if isinstance(obj, str) else obj
                        if o == 'npv':
                            pen.append(-1e12)
                        elif o == 'roi':
                            pen.append(-1e6)
                        elif o == 'emissions':
                            pen.append(1e12)
                        elif o == 'total_capex':
                            pen.append(1e15)
                    return tuple(pen)
                else:
                    obj_lower = self.objective_function.lower() if isinstance(self.objective_function, str) else self.objective_function
                    if obj_lower == 'roi':
                        return (-1e6,)
                    elif obj_lower == 'emissions':
                        return (-1e12,)
                    elif obj_lower == 'total_capex':
                        return (1e15,)
                    else:
                        return (-1e12,)
        except Exception:
            # If early check fails for any reason, fall back to full evaluation
            pass

        # Calculate emissions using calculate_district_emissions_new
        ind_tuple = tuple(individual)
        if ind_tuple in self.emissions_cache:
            # Use cached emissions results if available
            phase_emissions, has_non_district_scale = self.emissions_cache[ind_tuple]
            log().debug(f"Using cached emissions for individual {ind_tuple}")
        else:
            # Set current_individual for use in calculate_district_emissions_new
            self.current_individual = individual

            # Calculate emissions using calculate_district_emissions_new
            district_emissions = self.calculate_district_emissions_new()

            # Convert district_emissions to the format expected by the rest of the code
            phase_emissions = {}
            for phase in range(1, self.num_phases + 1):
                if phase in district_emissions:
                    phase_emissions[phase] = {
                        'operation': district_emissions[phase]['district_operation_emission [t CO2eq/yr]']
                    }

            # Check if cluster 0 has non-DISTRICT scale systems
            # This check is already done inside calculate_district_emissions_new
            # but we need to extract the result for the rest of the evaluation
            has_non_district_scale = False

            # Cache the results
            self._cache_emissions_for_individual(ind_tuple, phase_emissions, has_non_district_scale)
            log().debug(f"Calculated and cached emissions for individual {ind_tuple}")

        # Calculate weighted emissions across all phases
        weighted_emissions = 0
        for phase in range(1, self.num_phases + 1):
            phase_duration = self.phase_durations[phase - 1]
            if phase in phase_emissions:
                weighted_emissions += phase_emissions[phase]['operation'] * phase_duration

        # Keep final phase emissions for backward compatibility
        final_phase_emissions = phase_emissions[self.num_phases][
            'operation'] if self.num_phases in phase_emissions else 0

        # Apply penalty if cluster 0 has non-DISTRICT scale systems
        if has_non_district_scale:
            log().debug(f"Applying penalty for non-DISTRICT scale systems in cluster 0 for individual {ind_tuple}")
            total_roi = -1000
            total_npv = -1000000
            weighted_emissions = 1000000  # Penalize weighted emissions for multi-objective mode
            final_phase_emissions = 1000000  # Also penalize final phase emissions for backward compatibility

        # Group clusters by phase
        clusters_by_phase = {}
        for cluster, phase in cluster_phase_map.items():
            if phase > 0:  # Skip unconnected clusters (phase 0)
                if phase not in clusters_by_phase:
                    clusters_by_phase[phase] = []
                clusters_by_phase[phase].append(cluster)

        # Calculate ROI, NPV, and costs for each phase
        for phase, clusters in clusters_by_phase.items():
            # Calculate CAPEX for this phase with interest rate adjustment
            capex, capex_components = self._calculate_phase_capex(clusters, phase)
            phase_capex[phase - 1] = capex

            # Calculate total expenditure for this phase
            total_expenditure = self._calculate_phase_total_expenditure(clusters, phase)
            phase_total_expenditure[phase - 1] = total_expenditure

            # Calculate ROI and NPV for this phase
            roi = self.calculate_roi(tuple(clusters), phase)
            npv = self.calculate_npv(tuple(clusters), phase)

            total_roi += roi
            total_npv += npv

        # Per-phase relaxed ceilings: hard violation (dominant penalty) or mild penalty
        per_phase_soft_violated = False
        soft_excess_sum = 0.0
        try:
            if getattr(self, 'enforce_per_phase_soft_ceiling', False):
                cap_soft = getattr(self, 'capex_soft_per_phase', []) or []
                te_soft = getattr(self, 'total_exp_soft_per_phase', []) or []
                for p_idx in range(self.num_phases):
                    # CAPEX soft exceedance
                    if p_idx < len(phase_capex) and p_idx < len(cap_soft):
                        cap_limit = cap_soft[p_idx]
                        if cap_limit is not None and np.isfinite(cap_limit):
                            if float(phase_capex[p_idx]) > float(cap_limit):
                                per_phase_soft_violated = True
                            soft_excess_sum += max(0.0, float(phase_capex[p_idx]) - float(cap_limit))
                    # Total Expenditure soft exceedance
                    if p_idx < len(phase_total_expenditure) and p_idx < len(te_soft):
                        te_limit = te_soft[p_idx]
                        if te_limit is not None and np.isfinite(te_limit):
                            if float(phase_total_expenditure[p_idx]) > float(te_limit):
                                per_phase_soft_violated = True
                            soft_excess_sum += max(0.0, float(phase_total_expenditure[p_idx]) - float(te_limit))
                # Apply penalties: always dominant (baseline-like)
                if per_phase_soft_violated:
                    total_roi = -1000
                    total_npv = -1000000
                    weighted_emissions = 10_000_000.0
        except Exception:
            # Be robust: never fail evaluation due to penalty computation
            pass

        # Final cumulative budget checks only (Q2): per-phase budgets are NOT enforced here
        budget_violated = False
        violation_amount = 0.0

        total_capex_sum = float(sum(phase_capex)) if phase_capex else 0.0
        total_totexp_sum = float(sum(phase_total_expenditure)) if phase_total_expenditure else 0.0

        if getattr(self, 'enforce_final_cumulative_capex', False) and (self.final_cumulative_capex_budget_total not in (None, 0)):
            if total_capex_sum > self.final_cumulative_capex_budget_total:
                violation_amount += (total_capex_sum - self.final_cumulative_capex_budget_total)
                log().info(f"Individual {ind_tuple} exceeds FINAL cumulative CAPEX budget: {total_capex_sum} > {self.final_cumulative_capex_budget_total}")
                budget_violated = True

        if getattr(self, 'enforce_final_cumulative_total_exp', False) and (self.final_cumulative_total_exp_budget_total not in (None, 0)):
            if total_totexp_sum > self.final_cumulative_total_exp_budget_total:
                violation_amount += (total_totexp_sum - self.final_cumulative_total_exp_budget_total)
                log().info(f"Individual {ind_tuple} exceeds FINAL cumulative Total Expenditure budget: {total_totexp_sum} > {self.final_cumulative_total_exp_budget_total}")
                budget_violated = True

        # Apply penalties if budget is violated
        if budget_violated:
            # Base penalty (make clearly dominated)
            total_roi = -1000
            total_npv = -1000000

            # For emissions objective, use a much larger penalty proportional to violation amount
            penalty_factor = max(1.0, violation_amount / 1_000_000.0)
            weighted_emissions = 10_000_000.0 * penalty_factor

        # Check GHG budget constraints if specified (apply in both single and multi-objective modes)
        if self.ghg_budget_per_phase:
            for phase in range(self.num_phases):
                if phase < self.num_phases and self.ghg_budget_per_phase[phase] > 0:
                    phase_ghg = phase_emissions[phase + 1][
                        'operation']  # +1 because phases are 1-indexed in the results
                    if phase_ghg > self.ghg_budget_per_phase[phase]:
                        # Apply penalty for exceeding GHG budget
                        log().info(
                            f"Individual {ind_tuple} exceeds GHG budget in phase {phase + 1}: {phase_ghg} > {self.ghg_budget_per_phase[phase]}")
                        total_roi = -1000
                        total_npv = -1000000
                        weighted_emissions = 1000000  # Penalize weighted emissions for multi-objective mode
                        final_phase_emissions = 1000000  # Also penalize final phase emissions for backward compatibility
                        break

        # Calculate discounted total CAPEX across all phases
        total_capex = sum(phase_capex)

        # Return fitness based on optimization mode
        if self.multi_objective_mode:
            # Return selected objectives
            fitness_values = []

            # If no objectives specified, use default (NPV/ROI and emissions)
            objectives = self.multi_objective_functions
            if not objectives or len(objectives) == 0:
                objectives = ['NPV', 'emissions']

            # Limit to 3 objectives maximum
            objectives = objectives[:3]

            # Add fitness values based on selected objectives
            for obj in objectives:
                # Convert to lowercase for case-insensitive comparison
                obj_lower = obj.lower() if isinstance(obj, str) else obj
                if obj_lower == 'npv':
                    fitness_values.append(total_npv)
                elif obj_lower == 'roi':
                    fitness_values.append(total_roi)
                elif obj_lower == 'emissions':
                    fitness_values.append(weighted_emissions)
                elif obj_lower == 'total_capex':
                    fitness_values.append(total_capex)

            return tuple(fitness_values)
        else:
            # Return single objective
            # Convert to lowercase for case-insensitive comparison
            obj_func_lower = self.objective_function.lower() if isinstance(self.objective_function,
                                                                           str) else self.objective_function
            if obj_func_lower == 'roi':
                return (total_roi,)
            elif obj_func_lower == 'emissions':
                # Return negative emissions since we want to minimize emissions
                # but the single-objective mode is set up for maximization
                return (-weighted_emissions,)
            else:  # Default to NPV
                return (total_npv,)

    def _verify_cluster0_supply_systems(self):
        """Verify that buildings in cluster 0 have DISTRICT scale supply systems"""
        cluster0_buildings = self._get_buildings_in_specific_cluster(0)
        if not cluster0_buildings:
            log().warning("No buildings found in cluster 0 (existing DTN)")
            return

        # Get supply systems
        supply_df = pd.read_csv(self.locator.get_building_supply())
        cluster0_supply = supply_df[supply_df['name'].isin(cluster0_buildings)]

        # Get supply system definitions
        heating_df = pd.read_csv(self.locator.get_database_assemblies_supply_heating())
        cooling_df = pd.read_csv(self.locator.get_database_assemblies_supply_cooling())
        dhw_df = pd.read_csv(self.locator.get_database_assemblies_supply_hot_water())

        # Check DISTRICT scale only for the relevant system based on network type
        nt = (self.network_type or '').upper()
        for _, row in cluster0_supply.iterrows():
            hs_code = row['supply_type_hs']
            cs_code = row['supply_type_cs']
            dhw_code = row['supply_type_dhw']

            hs_scale = heating_df[heating_df['code'] == hs_code]['scale'].iloc[0] if len(heating_df[heating_df['code'] == hs_code]) > 0 else 'UNKNOWN'
            cs_scale = cooling_df[cooling_df['code'] == cs_code]['scale'].iloc[0] if len(cooling_df[cooling_df['code'] == cs_code]) > 0 else 'UNKNOWN'
            dhw_scale = dhw_df[dhw_df['code'] == dhw_code]['scale'].iloc[0] if len(dhw_df[dhw_df['code'] == dhw_code]) > 0 else 'UNKNOWN'

            if nt == 'DC':
                # For DC, check only cooling
                if cs_scale != 'DISTRICT':
                    log().warning(
                        f"Building {row['name']} in cluster 0 (DC) does not use DISTRICT scale for cooling (CS={cs_scale}).")
            elif nt == 'DH':
                # For DH, check only heating
                if hs_scale != 'DISTRICT':
                    log().warning(
                        f"Building {row['name']} in cluster 0 (DH) does not use DISTRICT scale for heating (HS={hs_scale}).")
            else:
                # Fallback (unknown network type) – keep broader warning as a safe default
                if hs_scale != 'DISTRICT' or cs_scale != 'DISTRICT' or dhw_scale != 'DISTRICT':
                    log().warning(
                        f"Building {row['name']} in cluster 0 has non-DISTRICT systems: HS={hs_scale}, CS={cs_scale}, DHW={dhw_scale}.")

    def optimize(self, population_size=50, num_generations=30):
        """
        Run the genetic algorithm to find the optimal phase assignment.

        Parameters:
        -----------
        population_size : int
            Size of the population
        num_generations : int
            Number of generations

        Returns:
        --------
        dict or list
            Dictionary with the optimal solution (single-objective) or
            list of non-dominated solutions (multi-objective)
        """
        # Verify cluster 0 buildings have district supply systems
        self._verify_cluster0_supply_systems()

        # Create initial population
        pop = self.toolbox.population(n=population_size)

        # Add a list to track all evaluated individuals
        all_evaluated_individuals = []

        # Store the original evaluate function
        original_evaluate = self.toolbox.evaluate

        # Define a new evaluate function that tracks individuals
        def evaluate_and_track(individual):
            fitness = original_evaluate(individual)
            # Create a proper DEAP Individual copy using the creator
            if self.multi_objective_mode:
                ind_copy = creator.Individual(individual)
            else:
                ind_copy = creator.Individual(individual)
            ind_copy.fitness.values = fitness
            all_evaluated_individuals.append(ind_copy)
            return fitness

        # Replace the evaluate function in the toolbox
        self.toolbox.evaluate = evaluate_and_track

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)

        # Assign fitness values to individuals
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        if self.multi_objective_mode:
            # For multi-objective optimization, use NSGA-III

            # Determine number of objectives
            num_objectives = len(self.multi_objective_functions) if self.multi_objective_functions else 2
            num_objectives = min(num_objectives, 3)  # Limit to 3 objectives maximum

            # Reference point for NSGA-III (automatically determined)
            # Use different p values based on number of objectives to get reasonable number of reference points
            if num_objectives == 2:
                ref_points = tools.uniform_reference_points(num_objectives, p=12)
            elif num_objectives == 3:
                ref_points = tools.uniform_reference_points(num_objectives, p=6)
            else:
                ref_points = tools.uniform_reference_points(num_objectives, p=4)

            # Create the NSGA-III selection operator
            self.toolbox.register("select", tools.selNSGA3, ref_points=ref_points)

            # Track the Pareto front
            pareto = tools.ParetoFront()

            # Use the custom evaluation function in the algorithm
            algorithms.eaMuPlusLambda(pop, self.toolbox, mu=population_size,
                                      lambda_=population_size,
                                      cxpb=getattr(self, 'ga_crossover_probability', 0.5), mutpb=getattr(self, 'ga_mutation_probability', 0.2),
                                      ngen=num_generations,
                                      stats=None, halloffame=pareto, verbose=True)

            # Process the Pareto front solutions
            pareto_solutions = []
            for i, ind in enumerate(pareto):
                # Find the corresponding individual_id from all_evaluated_individuals
                individual_id = None
                ind_tuple = tuple(ind)

                # Search for this individual in all_evaluated_individuals to get its index
                for j, eval_ind in enumerate(all_evaluated_individuals):
                    if tuple(eval_ind) == ind_tuple:
                        individual_id = j
                        break

                if individual_id is None:
                    log().warning(f"Could not find individual_id for Pareto solution {i}")
                    individual_id = f"unknown_{i}"

                # Convert to cluster-phase mapping
                solution = {
                    'individual_id': individual_id,  # Use actual individual_id
                    'cluster_phase_map': {cluster: phase for cluster, phase in
                                          zip(self.all_clusters, ind) if phase > 0},
                    'phases': {}
                }

                # Add fitness values based on selected objectives
                objectives = self.multi_objective_functions
                if not objectives or len(objectives) == 0:
                    objectives = ['NPV', 'emissions']

                # Limit to 3 objectives maximum
                objectives = objectives[:3]

                # Add fitness values to solution
                for i, obj in enumerate(objectives):
                    solution[f'fitness_{obj}'] = ind.fitness.values[i]

                # Group clusters by phase
                for cluster, phase in solution['cluster_phase_map'].items():
                    if phase not in solution['phases']:
                        solution['phases'][phase] = []
                    solution['phases'][phase].append(cluster)

                # Calculate metrics for each phase
                for phase, clusters in solution['phases'].items():
                    solution[f'phase_{phase}_roi'] = self.calculate_roi(tuple(clusters), phase)
                    solution[f'phase_{phase}_npv'] = self.calculate_npv(tuple(clusters), phase)
                    solution[f'phase_{phase}_capex'], _ = self._calculate_phase_capex(clusters, phase)

                    # Get emissions for this phase
                    ind_tuple = tuple(ind)
                    if ind_tuple in self.emissions_cache:
                        phase_emissions, _ = self.emissions_cache[ind_tuple]
                        solution[f'phase_{phase}_emissions'] = phase_emissions[phase]['operation']

                pareto_solutions.append(solution)

            # Create a scatter plot of all solutions tested during multi-objective optimization
            self.plot_multi_objective_results(all_evaluated_individuals, pareto, self.multi_objective_functions)

            # Save metrics for all evaluated individuals
            self.save_all_evaluated_individuals(all_evaluated_individuals,
                                                self.locator.get_dtn_expansion_optimization_results_folder())

            return pareto_solutions

        else:
            # For single-objective optimization, use the original approach
            # Track the best individual
            hof = tools.HallOfFame(1)

            # Track statistics
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("avg", np.mean)
            stats.register("min", np.min)
            stats.register("max", np.max)

            # Use the custom evaluation function in the algorithm
            pop, logbook = algorithms.eaSimple(pop, self.toolbox, cxpb=getattr(self, 'ga_crossover_probability', 0.5), mutpb=getattr(self, 'ga_mutation_probability', 0.2),
                                               ngen=num_generations, stats=stats,
                                               halloffame=hof, verbose=True)

            # Get the best solution
            best_individual = hof[0]

            # Convert to cluster-phase mapping
            solution = {
                'genome': list(best_individual),
                'cluster_phase_map': {cluster: phase for cluster, phase in
                                      zip(self.all_clusters, best_individual) if phase > 0},
                'fitness': best_individual.fitness.values[0],
                'phases': {}
            }

            # Sync internal state to final best genome for downstream emissions/saving
            try:
                self.solution = solution
                self.current_individual = list(best_individual)
            except Exception:
                pass

            # Group clusters by phase
            for cluster, phase in solution['cluster_phase_map'].items():
                if phase not in solution['phases']:
                    solution['phases'][phase] = []
                solution['phases'][phase].append(cluster)

            # Calculate metrics for each phase
            for phase, clusters in solution['phases'].items():
                solution[f'phase_{phase}_roi'] = self.calculate_roi(tuple(clusters), phase)
                solution[f'phase_{phase}_npv'] = self.calculate_npv(tuple(clusters), phase)
                solution[f'phase_{phase}_capex'], _ = self._calculate_phase_capex(clusters, phase)

            # Save metrics for all evaluated individuals
            self.save_all_evaluated_individuals(all_evaluated_individuals,
                                                self.locator.get_dtn_expansion_optimization_results_folder())

            return solution

    def save_all_evaluated_individuals(self, all_individuals, output_dir):
        """
        Save metrics for all evaluated individuals during optimization to a CSV file.

        Parameters:
        -----------
        all_individuals : list
            List of all individuals evaluated during optimization
        output_dir : Path
            Output directory

        Returns:
        --------
        Path
            Path to the saved CSV file
        """
        # Force output into the dynamic temp scenario dtn_expansion folder
        output_dir = Path(self.locator.get_dynamic_dtn_optimization_temp_scenario_dtn_expansion_folder())
        output_dir.mkdir(parents=True, exist_ok=True)
        temp_root = Path(self.locator.get_dynamic_dtn_optimization_temp_scenario_folder())
        _assert_under_temp(output_dir, temp_root, "all_evaluated_individuals output folder")
        _log_io("ENSURE DIR all_evaluated_individuals", output_dir)

        # Create a list to store all individual metrics
        all_metrics = []

        # Process each individual
        for i, ind in enumerate(all_individuals):
            # Skip individuals without valid fitness
            if not hasattr(ind, 'fitness') or not hasattr(ind.fitness, 'values') or len(ind.fitness.values) == 0:
                continue

            # Create a dictionary to store metrics for this individual
            metrics = {
                'individual_id': i,
                'genome': str(list(ind)),
            }

            # Add fitness values
            if self.multi_objective_mode:
                objectives = self.multi_objective_functions
                if not objectives or len(objectives) == 0:
                    objectives = ['NPV', 'emissions']

                # Limit to 3 objectives maximum
                objectives = objectives[:3]
            else:
                # For single-objective mode, use the specified objective function
                objectives = [self.objective_function]

            # Add fitness values to metrics
            for j, obj in enumerate(objectives):
                if j < len(ind.fitness.values):
                    # For emissions in single-objective mode, we need to negate the value
                    # since we store it as negative for maximization
                    if not self.multi_objective_mode and obj.lower() == 'emissions':
                        metrics[f'fitness_{obj}'] = -ind.fitness.values[j]
                    else:
                        metrics[f'fitness_{obj}'] = ind.fitness.values[j]

            # Convert individual to cluster-phase mapping
            cluster_phase_map = {cluster: phase for cluster, phase in zip(self.all_clusters, ind)}

            # Group clusters by phase
            clusters_by_phase = {}
            for cluster, phase in cluster_phase_map.items():
                if phase > 0:  # Skip unconnected clusters (phase 0)
                    if phase not in clusters_by_phase:
                        clusters_by_phase[phase] = []
                    clusters_by_phase[phase].append(cluster)

            # Add phase information
            metrics['num_phases'] = len(clusters_by_phase)
            for phase, clusters in clusters_by_phase.items():
                metrics[f'phase_{phase}_clusters'] = str(clusters)
                metrics[f'phase_{phase}_num_clusters'] = len(clusters)

                # Calculate metrics for this phase
                if len(clusters) > 0:
                    metrics[f'phase_{phase}_roi'] = self.calculate_roi(tuple(clusters), phase)
                    metrics[f'phase_{phase}_npv'] = self.calculate_npv(tuple(clusters), phase)
                    metrics[f'phase_{phase}_capex'], _ = self._calculate_phase_capex(clusters, phase)

                    # Get emissions for this phase if available
                    ind_tuple = tuple(ind)
                    if ind_tuple in self.emissions_cache:
                        phase_emissions, _ = self.emissions_cache[ind_tuple]
                        if phase in phase_emissions:
                            metrics[f'phase_{phase}_emissions'] = phase_emissions[phase]['operation']

            # Add to list of all metrics
            all_metrics.append(metrics)

        # Convert to DataFrame
        df = pd.DataFrame(all_metrics)

        # Save to CSV
        csv_file = output_dir / f"all_evaluated_individuals_{self.network_type}.csv"
        _assert_under_temp(csv_file, temp_root, "all_evaluated_individuals CSV")
        _log_io("WRITE all_evaluated_individuals", csv_file)
        df.to_csv(csv_file, index=False)

        log().info(f"Metrics for all {len(all_metrics)} evaluated individuals saved to {csv_file}")

        return csv_file

    def save_detailed_results(self, results, output_dir, return_df=False):
        """
        Save detailed optimization results to CSV.

        Parameters:
        -----------
        results : list
            List of result dictionaries
        output_dir : Path
            Output directory
        return_df : bool, optional
            If True, return the DataFrame instead of saving to CSV

        Returns:
        --------
        Path or DataFrame
            Path to the detailed results file or the DataFrame if return_df is True
        """
        # Create a list to store detailed results
        detailed_results = []

        # Determine the network type and demand type
        if self.network_type == 'DH':
            demand_type = 'Qh'
        else:
            demand_type = 'Qc'

        # Process each solution in the results
        for original_result in results:
            phase_results = []
            if 'phases' in original_result and 'phase' not in original_result:

                # Multi-objective solution with multiple phases
                cumulative_clusters = []
                cumulative_clusters: list[int] = []

                # Initialize prev_cumulative_metrics with zeros
                prev_cumulative_metrics = {
                    'total_pipe_length_m': 0,
                    f'total_annual_{demand_type}_MWh': 0
                }

                # Add a row for phase 0 (existing DTN)
                # Get district emissions
                district_emissions = self.calculate_district_emissions_new()

                # Get buildings in cluster 0 (existing DTN)
                cluster0_buildings = self._get_buildings_in_specific_cluster(0)

                # Create phase 0 result
                phase0_result = {
                    'phase': 0,
                    'newly_connected_cluster(s)': '0',
                    'cumulative_cluster(s)': '0',
                    'new_cluster(s)_discounted_roi [-]': 0,  # No ROI for existing DTN
                    'new_cluster(s)_npv [USD]': 0,  # No NPV for existing DTN
                    'new_cluster(s)_capex [USD]': 0,  # No CAPEX for existing DTN
                    'district_operation_emission [t CO2eq/yr]': district_emissions.get(0, {}).get(
                        'district_operation_emission [t CO2eq/yr]', 0),
                    'new_cluster(s)_pipe_length [m]': 0,  # Will be updated if data is available
                    'cumulative_pipe_length [m]': 0,  # Will be updated if data is available
                    f'new_cluster(s)_annual_{demand_type} [MWh/yr]': 0,  # Will be updated if data is available
                    f'cumulative_annual_{demand_type} [MWh/yr]': 0,  # Will be updated if data is available
                    f'new_cluster(s)_linear_{demand_type}_density [MWh/km/yr]': 0,
                    # Will be calculated if data is available
                    f'overall_linear_{demand_type}_density [MWh/km/yr]': 0,  # Will be calculated if data is available
                    'individual': original_result.get('individual'),

                    # CAPEX components
                    'new_cluster(s)_pipe_capex [USD]': 0,  # No CAPEX for existing DTN
                    'new_cluster(s)_hex_capex [USD]': 0,  # No CAPEX for existing DTN
                    'new_cluster(s)_pump_capex [USD]': 0,  # No CAPEX for existing DTN
                    'new_cluster(s)_cooling_plant_capex [USD]': 0,  # No CAPEX for existing DTN
                    'new_cluster(s)_total_capex [USD]': 0,  # No CAPEX for existing DTN

                    # Cumulative CAPEX components
                    'cumulative_pipe_capex [USD]': 0,  # No CAPEX for existing DTN
                    'cumulative_hex_capex [USD]': 0,  # No CAPEX for existing DTN
                    'cumulative_pump_capex [USD]': 0,  # No CAPEX for existing DTN
                    'cumulative_cooling_plant_capex [USD]': 0,  # No CAPEX for existing DTN
                    'cumulative_total_capex [USD]': 0,  # No CAPEX for existing DTN

                    # Annualized CAPEX components
                    'new_cluster(s)_pipe_annual_capex [USD/yr]': 0,  # No CAPEX for existing DTN
                    'new_cluster(s)_hex_annual_capex [USD/yr]': 0,  # No CAPEX for existing DTN
                    'new_cluster(s)_pump_annual_capex [USD/yr]': 0,  # No CAPEX for existing DTN
                    'new_cluster(s)_cooling_plant_annual_capex [USD/yr]': 0,  # No CAPEX for existing DTN
                    'new_cluster(s)_total_annual_capex [USD/yr]': 0,  # No CAPEX for existing DTN

                    # Cumulative annualized CAPEX components
                    'cumulative_pipe_annual_capex [USD/yr]': 0,  # No CAPEX for existing DTN
                    'cumulative_hex_annual_capex [USD/yr]': 0,  # No CAPEX for existing DTN
                    'cumulative_pump_annual_capex [USD/yr]': 0,  # No CAPEX for existing DTN
                    'cumulative_cooling_plant_annual_capex [USD/yr]': 0,  # No CAPEX for existing DTN
                    'cumulative_total_annual_capex [USD/yr]': 0,  # No CAPEX for existing DTN

                    # O&M costs - fixed components
                    'new_cluster(s)_pipe_annual_fixed_om [USD/yr]': 0,  # Will be updated if data is available
                    'new_cluster(s)_hex_annual_fixed_om [USD/yr]': 0,  # Will be updated if data is available
                    'new_cluster(s)_pump_annual_fixed_om [USD/yr]': 0,  # Will be updated if data is available
                    'new_cluster(s)_cooling_plant_annual_fixed_om [USD/yr]': 0,  # Will be updated if data is available
                    'new_cluster(s)_total_annual_fixed_om [USD/yr]': 0,  # Will be updated if data is available

                    # Cumulative O&M costs - fixed components
                    'cumulative_pipe_annual_fixed_om [USD/yr]': 0,  # Will be updated if data is available
                    'cumulative_hex_annual_fixed_om [USD/yr]': 0,  # Will be updated if data is available
                    'cumulative_pump_annual_fixed_om [USD/yr]': 0,  # Will be updated if data is available
                    'cumulative_cooling_plant_annual_fixed_om [USD/yr]': 0,  # Will be updated if data is available
                    'cumulative_total_annual_fixed_om [USD/yr]': 0,  # Will be updated if data is available

                    # O&M costs - variable components
                    'new_cluster(s)_pump_annual_variable_om [USD/yr]': 0,  # Will be updated if data is available
                    'new_cluster(s)_cooling_plant_annual_variable_om [USD/yr]': 0,
                    # Will be updated if data is available
                    'new_cluster(s)_total_annual_variable_om [USD/yr]': 0,  # Will be updated if data is available

                    # Cumulative O&M costs - variable components
                    'cumulative_pump_annual_variable_om [USD/yr]': 0,  # Will be updated if data is available
                    'cumulative_cooling_plant_annual_variable_om [USD/yr]': 0,  # Will be updated if data is available
                    'cumulative_total_annual_variable_om [USD/yr]': 0,  # Will be updated if data is available

                    # O&M costs - total (for backward compatibility)
                    'new_cluster(s)_pipe_annual_om [USD/yr]': 0,  # Will be updated if data is available
                    'new_cluster(s)_hex_annual_om [USD/yr]': 0,  # Will be updated if data is available
                    'new_cluster(s)_pump_annual_om [USD/yr]': 0,  # Will be updated if data is available
                    'new_cluster(s)_cooling_plant_annual_om [USD/yr]': 0,  # Will be updated if data is available
                    'new_cluster(s)_total_annual_om [USD/yr]': 0,  # Will be updated if data is available

                    # Cumulative O&M costs - total
                    'cumulative_pipe_annual_om [USD/yr]': 0,  # Will be updated if data is available
                    'cumulative_hex_annual_om [USD/yr]': 0,  # Will be updated if data is available
                    'cumulative_pump_annual_om [USD/yr]': 0,  # Will be updated if data is available
                    'cumulative_cooling_plant_annual_om [USD/yr]': 0,  # Will be updated if data is available
                    'cumulative_total_annual_om [USD/yr]': 0,  # Will be updated if data is available

                    # Energy consumption
                    'new_cluster(s)_pump_electricity [kWh/yr]': 0,  # Will be updated if data is available
                    'new_cluster(s)_cooling_plant_electricity [kWh/yr]': 0,  # Will be updated if data is available

                    # Cumulative energy consumption
                    'cumulative_pump_electricity [kWh/yr]': 0,  # Will be updated if data is available
                    'cumulative_cooling_plant_electricity [kWh/yr]': 0  # Will be updated if data is available
                }

                # Total expenditure for phase 0 is just CAPEX (no O&M)
                total_exp_p0 = self._calculate_phase_total_expenditure((0,), 0)

                # Annual O&M (simple 2.5 % of CAPEX) - set to 0 for phase 0
                annual_om_p0 = 0

                # For phase 0, revenue should be 0 (no duration, so no "annual demand" and "heat sales")
                annual_rev_p0 = 0

                # One-phase ROI and NPV - for phase 0, ROI=0 and NPV=-CAPEX
                roi_p0 = self.calculate_roi((0,), 0)
                npv_p0 = self.calculate_npv((0,), 0)

                # Write back into the dictionary
                phase0_result.update({
                    'new_cluster(s)_total_expenditure [USD]': total_exp_p0,
                    'cumulative_total_expenditure [USD]': total_exp_p0,
                    'new_cluster(s)_revenue [USD]': annual_rev_p0,
                    'cumulative_revenue [USD]': annual_rev_p0,
                    'new_cluster(s)_om_cost [USD]': annual_om_p0,
                    'cumulative_om_cost [USD]': annual_om_p0,
                    'new_cluster(s)_discounted_roi [-]': roi_p0,
                    'overall_discounted_roi [-]': roi_p0,
                    'new_cluster(s)_npv [USD]': npv_p0,
                    'overall_npv [USD]': npv_p0,
                })

                # Try to get metrics for cluster 0 if available
                cluster0_key = '0'
                annual_demand = 0
                pipe_length = 0
                pipe_length_recalculated = 0

                if cluster0_key in self.cluster_metrics:
                    metrics = self.cluster_metrics[cluster0_key]
                    annual_demand = metrics.get(f'total_annual_{demand_type}_MWh', 0)
                    pipe_length = metrics.get('total_pipe_length_m', 0)

                # Get required pipes for cluster 0
                required_pipes = self.get_required_pipes_for_clusters((0,))
                pipe_length_recalculated = required_pipes['length_m'].sum()

                # Update phase 0 metrics
                phase0_result['new_cluster(s)_pipe_length [m]'] = pipe_length_recalculated
                phase0_result['cumulative_pipe_length [m]'] = pipe_length_recalculated
                phase0_result[f'new_cluster(s)_annual_{demand_type} [MWh/yr]'] = annual_demand
                phase0_result[f'cumulative_annual_{demand_type} [MWh/yr]'] = annual_demand

                if cluster0_key in self.cluster_metrics:
                    metrics = self.cluster_metrics[cluster0_key]
                    phase0_result[f'new_cluster(s)_linear_{demand_type}_density [MWh/km/yr]'] = metrics.get(
                        f'linear_{demand_type}_density_MWh_per_km', 0)

                    # Recalculate overall linear heat density for consistency
                    if pipe_length_recalculated > 0:
                        phase0_result[f'overall_linear_{demand_type}_density [MWh/km/yr]'] = annual_demand / (
                                    pipe_length_recalculated / 1000)
                    else:
                        phase0_result[f'overall_linear_{demand_type}_density [MWh/km/yr]'] = metrics.get(
                            f'linear_{demand_type}_density_MWh_per_km', 0)

                # Calculate CAPEX components for cluster 0
                pipe_capex = self.calculate_detailed_capex((0,))
                hex_capex = self.calculate_hex_costs((0,))
                pump_capex, pump_electricity = self.calculate_pump_costs((0,))
                cooling_plant_capex = 0
                cooling_plant_electricity = 0
                if self.network_type == 'DC':
                    cooling_plant_capex, cooling_plant_electricity = self.calculate_cooling_plant_costs((0,))

                # Update phase0_result with calculated CAPEX components
                phase0_result['new_cluster(s)_pipe_capex [USD]'] = pipe_capex
                phase0_result['new_cluster(s)_hex_capex [USD]'] = hex_capex
                phase0_result['new_cluster(s)_pump_capex [USD]'] = pump_capex
                phase0_result['new_cluster(s)_cooling_plant_capex [USD]'] = cooling_plant_capex

                # Update cumulative CAPEX components
                phase0_result['cumulative_pipe_capex [USD]'] = pipe_capex
                phase0_result['cumulative_hex_capex [USD]'] = hex_capex
                phase0_result['cumulative_pump_capex [USD]'] = pump_capex
                phase0_result['cumulative_cooling_plant_capex [USD]'] = cooling_plant_capex

                # Explicitly calculate cumulative_total_capex for phase 0 as the sum of individual components
                cumulative_total_capex = (
                        pipe_capex +  # Use the direct variables instead of dict lookup
                        hex_capex +
                        pump_capex +
                        cooling_plant_capex
                )

                # Assign the calculated total CAPEX
                phase0_result['cumulative_total_capex [USD]'] = cumulative_total_capex

                # Set total_capex_p0 for use elsewhere in the code
                total_capex_p0 = cumulative_total_capex

                # Debug logging to verify the calculation
                log().debug(
                    f"Phase 0 CAPEX components: pipe={pipe_capex}, hex={hex_capex}, pump={pump_capex}, cooling={cooling_plant_capex}")
                log().debug(f"Phase 0 cumulative_total_capex calculated as: {cumulative_total_capex}")

                # Calculate annualized costs (assuming 25 year lifetime and 5% interest rate)
                pipe_annual_capex = calc_capex_annualized(pipe_capex, 5, 25)
                hex_annual_capex = calc_capex_annualized(hex_capex, 5, 25)
                pump_annual_capex = calc_capex_annualized(pump_capex, 5, 25)
                cooling_plant_annual_capex = calc_capex_annualized(cooling_plant_capex, 5, 25)
                total_annual_capex = pipe_annual_capex + hex_annual_capex + pump_annual_capex + cooling_plant_annual_capex

                # Update annualized CAPEX components
                phase0_result['new_cluster(s)_pipe_annual_capex [USD/yr]'] = pipe_annual_capex
                phase0_result['new_cluster(s)_hex_annual_capex [USD/yr]'] = hex_annual_capex
                phase0_result['new_cluster(s)_pump_annual_capex [USD/yr]'] = pump_annual_capex
                phase0_result['new_cluster(s)_cooling_plant_annual_capex [USD/yr]'] = cooling_plant_annual_capex
                phase0_result['new_cluster(s)_total_annual_capex [USD/yr]'] = total_annual_capex

                # Update cumulative annualized CAPEX components
                phase0_result['cumulative_pipe_annual_capex [USD/yr]'] = pipe_annual_capex
                phase0_result['cumulative_hex_annual_capex [USD/yr]'] = hex_annual_capex
                phase0_result['cumulative_pump_annual_capex [USD/yr]'] = pump_annual_capex
                phase0_result['cumulative_cooling_plant_annual_capex [USD/yr]'] = cooling_plant_annual_capex
                phase0_result['cumulative_total_annual_capex [USD/yr]'] = total_annual_capex

                # Calculate O&M costs
                # Fixed O&M costs (infrastructure components like pipes and HEX use 1.5% of CAPEX)
                pipe_annual_fixed_om = 0.015 * pipe_capex
                hex_annual_fixed_om = 0.015 * hex_capex

                # Fixed O&M costs (equipment like pumps and cooling plants use 1.0% of CAPEX)
                pump_annual_fixed_om = 0.01 * pump_capex
                cooling_plant_annual_fixed_om = 0.01 * cooling_plant_capex

                # Update O&M costs - fixed components
                phase0_result['new_cluster(s)_pipe_annual_fixed_om [USD/yr]'] = pipe_annual_fixed_om
                phase0_result['new_cluster(s)_hex_annual_fixed_om [USD/yr]'] = hex_annual_fixed_om
                phase0_result['new_cluster(s)_pump_annual_fixed_om [USD/yr]'] = pump_annual_fixed_om
                phase0_result['new_cluster(s)_cooling_plant_annual_fixed_om [USD/yr]'] = cooling_plant_annual_fixed_om
                phase0_result[
                    'new_cluster(s)_total_annual_fixed_om [USD/yr]'] = pipe_annual_fixed_om + hex_annual_fixed_om + pump_annual_fixed_om + cooling_plant_annual_fixed_om

                # Update cumulative O&M costs - fixed components
                phase0_result['cumulative_pipe_annual_fixed_om [USD/yr]'] = pipe_annual_fixed_om
                phase0_result['cumulative_hex_annual_fixed_om [USD/yr]'] = hex_annual_fixed_om
                phase0_result['cumulative_pump_annual_fixed_om [USD/yr]'] = pump_annual_fixed_om
                phase0_result['cumulative_cooling_plant_annual_fixed_om [USD/yr]'] = cooling_plant_annual_fixed_om
                phase0_result[
                    'cumulative_total_annual_fixed_om [USD/yr]'] = pipe_annual_fixed_om + hex_annual_fixed_om + pump_annual_fixed_om + cooling_plant_annual_fixed_om

                # Variable O&M costs based on electricity consumption
                # Get electricity price (USD/kWh)
                from cea.technologies.supply_systems_database import SupplySystemsDatabase
                try:
                    supply_systems = SupplySystemsDatabase(self.locator)
                    from cea.optimization.prices import Prices  # Import Prices class
                    prices = Prices(supply_systems)
                    electricity_price = np.mean(prices.ELEC_PRICE, dtype=np.float64)  # [USD/W]
                except Exception as e:
                    log().warning(f"Could not get electricity price: {e}. Using default value of 0.1 USD/kWh")
                    electricity_price = 0.1 / 1000  # Convert from USD/kWh to USD/W

                # Calculate variable O&M costs
                pump_annual_variable_om = pump_electricity * electricity_price  # Wh * USD/W = USD
                cooling_plant_annual_variable_om = cooling_plant_electricity * 1000 * electricity_price  # kWh * 1000 * USD/W = USD

                # Update O&M costs - variable components
                phase0_result['new_cluster(s)_pump_annual_variable_om [USD/yr]'] = pump_annual_variable_om
                phase0_result[
                    'new_cluster(s)_cooling_plant_annual_variable_om [USD/yr]'] = cooling_plant_annual_variable_om
                phase0_result[
                    'new_cluster(s)_total_annual_variable_om [USD/yr]'] = pump_annual_variable_om + cooling_plant_annual_variable_om

                # Update cumulative O&M costs - variable components
                phase0_result['cumulative_pump_annual_variable_om [USD/yr]'] = pump_annual_variable_om
                phase0_result['cumulative_cooling_plant_annual_variable_om [USD/yr]'] = cooling_plant_annual_variable_om
                phase0_result[
                    'cumulative_total_annual_variable_om [USD/yr]'] = pump_annual_variable_om + cooling_plant_annual_variable_om

                # Total O&M costs
                pipe_annual_om = pipe_annual_fixed_om
                hex_annual_om = hex_annual_fixed_om
                pump_annual_om = pump_annual_fixed_om + pump_annual_variable_om
                cooling_plant_annual_om = cooling_plant_annual_fixed_om + cooling_plant_annual_variable_om
                total_annual_om = pipe_annual_om + hex_annual_om + pump_annual_om + cooling_plant_annual_om

                # Update O&M costs - total
                phase0_result['new_cluster(s)_pipe_annual_om [USD/yr]'] = pipe_annual_om
                phase0_result['new_cluster(s)_hex_annual_om [USD/yr]'] = hex_annual_om
                phase0_result['new_cluster(s)_pump_annual_om [USD/yr]'] = pump_annual_om
                phase0_result['new_cluster(s)_cooling_plant_annual_om [USD/yr]'] = cooling_plant_annual_om
                phase0_result['new_cluster(s)_total_annual_om [USD/yr]'] = total_annual_om

                # Update cumulative O&M costs - total
                phase0_result['cumulative_pipe_annual_om [USD/yr]'] = pipe_annual_om
                phase0_result['cumulative_hex_annual_om [USD/yr]'] = hex_annual_om
                phase0_result['cumulative_pump_annual_om [USD/yr]'] = pump_annual_om
                phase0_result['cumulative_cooling_plant_annual_om [USD/yr]'] = cooling_plant_annual_om
                phase0_result['cumulative_total_annual_om [USD/yr]'] = total_annual_om

                # Update energy consumption
                phase0_result['new_cluster(s)_pump_electricity [kWh/yr]'] = pump_electricity / 1000  # Convert Wh to kWh
                phase0_result['new_cluster(s)_cooling_plant_electricity [kWh/yr]'] = cooling_plant_electricity

                # Update cumulative energy consumption
                phase0_result['cumulative_pump_electricity [kWh/yr]'] = pump_electricity / 1000
                phase0_result['cumulative_cooling_plant_electricity [kWh/yr]'] = cooling_plant_electricity

                # Update prev_cumulative_metrics for phase 1
                prev_cumulative_metrics = {
                    'total_pipe_length_m': pipe_length_recalculated,
                    f'total_annual_{demand_type}_MWh': annual_demand
                }

                # Add phase 0 result to phase_results
                phase_results.append(phase0_result)

                # Process each phase, including empty phases
                for phase_num in range(1, self.num_phases + 1):
                    # Check if this phase exists in the solution
                    if phase_num in original_result['phases']:
                        newly_connected = sorted(original_result['phases'][phase_num])
                    else:
                        # Empty phase - no clusters connected
                        newly_connected = []

                    # Update cumulative clusters (only if there are newly connected clusters)
                    if newly_connected:
                        cumulative_clusters.extend(newly_connected)
                        cumulative_clusters = sorted(set(cumulative_clusters))

                    cumulative_key = '+'.join(map(str, cumulative_clusters))
                    cumulative_metrics = self.cluster_metrics.get(cumulative_key, {})

                    # Calculate incremental metrics
                    new_pipe_length = 0
                    new_annual_demand = 0
                    new_linear_density = 0

                    if newly_connected:
                        new_pipe_length = (
                                cumulative_metrics.get('total_pipe_length_m', 0)
                                - prev_cumulative_metrics['total_pipe_length_m']
                        )
                        new_annual_demand = (
                                cumulative_metrics.get(f'total_annual_{demand_type}_MWh', 0)
                                - prev_cumulative_metrics[f'total_annual_{demand_type}_MWh']
                        )
                        new_linear_density = (
                            new_annual_demand / (new_pipe_length / 1000)
                            if new_pipe_length > 0 else 0
                        )

                    # Create phase result
                    phase_result = {
                        'phase': phase_num,
                        'newly_connected_cluster(s)': '+'.join(
                            map(str, newly_connected)) if newly_connected else 'none',
                        'cumulative_cluster(s)': '+'.join(map(str, [0] + cumulative_clusters)),

                        'new_cluster(s)_discounted_roi [-]': original_result.get(f'phase_{phase_num}_roi', 0),
                        'new_cluster(s)_npv [USD]': original_result.get(f'phase_{phase_num}_npv', 0),
                        'new_cluster(s)_capex [USD]': original_result.get(f'phase_{phase_num}_capex', 0),
                        'district_operation_emission [t CO2eq/yr]':
                            original_result.get(f'phase_{phase_num}_emissions', 0),

                        'new_cluster(s)_pipe_length [m]': new_pipe_length,
                        'cumulative_pipe_length [m]':
                            cumulative_metrics.get('total_pipe_length_m', 0),

                        f'new_cluster(s)_annual_{demand_type} [MWh/yr]': new_annual_demand,
                        f'cumulative_annual_{demand_type} [MWh/yr]':
                            cumulative_metrics.get(f'total_annual_{demand_type}_MWh', 0),

                        f'new_cluster(s)_linear_{demand_type}_density [MWh/km/yr]': new_linear_density,
                        f'overall_linear_{demand_type}_density [MWh/km/yr]':
                            cumulative_metrics.get(f'linear_{demand_type}_density_MWh_per_km', 0),

                        'individual': original_result.get('individual')
                    }
                    phase_results.append(phase_result)

                    # ------------------------------------------------------------------
                    # 4) store current cumulative values for next iteration
                    # ------------------------------------------------------------------
                    prev_cumulative_metrics = {
                        'total_pipe_length_m': cumulative_metrics.get('total_pipe_length_m', 0),
                        f'total_annual_{demand_type}_MWh':
                            cumulative_metrics.get(f'total_annual_{demand_type}_MWh', 0)
                    }
            else:
                # Single-objective solution or already processed phase
                if 'phase' in original_result:
                    phase_results.append(original_result)

            # Process each phase result (either from multi-objective or single-objective)
            for result in phase_results:
                phase = result['phase']

                # Get clusters for this phase
                if result['newly_connected_cluster(s)'] == 'none':
                    clusters = []
                    # For phases with no newly connected clusters, set all CAPEX components to 0
                    pipe_capex = 0
                    hex_capex = 0
                    pump_capex = 0
                    cooling_plant_capex = 0
                    pump_electricity = 0
                    cooling_plant_electricity = 0
                else:
                    clusters = [int(c) for c in result['newly_connected_cluster(s)'].split('+')]

                    # Calculate detailed CAPEX components
                    pipe_capex = self.calculate_detailed_capex(tuple(clusters))

                    # Get buildings in the clusters
                    buildings = self._get_buildings_in_clusters(tuple(clusters))

                    # Calculate heat exchanger costs
                    hex_capex = self.calculate_hex_costs(tuple(clusters))

                    # Calculate pump costs
                    pump_capex, pump_electricity = self.calculate_pump_costs(tuple(clusters))

                    # Calculate cooling plant costs for DC networks
                    if self.network_type == 'DC':
                        cooling_plant_capex, cooling_plant_electricity = self.calculate_cooling_plant_costs(
                            tuple(clusters))
                    else:
                        cooling_plant_capex = 0
                        cooling_plant_electricity = 0

                # Calculate annualized costs (assuming 25 year lifetime and 5% interest rate)
                pipe_annual_capex = calc_capex_annualized(pipe_capex, 5, 25)
                hex_annual_capex = calc_capex_annualized(hex_capex, 5, 25)
                pump_annual_capex = calc_capex_annualized(pump_capex, 5, 25)
                cooling_plant_annual_capex = calc_capex_annualized(cooling_plant_capex, 5, 25)

                total_capex = (
                        pipe_capex +
                        hex_capex +
                        pump_capex +
                        cooling_plant_capex
                )

                # UPDATE new-cluster column
                result['new_cluster(s)_capex [USD]'] = total_capex

                # Calculate O&M costs
                # Fixed O&M costs (infrastructure components like pipes and HEX use 1.5% of CAPEX)
                pipe_annual_fixed_om = 0.015 * pipe_capex
                hex_annual_fixed_om = 0.015 * hex_capex

                # Fixed O&M costs (equipment like pumps and cooling plants use 1.0% of CAPEX)
                pump_annual_fixed_om = 0.01 * pump_capex
                cooling_plant_annual_fixed_om = 0.01 * cooling_plant_capex

                # Variable O&M costs based on electricity consumption
                # Get electricity price (USD/kWh)
                from cea.technologies.supply_systems_database import SupplySystemsDatabase
                try:
                    supply_systems = SupplySystemsDatabase(self.locator)
                    from cea.optimization.prices import Prices  # Import Prices class
                    prices = Prices(supply_systems)
                    electricity_price = np.mean(prices.ELEC_PRICE, dtype=np.float64)  # [USD/W]
                    log().debug(f"Using electricity price: {electricity_price} USD/W")
                except Exception as e:
                    log().warning(f"Could not get electricity price: {e}. Using default value of 0.1 USD/kWh")
                    electricity_price = 0.1 / 1000  # Convert from USD/kWh to USD/W

                # Calculate variable O&M costs
                pump_annual_variable_om = pump_electricity * electricity_price  # Wh * USD/W = USD
                cooling_plant_annual_variable_om = cooling_plant_electricity * 1000 * electricity_price  # kWh * 1000 * USD/W = USD

                # Total O&M costs
                pipe_annual_om = pipe_annual_fixed_om
                hex_annual_om = hex_annual_fixed_om
                pump_annual_om = pump_annual_fixed_om + pump_annual_variable_om
                cooling_plant_annual_om = cooling_plant_annual_fixed_om + cooling_plant_annual_variable_om

                # Get previous cumulative values from the last detailed result
                prev_cumulative_values = {}
                if detailed_results:
                    prev_result = detailed_results[-1]
                    for key in prev_result:
                        if key.startswith('cumulative_'):
                            prev_cumulative_values[key] = prev_result[key]

                # Create detailed result dictionary
                detailed_result = {
                    'phase': phase,
                    'newly_connected_cluster(s)': result['newly_connected_cluster(s)'],
                    'cumulative_cluster(s)': result['cumulative_cluster(s)'],

                    # CAPEX components
                    'new_cluster(s)_pipe_capex [USD]': pipe_capex,
                    'new_cluster(s)_hex_capex [USD]': hex_capex,
                    'new_cluster(s)_pump_capex [USD]': pump_capex,
                    'new_cluster(s)_cooling_plant_capex [USD]': cooling_plant_capex,
                    'new_cluster(s)_total_capex [USD]': total_capex,

                    # Cumulative CAPEX components
                    'cumulative_pipe_capex [USD]': prev_cumulative_values.get('cumulative_pipe_capex [USD]',
                                                                              0) + pipe_capex,
                    'cumulative_hex_capex [USD]': prev_cumulative_values.get('cumulative_hex_capex [USD]',
                                                                             0) + hex_capex,
                    'cumulative_pump_capex [USD]': prev_cumulative_values.get('cumulative_pump_capex [USD]',
                                                                              0) + pump_capex,
                    'cumulative_cooling_plant_capex [USD]': prev_cumulative_values.get(
                        'cumulative_cooling_plant_capex [USD]', 0) + cooling_plant_capex,
                    'cumulative_total_capex [USD]': prev_cumulative_values.get('cumulative_total_capex [USD]',
                                                                               0) + total_capex,

                    # Annualized CAPEX components
                    'new_cluster(s)_pipe_annual_capex [USD/yr]': pipe_annual_capex,
                    'new_cluster(s)_hex_annual_capex [USD/yr]': hex_annual_capex,
                    'new_cluster(s)_pump_annual_capex [USD/yr]': pump_annual_capex,
                    'new_cluster(s)_cooling_plant_annual_capex [USD/yr]': cooling_plant_annual_capex,
                    'new_cluster(s)_total_annual_capex [USD/yr]': pipe_annual_capex + hex_annual_capex + pump_annual_capex + cooling_plant_annual_capex,

                    # Cumulative annualized CAPEX components
                    'cumulative_pipe_annual_capex [USD/yr]': prev_cumulative_values.get(
                        'cumulative_pipe_annual_capex [USD/yr]', 0) + pipe_annual_capex,
                    'cumulative_hex_annual_capex [USD/yr]': prev_cumulative_values.get(
                        'cumulative_hex_annual_capex [USD/yr]', 0) + hex_annual_capex,
                    'cumulative_pump_annual_capex [USD/yr]': prev_cumulative_values.get(
                        'cumulative_pump_annual_capex [USD/yr]', 0) + pump_annual_capex,
                    'cumulative_cooling_plant_annual_capex [USD/yr]': prev_cumulative_values.get(
                        'cumulative_cooling_plant_annual_capex [USD/yr]', 0) + cooling_plant_annual_capex,
                    'cumulative_total_annual_capex [USD/yr]': prev_cumulative_values.get(
                        'cumulative_total_annual_capex [USD/yr]',
                        0) + pipe_annual_capex + hex_annual_capex + pump_annual_capex + cooling_plant_annual_capex,

                    # O&M costs - fixed components
                    'new_cluster(s)_pipe_annual_fixed_om [USD/yr]': pipe_annual_fixed_om,
                    'new_cluster(s)_hex_annual_fixed_om [USD/yr]': hex_annual_fixed_om,
                    'new_cluster(s)_pump_annual_fixed_om [USD/yr]': pump_annual_fixed_om,
                    'new_cluster(s)_cooling_plant_annual_fixed_om [USD/yr]': cooling_plant_annual_fixed_om,
                    'new_cluster(s)_total_annual_fixed_om [USD/yr]': pipe_annual_fixed_om + hex_annual_fixed_om + \
                                                                     pump_annual_fixed_om + cooling_plant_annual_fixed_om,

                    # Cumulative O&M costs - fixed components
                    'cumulative_pipe_annual_fixed_om [USD/yr]': prev_cumulative_values.get(
                        'cumulative_pipe_annual_fixed_om [USD/yr]', 0) + pipe_annual_fixed_om,
                    'cumulative_hex_annual_fixed_om [USD/yr]': prev_cumulative_values.get(
                        'cumulative_hex_annual_fixed_om [USD/yr]', 0) + hex_annual_fixed_om,
                    'cumulative_pump_annual_fixed_om [USD/yr]': prev_cumulative_values.get(
                        'cumulative_pump_annual_fixed_om [USD/yr]', 0) + pump_annual_fixed_om,
                    'cumulative_cooling_plant_annual_fixed_om [USD/yr]': prev_cumulative_values.get(
                        'cumulative_cooling_plant_annual_fixed_om [USD/yr]', 0) + cooling_plant_annual_fixed_om,
                    'cumulative_total_annual_fixed_om [USD/yr]': prev_cumulative_values.get(
                        'cumulative_total_annual_fixed_om [USD/yr]', 0) + pipe_annual_fixed_om + hex_annual_fixed_om + \
                                                                 pump_annual_fixed_om + cooling_plant_annual_fixed_om,

                    # O&M costs - variable components
                    'new_cluster(s)_pump_annual_variable_om [USD/yr]': pump_annual_variable_om,
                    'new_cluster(s)_cooling_plant_annual_variable_om [USD/yr]': cooling_plant_annual_variable_om,
                    'new_cluster(s)_total_annual_variable_om [USD/yr]': pump_annual_variable_om + cooling_plant_annual_variable_om,

                    # Cumulative O&M costs - variable components
                    'cumulative_pump_annual_variable_om [USD/yr]': prev_cumulative_values.get(
                        'cumulative_pump_annual_variable_om [USD/yr]', 0) + pump_annual_variable_om,
                    'cumulative_cooling_plant_annual_variable_om [USD/yr]': prev_cumulative_values.get(
                        'cumulative_cooling_plant_annual_variable_om [USD/yr]', 0) + cooling_plant_annual_variable_om,
                    'cumulative_total_annual_variable_om [USD/yr]': prev_cumulative_values.get(
                        'cumulative_total_annual_variable_om [USD/yr]',
                        0) + pump_annual_variable_om + cooling_plant_annual_variable_om,

                    # O&M costs - total (for backward compatibility)
                    'new_cluster(s)_pipe_annual_om [USD/yr]': pipe_annual_om,
                    'new_cluster(s)_hex_annual_om [USD/yr]': hex_annual_om,
                    'new_cluster(s)_pump_annual_om [USD/yr]': pump_annual_om,
                    'new_cluster(s)_cooling_plant_annual_om [USD/yr]': cooling_plant_annual_om,
                    'new_cluster(s)_total_annual_om [USD/yr]': pipe_annual_om + hex_annual_om + pump_annual_om + cooling_plant_annual_om,

                    # Cumulative O&M costs - total
                    'cumulative_pipe_annual_om [USD/yr]': prev_cumulative_values.get(
                        'cumulative_pipe_annual_om [USD/yr]', 0) + pipe_annual_om,
                    'cumulative_hex_annual_om [USD/yr]': prev_cumulative_values.get('cumulative_hex_annual_om [USD/yr]',
                                                                                    0) + hex_annual_om,
                    'cumulative_pump_annual_om [USD/yr]': prev_cumulative_values.get(
                        'cumulative_pump_annual_om [USD/yr]', 0) + pump_annual_om,
                    'cumulative_cooling_plant_annual_om [USD/yr]': prev_cumulative_values.get(
                        'cumulative_cooling_plant_annual_om [USD/yr]', 0) + cooling_plant_annual_om,
                    'cumulative_total_annual_om [USD/yr]': prev_cumulative_values.get(
                        'cumulative_total_annual_om [USD/yr]',
                        0) + pipe_annual_om + hex_annual_om + pump_annual_om + cooling_plant_annual_om,

                    # Energy consumption
                    'new_cluster(s)_pump_electricity [kWh/yr]': pump_electricity / 1000,  # Convert Wh to kWh
                    'new_cluster(s)_cooling_plant_electricity [kWh/yr]': cooling_plant_electricity,

                    # Cumulative energy consumption
                    'cumulative_pump_electricity [kWh/yr]': prev_cumulative_values.get(
                        'cumulative_pump_electricity [kWh/yr]', 0) + pump_electricity / 1000,
                    'cumulative_cooling_plant_electricity [kWh/yr]': prev_cumulative_values.get(
                        'cumulative_cooling_plant_electricity [kWh/yr]', 0) + cooling_plant_electricity,

                    # Other metrics
                    # Get annual demand directly from cluster metrics
                    f'new_cluster(s)_annual_{demand_type} [MWh/yr]': self.cluster_metrics.get(
                        '+'.join(map(str, sorted(clusters))), {}).get(f'total_annual_{demand_type}_MWh', 0),
                    f'cumulative_annual_{demand_type} [MWh/yr]': result[f'cumulative_annual_{demand_type} [MWh/yr]'],
                    'new_cluster(s)_pipe_length [m]': result['new_cluster(s)_pipe_length [m]'],
                    'cumulative_pipe_length [m]': result['cumulative_pipe_length [m]'],
                    f'new_cluster(s)_linear_{demand_type}_density [MWh/km/yr]': result[
                        f'new_cluster(s)_linear_{demand_type}_density [MWh/km/yr]'],
                    f'overall_linear_{demand_type}_density [MWh/km/yr]': result[
                        f'overall_linear_{demand_type}_density [MWh/km/yr]']
                }

                # Add emissions data from the best individual
                if 'individual' in result and result['individual'] is not None:
                    ind_tuple = tuple(result['individual'])
                    if ind_tuple in self.emissions_cache:
                        emissions, _ = self.emissions_cache[ind_tuple]
                        # Add emissions data for this phase - only operational emissions
                        detailed_result['phase_operation_emissions_tonCO2'] = emissions[phase]['operation']

                        # Add cumulative emissions data (sum of all phases up to this one)
                        cumulative_operation = sum(emissions[p]['operation'] for p in range(1, phase + 1))
                        detailed_result['cumulative_operation_emissions_tonCO2'] = cumulative_operation
                        detailed_result['district_operation_emission_tonCO2'] = cumulative_operation

                detailed_results.append(detailed_result)

        # Create DataFrame with ordered columns
        # Define column order to put cumulative values next to new_cluster values
        column_order = [
            'solution_id', 'phase', 'newly_connected_cluster(s)', 'cumulative_cluster(s)',
            'new_cluster(s)_pipe_capex [USD]', 'cumulative_pipe_capex [USD]',
            'new_cluster(s)_hex_capex [USD]', 'cumulative_hex_capex [USD]',
            'new_cluster(s)_pump_capex [USD]', 'cumulative_pump_capex [USD]',
            'new_cluster(s)_cooling_plant_capex [USD]', 'cumulative_cooling_plant_capex [USD]',
            'new_cluster(s)_total_capex [USD]', 'cumulative_total_capex [USD]',
            'new_cluster(s)_pipe_annual_capex [USD/yr]', 'cumulative_pipe_annual_capex [USD/yr]',
            'new_cluster(s)_hex_annual_capex [USD/yr]', 'cumulative_hex_annual_capex [USD/yr]',
            'new_cluster(s)_pump_annual_capex [USD/yr]', 'cumulative_pump_annual_capex [USD/yr]',
            'new_cluster(s)_cooling_plant_annual_capex [USD/yr]', 'cumulative_cooling_plant_annual_capex [USD/yr]',
            'new_cluster(s)_total_annual_capex [USD/yr]', 'cumulative_total_annual_capex [USD/yr]',
            'new_cluster(s)_pipe_annual_fixed_om [USD/yr]', 'cumulative_pipe_annual_fixed_om [USD/yr]',
            'new_cluster(s)_hex_annual_fixed_om [USD/yr]', 'cumulative_hex_annual_fixed_om [USD/yr]',
            'new_cluster(s)_pump_annual_fixed_om [USD/yr]', 'cumulative_pump_annual_fixed_om [USD/yr]',
            'new_cluster(s)_cooling_plant_annual_fixed_om [USD/yr]',
            'cumulative_cooling_plant_annual_fixed_om [USD/yr]',
            'new_cluster(s)_total_annual_fixed_om [USD/yr]', 'cumulative_total_annual_fixed_om [USD/yr]',
            'new_cluster(s)_pump_annual_variable_om [USD/yr]', 'cumulative_pump_annual_variable_om [USD/yr]',
            'new_cluster(s)_cooling_plant_annual_variable_om [USD/yr]',
            'cumulative_cooling_plant_annual_variable_om [USD/yr]',
            'new_cluster(s)_total_annual_variable_om [USD/yr]', 'cumulative_total_annual_variable_om [USD/yr]',
            'new_cluster(s)_pipe_annual_om [USD/yr]', 'cumulative_pipe_annual_om [USD/yr]',
            'new_cluster(s)_hex_annual_om [USD/yr]', 'cumulative_hex_annual_om [USD/yr]',
            'new_cluster(s)_pump_annual_om [USD/yr]', 'cumulative_pump_annual_om [USD/yr]',
            'new_cluster(s)_cooling_plant_annual_om [USD/yr]', 'cumulative_cooling_plant_annual_om [USD/yr]',
            'new_cluster(s)_total_annual_om [USD/yr]', 'cumulative_total_annual_om [USD/yr]',
            'new_cluster(s)_om_cost [USD]', 'cumulative_om_cost [USD]',
            'new_cluster(s)_total_expenditure [USD]', 'cumulative_total_expenditure [USD]',
            'new_cluster(s)_revenue [USD]', 'cumulative_revenue [USD]',
            'new_cluster(s)_pump_electricity [kWh/yr]', 'cumulative_pump_electricity [kWh/yr]',
            'new_cluster(s)_cooling_plant_electricity [kWh/yr]', 'cumulative_cooling_plant_electricity [kWh/yr]',
            'new_cluster(s)_pipe_length [m]', 'cumulative_pipe_length [m]',
            f'new_cluster(s)_annual_{demand_type} [MWh/yr]', f'cumulative_annual_{demand_type} [MWh/yr]',
            f'new_cluster(s)_linear_{demand_type}_density [MWh/km/yr]',
            f'overall_linear_{demand_type}_density [MWh/km/yr]',
            'new_cluster(s)_discounted_roi [-]', 'overall_discounted_roi [-]',
            'new_cluster(s)_npv [USD]', 'overall_npv [USD]',
            'district_operation_emission [t CO2eq/yr]',
            'individual'
        ]

        # Filter column_order to only include columns that exist in detailed_results
        available_columns = set()
        for result in detailed_results:
            available_columns.update(result.keys())

        filtered_column_order = [col for col in column_order if col in available_columns]

        # Create DataFrame with ordered columns
        detailed_results_df = pd.DataFrame(detailed_results, columns=filtered_column_order)

        # Add any remaining columns that weren't in column_order
        remaining_columns = [col for col in available_columns if col not in filtered_column_order]
        if remaining_columns:
            remaining_df = pd.DataFrame(detailed_results)[remaining_columns]
            detailed_results_df = pd.concat([detailed_results_df, remaining_df], axis=1)

        # Apply formatting to specific columns
        # Integer formatting
        integer_columns = [col for col in detailed_results_df.columns if any(substr in col for substr in [
            'capex [USD]', 'total_expenditure [USD]', 'revenue [USD]', 'om_cost [USD]', 'npv [USD]', 'pipe_length [m]'
        ])]
        for col in integer_columns:
            if col in detailed_results_df.columns:
                detailed_results_df[col] = detailed_results_df[col].apply(
                    lambda x: int(round(x, 0)) if isinstance(x, (int, float)) and not pd.isna(x) else x)

        # 2 decimal places
        decimal2_columns = [col for col in detailed_results_df.columns if any(substr in col for substr in [
            'operation_emission', 'annual_Qh [MWh/yr]', 'annual_Qc [MWh/yr]', 'linear_Qh_density [MWh/km/yr]',
            'linear_Qc_density [MWh/km/yr]'
        ])]
        for col in decimal2_columns:
            if col in detailed_results_df.columns:
                detailed_results_df[col] = detailed_results_df[col].apply(
                    lambda x: round(x, 2) if isinstance(x, (int, float)) and not pd.isna(x) else x)

        # 4 decimal places
        decimal4_columns = [col for col in detailed_results_df.columns if 'discounted_roi [-]' in col]
        for col in decimal4_columns:
            if col in detailed_results_df.columns:
                detailed_results_df[col] = detailed_results_df[col].apply(
                    lambda x: round(x, 4) if isinstance(x, (int, float)) and not pd.isna(x) else x)

        # Save to CSV with network-type specific filename
        if return_df:
            return detailed_results_df
        else:
            detailed_results_file = output_dir / f"dtn_expansion_opt_results_{self.network_type}_detailed.csv"
            detailed_results_df.to_csv(detailed_results_file, index=False)
            return detailed_results_file

    def save_results(self, solution):
        """
        Save optimization results to CSV.

        Parameters:
        -----------
        solution : dict or list
            Optimization solution (single-objective) or list of solutions (multi-objective)

        Returns:
        --------
        Path
            Path to the results file
        """
        # Force output into the dynamic temp scenario dtn_expansion folder
        output_dir = Path(self.locator.get_dynamic_dtn_optimization_temp_scenario_dtn_expansion_folder())
        output_dir.mkdir(parents=True, exist_ok=True)
        temp_root = Path(self.locator.get_dynamic_dtn_optimization_temp_scenario_folder())
        _assert_under_temp(output_dir, temp_root, "results folder")
        _log_io("ENSURE DIR results", output_dir)

        # Sync to final best genome (if provided) so emissions and files reflect the chosen solution
        try:
            if isinstance(solution, dict):
                self.solution = solution
                if 'genome' in solution and solution['genome']:
                    self.current_individual = list(solution['genome'])
                    log().info(f"[FINAL] Best genome (objective={self.objective_function}): {solution['genome']}")
                    if 'phases' in solution and isinstance(solution['phases'], dict):
                        for ph in range(1, self.num_phases + 1):
                            clusters_ph = solution['phases'].get(ph, [])
                            log().info(f"[FINAL] Phase {ph}: Connecting clusters {clusters_ph}")
        except Exception:
            pass

        # Calculate district emissions using the new methodology
        district_emissions = self.calculate_district_emissions_new()

        # Write final-best metadata JSON for traceability
        try:
            meta = {
                "objective_function": self.objective_function,
                "genome": solution.get('genome', []) if isinstance(solution, dict) else None,
                "clusters": self.all_clusters,
                "num_phases": self.num_phases,
            }
            meta_path = output_dir / "final_best_genome.json"
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)
            log().info(f"[FINAL] Wrote {meta_path}")
        except Exception:
            pass

        # Check if this is a multi-objective result (list of solutions)
        if isinstance(solution, list):
            # For multi-objective, solution is a list of Pareto-optimal solutions
            # Create a summary dataframe for the Pareto front (keep this for backward compatibility)
            pareto_summary = []

            # Determine objectives
            objectives = self.multi_objective_functions
            if not objectives or len(objectives) == 0:
                objectives = ['NPV', 'emissions']

            # Limit to 3 objectives maximum
            objectives = objectives[:3]

            for i, sol in enumerate(solution):
                row = {
                    'solution_id': sol.get('individual_id', i),  # Use individual_id if available, fallback to i
                }

                # Add objective values with units
                for obj in objectives:
                    # Convert to lowercase for case-insensitive comparison
                    obj_lower = obj.lower() if isinstance(obj, str) else obj
                    if obj_lower == 'npv':
                        row[f'objective_{obj} [USD]'] = sol.get(f'fitness_{obj}', 0)
                    elif obj_lower == 'roi':
                        row[f'objective_{obj} [-]'] = sol.get(f'fitness_{obj}', 0)
                    elif obj_lower == 'emissions':
                        row[f'objective_{obj} [t CO2eq]'] = sol.get(f'fitness_{obj}', 0)
                    elif obj_lower == 'total_capex':
                        row[f'objective_{obj} [USD]'] = sol.get(f'fitness_{obj}', 0)

                # Add phase information
                for phase in range(1, self.num_phases + 1):
                    if phase in sol['phases']:
                        clusters = sol['phases'][phase]
                        row[f'phase_{phase}_clusters'] = '+'.join(map(str, clusters))
                        row[f'phase_{phase}_roi [-]'] = sol.get(f'phase_{phase}_roi', 0)
                        row[f'phase_{phase}_npv [USD]'] = sol.get(f'phase_{phase}_npv', 0)
                        row[f'phase_{phase}_capex [USD]'] = sol.get(f'phase_{phase}_capex', 0)
                        row[f'phase_{phase}_emissions [t CO2eq]'] = sol.get(f'phase_{phase}_emissions', 0)
                    else:
                        row[f'phase_{phase}_clusters'] = ''
                        row[f'phase_{phase}_roi [-]'] = 0
                        row[f'phase_{phase}_npv [USD]'] = 0
                        row[f'phase_{phase}_capex [USD]'] = 0
                        row[f'phase_{phase}_emissions [t CO2eq]'] = 0

                pareto_summary.append(row)

            # Save to CSV
            pareto_df = pd.DataFrame(pareto_summary)
            # Save Pareto summary in the temp dtn_expansion folder (not the original folder)
            pareto_file = output_dir / f"dtn_expansion_opt_pareto_{self.network_type}.csv"
            pareto_df.to_csv(pareto_file, index=False)

            # Create a combined detailed results file with all Pareto solutions
            # Each solution will be separated by an empty row
            combined_results = []

            for i, sol in enumerate(solution):
                # Add a header row identifying the solution
                header_row = {
                    'solution_id': sol.get('individual_id', i),  # Use individual_id if available
                    'phase': '',  # Empty phase for the header row
                }

                # Add objective values with units
                for obj in objectives:
                    # Convert to lowercase for case-insensitive comparison
                    obj_lower = obj.lower() if isinstance(obj, str) else obj
                    if obj_lower == 'npv':
                        header_row[f'objective_{obj} [USD]'] = sol.get(f'fitness_{obj}', 0)
                    elif obj_lower == 'roi':
                        header_row[f'objective_{obj} [-]'] = sol.get(f'fitness_{obj}', 0)
                    elif obj_lower == 'emissions':
                        header_row[f'objective_{obj} [t CO2eq]'] = sol.get(f'fitness_{obj}', 0)
                    elif obj_lower == 'total_capex':
                        header_row[f'objective_{obj} [USD]'] = sol.get(f'fitness_{obj}', 0)
                combined_results.append(header_row)

                # Get detailed results for this solution
                detailed_results = self.save_detailed_results([sol], output_dir, return_df=True)

                # Add the detailed results to the combined results
                combined_results.extend(detailed_results.to_dict('records'))

                # Add an empty row as separator (unless it's the last solution)
                if i < len(solution) - 1:
                    combined_results.append({})

            # Save the combined results to CSV
            combined_df = pd.DataFrame(combined_results)
            combined_file = output_dir / f"dtn_expansion_opt_results_{self.network_type}_detailed_combined.csv"
            combined_df.to_csv(combined_file, index=False)

            # Also save individual detailed results for each solution (for backward compatibility)
            # Save combined and per-solution detailed results also under output_dir
            # (you already save combined_file under output_dir; just ensure per-solution dirs also live under output_dir)
            for i, sol in enumerate(solution):
                pareto_dir = output_dir / f"pareto_solution_{i}"
                pareto_dir.mkdir(parents=True, exist_ok=True)
                self.save_detailed_results([sol], pareto_dir)

            return output_dir / f"dtn_expansion_opt_results_{self.network_type}_detailed_combined.csv"

        # For single-objective optimization, use the original approach
        # Create a DataFrame with the results
        results = []
        total_capex = 0
        total_npv = 0

        # Determine the network type and demand type
        if self.network_type == 'DH':
            demand_type = 'Qh'
        else:
            demand_type = 'Qc'

        # Track cumulative values
        cumulative_clusters = {0}  # Start with cluster 0 (existing DTN)
        cumulative_pipe_length = 0

        # Get buildings in cluster 0 (existing DTN)
        cluster0_buildings = self._get_buildings_in_specific_cluster(0)
        cumulative_buildings = set(cluster0_buildings)

        # Add a row for phase 0 (existing DTN)
        phase0_result = {
            'phase': 0,
            'year': 'Year 0',  # Year 0 for existing DTN
            'newly_connected_cluster(s)': '0',
            'cumulative_cluster(s)': '0',
            'number_of_newly_connected_buildings': len(cluster0_buildings),
            'cumulative_number_of_buildings_connected': len(cluster0_buildings),
            'capex_budget_per_phase [USD]': '-',  # No budget for existing DTN
            'new_cluster(s)_capex [USD]': 0,  # No CAPEX for existing DTN
            'cumulative_capex_budget [USD]': '-',  # No cumulative budget for existing DTN
            'cumulative_capex [USD]': 0,  # No cumulative CAPEX for existing DTN
            'total_expenditure_budget_per_phase [USD]': '-',  # No total expenditure budget for existing DTN
            'new_cluster(s)_total_expenditure [USD]': 0,  # Will be calculated if data is available
            'cumulative_total_expenditure [USD]': 0,  # Will be calculated if data is available
            'new_cluster(s)_revenue [USD]': 0,  # Will be calculated if data is available
            'cumulative_revenue [USD]': 0,  # Will be calculated if data is available
            'new_cluster(s)_om_cost [USD]': 0,  # No OM costs for existing DTN
            'cumulative_om_cost [USD]': 0,  # No cumulative OM costs for phase 0
            'ghg_cap [t CO2eq/yr]': '-',  # No GHG cap for existing DTN
            'district_operation_emission [t CO2eq/yr]': district_emissions.get(0, {}).get(
                'district_operation_emission [t CO2eq/yr]', 0),
            'district_operation_emission_per_total_gfa [kg CO2eq/yr/m2]': district_emissions.get(0, {}).get(
                'district_operation_emission_per_total_gfa [kg CO2eq/yr/m2]', 0),
            'new_cluster(s)_discounted_roi [-]': 0,  # Will be calculated if data is available
            'overall_discounted_roi [-]': 0,  # Will be calculated if data is available
            'new_cluster(s)_npv [USD]': 0,  # Will be calculated if data is available
            'overall_npv [USD]': 0,  # Will be calculated if data is available
            'new_cluster(s)_pipe_length [m]': 0,  # No new pipes for existing DTN
            'cumulative_pipe_length [m]': 0,  # Will be updated if data is available
            f'new_cluster(s)_annual_{demand_type} [MWh/yr]': 0,  # Will be updated if data is available
            f'cumulative_annual_{demand_type} [MWh/yr]': 0,  # Will be updated if data is available
            f'new_cluster(s)_linear_{demand_type}_density [MWh/km/yr]': 0,  # Will be calculated if data is available
            f'overall_linear_{demand_type}_density [MWh/km/yr]': 0  # Will be calculated if data is available
        }

        # Try to freeze Phase 0 to baseline results; else use header-only semantics
        try:
            baseline_file = Path(self.locator.get_dtn_expansion_optimization_results_folder()) / f"dtn_expansion_opt_results_{self.network_type}.csv"
            if baseline_file.exists():
                base_df = pd.read_csv(baseline_file)
                base0 = None
                if 'phase' in base_df.columns:
                    try:
                        base0 = base_df.loc[base_df['phase'].astype(str) == '0']
                    except Exception:
                        base0 = base_df.loc[base_df['phase'] == 0]
                    if base0 is not None and base0.empty:
                        base0 = None
                if base0 is None and 'year' in base_df.columns:
                    base0 = base_df[base_df['year'].astype(str).str.contains('Year 0', na=False)]
                if base0 is not None and not base0.empty:
                    base0d = base0.iloc[0].to_dict()
                    keys_to_copy = [
                        'new_cluster(s)_capex [USD]',
                        'cumulative_capex [USD]',
                        'new_cluster(s)_total_expenditure [USD]',
                        'cumulative_total_expenditure [USD]',
                        'new_cluster(s)_revenue [USD]',
                        'cumulative_revenue [USD]',
                        'new_cluster(s)_om_cost [USD]',
                        'cumulative_om_cost [USD]',
                        'new_cluster(s)_npv [USD]',
                        'overall_npv [USD]',
                        'new_cluster(s)_discounted_roi [-]',
                        'overall_discounted_roi [-]',
                        'new_cluster(s)_pipe_length [m]',
                        'cumulative_pipe_length [m]',
                        f'new_cluster(s)_annual_{demand_type} [MWh/yr]',
                        f'cumulative_annual_{demand_type} [MWh/yr]',
                        f'new_cluster(s)_linear_{demand_type}_density [MWh/km/yr]',
                        f'overall_linear_{demand_type}_density [MWh/km/yr]',
                    ]
                    for k in keys_to_copy:
                        if k in base0d:
                            phase0_result[k] = base0d[k]
                    # Set cumulative pipe length tracker
                    try:
                        cumulative_pipe_length = float(phase0_result.get('cumulative_pipe_length [m]', phase0_result.get('new_cluster(s)_pipe_length [m]', 0)))
                    except Exception:
                        cumulative_pipe_length = 0
                else:
                    log().warning("Baseline Phase 0 row not found; keeping header-only Phase 0 with zeros.")
            else:
                log().warning(f"Baseline results not found at {baseline_file}; keeping header-only Phase 0 with zeros.")
        except Exception as e:
            log().warning(f"Could not load/copy baseline Phase 0 row: {e}. Keeping header-only Phase 0.")

        # Emit diagnostics comparing baseline-frozen P0 row vs dynamic rerun P0 technicals (if available)
        try:
            diagnostics_rows = []
            dyn_p0_metrics = self.cluster_metrics.get('0', {})
            if dyn_p0_metrics:
                # Determine demand key
                demand_key_total = f'total_annual_{demand_type}_MWh'
                dyn_p0_pipe_len = dyn_p0_metrics.get('total_pipe_length_m', None)
                dyn_p0_ann_demand = dyn_p0_metrics.get(demand_key_total, None)
                base_p0_pipe_len = phase0_result.get('cumulative_pipe_length [m]', phase0_result.get('new_cluster(s)_pipe_length [m]', None))
                base_p0_ann_demand = phase0_result.get(f'cumulative_annual_{demand_type} [MWh/yr]', phase0_result.get(f'new_cluster(s)_annual_{demand_type} [MWh/yr]', None))
                diagnostics_rows.append({
                    'metric': 'phase0_pipe_length_m',
                    'baseline_value': base_p0_pipe_len,
                    'dynamic_value': dyn_p0_pipe_len,
                    'delta_dynamic_minus_baseline': (float(dyn_p0_pipe_len) - float(base_p0_pipe_len)) if (dyn_p0_pipe_len is not None and base_p0_pipe_len is not None) else None
                })
                diagnostics_rows.append({
                    'metric': f'phase0_cumulative_annual_{demand_type}_MWh',
                    'baseline_value': base_p0_ann_demand,
                    'dynamic_value': dyn_p0_ann_demand,
                    'delta_dynamic_minus_baseline': (float(dyn_p0_ann_demand) - float(base_p0_ann_demand)) if (dyn_p0_ann_demand is not None and base_p0_ann_demand is not None) else None
                })
            if diagnostics_rows:
                diag_df = pd.DataFrame(diagnostics_rows)
                diag_path = output_dir / 'phase0_freeze_diagnostics.csv'
                diag_df.to_csv(diag_path, index=False)
                log().info(f"Wrote Phase 0 freeze diagnostics to {diag_path}")
        except Exception as e:
            log().warning(f"Could not write Phase 0 diagnostics: {e}")

        results.append(phase0_result)

        # Initialize internal trackers for previous cumulative values to dynamic rerun Phase 0 (if available)
        prev_cum_pipe_len_internal = None
        prev_cum_ann_demand_internal = None
        try:
            dyn0 = self.cluster_metrics.get('0', {})
            if dyn0:
                prev_cum_pipe_len_internal = float(dyn0.get('total_pipe_length_m', 0) or 0)
                prev_cum_ann_demand_internal = float(dyn0.get(f'total_annual_{demand_type}_MWh', 0) or 0)
        except Exception:
            prev_cum_pipe_len_internal = None
            prev_cum_ann_demand_internal = None
        # Fallback to baseline-frozen outputs if dynamic values are not available
        if prev_cum_pipe_len_internal is None:
            try:
                prev_cum_pipe_len_internal = float(phase0_result.get('cumulative_pipe_length [m]', phase0_result.get('new_cluster(s)_pipe_length [m]', 0)) or 0)
            except Exception:
                prev_cum_pipe_len_internal = 0.0
        if prev_cum_ann_demand_internal is None:
            try:
                prev_cum_ann_demand_internal = float(phase0_result.get(f'cumulative_annual_{demand_type} [MWh/yr]', phase0_result.get(f'new_cluster(s)_annual_{demand_type} [MWh/yr]', 0)) or 0)
            except Exception:
                prev_cum_ann_demand_internal = 0.0

        log().info("Phase 0 KPIs frozen to baseline for reporting. Internal deltas initialized from dynamic Phase 0 to avoid negative/biased deltas.")

        # Calculate metrics for each phase using internal trackers for deltas
        for phase, clusters in sorted(solution['phases'].items()):
            # Get buildings in each cluster individually
            newly_connected_buildings = set()
            for cluster in clusters:
                cluster_buildings = self._get_buildings_in_specific_cluster(cluster)
                newly_connected_buildings.update(cluster_buildings)

            # Update cumulative values
            cumulative_clusters.update(clusters)
            cumulative_buildings.update(newly_connected_buildings)

            # Get metrics for this cluster set
            key = '+'.join(map(str, sorted(cumulative_clusters)))
            metrics = self.cluster_metrics.get(key, {})

            # Calculate financial metrics
            roi = solution[f'phase_{phase}_roi']
            npv = solution[f'phase_{phase}_npv']
            capex = solution[f'phase_{phase}_capex']

            # Calculate annual O&M costs (2.5% of CAPEX)
            annual_om_cost = 0.025 * capex

            # Calculate annual revenue
            annual_demand_kwh = metrics.get(f'total_annual_{demand_type}_MWh', 0) * 1000  # Convert MWh to kWh
            annual_revenue = annual_demand_kwh * self.energy_price

            # Get cumulative values directly from metrics (these already include cluster 0)
            cumulative_pipe_length = metrics.get('total_pipe_length_m', 0)
            cumulative_annual_demand = metrics.get(f'total_annual_{demand_type}_MWh', 0)
            linear_heat_density = metrics.get(f'linear_{demand_type}_density_MWh_per_km', 0)

            num_newly_connected_buildings = len(newly_connected_buildings)

            # Calculate new values as the difference between current and previous phase
            # Get the previous phase's cumulative values (phase 0 if this is phase 1)
            # Calculate new values as the difference using internal dynamic P0 trackers
            prev_pipe_length = float(prev_cum_pipe_len_internal or 0)
            prev_annual_demand = float(prev_cum_ann_demand_internal or 0)
            pipe_length = cumulative_pipe_length - prev_pipe_length
            new_annual_demand = cumulative_annual_demand - prev_annual_demand
            # Update internal trackers for next phase loop
            try:
                prev_cum_pipe_len_internal = float(cumulative_pipe_length or 0)
                prev_cum_ann_demand_internal = float(cumulative_annual_demand or 0)
            except Exception:
                pass

            # Calculate overall linear heat density
            overall_linear_density = cumulative_annual_demand / (
                        cumulative_pipe_length / 1000) if cumulative_pipe_length > 0 else 0

            # Add to totals
            total_capex += capex
            total_npv += npv

            # Calculate cumulative values for overall columns
            # Sum up values from all previous phases including current phase
            overall_capex = sum(r.get('new_cluster(s)_capex [USD]', 0) for r in results) + capex
            overall_npv = sum(r.get('new_cluster(s)_npv [USD]', 0) for r in results) + npv

            # Calculate overall ROI (weighted by CAPEX)
            if overall_capex > 0:
                # Include current phase in the calculation
                overall_roi = (sum(r['new_cluster(s)_discounted_roi [-]'] * r['new_cluster(s)_capex [USD]'] for r in
                                   results) + roi * capex) / \
                              (sum(r['new_cluster(s)_capex [USD]'] for r in results) + capex)
            else:
                overall_roi = 0

            # Calculate year range for this phase
            year_start = 1
            for p in range(1, phase):
                year_start += self.phase_durations[p - 1]
            year_end = year_start + self.phase_durations[phase - 1] - 1
            # Format year range to avoid Excel interpreting it as a date
            year_range = f"Year {year_start}-{year_end}"

            # Calculate cumulative budget and capex
            cumulative_capex_budget = sum(self.capex_budget_per_phase[:phase])
            cumulative_capex = sum(r.get('new_cluster(s)_capex [USD]', 0) for r in results) + capex

            # Calculate total expenditure for this phase
            total_expenditure = self._calculate_phase_total_expenditure(clusters, phase)

            # Calculate total revenue and OM costs for this phase
            phase_duration = self.phase_durations[phase - 1]
            total_revenue = 0
            total_om_cost = 0

            # Calculate present value of revenue and OM costs for all years in the phase
            for year in range(phase_duration):
                discount_factor = 1 / ((1 + self.interest_rate) ** (year + 1))
                total_revenue += annual_revenue * discount_factor
                total_om_cost += annual_om_cost * discount_factor

            # Calculate cumulative total expenditure
            cumulative_total_expenditure = sum(
                r.get('new_cluster(s)_total_expenditure [USD]', 0) for r in results) + total_expenditure

            # Calculate cumulative revenue and OM costs
            cumulative_revenue = sum(r.get('new_cluster(s)_revenue [USD]', 0) for r in results) + total_revenue
            cumulative_om_cost = sum(r.get('new_cluster(s)_om_cost [USD]', 0) for r in results) + total_om_cost

            # Create result dictionary with units
            # Note: All costs are in USD as per the internal calculations (e.g., Inv_USD2015perm, capex_hex_USD)
            result = {
                'phase': phase,
                'year': year_range,
                'newly_connected_cluster(s)': '+'.join(map(str, sorted(clusters))),
                'cumulative_cluster(s)': '+'.join(map(str, sorted(cumulative_clusters))),
                'number_of_newly_connected_buildings': num_newly_connected_buildings,
                'cumulative_number_of_buildings_connected': len(cumulative_buildings),
                'capex_budget_per_phase [USD]': self.capex_budget_per_phase[phase - 1] if phase - 1 < len(
                    self.capex_budget_per_phase) else 0,
                'new_cluster(s)_capex [USD]': capex,
                'cumulative_capex_budget [USD]': cumulative_capex_budget,
                'cumulative_capex [USD]': cumulative_capex,
                'total_expenditure_budget_per_phase [USD]': self.total_expenditure_budget_per_phase[
                    phase - 1] if phase - 1 < len(self.total_expenditure_budget_per_phase) else 0,
                'new_cluster(s)_total_expenditure [USD]': total_expenditure,
                'cumulative_total_expenditure [USD]': cumulative_total_expenditure,
                'new_cluster(s)_revenue [USD]': total_revenue,
                'cumulative_revenue [USD]': cumulative_revenue,
                'new_cluster(s)_om_cost [USD]': total_om_cost,
                'cumulative_om_cost [USD]': cumulative_om_cost,
                'ghg_cap [t CO2eq/yr]': self.ghg_budget_per_phase[
                    phase - 1] if self.ghg_budget_per_phase and phase - 1 < len(
                    self.ghg_budget_per_phase) else 'no_limit',
                'district_operation_emission [t CO2eq/yr]': district_emissions.get(phase, {}).get(
                    'district_operation_emission [t CO2eq/yr]', 0),
                'district_operation_emission_per_total_gfa [kg CO2eq/yr/m2]': district_emissions.get(phase, {}).get(
                    'district_operation_emission_per_total_gfa [kg CO2eq/yr/m2]', 0),
                'new_cluster(s)_discounted_roi [-]': roi,
                'overall_discounted_roi [-]': overall_roi,
                'new_cluster(s)_npv [USD]': npv,
                'overall_npv [USD]': overall_npv,
                'new_cluster(s)_pipe_length [m]': pipe_length,
                'cumulative_pipe_length [m]': cumulative_pipe_length,
                f'new_cluster(s)_annual_{demand_type} [MWh/yr]': new_annual_demand,
                f'cumulative_annual_{demand_type} [MWh/yr]': cumulative_annual_demand,
                f'new_cluster(s)_linear_{demand_type}_density [MWh/km/yr]': (
                            new_annual_demand / (pipe_length / 1000)) if pipe_length > 0 else 0,
                f'overall_linear_{demand_type}_density [MWh/km/yr]': overall_linear_density
            }

            results.append(result)

        # Overall ROI is now calculated for each phase, so we don't need this calculation anymore
        # The final overall ROI is in the last phase's result
        if results and any(r['phase'] != 0 for r in results):
            last_phase_result = [r for r in results if r['phase'] != 0][-1]
            final_overall_roi = last_phase_result['overall_discounted_roi [-]']
        else:
            final_overall_roi = 0

        # Create DataFrame
        results_df = pd.DataFrame(results)

        # Calculate total years
        total_years = sum(self.phase_durations)

        # Add overall summary row
        summary = {
            'phase': 'Total',
            'year': f"Year 1-{total_years}",
            'newly_connected_cluster(s)': '+'.join(map(str, sorted(
                [cluster for phase_result in results if phase_result['phase'] != 0 for cluster in
                 map(int, phase_result['newly_connected_cluster(s)'].split('+'))]))),
            'number_of_newly_connected_buildings': sum(
                result['number_of_newly_connected_buildings'] for result in results if result['phase'] != 0),
            'capex_budget_per_phase [USD]': sum(self.capex_budget_per_phase),
            'new_cluster(s)_capex [USD]': sum(
                result['new_cluster(s)_capex [USD]'] for result in results if result['phase'] != 0),
            'cumulative_capex_budget [USD]': sum(self.capex_budget_per_phase),
            'cumulative_capex [USD]': sum(
                result['new_cluster(s)_capex [USD]'] for result in results if result['phase'] != 0),
            'total_expenditure_budget_per_phase [USD]': sum(self.total_expenditure_budget_per_phase),
            'new_cluster(s)_total_expenditure [USD]': sum(
                result.get('new_cluster(s)_total_expenditure [USD]', 0) for result in results if result['phase'] != 0),
            'cumulative_total_expenditure [USD]': sum(
                result.get('new_cluster(s)_total_expenditure [USD]', 0) for result in results if result['phase'] != 0),
            'new_cluster(s)_revenue [USD]': sum(
                result.get('new_cluster(s)_revenue [USD]', 0) for result in results if result['phase'] != 0),
            'cumulative_revenue [USD]': sum(
                result.get('new_cluster(s)_revenue [USD]', 0) for result in results if result['phase'] != 0),
            'new_cluster(s)_om_cost [USD]': sum(
                result.get('new_cluster(s)_om_cost [USD]', 0) for result in results if result['phase'] != 0),
            'cumulative_om_cost [USD]': sum(
                result.get('new_cluster(s)_om_cost [USD]', 0) for result in results if result['phase'] != 0),
            'ghg_cap [t CO2eq/yr]': self.ghg_budget_per_phase[-1] if self.ghg_budget_per_phase else '-',
            'district_operation_emission [t CO2eq/yr]': district_emissions.get(self.num_phases, {}).get(
            'district_operation_emission [t CO2eq/yr]', 0),
        'district_operation_emission_per_total_gfa [kg CO2eq/yr/m2]': district_emissions.get(self.num_phases, {}).get(
            'district_operation_emission_per_total_gfa [kg CO2eq/yr/m2]', 0),
            'new_cluster(s)_discounted_roi [-]': sum(
                result['new_cluster(s)_discounted_roi [-]'] * result['new_cluster(s)_capex [USD]'] for result in results
                if result['phase'] != 0) / sum(
                result['new_cluster(s)_capex [USD]'] for result in results if result['phase'] != 0) if sum(
                result['new_cluster(s)_capex [USD]'] for result in results if result['phase'] != 0) > 0 else 0,
            'new_cluster(s)_npv [USD]': sum(
                result['new_cluster(s)_npv [USD]'] for result in results if result['phase'] != 0),
            'new_cluster(s)_pipe_length [m]': sum(
                result.get('new_cluster(s)_pipe_length [m]', result.get('newly_added_pipe_length [m]', 0)) for result in
                results if result['phase'] != 0),
            f'new_cluster(s)_annual_{demand_type} [MWh/yr]': sum(
                result.get(f'new_cluster(s)_annual_{demand_type} [MWh/yr]', 0) for result in results if
                result['phase'] != 0),
            f'cumulative_annual_{demand_type} [MWh/yr]': 0,  # Will be updated from the last phase
            f'new_cluster(s)_linear_{demand_type}_density [MWh/km/yr]': 0  # Will be calculated below
        }

        # For columns with "overall" or "cumulative" in their name, use the values from the last phase row
        if results and any(r['phase'] != 0 for r in results):
            last_phase_result = [r for r in results if r['phase'] != 0][-1]
            # Add overall and cumulative values from the last phase
            summary['cumulative_cluster(s)'] = last_phase_result['cumulative_cluster(s)']
            summary['cumulative_number_of_buildings_connected'] = last_phase_result[
                'cumulative_number_of_buildings_connected']
            summary['cumulative_capex [USD]'] = last_phase_result['cumulative_capex [USD]']
            summary['cumulative_total_expenditure [USD]'] = last_phase_result['cumulative_total_expenditure [USD]']
            summary['overall_discounted_roi [-]'] = final_overall_roi  # Already set to last phase value
            summary['overall_npv [USD]'] = last_phase_result['overall_npv [USD]']
            summary['cumulative_pipe_length [m]'] = last_phase_result['cumulative_pipe_length [m]']
            summary[f'cumulative_annual_{demand_type} [MWh/yr]'] = last_phase_result[
                f'cumulative_annual_{demand_type} [MWh/yr]']
            summary[f'overall_linear_{demand_type}_density [MWh/km/yr]'] = last_phase_result[
                f'overall_linear_{demand_type}_density [MWh/km/yr]']
        else:
            # If there are no non-zero phases, use Phase 0 values
            summary['cumulative_cluster(s)'] = '+'.join(map(str, sorted(cumulative_clusters)))
            summary['cumulative_number_of_buildings_connected'] = len(cumulative_buildings)
            summary['cumulative_capex [USD]'] = 0
            summary['cumulative_total_expenditure [USD]'] = 0
            summary['overall_discounted_roi [-]'] = 0
            summary['overall_npv [USD]'] = 0
            summary['cumulative_pipe_length [m]'] = cumulative_pipe_length
            summary[f'cumulative_annual_{demand_type} [MWh/yr]'] = phase0_result.get(
                f'cumulative_annual_{demand_type} [MWh/yr]', 0)
            summary[f'overall_linear_{demand_type}_density [MWh/km/yr]'] = phase0_result.get(
                f'overall_linear_{demand_type}_density [MWh/km/yr]', 0)
            summary['district_operation_emission [t CO2eq/yr]'] = district_emissions.get(0, {}).get('district_operation_emission [t CO2eq/yr]', 0)
            summary['district_operation_emission_per_total_gfa [kg CO2eq/yr/m2]'] = district_emissions.get(0, {}).get('district_operation_emission_per_total_gfa [kg CO2eq/yr/m2]', 0)

        # Calculate average newly connected linear heat density (weighted by pipe length) only if not already set
        if summary[f'new_cluster(s)_linear_{demand_type}_density [MWh/km/yr]'] == 0:
            total_pipe_length = sum(
                result.get('new_cluster(s)_pipe_length [m]', result.get('newly_added_pipe_length [m]', 0)) for result in
                results if result['phase'] != 0)
            if total_pipe_length > 0:
                summary[f'new_cluster(s)_linear_{demand_type}_density [MWh/km/yr]'] = sum(
                    result.get(f'new_cluster(s)_linear_{demand_type}_density [MWh/km/yr]', 0) *
                    result.get('new_cluster(s)_pipe_length [m]', result.get('newly_added_pipe_length [m]', 0))
                    for result in results if result['phase'] != 0
                ) / total_pipe_length

        # Append summary row
        results_df = pd.concat([results_df, pd.DataFrame([summary])], ignore_index=True)

        # Add optimization settings as metadata
        metadata = {
            'network_type': self.network_type,
            'num_phases': self.num_phases,
            'phase_durations [years]': ','.join(map(str, self.phase_durations)),
            'cost_model': self.cost_model,
            'objective_function': self.objective_function,
            'energy_price [USD/kWh]': self.energy_price,
            'interest_rate [-]': self.interest_rate,
            'capex_budget_per_phase [USD]': ','.join(map(str, self.capex_budget_per_phase)),
            'total_expenditure_budget_per_phase [USD]': ','.join(map(str, self.total_expenditure_budget_per_phase)),
            'diversity_factor [-]': self.diversity_factor,
            'temperature_difference_dh [K]': self.temperature_difference_dh,
            'temperature_difference_dc [K]': self.temperature_difference_dc,
            'pressure_loss_pa_per_m [Pa/m]': self.pressure_loss_pa_per_m,
            'pump_operation_hours [h]': self.pump_operation_hours,
            'pump_efficiency [-]': self.pump_efficiency,
            'pump_load_factor [-]': self.pump_load_factor,
            'pump_capex_a': self.pump_capex_a,
            'pump_capex_b': self.pump_capex_b,
            'cooling_cop [-]': self.cooling_cop
        }

        # Add GHG budget if specified
        if self.ghg_budget_per_phase:
            metadata['ghg_budget_per_phase [tonCO2]'] = ','.join(map(str, self.ghg_budget_per_phase))

        # Add testing_clusters if specified
        if self.testing_clusters:
            if isinstance(self.testing_clusters, list):
                metadata['testing_clusters'] = ','.join(map(str, self.testing_clusters))
            else:
                metadata['testing_clusters'] = str(self.testing_clusters)
        else:
            # If no testing clusters specified, include all clusters found in building clustering
            all_clusters = sorted([c for c in self.cluster_nodes['cluster'].unique() if c > 0])
            metadata['testing_clusters'] = ','.join(map(str, all_clusters))

        # Save metadata to a separate CSV
        metadata_df = pd.DataFrame([metadata])
        metadata_file = output_dir / "optimization_settings.csv"
        _assert_under_temp(metadata_file, temp_root, "metadata CSV")
        _log_io("WRITE optimization_settings CSV", metadata_file)
        metadata_df.to_csv(metadata_file, index=False)

        # Apply formatting to specific columns
        # Integer formatting
        integer_columns = [col for col in results_df.columns if any(substr in col for substr in [
            'capex [USD]', 'total_expenditure [USD]', 'revenue [USD]', 'om_cost [USD]', 'npv [USD]', 'pipe_length [m]'
        ])]
        for col in integer_columns:
            if col in results_df.columns:
                results_df[col] = results_df[col].apply(
                    lambda x: int(round(x, 0)) if isinstance(x, (int, float)) and not pd.isna(x) else x)

        # 2 decimal places
        decimal2_columns = [col for col in results_df.columns if any(substr in col for substr in [
            'operation_emission [t CO2eq/yr]', 'annual_Qh [MWh/yr]', 'annual_Qc [MWh/yr]',
            'linear_Qh_density [MWh/km/yr]', 'linear_Qc_density [MWh/km/yr]'
        ])]
        for col in decimal2_columns:
            if col in results_df.columns:
                results_df[col] = results_df[col].apply(
                    lambda x: round(x, 2) if isinstance(x, (int, float)) and not pd.isna(x) else x)

        # 4 decimal places
        decimal4_columns = [col for col in results_df.columns if 'discounted_roi [-]' in col]
        for col in decimal4_columns:
            if col in results_df.columns:
                results_df[col] = results_df[col].apply(
                    lambda x: round(x, 4) if isinstance(x, (int, float)) and not pd.isna(x) else x)

        # Save results to CSV with network-type specific filename
        results_file = output_dir / f"dtn_expansion_opt_results_{self.network_type}.csv"
        _assert_under_temp(results_file, temp_root, "results CSV")
        _log_io("WRITE results CSV", results_file)
        results_df.to_csv(results_file, index=False)

        # Save detailed results
        detailed_results_file = self.save_detailed_results(results, output_dir)
        _assert_under_temp(detailed_results_file, temp_root, "detailed results CSV")
        _log_io("WROTE detailed results CSV", detailed_results_file)

        # IO Ledger summary
        try:
            io_ledger = {
                "temp_root": str(temp_root),
                "phase_supply_dir": str(self.locator.get_dynamic_dtn_optimization_temp_scenario_phase_supply_files_folder()),
                "cluster_nodes": str(Path(self.locator.get_dynamic_dtn_optimization_temp_scenario_dtn_expansion_folder()) / "cluster_nodes.csv"),
                "cluster_edges": str(Path(self.locator.get_dynamic_dtn_optimization_temp_scenario_dtn_expansion_folder()) / "cluster_edges.csv"),
                "results_csv": str(results_file),
                "detailed_results_csv": str(detailed_results_file),
                "metadata_csv": str(metadata_file),
            }
            log().info("IO Ledger (Dynamic Part 2):\n" + "\n".join([f"  - {k}: {v}" for k, v in io_ledger.items()]))
        except Exception:
            pass

        log().info(f"Optimization results saved to {results_file}")
        log().info(f"Detailed optimization results saved to {detailed_results_file}")
        log().info(f"Optimization settings saved to {metadata_file}")

        # Q7: Cleanup empty legacy phase_* directories (keep phase_supply_files)
        try:
            dtn_dir = Path(self.locator.get_dynamic_dtn_optimization_temp_scenario_dtn_expansion_folder())
            for p in dtn_dir.glob("phase_*"):
                if p.name == "phase_supply_files":
                    continue
                if p.is_dir():
                    try:
                        next(p.iterdir())
                    except StopIteration:
                        p.rmdir()
                        log().info(f"Removed empty legacy directory: {p}")
        except Exception:
            pass

        return results_file


class PipeLayoutGenerator:
    """Generate pipe layouts for different sets of clusters and calculate metrics."""

    def __init__(self, locator: cea.inputlocator.InputLocator, network_type: str, phase: int = 1, testing_clusters=None,
                 chosen_buildings=None):
        """
        Initialize the PipeLayoutGenerator.

        Parameters:
        -----------
        locator : InputLocator
            CEA InputLocator object with direct path methods
        network_type : str
            'DH' for district heating or 'DC' for district cooling
        phase : int
            The phase number for the expansion
        testing_clusters : list, optional
            List of cluster IDs to include (if None, all clusters are included)
        chosen_buildings : list, optional
            List of building names to include (if None, all buildings are included)
        """
        self.locator = locator
        self.network_type = network_type
        self.phase = phase
        self.testing_clusters = testing_clusters
        self.chosen_buildings = chosen_buildings
        
        # Use direct path methods to get the output folder
        self.output_folder = Path(locator.get_dynamic_dtn_optimization_temp_scenario_dtn_expansion_folder()) / f"phase_{phase}"
        log().info(f"Output folder for phase {phase}: {self.output_folder}")
        self.output_folder.mkdir(parents=True, exist_ok=True)

        # Load input data
        self._load_inputs()

    def _load_inputs(self):
        """Load all necessary input data."""
        # Log the start of loading inputs
        log().info(f"Loading inputs for phase {self.phase}...")
        
        # Load cluster assignments (prefer baseline mapping for consistency)
        baseline_edges_path = Path(self.locator.get_dtn_expansion_optimization_results_folder()) / "cluster_edges.csv"
        baseline_nodes_path = Path(self.locator.get_dtn_cluster_nodes_file())
        if baseline_edges_path.exists():
            log().info(f"Loading cluster edges (baseline) from: {baseline_edges_path}")
            self.cluster_edges = pd.read_csv(baseline_edges_path)
        else:
            cluster_edges_path = Path(self.locator.get_dynamic_dtn_optimization_temp_scenario_dtn_expansion_folder()) / "cluster_edges.csv"
            log().info(f"Loading cluster edges (temp scenario) from: {cluster_edges_path}")
            self.cluster_edges = pd.read_csv(cluster_edges_path)

        if baseline_nodes_path.exists():
            log().info(f"Loading cluster nodes (baseline) from: {baseline_nodes_path}")
            self.cluster_nodes = pd.read_csv(baseline_nodes_path)
        else:
            cluster_nodes_path = Path(self.locator.get_dynamic_dtn_optimization_temp_scenario_dtn_expansion_folder()) / "cluster_nodes.csv"
            log().info(f"Loading cluster nodes (temp scenario) from: {cluster_nodes_path}")
            self.cluster_nodes = pd.read_csv(cluster_nodes_path)
        
        # Filter to CONSUMER nodes when available
        if 'type' in self.cluster_nodes.columns:
            self.cluster_nodes = self.cluster_nodes[self.cluster_nodes['type'] == 'CONSUMER']
        
        # Load edge-node matrix
        edge_node_path = Path(self.locator.get_thermal_network_edge_node_matrix_file(self.network_type))
        log().info(f"Loading edge-node matrix from: {edge_node_path}")
        self.edge_node_matrix = pd.read_csv(edge_node_path, index_col=0)
        
        # Load total demand from temp scenario
        total_demand_path = Path(self.locator.get_dynamic_dtn_optimization_temp_scenario_total_demand())
        log().info(f"Loading total demand from: {total_demand_path}")
        self.total_demand = pd.read_csv(total_demand_path)
        log().info(f"Loaded total demand with {len(self.total_demand)} buildings")

        # Get unique clusters (excluding 0 and -1)
        all_clusters = sorted([c for c in self.cluster_edges['cluster'].unique() if c > 0])

        # If testing_clusters is specified, use it directly
        if self.testing_clusters:
            # Convert to list if it's a string
            if isinstance(self.testing_clusters, str):
                self.testing_clusters = [int(c.strip()) for c in self.testing_clusters.split(',') if c.strip()]

            # No need to use chosen_buildings if testing_clusters is specified
            log().info(f"Using specified testing clusters: {self.testing_clusters}")
        # Otherwise, try to derive clusters from chosen buildings if specified
        elif self.chosen_buildings:
            # Convert to list if it's a string
            if isinstance(self.chosen_buildings, str):
                self.chosen_buildings = [b.strip() for b in self.chosen_buildings.split(',') if b.strip()]

            log().info(f"Filtering to include only {len(self.chosen_buildings)} chosen buildings")

            # Get clusters that contain the chosen buildings
            building_clusters = self.cluster_nodes[
                (self.cluster_nodes['type'] == 'CONSUMER') &
                (self.cluster_nodes['building'].isin(self.chosen_buildings))
                ]['cluster'].unique()

            # Use the clusters from chosen buildings
            self.testing_clusters = sorted([c for c in building_clusters if c > 0])
            log().info(f"Derived clusters from chosen buildings: {self.testing_clusters}")

        # Ensure all testing clusters exist
        if self.testing_clusters:
            valid_clusters = [c for c in self.testing_clusters if c in all_clusters]
            if len(valid_clusters) != len(self.testing_clusters):
                missing = set(self.testing_clusters) - set(valid_clusters)
                log().warning(f"Some testing clusters do not exist: {missing}")

            self.clusters = sorted(valid_clusters)
            log().info(f"Using {len(self.clusters)} testing clusters: {self.clusters}")
        else:
            self.clusters = all_clusters
            log().info(f"Using all {len(self.clusters)} clusters (excluding existing DTN and main roads)")

    def generate_all_cluster_combinations(self):
        """
        Generate all possible combinations of clusters.

        Returns:
        --------
        list
            List of all possible cluster combinations
        """
        all_combinations = []
        for r in range(1, len(self.clusters) + 1):
            combinations = list(itertools.combinations(self.clusters, r))
            all_combinations.extend(combinations)

        log().info(f"Generated {len(all_combinations)} possible cluster combinations")
        return all_combinations

    def get_required_pipes_for_clusters(self, cluster_set):
        """
        Determine the minimum required pipes for a set of clusters.

        Parameters:
        -----------
        cluster_set : tuple
            Tuple of cluster IDs to connect

        Returns:
        --------
        pd.DataFrame
            DataFrame of required pipes
        """
        # Include cluster 0 (existing DTN) in the set
        clusters_to_connect = set(cluster_set).union({0})

        # Ensure cluster, from_C, and to_C columns are numeric for comparison
        try:
            # Make a copy to avoid SettingWithCopyWarning
            cluster_edges_df = self.cluster_edges.copy()
            # Convert columns to numeric, errors='coerce' will convert non-numeric values to NaN
            for col in ['cluster', 'from_C', 'to_C']:
                if col in cluster_edges_df.columns:
                    cluster_edges_df[col] = pd.to_numeric(cluster_edges_df[col], errors='coerce')
                    # Fill NaN with a value that won't match our filters
                    cluster_edges_df[col] = cluster_edges_df[col].fillna(-999)
        except Exception as e:
            log().warning(f"Error converting columns to numeric: {e}. Using original dataframe.")
            cluster_edges_df = self.cluster_edges

        # Get edges that belong to the clusters in the set
        cluster_edges = cluster_edges_df[cluster_edges_df['cluster'].isin(clusters_to_connect)]

        # Get main road edges (-1) that connect the clusters
        main_road_edges = cluster_edges_df[
            (cluster_edges_df['cluster'] == -1) &
            (cluster_edges_df['from_C'].isin(clusters_to_connect)) &
            (cluster_edges_df['to_C'].isin(clusters_to_connect))
            ]

        # Combine the edges with error handling
        try:
            # First, try to concatenate with default settings
            required_pipes = pd.concat([cluster_edges, main_road_edges])
        except Exception as e:
            log().warning(f"Error during dataframe concatenation: {e}")

            # If that fails, try with more explicit settings
            try:
                # Reset index to avoid index-related issues
                cluster_edges_reset = cluster_edges.reset_index(drop=True)
                main_road_edges_reset = main_road_edges.reset_index(drop=True)

                # Try concatenation with ignore_index=True
                required_pipes = pd.concat([cluster_edges_reset, main_road_edges_reset], ignore_index=True)
            except Exception as e2:
                log().error(f"Failed to concatenate dataframes even with reset_index: {e2}")

                # As a last resort, if one of the dataframes is empty, return the other
                if len(cluster_edges) == 0:
                    required_pipes = main_road_edges
                elif len(main_road_edges) == 0:
                    required_pipes = cluster_edges
                else:
                    # If both have data but can't be concatenated, try to create a new dataframe with common columns
                    common_columns = set(cluster_edges.columns).intersection(set(main_road_edges.columns))
                    if common_columns:
                        log().warning(f"Using only common columns for concatenation: {common_columns}")
                        required_pipes = pd.concat([
                            cluster_edges[list(common_columns)],
                            main_road_edges[list(common_columns)]
                        ], ignore_index=True)
                    else:
                        # If no solution works, raise an error
                        raise ValueError("Cannot concatenate dataframes - no common columns found")

        return required_pipes

    def calculate_metrics(self, cluster_set, required_pipes):
        """
        Calculate metrics for a set of clusters.

        Parameters:
        -----------
        cluster_set : tuple
            Tuple of cluster IDs to connect
        required_pipes : pd.DataFrame
            DataFrame of required pipes

        Returns:
        --------
        dict
            Dictionary of metrics
        """
        # Include cluster 0 (existing DTN) in the set
        clusters_to_connect = set(cluster_set).union({0})

        # Calculate total pipe length
        total_pipe_length = required_pipes['length_m'].sum()

        # Get buildings in the clusters (using unique to avoid duplicates)
        buildings_in_clusters = self.cluster_nodes[
            (self.cluster_nodes['cluster'].isin(clusters_to_connect)) &
            (self.cluster_nodes['type'] == 'CONSUMER')
            ]['building'].unique().tolist()

        # Calculate total annual demand
        if self.network_type == 'DH':
            # For district heating, use Qhs_sys_MWhyr + Qww_sys_MWhyr
            total_annual_demand = self.total_demand[
                                      self.total_demand['name'].isin(buildings_in_clusters)
                                  ]['Qhs_sys_MWhyr'].sum() + self.total_demand[
                                      self.total_demand['name'].isin(buildings_in_clusters)
                                  ]['Qww_sys_MWhyr'].sum()
            demand_type = 'Qh'
        else:
            # For district cooling, use Qcs_sys_MWhyr + Qcre_sys_MWhyr + Qcdata_sys_MWhyr
            total_annual_demand = self.total_demand[
                                      self.total_demand['name'].isin(buildings_in_clusters)
                                  ]['Qcs_sys_MWhyr'].sum() + self.total_demand[
                                      self.total_demand['name'].isin(buildings_in_clusters)
                                  ]['Qcre_sys_MWhyr'].sum() + self.total_demand[
                                      self.total_demand['name'].isin(buildings_in_clusters)
                                  ]['Qcdata_sys_MWhyr'].sum()
            demand_type = 'Qc'

        # Calculate linear heat density (LHD)
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

    def generate_pipe_layouts(self):
        """
        Generate pipe layouts for all possible combinations of clusters.

        Returns:
        --------
        pd.DataFrame
            DataFrame with metrics for all cluster combinations
        """
        # Generate all possible combinations of clusters
        all_combinations = self.generate_all_cluster_combinations()

        # Calculate metrics for each combination
        all_metrics = []

        # Calculate metrics for cluster 0 alone first
        cluster0_pipes = self.get_required_pipes_for_clusters((0,))
        cluster0_metrics = self.calculate_metrics((0,), cluster0_pipes)
        all_metrics.append(cluster0_metrics)
        log().info(f"Calculated metrics for cluster 0 (existing DTN)")

        for i, cluster_set in enumerate(all_combinations):
            if i % 100 == 0 and i > 0:
                log().info(f"Processed {i}/{len(all_combinations)} cluster combinations")

            # Get required pipes for this set of clusters
            required_pipes = self.get_required_pipes_for_clusters(cluster_set)

            # Calculate metrics
            metrics = self.calculate_metrics(cluster_set, required_pipes)
            all_metrics.append(metrics)

        # Convert to DataFrame
        metrics_df = pd.DataFrame(all_metrics)

        # Save to CSV in the temp scenario's base dtn_expansion folder (not phase-specific)
        output_file = Path(self.locator.get_dynamic_dtn_optimization_temp_scenario_dtn_expansion_folder()) / "clusters_metrics.csv"
        metrics_df.to_csv(output_file, index=False)
        log().info(f"Saved metrics for {len(all_metrics)} cluster combinations to {output_file}")

        # Return the metrics DataFrame for use by the optimizer
        return metrics_df


def _detect_unmodified_clusters(locator, network_type, cluster_nodes_df, eps=1e-6):
    import pandas as _pd
    base_td = _pd.read_csv(locator.get_total_demand())
    temp_td = _pd.read_csv(locator.get_dynamic_dtn_optimization_temp_scenario_total_demand())
    nodes = cluster_nodes_df.copy()
    if 'type' in nodes.columns:
        nodes = nodes[nodes['type'] == 'CONSUMER']
    # choose energy columns by network type
    if str(network_type).upper() == 'DH':
        base_td['__Q__'] = base_td.get('Qhs_sys_MWhyr', 0).abs().fillna(0.0) + base_td.get('Qww_sys_MWhyr', 0).abs().fillna(0.0)
        temp_td['__Q__'] = temp_td.get('Qhs_sys_MWhyr', 0).abs().fillna(0.0) + temp_td.get('Qww_sys_MWhyr', 0).abs().fillna(0.0)
    else:
        base_td['__Q__'] = base_td.get('Qcs_sys_MWhyr', 0).abs().fillna(0.0) + base_td.get('Qcre_sys_MWhyr', 0).abs().fillna(0.0) + base_td.get('Qcdata_sys_MWhyr', 0).abs().fillna(0.0)
        temp_td['__Q__'] = temp_td.get('Qcs_sys_MWhyr', 0).abs().fillna(0.0) + temp_td.get('Qcre_sys_MWhyr', 0).abs().fillna(0.0) + temp_td.get('Qcdata_sys_MWhyr', 0).abs().fillna(0.0)
    # map building to cluster
    b2c = dict(zip(nodes['building'].astype(str), nodes['cluster'].astype(int)))
    base_sum = base_td.groupby(base_td['name'].astype(str).map(b2c)).agg({'__Q__': 'sum'})
    temp_sum = temp_td.groupby(temp_td['name'].astype(str).map(b2c)).agg({'__Q__': 'sum'})
    comp = base_sum.join(temp_sum, lsuffix='_base', rsuffix='_temp').fillna(0.0)
    unmodified = []
    for cid, row in comp.iterrows():
        if _pd.isna(cid):
            continue
        cid_int = int(cid)
        # always treat cluster 0 as unmodified reference
        if cid_int == 0:
            unmodified.append(0)
            continue
        b = float(row.get('__Q___base', 0.0))
        m = float(row.get('__Q___temp', 0.0))
        if b == 0.0 and m == 0.0:
            unmodified.append(cid_int)
        else:
            if abs(m - b) <= eps * max(1.0, abs(b)):
                unmodified.append(cid_int)
    return sorted(set(unmodified))

def _compare_unmodified_combinations(locator, network_type, cluster_nodes_df, updated_metrics_df, tol_rel=1e-6, tol_abs=1e-6, max_order=2):
    import pandas as _pd
    from itertools import combinations as _combinations
    baseline_metrics_file = Path(locator.get_dtn_expansion_optimization_results_folder()) / "clusters_metrics.csv"
    if not baseline_metrics_file.exists():
        log().info(f"No baseline clusters_metrics found at {baseline_metrics_file}; skipping sanity check.")
        return
    base_df = _pd.read_csv(baseline_metrics_file)
    unmod = _detect_unmodified_clusters(locator, network_type, cluster_nodes_df)
    # Build keys to check
    keys_to_check = set(['0'])
    for c in unmod:
        if c == 0:
            continue
        keys_to_check.add('+'.join(map(str, sorted([0, c]))))
    if max_order >= 2:
        for c1, c2 in _combinations([c for c in unmod if c != 0], 2):
            keys_to_check.add('+'.join(map(str, sorted([0, c1, c2]))))
    demand_col = 'total_annual_Qh_MWh' if str(network_type).upper() == 'DH' else 'total_annual_Qc_MWh'
    for key in sorted(keys_to_check):
        rb = base_df[base_df['clusters'] == key]
        ru = updated_metrics_df[updated_metrics_df['clusters'] == key]
        if rb.empty or ru.empty:
            log().error(f"[Sanity] Missing combination '{key}' in {'baseline' if rb.empty else 'updated'} metrics.")
            raise RuntimeError(f"Missing combination '{key}' in {'baseline' if rb.empty else 'updated'} metrics.")
        bL = float(rb['total_pipe_length_m'].iloc[0])
        uL = float(ru['total_pipe_length_m'].iloc[0])
        if abs(bL - uL) > tol_abs:
            log().error(f"[Sanity] Pipe length mismatch for '{key}': baseline={bL}, updated={uL}")
            raise RuntimeError(f"Pipe length mismatch for '{key}'. Please re-check Part 1 outputs or mapping.")
        bQ = float(rb[demand_col].iloc[0])
        uQ = float(ru[demand_col].iloc[0])
        if not (abs(uQ - bQ) <= max(tol_abs, tol_rel * max(1.0, abs(bQ)))):
            log().error(f"[Sanity] Demand mismatch for '{key}': baseline={bQ}, updated={uQ}")
            raise RuntimeError(f"Demand mismatch for '{key}'. Unmodified combos should match baseline. Check Part 1 export.")
    log().info(f"Sanity check passed for {len(keys_to_check)} unmodified combinations (incl. 0).")

def generate_updated_metrics(locator, network_type, testing_clusters=None):
    """
    Generate updated metrics based on demand files in the temp scenario.

    Parameters:
    -----------
    locator : InputLocator
        Original scenario locator with direct path methods
    network_type : str
        'DH' for district heating or 'DC' for district cooling
    testing_clusters : list, optional
        List of cluster IDs to include

    Returns:
    --------
    pd.DataFrame
        DataFrame containing updated metrics for all cluster combinations
    """
    log().info("Generating updated metrics using temp scenario demand files...")

    # Use the direct path methods to get the paths to the total demand files
    original_total_demand = Path(locator.get_total_demand())
    temp_total_demand = Path(locator.get_dynamic_dtn_optimization_temp_scenario_total_demand())

    log().info(f"Original total demand path: {original_total_demand}")
    log().info(f"Temp scenario total demand path: {temp_total_demand}")

    if str(original_total_demand) == str(temp_total_demand):
        log().warning("WARNING: The paths to the original and temp scenario total demand files are the same!")
        log().warning("This suggests that the temp scenario may not have been properly created.")
    else:
        log().info("✓ Successfully identified distinct total demand files for original and temp scenarios")

    # Create directory for updated metrics
    output_dir = Path(locator.get_dynamic_dtn_optimization_updated_metrics_folder())
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create a custom PipeLayoutGenerator that uses the temp scenario's total demand file
    class CustomPipeLayoutGenerator(PipeLayoutGenerator):
        def _load_inputs(self):
            """Override to ensure we use the temp scenario's total demand file, but baseline cluster mapping."""
            # Prefer baseline cluster assignments for consistency
            baseline_edges_path = Path(self.locator.get_dtn_expansion_optimization_results_folder()) / "cluster_edges.csv"
            baseline_nodes_path = Path(self.locator.get_dtn_cluster_nodes_file())
            if baseline_edges_path.exists():
                log().info(f"Loading cluster edges (baseline) from: {baseline_edges_path}")
                self.cluster_edges = pd.read_csv(baseline_edges_path)
            else:
                cluster_edges_path = Path(self.locator.get_dynamic_dtn_optimization_temp_scenario_dtn_expansion_folder()) / "cluster_edges.csv"
                log().info(f"Loading cluster edges (temp scenario) from: {cluster_edges_path}")
                self.cluster_edges = pd.read_csv(cluster_edges_path)

            if baseline_nodes_path.exists():
                log().info(f"Loading cluster nodes (baseline) from: {baseline_nodes_path}")
                self.cluster_nodes = pd.read_csv(baseline_nodes_path)
            else:
                cluster_nodes_path = Path(self.locator.get_dynamic_dtn_optimization_temp_scenario_dtn_expansion_folder()) / "cluster_nodes.csv"
                log().info(f"Loading cluster nodes (temp scenario) from: {cluster_nodes_path}")
                self.cluster_nodes = pd.read_csv(cluster_nodes_path)

            # Filter to CONSUMER nodes when available
            if 'type' in self.cluster_nodes.columns:
                self.cluster_nodes = self.cluster_nodes[self.cluster_nodes['type'] == 'CONSUMER']

            # Load edge-node matrix
            edge_node_path = Path(self.locator.get_thermal_network_edge_node_matrix_file(self.network_type))
            log().info(f"Loading edge-node matrix from: {edge_node_path}")
            self.edge_node_matrix = pd.read_csv(edge_node_path, index_col=0)

            # Use the temp scenario's total demand file
            total_demand_path = temp_total_demand
            log().info(f"Loading total demand from temp scenario: {total_demand_path}")

            # Verify the file exists
            if not total_demand_path.exists():
                log().error(f"Total demand file not found in temp scenario: {total_demand_path}")
                raise FileNotFoundError(f"Total demand file not found in temp scenario: {total_demand_path}")

            self.total_demand = pd.read_csv(total_demand_path)

            # Get unique clusters (excluding 0 and -1)
            all_clusters = sorted([c for c in self.cluster_edges['cluster'].unique() if c > 0])

            # If testing_clusters is specified, use it directly
            if self.testing_clusters:
                # Convert to list if it's a string
                if isinstance(self.testing_clusters, str):
                    self.testing_clusters = [int(c.strip()) for c in self.testing_clusters.split(',') if c.strip()]

                # No need to use chosen_buildings if testing_clusters is specified
                log().info(f"Using specified testing clusters: {self.testing_clusters}")
            # Otherwise, try to derive clusters from chosen buildings if specified
            elif self.chosen_buildings:
                # Convert to list if it's a string
                if isinstance(self.chosen_buildings, str):
                    self.chosen_buildings = [b.strip() for b in self.chosen_buildings.split(',') if b.strip()]

                log().info(f"Filtering to include only {len(self.chosen_buildings)} chosen buildings")

                # Get clusters that contain the chosen buildings
                building_clusters = self.cluster_nodes[
                    (self.cluster_nodes['type'] == 'CONSUMER') &
                    (self.cluster_nodes['building'].isin(self.chosen_buildings))
                    ]['cluster'].unique()

                # Use the clusters from chosen buildings
                self.testing_clusters = sorted([c for c in building_clusters if c > 0])
                log().info(f"Derived clusters from chosen buildings: {self.testing_clusters}")

            # Ensure all testing clusters exist
            if self.testing_clusters:
                valid_clusters = [c for c in self.testing_clusters if c in all_clusters]
                if len(valid_clusters) != len(self.testing_clusters):
                    missing = set(self.testing_clusters) - set(valid_clusters)
                    log().warning(f"Some testing clusters do not exist: {missing}")

                self.clusters = sorted(valid_clusters)
                log().info(f"Using {len(self.clusters)} testing clusters: {self.clusters}")
            else:
                self.clusters = all_clusters
                log().info(f"Using all {len(self.clusters)} clusters (excluding existing DTN and main roads)")

    log().info(
        f"Creating CustomPipeLayoutGenerator with network_type={network_type}, testing_clusters={testing_clusters}")
    generator = CustomPipeLayoutGenerator(
        locator=locator,  # Use the original locator with our custom override
        network_type=network_type,
        phase=1,
        testing_clusters=testing_clusters
    )

    # Generate the updated metrics
    log().info("Generating pipe layouts with temp scenario demand files...")
    metrics_df = generator.generate_pipe_layouts()

    # Q6: Sanity check for unmodified combinations (0, 0+c, 0+c1+c2) vs baseline; stop on mismatch
    try:
        log().info("Running sanity check for unmodified cluster combinations against baseline metrics...")
        _compare_unmodified_combinations(locator, network_type, generator.cluster_nodes, metrics_df)
    except Exception as e:
        log().error(f"Sanity check failed: {e}")
        raise

    # Save the updated metrics (overwrite)
    output_file = output_dir / "clusters_metrics_updated.csv"
    metrics_df.to_csv(output_file, index=False)
    log().info(f"Saved updated metrics to {output_file}")

    return metrics_df


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

    # Configure logging levels: keep HTTP/urllib3 noise suppressed; show our IO at INFO; enable module DEBUG when requested
    try:
        debug_flag = bool(getattr(config.general, 'debug', False))
    except Exception:
        debug_flag = False
    import logging as _logging_cfg
    # Always keep root at INFO to avoid 3rd-party DEBUG noise (e.g., urllib3)
    try:
        _logging_cfg.getLogger().setLevel(_logging_cfg.INFO)
    except Exception:
        pass
    # Set our module and the 'cea' namespace to DEBUG when requested, else INFO
    desired_level = _logging_cfg.DEBUG if debug_flag else _logging_cfg.INFO
    try:
        log().setLevel(desired_level)
        _logging_cfg.getLogger('cea').setLevel(desired_level)
    except Exception:
        pass
    # Silence noisy third-party debug logs explicitly
    for noisy in (
        'urllib3',
        'urllib3.connectionpool',
        'requests',
        'requests.packages.urllib3',
        'matplotlib',
    ):
        try:
            _logging_cfg.getLogger(noisy).setLevel(_logging_cfg.WARNING)
        except Exception:
            pass
    # Ensure at least one console handler exists (fail-safe)
    try:
        root_logger = _logging_cfg.getLogger()
        if not root_logger.handlers:
            _logging_cfg.basicConfig(level=_logging_cfg.INFO,
                                     format="%(asctime)s | %(levelname)5s | %(message)s",
                                     datefmt="%H:%M:%S")
    except Exception:
        pass
    log().info(f"Debug mode is {'ON' if debug_flag else 'OFF'}; module log level set to {'DEBUG' if debug_flag else 'INFO'} (HTTP client debug logs suppressed).")

    # Get the scenario and locator
    scenario = config.scenario
    locator = cea.inputlocator.InputLocator(scenario=scenario)

    # Get the network type
    network_type = config.dynamic_dtn_optimization.network_type

    # Check if the thermal network prerequisites are met
    if not check_thermal_network_prerequisites(locator, network_type, bypass_check=True):
        log().error("Thermal network prerequisites not met. Please run the thermal network script first.")
        return

    # Check if the temporary scenario exists
    temp_scenario_path = locator.get_dynamic_dtn_optimization_temp_scenario_folder()
    if not os.path.exists(temp_scenario_path):
        log().error(f"Temporary scenario not found at: {temp_scenario_path}")
        log().error("Please run dynamic_dtn_optimization.py first to create the temporary scenario.")
        return

    log().info(f"Found temporary scenario at: {temp_scenario_path}")

    # Q4: Mirror baseline cluster mapping files into temp_scenario for diagnostics (authoritative mapping is baseline)
    try:
        from shutil import copyfile as _copyfile
        temp_dtn_dir = Path(locator.get_dynamic_dtn_optimization_temp_scenario_dtn_expansion_folder())
        temp_dtn_dir.mkdir(parents=True, exist_ok=True)
        baseline_nodes = Path(locator.get_dtn_cluster_nodes_file())
        baseline_edges = Path(locator.get_dtn_expansion_optimization_results_folder()) / "cluster_edges.csv"
        if baseline_nodes.exists():
            _copyfile(baseline_nodes, temp_dtn_dir / "cluster_nodes.csv")
            log().info(f"Copied baseline cluster_nodes.csv to {temp_dtn_dir}")
        if baseline_edges.exists():
            _copyfile(baseline_edges, temp_dtn_dir / "cluster_edges.csv")
            log().info(f"Copied baseline cluster_edges.csv to {temp_dtn_dir}")
    except Exception as e:
        log().warning(f"Could not mirror baseline cluster mapping files to temp_scenario: {e}")

    # Parse testing clusters from config
    testing_clusters_str = config.dtn_expansion_optimization.testing_clusters
    if testing_clusters_str:
        testing_clusters = [int(c.strip()) for c in testing_clusters_str.split(',') if c.strip()]
    else:
        testing_clusters = None

    # Always regenerate updated metrics (overwrite mode) from temp scenario demand files (Q3)
    log().info("Regenerating updated metrics from temp scenario demand (overwrite mode)")
    metrics_df = generate_updated_metrics(
        locator=locator,
        network_type=network_type,
        testing_clusters=testing_clusters
    )

    # Load saved DTN optimization settings (if available) and resolve parameters with precedence
    saved = {}
    saved_path = Path(locator.get_dynamic_dtn_optimization_temp_scenario_dtn_expansion_folder()) / "run_settings.json"
    if saved_path.exists():
        log().info(f"Loading saved DTN run settings from {saved_path}")
        try:
            with open(saved_path, 'r') as f:
                saved = json.load(f)
        except Exception as e:
            log().warning(f"Failed to load saved settings JSON: {e}")
    else:
        log().info("No run_settings.json found in temp scenario. Using config/defaults only unless CLI flags provided.")

    def pick(cfg_value, saved_value, default_value):
        return cfg_value if cfg_value not in (None, "", []) else (saved_value if saved_value not in (None, "", []) else default_value)

    # Core settings
    num_phases = pick(getattr(config.dtn_expansion_optimization, 'num_phases', None), saved.get('num_phases'), 3)

    # Phase durations
    phase_durations = None
    phase_durations_str = getattr(config.dtn_expansion_optimization, 'phase_durations', None)
    if phase_durations_str:
        phase_durations = [int(d.strip()) for d in str(phase_durations_str).split(',') if str(d).strip()]
    elif saved.get('phase_durations'):
        phase_durations = [int(d) for d in saved.get('phase_durations')]
    else:
        phase_durations = [10] * int(num_phases)

    # Budgets
    capex_budget_str = getattr(config.dtn_expansion_optimization, 'capex_budget_per_phase', None)
    capex_budget_per_phase = [float(b.strip()) for b in str(capex_budget_str).split(',') if str(b).strip()] if capex_budget_str else saved.get('capex_budget_per_phase')

    total_expenditure_budget_str = getattr(config.dtn_expansion_optimization, 'total_expenditure_budget_per_phase', None)
    total_expenditure_budget_per_phase = [float(b.strip()) for b in str(total_expenditure_budget_str).split(',') if str(b).strip()] if total_expenditure_budget_str else saved.get('total_expenditure_budget_per_phase')

    ghg_budget_str = getattr(config.dtn_expansion_optimization, 'ghg_budget_per_phase', None)
    ghg_budget_per_phase = [float(b.strip()) for b in str(ghg_budget_str).split(',') if str(b).strip()] if ghg_budget_str else saved.get('ghg_budget_per_phase')

    # Q2/Q3 additional parameters (relaxing pct with legacy fallback)
    capex_per_phase_relaxing_pct_cfg = getattr(config.dtn_expansion_optimization, 'capex_per_phase_relaxing_pct', None)
    if capex_per_phase_relaxing_pct_cfg in (None, ''):
        capex_per_phase_relaxing_pct_cfg = getattr(config.dtn_expansion_optimization, 'capex_per_phase_softening_pct', None)
    capex_per_phase_relaxing_pct_saved = saved.get('capex_per_phase_relaxing_pct', saved.get('capex_per_phase_softening_pct'))
    capex_per_phase_softening_pct = pick(capex_per_phase_relaxing_pct_cfg, capex_per_phase_relaxing_pct_saved, 0.20)

    total_exp_per_phase_relaxing_pct_cfg = getattr(config.dtn_expansion_optimization, 'total_exp_per_phase_relaxing_pct', None)
    if total_exp_per_phase_relaxing_pct_cfg in (None, ''):
        total_exp_per_phase_relaxing_pct_cfg = getattr(config.dtn_expansion_optimization, 'total_exp_per_phase_softening_pct', None)
    total_exp_per_phase_relaxing_pct_saved = saved.get('total_exp_per_phase_relaxing_pct', saved.get('total_exp_per_phase_softening_pct'))
    total_exp_per_phase_softening_pct = pick(total_exp_per_phase_relaxing_pct_cfg, total_exp_per_phase_relaxing_pct_saved, 0.20)

    enforce_final_cumulative_capex = bool(pick(getattr(config.dtn_expansion_optimization, 'enforce_final_cumulative_capex', None), saved.get('enforce_final_cumulative_capex'), True))
    enforce_final_cumulative_total_exp = bool(pick(getattr(config.dtn_expansion_optimization, 'enforce_final_cumulative_total_exp', None), saved.get('enforce_final_cumulative_total_exp'), True))

    final_cumulative_capex_budget_total = pick(getattr(config.dtn_expansion_optimization, 'final_cumulative_capex_budget_total', None), saved.get('final_cumulative_capex_budget_total'), 0)
    final_cumulative_total_exp_budget_total = pick(getattr(config.dtn_expansion_optimization, 'final_cumulative_total_exp_budget_total', None), saved.get('final_cumulative_total_exp_budget_total'), 0)
    try:
        final_cumulative_capex_budget_total = float(final_cumulative_capex_budget_total) if str(final_cumulative_capex_budget_total) not in ("", "None") else 0
    except Exception:
        final_cumulative_capex_budget_total = 0
    try:
        final_cumulative_total_exp_budget_total = float(final_cumulative_total_exp_budget_total) if str(final_cumulative_total_exp_budget_total) not in ("", "None") else 0
    except Exception:
        final_cumulative_total_exp_budget_total = 0

    lock_committed_early_phases = bool(pick(getattr(config.dtn_expansion_optimization, 'lock_committed_early_phases', None), saved.get('lock_committed_early_phases'), True))
    lock_reference_genome = str(pick(getattr(config.dtn_expansion_optimization, 'lock_reference_genome', None), saved.get('lock_reference_genome'), 'baseline-dtn-opt'))

    locked_phase_indices_raw = pick(getattr(config.dtn_expansion_optimization, 'locked_phase_indices', None), saved.get('locked_phase_indices'), '1')
    if isinstance(locked_phase_indices_raw, str):
        locked_phase_indices = [int(x.strip()) for x in locked_phase_indices_raw.split(',') if x.strip() != '']
    elif isinstance(locked_phase_indices_raw, list):
        locked_phase_indices = [int(x) for x in locked_phase_indices_raw]
    else:
        locked_phase_indices = [1]

    custom_locked_genome_raw = pick(getattr(config.dtn_expansion_optimization, 'custom_locked_genome', None), saved.get('custom_locked_genome'), '')
    custom_locked_genome = []
    if isinstance(custom_locked_genome_raw, str) and custom_locked_genome_raw:
        try:
            s = str(custom_locked_genome_raw).strip()
            if s.startswith('[') and s.endswith(']'):
                s = s[1:-1]
            custom_locked_genome = [int(x.strip()) for x in s.split(',') if x.strip() != '']
        except Exception:
            custom_locked_genome = []
    elif isinstance(custom_locked_genome_raw, list):
        try:
            custom_locked_genome = [int(x) for x in custom_locked_genome_raw]
        except Exception:
            custom_locked_genome = []

    # Per-phase ceiling enforcement is always ON (hidden from GUI)
    enforce_per_phase_soft_ceiling = True
    try:
        log().info("Per-phase relaxed ceiling enforcement is forced to True (hidden from GUI)")
    except Exception:
        pass

    # GA hyperparameters
    ga_crossover_probability = pick(getattr(config.dtn_expansion_optimization, 'ga_crossover_probability', None), saved.get('ga_crossover_probability'), 0.5)
    ga_mutation_probability = pick(getattr(config.dtn_expansion_optimization, 'ga_mutation_probability', None), saved.get('ga_mutation_probability'), 0.2)
    ga_tournament_size = pick(getattr(config.dtn_expansion_optimization, 'ga_tournament_size', None), saved.get('ga_tournament_size'), 3)
    ga_mutation_indpb = pick(getattr(config.dtn_expansion_optimization, 'ga_mutation_indpb', None), saved.get('ga_mutation_indpb'), 0.4)

    # random seed: 0 or blank => random
    ga_random_seed = pick(getattr(config.dtn_expansion_optimization, 'ga_random_seed', None), saved.get('ga_random_seed'), 0)

    log().info("Creating optimizer with original locator (using direct path methods)")

    # Resolve objective and technical parameters with precedence (config > saved > defaults)
    objective_function = pick(getattr(config.dtn_expansion_optimization, 'objective_function', None), saved.get('objective_function'), 'NPV')

    # Enforce single-objective-only mode
    requested_moo = bool(getattr(config.dtn_expansion_optimization, 'multi_objective_mode', False)) or bool(saved.get('multi_objective_mode', False))
    if requested_moo:
        log().warning("Dynamic Part 2 only supports single-objective reruns. Overriding multi_objective_mode=False and ignoring multi_objective_functions.")
    multi_objective_mode = False
    multi_objective_functions = None

    # Validate objective for SOO
    allowed_soo = {"npv", "roi", "emissions", "total_capex"}
    if str(objective_function).lower() not in allowed_soo:
        log().warning(f"Unsupported single-objective '{objective_function}'. Falling back to 'NPV'.")
        objective_function = 'NPV'

    # Technical / economic parameters
    interest_rate = pick(getattr(config.dtn_expansion_optimization, 'interest_rate', None), saved.get('interest_rate'), 0.05)
    cost_model = pick(getattr(config.dtn_expansion_optimization, 'cost_model', None), saved.get('cost_model'), 'detailed')
    diversity_factor = pick(getattr(config.dtn_expansion_optimization, 'diversity_factor', None), saved.get('diversity_factor'), 0.7)
    temperature_difference_dh = pick(getattr(config.dtn_expansion_optimization, 'temperature_difference_dh', None), saved.get('temperature_difference_dh'), 20)
    temperature_difference_dc = pick(getattr(config.dtn_expansion_optimization, 'temperature_difference_dc', None), saved.get('temperature_difference_dc'), 10)
    pressure_loss_pa_per_m = pick(getattr(config.dtn_expansion_optimization, 'pressure_loss_pa_per_m', None), saved.get('pressure_loss_pa_per_m'), 200)
    pump_operation_hours = pick(getattr(config.dtn_expansion_optimization, 'pump_operation_hours', None), saved.get('pump_operation_hours'), 4000)
    pump_efficiency = pick(getattr(config.dtn_expansion_optimization, 'pump_efficiency', None), saved.get('pump_efficiency'), 0.8)
    pump_load_factor = pick(getattr(config.dtn_expansion_optimization, 'pump_load_factor', None), saved.get('pump_load_factor'), 0.5)
    pump_capex_a = pick(getattr(config.dtn_expansion_optimization, 'pump_capex_a', None), saved.get('pump_capex_a'), 1230)
    pump_capex_b = pick(getattr(config.dtn_expansion_optimization, 'pump_capex_b', None), saved.get('pump_capex_b'), 0.65)
    cooling_cop = pick(getattr(config.dtn_expansion_optimization, 'cooling_cop', None), saved.get('cooling_cop'), 4.0)

    # Testing clusters fallback to saved if not provided
    if not testing_clusters and saved.get('testing_clusters'):
        testing_clusters = saved.get('testing_clusters')

    # Create the optimizer with the resolved parameters
    optimizer = DTNExpansionOptimizer(
        locator=locator,  # Use the original locator with direct path methods
        network_type=network_type,
        metrics_df=metrics_df,
        num_phases=int(num_phases),
        phase_durations=phase_durations,
        capex_budget_per_phase=capex_budget_per_phase,
        total_expenditure_budget_per_phase=total_expenditure_budget_per_phase,
        interest_rate=float(interest_rate),
        cost_model=str(cost_model),
        objective_function=str(objective_function),
        diversity_factor=float(diversity_factor),
        temperature_difference_dh=float(temperature_difference_dh),
        temperature_difference_dc=float(temperature_difference_dc),
        pressure_loss_pa_per_m=float(pressure_loss_pa_per_m),
        pump_operation_hours=int(pump_operation_hours),
        pump_efficiency=float(pump_efficiency),
        pump_load_factor=float(pump_load_factor),
        pump_capex_a=float(pump_capex_a),
        pump_capex_b=float(pump_capex_b),
        cooling_cop=float(cooling_cop),
        ghg_budget_per_phase=ghg_budget_per_phase,
        multi_objective_mode=multi_objective_mode,
        multi_objective_functions=multi_objective_functions,
        testing_clusters=testing_clusters,
        # New Q2/Q3
        capex_per_phase_softening_pct=float(capex_per_phase_softening_pct),
        total_exp_per_phase_softening_pct=float(total_exp_per_phase_softening_pct),
        enforce_final_cumulative_capex=bool(enforce_final_cumulative_capex),
        final_cumulative_capex_budget_total=float(final_cumulative_capex_budget_total),
        enforce_final_cumulative_total_exp=bool(enforce_final_cumulative_total_exp),
        final_cumulative_total_exp_budget_total=float(final_cumulative_total_exp_budget_total),
        lock_committed_early_phases=bool(lock_committed_early_phases),
        locked_phase_indices=locked_phase_indices,
        lock_reference_genome=str(lock_reference_genome),
        custom_locked_genome=custom_locked_genome,
        enforce_per_phase_soft_ceiling=bool(enforce_per_phase_soft_ceiling)
    )

    # Emissions are LCA-only; compute-emissions-from-cop is deprecated and ignored
    try:
        log().info("compute-emissions-from-cop parameter is deprecated and ignored. Using LCA Operation per phase.")
    except Exception:
        pass

    # Run the optimization (GA runtime controls with precedence)
    population_size = pick(getattr(config.dtn_expansion_optimization, 'population_size', None), saved.get('population_size'), 50)
    num_generations = pick(getattr(config.dtn_expansion_optimization, 'num_generations', None), saved.get('num_generations'), 30)

    # Additional run management options (may not exist in config; use defaults)
    try:
        include_in_part3 = bool(getattr(config.dynamic_dtn_optimization, 'include_in_part3'))
    except Exception:
        include_in_part3 = True
    try:
        run_label = getattr(config.dynamic_dtn_optimization, 'run_label')
    except Exception:
        run_label = None

    # Persist resolved run settings (including new constraint parameters) to temp scenario for Part 3 and reproducibility
    try:
        settings_out = {
            "network_type": network_type,
            "objective_function": str(objective_function),
            "num_phases": int(num_phases),
            "phase_durations": phase_durations,
            "capex_budget_per_phase": capex_budget_per_phase,
            "total_expenditure_budget_per_phase": total_expenditure_budget_per_phase,
            "ghg_budget_per_phase": ghg_budget_per_phase,
            # New constraint parameters (use snake_case keys Part 3 understands)
            "capex_per_phase_relaxing_pct": float(capex_per_phase_softening_pct),
            "total_exp_per_phase_relaxing_pct": float(total_exp_per_phase_softening_pct),
            "enforce_final_cumulative_capex": bool(enforce_final_cumulative_capex),
            "final_cumulative_capex_budget_total": float(final_cumulative_capex_budget_total),
            "enforce_final_cumulative_total_exp": bool(enforce_final_cumulative_total_exp),
            "final_cumulative_total_exp_budget_total": float(final_cumulative_total_exp_budget_total),
            "lock_committed_early_phases": bool(lock_committed_early_phases),
            "locked_phase_indices": locked_phase_indices,
            "lock_reference_genome": str(lock_reference_genome),
            "custom_locked_genome": custom_locked_genome,
            "enforce_per_phase_soft_ceiling": bool(enforce_per_phase_soft_ceiling),
            # GA runtime
            "population_size": int(population_size),
            "num_generations": int(num_generations),
            # Misc
            "testing_clusters": testing_clusters,
            # Run management flags
            "include_in_part3": bool(include_in_part3),
            "run_label": run_label,
        }
        # Remove None values to keep file clean
        settings_out = {k: v for k, v in settings_out.items() if v is not None}
        # Ensure parent folder exists and write
        try:
            Path(saved_path).parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        with open(saved_path, 'w') as f:
            json.dump(settings_out, f, indent=2)
        log().info(f"Saved resolved run settings to {saved_path}")
    except Exception as e:
        log().warning(f"Could not write resolved run settings JSON: {e}")

    solution = optimizer.optimize(population_size=int(population_size), num_generations=int(num_generations))

    # Save the results
    if isinstance(solution, list):
        # Multi-objective optimization - this branch can be removed if MOO is not supported
        log().warning(
            "Multi-objective optimization is not supported for dynamic DTN optimization. Using first solution only.")
        if solution:
            log().info("Saving first solution from multi-objective results")
            result_files = optimizer.save_results(solution[0])
            log().info(f"Results saved to: {result_files}")
        else:
            log().error("No solutions found in multi-objective optimization")
    else:
        # Single-objective optimization
        log().info("Saving optimization results")
        result_files = optimizer.save_results(solution)
        log().info(f"Results saved to: {result_files}")

    # Write a compact rerun summary JSON to help Part 3 analysis
    try:
        summary = {
            "network_type": network_type,
            "objective_function": objective_function,
            "num_phases": int(num_phases),
            "phase_durations": phase_durations,
            "population_size": int(population_size),
            "num_generations": int(num_generations),
            "testing_clusters": testing_clusters,
            # New constraint parameters (mirrors run_settings.json)
            "capex_per_phase_relaxing_pct": float(capex_per_phase_softening_pct),
            "total_exp_per_phase_relaxing_pct": float(total_exp_per_phase_softening_pct),
            "enforce_final_cumulative_capex": bool(enforce_final_cumulative_capex),
            "final_cumulative_capex_budget_total": float(final_cumulative_capex_budget_total),
            "enforce_final_cumulative_total_exp": bool(enforce_final_cumulative_total_exp),
            "final_cumulative_total_exp_budget_total": float(final_cumulative_total_exp_budget_total),
            "lock_committed_early_phases": bool(lock_committed_early_phases),
            "locked_phase_indices": locked_phase_indices,
            "lock_reference_genome": str(lock_reference_genome),
            "custom_locked_genome": custom_locked_genome,
            "enforce_per_phase_soft_ceiling": bool(enforce_per_phase_soft_ceiling),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
        }
        # get genome if available
        try:
            if isinstance(solution, dict) and 'genome' in solution:
                summary["genome"] = solution['genome']
            elif hasattr(optimizer, 'solution') and isinstance(optimizer.solution, dict) and 'genome' in optimizer.solution:
                summary["genome"] = optimizer.solution['genome']
        except Exception:
            pass
        # save to rerun_results root
        rerun_root = Path(locator.get_dynamic_dtn_optimization_results_folder())
        rerun_root.mkdir(parents=True, exist_ok=True)
        with open(rerun_root / 'rerun_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        log().info(f"Saved rerun summary to {rerun_root / 'rerun_summary.json'}")
    except Exception as e:
        log().warning(f"Could not write rerun summary JSON: {e}")

    # Copy results from temp scenario to rerun_results folder
    temp_results_dir = Path(locator.get_dynamic_dtn_optimization_temp_scenario_dtn_expansion_folder())
    rerun_results_dir = Path(locator.get_dynamic_dtn_optimization_results_folder())
    rerun_opt_results_dir = rerun_results_dir / "optimization_results"
    rerun_opt_results_dir.mkdir(parents=True, exist_ok=True)

    log().info(f"Copying results from {temp_results_dir} to {rerun_opt_results_dir}")

    # Copy optimization results
    for file in temp_results_dir.glob("*.csv"):
        target_file = rerun_opt_results_dir / file.name
        shutil.copy2(file, target_file)
        log().info(f"Copied {file.name} to {target_file}")

    # Copy phase supply files
    temp_phase_supply_dir = temp_results_dir / "phase_supply_files"
    rerun_phase_supply_dir = rerun_results_dir / "phase_supply_files"
    rerun_phase_supply_dir.mkdir(parents=True, exist_ok=True)
    if temp_phase_supply_dir.exists():
        for file in temp_phase_supply_dir.glob("*.csv"):
            target_file = rerun_phase_supply_dir / file.name
            shutil.copy2(file, target_file)
            log().info(f"Copied {file.name} to {target_file}")

    # Copy updated metrics file
    metrics_file = Path(locator.get_dynamic_dtn_optimization_updated_metrics_file())
    if metrics_file.exists():
        target_file = rerun_results_dir / metrics_file.name
        shutil.copy2(metrics_file, target_file)
        log().info(f"Copied {metrics_file.name} to {target_file}")

    # Log execution time
    time_elapsed = time.time() - start
    log().info(f"Execution time: {time_elapsed:.2f} seconds")


if __name__ == '__main__':
    args = parse_args()

    config = cea.config.Configuration()

    if args.scenario:
        config.scenario = args.scenario
    if args.config:
        config.load(args.config)

    main(config)