# Dynamic DTN Rerun Optimization Part 2 Documentation

## Overview

The Dynamic DTN Rerun Optimization Part 2 module extends the dynamic DTN optimization process to include the Genetic Algorithm (GA) optimization part. This module ensures that all input files required for rerunning the GA are properly located in the temporary scenario folders.

## Key Features

1. **File Preparation**: Copies all required files to the temporary scenario folder
2. **Metrics Generation**: Generates updated metrics based on the temporary scenario files
3. **GA Optimization**: Runs the genetic algorithm optimization with the updated metrics
4. **Results Saving**: Saves the optimization results to files

## File Structure

### Input Files

The following files are required for the GA optimization:

1. **Cluster Files**:
   - `cluster_edges.csv`: Copied from the original DTN expansion directory
   - `cluster_nodes.csv`: Copied from the original DTN expansion directory

2. **Demand Files**:
   - `Total_demand.csv`: Located in the temporary scenario's demand directory

3. **Building Supply Files**:
   - `supply.csv`: Copied from the original scenario to the temporary scenario

4. **Database Files**:
   - Supply systems:
     - `SUPPLY_HEATING.csv`: Copied to the temporary scenario's database directory
     - `SUPPLY_COOLING.csv`: Copied to the temporary scenario's database directory
     - `SUPPLY_HOTWATER.csv`: Copied to the temporary scenario's database directory
   - Cost data:
     - `THERMAL_GRID.csv`: Copied to the temporary scenario's database directory
     - `HEAT_EXCHANGERS.csv`: Copied to the temporary scenario's database directory

### Output Files

The optimization results are saved to the following files:

1. **Metrics File**:
   - `clusters_metrics_updated.csv`: Contains metrics for all cluster combinations

2. **Solution Files**:
   - `solution.json`: Contains the optimization solution in JSON format
   - `solution.csv`: Contains the cluster-phase mapping in CSV format

## Implementation Details

### File Copying

The `_prepare_optimization_files` method copies all required files to the temporary scenario folder:

```python
def _prepare_optimization_files(self):
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
    
    # Create a custom InputLocator for the temp scenario
    self.temp_locator = self._create_temp_scenario_locator()
    
    # Copy building supply file
    self._copy_building_supply_file()
    
    # Copy database files
    self._copy_database_files()
    
    # Verify Total_demand.csv exists in temp scenario
    total_demand_path = self.temp_demand_dir / "Total_demand.csv"
    if not total_demand_path.exists():
        raise FileNotFoundError(f"Total_demand.csv not found in temp scenario: {total_demand_path}")
    
    # Load and verify Total_demand.csv
    total_demand = pd.read_csv(total_demand_path)
    
    # Verify required columns exist
    # ...
    
    return total_demand
```

### Metrics Generation

The `_generate_cluster_metrics` method generates metrics for all cluster combinations:

```python
def _generate_cluster_metrics(self, total_demand):
    # Load cluster files
    temp_opt_dir = self.temp_scenario / "outputs" / "data" / "optimization" / "dtn_expansion"
    self.cluster_edges = pd.read_csv(temp_opt_dir / "cluster_edges.csv")
    self.cluster_nodes = pd.read_csv(temp_opt_dir / "cluster_nodes.csv")
    
    # Get unique clusters (excluding 0 and -1)
    self.all_clusters = sorted([c for c in self.cluster_edges['cluster'].unique() if c > 0])
    
    # Filter to testing clusters if specified
    # ...
    
    # Store total demand for later use
    self.total_demand = total_demand
    
    # Generate all possible combinations of clusters
    # ...
    
    # Calculate metrics for each combination
    # ...
    
    # Create a dictionary of metrics for each cluster combination
    self.cluster_metrics = {}
    for metrics in all_metrics:
        clusters_str = metrics['clusters']
        self.cluster_metrics[clusters_str] = metrics
    
    return metrics_df
```

### GA Optimization

The `run_optimization` method runs the genetic algorithm optimization:

```python
def run_optimization(self, population_size=50, num_generations=30):
    # Set up parameters for optimization
    self.num_phases = self.config.dtn_expansion_optimization.num_phases
    
    # Parse phase durations from config
    # ...
    
    # Set up the genetic algorithm
    setup_creator(self.multi_objective_mode, self.objective_function, self.multi_objective_functions)
    self._setup_genetic_algorithm()
    
    # Create initial population
    pop = self.toolbox.population(n=population_size)
    
    # Evaluate the individuals with an invalid fitness
    # ...
    
    # Run the genetic algorithm
    for gen in range(num_generations):
        # Select the next generation individuals
        # ...
        
        # Apply crossover and mutation
        # ...
        
        # Evaluate the individuals with an invalid fitness
        # ...
        
        # Replace the population with the offspring
        # ...
        
        # Update the best individual
        # ...
    
    # Convert the best individual to a solution
    solution = {
        'individual': list(best_ind),
        'fitness': best_ind.fitness.values,
        'cluster_phase_map': {cluster: phase for cluster, phase in zip(self.all_clusters, best_ind)},
        'phases': {}
    }
    
    # Group clusters by phase
    # ...
    
    # Save the solution
    self._save_optimization_results(solution)
    
    return solution
```

### Emissions Calculation

The `calculate_emissions_for_genome` method calculates emissions for a specific genome:

```python
def calculate_emissions_for_genome(self, cluster_phase_map):
    # Initialize results dictionary
    phase_emissions = {phase: {'operation': 0}
                      for phase in range(1, self.num_phases + 1)}
    
    # Get original supply file
    supply_file = self.temp_locator.get_building_supply()
    original_supply_df = pd.read_csv(supply_file)
    
    # Get district supply systems from cluster 0 buildings
    # ...
    
    # Create phase-specific supply files
    # ...
    
    # Process each phase separately
    for phase in range(1, self.num_phases + 1):
        # Get clusters connected in this phase
        # ...
        
        # Make a copy of the original supply file for this phase
        # ...
        
        # Update supply systems for connected buildings
        # ...
        
        # Save the phase-specific supply file
        # ...
        
        # Run LCA operation module with the phase-specific supply file
        lca_operation(self.temp_locator, custom_supply_path=str(phase_supply_path))
        
        # Load LCA results
        lca_operation_results = pd.read_csv(self.temp_locator.get_lca_operation())
        
        # Calculate total emissions for all buildings in testing clusters
        phase_emissions[phase]['operation'] = lca_operation_results['GHG_sys_tonCO2'].sum()
    
    return phase_emissions, has_non_district_scale
```

### Cost Calculation

The `_calculate_phase_capex`, `calculate_roi`, and `calculate_npv` methods calculate costs for a specific phase:

```python
def _calculate_phase_capex(self, clusters, phase):
    # Get required pipes for the clusters
    required_pipes = self._get_required_pipes(set(clusters), self.cluster_edges)
    
    # Calculate pipe CAPEX
    # ...
    
    # Calculate HEX CAPEX
    # ...
    
    # Calculate pump CAPEX
    # ...
    
    # Calculate cooling plant CAPEX
    # ...
    
    # Apply interest rate adjustment based on phase
    # ...
    
    return total_capex

def calculate_roi(self, clusters, phase):
    # Get metrics for this cluster set
    # ...
    
    # Calculate CAPEX with interest rate adjustment
    # ...
    
    # Calculate annual revenue and O&M costs
    # ...
    
    # Calculate present value of net annual returns
    # ...
    
    # Calculate Discounted ROI
    # ...
    
    return roi

def calculate_npv(self, clusters, phase, years=20):
    # Calculate CAPEX with interest rate adjustment
    # ...
    
    # Calculate annual revenue and O&M costs
    # ...
    
    # Calculate NPV
    # ...
    
    return npv
```

## Usage

The Dynamic DTN Rerun Optimization Part 2 module is used as follows:

```python
from cea.optimization_new.dynamic_dtn_optimization_part2 import CustomDTNRerunOptimizer

# Create the optimizer
optimizer = CustomDTNRerunOptimizer(locator, config)

# Run the optimization
results = optimizer.run()

# Check the results
if results["status"] == "success":
    print(f"Optimization completed successfully!")
    print(f"Results saved to: {results['output_dir']}")
else:
    print(f"Optimization failed: {results['error']}")
```

## Conclusion

The Dynamic DTN Rerun Optimization Part 2 module extends the dynamic DTN optimization process to include the Genetic Algorithm optimization part. It ensures that all input files required for rerunning the GA are properly located in the temporary scenario folders, and it uses the updated metrics to find the optimal phase assignment for the DTN expansion.