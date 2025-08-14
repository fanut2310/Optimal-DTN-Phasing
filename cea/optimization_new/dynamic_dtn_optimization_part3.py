from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

import cea.config
import cea.inputlocator

__author__ = "Fan Ut Chang"
__copyright__ = "Copyright 2025, City Energy Analyst"
__license__ = "MIT"


def log() -> logging.Logger:
    lg = logging.getLogger("cea.dynamic_dtn_optimization_part3")
    if not lg.handlers:
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s | %(levelname)5s | %(message)s",
                            datefmt="%H:%M:%S")
    return lg


def parse_args():
    parser = argparse.ArgumentParser(description="Dynamic DTN Optimization Part 3: Sensitivity Analysis")
    parser.add_argument('-s', '--scenario', help='Path to the scenario folder')
    parser.add_argument('-c', '--config', help='Path to the config file')
    return parser.parse_args()


def _safe_literal_list(x) -> List[int]:
    """Safely parse a genome string like "[2,3,1]" to a list of ints."""
    import ast
    if isinstance(x, list):
        return [int(v) for v in x]
    if isinstance(x, str):
        try:
            val = ast.literal_eval(x)
            if isinstance(val, (list, tuple)):
                return [int(v) for v in val]
        except Exception:
            pass
    return []


def _ensure_analysis_folder(locator: cea.inputlocator.InputLocator) -> Path:
    base = Path(locator.get_dynamic_dtn_optimization_folder())
    out = base / 'analysis'
    out.mkdir(parents=True, exist_ok=True)
    return out


def _load_run_settings(locator: cea.inputlocator.InputLocator) -> Dict:
    settings_path = Path(locator.get_dynamic_dtn_optimization_temp_scenario_dtn_expansion_folder()) / 'run_settings.json'
    if settings_path.exists():
        try:
            with open(settings_path, 'r') as f:
                data = json.load(f)
                log().info(f"Loaded run settings from {settings_path}")
                return data
        except Exception as e:
            log().warning(f"Failed to load run settings JSON: {e}")
    else:
        log().info("run_settings.json not found; proceeding with defaults where needed")
    return {}


def _detect_energy_columns(df: pd.DataFrame) -> Dict[str, str]:
    """
    Heuristically detect column names for heating, cooling, dhw, electricity annual demands and GFA in Total_demand.csv.
    Returns mapping keys: 'heating','cooling','dhw','electricity','gfa' with value None if not found.
    """
    cols = {k: None for k in ['heating', 'cooling', 'dhw', 'electricity', 'gfa']}
    lc = {c.lower(): c for c in df.columns}
    # GFA
    for key in ['gfa_m2', 'Aref_m2', 'aref_m2', 'gfa', 'GFA_m2'.lower()]:
        if key in lc:
            cols['gfa'] = lc[key]
            break
    # Heating
    for key in ['heating_kwhyr', 'qh_sys_mwhyr', 'qh_sys_kwhyr', 'qh_kwhyr', 'qh_demand_mwhyr', 'heating']:
        if key in lc:
            cols['heating'] = lc[key]
            break
    # Cooling
    for key in ['cooling_kwhyr', 'qc_sys_mwhyr', 'qc_sys_kwhyr', 'qc_kwhyr', 'cooling']:
        if key in lc:
            cols['cooling'] = lc[key]
            break
    # DHW
    for key in ['dhw_kwhyr', 'qww_sys_mwhyr', 'qww_sys_kwhyr', 'qww_kwhyr', 'dhw']:
        if key in lc:
            cols['dhw'] = lc[key]
            break
    # Electricity
    for key in ['electricity_kwhyr', 'E_sys_kWhyr'.lower(), 'e_sys_kwhyr', 'electricity']:
        if key in lc:
            cols['electricity'] = lc[key]
            break
    return cols


def _sum_energy(df: pd.DataFrame, cols_map: Dict[str, Optional[str]]) -> Dict[str, float]:
    out = {}
    for k in ['heating', 'cooling', 'dhw', 'electricity']:
        col = cols_map.get(k)
        out[k] = float(df[col].sum()) if col and col in df.columns else np.nan
    gfa_col = cols_map.get('gfa')
    out['gfa'] = float(df[gfa_col].sum()) if gfa_col and gfa_col in df.columns else np.nan
    return out


def _compute_district_deltas(baseline_td: pd.DataFrame, modified_td: pd.DataFrame) -> Dict[str, float]:
    bcols = _detect_energy_columns(baseline_td)
    mcols = _detect_energy_columns(modified_td)
    # use baseline-detected columns for both if possible
    if any(v is None for v in bcols.values()):
        log().warning("Some energy/GFA columns not detected in baseline Total_demand.csv; deltas may be partial.")
    bsum = _sum_energy(baseline_td, bcols)
    msum = _sum_energy(modified_td, bcols if all((c in modified_td.columns if c else True) for c in bcols.values()) else mcols)
    deltas = {}
    for k in ['heating', 'cooling', 'dhw', 'electricity', 'gfa']:
        b = bsum.get(k, np.nan)
        m = msum.get(k, np.nan)
        deltas[f"baseline_{k}"] = b
        deltas[f"modified_{k}"] = m
        if pd.notna(b) and b != 0 and pd.notna(m):
            deltas[f"delta_{k}"] = m - b
            deltas[f"delta_{k}_pct"] = (m - b) / b
        else:
            deltas[f"delta_{k}"] = np.nan
            deltas[f"delta_{k}_pct"] = np.nan
    return deltas


def _load_cluster_nodes(locator: cea.inputlocator.InputLocator) -> pd.DataFrame:
    path = Path(locator.get_dynamic_dtn_optimization_temp_scenario_dtn_expansion_folder()) / 'cluster_nodes.csv'
    if not path.exists():
        raise FileNotFoundError(f"cluster_nodes.csv not found in temp scenario: {path}")
    df = pd.read_csv(path)
    if 'building' not in df.columns or 'cluster' not in df.columns:
        raise ValueError("cluster_nodes.csv must contain 'building' and 'cluster' columns")
    return df


def _compute_cluster_level_deltas(baseline_td: pd.DataFrame, modified_td: pd.DataFrame, cluster_nodes: pd.DataFrame,
                                   testing_clusters: Optional[List[int]] = None) -> pd.DataFrame:
    # filter to consumer buildings
    nodes = cluster_nodes.copy()
    if 'type' in nodes.columns:
        nodes = nodes[nodes['type'] == 'CONSUMER']
    # merge baseline and modified totals on building name
    # Detect building id column in Total_demand
    building_col = None
    for cand in ['name', 'Name', 'BUILDING', 'building']:
        if cand in baseline_td.columns and cand in modified_td.columns:
            building_col = cand
            break
    if not building_col:
        raise ValueError("Could not detect building name column in Total_demand.csv (expected 'name').")

    # Detect energy columns
    bcols = _detect_energy_columns(baseline_td)
    cols_needed = [c for c in [bcols.get('heating'), bcols.get('cooling'), bcols.get('dhw'), bcols.get('electricity'), bcols.get('gfa')] if c]

    b = baseline_td[[building_col] + [c for c in cols_needed if c in baseline_td.columns]].copy()
    m = modified_td[[building_col] + [c for c in cols_needed if c in modified_td.columns]].copy()

    bm = b.merge(m, on=building_col, suffixes=('_base', '_mod'))
    # attach cluster id
    if 'building' in nodes.columns:
        bm = bm.merge(nodes[['building', 'cluster']], left_on=building_col, right_on='building', how='left')
        bm.drop(columns=['building'], inplace=True)
    else:
        # fallback if nodes has different column name
        raise ValueError("cluster_nodes.csv missing 'building' column")

    # compute deltas per building, then aggregate by cluster (sum then percent wrt baseline sums)
    agg_rows = []
    for cluster_id, grp in bm.groupby('cluster'):
        if cluster_id is None or (isinstance(cluster_id, float) and np.isnan(cluster_id)):
            continue
        if testing_clusters and int(cluster_id) not in set(testing_clusters):
            continue
        row: Dict[str, float] = {"cluster": int(cluster_id)}
        for k, base_col in [('heating', bcols.get('heating')), ('cooling', bcols.get('cooling')),
                            ('dhw', bcols.get('dhw')), ('electricity', bcols.get('electricity')), ('gfa', bcols.get('gfa'))]:
            if base_col and f"{base_col}_base" in grp.columns and f"{base_col}_mod" in grp.columns:
                bsum = float(grp[f"{base_col}_base"].sum())
                msum = float(grp[f"{base_col}_mod"].sum())
                row[f"baseline_{k}"] = bsum
                row[f"modified_{k}"] = msum
                row[f"delta_{k}"] = msum - bsum
                row[f"delta_{k}_pct"] = (msum - bsum) / bsum if bsum != 0 else np.nan
        agg_rows.append(row)
    return pd.DataFrame(agg_rows).sort_values(by='cluster')


def _load_best_genome_from_results(results_csv: Path, objective: str) -> Tuple[List[int], Dict[str, float]]:
    df = pd.read_csv(results_csv)
    if df.empty or 'genome' not in df.columns:
        return [], {}
    obj = (objective or 'NPV').lower()
    col = None
    if obj == 'npv' and 'fitness_NPV' in df.columns:
        col = 'fitness_NPV'
        best_row = df.loc[df[col].idxmax()]
    elif obj == 'roi' and 'fitness_ROI' in df.columns:
        col = 'fitness_ROI'
        best_row = df.loc[df[col].idxmax()]
    elif obj == 'emissions' and 'fitness_emissions' in df.columns:
        col = 'fitness_emissions'
        best_row = df.loc[df[col].idxmin()]
    elif 'fitness_NPV' in df.columns:
        best_row = df.loc[df['fitness_NPV'].idxmax()]
    else:
        best_row = df.iloc[0]
    genome = _safe_literal_list(best_row['genome'])
    metrics = {c: best_row[c] for c in df.columns if c.startswith('fitness_')}
    return genome, metrics


def _load_baseline_and_rerun_genomes(locator: cea.inputlocator.InputLocator, network_type: str,
                                     settings: Dict) -> Tuple[List[int], List[int], Dict, Dict]:
    # Baseline
    baseline_results = Path(locator.get_dtn_expansion_optimization_results_folder()) / f"all_evaluated_individuals_{network_type}.csv"
    baseline_genome: List[int] = []
    baseline_metrics: Dict = {}
    objective = settings.get('objective_function', 'NPV')
    if baseline_results.exists():
        baseline_genome, baseline_metrics = _load_best_genome_from_results(baseline_results, objective)
    else:
        log().warning(f"Baseline optimization results not found at {baseline_results}")

    # Rerun
    rerun_root = Path(locator.get_dynamic_dtn_optimization_results_folder())
    rerun_summary = rerun_root / 'rerun_summary.json'
    rerun_genome: List[int] = []
    rerun_metrics: Dict = {}
    if rerun_summary.exists():
        try:
            with open(rerun_summary, 'r') as f:
                data = json.load(f)
                rerun_genome = _safe_literal_list(data.get('genome', []))
                rerun_metrics['objective_function'] = data.get('objective_function')
        except Exception as e:
            log().warning(f"Failed to load rerun_summary.json: {e}")
    if not rerun_genome:
        # Try parse optimization_results CSV
        opt_dir = rerun_root / 'optimization_results'
        rerun_csv = opt_dir / f"all_evaluated_individuals_{network_type}.csv"
        if rerun_csv.exists():
            rerun_genome, rerun_metrics = _load_best_genome_from_results(rerun_csv, objective)
        else:
            log().warning(f"Rerun optimization results not found at {rerun_csv}")
    return baseline_genome, rerun_genome, baseline_metrics, rerun_metrics


def _compute_genome_deltas(baseline_genome: List[int], rerun_genome: List[int], clusters: List[int]) -> Dict[str, float]:
    if not baseline_genome or not rerun_genome or not clusters:
        return {
            'clusters_compared': 0,
            'hamming': np.nan,
            'hamming_norm': np.nan,
            'directional_shift': np.nan,
            'earlier': np.nan,
            'later': np.nan,
            'same': np.nan
        }
    # Map clusters to phases
    base_map = {c: p for c, p in zip(clusters, baseline_genome)}
    new_map = {c: p for c, p in zip(clusters, rerun_genome)}
    shared = [c for c in clusters if c in base_map and c in new_map]
    if not shared:
        return {
            'clusters_compared': 0,
            'hamming': np.nan,
            'hamming_norm': np.nan,
            'directional_shift': np.nan,
            'earlier': np.nan,
            'later': np.nan,
            'same': np.nan
        }
    base = np.array([base_map[c] for c in shared])
    new = np.array([new_map[c] for c in shared])
    diffs = new - base
    hamming = float(np.sum(diffs != 0))
    hamming_norm = hamming / len(shared)
    directional = float(np.sum(diffs))
    earlier = int(np.sum(diffs < 0))
    later = int(np.sum(diffs > 0))
    same = int(len(shared) - earlier - later)
    return {
        'clusters_compared': len(shared),
        'hamming': hamming,
        'hamming_norm': hamming_norm,
        'directional_shift': directional,
        'earlier': earlier,
        'later': later,
        'same': same
    }


def main(config):
    logger = log()
    logger.info("=" * 80)
    logger.info("Starting Dynamic DTN Optimization Part 3: Sensitivity Analysis")
    logger.info(f"Scenario: {config.scenario}")
    network_type = config.dynamic_dtn_optimization.network_type
    logger.info(f"Network type: {network_type}")
    logger.info("=" * 80)

    locator = cea.inputlocator.InputLocator(config.scenario)

    # Optional testing_clusters filter
    testing_clusters = None
    if hasattr(config.dtn_expansion_optimization, 'testing_clusters') and config.dtn_expansion_optimization.testing_clusters:
        if isinstance(config.dtn_expansion_optimization.testing_clusters, str):
            testing_clusters = [int(c.strip()) for c in config.dtn_expansion_optimization.testing_clusters.split(',') if c.strip()]
        else:
            testing_clusters = list(config.dtn_expansion_optimization.testing_clusters)

    analysis_dir = _ensure_analysis_folder(locator)

    # Load settings (objective function, etc.)
    settings = _load_run_settings(locator)

    # Load total demand (baseline vs modified temp scenario)
    baseline_total_path = Path(locator.get_total_demand())
    modified_total_path = Path(locator.get_dynamic_dtn_optimization_temp_scenario_total_demand())
    if not baseline_total_path.exists() or not modified_total_path.exists():
        logger.error("Total_demand.csv missing: baseline or modified temp scenario not found.")
        logger.error(f"Baseline: {baseline_total_path}")
        logger.error(f"Modified: {modified_total_path}")
        return

    baseline_td = pd.read_csv(baseline_total_path)
    modified_td = pd.read_csv(modified_total_path)

    # Compute district-level deltas
    district_deltas = _compute_district_deltas(baseline_td, modified_td)

    # Compute cluster-level deltas
    cluster_nodes = _load_cluster_nodes(locator)
    cluster_deltas_df = _compute_cluster_level_deltas(baseline_td, modified_td, cluster_nodes, testing_clusters)

    # Determine the clusters order used by the optimizer (exclude 0 and negatives)
    clusters_sorted = sorted([int(c) for c in cluster_nodes['cluster'].unique() if isinstance(c, (int, np.integer)) and c > 0])
    if testing_clusters:
        clusters_sorted = [c for c in clusters_sorted if c in set(testing_clusters)]

    # Load genomes and compute genome deltas
    baseline_genome, rerun_genome, baseline_metrics, rerun_metrics = _load_baseline_and_rerun_genomes(locator, network_type, settings)
    genome_deltas = _compute_genome_deltas(baseline_genome, rerun_genome, clusters_sorted)

    # Try to load emissions totals (optional)
    baseline_lca = Path(locator.scenario) / 'outputs' / 'data' / 'emissions' / 'Total_LCA_operation.csv'
    modified_lca = Path(locator.get_dynamic_dtn_optimization_temp_scenario_lca_operation_file())
    emissions_summary = {}
    try:
        if baseline_lca.exists():
            b = pd.read_csv(baseline_lca)
            emissions_summary['baseline_emissions_tonCO2'] = float(b['GHG_sys_tonCO2'].sum()) if 'GHG_sys_tonCO2' in b.columns else np.nan
            emissions_summary['baseline_gfa_m2'] = float(b['GFA_m2'].sum()) if 'GFA_m2' in b.columns else np.nan
        if modified_lca.exists():
            m = pd.read_csv(modified_lca)
            emissions_summary['modified_emissions_tonCO2'] = float(m['GHG_sys_tonCO2'].sum()) if 'GHG_sys_tonCO2' in m.columns else np.nan
            emissions_summary['modified_gfa_m2'] = float(m['GFA_m2'].sum()) if 'GFA_m2' in m.columns else np.nan
        if 'baseline_emissions_tonCO2' in emissions_summary and 'modified_emissions_tonCO2' in emissions_summary:
            b_e = emissions_summary['baseline_emissions_tonCO2']
            m_e = emissions_summary['modified_emissions_tonCO2']
            emissions_summary['delta_emissions_tonCO2'] = m_e - b_e if all(pd.notna([b_e, m_e])) else np.nan
            emissions_summary['delta_emissions_pct'] = (m_e - b_e) / b_e if (pd.notna(b_e) and b_e != 0 and pd.notna(m_e)) else np.nan
    except Exception as e:
        logger.warning(f"Failed to load emissions totals: {e}")

    # Aggregate master summary
    master_row = {
        'network_type': network_type,
        'objective_function': settings.get('objective_function', 'NPV'),
        'num_phases': settings.get('num_phases'),
        'phase_durations': settings.get('phase_durations'),
        'testing_clusters': ','.join(str(c) for c in testing_clusters) if testing_clusters else '',
        'baseline_genome': json.dumps(baseline_genome),
        'rerun_genome': json.dumps(rerun_genome)
    }
    master_row.update(district_deltas)
    master_row.update(genome_deltas)
    master_row.update({f"baseline_{k}": v for k, v in baseline_metrics.items()})
    master_row.update({f"rerun_{k}": v for k, v in rerun_metrics.items()})
    master_row.update(emissions_summary)

    # Save outputs
    master_summary_csv = analysis_dir / 'master_summary.csv'
    if master_summary_csv.exists():
        # append
        pd.DataFrame([master_row]).to_csv(master_summary_csv, mode='a', header=False, index=False)
    else:
        pd.DataFrame([master_row]).to_csv(master_summary_csv, index=False)
    logger.info(f"Saved master summary to {master_summary_csv}")

    # Save cluster-level deltas
    cluster_deltas_csv = analysis_dir / 'cluster_level_deltas.csv'
    cluster_deltas_df.to_csv(cluster_deltas_csv, index=False)
    logger.info(f"Saved cluster-level deltas to {cluster_deltas_csv}")

    # Save a JSON summary for quick inspection
    json_summary_path = analysis_dir / 'rerun_analysis_summary.json'
    try:
        with open(json_summary_path, 'w') as f:
            json.dump(master_row, f, indent=2)
        logger.info(f"Saved JSON summary to {json_summary_path}")
    except Exception as e:
        logger.warning(f"Failed to save JSON summary: {e}")

    logger.info("Sensitivity analysis (core) completed.")
    return None


if __name__ == '__main__':
    args = parse_args()
    config = cea.config.Configuration(args.config)
    config.scenario = args.scenario
    main(config)
