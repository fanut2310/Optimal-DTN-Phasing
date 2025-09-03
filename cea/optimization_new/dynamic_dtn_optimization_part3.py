from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

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
    for key in [
        'electricity_kwhyr',
        'e_sys_kwhyr',
        'el_sys_kwhyr',
        'electricity',
        'e_sys_mwhyr',
        'el_sys_mwhyr',
        'e_sys_kwhyr',
        'e_sys_mwhyr'
    ]:
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
    """Load cluster_nodes.csv with robust fallbacks.
    Order of attempts:
    1) Temp scenario DTN expansion folder
    2) Baseline locator.get_dtn_cluster_nodes_file()
    3) Baseline DTN expansion results folder / 'cluster_nodes.csv'
    If none found or invalid, returns an empty DataFrame (columns: building, cluster, type) and logs a warning.
    """
    candidates: List[Path] = []
    try:
        candidates.append(Path(locator.get_dynamic_dtn_optimization_temp_scenario_dtn_expansion_folder()) / 'cluster_nodes.csv')
    except Exception:
        pass
    try:
        # Known baseline path exposed by locator
        baseline_nodes = Path(locator.get_dtn_cluster_nodes_file())
        candidates.append(baseline_nodes)
    except Exception:
        pass
    try:
        # Common baseline results location
        candidates.append(Path(locator.get_dtn_expansion_optimization_results_folder()) / 'cluster_nodes.csv')
    except Exception:
        pass

    for path in candidates:
        try:
            if path and path.exists():
                df = pd.read_csv(path)
                if 'building' in df.columns and 'cluster' in df.columns:
                    log().info(f"Loaded cluster_nodes from: {path}")
                    return df
                else:
                    log().warning(f"cluster_nodes at {path} missing required columns; trying next candidate.")
        except Exception as e:
            log().warning(f"Failed to load cluster_nodes from {path}: {e}")

    log().warning("cluster_nodes.csv not found in temp or baseline; proceeding without cluster-level plots.")
    return pd.DataFrame(columns=['building', 'cluster', 'type'])


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


def _generate_figures(analysis_dir: Path,
                      district_deltas: Dict[str, float],
                      cluster_deltas_df: pd.DataFrame,
                      baseline_genome: List[int],
                      rerun_genome: List[int],
                      clusters_sorted: List[int],
                      genome_deltas: Optional[Dict[str, float]] = None,
                      genome_comparison_df: Optional[pd.DataFrame] = None,
                      lock_committed_early_phases: bool = False,
                      locked_phase_indices: Optional[List[int]] = None,
                      emissions_summary: Optional[Dict[str, float]] = None) -> None:
    # Ensure non-interactive backend for headless environments
    try:
        plt.switch_backend('Agg')
    except Exception:
        pass

    figs_dir = analysis_dir / 'figures'
    try:
        figs_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        log().warning(f"Could not create figures folder: {figs_dir}")
        return

    # 1) District-level totals bar chart
    try:
        labels = []
        base_vals = []
        mod_vals = []
        for k_base, k_mod, label in [
            ('baseline_heating', 'modified_heating', 'Heating'),
            ('baseline_cooling', 'modified_cooling', 'Cooling'),
            ('baseline_dhw', 'modified_dhw', 'DHW'),
            ('baseline_electricity', 'modified_electricity', 'Electricity'),
        ]:
            b = district_deltas.get(k_base)
            m = district_deltas.get(k_mod)
            if b is not None and m is not None and not (pd.isna(b) or pd.isna(m)):
                labels.append(label)
                base_vals.append(float(b))
                mod_vals.append(float(m))
        if labels:
            x = np.arange(len(labels))
            w = 0.28
            plt.figure(figsize=(8, 5))
            plt.bar(x - w/2, base_vals, width=w, label='Baseline')
            plt.bar(x + w/2, mod_vals, width=w, label='Modified')
            plt.xticks(x, labels)
            plt.ylabel('Annual energy (as in Total_demand units)')
            plt.title('District totals: Baseline vs Modified')
            plt.legend()
            plt.tight_layout()
            out_path = figs_dir / 'district_totals_baseline_vs_modified.png'
            plt.savefig(out_path, dpi=200)
            log().info(f"Saved figure: {out_path}")
            plt.close()
    except Exception as e:
        log().warning(f"Failed to create district totals figure: {e}")

    # 2) Cluster-level percent change grouped bars
    try:
        if isinstance(cluster_deltas_df, pd.DataFrame) and not cluster_deltas_df.empty:
            pct_cols = [c for c in cluster_deltas_df.columns if c.endswith('_pct') and any(s in c for s in ['heating','cooling','dhw','electricity'])]
            if pct_cols and 'cluster' in cluster_deltas_df.columns:
                dfm = cluster_deltas_df[['cluster'] + pct_cols].copy()
                # Melt
                dfm = dfm.melt(id_vars='cluster', var_name='metric', value_name='pct_delta')
                # Clean names
                dfm['metric'] = (dfm['metric']
                                 .str.replace('delta_', '', regex=False)
                                 .str.replace('_pct', '', regex=False)
                                 .str.title())
                # Scale fractions to percent for plotting
                dfm['pct_delta'] = dfm['pct_delta'] * 100.0
                # Order clusters numerically ascending
                try:
                    cluster_order = sorted([int(c) for c in dfm['cluster'].dropna().unique()])
                except Exception:
                    cluster_order = list(pd.to_numeric(dfm['cluster'], errors='coerce').dropna().astype(int).sort_values().unique())
                dfm['cluster'] = pd.Categorical(dfm['cluster'], cluster_order)
                metrics = sorted(dfm['metric'].dropna().unique().tolist())
                x = np.arange(len(cluster_order))
                w = min(0.6/ max(1, len(metrics)), 0.12)
                plt.figure(figsize=(max(8, len(cluster_order)*0.35), 5))
                for i, met in enumerate(metrics):
                    yi = dfm[dfm['metric'] == met].set_index('cluster').reindex(cluster_order)['pct_delta'].values
                    plt.bar(x + (i - (len(metrics)-1)/2)*w, yi, width=w, label=met)
                plt.axhline(0, color='k', linewidth=0.8)
                plt.xticks(x, [str(c) for c in cluster_order], rotation=45)
                plt.ylabel('Percent change (%)')
                plt.title('Cluster-level percent changes')
                plt.legend()
                plt.tight_layout()
                out_path = figs_dir / 'cluster_level_percent_changes.png'
                plt.savefig(out_path, dpi=200)
                log().info(f"Saved figure: {out_path}")
                plt.close()
    except Exception as e:
        log().warning(f"Failed to create cluster-level figure: {e}")

    # 3) Genome phase change plot (optional)
    try:
        if baseline_genome and rerun_genome and clusters_sorted:
            n = min(len(baseline_genome), len(rerun_genome), len(clusters_sorted))
            if n > 0:
                base = list(map(int, baseline_genome[:n]))
                new = list(map(int, rerun_genome[:n]))
                cl = list(map(int, clusters_sorted[:n]))
                plt.figure(figsize=(max(8, n*0.3), 4))
                plt.plot(cl, base, marker='o', label='Baseline phase')
                plt.plot(cl, new, marker='s', label='Rerun phase')
                for k in range(n):
                    if base[k] != new[k]:
                        plt.plot([cl[k], cl[k]], [base[k], new[k]], color='gray', linewidth=0.8)
                plt.xlabel('Cluster ID')
                plt.ylabel('Phase')
                # enforce integer-only ticks for phases
                try:
                    max_phase_plot = int(max(max(base), max(new))) if base and new else None
                except Exception:
                    max_phase_plot = None
                ax = plt.gca()
                ax.yaxis.set_major_locator(MaxNLocator(integer=True))
                if max_phase_plot and max_phase_plot > 0:
                    ax.set_ylim(0.5, max_phase_plot + 0.5)
                    ax.set_yticks(range(1, max_phase_plot + 1))
                plt.title('Genome phase changes by cluster')
                plt.legend()
                plt.tight_layout()
                out_path = figs_dir / 'genome_phase_changes.png'
                plt.savefig(out_path, dpi=200)
                log().info(f"Saved figure: {out_path}")
                plt.close()

                # 3b) With locks overlay
                try:
                    if lock_committed_early_phases and locked_phase_indices:
                        locked_set = set(int(x) for x in locked_phase_indices)
                        plt.figure(figsize=(max(8, n*0.3), 4))
                        plt.plot(cl, base, marker='o', label='Baseline phase')
                        plt.plot(cl, new, marker='s', label='Rerun phase')
                        # Highlight locked clusters (based on baseline phase in locked phases)
                        locked_x = [cl[k] for k in range(n) if base[k] in locked_set]
                        locked_y = [base[k] for k in range(n) if base[k] in locked_set]
                        if locked_x:
                            plt.scatter(locked_x, locked_y, s=80, facecolors='none', edgecolors='red', linewidths=1.5, label='Locked (by phase)')
                        for k in range(n):
                            if base[k] != new[k]:
                                plt.plot([cl[k], cl[k]], [base[k], new[k]], color='gray', linewidth=0.8)
                        plt.xlabel('Cluster ID')
                        plt.ylabel('Phase')
                        # enforce integer-only ticks for phases
                        try:
                            max_phase_plot = int(max(max(base), max(new))) if base and new else None
                        except Exception:
                            max_phase_plot = None
                        ax = plt.gca()
                        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
                        if max_phase_plot and max_phase_plot > 0:
                            ax.set_ylim(0.5, max_phase_plot + 0.5)
                            ax.set_yticks(range(1, max_phase_plot + 1))
                        plt.title('Genome phase changes (locked phases highlighted)')
                        plt.legend()
                        plt.tight_layout()
                        out_path = figs_dir / 'genome_phase_changes_with_locks.png'
                        plt.savefig(out_path, dpi=200)
                        log().info(f"Saved figure: {out_path}")
                        plt.close()
                except Exception as e2:
                    log().warning(f"Failed to create locked overlay figure: {e2}")
    except Exception as e:
        log().warning(f"Failed to create genome phase figure: {e}")

    # 4) Genome change counts (earlier/later/same)
    try:
        if isinstance(genome_deltas, dict):
            earlier = genome_deltas.get('earlier')
            later = genome_deltas.get('later')
            same = genome_deltas.get('same')
            if all(x is not None and not pd.isna(x) for x in [earlier, later, same]):
                vals = [int(earlier), int(later), int(same)]
                labels = ['Earlier', 'Later', 'Same']
                plt.figure(figsize=(6, 4))
                plt.bar(np.arange(3), vals, width=0.35, color=['#1b9e77','#d95f02','#7570b3'])
                plt.xticks(np.arange(3), labels)
                plt.ylabel('Cluster count')
                # integer y-axis ticks
                ax = plt.gca()
                ax.yaxis.set_major_locator(MaxNLocator(integer=True))
                try:
                    y_max = max(vals)
                    ax.set_ylim(0, y_max + 0.5)
                    ax.set_yticks(range(0, y_max + 1))
                except Exception:
                    pass
                plt.title('Genome changes summary')
                plt.tight_layout()
                out_path = figs_dir / 'genome_changes_counts.png'
                plt.savefig(out_path, dpi=200)
                log().info(f"Saved figure: {out_path}")
                plt.close()
    except Exception as e:
        log().warning(f"Failed to create genome changes counts figure: {e}")

    # 5) Phase volatility and baseline→destination stacks (single-run)
    try:
        if isinstance(genome_comparison_df, pd.DataFrame) and not genome_comparison_df.empty:
            gdf = genome_comparison_df.copy()
            if {'baseline_phase','rerun_phase','phase_delta'}.issubset(gdf.columns):
                # Phase volatility counts and avg |Δ|
                vol = gdf.groupby('baseline_phase').agg(
                    moved_count=('phase_delta', lambda s: int((s != 0).sum())),
                    avg_abs_delta=('phase_delta', lambda s: float(np.nanmean(np.abs(s))))
                ).reset_index().sort_values('baseline_phase')
                if not vol.empty:
                    # Bar for moved_count and line for avg_abs_delta (twin axis)
                    plt.figure(figsize=(max(6, len(vol)*0.6), 4))
                    x = np.arange(len(vol))
                    b = plt.bar(x, vol['moved_count'].values, width=0.5, color='#4c78a8', label='Moved count')
                    ax1 = plt.gca()
                    ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
                    ax2 = ax1.twinx()
                    ax2.plot(x, vol['avg_abs_delta'].values, color='#f58518', marker='o', label='Avg |Δphase|')
                    ax1.set_xlabel('Baseline phase')
                    ax1.set_ylabel('Moved count')
                    ax2.set_ylabel('Avg |Δphase|')
                    plt.xticks(x, vol['baseline_phase'].astype(int).astype(str))
                    plt.title('Phase volatility by baseline phase')
                    plt.tight_layout()
                    out_path = figs_dir / 'phase_volatility_counts.png'
                    plt.savefig(out_path, dpi=200)
                    try:
                        log().info(f"Saved figure: {out_path}")
                    except Exception:
                        pass
                    plt.close()
                else:
                    try:
                        log().info("[Part3] Skipping phase_volatility_counts: no movements detected (vol table empty).")
                    except Exception:
                        pass
                # Baseline phase to earlier/same/later stacks (shares)
                buckets = gdf.assign(change_bucket=np.sign(gdf['phase_delta']).map({-1:'Earlier',0:'Same',1:'Later'}))
                stacks = (buckets
                          .groupby(['baseline_phase','change_bucket'])
                          .size()
                          .unstack(fill_value=0)
                          .sort_index())
                if not stacks.empty:
                    stacks = stacks[['Earlier','Same','Later']] if {'Earlier','Same','Later'}.issubset(stacks.columns) else stacks
                    shares = stacks.div(stacks.sum(axis=1), axis=0)
                    plt.figure(figsize=(max(6, shares.shape[0]*0.6), 4))
                    bottom = np.zeros(shares.shape[0])
                    colors = {'Earlier':'#1b9e77','Same':'#7570b3','Later':'#d95f02'}
                    x = np.arange(shares.shape[0])
                    for col in shares.columns:
                        plt.bar(x, shares[col].values, bottom=bottom, color=colors.get(col, None), label=col, width=0.6)
                        bottom += shares[col].values
                    plt.xticks(x, shares.index.astype(int).astype(str))
                    plt.xlabel('Baseline phase')
                    plt.ylabel('Share')
                    plt.title('Baseline phase → earlier/same/later shares')
                    plt.legend()
                    plt.tight_layout()
                    out_path = figs_dir / 'baseline_phase_to_dest_stacks.png'
                    plt.savefig(out_path, dpi=200)
                    try:
                        log().info(f"Saved figure: {out_path}")
                    except Exception:
                        pass
                    plt.close()
                else:
                    try:
                        log().info("[Part3] Skipping baseline_phase_to_dest_stacks: no data for stacks (empty table).")
                    except Exception:
                        pass
            else:
                try:
                    log().info("[Part3] Skipping phase-volatility/stack plots: genome_comparison missing required columns.")
                except Exception:
                    pass
        else:
            try:
                log().info("[Part3] Skipping phase-volatility/stack plots: genome_comparison is empty or not provided.")
            except Exception:
                pass
    except Exception as e:
        log().warning(f"Failed to create phase volatility/stack plots: {e}")

    # 6) Phase transition matrix heatmap (optional; disabled by default)
    generate_phase_transition_matrix = False
    try:
        if generate_phase_transition_matrix and baseline_genome and rerun_genome and clusters_sorted:
            n = min(len(baseline_genome), len(rerun_genome), len(clusters_sorted))
            if n > 0:
                base = np.array(list(map(int, baseline_genome[:n])))
                new = np.array(list(map(int, rerun_genome[:n])))
                max_phase = int(max(base.max(), new.max())) if len(base) and len(new) else 0
                if max_phase > 0:
                    mat = np.zeros((max_phase, max_phase), dtype=int)
                    for i in range(n):
                        bi = max(1, int(base[i]))
                        nj = max(1, int(new[i]))
                        if bi <= max_phase and nj <= max_phase:
                            mat[bi-1, nj-1] += 1
                    plt.figure(figsize=(max(6, max_phase), max(5, max_phase)))
                    im = plt.imshow(mat, cmap='Blues', origin='upper')
                    plt.colorbar(im, fraction=0.046, pad=0.04, label='Cluster count')
                    plt.xticks(np.arange(max_phase), [str(j) for j in range(1, max_phase+1)])
                    plt.yticks(np.arange(max_phase), [str(i) for i in range(1, max_phase+1)])
                    ax = plt.gca()
                    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
                    plt.xlabel('Rerun phase')
                    plt.ylabel('Baseline phase')
                    # Add summary stats to title
                    try:
                        earlier = int(genome_deltas.get('earlier', 0)) if isinstance(genome_deltas, dict) else 0
                        later = int(genome_deltas.get('later', 0)) if isinstance(genome_deltas, dict) else 0
                        same = int(genome_deltas.get('same', 0)) if isinstance(genome_deltas, dict) else 0
                        hamming = int(genome_deltas.get('hamming', 0)) if isinstance(genome_deltas, dict) and pd.notna(genome_deltas.get('hamming')) else 0
                        bias = int(genome_deltas.get('directional_shift', 0)) if isinstance(genome_deltas, dict) and pd.notna(genome_deltas.get('directional_shift')) else 0
                        plt.title(f'Phase transition matrix (counts)\nEarlier: {earlier}, Later: {later}, Same: {same}; Hamming: {hamming}; Bias: {bias}')
                    except Exception:
                        plt.title('Phase transition matrix (counts)')
                    # Annotations
                    for i in range(max_phase):
                        for j in range(max_phase):
                            val = mat[i, j]
                            if val > 0:
                                plt.text(j, i, str(val), ha='center', va='center', color='black')
                    plt.tight_layout()
                    out_path = figs_dir / 'phase_transition_matrix.png'
                    plt.savefig(out_path, dpi=200)
                    log().info(f"Saved figure: {out_path}")
                    plt.close()
    except Exception as e:
        log().warning(f"Failed to create phase transition matrix: {e}")

    # 6) Emissions bar (optional)
    try:
        if isinstance(emissions_summary, dict):
            b = emissions_summary.get('baseline_emissions_tonCO2')
            m = emissions_summary.get('modified_emissions_tonCO2')
            if b is not None and m is not None and not (pd.isna(b) or pd.isna(m)):
                plt.figure(figsize=(6,4))
                plt.bar([0,1], [float(b), float(m)], width=0.28, tick_label=['Baseline','Modified'])
                plt.ylabel('Operational emissions (ton CO2)')
                delta_pct = emissions_summary.get('delta_emissions_pct')
                if delta_pct is not None and not pd.isna(delta_pct):
                    plt.title(f'Emissions: Baseline vs Modified (Δ {delta_pct*100:.1f}%)')
                else:
                    plt.title('Emissions: Baseline vs Modified')
                plt.tight_layout()
                out_path = figs_dir / 'emissions_baseline_vs_modified.png'
                plt.savefig(out_path, dpi=200)
                log().info(f"Saved figure: {out_path}")
                plt.close()
    except Exception as e:
        log().warning(f"Failed to create emissions figure: {e}")


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

    # Optionally reset analysis folder (clear cross-run records)
    try:
        reset_analysis_flag = bool(getattr(config.dynamic_dtn_optimization, 'reset_analysis', False))
    except Exception:
        reset_analysis_flag = False
    analysis_root = Path(locator.get_dynamic_dtn_optimization_folder()) / 'analysis'
    if reset_analysis_flag:
        try:
            shutil.rmtree(analysis_root, ignore_errors=True)
            logger.info(f"Reset-analysis: cleared {analysis_root}")
        except Exception as e:
            logger.warning(f"Reset-analysis: failed to clear {analysis_root}: {e}")

    analysis_dir = _ensure_analysis_folder(locator)

    # Load settings (objective function, etc.)
    settings = _load_run_settings(locator)

    # Run selection flags
    try:
        include_in_part3 = bool(settings.get('include_in_part3', settings.get('include-in-part3', True)))
    except Exception:
        include_in_part3 = True
    try:
        run_label = settings.get('run_label', settings.get('run-label', ''))
    except Exception:
        run_label = ''

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
    try:
        cluster_deltas_df = _compute_cluster_level_deltas(baseline_td, modified_td, cluster_nodes, testing_clusters)
    except Exception as e:
        logger.warning(f"Cluster-level deltas computation failed, will proceed without cluster plots: {e}")
        cluster_deltas_df = pd.DataFrame()

    # Determine the clusters order used by the optimizer (exclude 0 and negatives)
    try:
        if isinstance(cluster_nodes, pd.DataFrame) and 'cluster' in cluster_nodes.columns and not cluster_nodes.empty:
            clusters_sorted = sorted([int(c) for c in cluster_nodes['cluster'].unique() if pd.notna(c) and int(c) > 0])
        else:
            clusters_sorted = []
    except Exception:
        clusters_sorted = []
    if testing_clusters and clusters_sorted:
        clusters_sorted = [c for c in clusters_sorted if c in set(testing_clusters)]

    # Load genomes and compute genome deltas
    baseline_genome, rerun_genome, baseline_metrics, rerun_metrics = _load_baseline_and_rerun_genomes(locator, network_type, settings)

    # If we have genomes but no clusters, use position indices as surrogate cluster IDs
    if (not clusters_sorted) and baseline_genome and rerun_genome:
        n_sur = min(len(baseline_genome), len(rerun_genome))
        if n_sur > 0:
            clusters_sorted = list(range(1, n_sur + 1))

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

    # Read new Part 2 constraint parameters (prefer run_settings.json, fallback to config)
    def _get_setting(key_snake: str, key_hyphen: str, default=None):
        if key_snake in settings:
            return settings.get(key_snake)
        if key_hyphen in settings:
            return settings.get(key_hyphen)
        try:
            return getattr(config.dtn_expansion_optimization, key_snake)
        except Exception:
            return default

    capex_relax_pct = _get_setting('capex_per_phase_relaxing_pct', 'capex-per-phase-relaxing-pct')
    totalexp_relax_pct = _get_setting('total_exp_per_phase_relaxing_pct', 'total-exp-per-phase-relaxing-pct')
    enforce_final_capex = bool(_get_setting('enforce_final_cumulative_capex', 'enforce-final-cumulative-capex', False))
    final_capex_budget_total = _get_setting('final_cumulative_capex_budget_total', 'final-cumulative-capex-budget-total')
    enforce_final_totalexp = bool(_get_setting('enforce_final_cumulative_total_exp', 'enforce-final-cumulative-total-exp', False))
    final_totalexp_budget_total = _get_setting('final_cumulative_total_exp_budget_total', 'final-cumulative-total-exp-budget-total')
    lock_committed = bool(_get_setting('lock_committed_early_phases', 'lock-committed-early-phases', False))

    locked_phase_indices_val = _get_setting('locked_phase_indices', 'locked-phase-indices')
    # Normalize locked_phase_indices to List[int]
    locked_phase_indices: List[int] = []
    try:
        if isinstance(locked_phase_indices_val, str):
            locked_phase_indices = [int(x.strip()) for x in locked_phase_indices_val.split(',') if x.strip()]
        elif isinstance(locked_phase_indices_val, (list, tuple)):
            locked_phase_indices = [int(x) for x in locked_phase_indices_val]
        elif locked_phase_indices_val is not None:
            locked_phase_indices = [int(locked_phase_indices_val)]
    except Exception:
        locked_phase_indices = []

    lock_reference_genome = bool(_get_setting('lock_reference_genome', 'lock-reference-genome', False))
    custom_locked_genome = _get_setting('custom_locked_genome', 'custom-locked-genome')

    # Build genome comparison DataFrame
    genome_rows = []
    try:
        base_map = {c: p for c, p in zip(clusters_sorted, baseline_genome)}
        new_map = {c: p for c, p in zip(clusters_sorted, rerun_genome)}
        for c in clusters_sorted:
            if c in base_map and c in new_map:
                bp = int(base_map[c])
                rp = int(new_map[c])
                genome_rows.append({
                    'cluster': int(c),
                    'baseline_phase': bp,
                    'rerun_phase': rp,
                    'phase_delta': rp - bp,
                    'is_locked': bool(lock_committed and (bp in locked_phase_indices))
                })
    except Exception:
        pass
    genome_comparison_df = pd.DataFrame(genome_rows)
    genome_comp_csv = analysis_dir / 'genome_comparison.csv'
    try:
        if not genome_comparison_df.empty:
            genome_comparison_df.to_csv(genome_comp_csv, index=False)
            logger.info(f"Saved genome comparison to {genome_comp_csv}")
    except Exception as e:
        logger.warning(f"Failed to save genome comparison CSV: {e}")

    # Persist per-run, per-phase movement stats for cross-run phase-wise plots
    try:
        if not genome_comparison_df.empty and {'baseline_phase','phase_delta'}.issubset(genome_comparison_df.columns):
            mv = genome_comparison_df.copy()
            # Compute counts per baseline phase
            phase_stats = mv.groupby('baseline_phase').agg(
                moved_count=('phase_delta', lambda s: int((s != 0).sum())),
                earlier_count=('phase_delta', lambda s: int((s < 0).sum())),
                later_count=('phase_delta', lambda s: int((s > 0).sum())),
                same_count=('phase_delta', lambda s: int((s == 0).sum())),
                avg_abs_delta=('phase_delta', lambda s: float(np.nanmean(np.abs(s))))
            ).reset_index()
            # Build rows with run metadata
            run_id = time.strftime('%Y-%m-%dT%H:%M:%S')
            rows = []
            for _, r in phase_stats.iterrows():
                rows.append({
                    'run_id': run_id,
                    'run_label': run_label,
                    'network_type': network_type,
                    'objective_function': settings.get('objective_function', 'NPV'),
                    'capex_per_phase_relaxing_pct': capex_relax_pct,
                    'total_exp_per_phase_relaxing_pct': totalexp_relax_pct,
                    'lock_committed_early_phases': int(bool(lock_committed)),
                    'locked_phases_count': len(locked_phase_indices) if locked_phase_indices else 0,
                    'baseline_phase': int(r['baseline_phase']) if pd.notna(r['baseline_phase']) else np.nan,
                    'moved_count': int(r['moved_count']) if pd.notna(r['moved_count']) else np.nan,
                    'earlier_count': int(r['earlier_count']) if pd.notna(r['earlier_count']) else np.nan,
                    'later_count': int(r['later_count']) if pd.notna(r['later_count']) else np.nan,
                    'same_count': int(r['same_count']) if pd.notna(r['same_count']) else np.nan,
                    'avg_abs_delta': float(r['avg_abs_delta']) if pd.notna(r['avg_abs_delta']) else np.nan,
                })
            mv_df = pd.DataFrame(rows)
            mv_csv = analysis_dir / 'genome_phase_movements.csv'
            if include_in_part3:
                if mv_csv.exists():
                    # aligned append
                    try:
                        old = pd.read_csv(mv_csv)
                    except Exception:
                        old = pd.DataFrame()
                    all_cols = sorted(set(old.columns.tolist()) | set(mv_df.columns.tolist()))
                    old = old.reindex(columns=all_cols)
                    mv_df = mv_df.reindex(columns=all_cols)
                    pd.concat([old, mv_df], ignore_index=True).to_csv(mv_csv, index=False)
                else:
                    mv_df.to_csv(mv_csv, index=False)
                logger.info(f"Saved per-phase movement stats to {mv_csv}")
            else:
                logger.info("include_in_part3 is False; skipping cross-run movement stats append.")
    except Exception as e:
        logger.warning(f"Failed to persist per-phase movement stats: {e}")

    # Aggregate master summary
    master_row = {
        'network_type': network_type,
        'objective_function': settings.get('objective_function', 'NPV'),
        'num_phases': settings.get('num_phases'),
        'phase_durations': settings.get('phase_durations'),
        'testing_clusters': ','.join(str(c) for c in testing_clusters) if testing_clusters else '',
        'run_label': run_label,
        'include_in_part3': bool(include_in_part3),
        'baseline_genome': json.dumps(baseline_genome),
        'rerun_genome': json.dumps(rerun_genome),
        # New constraint parameters
        'capex_per_phase_relaxing_pct': capex_relax_pct,
        'total_exp_per_phase_relaxing_pct': totalexp_relax_pct,
        'enforce_final_cumulative_capex': enforce_final_capex,
        'final_cumulative_capex_budget_total': final_capex_budget_total,
        'enforce_final_cumulative_total_exp': enforce_final_totalexp,
        'final_cumulative_total_exp_budget_total': final_totalexp_budget_total,
        'lock_committed_early_phases': lock_committed,
        'locked_phase_indices': ','.join(str(x) for x in locked_phase_indices) if locked_phase_indices else '',
        'lock_reference_genome': lock_reference_genome,
        'custom_locked_genome': json.dumps(custom_locked_genome) if custom_locked_genome is not None else ''
    }
    master_row.update(district_deltas)
    # Demand modification proxies (portfolio-level)
    try:
        def _pct100(x):
            try:
                return float(x) * 100.0 if pd.notna(x) else x
            except Exception:
                return x
        master_row['mean_reduction_heating_pct'] = _pct100(district_deltas.get('delta_heating_pct'))
        master_row['mean_reduction_cooling_pct'] = _pct100(district_deltas.get('delta_cooling_pct'))
        master_row['mean_reduction_dhw_pct'] = _pct100(district_deltas.get('delta_dhw_pct'))
        master_row['mean_reduction_electricity_pct'] = _pct100(district_deltas.get('delta_electricity_pct'))
        master_row['densification_pct'] = _pct100(district_deltas.get('delta_gfa_pct'))
    except Exception:
        pass
    master_row.update(genome_deltas)
    # Derived shares from genome deltas
    try:
        _total = float(master_row.get('clusters_compared')) if master_row.get('clusters_compared') not in (None, np.nan) else np.nan
        _earlier = float(master_row.get('earlier')) if master_row.get('earlier') not in (None, np.nan) else np.nan
        _later = float(master_row.get('later')) if master_row.get('later') not in (None, np.nan) else np.nan
        _same = float(master_row.get('same')) if master_row.get('same') not in (None, np.nan) else np.nan
        if pd.notna(_total) and _total > 0 and all(pd.notna(x) for x in [_earlier, _later, _same]):
            master_row['earlier_share'] = _earlier / _total
            master_row['later_share'] = _later / _total
            master_row['same_share'] = _same / _total
    except Exception:
        pass
    # Locking KPIs
    try:
        master_row['locked_phases_count'] = len(locked_phase_indices) if locked_phase_indices else 0
        if isinstance(genome_comparison_df, pd.DataFrame) and not genome_comparison_df.empty and 'is_locked' in genome_comparison_df.columns:
            locked_df = genome_comparison_df[genome_comparison_df['is_locked'] == True]
            if not locked_df.empty and 'phase_delta' in locked_df.columns:
                prot = float((locked_df['phase_delta'] == 0).sum()) / float(len(locked_df)) if len(locked_df) > 0 else np.nan
                master_row['protected_share'] = prot
    except Exception:
        pass
    master_row.update({f"baseline_{k}": v for k, v in baseline_metrics.items()})
    master_row.update({f"rerun_{k}": v for k, v in rerun_metrics.items()})
    master_row.update(emissions_summary)

    # Save outputs (robust aligned writer to avoid schema drift)
    master_summary_csv = analysis_dir / 'master_summary.csv'
    if include_in_part3:
        try:
            new_df = pd.DataFrame([master_row])
            if master_summary_csv.exists():
                try:
                    old_df = pd.read_csv(master_summary_csv)
                except Exception:
                    old_df = pd.DataFrame()
                all_cols = sorted(set(old_df.columns.tolist()) | set(new_df.columns.tolist()))
                old_df = old_df.reindex(columns=all_cols)
                new_df = new_df.reindex(columns=all_cols)
                combined = pd.concat([old_df, new_df], ignore_index=True)
                combined.to_csv(master_summary_csv, index=False)
            else:
                new_df.to_csv(master_summary_csv, index=False)
            logger.info(f"Saved master summary to {master_summary_csv}")
        except Exception as e:
            logger.warning(f"Failed aligned write of master_summary.csv: {e}. Falling back to append mode.")
            if master_summary_csv.exists():
                pd.DataFrame([master_row]).to_csv(master_summary_csv, mode='a', header=False, index=False)
            else:
                pd.DataFrame([master_row]).to_csv(master_summary_csv, index=False)
    else:
        logger.info("include_in_part3 is False; skipping append to master_summary.csv.")

    # Save cluster-level deltas (scale *_pct as percent for user-facing CSV)
    cluster_deltas_csv = analysis_dir / 'cluster_level_deltas.csv'
    try:
        cluster_save = cluster_deltas_df.copy()
        pct_cols = [c for c in cluster_save.columns if c.endswith('_pct')]
        for c in pct_cols:
            try:
                cluster_save[c] = pd.to_numeric(cluster_save[c], errors='coerce') * 100.0
            except Exception:
                pass
        cluster_save.to_csv(cluster_deltas_csv, index=False)
    except Exception:
        # fallback to original if scaling failed for any unexpected reason
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

    # Generate figures (non-blocking, best-effort)
    try:
        _generate_figures(
            analysis_dir=analysis_dir,
            district_deltas=district_deltas,
            cluster_deltas_df=cluster_deltas_df,
            baseline_genome=baseline_genome,
            rerun_genome=rerun_genome,
            clusters_sorted=clusters_sorted,
            genome_deltas=genome_deltas,
            genome_comparison_df=genome_comparison_df,
            lock_committed_early_phases=lock_committed,
            locked_phase_indices=locked_phase_indices,
            emissions_summary=emissions_summary
        )
    except Exception as e:
        logger.warning(f"Plot generation failed: {e}")

    # Cross-run insight plots (non-blocking)
    try:
        _generate_cross_run_insight_plots(analysis_dir)
    except Exception as e:
        logger.warning(f"Cross-run insight plots skipped: {e}")

    logger.info("Sensitivity analysis (core) completed.")
    return None


def _generate_cross_run_insight_plots(analysis_dir: Path) -> None:
    """Create cross-run plots that relate genome changes to policy levers and demand changes.
    Reads analysis/master_summary.csv and outputs PNGs to analysis/figures.
    Safe no-op if file or required fields are missing.
    """
    try:
        try:
            plt.switch_backend('Agg')
        except Exception:
            pass
        ms_path = analysis_dir / 'master_summary.csv'
        if not ms_path.exists():
            return
        df = pd.read_csv(ms_path)
        figs = analysis_dir / 'figures'
        try:
            figs.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        def col(name: str):
            if name in df.columns:
                return pd.to_numeric(df[name], errors='coerce') if df[name].dtype == object else df[name]
            return None

        def line_or_scatter(xs: np.ndarray, ys: np.ndarray, xlabel: str, ylabel: str, title: str, out: Path):
            if len(xs) < 2:
                return
            order = np.argsort(xs)
            xs_sorted = np.array(xs)[order]
            ys_sorted = np.array(ys)[order]
            plt.figure(figsize=(6, 4))
            # If there are repeated x values, aggregate by mean for line; also plot scatter
            try:
                x_unique = np.unique(xs_sorted)
                y_mean = [np.nanmean(ys_sorted[xs_sorted == xv]) for xv in x_unique]
                plt.plot(x_unique, y_mean, marker='o', linewidth=1.5, label='mean by x')
                plt.scatter(xs_sorted, ys_sorted, s=16, alpha=0.6, label='runs')
            except Exception:
                plt.plot(xs_sorted, ys_sorted, marker='o', linewidth=1.5)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(title)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.legend(loc='best')
            plt.savefig(out, dpi=200)
            try:
                log().info(f"Saved figure: {out}")
            except Exception:
                pass
            plt.close()

        def scatter_with_fit(x: np.ndarray, y: np.ndarray, xlabel: str, ylabel: str, title: str, out: Path):
            if len(x) < 2:
                return
            plt.figure(figsize=(6, 4))
            plt.scatter(x, y, s=20, alpha=0.7)
            # simple linear fit
            try:
                coeffs = np.polyfit(x, y, 1)
                x_line = np.linspace(np.nanmin(x), np.nanmax(x), 50)
                y_line = coeffs[0] * x_line + coeffs[1]
                plt.plot(x_line, y_line, color='orange', linewidth=1.5, label=f'fit: y={coeffs[0]:.2f}x+{coeffs[1]:.2f}')
                plt.legend(loc='best')
            except Exception:
                pass
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(title)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(out, dpi=200)
            try:
                log().info(f"Saved figure: {out}")
            except Exception:
                pass
            plt.close()

        hamming_norm = col('hamming_norm')
        directional_shift = col('directional_shift')
        capex_relax = col('capex_per_phase_relaxing_pct')
        totalexp_relax = col('total_exp_per_phase_relaxing_pct')
        lock_flag = df['lock_committed_early_phases'] if 'lock_committed_early_phases' in df.columns else None

        # Hamming vs CAPEX relaxation
        if hamming_norm is not None and capex_relax is not None:
            d = pd.DataFrame({'x': capex_relax, 'y': hamming_norm}).dropna()
            if len(d) >= 2:
                line_or_scatter(d['x'].values, d['y'].values,
                                'CAPEX per-phase relaxing pct', 'Hamming (normalized)',
                                f'Hamming vs CAPEX relaxation (n={len(d)})',
                                figs / 'hamming_vs_capex_relaxation.png')

        # Hamming vs Total Expenditure relaxation
        if hamming_norm is not None and totalexp_relax is not None:
            d = pd.DataFrame({'x': totalexp_relax, 'y': hamming_norm}).dropna()
            if len(d) >= 2:
                line_or_scatter(d['x'].values, d['y'].values,
                                'Total-exp per-phase relaxing pct', 'Hamming (normalized)',
                                f'Hamming vs Total-exp relaxation (n={len(d)})',
                                figs / 'hamming_vs_totalexp_relaxation.png')

        # Bias (directional shift) vs relaxations
        if directional_shift is not None and capex_relax is not None:
            d = pd.DataFrame({'x': capex_relax, 'y': directional_shift}).dropna()
            if len(d) >= 2:
                line_or_scatter(d['x'].values, d['y'].values,
                                'CAPEX per-phase relaxing pct', 'Directional shift (ΣΔphase)',
                                f'Bias vs CAPEX relaxation (n={len(d)})',
                                figs / 'bias_vs_capex_relaxation.png')
        if directional_shift is not None and totalexp_relax is not None:
            d = pd.DataFrame({'x': totalexp_relax, 'y': directional_shift}).dropna()
            if len(d) >= 2:
                line_or_scatter(d['x'].values, d['y'].values,
                                'Total-exp per-phase relaxing pct', 'Directional shift (ΣΔphase)',
                                f'Bias vs Total-exp relaxation (n={len(d)})',
                                figs / 'bias_vs_totalexp_relaxation.png')

        # Hamming by lock flag (bar of means with counts)
        if hamming_norm is not None and lock_flag is not None:
            try:
                d = pd.DataFrame({'lock': lock_flag.astype(int), 'h': pd.to_numeric(hamming_norm, errors='coerce')}).dropna()
                if len(d) >= 2 and d['lock'].nunique() >= 1:
                    stats = d.groupby('lock')['h'].agg(['mean', 'count']).reset_index()
                    plt.figure(figsize=(5, 4))
                    plt.bar(stats['lock'].astype(str), stats['mean'], width=0.5)
                    for i, row in stats.iterrows():
                        plt.text(i, row['mean'], f"n={int(row['count'])}", ha='center', va='bottom', fontsize=8)
                    plt.xlabel('Lock committed early phases (0/1)')
                    plt.ylabel('Hamming (normalized)')
                    plt.title('Hamming by locking policy')
                    plt.tight_layout()
                    out_path = figs / 'hamming_by_lock_flag.png'
                    plt.savefig(out_path, dpi=200)
                    try:
                        log().info(f"Saved figure: {out_path}")
                    except Exception:
                        pass
                    plt.close()
            except Exception:
                pass

        # Hamming vs end-use reductions and densification
        for key, label in [
            ('mean_reduction_heating_pct', 'Heating reduction (pct)'),
            ('mean_reduction_cooling_pct', 'Cooling reduction (pct)'),
            ('mean_reduction_dhw_pct', 'DHW reduction (pct)'),
            ('mean_reduction_electricity_pct', 'Electricity reduction (pct)'),
            ('densification_pct', 'Densification (pct)')
        ]:
            xcol = col(key)
            if xcol is None or hamming_norm is None:
                continue
            d = pd.DataFrame({'x': xcol, 'y': hamming_norm}).dropna()
            if len(d) >= 2:
                scatter_with_fit(d['x'].values, d['y'].values,
                                 label, 'Hamming (normalized)',
                                 f'Hamming vs {label} (n={len(d)})',
                                 figs / f"hamming_vs_{key}.png")

        # Earlier / Later shares vs relaxations
        earlier_share = col('earlier_share')
        later_share = col('later_share')
        if earlier_share is not None and capex_relax is not None:
            d = pd.DataFrame({'x': capex_relax, 'y': earlier_share}).dropna()
            if len(d) >= 2:
                line_or_scatter(d['x'].values, d['y'].values,
                                'CAPEX per-phase relaxing pct', 'Earlier share',
                                f'Earlier share vs CAPEX relaxation (n={len(d)})',
                                figs / 'earlier_share_vs_capex_relaxation.png')
        if later_share is not None and capex_relax is not None:
            d = pd.DataFrame({'x': capex_relax, 'y': later_share}).dropna()
            if len(d) >= 2:
                line_or_scatter(d['x'].values, d['y'].values,
                                'CAPEX per-phase relaxing pct', 'Later share',
                                f'Later share vs CAPEX relaxation (n={len(d)})',
                                figs / 'later_share_vs_capex_relaxation.png')
        if earlier_share is not None and totalexp_relax is not None:
            d = pd.DataFrame({'x': totalexp_relax, 'y': earlier_share}).dropna()
            if len(d) >= 2:
                line_or_scatter(d['x'].values, d['y'].values,
                                'Total-exp per-phase relaxing pct', 'Earlier share',
                                f'Earlier share vs Total-exp relaxation (n={len(d)})',
                                figs / 'earlier_share_vs_totalexp_relaxation.png')
        if later_share is not None and totalexp_relax is not None:
            d = pd.DataFrame({'x': totalexp_relax, 'y': later_share}).dropna()
            if len(d) >= 2:
                line_or_scatter(d['x'].values, d['y'].values,
                                'Total-exp per-phase relaxing pct', 'Later share',
                                f'Later share vs Total-exp relaxation (n={len(d)})',
                                figs / 'later_share_vs_totalexp_relaxation.png')

        # Earlier / Later shares vs demand-change proxies
        for key, label in [
            ('mean_reduction_heating_pct', 'Heating reduction (pct)'),
            ('mean_reduction_cooling_pct', 'Cooling reduction (pct)'),
            ('mean_reduction_dhw_pct', 'DHW reduction (pct)'),
            ('mean_reduction_electricity_pct', 'Electricity reduction (pct)'),
            ('densification_pct', 'Densification (pct)')
        ]:
            xcol = col(key)
            if xcol is None:
                continue
            if earlier_share is not None:
                d = pd.DataFrame({'x': xcol, 'y': earlier_share}).dropna()
                if len(d) >= 2:
                    scatter_with_fit(d['x'].values, d['y'].values,
                                     label, 'Earlier share',
                                     f'Earlier share vs {label} (n={len(d)})',
                                     figs / f"earlier_share_vs_{key}.png")
            if later_share is not None:
                d = pd.DataFrame({'x': xcol, 'y': later_share}).dropna()
                if len(d) >= 2:
                    scatter_with_fit(d['x'].values, d['y'].values,
                                     label, 'Later share',
                                     f'Later share vs {label} (n={len(d)})',
                                     figs / f"later_share_vs_{key}.png")

        # Protected share by number of locked phases (bars)
        if 'protected_share' in df.columns and 'locked_phases_count' in df.columns:
            d = df[['protected_share','locked_phases_count']].copy()
            d['protected_share'] = pd.to_numeric(d['protected_share'], errors='coerce')
            d['locked_phases_count'] = pd.to_numeric(d['locked_phases_count'], errors='coerce')
            d = d.dropna()
            if len(d) >= 2 and d['locked_phases_count'].nunique() >= 1:
                stats = d.groupby('locked_phases_count')['protected_share'].agg(['mean','count']).reset_index()
                plt.figure(figsize=(6,4))
                plt.bar(stats['locked_phases_count'].astype(int).astype(str), stats['mean'], width=0.5)
                for i, row in stats.iterrows():
                    plt.text(i, row['mean'], f"n={int(row['count'])}", ha='center', va='bottom', fontsize=8)
                plt.xlabel('Locked phases count')
                plt.ylabel('Protected share (locked unchanged)')
                plt.title('Protected share by locked phases count')
                plt.tight_layout()
                plt.savefig(figs / 'protected_share_by_locked_phases_count.png', dpi=200)
                plt.close()

        # Phase-wise relaxation plots from genome_phase_movements.csv
        mv_csv = analysis_dir / 'genome_phase_movements.csv'
        if mv_csv.exists():
            try:
                mv = pd.read_csv(mv_csv)
                # Ensure numeric types
                for c in ['capex_per_phase_relaxing_pct', 'total_exp_per_phase_relaxing_pct',
                          'baseline_phase', 'moved_count', 'avg_abs_delta', 'lock_committed_early_phases']:
                    if c in mv.columns:
                        mv[c] = pd.to_numeric(mv[c], errors='coerce')
                # Helper to draw lines per baseline phase for a given x and y and lock flag subset
                def plot_phase_lines(df_in: pd.DataFrame, xcol: str, ycol: str, lock_flag_val: int, out_name: str, xlabel: str, ylabel: str, title_prefix: str):
                    d = df_in.copy()
                    d = d[(~d[xcol].isna()) & (~d[ycol].isna())]
                    if 'lock_committed_early_phases' in d.columns:
                        d = d[d['lock_committed_early_phases'] == lock_flag_val]
                    if d.empty:
                        return
                    # Need at least two distinct x values
                    if d[xcol].nunique() < 2:
                        return
                    plt.figure(figsize=(7, 4.5))
                    phases = sorted([int(p) for p in d['baseline_phase'].dropna().unique()]) if 'baseline_phase' in d.columns else []
                    for ph in phases:
                        dp = d[d['baseline_phase'] == ph]
                        if dp.empty:
                            continue
                        # aggregate by x (mean y)
                        grp = dp.groupby(xcol)[ycol].mean().reset_index().sort_values(xcol)
                        plt.plot(grp[xcol].values, grp[ycol].values, marker='o', linewidth=1.5, label=f'Phase {ph}')
                    # tipping point: first x with any movement (>0) across phases when y is moved_count
                    try:
                        if ycol == 'moved_count':
                            agg = d.groupby(xcol)['moved_count'].sum().reset_index().sort_values(xcol)
                            tip = agg.loc[agg['moved_count'] > 0, xcol].min()
                            if pd.notna(tip):
                                plt.axvline(tip, color='red', linestyle='--', linewidth=1.0, alpha=0.7)
                                plt.text(tip, plt.gca().get_ylim()[1]*0.95, f'first change @ {tip:g}%', color='red', ha='right', va='top', fontsize=8)
                    except Exception:
                        pass
                    plt.xlabel(xlabel)
                    plt.ylabel(ylabel)
                    plt.title(f"{title_prefix} (lock={'ON' if lock_flag_val==1 else 'OFF'})")
                    plt.legend(ncol=2, fontsize=8)
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    out_path = figs / out_name
                    plt.savefig(out_path, dpi=200)
                    try:
                        log().info(f"Saved figure: {out_path}")
                    except Exception:
                        pass
                    plt.close()

                # Create the four requested families of plots
                if {'baseline_phase','moved_count','avg_abs_delta'}.issubset(mv.columns):
                    for lock_val in (0, 1):
                        # CAPEX relaxation based
                        if 'capex_per_phase_relaxing_pct' in mv.columns:
                            plot_phase_lines(mv, 'capex_per_phase_relaxing_pct', 'moved_count', lock_val,
                                             f"phase_moved_vs_capex_relaxation_lock{lock_val}.png",
                                             'CAPEX per-phase relaxing pct', 'Moved count', 'Phase movement vs CAPEX relaxation')
                            plot_phase_lines(mv, 'capex_per_phase_relaxing_pct', 'avg_abs_delta', lock_val,
                                             f"phase_avg_abs_delta_vs_capex_relaxation_lock{lock_val}.png",
                                             'CAPEX per-phase relaxing pct', 'Avg |Δphase|', 'Avg |Δ| vs CAPEX relaxation')
                        # Total expenditure relaxation based
                        if 'total_exp_per_phase_relaxing_pct' in mv.columns:
                            plot_phase_lines(mv, 'total_exp_per_phase_relaxing_pct', 'moved_count', lock_val,
                                             f"phase_moved_vs_totalexp_relaxation_lock{lock_val}.png",
                                             'Total-exp per-phase relaxing pct', 'Moved count', 'Phase movement vs Total-exp relaxation')
                            plot_phase_lines(mv, 'total_exp_per_phase_relaxing_pct', 'avg_abs_delta', lock_val,
                                             f"phase_avg_abs_delta_vs_totalexp_relaxation_lock{lock_val}.png",
                                             'Total-exp per-phase relaxing pct', 'Avg |Δphase|', 'Avg |Δ| vs Total-exp relaxation')
            except Exception as e:
                try:
                    log().warning(f"Phase-wise relaxation plots skipped: {e}")
                except Exception:
                    pass
    except Exception:
        # Never fail main flow because of cross-run plots
        return


if __name__ == '__main__':
    args = parse_args()
    config = cea.config.Configuration(args.config)
    config.scenario = args.scenario
    main(config)
