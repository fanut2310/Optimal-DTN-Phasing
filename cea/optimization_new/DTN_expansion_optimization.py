#!/usr/bin/env python3
"""
DTN_expansion_optimization.py
=============================
A workflow to stage the expansion of an existing (cluster‑0) District Thermal
Network (DTN) into an arbitrary number of *phases* while maximising a financial
objective (ROI, NPV or simple pay‑back).

Key design philosophy
---------------------
* **Single network layout** – we reuse the *Part 1* master layout that connects
  **all** candidate buildings.  This avoids having to regenerate geometry for
  each phase and ensures pipe positions & diameters stay consistent.
* **Edge activation** – every pipe (edge) in that master graph carries a
  Boolean `active` flag.  When a cluster is assigned to a phase we activate the
  path that links its nodes to the already‑active network.  CAPEX is charged
  *only once* per edge – the phase that first activates it.
* **Modular optimization layer** – the cost / hydraulic model is kept strictly
  separate from the search algorithm.  You may plug‑in Greedy, A*, Genetic
  Algorithm, etc. simply by switching the `choose_next_phase()` strategy.
* **CEA integration ready** – inputs are read via `cea.inputlocator.InputLocator`
  and parameters via `cea.config.Configuration`.  The script therefore runs
  seamlessly from the CEA CLI or Dashboard once registered in `scripts.yml`.
"""

import argparse
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import pandas as pd

import cea.config               # type: ignore
import cea.inputlocator         # type: ignore
from cea.analysis.costs.equations import (
    calc_capex_annualized,
    calc_opex_annualized,
)

# ----------------------------------------------------------------------------
# Default economic assumptions (if missing in config)
# ----------------------------------------------------------------------------
DEFAULT_IR_PERC          = 5.0    # [%]
DEFAULT_LIFETIME_YR      = 25     # [yr]
DEFAULT_FUEL_COST_USDkWh = 0.04   # [USD/kWh]
DEFAULT_OANDM_PERC       = 1.0    # [% of CAPEX]

# ----------------------------------------------------------------------------
# Logging setup
# ----------------------------------------------------------------------------

def log() -> logging.Logger:
    logger = logging.getLogger("cea.dtn_expansion")
    if not logger.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)5s | %(message)s",
            datefmt="%H:%M:%S",
        )
    return logger

# ----------------------------------------------------------------------------
# Safe config fetch
# ----------------------------------------------------------------------------

def safe_cfg(section, attr: str, default):
    try:
        return getattr(section, attr)
    except Exception:
        log().warning("Missing config.%s.%s → default %s", section, attr, default)
        return default

# ----------------------------------------------------------------------------
# Data classes
# ----------------------------------------------------------------------------
@dataclass
class Cluster:
    cid: int
    buildings: List[str]
    peak_kW: float
    annual_MWh: float
    centroid: Tuple[float, float]
    phase: Optional[int] = None
    capex_USD: float = 0.0
    opex_a_USD: float = 0.0
    npv_USD: float = 0.0
    path_nodes: List[str] = field(default_factory=list)

@dataclass
class Econ:
    IR: float
    LT: int
    fuel: float
    fixed_pct: float

@dataclass
class PipeCat:
    df: Optional[pd.DataFrame]
    default_cost: float = 450.0

    @classmethod
    def load(cls, path: Path):
        if path.exists():
            return cls(pd.read_csv(path))
        log().warning("Thermal grid DB not found: %s", path)
        return cls(None)

    def cost_per_m(self, dn_mm: float) -> float:
        if self.df is None:
            return self.default_cost
        df = self.df[df["DN_int_m"] * 1000 >= dn_mm]
        if df.empty:
            row = self.df.iloc[-1]
        else:
            row = df.iloc[0]
        return float(row.get("Pipe_cost_USD2015perm", self.default_cost))

# ----------------------------------------------------------------------------
# Core Optimiser
# ----------------------------------------------------------------------------
class DTNOptimizer:
    def __init__(self, scenario: str, net_type: str, net_name: str):
        cfg = cea.config.Configuration()
        cfg.scenario = scenario
        self.loc = cea.inputlocator.InputLocator(cfg.scenario)
        # economics
        costs = getattr(cfg, 'costs', cfg)
        self.econ = Econ(
            IR        = safe_cfg(costs, 'interest_rate', DEFAULT_IR_PERC),
            LT        = safe_cfg(costs, 'lifetime', DEFAULT_LIFETIME_YR),
            fuel      = safe_cfg(costs, 'fuel_cost_usdperkwh', DEFAULT_FUEL_COST_USDkWh),
            fixed_pct = safe_cfg(costs, 'maintenance_percent', DEFAULT_OANDM_PERC)
        )
        # load graph with fallback for edges.shp
        try:
            edges_shp = self.loc.get_network_layout_edges_shapefile(net_type, net_name)
            if not Path(edges_shp).exists():
                raise FileNotFoundError
        except Exception:
            scenario_root = Path(self.loc.scenario)
            edges_shp = scenario_root / "outputs" / "data" / "thermal-network" / net_type / "edges.shp"
            log().warning("Using fallback edges shapefile path: %s", edges_shp)
        g0 = nx.read_shp(str(edges_shp), simplify=True)
        self.graph = nx.Graph()
        for u, v, d in g0.edges(data=True):
            self.graph.add_edge(u, v,
                length_m=float(d.get('length_m', 0.0)),
                diameter_mm=float(d.get('D_int_m', 0.15)) * 1000,
                active=False)
        for n, d in g0.nodes(data=True):
            self.graph.add_node(n, **d)
        log().info("Loaded graph: %d nodes, %d edges", self.graph.number_of_nodes(), self.graph.number_of_edges())
        # pipe catalogue
        grid_csv = Path(self.loc.get_database_thermal_grid())
        self.pipe_cat = PipeCat.load(grid_csv)
        # clusters
        csvc = self.loc.get_building_cluster_assignment_file()
        df = pd.read_csv(csvc)
        self.clusters: Dict[int, Cluster] = {}
        for cid, grp in df.groupby('cluster'):
            self.clusters[cid] = Cluster(
                cid,
                grp['name'].tolist(),
                grp['peak_kW'].sum(),
                grp['annual_MWh'].sum(),
                (grp['x'].mean(), grp['y'].mean())
            )
        log().info("Loaded %d clusters", len(self.clusters))
        # plant node
        plants = [n for n, d in self.graph.nodes(data=True) if d.get('type', '').upper() == 'PLANT']
        if not plants:
            raise RuntimeError("No PLANT node found in network shapefile")
        self.plant = plants[0]
        self.active_edges: Set[Tuple[str, str]] = set()

    def _nearest(self, xy: Tuple[float, float]) -> str:
        return min(self.graph.nodes, key=lambda n: math.hypot(n[0] - xy[0], n[1] - xy[1]))

    def _compute_incremental(self, cluster: Cluster, phase_idx: int):
        start = self._nearest(cluster.centroid)
        path = nx.shortest_path(self.graph, self.plant, start, weight='length_m')
        edges = {tuple(sorted(e)) for e in zip(path[:-1], path[1:])}
        new = {e for e in edges if e not in self.active_edges}
        cap_p = sum(
            self.pipe_cat.cost_per_m(self.graph.edges[e]['diameter_mm']) * self.graph.edges[e]['length_m']
            for e in new
        )
        cap_h = 50 * cluster.peak_kW
        cap_tot = cap_p + cap_h
        cap_a = calc_capex_annualized(cap_tot, self.econ.IR, self.econ.LT)
        o_f = cap_tot * self.econ.fixed_pct / 100
        o_v = cluster.annual_MWh * 1000 * self.econ.fuel
        o_tot = o_f + o_v
        o_a = calc_opex_annualized(o_tot, self.econ.IR, self.econ.LT)
        disc = (1 + self.econ.IR / 100) ** -(phase_idx - 1)
        npv = -(cap_tot + o_a * self.econ.LT) * disc
        return new, cap_tot, cap_a, o_a, npv, path

    def greedy(self, max_phases: int):
        connected = {0}
        phase = 1
        while len(connected) < len(self.clusters) and phase <= max_phases:
            candidates = [cid for cid in self.clusters if cid not in connected]
            best_c, best_roi = None, -1e9
            for cid in candidates:
                new, cap, cap_a, op_a, npv, _ = self._compute_incremental(self.clusters[cid], phase)
                roi = op_a / cap_a if cap_a else 0
                if roi > best_roi:
                    best_roi, best_c = roi, cid
            self._activate(best_c, phase)
            connected.add(best_c)
            phase += 1

    def _activate(self, cid: int, phase: int):
        cl = self.clusters[cid]
        new, cap, cap_a, op_a, npv, path = self._compute_incremental(cl, phase)
        self.active_edges.update(new)
        cl.phase, cl.capex_USD, cl.opex_a_USD, cl.npv_USD, cl.path_nodes = (
            phase, cap, op_a, npv, path
        )
        for e in new:
            self.graph.edges[e]['active'] = True
        log().info("Phase %d: cluster %d | cap %0.f USD | ROI %0.1f%%",
                   phase, cid, cap, 100 * op_a / cap_a)

    def genetic(self):
        raise NotImplementedError("GA not implemented in rev‑4")

    def write(self):
        out = Path(self.loc.get_dtn_expansion_optimization_results_folder())
        out.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame([
            {
                'cluster': c.cid,
                'phase': c.phase,
                'Capex_USD': c.capex_USD,
                'Opex_a_USD': c.opex_a_USD,
                'NPV_USD': c.npv_USD
            }
            for c in self.clusters.values()
        ])
        df.to_csv(out / "cluster_phasing.csv", index=False)
        log().info("Wrote %s", out / "cluster_phasing.csv")

# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="DTN expansion phasing optimiser")
    p.add_argument("scenario", help="CEA scenario root folder")
    p.add_argument("--network-type", default="DH", choices=["DH", "DC"])
    p.add_argument("--network-name", default="DH")
    p.add_argument("--algorithm", default="greedy", choices=["greedy", "ga"],
                   help="optimisation method")
    p.add_argument("--max-phases", type=int, default=3, help="maximum phases")
    return p.parse_args()

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    args = parse_args()
    t0 = time.time()
    opt = DTNOptimizer(args.scenario, args.network_type, args.network_name)
    if args.algorithm == 'greedy':
        opt.greedy(args.max_phases)
    else:
        opt.genetic()
    opt.write()
    log().info("Done in %.1fs", time.time() - t0)

if __name__ == '__main__':
    main()
