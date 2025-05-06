#!/usr/bin/env python3
"""
District‑Network‑Expansion

Runs CEA network‑layout (Part 1) and thermal‑network (Part 2) on *all*
buildings that are within SNAP_TOLERANCE of a street centre‑line.

Outputs land in:  outputs/data/optimization/
Diagnostics CSV : outputs/data/optimization/diagnostics/building_street_distances.csv
"""

from __future__ import annotations
import os, types
import pandas as pd
import geopandas as gpd
import cea.config
from cea.inputlocator import InputLocator
import cea.constants as consts
from cea.technologies.network_layout.main import layout_network, NetworkLayout
from cea.technologies.thermal_network.thermal_network import (
    ThermalNetwork, thermal_network_main)

SHAPEFILE_TOLERANCE_MM = consts.SHAPEFILE_TOLERANCE
SNAP_TOLERANCE_M = consts.SNAP_TOLERANCE

print("Using SHAPEFILE_TOLERANCE =", consts.SHAPEFILE_TOLERANCE,  "mm")
print("Using SNAP_TOLERANCE      =", consts.SNAP_TOLERANCE,  "m")

# -------------------------------------------------------------------------
def nearest_distance(points: gpd.GeoSeries,
                     lines : gpd.GeoSeries) -> pd.Series:
    """Vectorised min distance (m) from each point to its nearest line."""
    idx = lines.sindex.nearest(points, return_all=False)[1]
    # idx may be shorter than points if some return []  → pad with None
    full_idx, it = [], iter(idx)
    for _ in points:
        full_idx.append(next(it, None))
    nearest = [lines.iloc[i] if i is not None else None for i in full_idx]
    return pd.Series([
        p.distance(l) if l is not None else float("inf")
        for p, l in zip(points, nearest)
    ], index=points.index)

# -------------------------------------------------------------------------
def main(config):
    locator = InputLocator(config.scenario)

    # redirect thermal‑network outputs → optimization/
    locator.get_thermal_network_folder = locator.get_optimization_results_folder

    # load geometry
    zone   = gpd.read_file(locator.get_zone_geometry())
    streets = gpd.read_file(locator.get_street_network())

    # compute point‑to‑line distances
    zone["centroid"]  = zone.geometry.centroid
    zone["dist_m"]    = nearest_distance(zone["centroid"], streets.geometry)

    # diagnostics
    diag_dir = os.path.join(
        config.scenario, "outputs", "data", "optimization", "diagnostics")
    os.makedirs(diag_dir, exist_ok=True)
    diag_csv = os.path.join(diag_dir, "building_street_distances.csv")
    zone[["name", "dist_m"]].to_csv(diag_csv, index=False)
    print("Diagnostics written ->", diag_csv)

    # filter buildings that can be snapped
    keep   = zone["dist_m"] <= SNAP_TOLERANCE_M
    kept_names = zone.loc[keep, "name"].astype(str).tolist()
    dropped    = (~keep).sum()
    if dropped:
        print(f"{dropped} buildings farther than {SNAP_TOLERANCE_M} m "
              "were excluded from the layout.")

    # ------------- network‑layout configuration -----------------------
    nl_cfg = config.network_layout
    nl_cfg.connected_buildings = kept_names
    nl_cfg.consider_only_buildings_with_demand = False

    if not hasattr(config, "thermal_network"):
        config.thermal_network = types.SimpleNamespace(network_type="DH")

    # ------------------ Part 1: layout --------------------------------
    print("Running network‑layout …")
    netlay = NetworkLayout(nl_cfg)
    netlay.network_type = config.thermal_network.network_type
    layout_network(netlay, locator, output_name_network="")

    # ------------------ Part 2: hydraulics / thermal ------------------
    print("Running thermal‑network …")
    tn = ThermalNetwork(locator, network_name="", config=config.thermal_network)
    thermal_network_main(locator, tn)

    print("Finished – outputs in", locator.get_optimization_results_folder())

# -------------------------------------------------------------------------
if __name__ == "__main__":
    main(cea.config.Configuration())
