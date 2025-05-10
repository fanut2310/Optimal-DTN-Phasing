#!/usr/bin/env python3
"""
DTN_expansion_optimization.py

  • --test : run only TEST_BUILDINGS (hard-coded below)
  • (no --test) : GUI/CLI mode, uses CEA’s connected_buildings or “Select all”
Results & diagnostics land under:
  ${SCENARIO}/outputs/data/optimization/dtn_expansion/
"""

import os
import argparse
from typing import List, Optional

# ┌───────────────────────────────┐
# │ 1) PATCH constants up front  │
# └───────────────────────────────┘
import cea.constants as constants
# > set these larger than your ~130 m max building–street gap
constants.SHAPEFILE_TOLERANCE = 1000  # mm
constants.SNAP_TOLERANCE      = 150   # m
print(f"Using SHAPEFILE_TOLERANCE = {constants.SHAPEFILE_TOLERANCE} mm")
print(f"Using SNAP_TOLERANCE      = {constants.SNAP_TOLERANCE} m")

# ┌───────────────────────────────────────────────────────────────────┐
# │ 2) defer all other CEA imports until after the patch above     │
# └───────────────────────────────────────────────────────────────────┘
import types
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString
from cea.config     import Configuration
from cea.inputlocator import InputLocator
from cea.technologies.network_layout.main import layout_network, NetworkLayout
import cea.technologies.network_layout.connectivity_potential as cp
from cea.technologies.thermal_network.thermal_network import thermal_network_main, ThermalNetwork

# ┌─────────────────────────────────────────┐
# │ 3) hard-coded test set of building IDs │
# └─────────────────────────────────────────┘
TEST_BUILDINGS = [
    'B0000','B0001','B0002','B0003','B0019','B0020','B0021','B0022',
    'B0034','B0035','B0036','B0041','B0053','B0054','B0055','B0056',
    'B0074','B0075','B0076'
]

def strip_z(zone_path: str) -> None:
    """Load zone.shp, drop any Z coords, overwrite it 2D."""
    gdf = gpd.read_file(zone_path)
    def dropz(g):
        if not getattr(g, "has_z", False):
            return g
        return type(g)([(x, y) for x, y, *rest in g.coords])
    gdf.geometry = gdf.geometry.map(dropz)
    gdf.to_file(zone_path, driver="ESRI Shapefile")

def run_dtn_expansion(config: Configuration, test_ids: Optional[List[str]]):
    locator = InputLocator(config.scenario)

    # ┌────────────────────────────────────────────────────────────────┐
    # │ 4) monkey-patch the broken near_analysis using this locator    │
    # └────────────────────────────────────────────────────────────────┘
    def near_analysis_fixed(blds: gpd.GeoDataFrame,
                            streets: gpd.GeoDataFrame,
                            crs) -> gpd.GeoDataFrame:

        # Diagnostics for streets data
        print(f"Street data: {len(streets)} rows, CRS: {streets.crs}")
        if not streets.empty:
            print(f"First street geometry type: {streets.iloc[0].geometry.type}")
        else:
            print("WARNING: Streets dataframe is empty!")

        streets_p = streets.to_crs(crs).geometry.reset_index(drop=True)
        blds_p = blds.to_crs(crs).reset_index(drop=True)

        # Check if streets dataframe is empty
        if streets.empty:
            print("WARNING: Streets dataframe is empty! Using building centroids directly.")
            return blds_p

        # Process each point individually
        names, snaps, dists = [], [], []
        for i, row in blds_p.iterrows():
            pt = row.geometry
            # Find nearest street segment for a single point
            nearest_results = streets_p.sindex.nearest(pt)

            # Check if any results were found
            if len(nearest_results[1]) > 0:
                nearest_idx = nearest_results[1][0]
                line = streets_p.iloc[nearest_idx]
                dist = pt.distance(line)
                snap = line.interpolate(line.project(pt))
            else:
                # Handle case where no streets are found near this building
                print(f"Warning: No streets found near building {row['name']}. Using original point position.")
                dist = float('inf')  # Set a large distance
                snap = pt

            names.append(row["name"])
            snaps.append(snap)
            dists.append(dist)

        # --- diagnostics
        df = pd.DataFrame({
            "name": names,
            "distance_to_street_m": dists
        }).sort_values("distance_to_street_m", ascending=False)

        diag_dir = os.path.join(
            locator.get_dtn_expansion_optimization_results_folder(), "diagnostics"
        )
        os.makedirs(diag_dir, exist_ok=True)
        df.to_csv(os.path.join(diag_dir, "building_street_distances.csv"), index=False)
        print("Diagnostics written →", diag_dir)

        # --- Important modification: If all distances are beyond tolerance, use a larger tolerance
        if all(dist > constants.SNAP_TOLERANCE for dist in dists):
            print(f"All buildings are beyond {constants.SNAP_TOLERANCE}m from streets.")
            print("Using building centroids directly instead of snapping to streets.")
            return blds_p[["name"]].set_geometry(blds_p.geometry)

        # Original filtering logic
        keep = df["distance_to_street_m"] <= constants.SNAP_TOLERANCE
        dropped = (~keep).sum()
        if dropped:
            print(f"Dropping {dropped} buildings beyond "
                  f"{constants.SNAP_TOLERANCE} m")
        kept = df[keep]["name"].tolist()
        kept_pts = [snap for snap, ok in zip(snaps, keep) if ok]

        return gpd.GeoDataFrame({"name": kept}, geometry=kept_pts, crs=crs)

    cp.near_analysis = near_analysis_fixed

    # ┌────────────────────────────────────────────────────┐
    # │ 5) strip any Z dimension out of zone.shp once     │
    # └────────────────────────────────────────────────────┘
    strip_z(locator.get_zone_geometry())

    # ┌─────────────────────────────────────────────────────────┐
    # │ 6) pick your building list: TEST or GUI/CLI (Select all) │
    # └─────────────────────────────────────────────────────────┘
    if test_ids:
        sel = test_ids
        print(f"[TEST MODE] running on {len(sel)} buildings")
    else:
        sel = config.network_layout.connected_buildings or []
        if not sel:
            sel = gpd.read_file(locator.get_zone_geometry())["name"].astype(str).tolist()

    # ┌──────────────────────────────────────────────────┐
    # │ 7) configure & run network-layout (Part 1)      │
    # └──────────────────────────────────────────────────┘
    cfg_nl = config.network_layout
    cfg_nl.connected_buildings = sel
    cfg_nl.consider_only_buildings_with_demand = False

    print("Running network-layout …")
    netlay = NetworkLayout(cfg_nl)
    netlay.network_type = getattr(config, "thermal_network", types.SimpleNamespace(network_type="DH")).network_type
    layout_network(netlay, locator, output_name_network="")

    # ┌──────────────────────────────────────────────────┐
    # │ 8) run thermal-network (Part 2)                │
    # └──────────────────────────────────────────────────┘
    print("Running thermal-network …")
    tn = ThermalNetwork(locator, network_name="", config=config.thermal_network)
    thermal_network_main(locator, tn)

    print("✅ Done. Results under",
          locator.get_dtn_expansion_optimization_results_folder())

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", required=True,
                   help="CEA scenario folder")
    p.add_argument("--test",   action="store_true",
                   help="Run only TEST_BUILDINGS list")
    return p.parse_args()

if __name__ == "__main__":
    args   = parse_args()
    cfg    = Configuration()
    cfg.scenario = args.scenario
    # ensure thermal_network exists
    if not hasattr(cfg, "thermal_network"):
        cfg.thermal_network = types.SimpleNamespace(network_type="DH")
    run_dtn_expansion(cfg, TEST_BUILDINGS if args.test else None)
