#!/usr/bin/env python3
"""
DTN_expansion_optimization.py

Usage:
  # PyCharm test mode:
  python DTN_expansion_optimization.py --test \
    --scenario "C:\…\01_base_design_2025"

  # Normal GUI/CLI mode:
  cea district-network-expansion --scenario "C:\…\01_base_design_2025"
"""

from __future__ import annotations
import argparse, os, sys, types
import pandas as pd
import geopandas as gpd
import shapely.ops

# -----------------------------------------------------------------------------
# 1) Patch constants *before* any network‐layout import
# -----------------------------------------------------------------------------
import cea.constants as consts
consts.SHAPEFILE_TOLERANCE = 1000   # mm
consts.SNAP_TOLERANCE      = 75     # m
print("PATCHED constants:",
      "SHAPEFILE_TOLERANCE=", consts.SHAPEFILE_TOLERANCE,
      "mm;", "SNAP_TOLERANCE=", consts.SNAP_TOLERANCE, "m")

# -----------------------------------------------------------------------------
# 2) Override the broken near_analysis in connectivity_potential
# -----------------------------------------------------------------------------
import cea.technologies.network_layout.connectivity_potential as cp

def near_analysis_fixed(blds: gpd.GeoDataFrame,
                        streets: gpd.GeoDataFrame,
                        crs) -> gpd.GeoDataFrame:
    """
    Safe replacement for cp.near_analysis.

    • ensures *exactly one* snapped point per building
    • keeps only those ≤ SNAP_TOLERANCE
    • never crashes when sindex.nearest returns too few indices
    """
    streets_proj = streets.to_crs(crs).geometry.reset_index(drop=True)
    blds_proj    = blds.to_crs(crs).reset_index(drop=True)

    cents = blds_proj.geometry
    idxs  = streets_proj.sindex.nearest(cents, return_all=False)[1]

    # Pad idxs to full length (None means “no street found”)
    it = iter(idxs)
    full_idxs = [next(it, None) for _ in range(len(cents))]

    snapped_pts, distances = [], []
    for pt, ix in zip(cents, full_idxs):
        if ix is None:            # no street at all
            distances.append(float("inf"))
            snapped_pts.append(None)
        else:
            line = streets_proj.iloc[ix]
            distances.append(pt.distance(line))
            snapped_pts.append(line.interpolate(line.project(pt)))

    # Diagnostics ----------------------------------------------------
    diag = pd.DataFrame({
        "name":   blds_proj["name"],
        "dist_m": distances
    }).sort_values("dist_m", ascending=False)

    diag_dir = os.path.join(
        "outputs", "data", "optimization", "dtn_expansion", "diagnostics")
    os.makedirs(diag_dir, exist_ok=True)
    diag.to_csv(os.path.join(diag_dir, "building_street_distances.csv"),
                index=False)
    print("Diagnostics written ->", diag_dir)

    # Keep only those within tolerance ------------------------------
    kept_names, kept_pts = [], []
    for name, d, pt in zip(blds_proj["name"], distances, snapped_pts):
        if d <= consts.SNAP_TOLERANCE and pt is not None:
            kept_names.append(name)
            kept_pts.append(pt)

    dropped = len(blds) - len(kept_names)
    if dropped:
        print(f"Dropping {dropped} buildings beyond {consts.SNAP_TOLERANCE} m")

    return gpd.GeoDataFrame({"name": kept_names},
                            geometry=kept_pts,
                            crs=crs)

cp.near_analysis = near_analysis_fixed

# -----------------------------------------------------------------------------
# 3) Now import the rest of CEA’s APIs
# -----------------------------------------------------------------------------
import cea.config
from cea.inputlocator import InputLocator
from cea.technologies.network_layout.main import layout_network, NetworkLayout
from cea.technologies.thermal_network.thermal_network import (
    ThermalNetwork, thermal_network_main
)

# -----------------------------------------------------------------------------
# 4) TEST_BUILDINGS (clusters 0–4)
# -----------------------------------------------------------------------------
TEST_BUILDINGS = [
    'B0000', 'B0001', 'B0002', 'B0003', 'B0019', 'B0020', 'B0021', 'B0022', 'B0034', 'B0035', 'B0036', 'B0041', 'B0053',
    'B0054', 'B0055', 'B0056', 'B0074', 'B0075', 'B0076', 'B0018', 'B0033', 'B0071', 'B101', 'B102', 'B103', 'B104',
    'B105', 'B106', 'B107', 'B108', 'B109', 'B110', 'B111', 'B112', 'B113', 'B114', 'B0023', 'B0047', 'B0062', 'B0077',
    'B0082', 'B0083', 'B0087', 'B115', 'B116', 'B117', 'B118', 'B119', 'B121', 'B122', 'B123', 'B124', 'B125', 'B193',
    'B194', 'B195', 'B196', 'B0012', 'B0037', 'B0038', 'B0045', 'B0057', 'B0058', 'B0060', 'B0061', 'B0078', 'B120',
    'B164', 'B0008', 'B0015', 'B0016', 'B0017', 'B0048', 'B0049', 'B0050', 'B0051', 'B0073', 'B126', 'B127', 'B128',
    'B129', 'B192'
]

# -----------------------------------------------------------------------------
def drop_z_dimension(path_zone: str):
    """Overwrite zone.shp as 2D by dropping any Z coordinates."""
    gdf = gpd.read_file(path_zone)
    def strip_z(geom):
        if geom.is_empty or not getattr(geom, "has_z", False):
            return geom
        return shapely.ops.transform(lambda x,y,z=None: (x,y), geom)
    gdf["geometry"] = gdf.geometry.apply(strip_z)
    gdf.to_file(path_zone, driver="ESRI Shapefile")

# -----------------------------------------------------------------------------
def run_dtn_expansion(cfg: cea.config.Configuration,
                      test_buildings: list[str] | None = None):
    locator = InputLocator(cfg.scenario)

    # strip Z once
    drop_z_dimension(locator.get_zone_geometry())

    # redirect all outputs under optimization/dtn_expansion/
    try:
        locator.get_thermal_network_folder = (
            locator.get_dtn_expansion_optimization_results_folder
        )
    except AttributeError:
        from cea.inputlocator import InputLocator as _IL
        locator.get_thermal_network_folder = (
            lambda: _IL.get_optimization_results_folder(locator)
        )
    out_base = locator.get_thermal_network_folder()

    # choose building list
    if test_buildings:
        sel = test_buildings
        print("TEST mode: running on", len(sel), "buildings")
    else:
        sel = cfg.network_layout.connected_buildings or []
        if not sel:
            # GUI Select All
            sel = list(gpd.read_file(locator.get_zone_geometry())["name"].astype(str))

    # configure network-layout
    nl_cfg = cfg.network_layout
    nl_cfg.connected_buildings                = sel
    nl_cfg.consider_only_buildings_with_demand = False
    if not hasattr(cfg, "thermal_network"):
        cfg.thermal_network = types.SimpleNamespace(network_type="DH")

    # Part 1: layout
    print("Running network-layout …")
    netlay = NetworkLayout(nl_cfg)
    netlay.network_type = cfg.thermal_network.network_type
    layout_network(netlay, locator, output_name_network="")

    # Part 2: thermal
    print("Running thermal-network …")
    tn = ThermalNetwork(locator, network_name="")
    thermal_network_main(locator, tn)

    print("Done. Results in", out_base)

# -----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", required=True)
    p.add_argument("--test", action="store_true")
    return p.parse_args()

# -----------------------------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()
    cfg  = cea.config.Configuration()
    cfg.scenario = args.scenario
    run_dtn_expansion(cfg, TEST_BUILDINGS if args.test else None)
