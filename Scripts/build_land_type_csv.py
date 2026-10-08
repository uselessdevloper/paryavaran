"""
Scripts/build_land_type_csv.py
==============================
Delhi NCT data pre-processing script.

Generates:
  ../points_df_025.csv        — 250 m Delhi grid (lat/lon)
  ../land_type_025.csv        — grid + all feature columns
  ../Data/cpcb_stations.csv   — CPCB AQ station averages (if not present)

Reads (must be downloaded first — see README for download links):
  ../Data/worldpop_india_2024_100m.tif   — WorldPop India 2024 population (clip to Delhi NCT)
  ../Data/worldcover_delhi.tif           — ESA WorldCover 2021 (10 m, tile N28E077)
  ../Data/modis_lst_summer_mean.tif      — MODIS MOD11A2 summer mean LST (K)
  ../Data/osm_delhi.gpkg                 — OSM Delhi GeoPackage
  ../Data/open_buildings_delhi.gpkg      — Google Open Buildings polygons (Delhi)

Usage:
  cd Paryavaran/
  python Scripts/build_land_type_csv.py

Runtime: ~10–25 minutes depending on machine (Delhi NCT is ~1500 km²).
"""

import os
import sys
import pathlib
import warnings

import numpy as np
import pandas as pd

# ── Path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR  = pathlib.Path(__file__).resolve().parent
ROOT        = SCRIPT_DIR.parent
DATA_DIR    = ROOT / "Data"
DATA_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(ROOT))
from helpers import (
    BBSR_BOTTOM_LEFT, BBSR_TOP_RIGHT, GRID_RESOLUTION_KM,
    get_spaced_point_set_in_bbox, ROOT_FOLDER_PATH,
)

POINTS_CSV     = ROOT / "points_df_025.csv"
LAND_TYPE_CSV  = ROOT / "land_type_025.csv"
CPCB_CSV       = DATA_DIR / "cpcb_stations.csv"

# WorldCover class values (ESA 2021 v200)
WC = {
    "tree":       10,
    "shrub":      20,
    "grass":      30,
    "cropland":   40,
    "builtup":    50,
    "bareland":   60,
    "snow":       70,
    "water":      80,
    "wetland":    90,
    "mangrove":   95,
    "moss":       100,
}

# ─── helpers ──────────────────────────────────────────────────────────────────
def _raster_zonal_stats(raster_path, points_df, stat="mean", nodata=None):
    """Extract per-cell mean/sum from a GeoTIFF using rasterstats."""
    try:
        from rasterstats import point_query
        import geopandas as gpd
        from shapely.geometry import Point

        gdf = gpd.GeoDataFrame(
            points_df,
            geometry=[Point(lon, lat) for lat, lon in
                      zip(points_df["Latitude"], points_df["Longitude"])],
            crs="EPSG:4326",
        )
        results = point_query(gdf, str(raster_path), interpolate="nearest", nodata=nodata)
        return pd.array([r if r is not None else np.nan for r in results])
    except ImportError:
        warnings.warn("rasterstats not installed — returning NaN column.")
        return np.full(len(points_df), np.nan)


def _worldcover_fraction(raster_path, points_df, wc_class, resolution_m=250):
    """Fraction of 250 m cell covered by a given WorldCover class."""
    try:
        import rasterio
        from rasterio.windows import from_bounds
        from rasterio.transform import rowcol
        from haversine import inverse_haversine, Direction

        fractions = []
        with rasterio.open(raster_path) as src:
            transform = src.transform
            data = src.read(1)
            for _, row in points_df.iterrows():
                lat, lon = row["Latitude"], row["Longitude"]
                half_deg = resolution_m / 2 / 111_320  # rough deg
                row_min = int((lat - half_deg - src.bounds.top) / transform.e)
                row_max = int((lat + half_deg - src.bounds.top) / transform.e)
                col_min = int((lon - half_deg - src.bounds.left) / transform.a)
                col_max = int((lon + half_deg - src.bounds.left) / transform.a)
                row_min, row_max = sorted([row_min, row_max])
                col_min, col_max = sorted([col_min, col_max])
                row_min = max(row_min, 0); col_min = max(col_min, 0)
                row_max = min(row_max, data.shape[0]-1)
                col_max = min(col_max, data.shape[1]-1)
                patch = data[row_min:row_max+1, col_min:col_max+1]
                if patch.size == 0:
                    fractions.append(0.0)
                else:
                    fractions.append((patch == wc_class).sum() / patch.size)
        return np.array(fractions)
    except (ImportError, Exception) as e:
        warnings.warn(f"rasterio error for WorldCover class {wc_class}: {e}")
        return np.zeros(len(points_df))


def _osm_binary_flag(points_df, tags, resolution_m=250, place="Delhi, India"):
    """Return 1 for cells that intersect any OSM feature matching `tags`."""
    try:
        import osmnx as ox
        import geopandas as gpd
        from shapely.geometry import Point, box
        import math

        # Fetch all matching OSM features for the target city at once
        gdf = ox.features_from_place(place, tags=tags)
        if gdf.empty:
            return np.zeros(len(points_df), dtype=int)
        gdf = gdf.to_crs("EPSG:32643")  # UTM zone 43N – metres (covers Delhi)

        r = resolution_m / 2
        boxes_geom = []
        for _, row in points_df.iterrows():
            lat, lon = row["Latitude"], row["Longitude"]
            # Convert to approximate metres
            cx = lon * 111_320 * math.cos(math.radians(lat))
            cy = lat * 110_540
            boxes_geom.append(box(cx-r, cy-r, cx+r, cy+r))
        
        cells_gdf = gpd.GeoDataFrame(geometry=boxes_geom, crs="EPSG:32643")
        # Ensure geometries are valid
        gdf["geometry"] = gdf.geometry.buffer(0)
        
        joined = gpd.sjoin(cells_gdf, gdf, how="inner", predicate="intersects")
        hit_indices = joined.index.unique()
        
        flags = np.zeros(len(points_df), dtype=int)
        flags[hit_indices] = 1
        return flags
    except (ImportError, Exception) as e:
        warnings.warn(f"OSM fetch failed for tags {tags}: {e} — returning zeros.")
        return np.zeros(len(points_df), dtype=int)


def _building_density(points_df, buildings_path=None, resolution_m=250):
    """Building footprint fraction per cell from Google Open Buildings."""
    if buildings_path is None or not os.path.exists(buildings_path):
        warnings.warn("open_buildings_delhi.gpkg not found — returning zeros.")
        return np.zeros(len(points_df))
    try:
        import geopandas as gpd
        from shapely.geometry import box

        buildings = gpd.read_file(buildings_path).to_crs("EPSG:32644")
        cell_area = resolution_m ** 2
        r = resolution_m / 2
        densities = []
        for _, row in points_df.iterrows():
            lat, lon = row["Latitude"], row["Longitude"]
            import math
            cx = lon * 111_320 * math.cos(math.radians(lat))
            cy = lat * 110_540
            cell_poly = gpd.GeoSeries([box(cx-r, cy-r, cx+r, cy+r)], crs="EPSG:32644")
            clipped = buildings.clip(cell_poly.iloc[0])
            footprint = clipped.geometry.area.sum()
            densities.append(min(footprint / cell_area, 1.0))
        return np.array(densities)
    except Exception as e:
        warnings.warn(f"Building density error: {e}")
        return np.zeros(len(points_df))


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("Paryavaran — Delhi NCT land_type_025.csv builder")
    print("=" * 65)

    # ── 1. Generate 250 m grid ────────────────────────────────────────────────
    if POINTS_CSV.exists():
        print(f"\n[1] Loading existing grid: {POINTS_CSV}")
        grid = pd.read_csv(POINTS_CSV, index_col=0)
    else:
        print(f"\n[1] Generating 250 m grid over Delhi…")
        grid = get_spaced_point_set_in_bbox(
            GRID_RESOLUTION_KM, BBSR_BOTTOM_LEFT, BBSR_TOP_RIGHT
        )
        grid.to_csv(POINTS_CSV, index=True)
        print(f"    Saved {len(grid):,} points → {POINTS_CSV}")
    print(f"    Grid size: {len(grid):,} cells")

    df = grid.copy()

    # ── 2. WorldPop population ────────────────────────────────────────────────
    print("\n[2] WorldPop population…")
    wp_path = DATA_DIR / "worldpop_india_2024_100m.tif"
    if wp_path.exists():
        df["Pop_density"] = _raster_zonal_stats(wp_path, df, stat="sum", nodata=-99999)
        df["Pop_density"] = df["Pop_density"].fillna(0).clip(lower=0)
    else:
        warnings.warn(f"WorldPop raster not found at {wp_path} — using zeros.")
        df["Pop_density"] = 0.0
    print(f"    Pop_density range: {df['Pop_density'].min():.1f} – {df['Pop_density'].max():.1f}")

    # ── 3. ESA WorldCover land cover ──────────────────────────────────────────
    print("\n[3] ESA WorldCover land cover…")
    wc_path = DATA_DIR / "worldcover_delhi.tif"
    if wc_path.exists():
        df["tree_frac"]      = _worldcover_fraction(wc_path, df, WC["tree"])
        df["grass_frac"]     = _worldcover_fraction(wc_path, df, WC["grass"])
        df["builtup_frac"]   = _worldcover_fraction(wc_path, df, WC["builtup"])
        df["water_frac"]     = _worldcover_fraction(wc_path, df, WC["water"])
        df["wetland_frac"]   = _worldcover_fraction(wc_path, df, WC["wetland"])
        df["bareland_frac"]  = _worldcover_fraction(wc_path, df, WC["bareland"])
        df["cropland_frac"]  = _worldcover_fraction(wc_path, df, WC["cropland"])
        # Binary flags (> 10 % of cell)
        df["Green_Space"]    = ((df["tree_frac"] + df["grass_frac"]) > 0.10).astype(int)
        df["Water"]          = (df["water_frac"] > 0.10).astype(int)
        df["Wetland"]        = (df["wetland_frac"] > 0.10).astype(int)
        df["Bare_Land"]      = (df["bareland_frac"] > 0.10).astype(int)
        df["Cropland"]       = (df["cropland_frac"] > 0.10).astype(int)
        df["Building"]       = (df["builtup_frac"] > 0.10).astype(int)
        df = df.drop(columns=[c for c in df.columns if c.endswith("_frac")])
    else:
        warnings.warn(f"WorldCover raster not found at {wc_path} — using zeros.")
        for col in ["Green_Space","Water","Wetland","Bare_Land","Cropland","Building"]:
            df[col] = 0

    # ── 4. MODIS LST heat stress ───────────────────────────────────────────────
    print("\n[4] MODIS LST heat stress…")
    lst_path = DATA_DIR / "modis_lst_summer_mean.tif"
    if lst_path.exists():
        df["Heat_LST"] = _raster_zonal_stats(lst_path, df, stat="mean")
        df["Heat_LST"] = df["Heat_LST"].fillna(df["Heat_LST"].median())
    else:
        warnings.warn(f"MODIS LST raster not found at {lst_path} — using constant 305 K.")
        df["Heat_LST"] = 305.0
    print(f"    Heat_LST range: {df['Heat_LST'].min():.1f} – {df['Heat_LST'].max():.1f} K")

    # ── 5. OSM features ───────────────────────────────────────────────────────
    print("\n[5] OpenStreetMap features (may take a few minutes)…")
    # Airports
    df["Airport"] = _osm_binary_flag(df, {"aeroway": ["aerodrome","runway"]}, place="Delhi, India")
    print("    Airports done")
    # Railway
    df["Railway"] = _osm_binary_flag(df, {"railway": True}, place="Delhi, India")
    print("    Railway done")
    # Roads (only trunk/primary to avoid every footpath being flagged)
    df["Road"] = _osm_binary_flag(df, {"highway": ["trunk","primary","secondary"]}, place="Delhi, India")
    print("    Roads done")
    # Hospitals
    df["Hospital"] = _osm_binary_flag(df, {"amenity": "hospital"}, place="Delhi, India")
    print("    Hospitals done")
    # Schools
    df["School"] = _osm_binary_flag(df, {"amenity": ["school","college","university"]}, place="Delhi, India")
    print("    Schools done")
    # Supplement Green_Space with OSM parks
    osm_gs = _osm_binary_flag(df, {
        "leisure": ["park","garden","recreation_ground","playground","nature_reserve"],
        "landuse": ["forest","grass"],
    }, place="Delhi, India")
    df["Green_Space"] = np.clip(df["Green_Space"] + osm_gs, 0, 1)
    print("    Green_Space supplemented with OSM")

    # ── 6. Building density (Google Open Buildings) ───────────────────────────
    print("\n[6] Building density (Google Open Buildings)…")
    bldg_path = DATA_DIR / "open_buildings_delhi.gpkg"
    df["Building_Density"] = _building_density(df, buildings_path=bldg_path)
    # If we got building footprints, use them to override WorldCover builtup flag
    if bldg_path.exists():
        df["Building"] = (df["Building_Density"] > 0.05).astype(int)
    print(f"    Building_Density range: {df['Building_Density'].min():.3f} – {df['Building_Density'].max():.3f}")

    # ── 7. Save land_type_025.csv ─────────────────────────────────────────────
    print(f"\n[7] Saving → {LAND_TYPE_CSV}")
    df.to_csv(LAND_TYPE_CSV, index=True)
    print(f"    Done. {len(df):,} rows, {len(df.columns)} columns.")
    print(f"    Columns: {list(df.columns)}")

    # ── 8. CPCB stub (if missing) ─────────────────────────────────────────────
    if not CPCB_CSV.exists():
        print(f"\n[8] Creating stub cpcb_stations.csv at {CPCB_CSV}")
        print("    ⚠️  Replace this with real CPCB data from data.gov.in")
        stub = pd.DataFrame({
            "Station": ["Delhi_Anand_Vihar", "Delhi_IGI_Airport", "Delhi_Rohini"],
            "Latitude":  [28.6469, 28.5665, 28.7315],
            "Longitude": [77.3152, 77.1011, 77.1025],
            "PM25": [89.0, 71.0, 95.0],
            "PM10": [145.0, 118.0, 152.0],
            "NO2":  [38.0, 31.0, 42.0],
        })
        stub.to_csv(CPCB_CSV, index=False)
        print("    Stub saved.")

    print("\n✅  Pre-processing complete.")
    print("   Next steps:")
    print("   1. Download real data files into Data/ (see README)")
    print("   2. Run Notebooks/create_penultimate_df.ipynb")
    print("   3. Run Notebooks/create_final_df.ipynb")
    print("   4. Run Plotly_Dash_App/app.py")


if __name__ == "__main__":
    main()
