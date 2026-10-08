"""
helpers.py  –  Paryavaran  (India / Delhi edition)
===================================================
Replaces the original Kainos ASDI-Hackathon helpers.py.

Changes vs. original
---------------------
* All Ordnance Survey WFS API calls removed
  (get_feature_type_in_bbox, is_airport, is_water, is_building,
   is_railway_station, is_green_space, is_urban_area).
* Land-type detection now uses pre-processed CSVs derived from
  ESA WorldCover 2021 + OpenStreetMap (no live API calls required
  during scoring).
* Air-quality (AQ) score now uses CPCB PM2.5 / PM10 / NO2 stations
  interpolated via distance-weighted KNN – same BallTree/haversine
  approach as the original, just with Indian station data.
* Population density is read from a pre-aggregated WorldPop raster
  (100 m → 250 m zonal sum), stored in land_type_025.csv.
* Heat stress (MODIS LST) and building density (Google Open Buildings)
  are new columns carried through from land_type_025.csv.
* Priority score formula updated for Indian urban-planning context.

Data pipeline
-------------
  land_type_025.csv  (grid + land-cover + pop + heat + buildings)
          ↓  create_penultimate_df.ipynb  →  fill_points_land_type_df()
  penultimate_df.csv  (adds AQ score, greenspace distance, pop norm)
          ↓  create_final_df.ipynb        →  fill_penultimate_df()
  final_df.csv        (adds priority score)
          ↓  Plotly_Dash_App/app.py
"""

import math
import pathlib
import pickle
import statistics
from time import time

import numpy as np
import pandas as pd
from haversine import Direction, haversine, inverse_haversine
from multiprocess import Pool, cpu_count
from sklearn.neighbors import BallTree
from tqdm import tqdm

from Enums.land_type import LAND_TYPE

# ── Path helpers ──────────────────────────────────────────────────────────────
ROOT_FOLDER_PATH = pathlib.Path(__file__).resolve().parent.as_posix()
PICKLE_FOLDER_PATH = ROOT_FOLDER_PATH + "/Pickles/"

# ── Delhi NCT bounding box (approx.) ─────────────────────────────────────────
# Covers the full National Capital Territory of Delhi
BBSR_BOTTOM_LEFT = (28.40, 76.84)   # (lat, lon) SW corner
BBSR_TOP_RIGHT   = (28.88, 77.35)   # (lat, lon) NE corner
BBSR_CENTER      = (28.6139, 77.2090)  # Connaught Place / city centre
GRID_RESOLUTION_KM = 0.25           # 250 m


# ═══════════════════════════════════════════════════════════════════════════════
#  GRID GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def convert_point_list_to_df(points):
    """Convert a list of (lat, lon) tuples to a DataFrame."""
    lats, lons = zip(*points)
    return pd.DataFrame({"Latitude": list(lats), "Longitude": list(lons)})


def get_spaced_point_set_in_bbox(d, bottom_left, top_right):
    """Return a 250 m evenly-spaced grid inside the given bounding box.

    Args:
        d (float): grid spacing in km (use 0.25 for 250 m).
        bottom_left (tuple): (lat, lon) of SW corner.
        top_right   (tuple): (lat, lon) of NE corner.

    Returns:
        pd.DataFrame: columns Latitude, Longitude.
    """
    top_left = (top_right[0], bottom_left[1])
    length = haversine(top_left, bottom_left)
    num_rows = int(length // d)
    print(f"Grid rows: {num_rows}")

    points = []
    for r in tqdm(range(num_rows)):
        row_start = inverse_haversine(top_left, r * d, Direction.SOUTH)
        row_end   = inverse_haversine(top_right, r * d, Direction.SOUTH)
        row_width = haversine(row_start, row_end)
        n_cols    = int(row_width // d)
        shift     = ((row_width / d) % 1) / 2
        row_start = inverse_haversine(row_start, shift, Direction.EAST)
        for i in range(n_cols):
            points.append(inverse_haversine(row_start, i * d, Direction.EAST))

    return convert_point_list_to_df(points)


def generate_delhi_grid(resolution_km=GRID_RESOLUTION_KM):
    """Convenience wrapper – generate a Delhi NCT 250 m grid."""
    return get_spaced_point_set_in_bbox(
        resolution_km, BBSR_BOTTOM_LEFT, BBSR_TOP_RIGHT
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  BOUNDING-BOX HELPER  (kept for OSM querying during pre-processing)
# ═══════════════════════════════════════════════════════════════════════════════

def get_bbox_of_point(latitude, longitude, resolution_diameter_km):
    """Return a WGS-84 bbox string around a point at the given resolution.

    Returns:
        str: 'minlat,minlon,maxlat,maxlon'
    """
    radius = resolution_diameter_km / 2
    hypot  = radius / math.cos(math.radians(45))
    sw = inverse_haversine((latitude, longitude), hypot, Direction.SOUTHWEST)
    ne = inverse_haversine((latitude, longitude), hypot, Direction.NORTHEAST)
    return f"{sw[0]:.8f},{sw[1]:.8f},{ne[0]:.8f},{ne[1]:.8f}"


# ═══════════════════════════════════════════════════════════════════════════════
#  AIR QUALITY  –  CPCB station KNN
# ═══════════════════════════════════════════════════════════════════════════════

def _load_cpcb_stations(stations_csv_path=None):
    """Load CPCB station data.

    Expects a CSV with columns: Latitude, Longitude, PM25, PM10, NO2.
    If the file doesn't exist yet, returns a tiny synthetic stub so the
    pipeline doesn't crash before real data is downloaded.
    """
    if stations_csv_path is None:
        stations_csv_path = ROOT_FOLDER_PATH + "/Data/cpcb_stations.csv"
    try:
        return pd.read_csv(stations_csv_path)
    except FileNotFoundError:
        # Synthetic fallback – replace with real CPCB data before production
        print("[helpers] WARNING: cpcb_stations.csv not found, using stub data.")
        return pd.DataFrame({
            "Latitude":  [20.2961, 20.35, 20.25],
            "Longitude": [85.8245, 85.80, 85.85],
            "PM25": [45.0, 60.0, 38.0],
            "PM10": [80.0, 95.0, 65.0],
            "NO2":  [25.0, 35.0, 20.0],
        })


def _build_aq_balltree(stations_df):
    coords_rad = np.radians(stations_df[["Latitude", "Longitude"]].values)
    return BallTree(coords_rad, leaf_size=40, metric="haversine")


def interpolate_aq_for_grid(grid_df, stations_csv_path=None, k=3):
    """Distance-weighted KNN interpolation of CPCB AQ metrics onto the grid.

    Args:
        grid_df (pd.DataFrame): must have Latitude, Longitude columns.
        stations_csv_path (str | None): path to cpcb_stations.csv.
        k (int): number of nearest stations to use.

    Returns:
        pd.DataFrame: grid_df with added PM25, PM10, NO2, AQ_score columns.
    """
    stations = _load_cpcb_stations(stations_csv_path)
    tree = _build_aq_balltree(stations)

    query_rad = np.radians(grid_df[["Latitude", "Longitude"]].values)
    dists, idxs = tree.query(query_rad, k=min(k, len(stations)))

    # Avoid division-by-zero for exact station hits
    dists = np.where(dists == 0, 1e-9, dists)
    weights = 1.0 / dists

    for col in ["PM25", "PM10", "NO2"]:
        vals = stations[col].values[idxs]           # shape (n_grid, k)
        interpolated = (vals * weights).sum(axis=1) / weights.sum(axis=1)
        grid_df[col] = interpolated

    return grid_df


def compute_aq_score(df):
    """Compute a normalised AQ score (0–1, higher = worse air quality).

    Uses PM2.5, PM10, NO2 weighted 50 / 30 / 20 % respectively,
    then min-max normalised across the city grid.
    """
    # Weight individual pollutants
    df["_aq_raw"] = (
        df["PM25"] * 0.50
        + df["PM10"] * 0.30
        + df["NO2"]  * 0.20
    )
    # Normalise 0–1
    mn, mx = df["_aq_raw"].min(), df["_aq_raw"].max()
    df["AQ_score"] = (df["_aq_raw"] - mn) / (mx - mn + 1e-9)
    df = df.drop(columns=["_aq_raw"])
    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  GREEN SPACE  –  BallTree / haversine  (same logic as original)
# ═══════════════════════════════════════════════════════════════════════════════

def dist_nearest_greenspace_function(df):
    """Mean haversine distance (km) to the 3 nearest greenspace cells.

    Identical algorithm to the original Kainos implementation.
    """
    gs = df.loc[df["Green_Space"] == 1, ["Latitude", "Longitude"]].apply(np.radians)
    if len(gs) == 0:
        df["Distance_Nearest_Greenspace"] = 10.0
        return df

    all_pts = df[["Latitude", "Longitude"]].apply(np.radians)

    tree = BallTree(gs, leaf_size=40, metric="haversine")
    dist, _ = tree.query(all_pts, k=min(3, len(gs)))

    mean_dist_rad = dist.mean(axis=1)
    distances_km  = mean_dist_rad * 6371  # Earth radius in km

    df.insert(
        loc=min(8, len(df.columns)),
        column="Distance_Nearest_Greenspace",
        value=distances_km,
    )
    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  POPULATION  –  WorldPop raster (pre-aggregated into land_type_025.csv)
# ═══════════════════════════════════════════════════════════════════════════════

def normalise_pop_density(df):
    """Min-max normalise the Pop_density column."""
    mn, mx = df["Pop_density"].min(), df["Pop_density"].max()
    df["norm_Pop_density"] = (df["Pop_density"] - mn) / (mx - mn + 1e-9)
    return df


def calculate_popd_weight(df, resolution_m=250):
    """WHO greenspace-per-capita weight (same logic as original).

    WHO standard: 50 m² per capita.
    Weight > 1 → city is below standard (need more greenspace).
    Weight < 1 → city exceeds standard.
    """
    standard_gs_per_pop_m2 = 50.0
    cell_area_m2 = resolution_m ** 2
    total_pop = df["Pop_density"].sum()
    total_gs_m2 = (df["Green_Space"] == 1).sum() * cell_area_m2
    gs_per_capita = total_gs_m2 / max(total_pop, 1)
    return standard_gs_per_pop_m2 / max(gs_per_capita, 1e-6)


# ═══════════════════════════════════════════════════════════════════════════════
#  HEAT STRESS  –  MODIS LST (pre-aggregated into land_type_025.csv)
# ═══════════════════════════════════════════════════════════════════════════════

def normalise_heat_lst(df):
    """Normalise summer-mean LST to 0–1 (higher = hotter)."""
    if "Heat_LST" not in df.columns:
        df["Heat_LST"] = 0.0
    mn, mx = df["Heat_LST"].min(), df["Heat_LST"].max()
    df["norm_Heat_LST"] = (df["Heat_LST"] - mn) / (mx - mn + 1e-9)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  BUILDING DENSITY  –  Google Open Buildings (pre-aggregated)
# ═══════════════════════════════════════════════════════════════════════════════

def normalise_building_density(df):
    """Normalise building footprint fraction to 0–1."""
    if "Building_Density" not in df.columns:
        df["Building_Density"] = df["Building"].astype(float)
    mn, mx = df["Building_Density"].min(), df["Building_Density"].max()
    df["norm_Building_Density"] = (df["Building_Density"] - mn) / (mx - mn + 1e-9)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  LAND-FEASIBILITY  –  OSM + WorldCover (binary columns in land_type_025.csv)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_land_feasibility(row):
    """Return a land-feasibility multiplier for the priority score.

    Rules (same penalty/reward philosophy as original Kainos):
      Airport  → 0    (no greenspace here)
      Water    → 0    (preserve water bodies)
      Hospital → 0.3  (minimal space, critical infrastructure)
      School   → 0.4  (campus grounds only)
      Railway  → 0.5  (corridor exists but limited)
      Building → 0.75 (demolition costly)
      Wetland  → 0.6  (partial credit – flood buffer value)
      Cropland → 0.8  (agricultural land, limited urban repurposing)
      Bare land→ 1.4  (excellent candidate)
      Green_Space (existing) → 0.5  (already counts; may expand edges)
      None of the above    → 1.0
    """
    f = 1.0
    if row.get("Airport", 0):
        return 0.0
    if row.get("Water", 0):
        return 0.0
    if row.get("Hospital", 0):
        f *= 0.30
    if row.get("School", 0):
        f *= 0.40
    if row.get("Railway", 0):
        f *= 0.50
    if row.get("Building", 0):
        f *= 0.75
    if row.get("Wetland", 0):
        f *= 0.60
    if row.get("Cropland", 0):
        f *= 0.80
    if row.get("Bare_Land", 0):
        f *= 1.40
    if row.get("Green_Space", 0):
        f *= 0.50   # already green; score penalised so other cells rank higher
    return f


# ═══════════════════════════════════════════════════════════════════════════════
#  PRIORITY SCORE  –  India urban greenspace suitability
# ═══════════════════════════════════════════════════════════════════════════════
#
#  Priority = (
#    0.25 × Population Need
#  + 0.20 × Greenspace Deficit
#  + 0.20 × Heat Stress
#  + 0.15 × Air Pollution
#  + 0.10 × Accessibility Deficit
#  + 0.10 × Land Suitability component
#  ) × Land Feasibility multiplier
#
# ═══════════════════════════════════════════════════════════════════════════════

def greenspace_score_function(
    aq_score,
    norm_pop,
    norm_heat,
    norm_building,
    green_space,
    dist_nearest_gs,
    land_feasibility,
    popd_weight,
):
    """Compute priority score for one grid cell.

    Returns:
        (priority_score, land_feasibility) tuple – matches original signature
        convention so the DataFrame assignment stays identical.
    """
    # 1. Population Need  (higher pop density in areas without greenspace)
    population_need = norm_pop * popd_weight          # amplified when WHO std not met

    # 2. Greenspace Deficit  (normalised; existing greenspace reduces deficit)
    gs_deficit = (1.0 - green_space) * 1.0            # 0 if already green, 1 if not

    # 3. Heat Stress
    heat_stress = norm_heat

    # 4. Air Pollution
    air_pollution = aq_score

    # 5. Accessibility Deficit  (further from nearest park = higher deficit)
    #    +1 so score increases with distance, same trick as original
    accessibility_deficit = min(dist_nearest_gs + 1, 5.0) / 5.0

    raw = (
        0.25 * population_need
        + 0.20 * gs_deficit
        + 0.20 * heat_stress
        + 0.15 * air_pollution
        + 0.10 * accessibility_deficit
        + 0.10 * norm_building          # high built-up → more need
    )

    priority = raw * land_feasibility
    return [priority, land_feasibility]


def apply_greenspace_score_function(df, resolution=250):
    """Apply priority scoring to every row of the penultimate dataframe."""
    popd_weight = calculate_popd_weight(df, resolution_m=resolution)
    print(f"popd_weight = {popd_weight:.4f}")

    # Pre-compute feasibility per row (vectorisable but kept row-wise for clarity)
    land_cols = [
        "Airport", "Water", "Hospital", "School", "Railway",
        "Building", "Wetland", "Cropland", "Bare_Land", "Green_Space",
    ]
    for c in land_cols:
        if c not in df.columns:
            df[c] = 0

    df[["Greenspace_score", "penalty_reward"]] = df.apply(
        lambda row: greenspace_score_function(
            aq_score         = row["AQ_score"],
            norm_pop         = row["norm_Pop_density"],
            norm_heat        = row.get("norm_Heat_LST", 0.0),
            norm_building    = row.get("norm_Building_Density", 0.0),
            green_space      = row.get("Green_Space", 0),
            dist_nearest_gs  = row["Distance_Nearest_Greenspace"],
            land_feasibility = compute_land_feasibility(row),
            popd_weight      = popd_weight,
        ),
        axis=1,
        result_type="expand",
    )

    df = df.drop(columns=["norm_Pop_density"], errors="ignore")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  PARALLELISATION  (unchanged from original)
# ═══════════════════════════════════════════════════════════════════════════════

def parallelise(df, func):
    n_cores = cpu_count()
    splits  = np.array_split(df, n_cores)
    pool    = Pool(n_cores)
    results = pool.map(func, splits)
    pool.close()
    pool.join()
    return pd.concat(results)


# ═══════════════════════════════════════════════════════════════════════════════
#  PIPELINE ENTRY POINTS  (called by the notebooks)
# ═══════════════════════════════════════════════════════════════════════════════

def fill_points_land_type_df(local_path=None):
    """Read land_type_025.csv and compute penultimate_df features.

    Steps
    -----
    1. Load land_type_025.csv  (grid + land-cover + Pop_density + Heat_LST +
       Building_Density, all pre-processed from raster/vector sources).
    2. Binary-cast land-type columns.
    3. Calculate distance to nearest greenspace.
    4. Interpolate CPCB air-quality values → AQ_score.
    5. Normalise population density.
    6. Normalise heat LST.
    7. Normalise building density.

    Returns:
        pd.DataFrame: penultimate_df ready for final scoring.
    """
    if local_path is None:
        local_path = ROOT_FOLDER_PATH + "/land_type_025.csv"

    df = pd.read_csv(local_path, index_col=0)

    # Binary-cast land-type columns
    land_cols = [
        "Airport", "Water", "Building", "Green_Space",
        "Railway", "Road", "Hospital", "School",
        "Bare_Land", "Cropland", "Wetland",
    ]
    for col in land_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)

    # Ensure required columns exist with sensible defaults
    for col, default in [
        ("Pop_density", 0.0),
        ("Heat_LST",    df.get("Heat_LST", pd.Series([300.0])).mean() if "Heat_LST" not in df.columns else None),
        ("Building_Density", 0.0),
    ]:
        if col not in df.columns:
            df[col] = default

    t0 = time()
    df = dist_nearest_greenspace_function(df)
    print(f"dist_nearest_greenspace_function complete  [{round(time()-t0,2)}s]")

    t0 = time()
    df = interpolate_aq_for_grid(df)
    df = compute_aq_score(df)
    print(f"interpolate_aq_for_grid + compute_aq_score complete  [{round(time()-t0,2)}s]")

    t0 = time()
    df = normalise_pop_density(df)
    df = normalise_heat_lst(df)
    df = normalise_building_density(df)
    print(f"normalise columns complete  [{round(time()-t0,2)}s]")

    return df


def fill_penultimate_df(local_path=None):
    """Read penultimate_df.csv and compute the final priority score.

    Returns:
        pd.DataFrame: final_df ready for the Dash application.
    """
    if local_path is None:
        local_path = ROOT_FOLDER_PATH + "/penultimate_df.csv"

    df = pd.read_csv(local_path, index_col=0)

    t0 = time()
    df = apply_greenspace_score_function(df, resolution=250)
    print(f"apply_greenspace_score_function complete  [{round(time()-t0,2)}s]")

    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  LAND-TYPE CSV HELPERS  (used by pre-processing scripts)
# ═══════════════════════════════════════════════════════════════════════════════

def preprocess_land_type_dataframes(dataframes_list, save_path, points_df_path):
    """Stack partial land-type dataframes and join to the grid.

    Kept for compatibility with legacy processing scripts.
    """
    stacked = pd.concat(dataframes_list, axis=0)
    points  = pd.read_csv(points_df_path, header=0, index_col=0)

    if len(stacked) != len(points):
        raise ValueError(
            f"Length mismatch: stacked={len(stacked)}, grid={len(points)}"
        )

    land_type_points_df = pd.concat([points, stacked], axis=1)

    # Expand pipe-separated land-type strings into binary columns
    all_land_types = [lt.value for lt in LAND_TYPE]
    for lt in all_land_types:
        land_type_points_df[lt] = (
            land_type_points_df["Land_Type"]
            .str.contains(lt, na=False)
            .astype(int)
        )

    land_type_points_df = land_type_points_df.drop(columns=["Land_Type"])
    land_type_points_df.to_csv(save_path)
    print(f"Saved land_type_025.csv → {save_path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  LEGACY AWS STUBS  (kept so old notebook cells don't crash)
#  Replace bucket/key values with your own S3 bucket if using cloud storage.
# ═══════════════════════════════════════════════════════════════════════════════

def upload_pickle_to_s3(bucket, model, key):
    try:
        import boto3
        s3  = boto3.resource("s3")
        s3.Object(bucket, key).put(Body=pickle.dumps([model]))
        print("Successful upload")
    except Exception as e:
        print(f"Failed upload: {e}")


def upload_df_to_s3(bucket, df, key):
    try:
        import boto3
        from io import StringIO
        s3 = boto3.resource("s3")
        buf = StringIO()
        df.to_csv(buf)
        s3.Object(bucket, key).put(Body=buf.getvalue())
        print("Successful upload")
    except Exception as e:
        print(f"Failed upload: {e}")


def read_csv_from_s3(bucket, key):
    import boto3
    client = boto3.client("s3")
    obj    = client.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(obj["Body"])
