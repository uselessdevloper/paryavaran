# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import sys
import pathlib

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
from dash import Dash, dcc, html
from dash.dependencies import Input, Output

ROOT = pathlib.Path(__file__).resolve().parent.parent.as_posix()
if ROOT not in sys.path:
    sys.path.append(ROOT)

app = Dash(__name__)
app.title = "Paryavaran – Delhi Greenspace Dashboard"

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(ROOT + "/final_df.csv", engine="c")

# ── Rename columns to human-readable display names ────────────────────────────
df = df.rename(columns={
    # Priority / scores
    "Greenspace_score":           "Priority Score",
    "penalty_reward":             "Land Feasibility Multiplier",
    "AQ_score":                   "Air Quality Score",
    # Population
    "Pop_density":                "Population Density",
    # Air quality pollutants
    "PM25":                       "PM2.5 (µg/m³)",
    "PM10":                       "PM10 (µg/m³)",
    "NO2":                        "NO₂ (µg/m³)",
    # Distance / greenspace
    "Distance_Nearest_Greenspace":"Distance to Nearest Green Space (km)",
    "Green_Space":                "Existing Green Spaces",
    # Heat
    "Heat_LST":                   "Heat Stress (LST K)",
    # Land cover – binary
    "Building":                   "Buildings",
    "Building_Density":           "Building Density",
    "Airport":                    "Airports",
    "Water":                      "Water Bodies",
    "Railway":                    "Railway / Metro",
    "Road":                       "Roads",
    "Hospital":                   "Hospitals",
    "School":                     "Schools",
    "Bare_Land":                  "Bare Land",
    "Cropland":                   "Cropland",
    "Wetland":                    "Wetlands",
})

# ── Delhi NCT city centre (Connaught Place) ───────────────────────────────────
CITY_LAT = 28.6139
CITY_LON  = 77.2090
DEFAULT_ZOOM = 10.5

# ── Dropdown option groups ────────────────────────────────────────────────────
SCORE_COLS = [
    "Priority Score",
    "Air Quality Score",
    "Land Feasibility Multiplier",
]

CONTINUOUS_COLS = [
    "Population Density",
    "PM2.5 (µg/m³)",
    "PM10 (µg/m³)",
    "NO₂ (µg/m³)",
    "Distance to Nearest Green Space (km)",
    "Heat Stress (LST K)",
    "Building Density",
]

BINARY_COLS = [
    "Existing Green Spaces",
    "Buildings",
    "Airports",
    "Water Bodies",
    "Railway / Metro",
    "Roads",
    "Hospitals",
    "Schools",
    "Bare Land",
    "Cropland",
    "Wetlands",
]

# Keep only columns that actually exist in the dataframe
def _existing(cols):
    return [c for c in cols if c in df.columns]

SCORE_COLS     = _existing(SCORE_COLS)
CONTINUOUS_COLS = _existing(CONTINUOUS_COLS)
BINARY_COLS    = _existing(BINARY_COLS)

all_options = (
    [{"label": f"🏆 {c}", "value": c} for c in SCORE_COLS]
    + [{"label": f"📊 {c}", "value": c} for c in CONTINUOUS_COLS]
    + [{"label": f"🗺️ {c}", "value": c} for c in BINARY_COLS]
)

# ── Layout ────────────────────────────────────────────────────────────────────
app.layout = html.Div(
    children=[
        html.H1(
            "Paryavaran — Delhi Greenspace Suitability Dashboard",
            style={
                "font-family": "Arial",
                "textAlign": "center",
                "color": "black",
                "font-size": "32px",
                "border-bottom": "4px solid #2ecc71",
                "padding-bottom": "0.5em",
                "margin-bottom": "0.5em",
            },
        ),

        html.P(
            "Identifies priority zones for new urban greenspaces in Delhi "
            "(NCT) using WorldPop population data, CPCB air-quality monitoring, "
            "ESA WorldCover land cover, MODIS heat stress, and OpenStreetMap.",
            style={
                "font-family": "Arial",
                "font-size": "13px",
                "color": "#555",
                "textAlign": "center",
                "margin-bottom": "1em",
            },
        ),

        html.Div(
            [
                html.Div(id="output_data"),
                html.Label(
                    "Select overlay",
                    style={"font-family": "Arial", "font-size": "14px", "padding": "4px"},
                ),
                dcc.Dropdown(
                    id="my_dropdown",
                    options=all_options,
                    optionHeight=22,
                    value="Priority Score",
                    disabled=False,
                    multi=False,
                    searchable=True,
                    search_value="",
                    placeholder="Select data overlay…",
                    clearable=True,
                    style={
                        "width": "60%",
                        "font-family": "Arial",
                        "font-size": "15px",
                        "textAlign": "left",
                    },
                ),
            ],
            className="fifteen columns",
        ),

        html.Div(
            [dcc.Graph(id="our_graph")],
            className="fifteen columns",
        ),

        html.Footer(
            "Data sources: WorldPop (CC BY 4.0) · CPCB / data.gov.in · "
            "ESA WorldCover (CC BY 4.0) · OpenStreetMap (ODbL) · MODIS LST (NASA)",
            style={
                "font-family": "Arial",
                "font-size": "11px",
                "color": "#aaa",
                "textAlign": "center",
                "padding": "1em",
            },
        ),
    ]
)


# ── Callback ──────────────────────────────────────────────────────────────────
@app.callback(
    Output(component_id="our_graph", component_property="figure"),
    [Input(component_id="my_dropdown", component_property="value")],
)
def build_graph(column_chosen):
    if column_chosen is None:
        column_chosen = "Priority Score"

    center = dict(lat=CITY_LAT, lon=CITY_LON)

    if column_chosen == "Population Density":
        fig = px.density_mapbox(
            df, lat="Latitude", lon="Longitude", z=column_chosen,
            radius=15, opacity=0.45,
            center=center, zoom=DEFAULT_ZOOM,
            mapbox_style="open-street-map",
            color_continuous_scale="Turbo",
        )

    elif column_chosen in BINARY_COLS:
        fig = px.scatter_mapbox(
            df, lat="Latitude", lon="Longitude",
            opacity=0.30, color=column_chosen,
            zoom=DEFAULT_ZOOM, mapbox_style="open-street-map",
            color_continuous_scale=["white", "#2ecc71"],
        )

    elif column_chosen in ("Priority Score", "Distance to Nearest Green Space (km)"):
        fig = px.scatter_mapbox(
            df, lat="Latitude", lon="Longitude",
            size=column_chosen, opacity=0.35,
            zoom=DEFAULT_ZOOM + 0.25, mapbox_style="open-street-map",
            color=column_chosen, color_continuous_scale="Turbo",
        )

    else:
        # Hexbin for continuous overlays (AQ, heat, PM2.5, etc.)
        fig = ff.create_hexbin_mapbox(
            df, lat="Latitude", lon="Longitude",
            color=column_chosen, nx_hexagon=120,
            opacity=0.30, center=center,
            mapbox_style="open-street-map",
            zoom=DEFAULT_ZOOM,
            color_continuous_scale="Turbo",
            labels={"color": column_chosen},
            agg_func=np.mean,
        )
        fig.update_traces(marker_line_width=0)

    fig.update_layout(
        autosize=True,
        height=820,
        margin=dict(l=20, r=20, t=20, b=20),
    )
    fig.layout.coloraxis.colorbar.title = ""
    fig.update_mapboxes(pitch=40)
    return fig


if __name__ == "__main__":
    app.run(debug=True)
