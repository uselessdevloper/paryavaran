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
server = app.server

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

# ── Build Graph Function ──────────────────────────────────────────────────────
def build_graph(column_chosen="Priority Score"):
    try:
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
        if hasattr(fig.layout, "coloraxis") and fig.layout.coloraxis and hasattr(fig.layout.coloraxis, "colorbar") and fig.layout.coloraxis.colorbar:
            fig.layout.coloraxis.colorbar.title = ""
        fig.update_mapboxes(pitch=40)
        return fig
    except Exception as e:
        import traceback
        traceback.print_exc()
        err_fig = px.scatter()
        err_fig.add_annotation(
            text=f"Error rendering map: {e}",
            showarrow=False,
            font=dict(size=16, color="red")
        )
        return err_fig


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
            [dcc.Graph(id="our_graph", figure=build_graph("Priority Score"))],
            className="fifteen columns",
        ),

        # ── OpenCV 5 Agentic Vision Inspector ──────────────────────────────────
        html.Div(
            [
                html.H3(
                    "🔍 OpenCV 5 & Agentic Vision Engine: Micro-Zone Aerial Inspector",
                    style={
                        "font-family": "Arial",
                        "textAlign": "left",
                        "color": "#1b4d3e",
                        "font-size": "20px",
                        "margin-top": "2em",
                        "margin-bottom": "0.4em",
                        "border-bottom": "2px solid #2ecc71",
                        "padding-bottom": "6px",
                    },
                ),
                html.P(
                    "Autonomous Perception-Decision-Action loop driven by OpenCV 5 computer vision. "
                    "When high-resolution 250m aerial micro-zones are inspected, OpenCV 5 extracts ground-truth "
                    "canopy (HSV+ExG), unpaved bare soil parcels, water bodies, and structural edge density "
                    "to adaptively override tabular estimates and trigger targeted municipal interventions.",
                    style={
                        "font-family": "Arial",
                        "font-size": "13px",
                        "color": "#555",
                        "margin-bottom": "1.2em",
                    },
                ),
                html.Div(
                    [
                        html.Label(
                            "Select Micro-Zone for OpenCV 5 Agentic Inspection:",
                            style={"font-family": "Arial", "font-weight": "bold", "font-size": "14px"},
                        ),
                        dcc.RadioItems(
                            id="zone_selector",
                            options=[
                                {"label": " Okhla Industrial Fringe (Bare Soil Opportunity)", "value": "ZONE_OKHLA_BARE"},
                                {"label": " Connaught Place Commercial Core (Dense Concrete)", "value": "ZONE_CP_DENSE"},
                                {"label": " Central Ridge Reserve (Existing Canopy Verification)", "value": "ZONE_RIDGE_CANOPY"},
                                {"label": " Yamuna River Floodplain (Water Body Preservation)", "value": "ZONE_YAMUNA_RIVER"},
                            ],
                            value="ZONE_OKHLA_BARE",
                            inline=True,
                            style={"margin-top": "6px", "margin-bottom": "14px", "font-family": "Arial", "font-size": "13px"},
                        ),
                        html.Div(id="agentic_inspection_container"),
                    ],
                    style={
                        "background": "#f8fdf9",
                        "border": "1px solid #d4edda",
                        "border-radius": "8px",
                        "padding": "16px",
                        "margin-bottom": "1.5em",
                    },
                ),
            ],
            className="fifteen columns",
        ),

        html.Footer(
            "Data sources: WorldPop (CC BY 4.0) · CPCB / data.gov.in · "
            "ESA WorldCover (CC BY 4.0) · OpenStreetMap (ODbL) · MODIS LST (NASA) · OpenCV 5.0",
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


# ── Callbacks ─────────────────────────────────────────────────────────────────
@app.callback(
    Output(component_id="our_graph", component_property="figure"),
    [Input(component_id="my_dropdown", component_property="value")],
)
def update_graph(column_chosen):
    return build_graph(column_chosen)


import json
import base64

DECISION_TRACES_PATH = pathlib.Path(ROOT) / "vision_engine" / "output_inspections" / "agentic_decision_traces.json"
DECISION_TRACES = {}
if DECISION_TRACES_PATH.exists():
    try:
        with open(DECISION_TRACES_PATH, "r", encoding="utf-8") as f:
            for item in json.load(f):
                DECISION_TRACES[item["tile_id"]] = item
    except Exception:
        pass


@app.callback(
    Output("agentic_inspection_container", "children"),
    [Input("zone_selector", "value")],
)
def render_agentic_inspection(selected_zone):
    if not selected_zone or selected_zone not in DECISION_TRACES:
        return html.Div("No inspection trace available for this zone.", style={"color": "#888"})

    trace = DECISION_TRACES[selected_zone]
    metrics = trace.get("perception_metrics", {})
    tools = trace.get("tools_invoked", [])
    reasoning = trace.get("reasoning_trace", [])
    img_path = trace.get("inspection_image", "")

    # Encode image if exists
    img_src = None
    if img_path and pathlib.Path(img_path).exists():
        try:
            with open(img_path, "rb") as img_f:
                b64 = base64.b64encode(img_f.read()).decode("utf-8")
                img_src = f"data:image/png;base64,{b64}"
        except Exception:
            img_src = None

    badge_color = {
        "BRANCH_MIYAWAKI_MICRO_FOREST": "#28a745",
        "BRANCH_ROOFTOP_CORRIDOR": "#fd7e14",
        "BRANCH_CANOPY_PROTECTION": "#17a2b8",
        "BRANCH_WATER_EXCLUSION": "#007bff",
    }.get(trace.get("decision_branch", ""), "#6c757d")

    return html.Div(
        [
            html.Div(
                [
                    # Left column: Annotated OpenCV 5 Vision Overlay
                    html.Div(
                        [
                            html.H5("OpenCV 5 Visual Overlay & Contour Detection", style={"font-size": "14px", "margin-bottom": "8px"}),
                            html.Img(
                                src=img_src,
                                style={
                                    "width": "100%",
                                    "max-width": "460px",
                                    "border": "1px solid #ddd",
                                    "border-radius": "6px",
                                    "box-shadow": "0 2px 4px rgba(0,0,0,0.1)",
                                },
                            ) if img_src else html.P("Inspection image not found."),
                        ],
                        style={"flex": "1", "minWidth": "320px", "padding-right": "16px"},
                    ),
                    # Right column: Agentic Decision Trace
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Span(
                                        trace.get("decision_branch", ""),
                                        style={
                                            "background": badge_color,
                                            "color": "white",
                                            "padding": "4px 10px",
                                            "border-radius": "4px",
                                            "font-weight": "bold",
                                            "font-size": "12px",
                                            "display": "inline-block",
                                            "margin-bottom": "8px",
                                        },
                                    ),
                                    html.H4(
                                        trace.get("recommended_action", ""),
                                        style={"font-size": "15px", "color": "#2c3e50", "margin-top": "4px", "margin-bottom": "8px"},
                                    ),
                                ]
                            ),
                            html.P(
                                [
                                    html.Strong("Score Adjustment: "),
                                    f"Initial Tabular: {trace.get('initial_score', 0):.2f} → ",
                                    html.Span(
                                        f"Adjusted: {trace.get('adjusted_score', 0):.2f}",
                                        style={"font-weight": "bold", "color": "#28a745" if trace.get('adjusted_score', 0) > trace.get('initial_score', 0) else "#c0392b"},
                                    ),
                                ],
                                style={"font-size": "13px", "margin-bottom": "6px"},
                            ),
                            html.P(
                                [
                                    html.Strong("OpenCV 5 Visual Perception Metrics:"),
                                    html.Br(),
                                    f"• Tree Canopy: {metrics.get('canopy_pct', 0):.1f}% | Bare Ground: {metrics.get('bare_ground_pct', 0):.1f}%",
                                    html.Br(),
                                    f"• Impervious/Built: {metrics.get('impervious_pct', 0):.1f}% | Water: {metrics.get('water_pct', 0):.1f}%",
                                    html.Br(),
                                    f"• Largest Continuous Parcel: {metrics.get('largest_parcel_sqm', 0):,.0f} m² | Edge Density: {metrics.get('edge_density', 0):.4f}",
                                ],
                                style={"font-size": "12px", "color": "#333", "background": "#fff", "padding": "8px", "border-radius": "4px", "border": "1px solid #eee"},
                            ),
                            html.P(
                                [
                                    html.Strong("Automated Tool Invocations:"),
                                    html.Br(),
                                    html.Code(", ".join(tools), style={"font-size": "11px", "color": "#8e44ad"}),
                                ],
                                style={"font-size": "12px", "margin-top": "6px", "margin-bottom": "6px"},
                            ),
                            html.P(
                                [
                                    html.Strong("Reasoning Trace:"),
                                    html.Ul([html.Li(r, style={"font-size": "12px", "color": "#444"}) for r in reasoning], style={"padding-left": "18px", "margin-top": "4px"}),
                                ],
                                style={"margin-top": "6px"},
                            ),
                        ],
                        style={"flex": "1.2", "minWidth": "340px"},
                    ),
                ],
                style={"display": "flex", "flexWrap": "wrap", "gap": "16px", "alignItems": "flex-start"},
            )
        ]
    )


if __name__ == "__main__":
    app.run(debug=True)

