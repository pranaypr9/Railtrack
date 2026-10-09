"""
RailConnect FastAPI backend
- ALL trains can be searched through /trains and /trains/catalogue
- ML-supported trains use the LSTM ETA model
- Any train available in the RailRadar catalogue can be opened for live tracking
- Non-ML trains return live RailRadar data without pretending to have an AI forecast

Run:
    python -m uvicorn api.main:app --reload --port 8001
"""

import os
import json
import sys
from typing import Optional
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from api.live import get_live_train_status

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "models"))
from model import ETALSTM, FEATURE_COLS  # noqa: E402


# ===============================================================
# PATHS
# ===============================================================

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

TRAINS_FILE = os.path.join(DATA_DIR, "trains.csv")
RAILRADAR_TRAINS_FILE = os.path.join(DATA_DIR, "railradar_trains.csv")
STATIONS_FILE = os.path.join(DATA_DIR, "stations.csv")


# ===============================================================
# HELPERS
# ===============================================================

def normalize_train_number(value) -> str:
    """Keep train numbers consistent across CSV files and API calls."""
    if pd.isna(value):
        return ""
    return (
        str(value)
        .strip()
        .replace(".0", "")
    )


def find_column(df, candidates):
    """Return the first matching column name from candidates."""
    lookup = {str(c).strip().lower(): c for c in df.columns}

    for candidate in candidates:
        key = candidate.strip().lower()
        if key in lookup:
            return lookup[key]

    return None


# ===============================================================
# LOAD TRAIN DATA
# ===============================================================

# ML-supported train list
try:
    trains_df = pd.read_csv(
        TRAINS_FILE,
        dtype={"train_number": str}
    )

    if "train_number" in trains_df.columns:
        trains_df["train_number"] = (
            trains_df["train_number"]
            .map(normalize_train_number)
        )

except Exception as e:
    print("WARNING: Could not load trains.csv:", e)
    trains_df = pd.DataFrame(
        columns=["train_number", "train_name"]
    )


# Complete train catalogue
try:
    railradar_trains_df = pd.read_csv(
        RAILRADAR_TRAINS_FILE,
        dtype={"train_number": str}
    )

    number_col = find_column(
        railradar_trains_df,
        ["train_number", "train_no", "trainnumber", "number"]
    )

    if number_col and number_col != "train_number":
        railradar_trains_df = railradar_trains_df.rename(
            columns={number_col: "train_number"}
        )

    if "train_number" in railradar_trains_df.columns:
        railradar_trains_df["train_number"] = (
            railradar_trains_df["train_number"]
            .map(normalize_train_number)
        )

    name_col = find_column(
        railradar_trains_df,
        ["train_name", "trainname", "name"]
    )

    if name_col and name_col != "train_name":
        railradar_trains_df = railradar_trains_df.rename(
            columns={name_col: "train_name"}
        )

    if "train_name" not in railradar_trains_df.columns:
        railradar_trains_df["train_name"] = ""

    railradar_trains_df["train_name"] = (
        railradar_trains_df["train_name"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    railradar_trains_df = (
        railradar_trains_df[
            ["train_number", "train_name"]
        ]
        .drop_duplicates(subset=["train_number"])
    )

    print(
        f"Loaded {len(railradar_trains_df)} trains "
        "from RailRadar catalogue."
    )

except Exception as e:
    print("WARNING: Could not load railradar_trains.csv:", e)

    railradar_trains_df = pd.DataFrame(
        columns=["train_number", "train_name"]
    )


# ===============================================================
# LOAD STATION / MODEL DATA
# ===============================================================

stations_df = pd.read_csv(
    STATIONS_FILE,
    dtype={"train_number": str}
)

stations_df["train_number"] = (
    stations_df["train_number"]
    .map(normalize_train_number)
)


with open(
    os.path.join(MODEL_DIR, "norm_stats.json"),
    encoding="utf-8"
) as f:
    STATS = json.load(f)


with open(
    os.path.join(MODEL_DIR, "eval_report.json"),
    encoding="utf-8"
) as f:
    EVAL_REPORT = json.load(f)


model = ETALSTM(
    n_features=len(FEATURE_COLS)
)

model.load_state_dict(
    torch.load(
        os.path.join(MODEL_DIR, "eta_lstm.pt"),
        map_location="cpu"
    )
)

model.eval()


# ===============================================================
# FASTAPI
# ===============================================================

app = FastAPI(
    title="RailConnect - Dynamic Railway ETA Prediction API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


# ===============================================================
# CONSTANTS / STATE
# ===============================================================

WEATHER_STATES = [
    "Clear",
    "Rain",
    "Fog",
    "Storm"
]

CONGESTION_LEVELS = [
    "Low",
    "Medium",
    "High"
]

# In-memory live run store.
# Production version can use Redis/PostgreSQL.
LIVE_RUNS = {}


# ===============================================================
# MODEL FUNCTIONS
# ===============================================================

def norm(col, val):
    s = STATS[col]

    std = s["std"]
    if std == 0:
        std = 1

    return (val - s["mean"]) / std


def denorm(col, val):
    s = STATS[col]

    return (
        val * s["std"]
        + s["mean"]
    )


def build_feature_row(
    station_row,
    delay_so_far,
    weather,
    congestion,
    hour,
    day
):
    hour_sin = np.sin(
        2 * np.pi * hour / 24
    )

    hour_cos = np.cos(
        2 * np.pi * hour / 24
    )

    day_sin = np.sin(
        2 * np.pi * day / 7
    )

    day_cos = np.cos(
        2 * np.pi * day / 7
    )

    row = {
        "dist_remaining_km_norm":
            norm(
                "dist_remaining_km",
                station_row["dist_remaining_km"]
            ),

        "cum_distance_km_norm":
            norm(
                "cum_distance_km",
                station_row["cum_distance_km"]
            ),

        "avg_hist_delay_min_norm":
            norm(
                "avg_hist_delay_min",
                station_row["avg_hist_delay_min"]
            ),

        "delay_so_far_min_norm":
            norm(
                "delay_so_far_min",
                delay_so_far
            ),

        "hour_sin": hour_sin,
        "hour_cos": hour_cos,

        "day_sin": day_sin,
        "day_cos": day_cos,

        "weather_clear":
            float(weather == "Clear"),

        "weather_rain":
            float(weather == "Rain"),

        "weather_fog":
            float(weather == "Fog"),

        "weather_storm":
            float(weather == "Storm"),

        "congestion_low":
            float(congestion == "Low"),

        "congestion_medium":
            float(congestion == "Medium"),

        "congestion_high":
            float(congestion == "High"),
    }

    return [
        row[c]
        for c in FEATURE_COLS
    ]


def run_inference(
    train_number: str,
    delay_so_far: float,
    weather: str,
    congestion: str,
    start_hour: int,
    day: int,
    from_seq: int = 0
):
    """
    Predict delay at every remaining station
    for an ML-supported train.
    """

    train_number = normalize_train_number(
        train_number
    )

    route = (
        stations_df[
            stations_df["train_number"] == train_number
        ]
        .sort_values("seq")
        .copy()
    )

    if route.empty:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Train {train_number} does not "
                "have an ML route dataset."
            )
        )

    total_distance = float(
        route["cum_distance_km"].max()
    )

    remaining = route[
        route["seq"] >= from_seq
    ].copy()

    remaining["dist_remaining_km"] = (
        total_distance
        - remaining["cum_distance_km"]
    )

    feats = []

    hour = start_hour

    for _, station in remaining.iterrows():

        feats.append(
            build_feature_row(
                station,
                delay_so_far,
                weather,
                congestion,
                hour % 24,
                day
            )
        )

        hour += 1

    if not feats:
        return []

    x = torch.tensor(
        [feats],
        dtype=torch.float32
    )

    with torch.no_grad():

        pred_norm = (
            model(x)
            .squeeze(0)
            .numpy()
        )

    pred_delay = denorm(
        "target_delay_min",
        pred_norm
    )

    pred_delay = np.clip(
        pred_delay,
        0,
        None
    )

    results = []

    for (_, station), delay in zip(
        remaining.iterrows(),
        pred_delay
    ):

        results.append({

            "seq":
                int(station["seq"]),

            "station_code":
                str(station["station_code"]),

            "station_name":
                str(
                    station["station_name"]
                ).strip(),

            "distance_km":
                float(
                    station["cum_distance_km"]
                ),

            "predicted_delay_min":
                round(
                    float(delay),
                    1
                ),

            "static_baseline_delay_min":
                round(
                    float(
                        station[
                            "avg_hist_delay_min"
                        ]
                    ),
                    1
                )
        })

    return results
def run_universal_inference(
    train_number: str,
    route: list,
    delay_so_far: float,
    weather: str = "Clear",
    congestion: str = "Low",
    start_hour: int = 8,
    day: int = 0,
    from_seq: int = 0
):
    """
    Run the existing LSTM against any available train route.

    For the 42 trained trains, the original route data is used.
    For other trains, RailRadar route data is converted into
    the same feature structure expected by the LSTM.
    """

    if not route:
        return []

    # Convert RailRadar route into a dataframe
    rows = []

    for i, station in enumerate(route):
        rows.append({
            "seq": int(
                station.get("sequence", station.get("seq", i))
            ),
            "station_code": str(
                station.get(
                    "stationCode",
                    station.get("station_code", "")
                )
            ),
            "station_name": str(
                station.get(
                    "stationName",
                    station.get("station_name", "")
                )
            ),
            "cum_distance_km": float(
                station.get(
                    "distanceFromOriginKm",
                    station.get(
                        "cum_distance_km",
                        0
                    )
                ) or 0
            ),
            "avg_hist_delay_min": float(
                station.get(
                    "avg_hist_delay_min",
                    0
                ) or 0
            )
        })

    df = pd.DataFrame(rows)

    if df.empty:
        return []

    df = df.sort_values("seq").reset_index(drop=True)

    total_distance = float(
        df["cum_distance_km"].max()
    )

    df["dist_remaining_km"] = (
        total_distance -
        df["cum_distance_km"]
    )

    remaining = df[
        df["seq"] >= from_seq
    ].copy()

    if remaining.empty:
        return []

    features = []
    hour = start_hour

    for _, station in remaining.iterrows():

        features.append(
            build_feature_row(
                station,
                delay_so_far,
                weather,
                congestion,
                hour % 24,
                day
            )
        )

        hour += 1

    x = torch.tensor(
        [features],
        dtype=torch.float32
    )

    with torch.no_grad():
        pred_norm = (
            model(x)
            .squeeze(0)
            .numpy()
        )

    pred_delay = denorm(
        "target_delay_min",
        pred_norm
    )

    pred_delay = np.clip(
        pred_delay,
        0,
        None
    )

    results = []

    for (_, station), delay in zip(
        remaining.iterrows(),
        pred_delay
    ):

        results.append({
            "seq": int(station["seq"]),

            "station_code":
                str(station["station_code"]),

            "station_name":
                str(station["station_name"]).strip(),

            "distance_km":
                round(
                    float(
                        station["cum_distance_km"]
                    ),
                    1
                ),

            "predicted_delay_min":
                round(
                    float(delay),
                    1
                ),

            "static_baseline_delay_min":
                round(
                    float(
                        station["avg_hist_delay_min"]
                    ),
                    1
                )
        })

    return results

# ===============================================================
# API MODELS
# ===============================================================

class SimulateRequest(BaseModel):

    run_id: Optional[str] = "demo"

    current_seq: int = 0

    weather: str = "Clear"

    congestion: str = "Low"

    hour_of_day: int = 8

    day_of_week: int = 0

    live_delay_observed_min: float = 0.0


# ===============================================================
# ROOT
# ===============================================================

@app.get("/")
def root():

    return {
        "service":
            "RailConnect - Dynamic Railway ETA Prediction API",

        "model":
            "PyTorch LSTM sequence regressor",

        "validation_report":
            EVAL_REPORT,

        "ml_supported_trains":
            int(
                stations_df[
                    "train_number"
                ].nunique()
            ),

        "catalogue_trains":
            int(
                railradar_trains_df[
                    "train_number"
                ].nunique()
            ),

        "docs":
            "/docs"
    }


# ===============================================================
# ALL TRAIN CATALOGUE
# ===============================================================

@app.get("/trains")
def list_all_trains():

    """
    Return ALL trains from the RailRadar catalogue.

    ML-supported trains are marked:
        ml_supported = True

    Other trains:
        ml_supported = False
    """

    catalogue = (
        railradar_trains_df
        .copy()
    )

    if catalogue.empty:

        # Fallback to trains.csv
        catalogue = (
            trains_df
            .copy()
        )

        if "train_name" not in catalogue.columns:

            catalogue["train_name"] = ""

    # Set of trains for which stations.csv exists
    ml_numbers = set(
        stations_df[
            "train_number"
        ]
        .astype(str)
        .unique()
    )

    out = []

    for _, train in catalogue.iterrows():

        number = normalize_train_number(
            train.get(
                "train_number",
                ""
            )
        )

        if not number:
            continue

        name = str(
            train.get(
                "train_name",
                ""
            )
        ).strip()

        route = (
            stations_df[
                stations_df["train_number"]
                == number
            ]
            .sort_values("seq")
        )

        item = {
            "train_number":
                number,

            "train_name":
                name,

            "ml_supported":
                number in ml_numbers,

            "origin":
                None,

            "destination":
                None,

            "n_stations":
                0,

            "total_distance_km":
                0
        }

        # Add route information if ML data exists
        if not route.empty:

            item["origin"] = (
                str(
                    route.iloc[0]["station_name"]
                ).strip()
            )

            item["destination"] = (
                str(
                    route.iloc[-1]["station_name"]
                ).strip()
            )

            item["n_stations"] = int(
                len(route)
            )

            item["total_distance_km"] = round(
                float(
                    route[
                        "cum_distance_km"
                    ].max()
                ),
                1
            )

        out.append(item)

    return sorted(
        out,
        key=lambda x: (
            int(x["train_number"])
            if x["train_number"].isdigit()
            else 999999999
        )
    )


# ===============================================================
# SEARCH ALL TRAINS
# ===============================================================

@app.get("/trains/catalogue")
def get_train_catalogue(
    search: Optional[str] = None,
    limit: int = 100
):
    """
    Search ALL trains by train number or train name.

    Examples:
        /trains/catalogue?search=20503
        /trains/catalogue?search=rajdhani
        /trains/catalogue?search=12919
    """

    df = (
        railradar_trains_df
        .copy()
    )

    if df.empty:
        return []

    if search:

        search = search.strip()

        number_mask = (
            df["train_number"]
            .astype(str)
            .str.contains(
                search,
                case=False,
                na=False,
                regex=False
            )
        )

        name_mask = (
            df["train_name"]
            .astype(str)
            .str.contains(
                search,
                case=False,
                na=False,
                regex=False
            )
        )

        df = df[
            number_mask
            | name_mask
        ]

    limit = max(
        1,
        min(
            int(limit),
            500
        )
    )

    ml_numbers = set(
        stations_df[
            "train_number"
        ]
        .astype(str)
        .unique()
    )

    results = []

    for _, row in df.head(limit).iterrows():

        number = normalize_train_number(
            row["train_number"]
        )

        results.append({

            "train_number":
                number,

            "train_name":
                str(
                    row.get(
                        "train_name",
                        ""
                    )
                ).strip(),

            "ml_supported":
                number in ml_numbers
        })

    return results


# ===============================================================
# TRAIN ROUTE
# ===============================================================

@app.get("/trains/{train_number}/route")
def get_route(train_number: str):

    train_number = normalize_train_number(
        train_number
    )

    route = (
        stations_df[
            stations_df["train_number"]
            == train_number
        ]
        .sort_values("seq")
    )

    if route.empty:

        raise HTTPException(
            status_code=404,
            detail=(
                "Train route is not available "
                "in the ML dataset."
            )
        )

    return route.to_dict(
        orient="records"
    )


# ===============================================================
# TRAIN DETAILS
# ===============================================================

@app.get("/trains/{train_number}/details")
def get_train_details(
    train_number: str
):

    train_number = normalize_train_number(
        train_number
    )

    catalogue_rows = (
        railradar_trains_df[
            railradar_trains_df[
                "train_number"
            ]
            == train_number
        ]
    )

    route = (
        stations_df[
            stations_df["train_number"]
            == train_number
        ]
        .sort_values("seq")
    )

    if (
        catalogue_rows.empty
        and route.empty
    ):

        raise HTTPException(
            status_code=404,
            detail="Train not found"
        )

    train_name = ""

    if not catalogue_rows.empty:

        train_name = str(
            catalogue_rows.iloc[0][
                "train_name"
            ]
        ).strip()

    details = {

        "train_number":
            train_number,

        "train_name":
            train_name,

        "ml_supported":
            not route.empty,

        "origin":
            None,

        "destination":
            None,

        "n_stations":
            0,

        "total_distance_km":
            0
    }

    if not route.empty:

        details["origin"] = str(
            route.iloc[0]["station_name"]
        ).strip()

        details["destination"] = str(
            route.iloc[-1]["station_name"]
        ).strip()

        details["n_stations"] = int(
            len(route)
        )

        details["total_distance_km"] = round(
            float(
                route[
                    "cum_distance_km"
                ].max()
            ),
            1
        )

    return details


# ===============================================================
# SIMULATION / ML ETA
# ===============================================================

@app.post(
    "/trains/{train_number}/simulate"
)
def simulate(
    train_number: str,
    req: SimulateRequest
):

    train_number = normalize_train_number(
        train_number
    )

    # Ensure this train has ML route data
    route = stations_df[
        stations_df["train_number"]
        == train_number
    ]

    if route.empty:

        raise HTTPException(
            status_code=400,
            detail=(
                f"AI ETA is not available "
                f"for train {train_number}. "
                "This train is not part of "
                "the trained ML route dataset."
            )
        )

    # Validate weather/congestion
    if req.weather not in WEATHER_STATES:

        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid weather. Choose: "
                + ", ".join(WEATHER_STATES)
            )
        )

    if req.congestion not in CONGESTION_LEVELS:

        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid congestion. Choose: "
                + ", ".join(CONGESTION_LEVELS)
            )
        )

    key = (
        f"{train_number}_"
        f"{req.run_id}"
    )

    LIVE_RUNS[key] = {

        "current_seq":
            req.current_seq,

        "delay_so_far_min":
            req.live_delay_observed_min,

        "weather":
            req.weather,

        "congestion":
            req.congestion,

        "hour_of_day":
            req.hour_of_day,

        "day_of_week":
            req.day_of_week
    }

    predictions = run_inference(

        train_number=
            train_number,

        delay_so_far=
            req.live_delay_observed_min,

        weather=
            req.weather,

        congestion=
            req.congestion,

        start_hour=
            req.hour_of_day,

        day=
            req.day_of_week,

        from_seq=
            req.current_seq
    )

    return {

        "train_number":
            train_number,

        "run_id":
            req.run_id,

        "as_of_seq":
            req.current_seq,

        "live_conditions":
            req.dict(),

        "eta_forecast":
            predictions
    }


# ===============================================================
# CURRENT ML ETA
# ===============================================================

@app.get(
    "/trains/{train_number}/eta"
)
def get_current_eta(
    train_number: str,
    run_id: str = "demo"
):

    train_number = normalize_train_number(
        train_number
    )

    key = (
        f"{train_number}_"
        f"{run_id}"
    )

    if key not in LIVE_RUNS:

        raise HTTPException(
            status_code=404,
            detail=(
                "No active run - "
                "POST to /simulate first"
            )
        )

    state = LIVE_RUNS[key]

    predictions = run_inference(

        train_number=
            train_number,

        delay_so_far=
            state["delay_so_far_min"],

        weather=
            state["weather"],

        congestion=
            state["congestion"],

        start_hour=
            state["hour_of_day"],

        day=
            state["day_of_week"],

        from_seq=
            state["current_seq"]
    )

    return {

        "train_number":
            train_number,

        "run_id":
            run_id,

        "state":
            state,

        "eta_forecast":
            predictions
    }


# ===============================================================
# LIVE RAILRADAR STATUS
# ===============================================================

@app.get(
    "/trains/{train_number}/live"
)
def get_live_status(
    train_number: str
):
    train_number = normalize_train_number(train_number)

    try:

        # -------------------------------------------------------
        # 1. GET LIVE DATA
        # -------------------------------------------------------

        live_data = get_live_train_status(
            train_number
        )

        current_location = live_data.get(
            "currentLocation",
            {}
        )

        current_station_code = (
            current_location.get(
                "stationCode"
            )
        )

        current_station_name = (
            current_location.get(
                "stationName"
            )
        )

        # -------------------------------------------------------
        # 2. LOCAL ML ROUTE
        # -------------------------------------------------------

        local_route = (
            stations_df[
                stations_df["train_number"]
                == train_number
            ]
            .sort_values("seq")
            .copy()
        )

        ml_supported = not local_route.empty

        predictions = []
        current_seq = 0
        live_delay = float(
            current_location.get(
                "delayMinutes",
                0
            ) or 0
        )

        # -------------------------------------------------------
        # 3. ML TRAIN
        # -------------------------------------------------------

        if ml_supported:

            current_rows = local_route[
                local_route["station_code"]
                .astype(str)
                ==
                str(current_station_code)
            ]

            if not current_rows.empty:

                current_seq = int(
                    current_rows.iloc[0]["seq"]
                )

            predictions = run_inference(
                train_number=train_number,
                delay_so_far=live_delay,
                weather="Clear",
                congestion="Low",
                start_hour=datetime.now().hour,
                day=datetime.now().weekday(),
                from_seq=current_seq
            )

            prediction_mode = (
                "LSTM - trained route"
            )

        # -------------------------------------------------------
        # 4. NON-ML TRAIN
        # -------------------------------------------------------

        else:

            radar_route = live_data.get(
                "route",
                []
            )

            if radar_route:

                # Find current sequence
                for i, station in enumerate(
                    radar_route
                ):

                    if (
                        str(
                            station.get(
                                "stationCode",
                                ""
                            )
                        )
                        ==
                        str(
                            current_station_code
                        )
                    ):

                        current_seq = i
                        break

                predictions = (
                    run_universal_inference(
                        train_number=train_number,
                        route=radar_route,
                        delay_so_far=live_delay,
                        weather="Clear",
                        congestion="Low",
                        start_hour=datetime.now().hour,
                        day=datetime.now().weekday(),
                        from_seq=current_seq
                    )
                )

                prediction_mode = (
                    "LSTM - generalized route"
                )

            else:

                prediction_mode = (
                    "No route available"
                )

        # -------------------------------------------------------
        # 5. RESPONSE
        # -------------------------------------------------------

        return {

            "train_number":
                train_number,

            "source":
                "RailRadar",

            "ml_supported":
                ml_supported,

            "prediction_available":
                len(predictions) > 0,

            "prediction_mode":
                prediction_mode,

            "message":
                (
                    "Live location and ETA "
                    "prediction available."
                    if predictions
                    else
                    "Live location available, "
                    "but route data is unavailable."
                ),

            "current_location":
                current_location,

            "current_station": {

                "code":
                    current_station_code,

                "name":
                    current_station_name,

                "sequence":
                    current_seq
            },

            "live_delay_min":
                live_delay,

            "prediction_time":
                datetime.now().isoformat(),

            "eta_forecast":
                predictions,

            "live_status":
                live_data
        }

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=502,
            detail=(
                "Unable to fetch live "
                "train data: "
                f"{str(e)}"
            )
        )