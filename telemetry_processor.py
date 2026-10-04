"""Telemetry ingestion, feature engineering, and grip anomaly inference.

FastF1 data is public sporting telemetry, not a tire-sensor measurement.
The inferred grip capacity and micro-slip labels are analytical proxies.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import fastf1
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error


# ============================================================
# FASTF1 CACHE
# ============================================================

CACHE_DIR = Path(__file__).resolve().parent / "f1_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

fastf1.Cache.enable_cache(str(CACHE_DIR))


# ============================================================
# MODEL FEATURES
# ============================================================

FEATURES = [
    "Speed",
    "RPM",
    "Throttle",
    "Brake",
    "Brake_Rate",
    "Gear",
    "TyreLife",
]

EPSILON = 1e-5


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def _engineer_features(telemetry: pd.DataFrame) -> pd.DataFrame:
    """Create engineering features from FastF1 telemetry."""

    df = telemetry.copy().reset_index(drop=True)

    # --------------------------------------------------------
    # Make sure the basic telemetry channels exist
    # --------------------------------------------------------

    required_raw = {
        "Time",
        "Speed",
        "X",
        "Y",
        "RPM",
        "Throttle",
        "Brake",
    }

    missing = required_raw.difference(df.columns)

    if missing:
        raise ValueError(
            "Telemetry is missing required channels: "
            + ", ".join(sorted(missing))
        )

    # --------------------------------------------------------
    # FastF1 normally calls the gear channel 'nGear'
    # Convert it to our internal name 'Gear'
    # --------------------------------------------------------

    if "Gear" not in df.columns:

        if "nGear" in df.columns:
            df["Gear"] = df["nGear"]

        else:
            # Some telemetry sources may not contain gear.
            # Use zero rather than stopping the entire simulation.
            df["Gear"] = 0

    # --------------------------------------------------------
    # TyreLife is not guaranteed to be present in telemetry.
    #
    # If it was supplied by the lap loader, use it.
    # Otherwise use a simple sample progression.
    # --------------------------------------------------------

    if "TyreLife" not in df.columns:

        # Tyre life is not directly available in every
        # FastF1 telemetry object.
        #
        # For this proxy model, a normalized progression
        # through the lap is safer than crashing.
        if len(df) > 1:
            df["TyreLife"] = np.linspace(
                1.0,
                float(len(df)),
                len(df)
            )
        else:
            df["TyreLife"] = 1.0

    # --------------------------------------------------------
    # Convert numerical columns
    # --------------------------------------------------------

    numerical_columns = [
        "Speed",
        "RPM",
        "Throttle",
        "Brake",
        "Gear",
        "TyreLife",
        "X",
        "Y",
    ]

    for col in numerical_columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Time → seconds
    # --------------------------------------------------------

    if pd.api.types.is_timedelta64_dtype(df["Time"]):

        time_s = df["Time"].dt.total_seconds()

    else:

        time_s = pd.to_timedelta(
            df["Time"]
        ).dt.total_seconds()

    df["Time_s"] = pd.to_numeric(
        time_s,
        errors="coerce"
    )

    # --------------------------------------------------------
    # Time difference
    # --------------------------------------------------------

    dt = (
        df["Time_s"]
        .diff()
        .fillna(0.0)
        .clip(lower=0.0)
    )

    df["dt"] = dt

    # --------------------------------------------------------
    # Speed conversion
    #
    # FastF1 Speed = km/h
    # --------------------------------------------------------

    speed_ms = df["Speed"] / 3.6

    df["Speed_ms"] = speed_ms

    # --------------------------------------------------------
    # Longitudinal G
    # --------------------------------------------------------

    df["G_long"] = (
        speed_ms.diff().fillna(0.0)
        / (dt + EPSILON)
        / 9.81
    )

    # --------------------------------------------------------
    # Position differences
    # --------------------------------------------------------

    dx = df["X"].diff().fillna(0.0)
    dy = df["Y"].diff().fillna(0.0)

    ds = np.sqrt(
        dx.pow(2) + dy.pow(2)
    )

    # --------------------------------------------------------
    # Heading
    # --------------------------------------------------------

    heading = np.unwrap(
        np.arctan2(
            dy.to_numpy(),
            dx.to_numpy()
        )
    )

    d_theta = (
        pd.Series(
            heading,
            index=df.index
        )
        .diff()
        .fillna(0.0)
    )

    # --------------------------------------------------------
    # Distance
    # --------------------------------------------------------

    df["ds"] = ds

    df["Distance"] = ds.cumsum()

    # --------------------------------------------------------
    # Turning radius
    # --------------------------------------------------------

    radius = (
        ds
        / (
            d_theta
            + np.where(
                d_theta >= 0,
                EPSILON,
                -EPSILON
            )
        )
    )

    radius = radius.where(
        radius.abs() > EPSILON,
        np.inf
    )

    # --------------------------------------------------------
    # Lateral G
    # --------------------------------------------------------

    lat_g = (
        speed_ms.pow(2)
        / radius
    ) / 9.81

    df["G_lat"] = (
        lat_g
        .clip(-6.0, 6.0)
        .replace(
            [np.inf, -np.inf],
            0.0
        )
        .fillna(0.0)
    )

    # --------------------------------------------------------
    # Total G
    # --------------------------------------------------------

    df["G_total"] = np.sqrt(
        df["G_long"].pow(2)
        + df["G_lat"].pow(2)
    )

    # --------------------------------------------------------
    # Brake rate
    #
    # IMPORTANT:
    # This is DERIVED here, so it must NOT be required
    # before feature engineering.
    # --------------------------------------------------------

    df["Brake_Rate"] = (
        df["Brake"].diff().fillna(0.0)
        / (dt + EPSILON)
    )

    df["Brake_Rate"] = (
        df["Brake_Rate"]
        .replace(
            [np.inf, -np.inf],
            0.0
        )
        .fillna(0.0)
    )

    # --------------------------------------------------------
    # Throttle rate
    # --------------------------------------------------------

    df["Throttle_Rate"] = (
        df["Throttle"].diff().fillna(0.0)
        / (dt + EPSILON)
    )

    df["Throttle_Rate"] = (
        df["Throttle_Rate"]
        .replace(
            [np.inf, -np.inf],
            0.0
        )
        .fillna(0.0)
    )

    # --------------------------------------------------------
    # Final cleanup
    # --------------------------------------------------------

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    return df


# ============================================================
# TELEMETRY LOADING
# ============================================================

def get_telemetry_matrix(
    year: int,
    track: str,
    session_type: str,
    driver_code: str,
) -> pd.DataFrame:
    """Load the driver's fastest lap and engineer telemetry."""

    year = int(year)

    track = str(track).strip()

    session_type = (
        str(session_type)
        .strip()
        .upper()
    )

    driver_code = (
        str(driver_code)
        .strip()
        .upper()
    )

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    if not track:
        raise ValueError(
            "Track name is required."
        )

    if (
        not driver_code
        or len(driver_code) != 3
        or not driver_code.isalpha()
    ):
        raise ValueError(
            "Driver code must be exactly three letters."
        )

    # --------------------------------------------------------
    # Get FastF1 session
    # --------------------------------------------------------

    session = fastf1.get_session(
        year,
        track,
        session_type
    )

    # --------------------------------------------------------
    # Load session
    # --------------------------------------------------------

    session.load(
        telemetry=True,
        laps=True,
        weather=False,
        messages=False,
    )

    # --------------------------------------------------------
    # Get driver's laps
    # --------------------------------------------------------

    laps = (
        session.laps
        .pick_drivers(driver_code)
        .pick_quicklaps()
    )

    # If no quick laps are available,
    # use all laps for that driver.
    if laps.empty:

        laps = session.laps.pick_drivers(
            driver_code
        )

    if laps.empty:
        raise ValueError(
            f"No lap data found for {driver_code} "
            f"at {track} ({year} {session_type})."
        )

    # --------------------------------------------------------
    # Fastest lap
    # --------------------------------------------------------

    fastest_lap = laps.pick_fastest()

    if fastest_lap is None or fastest_lap.empty:
        raise ValueError(
            "FastF1 did not return a valid fastest lap."
        )

    # --------------------------------------------------------
    # Get telemetry
    # --------------------------------------------------------

    raw = fastest_lap.get_telemetry().copy()

    if raw.empty:
        raise ValueError(
            "Fastest lap has no telemetry samples."
        )

    # --------------------------------------------------------
    # Add TyreLife if available from lap data
    # --------------------------------------------------------

    if "TyreLife" in fastest_lap.index:

        tyre_life = fastest_lap["TyreLife"]

        if pd.notna(tyre_life):
            raw["TyreLife"] = float(
                tyre_life
            )

    elif "TyreLife" in fastest_lap.index:

        raw["TyreLife"] = pd.to_numeric(
            fastest_lap["TyreLife"],
            errors="coerce"
        )

    # --------------------------------------------------------
    # FastF1 may use nGear instead of Gear.
    #
    # Do this before feature engineering.
    # --------------------------------------------------------

    if "Gear" not in raw.columns and "nGear" in raw.columns:

        raw["Gear"] = raw["nGear"]

    # --------------------------------------------------------
    # Engineer all features
    # --------------------------------------------------------

    return _engineer_features(raw)


# ============================================================
# MACHINE LEARNING
# ============================================================

def train_slip_inference_engine(
    df: pd.DataFrame,
) -> Tuple[
    RandomForestRegressor,
    StandardScaler,
    pd.DataFrame,
]:
    """Train grip proxy model and calculate micro-slip proxy."""

    result = df.copy()

    # --------------------------------------------------------
    # Check model features
    # --------------------------------------------------------

    missing = set(
        FEATURES + ["G_total"]
    ).difference(result.columns)

    if missing:
        raise ValueError(
            "Cannot train model; missing columns: "
            + ", ".join(sorted(missing))
        )

    # --------------------------------------------------------
    # Convert model columns to numeric
    # --------------------------------------------------------

    usable = (
        result[
            FEATURES + ["G_total"]
        ]
        .apply(
            pd.to_numeric,
            errors="coerce"
        )
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .dropna()
    )

    if len(usable) < 10:
        raise ValueError(
            "At least 10 complete telemetry samples "
            "are required for inference."
        )

    # --------------------------------------------------------
    # X and y
    # --------------------------------------------------------

    x = usable[FEATURES]

    y = usable["G_total"]

    # --------------------------------------------------------
    # Chronological 80/20 split
    # --------------------------------------------------------

    split = max(
        1,
        int(len(usable) * 0.2)
    )

    if len(usable) - split < 2:
        raise ValueError(
            "Not enough samples for an 80/20 "
            "train/validation split."
        )

    x_train = x.iloc[:-split]
    x_test = x.iloc[-split:]

    y_train = y.iloc[:-split]
    y_test = y.iloc[-split:]

    # --------------------------------------------------------
    # Scale features
    # --------------------------------------------------------

    scaler = StandardScaler()

    x_train_scaled = scaler.fit_transform(
        x_train
    )

    x_test_scaled = scaler.transform(
        x_test
    )

    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=12,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        x_train_scaled,
        y_train
    )

    # --------------------------------------------------------
    # Predict complete lap
    # --------------------------------------------------------

    result["Predicted_Grip"] = np.nan

    valid_mask = (
        result[FEATURES]
        .apply(
            pd.to_numeric,
            errors="coerce"
        )
        .notna()
        .all(axis=1)
    )

    if valid_mask.any():

        prediction_data = (
            result.loc[
                valid_mask,
                FEATURES
            ]
            .astype(float)
        )

        result.loc[
            valid_mask,
            "Predicted_Grip"
        ] = model.predict(
            scaler.transform(
                prediction_data
            )
        )

    # --------------------------------------------------------
    # Grip deficit
    # --------------------------------------------------------

    result["Micro-Slip Delta"] = (
        result["Predicted_Grip"]
        - result["G_total"]
    ).clip(
        lower=0.0
    )

    # --------------------------------------------------------
    # Micro-slip proxy
    #
    # Full throttle + >= 0.25g grip deficit
    # --------------------------------------------------------

    throttle = pd.to_numeric(
        result["Throttle"],
        errors="coerce"
    )

    result["Micro_Slip"] = (
        (throttle >= 99.5)
        & (
            result["Micro-Slip Delta"]
            >= 0.25
        )
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    test_predictions = model.predict(
        x_test_scaled
    )

    validation_mae = mean_absolute_error(
        y_test,
        test_predictions
    )

    result.attrs["validation_mae"] = float(
        validation_mae
    )

    result.attrs["validation_samples"] = int(
        len(y_test)
    )

    result.attrs["validation_note"] = (
        "chronological 80/20 holdout; "
        "grip is a telemetry-derived proxy"
    )

    return (
        model,
        scaler,
        result
    )