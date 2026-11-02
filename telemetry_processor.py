"""Telemetry ingestion, feature engineering, and grip anomaly inference.

FastF1 data is public sporting telemetry, not a tire-sensor measurement. The
inferred grip capacity and micro-slip labels are therefore analytical proxies.
"""
from __future__ import annotations

from typing import Tuple
from pathlib import Path

import fastf1
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error


Path("f1_cache").mkdir(parents=True, exist_ok=True)
fastf1.Cache.enable_cache("f1_cache")
FEATURES = ["Speed", "RPM", "Throttle", "Brake", "Brake_Rate", "Gear", "TyreLife"]
EPSILON = 1e-5


def _engineer_features(telemetry: pd.DataFrame) -> pd.DataFrame:
    """Return telemetry with physically motivated, numerically guarded features."""
    df = telemetry.copy().reset_index(drop=True)
    required = {"Time", "Speed", "X", "Y"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Telemetry is missing required channels: {', '.join(sorted(missing))}")
    # Some FastF1 sessions do not expose every optional channel. Keep those as
    # missing values so downstream experiments can select available features.
    for optional in FEATURES[1:]:
        if optional not in df.columns:
            df[optional] = np.nan

    if pd.api.types.is_timedelta64_dtype(df["Time"]):
        time_s = df["Time"].dt.total_seconds()
    else:
        time_s = pd.to_timedelta(df["Time"]).dt.total_seconds()
    df["Time_s"] = pd.to_numeric(time_s, errors="coerce")
    dt = df["Time_s"].diff().fillna(0.0).clip(lower=0.0)
    df["dt"] = dt

    for col in FEATURES + ["X", "Y"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    speed_ms = df["Speed"] / 3.6
    df["Speed_ms"] = speed_ms
    df["G_long"] = (speed_ms.diff().fillna(0.0) / (dt + EPSILON)) / 9.81

    dx, dy = df["X"].diff().fillna(0.0), df["Y"].diff().fillna(0.0)
    ds = np.sqrt(dx.pow(2) + dy.pow(2))
    heading = np.unwrap(np.arctan2(dy.to_numpy(), dx.to_numpy()))
    d_theta = pd.Series(heading, index=df.index).diff().fillna(0.0)
    df["ds"] = ds
    df["Distance"] = ds.cumsum()
    # Signed radius preserves turn direction; the epsilon prevents singularities.
    radius = ds / (d_theta + np.where(d_theta >= 0, EPSILON, -EPSILON))
    radius = radius.where(radius.abs() > EPSILON, np.inf)
    lat_g = (speed_ms.pow(2) / radius) / 9.81
    df["G_lat"] = lat_g.clip(-6.0, 6.0).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    df["G_total"] = np.sqrt(df["G_long"].pow(2) + df["G_lat"].pow(2))
    df["Brake_Rate"] = (df["Brake"].diff().fillna(0.0) / (dt + EPSILON)).replace(
        [np.inf, -np.inf], 0.0
    )
    df["Throttle_Rate"] = (df["Throttle"].diff().fillna(0.0) / (dt + EPSILON)).replace(
        [np.inf, -np.inf], 0.0
    )
    return df.replace([np.inf, -np.inf], np.nan)


def get_telemetry_matrix(
    year: int, track: str, session_type: str, driver_code: str
) -> pd.DataFrame:
    """Load the requested driver's fastest lap and engineer its telemetry."""
    year = int(year)
    track = str(track).strip()
    session_type = str(session_type).strip().upper()
    driver_code = str(driver_code).strip().upper()
    if not track or not driver_code or len(driver_code) != 3:
        raise ValueError("Track is required and driver code must be exactly three letters.")
    session = fastf1.get_session(year, track, session_type)
    session.load(telemetry=True, laps=True, weather=False, messages=False)
    laps = session.laps.pick_drivers(driver_code).pick_quicklaps()
    if laps.empty:
        laps = session.laps.pick_drivers(driver_code)
    if laps.empty:
        raise ValueError(f"No lap data found for {driver_code} at {track} ({year} {session_type}).")
    fastest_lap = laps.pick_fastest()
    if fastest_lap is None or fastest_lap.empty:
        raise ValueError("FastF1 did not return a valid fastest lap.")
    raw = fastest_lap.get_telemetry().copy()
    if raw.empty:
        raise ValueError("Fastest lap has no telemetry samples.")
    return _engineer_features(raw)


def train_slip_inference_engine(
    df: pd.DataFrame,
) -> Tuple[RandomForestRegressor, StandardScaler, pd.DataFrame]:
    """Fit grip proxy model and append prediction, delta, and slip classification.

    A chronological holdout prevents shuffled neighboring samples from leaking
    into both sets. The returned frame includes predictions for the complete lap.
    """
    result = df.copy()
    if "G_total" not in result or "Speed" not in result:
        raise ValueError("Cannot train grip proxy model without Speed and G_total.")
    feature_cols = [name for name in FEATURES if name in result.columns and
                    pd.to_numeric(result[name], errors="coerce").notna().any()]
    if not feature_cols:
        raise ValueError("No usable telemetry feature channels are available.")
    usable = result[feature_cols + ["G_total"]].apply(pd.to_numeric, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    if len(usable) < 10:
        raise ValueError("At least 10 complete telemetry samples are required for inference.")
    x, y = usable[feature_cols], usable["G_total"]
    split = max(1, int(len(usable) * 0.2))
    if len(usable) - split < 2:
        raise ValueError("Not enough samples for an 80/20 train/validation split.")
    # Time-ordered split retains the requested 80/20 evaluation structure.
    x_train, x_test = x.iloc[:-split], x.iloc[-split:]
    y_train, y_test = y.iloc[:-split], y.iloc[-split:]
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    model = RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1)
    model.fit(x_train_scaled, y_train)
    result["Predicted_Grip"] = np.nan
    valid_mask = result[feature_cols].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)
    result.loc[valid_mask, "Predicted_Grip"] = model.predict(
        scaler.transform(result.loc[valid_mask, feature_cols].astype(float))
    )
    result["Micro-Slip Delta"] = (result["Predicted_Grip"] - result["G_total"]).clip(lower=0.0)
    # A 0.25 g deficit is a conservative proxy threshold; full throttle is required.
    throttle = pd.to_numeric(result.get("Throttle", pd.Series(np.nan, index=result.index)), errors="coerce")
    result["Micro_Slip"] = (throttle >= 99.5) & (result["Micro-Slip Delta"] >= 0.25)
    result.attrs["validation_mae"] = float(mean_absolute_error(y_test, model.predict(x_test_scaled)))
    result.attrs["validation_samples"] = int(len(y_test))
    result.attrs["validation_note"] = "chronological 80/20 holdout; grip is a telemetry-derived proxy"
    return model, scaler, result





