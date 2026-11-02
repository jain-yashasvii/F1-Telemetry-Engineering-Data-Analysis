"""Streamlit pit-wall dashboard for telemetry-derived grip and slip proxies."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from telemetry_processor import get_telemetry_matrix, train_slip_inference_engine
from anomaly_experiment import run_experiment


logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("telemetry_dashboard")

st.set_page_config(page_title="Telemetry Intelligence | Pit Wall", layout="wide")
st.markdown(
    """
    <style>
    .stApp { background: #0b1018; color: #e6edf3; }
    [data-testid="stSidebar"] { background: #111923; }
    [data-testid="stMetric"] { background: #121c28; border: 1px solid #263648;
      padding: 14px; border-radius: 8px; }
    h1, h2, h3 { color: #e6edf3; }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("Adversarial Telemetry Intelligence")
st.caption("Non-invasive tire micro-slip estimator · telemetry-derived engineering proxies")

with st.sidebar:
    st.header("Session configuration")
    year = st.selectbox("Year", options=list(range(2025, 2017, -1)), index=0)
    track = st.text_input("Track name", value="Monza").strip()
    session_type = st.selectbox("Session", ["R", "Q", "S", "FP1", "FP2", "FP3"], index=0)
    driver_code = st.text_input("Driver code", value="LAW", max_chars=3).strip().upper()
    st.header("Simulation conditions")
    temp_shift = st.slider("Track surface temperature shift (°C)", -15, 15, 0)
    drizzle = st.checkbox("Light drizzle / elevated humidity", value=False)
    st.caption("Temperature scenario scales the grip proxy by 0.5% per °C (bounded ±7.5%).")
    st.header("Research experiment")
    anomaly_enabled = st.checkbox("Run feature-set anomaly comparison", value=True)
    contamination = st.slider("Expected anomaly fraction (contamination)", 0.01, 0.20, 0.02, 0.01, help="Expected training anomaly fraction; controls the Isolation Forest threshold.")
    execute = st.button("Execute Telemetry Simulation", type="primary", use_container_width=True)

if execute:
    if len(driver_code) != 3 or not driver_code.isalpha():
        st.error("Enter a valid three-letter driver code.")
    elif not track:
        st.error("Enter a track name.")
    else:
        try:
            with st.spinner(f"Loading {year} {track} {session_type} telemetry for {driver_code}…"):
                frame = get_telemetry_matrix(int(year), str(track), str(session_type), str(driver_code))
                # Explicit scenario factors modify observed grip before model fitting.
                temperature_factor = float(np.clip(1.0 - 0.005 * int(temp_shift), 0.925, 1.075))
                weather_factor = 0.80 if bool(drizzle) else 1.0
                frame["G_total"] = pd.to_numeric(frame["G_total"], errors="coerce") * temperature_factor * weather_factor
                model, scaler, result = train_slip_inference_engine(frame)
            mae = float(result.attrs["validation_mae"])
            slip_pct = float(result["Micro_Slip"].mean() * 100.0)
            max_g = float(result["G_total"].max())
            st.success(
                f"Simulation complete: {driver_code} · {year} {track} {session_type} · "
                f"{len(result):,} telemetry samples."
            )
            LOGGER.info("Simulation complete for %s %s %s (%s)", driver_code, year, track, session_type)

            metric_cols = st.columns(3)
            metric_cols[0].metric("Validation MAE", f"{mae:.3f} g")
            metric_cols[1].metric("Lap in micro-slip proxy", f"{slip_pct:.1f}%")
            metric_cols[2].metric("Maximum observed G", f"{max_g:.2f} g")
            st.caption("MAE uses a chronological 20% holdout. Slip labels are telemetry-based anomalies, not direct tire measurements.")

            left, right = st.columns(2)
            with left:
                fig = px.line(
                    result, x="Distance", y=["G_total", "Predicted_Grip"],
                    labels={"Distance": "Distance around lap (m)", "value": "Grip vector (g)", "variable": "Series"},
                    title="Observed grip vector vs. inferred grip potential",
                    color_discrete_map={"G_total": "#20c997", "Predicted_Grip": "#f2a541"},
                )
                fig.update_layout(template="plotly_dark", paper_bgcolor="#0b1018", plot_bgcolor="#0b1018")
                st.plotly_chart(fig, use_container_width=True)
            with right:
                scatter = result.copy()
                scatter["Slip state"] = np.where(scatter["Micro_Slip"], "Micro-slip proxy", "Normal")
                fig = px.scatter(
                    scatter, x="X", y="Y", color="Slip state", color_discrete_map={
                        "Normal": "#1769aa", "Micro-slip proxy": "#ff334f"
                    }, hover_data={"Distance": ":.0f", "Speed": ":.1f", "Micro-Slip Delta": ":.3f"},
                    title="Circuit footprint · full-throttle grip deficits",
                )
                fig.update_traces(marker={"size": 5, "opacity": 0.85})
                fig.update_yaxes(scaleanchor="x", scaleratio=1)
                fig.update_layout(template="plotly_dark", paper_bgcolor="#0b1018", plot_bgcolor="#0b1018")
                st.plotly_chart(fig, use_container_width=True)
            if anomaly_enabled:
                st.subheader("Unsupervised anomaly detection")
                st.caption("Flags identify unusual telemetry patterns only; they are not confirmed mistakes. The final 20% of the lap is scored after fitting on the earlier 80%.")
                try:
                    anomaly_rows, anomaly_summary = run_experiment(result, contamination=float(contamination))
                    st.dataframe(anomaly_summary, use_container_width=True, hide_index=True)
                    score_fig = px.scatter(anomaly_rows, x="Distance" if anomaly_rows["Distance"].notna().any() else "Time_s", y="isolation_score", color="isolation_flag", facet_row="feature_set", hover_data=["Speed", "RPM", "statistical_score"], title="Isolation Forest anomaly score on chronological holdout")
                    score_fig.update_layout(template="plotly_dark", paper_bgcolor="#0b1018", plot_bgcolor="#0b1018")
                    st.plotly_chart(score_fig, use_container_width=True)
                    comparison_fig = px.bar(anomaly_summary, x="feature_set", y=["isolationforest_rate_pct", "robust_z_rate_pct"], barmode="group", title="Flagged holdout samples by feature set and method", labels={"value": "Flagged samples (%)", "variable": "Method"})
                    comparison_fig.update_layout(template="plotly_dark", paper_bgcolor="#0b1018", plot_bgcolor="#0b1018")
                    st.plotly_chart(comparison_fig, use_container_width=True)
                except Exception as anomaly_exc:
                    st.warning(f"Anomaly experiment unavailable for this session: {anomaly_exc}")
            with st.expander("Telemetry and inference details"):
                st.dataframe(result, use_container_width=True, hide_index=True)
        except Exception as exc:
            LOGGER.exception("Telemetry simulation failed")
            st.error(f"Simulation failed: {exc}")
else:
    st.info("Choose a session and driver, adjust the simulated conditions, then execute the telemetry simulation.")


