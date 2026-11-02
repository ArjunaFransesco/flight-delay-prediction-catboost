import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Commercial Flight Departure Delay Predictor & Weather Impact Model",
    page_icon="🤖",
    layout="wide"
)

st.title("🎯 Commercial Flight Departure Delay Predictor & Weather Impact Model")
st.markdown("**Domain**: `Aviation / Transportation ML` | **Tech Stack**: `LightGBM, Scikit-Learn, Streamlit`")
st.markdown("**Author**: [Arjuna Fransesco](https://github.com/ArjunaFransesco) | **GitHub**: [Portfolio Repositories](https://github.com/ArjunaFransesco?tab=repositories)")
st.markdown("---")

col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("⚙️ Domain Input Telemetry")
    departure_lead_time_min = st.slider("Departure Lead Time Min", float(5.0), float(180.0), float(45.0))
    airport_congestion_index = st.slider("Airport Congestion Index", float(0.1), float(0.99), float(0.65))
    crosswind_speed_knots = st.slider("Crosswind Speed Knots", float(0.0), float(45.0), float(12.0))
    visibility_statute_miles = st.slider("Visibility Statute Miles", float(0.5), float(10.0), float(9.2))
    aircraft_age_years = st.slider("Aircraft Age Years", float(1.0), float(25.0), float(8.5))
    prior_leg_delay_min = st.slider("Prior Leg Delay Min", float(0.0), float(120.0), float(14.0))

with col2:
    st.subheader("🔮 Predictive Model Inference")
    model_path = os.path.join(os.path.dirname(__file__), "models/model_pipeline.joblib")
    scaler_path = os.path.join(os.path.dirname(__file__), "models/scaler.joblib")
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        input_df = pd.DataFrame([{"departure_lead_time_min": departure_lead_time_min, "airport_congestion_index": airport_congestion_index, "crosswind_speed_knots": crosswind_speed_knots, "visibility_statute_miles": visibility_statute_miles, "aircraft_age_years": aircraft_age_years, "prior_leg_delay_min": prior_leg_delay_min}])
        input_scaled = scaler.transform(input_df)
        pred = model.predict(input_scaled)[0]
        
        st.markdown("#### Real-Time Prediction Output")
        st.info(f"Predicted `arrival_delay_minutes`: **{pred}**")
        
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(input_scaled)[0]
            st.progress(float(probs[1]) if len(probs) > 1 else float(probs[0]))
            st.caption(f"Confidence Probability Score: **{np.max(probs):.2%}**")
    else:
        st.warning("Model or Scaler artifact not found in models/ directory.")

st.markdown("---")
st.markdown("### 📊 Benchmark Metrics")
metrics_path = os.path.join(os.path.dirname(__file__), "reports/metrics.json")
if os.path.exists(metrics_path):
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics_data = json.load(f)
    st.json(metrics_data)
