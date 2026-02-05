import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

# ---------------------------------------------------
# Page config
# ---------------------------------------------------

st.set_page_config(
    page_title="Predictive Maintenance Engine Risk Predictor",
    layout="centered"
)

# ---------------------------------------------------
# Load Model from Hugging Face
# ---------------------------------------------------

REPO_ID = "subratm62/predictive-maintenance"
MODEL_FILE = "predictive_maintenance_pipeline.joblib"

@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=MODEL_FILE
    )
    return joblib.load(model_path)

model = load_model()

# Classification threshold (your tuned value)
classification_threshold = 0.50

# ---------------------------------------------------
# UI Header
# ---------------------------------------------------

st.title("🔧 Predictive Maintenance — Engine Failure Risk")
st.write(
    """
    Enter live engine sensor readings to estimate **failure risk**.
    This tool supports proactive maintenance decisions.
    """
)

st.markdown("---")

# ---------------------------------------------------
# Sensor Inputs
# ---------------------------------------------------

st.subheader("Engine Sensor Inputs")

col1, col2 = st.columns(2)

with col1:
    engine_rpm = st.number_input(
        "Engine RPM",
        min_value=0.0,
        max_value=3000.0,
        value=750.0
    )

    lub_oil_pressure = st.number_input(
        "Lub Oil Pressure",
        min_value=0.0,
        max_value=10.0,
        value=3.0
    )

    fuel_pressure = st.number_input(
        "Fuel Pressure",
        min_value=0.0,
        max_value=25.0,
        value=6.0
    )

with col2:
    coolant_pressure = st.number_input(
        "Coolant Pressure",
        min_value=0.0,
        max_value=10.0,
        value=2.0
    )

    lub_oil_temp = st.number_input(
        "Lub Oil Temperature",
        min_value=60.0,
        max_value=120.0,
        value=77.0
    )

    coolant_temp = st.number_input(
        "Coolant Temperature",
        min_value=50.0,
        max_value=200.0,
        value=78.0
    )

st.markdown("---")

# ---------------------------------------------------
# Prepare input dataframe
# ---------------------------------------------------

input_data = pd.DataFrame([{
    "Engine rpm": engine_rpm,
    "Lub oil pressure": lub_oil_pressure,
    "Fuel pressure": fuel_pressure,
    "Coolant pressure": coolant_pressure,
    "lub oil temp": lub_oil_temp,
    "Coolant temp": coolant_temp
}])

# ---------------------------------------------------
# Prediction
# ---------------------------------------------------

if st.button("🔍 Predict Failure Risk"):

    probability = model.predict_proba(input_data)[0, 1]
    prediction = int(probability >= classification_threshold)

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(
            "⚠ HIGH FAILURE RISK — Maintenance inspection recommended."
        )
    else:
        st.success(
            "✅ Engine operating within normal range."
        )

    st.write(f"**Failure Probability:** {probability:.4f}")
    st.write(f"**Decision Threshold:** {classification_threshold:.2f}")

    # Business interpretation
    if probability > 0.75:
        st.warning("Critical condition — immediate inspection advised.")
    elif probability > 0.50:
        st.info("Moderate risk — schedule maintenance soon.")
    else:
        st.write("Low operational risk detected.")

st.markdown("---")

st.caption(
    "Model hosted on Hugging Face | Experiment tracking via MLflow | Built with Streamlit"
)
