import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ==================================================
# Configuration
# ==================================================
APP_TITLE = "Customer Churn Risk Analysis"
APP_ICON = "📉"

MODEL_PATH = Path("models/churn_model.pkl")
FEATURES_PATH = Path("models/feature_columns.pkl")

HIGH_RISK_THRESHOLD = 0.60
MEDIUM_RISK_THRESHOLD = 0.30

# ==================================================
# Page Configuration
# ==================================================
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="centered"
)

st.title(f"{APP_ICON} {APP_TITLE}")
st.markdown(
    """
    This application predicts **customer churn risk** using a
    pretrained, **cost-sensitive machine learning model**.

    Predictions are **probabilistic** and designed to support
    **data-driven retention strategies**.
    """
)

# ==================================================
# Utility Functions
# ==================================================
@st.cache_resource
def load_artifacts():
    """Load trained model and feature schema."""
    if not MODEL_PATH.exists() or not FEATURES_PATH.exists():
        raise FileNotFoundError("Model artifacts not found.")

    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURES_PATH)
    return model, feature_columns


def prepare_input(data: dict, feature_columns: list) -> pd.DataFrame:
    """
    Prepare user input for inference:
    - Convert to DataFrame
    - Apply one-hot encoding
    - Align feature schema
    """
    df = pd.DataFrame([data])
    df = pd.get_dummies(df)
    df = df.reindex(columns=feature_columns, fill_value=0)
    return df


def assess_risk(probability: float) -> tuple[str, str]:
    """Map churn probability to risk level and message."""
    if probability >= HIGH_RISK_THRESHOLD:
        return (
            "High Risk",
            "🔴 Immediate retention action recommended.\n\n"
            "Suggested actions: proactive outreach, personalized offers."
        )
    elif probability >= MEDIUM_RISK_THRESHOLD:
        return (
            "Medium Risk",
            "🟡 Monitor closely and engage.\n\n"
            "Suggested actions: loyalty incentives, usage nudges."
        )
    else:
        return (
            "Low Risk",
            "🟢 Customer likely to stay.\n\n"
            "Suggested actions: maintain service quality."
        )


# ==================================================
# Load Model Artifacts
# ==================================================
try:
    model, feature_columns = load_artifacts()
except Exception as e:
    st.error("❌ Failed to load model artifacts.")
    st.stop()

# ==================================================
# Sidebar Inputs
# ==================================================
st.sidebar.header("🧾 Customer Profile")

tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.sidebar.slider("Monthly Charges", 20.0, 150.0, 70.0)
total_charges = st.sidebar.slider("Total Charges", 20.0, 10000.0, 1000.0)

contract = st.sidebar.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)

internet_service = st.sidebar.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

# ==================================================
# Prediction
# ==================================================
if st.button("🔍 Assess Churn Risk"):
    input_data = {
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Contract": contract,
        "InternetService": internet_service
    }

    input_df = prepare_input(input_data, feature_columns)

    try:
        probability = model.predict_proba(input_df)[0][1]
        risk_level, recommendation = assess_risk(probability)

        st.subheader("📊 Churn Risk Assessment")
        st.metric(
            label="Probability of Churn",
            value=f"{probability:.2f}"
        )

        if risk_level == "High Risk":
            st.error(recommendation)
        elif risk_level == "Medium Risk":
            st.warning(recommendation)
        else:
            st.success(recommendation)

    except Exception:
        st.error("❌ Prediction failed. Please verify input values.")

# ==================================================
# Footer
# ==================================================
st.markdown("---")
st.markdown(
    """
    **Author:** Hemant Kumar  
    *B.Tech | Aspiring Data Scientist*

    ⚠️ This application performs **inference only**.
    Model training and evaluation are handled offline.
    """
)
