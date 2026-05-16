from __future__ import annotations

import joblib
import pandas as pd
import streamlit as st


ARTIFACT_PATH = "models/churn_pipeline.joblib"


@st.cache_resource
def load_artifact() -> dict:
    return joblib.load(ARTIFACT_PATH)


def risk_label(probability: float) -> tuple[str, str]:
    if probability >= 0.65:
        return "High risk", "Prioritize proactive retention outreach."
    if probability >= 0.35:
        return "Medium risk", "Send targeted engagement and monitor usage."
    return "Low risk", "Maintain service quality and normal lifecycle messaging."


artifact = load_artifact()
model = artifact["model"]
threshold = artifact["threshold"]
options = artifact["feature_options"]

st.set_page_config(page_title="Customer Churn Risk Scoring", layout="wide")

st.title("Customer Churn Risk Scoring")
st.caption("Cost-sensitive churn prediction with business-threshold recommendations.")

left, right = st.columns([0.42, 0.58], gap="large")

with left:
    st.subheader("Customer Profile")

    tenure = st.slider("Tenure in months", 0, 72, 12)
    monthly_charges = st.slider("Monthly charges", 0.0, 150.0, 70.0, step=1.0)
    total_charges = st.number_input("Total charges", min_value=0.0, max_value=10000.0, value=1000.0, step=50.0)
    senior = st.selectbox("Senior citizen", [0, 1], format_func=lambda value: "Yes" if value else "No")

    st.subheader("Account Details")
    contract = st.selectbox("Contract", options["Contract"])
    payment = st.selectbox("Payment method", options["PaymentMethod"])
    paperless = st.selectbox("Paperless billing", options["PaperlessBilling"])
    gender = st.selectbox("Gender", options["gender"])
    partner = st.selectbox("Partner", options["Partner"])
    dependents = st.selectbox("Dependents", options["Dependents"])

    st.subheader("Services")
    phone = st.selectbox("Phone service", options["PhoneService"])
    multiple_lines = st.selectbox("Multiple lines", options["MultipleLines"])
    internet = st.selectbox("Internet service", options["InternetService"])
    online_security = st.selectbox("Online security", options["OnlineSecurity"])
    online_backup = st.selectbox("Online backup", options["OnlineBackup"])
    device_protection = st.selectbox("Device protection", options["DeviceProtection"])
    tech_support = st.selectbox("Tech support", options["TechSupport"])
    streaming_tv = st.selectbox("Streaming TV", options["StreamingTV"])
    streaming_movies = st.selectbox("Streaming movies", options["StreamingMovies"])

profile = pd.DataFrame(
    [
        {
            "SeniorCitizen": senior,
            "tenure": tenure,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "gender": gender,
            "Partner": partner,
            "Dependents": dependents,
            "PhoneService": phone,
            "MultipleLines": multiple_lines,
            "InternetService": internet,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
        }
    ]
)

probability = float(model.predict_proba(profile)[0, 1])
decision = probability >= threshold
label, recommendation = risk_label(probability)

with right:
    st.subheader("Prediction")

    metric_cols = st.columns(3)
    metric_cols[0].metric("Churn probability", f"{probability:.1%}")
    metric_cols[1].metric("Business threshold", f"{threshold:.0%}")
    metric_cols[2].metric("Decision", "Contact" if decision else "Monitor")

    st.progress(min(max(probability, 0.0), 1.0))
    st.write(f"**Risk tier:** {label}")
    st.write(f"**Recommended action:** {recommendation}")

    st.subheader("Model Quality")
    metrics = artifact["metrics"]
    quality_cols = st.columns(4)
    quality_cols[0].metric("Recall", f"{metrics['recall']:.1%}")
    quality_cols[1].metric("Precision", f"{metrics['precision']:.1%}")
    quality_cols[2].metric("ROC AUC", f"{metrics['roc_auc']:.3f}")
    quality_cols[3].metric("Accuracy", f"{metrics['accuracy']:.1%}")

    st.subheader("Business Context")
    roi = metrics["roi"]
    st.dataframe(
        pd.DataFrame(
            [
                ["Customers contacted", roi["contacted_customers"]],
                ["True churners found", roi["true_positives"]],
                ["Missed churners", roi["false_negatives"]],
                ["Estimated net value", f"${roi['net_value']:,.0f}"],
            ],
            columns=["Metric", "Validation result"],
        ),
        hide_index=True,
        use_container_width=True,
    )

    with st.expander("View encoded model input"):
        st.dataframe(profile, use_container_width=True, hide_index=True)
