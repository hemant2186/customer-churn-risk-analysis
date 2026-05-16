from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

from src.business import risk_label
from src.config import CATEGORICAL_FEATURES, MODEL_FEATURES, NUMERIC_FEATURES
from src.inference import explain_prediction, score_customers
from src.monitoring import compare_to_training_profile


ARTIFACT_PATH = ROOT / "models" / "churn_pipeline.joblib"


@st.cache_resource
def load_artifact() -> dict:
    return joblib.load(ARTIFACT_PATH)


def build_profile_form(options: dict) -> pd.DataFrame:
    left, middle = st.columns(2)

    with left:
        st.subheader("Customer Profile")
        tenure = st.slider("Tenure in months", 0, 72, 12)
        monthly_charges = st.slider("Monthly charges", 0.0, 150.0, 70.0, step=1.0)
        total_charges = st.number_input("Total charges", min_value=0.0, max_value=10000.0, value=1000.0, step=50.0)
        senior = st.selectbox("Senior citizen", [0, 1], format_func=lambda value: "Yes" if value else "No")
        gender = st.selectbox("Gender", options["gender"])
        partner = st.selectbox("Partner", options["Partner"])
        dependents = st.selectbox("Dependents", options["Dependents"])

    with middle:
        st.subheader("Account and Services")
        contract = st.selectbox("Contract", options["Contract"])
        payment = st.selectbox("Payment method", options["PaymentMethod"])
        paperless = st.selectbox("Paperless billing", options["PaperlessBilling"])
        phone = st.selectbox("Phone service", options["PhoneService"])
        multiple_lines = st.selectbox("Multiple lines", options["MultipleLines"])
        internet = st.selectbox("Internet service", options["InternetService"])
        online_security = st.selectbox("Online security", options["OnlineSecurity"])
        online_backup = st.selectbox("Online backup", options["OnlineBackup"])
        device_protection = st.selectbox("Device protection", options["DeviceProtection"])
        tech_support = st.selectbox("Tech support", options["TechSupport"])
        streaming_tv = st.selectbox("Streaming TV", options["StreamingTV"])
        streaming_movies = st.selectbox("Streaming movies", options["StreamingMovies"])

    return pd.DataFrame(
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


def show_prediction_result(artifact: dict, profile: pd.DataFrame) -> None:
    scored = score_customers(artifact, profile)
    probability = float(scored.loc[0, "churn_probability"])
    decision = scored.loc[0, "recommended_decision"]
    label, recommendation = risk_label(probability)
    metrics = artifact["metrics"]

    st.subheader("Prediction")
    metric_cols = st.columns(3)
    metric_cols[0].metric("Churn probability", f"{probability:.1%}")
    metric_cols[1].metric("Business threshold", f"{artifact['threshold']:.0%}")
    metric_cols[2].metric("Decision", decision)

    st.progress(min(max(probability, 0.0), 1.0))
    st.write(f"**Risk tier:** {label}")
    st.write(f"**Recommended action:** {recommendation}")

    st.subheader("Local Explanation")
    explanation = explain_prediction(artifact, profile)
    if explanation.empty:
        st.info("Local explanations are available for the selected linear model artifact.")
    else:
        chart_data = explanation.set_index("Feature")["Contribution"]
        st.bar_chart(chart_data)
        st.dataframe(explanation, use_container_width=True, hide_index=True)

    st.subheader("Model Quality")
    quality_cols = st.columns(4)
    quality_cols[0].metric("Recall", f"{metrics['recall']:.1%}")
    quality_cols[1].metric("Precision", f"{metrics['precision']:.1%}")
    quality_cols[2].metric("ROC AUC", f"{metrics['roc_auc']:.3f}")
    quality_cols[3].metric("Accuracy", f"{metrics['accuracy']:.1%}")


def show_batch_scoring(artifact: dict) -> None:
    st.subheader("Batch Prediction")
    st.write("Upload a CSV with the model input columns, then download churn scores for every row.")

    sample = pd.DataFrame(artifact["sample_input"])[MODEL_FEATURES]
    st.download_button(
        "Download sample input CSV",
        sample.to_csv(index=False).encode("utf-8"),
        file_name="sample_churn_input.csv",
        mime="text/csv",
    )

    upload = st.file_uploader("Upload customer CSV", type=["csv"])
    if upload is None:
        st.dataframe(sample.head(5), use_container_width=True, hide_index=True)
        return

    uploaded_df = pd.read_csv(upload)
    try:
        scored = score_customers(artifact, uploaded_df)
    except ValueError as error:
        st.error(str(error))
        return

    summary_cols = st.columns(3)
    summary_cols[0].metric("Rows scored", f"{len(scored):,}")
    summary_cols[1].metric("Customers to contact", f"{(scored['recommended_decision'] == 'Contact').sum():,}")
    summary_cols[2].metric("Average churn risk", f"{scored['churn_probability'].mean():.1%}")

    st.dataframe(
        scored.sort_values("churn_probability", ascending=False),
        use_container_width=True,
        hide_index=True,
    )
    st.download_button(
        "Download scored CSV",
        scored.to_csv(index=False).encode("utf-8"),
        file_name="churn_scores.csv",
        mime="text/csv",
    )


def show_monitoring(artifact: dict) -> None:
    st.subheader("Data Drift Check")
    st.write("Upload a customer CSV to compare its feature profile with the training baseline.")
    upload = st.file_uploader("Upload monitoring CSV", type=["csv"], key="monitoring_upload")

    if upload is None:
        st.info("Use the sample CSV from the Batch Prediction tab to test this monitoring view.")
        return

    uploaded_df = pd.read_csv(upload)
    drift = compare_to_training_profile(uploaded_df, artifact["training_profile"])
    review_count = int((drift["Status"] == "Review").sum())
    st.metric("Features flagged for review", review_count)
    st.dataframe(drift, use_container_width=True, hide_index=True)


def show_model_details(artifact: dict) -> None:
    metrics = artifact["metrics"]
    roi = metrics["roi"]

    st.subheader("Validation Metrics")
    st.dataframe(
        pd.DataFrame(
            [
                ["Selected model", artifact["model_name"]],
                ["Recall", f"{metrics['recall']:.1%}"],
                ["Precision", f"{metrics['precision']:.1%}"],
                ["ROC AUC", f"{metrics['roc_auc']:.3f}"],
                ["Accuracy", f"{metrics['accuracy']:.1%}"],
                ["Business threshold", f"{artifact['threshold']:.0%}"],
                ["Estimated net value", f"${roi['net_value']:,.0f}"],
            ],
            columns=["Metric", "Value"],
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Required Batch Columns")
    st.code(", ".join(MODEL_FEATURES), language="text")


st.set_page_config(page_title="Customer Churn Risk Scoring", layout="wide")

artifact = load_artifact()
st.title("Customer Churn Risk Scoring")
st.caption("Interactive churn prediction, batch scoring, local explanations, and drift monitoring.")

tab_single, tab_batch, tab_monitoring, tab_model = st.tabs(
    ["Single Prediction", "Batch Prediction", "Monitoring", "Model Details"]
)

with tab_single:
    form_col, result_col = st.columns([0.52, 0.48], gap="large")
    with form_col:
        profile = build_profile_form(artifact["feature_options"])
    with result_col:
        show_prediction_result(artifact, profile)

with tab_batch:
    show_batch_scoring(artifact)

with tab_monitoring:
    show_monitoring(artifact)

with tab_model:
    show_model_details(artifact)
