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
from src.saas import (
    authenticate_workspace,
    can_score_rows,
    create_api_key,
    create_workspace,
    fetch_api_keys,
    fetch_scoring_runs,
    fetch_workspace,
    get_plan_limit,
    initialize_database,
    PLAN_LIMITS,
    record_scoring_run,
    revoke_api_key,
    summarize_usage,
    update_workspace_plan,
)


ARTIFACT_PATH = ROOT / "models" / "churn_pipeline.joblib"
DB_PATH = ROOT / "data" / "saas_demo.sqlite3"


@st.cache_resource
def load_artifact() -> dict:
    return joblib.load(ARTIFACT_PATH)


def get_workspace() -> dict | None:
    return st.session_state.get("workspace")


def render_auth() -> dict | None:
    initialize_database(DB_PATH)

    workspace = get_workspace()
    if workspace:
        refreshed = fetch_workspace(DB_PATH, workspace["id"]) or workspace
        st.session_state["workspace"] = refreshed
        with st.sidebar:
            st.markdown("### Workspace")
            st.write(refreshed["name"])
            st.caption(f"{refreshed['plan']} plan")
            if st.button("Sign out", use_container_width=True):
                st.session_state.pop("workspace", None)
                st.rerun()
        return refreshed

    st.title("Customer Churn AI")
    st.caption("Retention analytics workspace for churn scoring, campaign planning, and monitoring.")

    auth_tab, signup_tab = st.tabs(["Sign in", "Create workspace"])

    with auth_tab:
        st.info("Demo account: demo@churnai.com / demo123")
        email = st.text_input("Email", value="demo@churnai.com")
        password = st.text_input("Password", value="demo123", type="password")
        if st.button("Sign in", type="primary"):
            workspace = authenticate_workspace(DB_PATH, email, password)
            if workspace:
                st.session_state["workspace"] = workspace
                st.rerun()
            st.error("Invalid email or password.")

    with signup_tab:
        workspace_name = st.text_input("Workspace name", value="Acme Telecom")
        signup_email = st.text_input("Work email")
        signup_password = st.text_input("Create password", type="password")
        if st.button("Create workspace"):
            if not workspace_name.strip() or not signup_email.strip() or not signup_password:
                st.error("Workspace name, email, and password are required.")
            else:
                try:
                    workspace = create_workspace(DB_PATH, workspace_name, signup_email, signup_password)
                except Exception:
                    st.error("That email already has a workspace.")
                else:
                    st.session_state["workspace"] = workspace
                    st.rerun()

    return None


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


def show_workspace_dashboard(artifact: dict, workspace: dict) -> None:
    runs = fetch_scoring_runs(DB_PATH, workspace["id"])
    usage = summarize_usage(runs)
    plan_limit = get_plan_limit(workspace["plan"])
    metrics = artifact["metrics"]
    roi = metrics["roi"]

    st.subheader("Workspace Overview")
    metric_cols = st.columns(4)
    metric_cols[0].metric("Scoring runs", f"{usage['runs']:,}")
    metric_cols[1].metric("Rows scored", f"{usage['rows_scored']:,}")
    metric_cols[2].metric("Customers to contact", f"{usage['contacts_recommended']:,}")
    metric_cols[3].metric("Avg. churn risk", f"{usage['avg_churn_probability']:.1%}")
    st.progress(min(int(usage["rows_scored"]) / plan_limit["monthly_rows"], 1.0))
    st.caption(f"Monthly quota: {usage['rows_scored']:,} of {plan_limit['monthly_rows']:,} rows used.")

    st.subheader("Retention Operating Model")
    model_cols = st.columns(4)
    model_cols[0].metric("Business threshold", f"{artifact['threshold']:.0%}")
    model_cols[1].metric("Recall", f"{metrics['recall']:.1%}")
    model_cols[2].metric("ROC AUC", f"{metrics['roc_auc']:.3f}")
    model_cols[3].metric("Validation net value", f"${roi['net_value']:,.0f}")

    st.subheader("Saved Campaign Runs")
    if runs.empty:
        st.info("Save a single prediction or batch scoring run to start building workspace history.")
        return

    display = runs.copy()
    display["avg_churn_probability"] = display["avg_churn_probability"].map(lambda value: f"{value:.1%}")
    display["threshold"] = display["threshold"].map(lambda value: f"{value:.0%}")
    st.dataframe(display, use_container_width=True, hide_index=True)


def show_prediction_result(artifact: dict, profile: pd.DataFrame, workspace: dict) -> None:
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
    if st.button("Save prediction to workspace"):
        runs = fetch_scoring_runs(DB_PATH, workspace["id"])
        allowed, message = can_score_rows(workspace, runs, len(scored))
        if allowed:
            record_scoring_run(DB_PATH, workspace["id"], "single_prediction", scored, artifact["threshold"])
            st.success(f"Saved this prediction to the workspace history. {message}")
        else:
            st.error(message)

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


def show_batch_scoring(artifact: dict, workspace: dict) -> None:
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
    if st.button("Save batch as campaign run"):
        runs = fetch_scoring_runs(DB_PATH, workspace["id"])
        allowed, message = can_score_rows(workspace, runs, len(scored))
        if allowed:
            record_scoring_run(DB_PATH, workspace["id"], "batch_campaign", scored, artifact["threshold"])
            st.success(f"Saved this batch run to the workspace history. {message}")
        else:
            st.error(message)

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

    upload = st.file_uploader(
        "Upload monitoring CSV",
        type=["csv"],
        key="monitoring_upload",
    )

    if upload is None:
        st.info(
            "Use the sample CSV from the Batch Prediction tab to test this monitoring view."
        )
        return

    uploaded_df = pd.read_csv(upload)

    # Validate schema
    missing_cols = set(MODEL_FEATURES) - set(uploaded_df.columns)

    if missing_cols:
        st.error(
            f"Uploaded file is missing {len(missing_cols)} required model columns."
        )

        st.write("Missing columns:")
        st.code("\n".join(sorted(missing_cols)))

        st.info(
            "Upload a churn dataset or the sample CSV from the Batch Prediction tab."
        )
        return

    drift = compare_to_training_profile(
        uploaded_df,
        artifact["training_profile"],
    )

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


def show_billing_and_api(workspace: dict) -> None:
    runs = fetch_scoring_runs(DB_PATH, workspace["id"])
    usage = summarize_usage(runs)
    current_limit = get_plan_limit(workspace["plan"])

    st.subheader("Subscription")
    plan_cols = st.columns(len(PLAN_LIMITS))
    for column, (plan_name, details) in zip(plan_cols, PLAN_LIMITS.items()):
        with column:
            st.markdown(f"### {plan_name}")
            st.metric("Monthly rows", f"{details['monthly_rows']:,}")
            st.metric("Seats", details["seats"])
            st.metric("Price", f"${details['price']}/mo")
            disabled = plan_name == workspace["plan"]
            if st.button(
                "Current plan" if disabled else f"Switch to {plan_name}",
                key=f"plan_{plan_name}",
                disabled=disabled,
                use_container_width=True,
            ):
                updated = update_workspace_plan(DB_PATH, workspace["id"], plan_name)
                st.session_state["workspace"] = updated
                st.success(f"Workspace moved to the {plan_name} plan.")
                st.rerun()

    st.subheader("Usage Controls")
    usage_cols = st.columns(3)
    usage_cols[0].metric("Current plan", workspace["plan"])
    usage_cols[1].metric("Rows used", f"{usage['rows_scored']:,}")
    usage_cols[2].metric("Rows remaining", f"{max(current_limit['monthly_rows'] - int(usage['rows_scored']), 0):,}")

    st.subheader("API Keys")
    st.caption("Use API keys for CRM or backend integrations. Store generated keys securely; only the prefix is shown later.")
    if st.button("Generate API key"):
        raw_key = create_api_key(DB_PATH, workspace["id"])
        st.success("API key generated. Copy it now; it will not be shown again.")
        st.code(raw_key, language="text")

    keys = fetch_api_keys(DB_PATH, workspace["id"])
    if keys.empty:
        st.info("No API keys have been created for this workspace.")
        return

    display = keys.copy()
    display["status"] = display["revoked_at"].map(lambda value: "Active" if pd.isna(value) else "Revoked")
    st.dataframe(display[["id", "key_prefix", "created_at", "status"]], use_container_width=True, hide_index=True)

    active_keys = display[display["status"] == "Active"]
    if not active_keys.empty:
        selected_key_id = st.selectbox("Revoke active key", active_keys["id"].tolist())
        if st.button("Revoke selected key"):
            revoke_api_key(DB_PATH, workspace["id"], int(selected_key_id))
            st.success("API key revoked.")
            st.rerun()


def main() -> None:
    st.set_page_config(page_title="Customer Churn AI", layout="wide")

    workspace = render_auth()
    if workspace is None:
        return

    artifact = load_artifact()
    st.title("Customer Churn AI")
    st.caption("SaaS retention analytics for churn scoring, campaign decisions, explanations, and monitoring.")

    tab_workspace, tab_single, tab_batch, tab_monitoring, tab_billing, tab_model = st.tabs(
        ["Workspace", "Single Prediction", "Batch Prediction", "Monitoring", "Billing & API", "Model Details"]
    )

    with tab_workspace:
        show_workspace_dashboard(artifact, workspace)

    with tab_single:
        form_col, result_col = st.columns([0.52, 0.48], gap="large")
        with form_col:
            profile = build_profile_form(artifact["feature_options"])
        with result_col:
            show_prediction_result(artifact, profile, workspace)

    with tab_batch:
        show_batch_scoring(artifact, workspace)

    with tab_monitoring:
        show_monitoring(artifact)

    with tab_billing:
        show_billing_and_api(workspace)

    with tab_model:
        show_model_details(artifact)


if __name__ == "__main__":
    main()
