import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Risk Analysis",
    page_icon="📉",
    layout="centered"
)

st.title("📉 Customer Churn Risk Analysis")
st.write(
    """
    This application predicts **customer churn risk**
    using a pretrained, cost-sensitive machine learning model.

    The output supports **data-driven retention strategies**.
    """
)

# --------------------------------------------------
# Load Model Artifacts
# --------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("models/churn_model.pkl")
    feature_columns = joblib.load("models/feature_columns.pkl")
    return model, feature_columns

model, feature_columns = load_artifacts()

# --------------------------------------------------
# Sidebar Inputs
# --------------------------------------------------
st.sidebar.header("🧾 Customer Profile")

tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.sidebar.slider("Monthly Charges", 20, 150, 70)
total_charges = st.sidebar.slider("Total Charges", 20, 10000, 1000)
contract = st.sidebar.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)
internet_service = st.sidebar.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

# --------------------------------------------------
# Prepare Input Data
# --------------------------------------------------
input_df = pd.DataFrame([{
    "tenure": tenure,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
    "Contract": contract,
    "InternetService": internet_service
}])

input_df = pd.get_dummies(input_df)
input_df = input_df.reindex(columns=feature_columns, fill_value=0)

# --------------------------------------------------
# Prediction
# --------------------------------------------------
if st.button("🔍 Assess Churn Risk"):
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("📊 Churn Risk Assessment")
    st.write(f"**Probability of Churn:** `{probability:.2f}`")

    if probability >= 0.6:
        st.error("🔴 High Risk — Immediate retention action recommended")
        st.write("Suggested action: proactive outreach, personalized offers")

    elif probability >= 0.3:
        st.warning("🟡 Medium Risk — Monitor and engage")
        st.write("Suggested action: loyalty incentives, usage nudges")

    else:
        st.success("🟢 Low Risk — Customer likely to stay")
        st.write("Suggested action: maintain service quality")

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.markdown(
    """
    **Author:** Hemant Kumar  
    *B.Tech | Aspiring Data Scientist*  

    ⚠️ This application performs inference only.
    Model training is handled offline in the notebook.
    """
)
