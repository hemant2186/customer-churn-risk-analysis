# End-to-End Customer Churn Analysis  
### Cost-Sensitive Modeling, Risk Segmentation & Cohort-Based Retention Strategy

## 📌 Project Overview
Customer churn refers to customers discontinuing a subscription-based service.  
In real-world businesses, predicting churn alone is insufficient unless predictions are translated into **actionable retention strategies**.

This project simulates how churn is handled in **real-world companies like FAANG**, moving beyond basic prediction to business-aware decision-making.

---

## 🎯 Objectives
- Optimize churn prediction models for **business cost**, not just accuracy
- Segment customers into **actionable risk tiers**
- Analyze churn behavior across **customer lifecycle cohorts**
- Translate model outputs into **data-driven retention strategies**

---

## 📊 Dataset
- **Source:** Telco Customer Churn Dataset  
- **Records:** ~7,000 customers  
- **Features:** Demographics, service usage, billing, tenure  
- **Target Variable:** Churn (Yes / No)

---

## 🛠️ Tech Stack
- Python  
- Pandas, NumPy  
- Scikit-learn  
- Matplotlib, Seaborn  

---

## 🧠 Methodology (Phase-wise)

### Phase 1 — Data Understanding & Baseline Modeling
- Data cleaning and preprocessing
- Feature encoding and scaling
- Baseline Logistic Regression model
- Identified low recall for churned customers as a business risk

### Phase 2 — Cost-Sensitive Modeling
- Implemented class-weighted Random Forest
- Prioritized recall to reduce missed churn cases
- Compared models using recall–precision trade-offs

### Phase 3 — Risk-Based Customer Segmentation
- Generated churn probabilities instead of binary predictions
- Segmented customers into Low, Medium, and High risk tiers
- Designed targeted retention actions for each segment

### Phase 4 — Cohort-Based Churn Analysis
- Grouped customers by tenure cohorts
- Identified highest churn during the first 6 months
- Highlighted onboarding as a critical retention phase

### Phase 5 — Insights & Business Recommendations
- High-risk customers exhibit low tenure and high monthly charges
- Majority of churn occurs during early customer lifecycle
- Retention efforts should prioritize early-stage, high-risk users

### Phase 6 — Production & Monitoring Considerations
- Monitor data drift and model performance degradation
- Periodic retraining and threshold recalibration
- Evaluate retention strategies through A/B testing

---

## 🚀 Key Takeaways
- Business-aware optimization is more valuable than accuracy alone
- Risk segmentation bridges ML predictions and product decisions
- Cohort analysis reveals lifecycle-specific churn patterns

---

## Deployment Architecture

Model training and experimentation are performed in the Jupyter notebook.
The trained model and feature schema are serialized and loaded by a
Streamlit application that performs inference-only prediction.

This separation mirrors real-world machine learning deployment workflows.

---

## License
This project is licensed under the MIT License.

---

## 👤 Author & Contact

**Hemant Kumar**  
B.Tech | Aspiring Data Scientist  

- 🔗 **LinkedIn:** https://www.linkedin.com/in/hemant-kumar-171472210  
- 💻 **GitHub:** https://github.com/hemant2186  
- 📧 **Email:** hemantkumar90089h@gmail.com  

---
