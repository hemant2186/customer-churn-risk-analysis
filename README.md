# 📉 Customer Churn Risk Analysis  
### Cost-Sensitive Machine Learning Pipeline with Risk Segmentation

---

## 📌 Project Overview

Customer churn occurs when customers discontinue a subscription-based service.  
In subscription-driven businesses, **missing a churn-prone customer is significantly more costly than incorrectly flagging a loyal one**.

This project implements an **end-to-end customer churn risk analysis pipeline** that frames churn prediction as a **probability-based, cost-sensitive decision problem**, rather than a simple binary classification task.

The project reflects how churn models are built, evaluated, and prepared for deployment in real-world data science and analytics teams.

---

## 🎯 Objectives

- Build a **clean and reproducible machine learning pipeline**  
- Optimize churn prediction for **business impact**, prioritizing recall  
- Generate **customer-level churn probabilities**  
- Convert predictions into **actionable risk segments**  
- Save trained artifacts for **deployment-ready inference**

---

## 📊 Dataset

- **Source:** Telco Customer Churn Dataset  
- **Records:** 7,043 customers  
- **Features Include:**
  - Customer demographics  
  - Service usage and subscriptions  
  - Contract type and billing information  
  - Tenure and payment behavior  
- **Target Variable:**  
  - `Churn` (Yes / No)

---

## 🛠️ Tech Stack

- **Programming Language:** Python  
- **Data Processing:** Pandas, NumPy  
- **Machine Learning:** Scikit-learn  
- **Modeling:** Cost-Sensitive Logistic Regression  
- **Model Serialization:** Joblib  

---

## 🧠 Methodology

### 1️⃣ Data Preparation
- Loaded and validated the dataset  
- Removed non-informative identifiers  
- Encoded churn target as a binary variable  
- Ensured consistent and reproducible preprocessing  

---

### 2️⃣ Feature Engineering
- Applied one-hot encoding to categorical variables  
- Standardized numerical features using `StandardScaler`  
- Preserved the feature schema for inference consistency  

---

### 3️⃣ Cost-Sensitive Model Training
- Trained a **Logistic Regression model with class weighting**  
- Penalized false negatives to reduce missed churn cases  
- Implemented preprocessing and modeling using a **Scikit-learn Pipeline**

---

### 4️⃣ Model Evaluation
- Evaluated model performance using:
  - Recall (primary metric)
  - Precision and F1-score
  - Confusion matrix  
- Achieved improved churn recall while maintaining acceptable precision  

---

### 5️⃣ Churn Probability & Risk Segmentation
- Generated **churn probabilities** for each customer  
- Converted probabilities into **actionable risk tiers**:
  - **High Risk**
  - **Medium Risk**
  - **Low Risk**  
- Enables targeted retention strategies instead of hard labels  

---

### 6️⃣ Model Serialization & Deployment Readiness
- Saved trained model pipeline  
- Saved feature column schema  
- Artifacts are designed for **inference-only deployment** (e.g., Streamlit)

---

## 🚀 Key Takeaways

- **Recall-focused optimization** is critical in churn prediction problems  
- Probabilistic outputs provide more business value than binary predictions  
- Risk segmentation bridges ML outputs and decision-making  
- Clean pipelines and saved artifacts are essential for production workflows  

---

## 🧩 Deployment Architecture

- Model training and experimentation are performed in Jupyter Notebook  
- Trained artifacts are serialized using Joblib  
- A Streamlit application can load these artifacts for real-time churn prediction  

This separation mirrors **real-world machine learning deployment best practices**.

---

## 📜 License

This project is licensed under the **MIT License**.

---

## 👤 Author

**Hemant Kumar**  
B.Tech | Aspiring Data Scientist  

- 🔗 LinkedIn: https://www.linkedin.com/in/hemant-kumar-171472210  
- 💻 GitHub: https://github.com/hemant2186  
- 📧 Email: hemantkumar90089h@gmail.com  

---
