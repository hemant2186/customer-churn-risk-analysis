# Model Card: Customer Churn Risk Model

## Intended use

This model estimates churn risk for telecom customers and supports retention prioritization. It is designed for portfolio demonstration and decision-support analysis, not automated customer treatment without human review.

## Data

- Dataset: Telco Customer Churn
- Rows: 7,043
- Target: `Churn`
- Excluded field: `customerID`, removed to avoid identifier leakage

## Model

- Selected model: balanced Logistic Regression
- Preprocessing: numeric imputation, scaling, categorical imputation, one-hot encoding
- Operating threshold: 31%, selected by retention ROI simulation
- App features: single scoring, batch scoring, local linear explanations, and drift monitoring

## Validation metrics

- Recall: 92.5%
- Precision: 43.4%
- ROC AUC: 0.841
- Accuracy: 65.9%
- Confusion matrix: `[[583, 452], [28, 346]]`

## Limitations

- The dataset is static and may not represent current telecom behavior.
- ROI assumptions are illustrative and should be replaced with real business economics.
- The model should be monitored for data drift before production use.
- Local explanations are based on linear-model contribution analysis and should be used as directional evidence, not causal proof.
