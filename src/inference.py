from __future__ import annotations

import numpy as np
import pandas as pd

from src.business import risk_label
from src.data import align_model_input


def score_customers(artifact: dict, df: pd.DataFrame) -> pd.DataFrame:
    model_input = align_model_input(df)
    probabilities = artifact["model"].predict_proba(model_input)[:, 1]
    scored = model_input.copy()
    scored["churn_probability"] = probabilities
    scored["risk_tier"] = [risk_label(float(probability))[0] for probability in probabilities]
    scored["recommended_decision"] = np.where(probabilities >= artifact["threshold"], "Contact", "Monitor")
    return scored


def explain_prediction(artifact: dict, row: pd.DataFrame, top_n: int = 8) -> pd.DataFrame:
    pipeline = artifact["model"]
    preprocessor = pipeline.named_steps["preprocess"]
    estimator = pipeline.named_steps["model"]
    transformed = preprocessor.transform(align_model_input(row))
    baseline = artifact.get("transformed_feature_means")
    feature_names = artifact.get("transformed_feature_names")

    if baseline is None or feature_names is None or not hasattr(estimator, "coef_"):
        return pd.DataFrame(columns=["Feature", "Contribution", "Direction"])

    contributions = (transformed[0] - np.asarray(baseline)) * estimator.coef_[0]
    explanation = pd.DataFrame(
        {
            "Feature": feature_names,
            "Contribution": contributions,
        }
    )
    explanation["Direction"] = np.where(explanation["Contribution"] >= 0, "Raises churn risk", "Lowers churn risk")
    explanation["abs_contribution"] = explanation["Contribution"].abs()
    explanation = explanation.sort_values("abs_contribution", ascending=False).head(top_n)
    return explanation.drop(columns=["abs_contribution"]).reset_index(drop=True)
