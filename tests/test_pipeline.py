from __future__ import annotations

import joblib
import pandas as pd

from src.inference import explain_prediction, score_customers
from src.monitoring import compare_to_training_profile


def load_artifact() -> dict:
    return joblib.load("models/churn_pipeline.joblib")


def test_model_artifact_scores_probability_between_zero_and_one() -> None:
    artifact = load_artifact()
    sample = pd.DataFrame(artifact["sample_input"]).head(1)
    scored = score_customers(artifact, sample)

    probability = float(scored.loc[0, "churn_probability"])
    assert 0 <= probability <= 1
    assert scored.loc[0, "recommended_decision"] in {"Contact", "Monitor"}


def test_local_explanation_returns_ranked_contributions() -> None:
    artifact = load_artifact()
    sample = pd.DataFrame(artifact["sample_input"]).head(1)
    explanation = explain_prediction(artifact, sample, top_n=5)

    assert len(explanation) == 5
    assert {"Feature", "Contribution", "Direction"}.issubset(explanation.columns)


def test_monitoring_profile_flags_expected_columns() -> None:
    artifact = load_artifact()
    sample = pd.DataFrame(artifact["sample_input"])
    drift = compare_to_training_profile(sample, artifact["training_profile"])

    assert {"Feature", "Type", "Shift score", "Uploaded summary", "Status"}.issubset(drift.columns)
    assert len(drift) == len(artifact["model_features"])
