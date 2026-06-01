from __future__ import annotations

import joblib
import pandas as pd

from src.inference import explain_prediction, score_customers
from src.monitoring import compare_to_training_profile
from src.saas import (
    authenticate_workspace,
    authenticate_api_key,
    can_score_rows,
    create_api_key,
    fetch_scoring_runs,
    initialize_database,
    record_scoring_run,
    revoke_api_key,
    summarize_usage,
    update_workspace_plan,
)


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


def test_saas_workspace_records_scoring_run(tmp_path) -> None:
    db_path = tmp_path / "saas.sqlite3"
    initialize_database(db_path)
    workspace = authenticate_workspace(db_path, "demo@churnai.com", "demo123")
    artifact = load_artifact()
    sample = pd.DataFrame(artifact["sample_input"]).head(3)
    scored = score_customers(artifact, sample)

    record_scoring_run(db_path, workspace["id"], "batch_campaign", scored, artifact["threshold"])
    runs = fetch_scoring_runs(db_path, workspace["id"])
    usage = summarize_usage(runs)

    assert len(runs) == 1
    assert usage["rows_scored"] == 3
    assert usage["runs"] == 1


def test_saas_plan_limits_and_api_keys(tmp_path) -> None:
    db_path = tmp_path / "saas.sqlite3"
    initialize_database(db_path)
    workspace = authenticate_workspace(db_path, "demo@churnai.com", "demo123")
    workspace = update_workspace_plan(db_path, workspace["id"], "Starter")
    empty_runs = fetch_scoring_runs(db_path, workspace["id"])

    allowed, _ = can_score_rows(workspace, empty_runs, 500)
    blocked, message = can_score_rows(workspace, empty_runs, 501)
    raw_key = create_api_key(db_path, workspace["id"])
    api_workspace = authenticate_api_key(db_path, raw_key)

    assert allowed is True
    assert blocked is False
    assert "Starter" in message
    assert api_workspace["id"] == workspace["id"]

    revoke_api_key(db_path, workspace["id"], 1)
    assert authenticate_api_key(db_path, raw_key) is None
