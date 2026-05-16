from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.business import find_best_threshold
from src.config import (
    CATEGORICAL_FEATURES,
    DROP_COLUMNS,
    MODEL_FEATURES,
    MODELS_DIR,
    NUMERIC_FEATURES,
    REPORTS_DIR,
    ROI_ASSUMPTIONS,
    TARGET,
)
from src.data import load_training_data
from src.features import make_preprocessor
from src.monitoring import build_training_profile


def evaluate_model(name: str, model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict:
    probabilities = model.predict_proba(x_test)[:, 1]
    threshold_result = find_best_threshold(y_test.to_numpy(), probabilities)
    predictions = probabilities >= threshold_result["threshold"]

    return {
        "model": name,
        "threshold": threshold_result["threshold"],
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision": round(float(precision_score(y_test, predictions)), 4),
        "recall": round(float(recall_score(y_test, predictions)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "classification_report": classification_report(y_test, predictions, output_dict=True),
        "roi": threshold_result,
    }


def make_candidates() -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("preprocess", make_preprocessor()),
                ("model", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocess", make_preprocessor()),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=350,
                        min_samples_leaf=8,
                        class_weight="balanced_subsample",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    df = load_training_data()
    x = df.drop(columns=[TARGET, *DROP_COLUMNS])
    y = df[TARGET].map({"No": 0, "Yes": 1})

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    results = {}
    for name, model in make_candidates().items():
        model.fit(x_train, y_train)
        results[name] = {
            "pipeline": model,
            "metrics": evaluate_model(name, model, x_test, y_test),
        }

    best_name = max(results, key=lambda key: results[key]["metrics"]["roi"]["net_value"])
    best_pipeline = results[best_name]["pipeline"]
    best_metrics = results[best_name]["metrics"]
    preprocessor = best_pipeline.named_steps["preprocess"]

    artifact = {
        "model": best_pipeline,
        "model_name": best_name,
        "threshold": best_metrics["threshold"],
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "model_features": MODEL_FEATURES,
        "feature_options": {
            column: sorted(df[column].dropna().astype(str).unique().tolist()) for column in CATEGORICAL_FEATURES
        },
        "transformed_feature_names": preprocessor.get_feature_names_out().tolist(),
        "transformed_feature_means": preprocessor.transform(x_train).mean(axis=0).tolist(),
        "training_profile": build_training_profile(x_train),
        "sample_input": x_test.head(25).to_dict(orient="records"),
        "roi_assumptions": ROI_ASSUMPTIONS,
        "metrics": best_metrics,
    }

    joblib.dump(artifact, MODELS_DIR / "churn_pipeline.joblib")
    joblib.dump(best_pipeline, MODELS_DIR / "churn_model.pkl")
    joblib.dump(MODEL_FEATURES, MODELS_DIR / "feature_columns.pkl")

    report = {
        "dataset_rows": int(df.shape[0]),
        "dataset_columns": int(df.shape[1]),
        "target_rate": round(float(y.mean()), 4),
        "selected_model": best_name,
        "models": {name: payload["metrics"] for name, payload in results.items()},
    }
    (REPORTS_DIR / "model_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
