from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
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
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "telco_churn.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

TARGET = "Churn"
DROP_COLUMNS = ["customerID"]

NUMERIC_FEATURES = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

ROI_ASSUMPTIONS = {
    "retained_customer_value": 450,
    "retention_cost": 45,
    "retention_success_rate": 0.30,
    "missed_churn_cost": 180,
}


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df


def make_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def estimate_roi(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    predictions = probabilities >= threshold
    true_positives = int(((predictions == 1) & (y_true == 1)).sum())
    false_positives = int(((predictions == 1) & (y_true == 0)).sum())
    false_negatives = int(((predictions == 0) & (y_true == 1)).sum())

    contacted = true_positives + false_positives
    saved_value = (
        true_positives
        * ROI_ASSUMPTIONS["retained_customer_value"]
        * ROI_ASSUMPTIONS["retention_success_rate"]
    )
    campaign_cost = contacted * ROI_ASSUMPTIONS["retention_cost"]
    missed_cost = false_negatives * ROI_ASSUMPTIONS["missed_churn_cost"]
    net_value = saved_value - campaign_cost - missed_cost

    return {
        "threshold": round(float(threshold), 2),
        "contacted_customers": contacted,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "estimated_saved_value": round(float(saved_value), 2),
        "campaign_cost": round(float(campaign_cost), 2),
        "missed_churn_cost": round(float(missed_cost), 2),
        "net_value": round(float(net_value), 2),
    }


def find_best_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> dict:
    candidates = [estimate_roi(y_true, probabilities, threshold) for threshold in np.arange(0.30, 0.86, 0.01)]
    return max(candidates, key=lambda item: item["net_value"])


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


def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    df = load_data()
    x = df.drop(columns=[TARGET, *DROP_COLUMNS])
    y = df[TARGET].map({"No": 0, "Yes": 1})

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    candidates = {
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

    results = {}
    for name, model in candidates.items():
        model.fit(x_train, y_train)
        results[name] = {
            "pipeline": model,
            "metrics": evaluate_model(name, model, x_test, y_test),
        }

    best_name = max(results, key=lambda key: results[key]["metrics"]["roi"]["net_value"])
    best_pipeline = results[best_name]["pipeline"]
    best_metrics = results[best_name]["metrics"]

    artifact = {
        "model": best_pipeline,
        "model_name": best_name,
        "threshold": best_metrics["threshold"],
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "feature_options": {
            column: sorted(df[column].dropna().astype(str).unique().tolist()) for column in CATEGORICAL_FEATURES
        },
        "roi_assumptions": ROI_ASSUMPTIONS,
        "metrics": best_metrics,
    }

    joblib.dump(artifact, MODELS_DIR / "churn_pipeline.joblib")
    joblib.dump(best_pipeline, MODELS_DIR / "churn_model.pkl")
    joblib.dump(NUMERIC_FEATURES + CATEGORICAL_FEATURES, MODELS_DIR / "feature_columns.pkl")

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
