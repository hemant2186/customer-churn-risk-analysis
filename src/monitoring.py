from __future__ import annotations

import pandas as pd

from src.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from src.data import clean_input_frame


def build_training_profile(df: pd.DataFrame) -> dict:
    clean = clean_input_frame(df)
    return {
        "numeric": {
            column: {
                "mean": float(clean[column].mean()),
                "std": float(clean[column].std() or 0),
            }
            for column in NUMERIC_FEATURES
        },
        "categorical": {
            column: clean[column].value_counts(normalize=True).to_dict() for column in CATEGORICAL_FEATURES
        },
    }


def compare_to_training_profile(df: pd.DataFrame, profile: dict) -> pd.DataFrame:
    clean = clean_input_frame(df)
    rows = []

    for column in NUMERIC_FEATURES:
        if column not in clean.columns:
            rows.append([column, "numeric", "missing", None, "Missing from uploaded data"])
            continue
        train_mean = profile["numeric"][column]["mean"]
        train_std = profile["numeric"][column]["std"]
        upload_mean = float(clean[column].mean())
        z_score = 0.0 if train_std == 0 else abs(upload_mean - train_mean) / train_std
        status = "Review" if z_score >= 0.5 else "OK"
        rows.append([column, "numeric", round(z_score, 3), round(upload_mean, 3), status])

    for column in CATEGORICAL_FEATURES:
        if column not in clean.columns:
            rows.append([column, "categorical", "missing", None, "Missing from uploaded data"])
            continue
        train_dist = profile["categorical"][column]
        upload_dist = clean[column].value_counts(normalize=True).to_dict()
        categories = set(train_dist) | set(upload_dist)
        max_shift = max(abs(upload_dist.get(category, 0) - train_dist.get(category, 0)) for category in categories)
        status = "Review" if max_shift >= 0.20 else "OK"
        rows.append([column, "categorical", round(max_shift, 3), clean[column].mode().iloc[0], status])

    return pd.DataFrame(rows, columns=["Feature", "Type", "Shift score", "Uploaded summary", "Status"])
