from __future__ import annotations

import pandas as pd

from src.config import CATEGORICAL_FEATURES, DATA_PATH, MODEL_FEATURES, NUMERIC_FEATURES


def load_training_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    return clean_input_frame(df)


def clean_input_frame(df: pd.DataFrame) -> pd.DataFrame:
    clean = df.copy()
    if "TotalCharges" in clean.columns:
        clean["TotalCharges"] = pd.to_numeric(clean["TotalCharges"], errors="coerce")
    if "SeniorCitizen" in clean.columns:
        clean["SeniorCitizen"] = pd.to_numeric(clean["SeniorCitizen"], errors="coerce").fillna(0).astype(int)
    for column in NUMERIC_FEATURES:
        if column in clean.columns:
            clean[column] = pd.to_numeric(clean[column], errors="coerce")
    for column in CATEGORICAL_FEATURES:
        if column in clean.columns:
            clean[column] = clean[column].astype(str)
    return clean


def align_model_input(df: pd.DataFrame) -> pd.DataFrame:
    clean = clean_input_frame(df)
    missing = [column for column in MODEL_FEATURES if column not in clean.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    return clean[MODEL_FEATURES]
