from __future__ import annotations

import numpy as np

from src.config import ROI_ASSUMPTIONS


def risk_label(probability: float) -> tuple[str, str]:
    if probability >= 0.65:
        return "High risk", "Prioritize proactive retention outreach."
    if probability >= 0.35:
        return "Medium risk", "Send targeted engagement and monitor usage."
    return "Low risk", "Maintain service quality and normal lifecycle messaging."


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
