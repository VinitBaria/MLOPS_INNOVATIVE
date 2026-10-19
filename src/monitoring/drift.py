"""
Data & Concept Drift Detection Module
Uses Kolmogorov-Smirnov (KS) Test & Population Stability Index (PSI)
to detect shifts in rating distribution and user engagement.
"""

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from typing import Dict, Any, List
from src.monitoring.metrics import DATA_DRIFT_SCORE, DRIFT_ALERT_TRIGGERED


def calculate_psi(expected: np.ndarray, actual: np.ndarray, num_buckets: int = 5) -> float:
    """
    Calculate Population Stability Index (PSI) between baseline and new ratings.
    PSI < 0.1: No significant change
    0.1 <= PSI < 0.2: Moderate drift
    PSI >= 0.2: Significant drift detected
    """
    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    bins = np.linspace(1, 5, num_buckets + 1)
    expected_counts, _ = np.histogram(expected, bins=bins)
    actual_counts, _ = np.histogram(actual, bins=bins)

    # Add epsilon to prevent division by zero
    eps = 1e-4
    expected_pct = (expected_counts + eps) / (len(expected) + eps * num_buckets)
    actual_pct = (actual_counts + eps) / (len(actual) + eps * num_buckets)

    psi_val = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi_val)


def detect_rating_drift(
    baseline_ratings: List[float],
    recent_ratings: List[float],
    p_value_threshold: float = 0.05,
    psi_threshold: float = 0.15,
) -> Dict[str, Any]:
    """
    Detects distribution drift between baseline training ratings and recent production ratings.
    
    Returns:
        Dict containing drift status, KS p-value, PSI, and recommendations.
    """
    if len(recent_ratings) < 20:
        return {
            "drift_detected": False,
            "status": "INSUFFICIENT_DATA",
            "message": f"Only {len(recent_ratings)} new ratings. Need at least 20 to compute drift.",
            "p_value": 1.0,
            "psi": 0.0,
        }

    base_arr = np.array(baseline_ratings)
    recent_arr = np.array(recent_ratings)

    # 1. Kolmogorov-Smirnov 2-sample test
    ks_stat, p_value = ks_2samp(base_arr, recent_arr)

    # 2. Population Stability Index (PSI)
    psi = calculate_psi(base_arr, recent_arr)

    # Drift decision: low p-value or elevated PSI indicates drift
    drift_detected = (p_value < p_value_threshold) or (psi >= psi_threshold)

    # Update Prometheus gauges
    DATA_DRIFT_SCORE.set(float(p_value))
    DRIFT_ALERT_TRIGGERED.set(1 if drift_detected else 0)

    return {
        "drift_detected": bool(drift_detected),
        "status": "DRIFT_DETECTED" if drift_detected else "STABLE",
        "ks_statistic": float(ks_stat),
        "p_value": float(p_value),
        "psi": float(psi),
        "baseline_mean": float(np.mean(base_arr)),
        "recent_mean": float(np.mean(recent_arr)),
        "recent_sample_size": len(recent_ratings),
        "recommendation": "Trigger automated retraining pipeline" if drift_detected else "Model is healthy",
    }
