"""
Drift Detection Tests
"""

import numpy as np
from src.monitoring.drift import detect_rating_drift, calculate_psi


def test_psi_identical_distribution():
    data = np.random.normal(3.5, 0.5, 1000)
    psi = calculate_psi(data, data)
    assert psi < 0.05  # No drift


def test_psi_drifted_distribution():
    baseline = np.random.normal(4.0, 0.3, 1000)
    drifted = np.random.normal(1.5, 0.3, 1000)
    psi = calculate_psi(baseline, drifted)
    assert psi >= 0.20  # Significant drift detected


def test_drift_detector_logic():
    baseline = [4.0, 5.0, 4.5, 3.5, 4.0, 5.0, 4.5, 4.0] * 10
    recent_shifted = [1.0, 1.5, 2.0, 1.0, 1.5, 2.0, 1.0, 1.5] * 10

    result = detect_rating_drift(baseline, recent_shifted)
    assert result["drift_detected"] is True
    assert result["status"] == "DRIFT_DETECTED"
