"""
Unit tests for Real-Time Data Drift Monitoring Engine.
"""

import pytest
import numpy as np
import pandas as pd
from backend.data.generator import generate_retail_data
from backend.models.drift import DataDriftDetector, check_data_drift


def test_no_drift_identical_distributions():
    """Test drift detector on data from identical distribution (should report NO drift)."""
    df_ref = generate_retail_data(n_samples=500, seed=42)
    df_curr = generate_retail_data(n_samples=500, seed=43)

    detector = DataDriftDetector(alpha=0.01)
    report = detector.detect_dataset_drift(df_ref, df_curr)

    assert "dataset_drift_detected" in report
    assert report["dataset_drift_detected"] is False
    assert report["drift_severity"] == "LOW"


def test_drift_detection_on_shifted_data():
    """Test drift detector when income and historical spend features undergo significant shift."""
    df_ref = generate_retail_data(n_samples=500, seed=42)
    df_curr = df_ref.copy()
    
    # Introduce deliberate extreme shift to income and spend
    df_curr["income"] = df_curr["income"] * 3.5 + 50000
    df_curr["historical_spend"] = df_curr["historical_spend"] * 4.0 + 1000

    report = check_data_drift(df_ref, df_curr)

    assert report["dataset_drift_detected"] is True
    assert report["drifted_features_count"] >= 2
    assert report["drift_severity"] in ["MEDIUM", "HIGH"]


def test_single_feature_ks_test():
    """Test single feature KS test statistics."""
    ref_vals = np.random.normal(50, 5, 300)
    curr_vals = np.random.normal(70, 5, 300)

    detector = DataDriftDetector()
    res = detector.detect_feature_drift(ref_vals, curr_vals, feature_name="income")

    assert res["drift_detected"] is True
    assert res["p_value"] < 0.05
    assert res["wasserstein_distance"] > 10.0
