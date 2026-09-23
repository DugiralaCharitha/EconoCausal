"""
Unit tests for Week 2: Double Machine Learning (EconML) & Uplift Modeling.
"""

import pytest
import numpy as np
import pandas as pd
from backend.data.generator import generate_retail_data
from backend.causal.estimator import DoubleMLEngine, train_double_ml


@pytest.fixture
def synthetic_data():
    """Fixture providing a synthetic retail dataset for DML testing."""
    return generate_retail_data(n_samples=600, seed=42, treatment_type="discrete")


def test_double_ml_fit_and_predict_causal_forest(synthetic_data):
    """Test CausalForestDML fitting and ITE predictions."""
    df = synthetic_data
    engine = DoubleMLEngine(model_type="causal_forest", n_estimators=40, random_state=42)
    
    # Fit engine
    engine.fit(
        df=df,
        outcome_col="converted",
        treatment_col="treatment_received",
        confounder_cols=["income", "age", "historical_spend"],
        feature_cols=["loyalty_score", "browsing_freq"]
    )

    assert engine.is_fitted is True

    # Predict ITE
    ites = engine.predict_ite(df)
    assert isinstance(ites, np.ndarray)
    assert len(ites) == len(df)
    assert not np.isnan(ites).any()


def test_double_ml_linear_dml(synthetic_data):
    """Test LinearDML fitting and confidence interval bounds."""
    df = synthetic_data
    engine = DoubleMLEngine(model_type="linear_dml", random_state=42)
    
    engine.fit(
        df=df,
        outcome_col="converted",
        treatment_col="treatment_received",
        confounder_cols=["income", "historical_spend"],
        feature_cols=["loyalty_score", "browsing_freq"]
    )

    ites = engine.predict_ite(df)
    lower, upper = engine.predict_ite_interval(df, alpha=0.05)

    assert len(lower) == len(df)
    assert len(upper) == len(df)
    assert (lower <= upper).all()


def test_evaluate_uplift(synthetic_data):
    """Test Qini curve, cumulative uplift, and decile metrics generation."""
    df = synthetic_data
    engine = DoubleMLEngine(model_type="causal_forest", n_estimators=40, random_state=42)
    engine.fit(df=df)

    metrics = engine.evaluate_uplift(df, n_bins=5)

    assert "qini_curve" in metrics
    assert "cumulative_uplift" in metrics
    assert "deciles" in metrics
    assert "overall_ate" in metrics
    assert "mean_predicted_ite" in metrics

    # Verify Qini curve structure
    assert len(metrics["qini_curve"]) > 0
    q_first = metrics["qini_curve"][0]
    assert "percentile" in q_first
    assert "qini_score" in q_first
    assert "random_score" in q_first

    # Verify deciles structure
    assert len(metrics["deciles"]) <= 5
    d_first = metrics["deciles"][0]
    assert "decile" in d_first
    assert "mean_predicted_ite" in d_first
    assert "actual_uplift" in d_first


def test_train_double_ml_convenience_wrapper(synthetic_data):
    """Test convenience function train_double_ml."""
    df = synthetic_data
    engine, metrics = train_double_ml(df, model_type="causal_forest")

    assert engine.is_fitted is True
    assert "qini_curve" in metrics
    summary = engine.get_model_summary()
    assert summary["fitted"] is True
    assert summary["model_type"] == "causal_forest"
