"""
Unit tests for Week 3: Prescriptive SciPy Budget Optimization Engine.
"""

import pytest
import numpy as np
import pandas as pd
from backend.data.generator import generate_retail_data
from backend.causal.estimator import DoubleMLEngine
from backend.optimization.solver import PrescriptiveBudgetOptimizer


@pytest.fixture
def fitted_causal_setup():
    """Fixture providing synthetic dataset and fitted Double ML model engine."""
    df = generate_retail_data(n_samples=500, seed=42)
    engine = DoubleMLEngine(model_type="causal_forest", n_estimators=40, random_state=42)
    engine.fit(df)
    return df, engine


def test_prescriptive_budget_optimizer_execution(fitted_causal_setup):
    """Test solver optimization under strict budget constraints."""
    df, engine = fitted_causal_setup
    optimizer = PrescriptiveBudgetOptimizer(random_state=42)

    result = optimizer.optimize_discounts(
        df=df,
        dml_engine=engine,
        total_budget=2000.0,
        margin_per_order=50.0,
        max_discount=20.0
    )

    assert "n_customers" in result
    assert result["n_customers"] == len(df)
    assert result["budget_spent"] <= 2000.0
    assert result["optimal_net_profit"] >= result["baseline_net_profit"]
    assert "discount_distribution" in result
    assert "persona_breakdown" in result


def test_zero_budget_constraint(fitted_causal_setup):
    """Test optimizer behavior under $0 budget (should assign $0 discount to all)."""
    df, engine = fitted_causal_setup
    optimizer = PrescriptiveBudgetOptimizer()

    result = optimizer.optimize_discounts(
        df=df,
        dml_engine=engine,
        total_budget=0.0
    )

    assert result["budget_spent"] == 0.0
    assert result["discount_distribution"].get("0.0", 0) == len(df)
