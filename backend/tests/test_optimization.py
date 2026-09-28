"""
Unit tests for Week 3: Prescriptive SciPy Budget Optimization Engine.
"""

import pytest
import numpy as np
import pandas as pd
from backend.data.generator import generate_retail_data
from backend.causal.estimator import DoubleMLEngine
from backend.optimization.solver import PrescriptiveBudgetOptimizer, solve_optimal_budget


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


def test_compare_against_baselines(fitted_causal_setup):
    """Test baseline benchmarking comparisons (Causal vs Blanket vs Zero)."""
    df, engine = fitted_causal_setup
    optimizer = PrescriptiveBudgetOptimizer()

    benchmarks = optimizer.compare_against_baselines(df, engine, total_budget=1500.0)

    assert "prescriptive_causal" in benchmarks
    assert "blanket_targeting" in benchmarks
    assert "no_discount_baseline" in benchmarks
    assert benchmarks["prescriptive_causal"]["net_profit"] >= benchmarks["blanket_targeting"]["net_profit"]


def test_simulate_budget_sensitivity(fitted_causal_setup):
    """Test budget sensitivity grid curve generation."""
    df, engine = fitted_causal_setup
    optimizer = PrescriptiveBudgetOptimizer()

    curve = optimizer.simulate_budget_sensitivity(df, engine, budget_grid=[500, 1500, 3000])

    assert len(curve) == 3
    assert curve[0]["budget_limit"] == 500
    assert curve[2]["budget_limit"] == 3000
    assert "roi_multiplier" in curve[0]


def test_solve_optimal_budget_wrapper(fitted_causal_setup):
    """Test solve_optimal_budget convenience function."""
    df, engine = fitted_causal_setup
    res = solve_optimal_budget(df, engine, total_budget=1000.0)

    assert "optimization" in res
    assert "benchmarks" in res
    assert res["optimization"]["budget_spent"] <= 1000.0
