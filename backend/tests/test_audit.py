"""
Unit tests for Mid-Project Causal Refutation Audit Suite.
"""

import pytest
import pandas as pd
from backend.data.generator import generate_retail_data
from backend.causal.graph import create_retail_causal_graph
from backend.causal.audit import CausalRefutationAuditor, audit_causal_model


@pytest.fixture
def sample_causal_setup():
    """Fixture initializing synthetic data, DoWhy graph model, estimand, and linear estimator."""
    df = generate_retail_data(n_samples=400, seed=42)
    dowhy_model, metadata = create_retail_causal_graph(df)
    
    identified_estimand = dowhy_model.identify_effect(proceed_when_unidentifiable=True)
    estimate = dowhy_model.estimate_effect(
        identified_estimand,
        method_name="backdoor.linear_regression"
    )
    return dowhy_model, identified_estimand, estimate, df


def test_propensity_overlap_audit(sample_causal_setup):
    """Test Propensity Score Overlap & Positivity Audit execution."""
    _, _, _, df = sample_causal_setup
    auditor = CausalRefutationAuditor(random_state=42)

    res = auditor.run_propensity_overlap_audit(df)

    assert "passed" in res
    assert "overlap_histogram" in res
    assert len(res["overlap_histogram"]) == 10
    assert "positivity_violation_warning" in res


def test_placebo_treatment_refutation(sample_causal_setup):
    """Test Placebo Treatment Refuter execution and pass condition."""
    dowhy_model, identified_estimand, estimate, _ = sample_causal_setup
    auditor = CausalRefutationAuditor(random_state=42)
    
    result = auditor.run_placebo_treatment_refutation(dowhy_model, identified_estimand, estimate, num_simulations=3)

    assert "passed" in result
    assert result["test_name"] == "Placebo Treatment Refuter"
    assert "new_effect" in result
    assert "original_effect" in result


def test_random_common_cause_refutation(sample_causal_setup):
    """Test Random Common Cause Refuter execution and effect stability."""
    dowhy_model, identified_estimand, estimate, _ = sample_causal_setup
    auditor = CausalRefutationAuditor(random_state=42)
    
    result = auditor.run_random_common_cause_refutation(dowhy_model, identified_estimand, estimate, num_simulations=3)

    assert "passed" in result
    assert result["test_name"] == "Random Common Cause Refuter"
    assert "effect_delta" in result


def test_data_subset_refutation(sample_causal_setup):
    """Test Data Subset Refuter execution across 80% bootstrap samples."""
    dowhy_model, identified_estimand, estimate, _ = sample_causal_setup
    auditor = CausalRefutationAuditor(random_state=42)
    
    result = auditor.run_data_subset_refutation(dowhy_model, identified_estimand, estimate, subset_fraction=0.8, num_simulations=3)

    assert "passed" in result
    assert result["test_name"] == "Data Subset Refuter"
    assert result["subset_fraction"] == 0.8


def test_full_audit_suite_and_markdown_report(sample_causal_setup):
    """Test run_full_audit_suite composite reliability score and markdown formatting."""
    dowhy_model, identified_estimand, estimate, df = sample_causal_setup
    auditor = CausalRefutationAuditor(random_state=42)
    
    suite_summary = auditor.run_full_audit_suite(dowhy_model, identified_estimand, estimate, df=df, num_simulations=3)

    assert "causal_reliability_score" in suite_summary
    assert "reliability_status" in suite_summary
    assert suite_summary["tests_passed"] >= 3
    assert suite_summary["total_tests"] == 4

    markdown_report = auditor.get_markdown_audit_report()
    assert "Refutation" in markdown_report
    assert "Placebo Treatment Refuter" in markdown_report


def test_audit_causal_model_wrapper():
    """Test convenience wrapper audit_causal_model."""
    df = generate_retail_data(n_samples=300, seed=42)
    estimate, summary = audit_causal_model(df, num_simulations=2)

    assert estimate is not None
    assert "causal_reliability_score" in summary
    assert summary["causal_reliability_score"] >= 60.0
