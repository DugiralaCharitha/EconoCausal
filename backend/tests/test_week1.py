"""
Unit tests for Week 1: Data Generator & DoWhy Causal Graphing.
"""

import pytest
import pandas as pd
from backend.data.generator import RetailDataGenerator, generate_retail_data
from backend.causal.graph import CausalGraphBuilder, create_retail_causal_graph


def test_data_generator_shape_and_columns():
    """Verify data generator output shape, schema, and value ranges."""
    df = generate_retail_data(n_samples=500, seed=42, treatment_type="discrete")
    
    assert len(df) == 500
    expected_cols = [
        "income", "age", "historical_spend", "browsing_freq",
        "loyalty_score", "user_segment", "discount_amount",
        "treatment_received", "propensity_score", "baseline_prob",
        "true_ite", "converted", "revenue", "persona"
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing column {col} in dataset"

    # Value range checks
    assert df["converted"].isin([0, 1]).all()
    assert df["treatment_received"].isin([0, 1]).all()
    assert (df["discount_amount"] >= 0).all()
    assert ((df["propensity_score"] >= 0) & (df["propensity_score"] <= 1)).all()
    assert set(df["persona"].unique()).issubset({"Organic Buyer", "Persuadable", "Lost Cause"})


def test_data_generator_multi_tier_and_summary():
    """Test multi-tier discount generation and summary statistics diagnostics."""
    generator = RetailDataGenerator(seed=42)
    df = generator.generate_data(n_samples=300, multi_tier_discounts=True, noise_level=0.05)
    
    stats = generator.summary_statistics(df)
    assert stats["n_samples"] == 300
    assert "conversion_rate" in stats
    assert "persona_counts" in stats
    assert len(stats["discount_distribution"]) <= 5


def test_causal_graph_builder_metadata():
    """Test DOT graph string creation, Mermaid diagram, and metadata structure for UI."""
    builder = CausalGraphBuilder()
    metadata = builder.get_graph_metadata()

    assert "nodes" in metadata
    assert "edges" in metadata
    assert "dot" in metadata
    assert "mermaid" in metadata
    assert "backdoor_paths" in metadata
    assert "digraph" in metadata["dot"]
    assert "graph TD" in metadata["mermaid"]
    
    # Check key node types
    types = [node["type"] for node in metadata["nodes"]]
    assert "confounder" in types
    assert "treatment" in types
    assert "outcome" in types
    assert "effect_modifier" in types


def test_dowhy_causal_model_creation():
    """Test initialization of Microsoft DoWhy CausalModel with synthetic dataset."""
    df = generate_retail_data(n_samples=200, seed=42)
    model, metadata = create_retail_causal_graph(df)

    assert model is not None
    assert metadata["summary"]["treatment"] == "treatment_received"
    assert metadata["summary"]["outcome"] == "converted"
