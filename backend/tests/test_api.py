"""
Unit tests for FastAPI REST Endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from backend.api.main import app

client = TestClient(app)


def test_root_endpoint():
    """Test GET / health check endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "version" in data


def test_generate_data_endpoint():
    """Test POST /api/data/generate endpoint."""
    payload = {
        "n_samples": 200,
        "seed": 42,
        "treatment_type": "discrete"
    }
    response = client.post("/api/data/generate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "summary" in data
    assert len(data["sample_data"]) == 10


def test_get_causal_graph_endpoint():
    """Test GET /api/causal/graph endpoint."""
    response = client.get("/api/causal/graph")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "metadata" in data
    assert "mermaid" in data["metadata"]


def test_train_causal_model_endpoint():
    """Test POST /api/causal/train endpoint."""
    payload = {
        "model_type": "causal_forest",
        "cv": 2,
        "n_estimators": 20
    }
    response = client.post("/api/causal/train", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "model_summary" in data
    assert "uplift_metrics" in data


def test_optimize_budget_endpoint():
    """Test POST /api/optimize endpoint."""
    payload = {
        "total_budget": 3000.0,
        "margin_per_order": 50.0
    }
    response = client.post("/api/optimize", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "optimization_result" in data
    assert "benchmarks" in data


def test_drift_check_endpoint():
    """Test POST /api/drift/check endpoint."""
    payload = {
        "shift_magnitude": 1.0
    }
    response = client.post("/api/drift/check", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "drift_report" in data
