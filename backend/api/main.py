"""
FastAPI REST Service for EconoCausal Backend.

Exposes endpoints for synthetic data generation, DoWhy Causal DAG visualization,
Double Machine Learning training, Refutation Auditing, Prescriptive SciPy Budget Solver,
and Real-Time Data Drift Monitoring.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Any, Optional

from backend.data.generator import generate_retail_data, RetailDataGenerator
from backend.causal.graph import create_retail_causal_graph, CausalGraphBuilder
from backend.causal.estimator import train_double_ml, DoubleMLEngine
from backend.causal.audit import audit_causal_model, CausalRefutationAuditor
from backend.optimization.solver import solve_optimal_budget, PrescriptiveBudgetOptimizer
from backend.models.drift import check_data_drift, DataDriftDetector


app = FastAPI(
    title="EconoCausal API",
    description="Causal & Prescriptive AI Engine for Dynamic Pricing via Double Machine Learning",
    version="1.0.0"
)

# Enable CORS for React frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global in-memory cache for demo state
_dataset_cache: Optional[Any] = None
_dml_engine_cache: Optional[DoubleMLEngine] = None


class DataGenRequest(BaseModel):
    n_samples: int = 2000
    seed: int = 42
    treatment_type: str = "discrete"
    multi_tier_discounts: bool = False
    noise_level: float = 0.0


class TrainDMLRequest(BaseModel):
    model_type: str = "causal_forest"
    cv: int = 3
    n_estimators: int = 40


class OptimizeRequest(BaseModel):
    total_budget: float = 5000.0
    margin_per_order: float = 50.0
    max_discount: float = 20.0


class DriftCheckRequest(BaseModel):
    shift_magnitude: float = 1.0


@app.get("/")
def read_root():
    return {
        "status": "online",
        "service": "EconoCausal Engine API",
        "version": "1.0.0",
        "endpoints": [
            "/api/data/generate",
            "/api/causal/graph",
            "/api/causal/train",
            "/api/causal/audit",
            "/api/optimize",
            "/api/drift/check"
        ]
    }


@app.post("/api/data/generate")
def api_generate_data(req: DataGenRequest):
    global _dataset_cache
    try:
        generator = RetailDataGenerator(seed=req.seed)
        df = generator.generate_data(
            n_samples=req.n_samples,
            treatment_type=req.treatment_type,
            multi_tier_discounts=req.multi_tier_discounts,
            noise_level=req.noise_level
        )
        _dataset_cache = df
        summary = generator.summary_statistics(df)
        sample_rows = df.head(10).to_dict(orient="records")
        return {
            "success": True,
            "summary": summary,
            "sample_data": sample_rows
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/causal/graph")
def api_get_causal_graph():
    try:
        builder = CausalGraphBuilder()
        metadata = builder.get_graph_metadata()
        return {
            "success": True,
            "metadata": metadata
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/causal/train")
def api_train_causal_model(req: TrainDMLRequest):
    global _dataset_cache, _dml_engine_cache
    if _dataset_cache is None:
        _dataset_cache = generate_retail_data(n_samples=2000, seed=42)

    try:
        engine, metrics = train_double_ml(
            df=_dataset_cache,
            model_type=req.model_type,
            cv=req.cv
        )
        _dml_engine_cache = engine
        summary = engine.get_model_summary()
        return {
            "success": True,
            "model_summary": summary,
            "uplift_metrics": metrics
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/causal/audit")
def api_audit_causal_model():
    global _dataset_cache
    if _dataset_cache is None:
        _dataset_cache = generate_retail_data(n_samples=1000, seed=42)

    try:
        estimate, audit_summary = audit_causal_model(_dataset_cache, num_simulations=2)
        auditor = CausalRefutationAuditor()
        auditor.audit_results = {t["method"]: t for t in audit_summary["test_details"]}
        markdown_report = auditor.get_markdown_audit_report()

        return {
            "success": True,
            "audit_summary": audit_summary,
            "markdown_report": markdown_report
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/optimize")
def api_optimize_budget(req: OptimizeRequest):
    global _dataset_cache, _dml_engine_cache
    if _dataset_cache is None:
        _dataset_cache = generate_retail_data(n_samples=2000, seed=42)

    if _dml_engine_cache is None:
        _dml_engine_cache, _ = train_double_ml(_dataset_cache, model_type="causal_forest")

    try:
        result = solve_optimal_budget(
            df=_dataset_cache,
            dml_engine=_dml_engine_cache,
            total_budget=req.total_budget
        )
        return {
            "success": True,
            "optimization_result": result["optimization"],
            "benchmarks": result["benchmarks"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/drift/check")
def api_check_drift(req: DriftCheckRequest):
    global _dataset_cache
    if _dataset_cache is None:
        _dataset_cache = generate_retail_data(n_samples=1000, seed=42)

    try:
        current_df = _dataset_cache.copy()
        if req.shift_magnitude != 1.0:
            current_df["income"] = current_df["income"] * req.shift_magnitude
            current_df["historical_spend"] = current_df["historical_spend"] * req.shift_magnitude

        report = check_data_drift(_dataset_cache, current_df)
        return {
            "success": True,
            "drift_report": report
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
