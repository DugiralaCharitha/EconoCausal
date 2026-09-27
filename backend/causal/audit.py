"""
Causal Refutation & Audit Engine for EconoCausal.

Evaluates the stability and robustness of causal effect estimates using Microsoft DoWhy
refutation tests (Placebo Treatment, Random Common Cause, Data Subset Refuters).
Computes composite Causal Reliability Index (0-100%) and formats Markdown audit reports.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import dowhy
from dowhy import CausalModel
from .graph import create_retail_causal_graph


class CausalRefutationAuditor:
    """
    Executes DoWhy refutation tests to audit causal inference models against hidden bias,
    spurious correlations, and sub-sample instability.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.audit_results = {}

    def run_placebo_treatment_refutation(
        self,
        model: CausalModel,
        estimand: Any,
        estimate: Any,
        num_simulations: int = 5
    ) -> Dict[str, Any]:
        """
        Placebo Treatment Refuter: Replaces the actual discount treatment with random noise.
        The estimated causal effect should drop to near zero.
        """
        refuter = model.refute_estimate(
            estimand=estimand,
            estimate=estimate,
            method_name="placebo_treatment_refuter",
            placebo_type="permute",
            num_simulations=num_simulations,
            random_state=self.random_state
        )

        new_effect = float(refuter.new_effect)
        orig_effect = float(refuter.estimated_effect)
        p_val = float(getattr(refuter, "p_value", 0.0) or 0.0)

        passed = abs(new_effect) < 0.08 or (abs(orig_effect) > 0 and abs(new_effect / orig_effect) < 0.30)

        res = {
            "test_name": "Placebo Treatment Refuter",
            "method": "placebo_treatment_refuter",
            "original_effect": round(orig_effect, 4),
            "new_effect": round(new_effect, 4),
            "p_value": round(p_val, 4),
            "passed": bool(passed),
            "description": "Replaces treatment with permuted random noise. Effect must drop to near 0."
        }
        self.audit_results["placebo_treatment"] = res
        return res

    def run_random_common_cause_refutation(
        self,
        model: CausalModel,
        estimand: Any,
        estimate: Any,
        num_simulations: int = 5
    ) -> Dict[str, Any]:
        """
        Random Common Cause Refuter: Adds a random unobserved confounder to the graph.
        The estimated causal effect should remain practically unchanged.
        """
        refuter = model.refute_estimate(
            estimand=estimand,
            estimate=estimate,
            method_name="random_common_cause",
            num_simulations=num_simulations,
            random_state=self.random_state
        )

        new_effect = float(refuter.new_effect)
        orig_effect = float(refuter.estimated_effect)
        delta = abs(new_effect - orig_effect)

        passed = delta < 0.08 or (abs(orig_effect) > 0 and (delta / abs(orig_effect)) < 0.25)

        res = {
            "test_name": "Random Common Cause Refuter",
            "method": "random_common_cause",
            "original_effect": round(orig_effect, 4),
            "new_effect": round(new_effect, 4),
            "effect_delta": round(delta, 4),
            "passed": bool(passed),
            "description": "Adds a random dummy confounder. Causal estimate should remain stable."
        }
        self.audit_results["random_common_cause"] = res
        return res

    def run_data_subset_refutation(
        self,
        model: CausalModel,
        estimand: Any,
        estimate: Any,
        subset_fraction: float = 0.8,
        num_simulations: int = 5
    ) -> Dict[str, Any]:
        """
        Data Subset Refuter: Evaluates model stability across random 80% data subsets.
        """
        refuter = model.refute_estimate(
            estimand=estimand,
            estimate=estimate,
            method_name="data_subset_refuter",
            subset_fraction=subset_fraction,
            num_simulations=num_simulations,
            random_state=self.random_state
        )

        new_effect = float(refuter.new_effect)
        orig_effect = float(refuter.estimated_effect)
        delta = abs(new_effect - orig_effect)

        passed = delta < 0.08 or (abs(orig_effect) > 0 and (delta / abs(orig_effect)) < 0.25)

        res = {
            "test_name": "Data Subset Refuter",
            "method": "data_subset_refuter",
            "subset_fraction": subset_fraction,
            "original_effect": round(orig_effect, 4),
            "new_effect": round(new_effect, 4),
            "effect_delta": round(delta, 4),
            "passed": bool(passed),
            "description": "Re-estimates effect on random 80% data subsets to verify stability."
        }
        self.audit_results["data_subset"] = res
        return res

    def run_full_audit_suite(
        self,
        model: CausalModel,
        estimand: Any,
        estimate: Any,
        num_simulations: int = 3
    ) -> Dict[str, Any]:
        """
        Executes full refutation audit suite and returns composite reliability score.
        """
        t1 = self.run_placebo_treatment_refutation(model, estimand, estimate, num_simulations=num_simulations)
        t2 = self.run_random_common_cause_refutation(model, estimand, estimate, num_simulations=num_simulations)
        t3 = self.run_data_subset_refutation(model, estimand, estimate, num_simulations=num_simulations)

        passed_count = sum([t1["passed"], t2["passed"], t3["passed"]])
        score = round((passed_count / 3.0) * 100, 1)

        if score >= 90:
            status = "HIGHLY_RELIABLE"
        elif score >= 60:
            status = "MODERATE_RELIABILITY"
        else:
            status = "UNRELIABLE"

        summary = {
            "causal_reliability_score": score,
            "reliability_status": status,
            "tests_passed": passed_count,
            "total_tests": 3,
            "test_details": [t1, t2, t3],
            "recommendation": "Model passed causal robustness checks. Safe for budget optimization." if passed_count >= 2 else "High risk of confounding bias. Refine confounder set before optimization."
        }
        return summary

    def get_markdown_audit_report(self) -> str:
        """
        Generates GitHub-Flavored Markdown summary table of latest audit suite.
        """
        if not self.audit_results:
            return "No audit results available. Run refutation suite first."

        lines = [
            "### 🛡️ Causal Model Refutation Audit Report",
            "",
            "| Refutation Test | Original Effect | New Effect | Pass Status | Description |",
            "| :--- | :--- | :--- | :--- | :--- |"
        ]
        for key, res in self.audit_results.items():
            status_icon = "✅ PASSED" if res.get("passed", False) else "❌ FAILED"
            orig = res.get("original_effect", 0.0)
            new = res.get("new_effect", 0.0)
            name = res.get("test_name", key)
            desc = res.get("description", "")
            lines.append(f"| **{name}** | `{orig}` | `{new}` | {status_icon} | {desc} |")

        return "\n".join(lines)


def audit_causal_model(df: pd.DataFrame, num_simulations: int = 3) -> Tuple[Any, Dict[str, Any]]:
    """
    Convenience wrapper to build DAG model, estimate effect, and run refutation audit.
    """
    auditor = CausalRefutationAuditor()
    dowhy_model, metadata = create_retail_causal_graph(df)
    estimand = dowhy_model.identify_effect(proceed_when_unidentifiable=True)
    estimate = dowhy_model.estimate_effect(estimand, method_name="backdoor.linear_regression")
    
    audit_summary = auditor.run_full_audit_suite(dowhy_model, estimand, estimate, num_simulations=num_simulations)
    return estimate, audit_summary
