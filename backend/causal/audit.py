"""
Causal Refutation & Audit Engine for EconoCausal.

Evaluates the stability and robustness of causal effect estimates using Microsoft DoWhy
refutation tests (Placebo Treatment, Random Common Cause, Data Subset Refuters) and
Propensity Score Overlap / Positivity audits.
Computes composite Causal Reliability Index (0-100%) and formats Markdown audit reports.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from sklearn.linear_model import LogisticRegression
import dowhy
from dowhy import CausalModel
from .graph import create_retail_causal_graph


class CausalRefutationAuditor:
    """
    Executes DoWhy refutation tests and propensity score positivity audits to verify causal models
    against hidden bias, positivity violations, and sub-sample instability.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.audit_results = {}

    def run_propensity_overlap_audit(
        self,
        df: pd.DataFrame,
        treatment_col: str = "treatment_received",
        confounder_cols: Optional[List[str]] = None,
        trim_threshold: float = 0.05
    ) -> Dict[str, Any]:
        """
        Positivity / Common Support Audit: Verifies propensity score overlap 0 < P(T=1|W) < 1.
        Extreme propensity scores (< 0.05 or > 0.95) violate positivity and distort DML estimates.
        """
        if confounder_cols is None:
            confounder_cols = ["income", "age", "historical_spend", "browsing_freq"]

        valid_confounders = [c for c in confounder_cols if c in df.columns]
        
        W = df[valid_confounders].values
        T = df[treatment_col].values

        clf = LogisticRegression(random_state=self.random_state, max_iter=500)
        clf.fit(W, T)
        propensities = clf.predict_proba(W)[:, 1]

        treated_ps = propensities[T == 1]
        control_ps = propensities[T == 0]

        extreme_low = float(np.mean(propensities < trim_threshold))
        extreme_high = float(np.mean(propensities > (1 - trim_threshold)))
        positivity_violation = (extreme_low + extreme_high) > 0.10

        # Histogram density bins for UI overlap visualization
        hist_treated, bin_edges = np.histogram(treated_ps, bins=10, range=(0.0, 1.0), density=True)
        hist_control, _ = np.histogram(control_ps, bins=10, range=(0.0, 1.0), density=True)

        overlap_bins = []
        for i in range(len(hist_treated)):
            overlap_bins.append({
                "bin_range": f"{round(bin_edges[i], 1)}-{round(bin_edges[i+1], 1)}",
                "treated_density": round(float(hist_treated[i]), 3),
                "control_density": round(float(hist_control[i]), 3)
            })

        res = {
            "test_name": "Propensity Overlap & Positivity Audit",
            "passed": not positivity_violation,
            "mean_treated_propensity": round(float(np.mean(treated_ps)), 4),
            "mean_control_propensity": round(float(np.mean(control_ps)), 4),
            "extreme_propensity_share": round(float(extreme_low + extreme_high), 4),
            "positivity_violation_warning": positivity_violation,
            "overlap_histogram": overlap_bins,
            "description": "Verifies common support overlap between treated and control groups to ensure positivity assumption holds."
        }
        self.audit_results["propensity_overlap"] = res
        return res

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
        df: Optional[pd.DataFrame] = None,
        num_simulations: int = 3
    ) -> Dict[str, Any]:
        """
        Executes full refutation audit suite and returns composite reliability score.
        """
        t1 = self.run_placebo_treatment_refutation(model, estimand, estimate, num_simulations=num_simulations)
        t2 = self.run_random_common_cause_refutation(model, estimand, estimate, num_simulations=num_simulations)
        t3 = self.run_data_subset_refutation(model, estimand, estimate, num_simulations=num_simulations)

        tests_list = [t1, t2, t3]
        
        if df is not None:
            t4 = self.run_propensity_overlap_audit(df)
            tests_list.append(t4)

        passed_count = sum([t["passed"] for t in tests_list])
        total_tests = len(tests_list)
        score = round((passed_count / float(total_tests)) * 100, 1)

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
            "total_tests": total_tests,
            "test_details": tests_list,
            "recommendation": "Model passed causal robustness & positivity checks. Safe for budget optimization." if passed_count >= (total_tests - 1) else "High risk of bias or positivity violation. Refine confounders before optimization."
        }
        return summary

    def get_markdown_audit_report(self) -> str:
        """
        Generates GitHub-Flavored Markdown summary table of latest audit suite.
        """
        if not self.audit_results:
            return "No audit results available. Run refutation suite first."

        lines = [
            "### 🛡️ Causal Model Refutation & Positivity Audit Report",
            "",
            "| Refutation Test | Original Effect | New Effect / Metric | Pass Status | Description |",
            "| :--- | :--- | :--- | :--- | :--- |"
        ]
        for key, res in self.audit_results.items():
            status_icon = "✅ PASSED" if res.get("passed", False) else "❌ FAILED"
            orig = res.get("original_effect", "N/A")
            new = res.get("new_effect", res.get("mean_treated_propensity", "N/A"))
            name = res.get("test_name", key)
            desc = res.get("description", "")
            lines.append(f"| **{name}** | `{orig}` | `{new}` | {status_icon} | {desc} |")

        return "\n".join(lines)


def audit_causal_model(df: pd.DataFrame, num_simulations: int = 3) -> Tuple[Any, Dict[str, Any]]:
    """
    Convenience wrapper to build DAG model, estimate effect, and run refutation + positivity audit.
    """
    auditor = CausalRefutationAuditor()
    dowhy_model, metadata = create_retail_causal_graph(df)
    estimand = dowhy_model.identify_effect(proceed_when_unidentifiable=True)
    estimate = dowhy_model.estimate_effect(estimand, method_name="backdoor.linear_regression")
    
    audit_summary = auditor.run_full_audit_suite(dowhy_model, estimand, estimate, df=df, num_simulations=num_simulations)
    return estimate, audit_summary
