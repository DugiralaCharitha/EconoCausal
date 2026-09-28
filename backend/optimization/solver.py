"""
Prescriptive Optimization Engine using SciPy.

Solves budget-constrained personalized discount allocation to maximize campaign net profit
and ROI using Double Machine Learning (DML) CATE/ITE estimates.
Includes baseline benchmarking and budget sensitivity curve analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from backend.causal.estimator import DoubleMLEngine


class PrescriptiveBudgetOptimizer:
    """
    Formulates and solves constrained discount allocation optimization problems.
    Categorizes users into Persuadables, Organic Buyers, Sleeping Dogs, and Lost Causes.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def optimize_discounts(
        self,
        df: pd.DataFrame,
        dml_engine: DoubleMLEngine,
        total_budget: float = 10000.0,
        margin_per_order: float = 50.0,
        max_discount: float = 20.0,
        discount_options: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Solves optimal discrete discount assignment per customer under total budget B constraint.
        """
        if discount_options is None:
            discount_options = [0.0, 5.0, 10.0, 15.0, 20.0]

        discount_options = [d for d in discount_options if d <= max_discount]

        N = len(df)
        predicted_ite = dml_engine.predict_ite(df)
        baseline_prob = df["baseline_prob"].values if "baseline_prob" in df.columns else np.full(N, 0.20)

        # Payoff matrix: Net Expected Revenue = (BaselineProb + ITE*(disc/20)) * Margin - disc
        payoff_matrix = np.zeros((N, len(discount_options)))
        cost_matrix = np.zeros((N, len(discount_options)))

        for j, disc in enumerate(discount_options):
            scale = (disc / max_discount) if max_discount > 0 else 1.0
            expected_conversion = np.clip(baseline_prob + predicted_ite * scale, 0.0, 0.99)
            net_profit = (expected_conversion * margin_per_order) - disc
            
            payoff_matrix[:, j] = net_profit
            cost_matrix[:, j] = disc

        chosen_option = np.zeros(N, dtype=int)
        current_budget_spent = 0.0

        upgrades = []
        for i in range(N):
            base_profit = payoff_matrix[i, 0]
            for j in range(1, len(discount_options)):
                cost = cost_matrix[i, j]
                profit_gain = payoff_matrix[i, j] - base_profit
                if profit_gain > 0 and cost > 0:
                    efficiency = profit_gain / cost
                    upgrades.append((efficiency, profit_gain, cost, i, j))

        upgrades.sort(key=lambda x: x[0], reverse=True)

        for eff, profit_gain, cost, i, j in upgrades:
            if current_budget_spent + cost <= total_budget and chosen_option[i] == 0:
                chosen_option[i] = j
                current_budget_spent += cost

        optimal_discounts = np.array([discount_options[idx] for idx in chosen_option])
        optimal_net_profit = sum(payoff_matrix[i, chosen_option[i]] for i in range(N))
        baseline_net_profit = sum(payoff_matrix[i, 0] for i in range(N))

        assigned_personas = []
        for i in range(N):
            bp = baseline_prob[i]
            ite_val = predicted_ite[i]

            if bp >= 0.45:
                assigned_personas.append("Organic Buyer")
            elif ite_val >= 0.15:
                assigned_personas.append("Persuadable")
            elif ite_val < 0.0:
                assigned_personas.append("Sleeping Dog")
            else:
                assigned_personas.append("Lost Cause")

        persona_counts = pd.Series(assigned_personas).value_counts().to_dict()
        discount_distrib = pd.Series(optimal_discounts).value_counts().to_dict()

        return {
            "n_customers": N,
            "total_budget_limit": total_budget,
            "budget_spent": round(float(current_budget_spent), 2),
            "budget_utilization_pct": round(float(current_budget_spent / max(1.0, total_budget) * 100), 2),
            "optimal_net_profit": round(float(optimal_net_profit), 2),
            "baseline_net_profit": round(float(baseline_net_profit), 2),
            "profit_lift": round(float(optimal_net_profit - baseline_net_profit), 2),
            "profit_lift_pct": round(float((optimal_net_profit - baseline_net_profit) / max(1.0, abs(baseline_net_profit)) * 100), 2),
            "optimal_discounts": optimal_discounts.tolist(),
            "discount_distribution": {str(k): int(v) for k, v in discount_distrib.items()},
            "persona_breakdown": persona_counts
        }

    def compare_against_baselines(
        self,
        df: pd.DataFrame,
        dml_engine: DoubleMLEngine,
        total_budget: float = 10000.0,
        margin_per_order: float = 50.0
    ) -> Dict[str, Any]:
        """
        Benchmarks Prescriptive Causal Optimization against Blanket and Random strategies.
        """
        N = len(df)
        opt_res = self.optimize_discounts(df, dml_engine, total_budget, margin_per_order)

        # 1. Blanket Strategy: $10 to everyone until budget runs out
        blanket_discounts = np.zeros(N)
        b_spent = 0.0
        for i in range(N):
            if b_spent + 10.0 <= total_budget:
                blanket_discounts[i] = 10.0
                b_spent += 10.0

        # 2. Zero Discount Baseline
        zero_discounts = np.zeros(N)

        predicted_ite = dml_engine.predict_ite(df)
        baseline_prob = df["baseline_prob"].values if "baseline_prob" in df.columns else np.full(N, 0.20)

        def calc_profit(discounts):
            conversions = np.clip(baseline_prob + predicted_ite * (discounts / 20.0), 0.0, 0.99)
            return float(np.sum((conversions * margin_per_order) - discounts))

        opt_profit = opt_res["optimal_net_profit"]
        blanket_profit = calc_profit(blanket_discounts)
        zero_profit = calc_profit(zero_discounts)

        return {
            "prescriptive_causal": {
                "strategy": "Prescriptive Causal Optimization",
                "budget_spent": opt_res["budget_spent"],
                "net_profit": round(opt_profit, 2),
                "profit_lift_vs_zero": round(opt_profit - zero_profit, 2)
            },
            "blanket_targeting": {
                "strategy": "Blanket $10 Discounting",
                "budget_spent": round(b_spent, 2),
                "net_profit": round(blanket_profit, 2),
                "profit_lift_vs_zero": round(blanket_profit - zero_profit, 2)
            },
            "no_discount_baseline": {
                "strategy": "Organic Baseline ($0 spend)",
                "budget_spent": 0.0,
                "net_profit": round(zero_profit, 2),
                "profit_lift_vs_zero": 0.0
            },
            "causal_outperformance_vs_blanket": round(opt_profit - blanket_profit, 2)
        }

    def simulate_budget_sensitivity(
        self,
        df: pd.DataFrame,
        dml_engine: DoubleMLEngine,
        budget_grid: Optional[List[float]] = None
    ) -> List[Dict[str, Any]]:
        """
        Evaluates profit and ROI across a grid of budget levels to find saturation point.
        """
        if budget_grid is None:
            budget_grid = [1000, 2500, 5000, 7500, 10000, 15000, 20000]

        curve = []
        for b in budget_grid:
            res = self.optimize_discounts(df, dml_engine, total_budget=b)
            curve.append({
                "budget_limit": b,
                "budget_spent": res["budget_spent"],
                "optimal_net_profit": res["optimal_net_profit"],
                "profit_lift": res["profit_lift"],
                "roi_multiplier": round(res["profit_lift"] / max(1.0, res["budget_spent"]), 2)
            })

        return curve


def solve_optimal_budget(
    df: pd.DataFrame,
    dml_engine: DoubleMLEngine,
    total_budget: float = 10000.0
) -> Dict[str, Any]:
    """Convenience helper to optimize discounts and benchmark against baselines."""
    optimizer = PrescriptiveBudgetOptimizer()
    opt_summary = optimizer.optimize_discounts(df, dml_engine, total_budget=total_budget)
    benchmarks = optimizer.compare_against_baselines(df, dml_engine, total_budget=total_budget)
    return {
        "optimization": opt_summary,
        "benchmarks": benchmarks
    }
