"""
Prescriptive Optimization Engine using SciPy.

Solves budget-constrained personalized discount allocation to maximize campaign net profit
and ROI using Double Machine Learning (DML) CATE/ITE estimates.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from scipy.optimize import minimize, linprog
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

        Parameters:
        -----------
        df : pd.DataFrame
            Customer dataset.
        dml_engine : DoubleMLEngine
            Fitted Double ML engine to predict CATE/ITE.
        total_budget : float
            Total marketing campaign budget ceiling ($B$).
        margin_per_order : float
            Gross profit margin per converted purchase before discount.
        max_discount : float
            Maximum allowed discount amount per customer.
        discount_options : List[float], optional
            Available discount tiers, defaults to [0.0, 5.0, 10.0, 15.0, 20.0].

        Returns:
        --------
        Dict containing allocation summary, optimal assignments, and revenue metrics.
        """
        if discount_options is None:
            discount_options = [0.0, 5.0, 10.0, 15.0, 20.0]

        discount_options = [d for d in discount_options if d <= max_discount]

        N = len(df)
        predicted_ite = dml_engine.predict_ite(df)
        
        baseline_prob = df["baseline_prob"].values if "baseline_prob" in df.columns else np.full(N, 0.20)

        # Build payoff matrix: Net Expected Revenue for each customer i and discount d_j
        # Net Profit = (BaselineProb + ITE*(d_j/20)) * Margin - d_j
        payoff_matrix = np.zeros((N, len(discount_options)))
        cost_matrix = np.zeros((N, len(discount_options)))

        for j, disc in enumerate(discount_options):
            scale = (disc / max_discount) if max_discount > 0 else 1.0
            expected_conversion = np.clip(baseline_prob + predicted_ite * scale, 0.0, 0.99)
            net_profit = (expected_conversion * margin_per_order) - disc
            
            payoff_matrix[:, j] = net_profit
            cost_matrix[:, j] = disc

        # Greedy / Knapsack optimization for discrete tier selection under total budget B
        # Baseline choice: 0 discount
        chosen_option = np.zeros(N, dtype=int)
        current_budget_spent = 0.0

        # Gain from upgrading from 0 discount to higher tier per dollar spent
        upgrades = []
        for i in range(N):
            base_profit = payoff_matrix[i, 0]
            for j in range(1, len(discount_options)):
                cost = cost_matrix[i, j]
                profit_gain = payoff_matrix[i, j] - base_profit
                if profit_gain > 0 and cost > 0:
                    efficiency = profit_gain / cost
                    upgrades.append((efficiency, profit_gain, cost, i, j))

        # Sort upgrades by efficiency (ROI per dollar spent) descending
        upgrades.sort(key=lambda x: x[0], reverse=True)

        for eff, profit_gain, cost, i, j in upgrades:
            if current_budget_spent + cost <= total_budget and chosen_option[i] == 0:
                chosen_option[i] = j
                current_budget_spent += cost

        optimal_discounts = np.array([discount_options[idx] for idx in chosen_option])
        optimal_net_profit = sum(payoff_matrix[i, chosen_option[i]] for i in range(N))
        baseline_net_profit = sum(payoff_matrix[i, 0] for i in range(N))

        # Categorize customer personas based on assignment
        assigned_personas = []
        for i in range(N):
            bp = baseline_prob[i]
            ite_val = predicted_ite[i]
            disc_given = optimal_discounts[i]

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
