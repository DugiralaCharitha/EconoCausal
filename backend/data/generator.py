"""
Retail Dataset Generator for EconoCausal.

Simulates observational customer dataset with known confounders, treatment (discount),
and outcome (conversion/revenue) to test Double Machine Learning (DML) & Causal Graphs.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional


class RetailDataGenerator:
    """
    Generates synthetic retail campaign customer data with realistic confounding
    and heterogeneous treatment effects.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(self.seed)

    def generate_data(
        self,
        n_samples: int = 5000,
        confounding_strength: float = 0.5,
        treatment_type: str = "discrete"
    ) -> pd.DataFrame:
        """
        Generate synthetic customer dataset with confounders, treatment, and outcome.

        Parameters:
        -----------
        n_samples : int
            Number of customer records to generate.
        confounding_strength : float
            Strength of relationship between confounders (income/spend) and treatment assignment.
        treatment_type : str
            'discrete' (0, 10, 20 discount) or 'binary' (0 or 1).

        Returns:
        --------
        pd.DataFrame containing features, confounders, treatment, outcome, and ground-truth ITE.
        """
        np.random.seed(self.seed)

        # 1. Generate Confounders (W)
        income = np.random.lognormal(mean=10.5, sigma=0.5, size=n_samples) # Mean ~$40,000 - $100,000
        age = np.random.randint(18, 70, size=n_samples)
        historical_spend = np.random.exponential(scale=200, size=n_samples)
        browsing_freq = np.random.poisson(lam=5, size=n_samples)

        # 2. Heterogeneity / Feature Modifiers (X)
        loyalty_score = np.random.beta(a=2, b=5, size=n_samples) * 100
        user_segment_prob = np.random.dirichlet((3, 5, 2), size=n_samples)
        user_segment = np.array(["budget", "regular", "premium"])[
            np.argmax(user_segment_prob, axis=1)
        ]

        # 3. Treatment Assignment (T) influenced by Confounders (Propensity)
        # Higher income / spend customers are more likely to receive targeted discounts
        propensity_score = (
            0.3 * (income - income.mean()) / income.std()
            + 0.4 * (historical_spend - historical_spend.mean()) / historical_spend.std()
            + 0.2 * (browsing_freq - browsing_freq.mean()) / browsing_freq.std()
            + 0.1 * (loyalty_score - loyalty_score.mean()) / loyalty_score.std()
        )
        propensity_score = 1 / (1 + np.exp(-confounding_strength * propensity_score))

        if treatment_type == "binary":
            discount = np.random.binomial(n=1, p=propensity_score)
            discount_amount = discount * 15.0
        else: # discrete: 0, 10, 20
            p0 = 1 - propensity_score
            p10 = propensity_score * 0.6
            p20 = propensity_score * 0.4
            
            # Normalize probabilities per row
            probs = np.column_stack([p0, p10, p20])
            probs = probs / probs.sum(axis=1, keepdims=True)
            
            discount_choices = np.array([0, 10, 20])
            discount_amount = np.array([
                np.random.choice(discount_choices, p=p) for p in probs
            ])
            discount = (discount_amount > 0).astype(int)

        # 4. Ground-Truth Individual Treatment Effect (ITE)
        # Heterogeneous effect: Budget segment responds strongly to discounts; Premium buys anyway
        base_ite = 0.15 # Baseline 15% conversion lift from discount
        segment_effect = np.where(
            user_segment == "budget", 0.25,
            np.where(user_segment == "regular", 0.10, 0.02)
        )
        loyalty_effect = (100 - loyalty_score) / 200.0 # Lower loyalty = higher responsiveness to discount

        true_ite = base_ite + segment_effect + loyalty_effect
        if treatment_type == "discrete":
            # Scale effect by discount size
            true_ite = true_ite * (discount_amount / 20.0)

        # 5. Baseline Outcome Y0 (without discount)
        baseline_prob = (
            0.1
            + 0.2 * (income - income.mean()) / income.std()
            + 0.3 * (historical_spend - historical_spend.mean()) / historical_spend.std()
            + 0.2 * (loyalty_score / 100.0)
        )
        baseline_prob = np.clip(baseline_prob, 0.02, 0.70)

        # 6. Observed Conversion Outcome Y (with treatment)
        conversion_prob = baseline_prob + (discount_amount > 0) * true_ite
        conversion_prob = np.clip(conversion_prob, 0.0, 0.99)
        converted = np.random.binomial(n=1, p=conversion_prob)

        # Revenue = base spend + effect - discount
        revenue = np.where(
            converted == 1,
            historical_spend * 0.5 + 50.0 - discount_amount,
            0.0
        )

        # Customer Persuadability Persona labeling
        # Organic: buys without discount (baseline_prob > 0.4)
        # Persuadable: responds to discount (true_ite > 0.15)
        # Lost cause: low baseline & low response
        persona = []
        for bp, ite_val in zip(baseline_prob, true_ite):
            if bp >= 0.45:
                persona.append("Organic Buyer")
            elif ite_val >= 0.18:
                persona.append("Persuadable")
            else:
                persona.append("Lost Cause")

        df = pd.DataFrame({
            "income": income.round(2),
            "age": age,
            "historical_spend": historical_spend.round(2),
            "browsing_freq": browsing_freq,
            "loyalty_score": loyalty_score.round(2),
            "user_segment": user_segment,
            "discount_amount": discount_amount,
            "treatment_received": discount,
            "propensity_score": propensity_score.round(4),
            "baseline_prob": baseline_prob.round(4),
            "true_ite": true_ite.round(4),
            "converted": converted,
            "revenue": revenue.round(2),
            "persona": persona
        })

        return df


def generate_retail_data(
    n_samples: int = 5000,
    seed: int = 42,
    treatment_type: str = "discrete"
) -> pd.DataFrame:
    """Helper utility function to generate retail dataframe directly."""
    generator = RetailDataGenerator(seed=seed)
    return generator.generate_data(n_samples=n_samples, treatment_type=treatment_type)
