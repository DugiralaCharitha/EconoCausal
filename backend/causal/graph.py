"""
Causal Graphing Module using Microsoft DoWhy.

Constructs Directed Acyclic Graphs (DAGs) defining Confounders, Treatments,
Outcomes, and Heterogeneous Effect Modifiers for EconoCausal dynamic pricing.
"""

import dowhy
from dowhy import CausalModel
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional


class CausalGraphBuilder:
    """
    Builds and manages Causal DAG specifications for DoWhy and frontend visualization.
    """

    def __init__(
        self,
        treatment_name: str = "treatment_received",
        outcome_name: str = "converted",
        confounder_names: Optional[List[str]] = None,
        effect_modifier_names: Optional[List[str]] = None
    ):
        self.treatment_name = treatment_name
        self.outcome_name = outcome_name
        self.confounder_names = confounder_names or ["income", "age", "historical_spend", "browsing_freq"]
        self.effect_modifier_names = effect_modifier_names or ["loyalty_score", "user_segment"]

    def build_dot_graph(self) -> str:
        """
        Generate DOT notation string for Directed Acyclic Graph (DAG).
        """
        edges = []
        
        # Confounders -> Treatment & Outcome
        for conf in self.confounder_names:
            edges.append(f'"{conf}" -> "{self.treatment_name}"')
            edges.append(f'"{conf}" -> "{self.outcome_name}"')
            
        # Effect Modifiers -> Outcome & Effect Modification on Treatment
        for mod in self.effect_modifier_names:
            edges.append(f'"{mod}" -> "{self.outcome_name}"')
            
        # Treatment -> Outcome (The primary causal effect of interest)
        edges.append(f'"{self.treatment_name}" -> "{self.outcome_name}"')

        dot_str = "digraph {\n" + ";\n".join(edges) + ";\n}"
        return dot_str

    def get_graph_metadata(self) -> Dict[str, Any]:
        """
        Returns JSON-serializable graph nodes and edges for UI visualization (React/Plotly).
        """
        nodes = []
        
        # Confounders
        for c in self.confounder_names:
            nodes.append({"id": c, "label": c, "type": "confounder", "description": "Confounding Variable"})
            
        # Effect Modifiers
        for m in self.effect_modifier_names:
            nodes.append({"id": m, "label": m, "type": "effect_modifier", "description": "Heterogeneity Modifier"})
            
        # Treatment
        nodes.append({
            "id": self.treatment_name,
            "label": "Discount Treatment",
            "type": "treatment",
            "description": "Target Intervention ($T$)"
        })
        
        # Outcome
        nodes.append({
            "id": self.outcome_name,
            "label": "Purchase Outcome",
            "type": "outcome",
            "description": "Target Variable ($Y$)"
        })

        edges = []
        # Confounder edges
        for conf in self.confounder_names:
            edges.append({"source": conf, "target": self.treatment_name, "relationship": "confounds_treatment"})
            edges.append({"source": conf, "target": self.outcome_name, "relationship": "confounds_outcome"})

        # Modifier edges
        for mod in self.effect_modifier_names:
            edges.append({"source": mod, "target": self.outcome_name, "relationship": "modifies_effect"})

        # Causal edge
        edges.append({
            "source": self.treatment_name,
            "target": self.outcome_name,
            "relationship": "causal_effect",
            "primary": True
        })

        return {
            "nodes": nodes,
            "edges": edges,
            "dot": self.build_dot_graph(),
            "summary": {
                "treatment": self.treatment_name,
                "outcome": self.outcome_name,
                "confounders": self.confounder_names,
                "effect_modifiers": self.effect_modifier_names
            }
        }

    def create_dowhy_model(self, df: pd.DataFrame) -> CausalModel:
        """
        Instantiates a Microsoft DoWhy CausalModel with dataset and DOT DAG graph.
        """
        dot_graph = self.build_dot_graph()
        
        model = CausalModel(
            data=df,
            treatment=self.treatment_name,
            outcome=self.outcome_name,
            common_causes=self.confounder_names,
            effect_modifiers=self.effect_modifier_names,
            graph=dot_graph,
            proceed_when_unidentifiable=True
        )
        return model


def create_retail_causal_graph(
    df: pd.DataFrame,
    treatment_name: str = "treatment_received",
    outcome_name: str = "converted"
) -> Tuple[CausalModel, Dict[str, Any]]:
    """
    Helper function to generate DAG and DoWhy model for retail dataset.
    """
    builder = CausalGraphBuilder(treatment_name=treatment_name, outcome_name=outcome_name)
    dowhy_model = builder.create_dowhy_model(df)
    metadata = builder.get_graph_metadata()
    return dowhy_model, metadata
