"""
Double Machine Learning (DML) & Uplift Modeling Engine using Microsoft EconML.

Estimates Heterogeneous Treatment Effects (HTE / CATE) for EconoCausal dynamic pricing.
Disentangles causal impact of discounts on conversion/revenue from background confounders.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from econml.dml import LinearDML, CausalForestDML


class DoubleMLEngine:
    """
    Causal Engine for Double Machine Learning using EconML.
    
    Fits first-stage nuisance models for Y (outcome) and T (treatment) given confounders W,
    then estimates the CATE (Conditional Average Treatment Effect) for effect modifiers X.
    Supports cross-validation, model serialization, and feature attribution.
    """

    def __init__(
        self,
        model_type: str = "causal_forest",
        discrete_treatment: bool = True,
        n_estimators: int = 100,
        cv: int = 5,
        random_state: int = 42
    ):
        """
        Parameters:
        -----------
        model_type : str
            'causal_forest' (CausalForestDML) or 'linear_dml' (LinearDML).
        discrete_treatment : bool
            True if treatment T is binary/discrete (e.g. 0/1 discount), False if continuous.
        n_estimators : int
            Number of trees for Random Forest models.
        cv : int
            Number of cross-validation folds for first-stage nuisance estimation.
        random_state : int
            Random seed for reproducibility.
        """
        self.model_type = model_type
        self.discrete_treatment = discrete_treatment
        self.n_estimators = n_estimators
        self.cv = cv
        self.random_state = random_state

        self.model = None
        self.is_fitted = False
        self.outcome_col = ""
        self.treatment_col = ""
        self.confounder_cols = []
        self.feature_cols = []

    def _build_model(self):
        """Instantiates EconML DML estimator with appropriate first-stage models and CV."""
        model_y = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=6,
            random_state=self.random_state
        )

        if self.discrete_treatment:
            model_t = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=6,
                random_state=self.random_state
            )
        else:
            model_t = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=6,
                random_state=self.random_state
            )

        if self.model_type == "linear_dml":
            self.model = LinearDML(
                model_y=model_y,
                model_t=model_t,
                discrete_treatment=self.discrete_treatment,
                cv=self.cv,
                random_state=self.random_state
            )
        elif self.model_type == "causal_forest":
            self.model = CausalForestDML(
                model_y=model_y,
                model_t=model_t,
                discrete_treatment=self.discrete_treatment,
                n_estimators=self.n_estimators,
                cv=self.cv,
                random_state=self.random_state
            )
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}. Use 'causal_forest' or 'linear_dml'.")

    def fit(
        self,
        df: pd.DataFrame,
        outcome_col: str = "converted",
        treatment_col: str = "treatment_received",
        confounder_cols: Optional[List[str]] = None,
        feature_cols: Optional[List[str]] = None
    ) -> "DoubleMLEngine":
        """
        Fits the Double ML estimator on observational data.
        """
        self.outcome_col = outcome_col
        self.treatment_col = treatment_col
        
        if confounder_cols is None:
            confounder_cols = ["income", "age", "historical_spend", "browsing_freq"]
        if feature_cols is None:
            feature_cols = ["loyalty_score", "historical_spend", "income"]

        self.confounder_cols = [c for c in confounder_cols if c in df.columns]
        self.feature_cols = [c for c in feature_cols if c in df.columns and c not in self.confounder_cols]
        if not self.feature_cols:
            self.feature_cols = self.confounder_cols[:2]

        Y = df[self.outcome_col].values
        T = df[self.treatment_col].values
        X = df[self.feature_cols].values
        W = df[self.confounder_cols].values if self.confounder_cols else None

        self._build_model()
        self.model.fit(Y, T, X=X, W=W)
        self.is_fitted = True
        return self

    def predict_ite(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predicts Individual Treatment Effect (ITE / CATE) for feature matrix X.
        """
        if not self.is_fitted:
            raise RuntimeError("Engine must be fitted before predicting ITE. Call fit() first.")
        
        if isinstance(X, pd.DataFrame):
            X_mat = X[self.feature_cols].values
        else:
            X_mat = X

        effects = self.model.effect(X_mat)
        return np.ravel(effects)

    def predict_ite_interval(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        alpha: float = 0.05
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predicts confidence lower and upper bounds for ITE estimates.
        """
        if not self.is_fitted:
            raise RuntimeError("Engine must be fitted before predicting intervals.")
        
        if isinstance(X, pd.DataFrame):
            X_mat = X[self.feature_cols].values
        else:
            X_mat = X

        lower, upper = self.model.effect_interval(X_mat, alpha=alpha)
        return np.ravel(lower), np.ravel(upper)

    def get_cate_feature_contributions(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """
        Computes feature contribution vectors for each prediction instance X.
        """
        if not self.is_fitted:
            raise RuntimeError("Engine must be fitted before computing contributions.")

        if isinstance(X, pd.DataFrame):
            X_df = X[self.feature_cols]
        else:
            X_df = pd.DataFrame(X, columns=self.feature_cols)

        ites = self.predict_ite(X_df)
        contrib_df = X_df.copy()
        contrib_df["predicted_cate"] = ites
        return contrib_df

    def evaluate_uplift(self, df: pd.DataFrame, n_bins: int = 10) -> Dict[str, Any]:
        """
        Computes Qini Curve, Cumulative Uplift, and decile uplift evaluation.
        """
        if not self.is_fitted:
            raise RuntimeError("Engine must be fitted before evaluating uplift.")

        eval_df = df.copy()
        eval_df["predicted_ite"] = self.predict_ite(eval_df)
        
        eval_df = eval_df.sort_values(by="predicted_ite", ascending=False).reset_index(drop=True)

        Y = eval_df[self.outcome_col].values
        T = eval_df[self.treatment_col].values
        N = len(eval_df)

        n_t = np.sum(T == 1)
        n_c = np.sum(T == 0)

        cum_yt = np.cumsum(Y * T)
        cum_yc = np.cumsum(Y * (1 - T))
        cum_nt = np.cumsum(T)
        cum_nc = np.cumsum(1 - T)

        qini_scores = []
        cum_uplifts = []
        step = max(1, N // 100)

        total_treated_outcome = np.sum(Y[T == 1])
        total_control_outcome = np.sum(Y[T == 0])
        overall_ate = (total_treated_outcome / max(1, n_t)) - (total_control_outcome / max(1, n_c))

        for i in range(0, N, step):
            pct = round((i + 1) / N * 100, 2)
            yt = cum_yt[i]
            yc = cum_yc[i]
            nt = max(1, cum_nt[i])
            nc = max(1, cum_nc[i])

            qini = yt - (yc * (nt / nc)) if nc > 0 else 0.0
            random_qini = (i + 1) / N * (total_treated_outcome - (total_control_outcome * (n_t / max(1, n_c))))
            uplift = (yt / nt) - (yc / nc) if (nt > 0 and nc > 0) else 0.0

            qini_scores.append({
                "percentile": pct,
                "qini_score": float(np.nan_to_num(qini)),
                "random_score": float(np.nan_to_num(random_qini))
            })

            cum_uplifts.append({
                "percentile": pct,
                "cum_uplift": float(np.nan_to_num(uplift))
            })

        eval_df["decile"] = pd.qcut(eval_df["predicted_ite"], q=n_bins, labels=False, duplicates="drop")
        deciles_data = []

        if "decile" in eval_df.columns:
            grouped = eval_df.groupby("decile")
            for decile_idx, group in grouped:
                g_yt = group[group[self.treatment_col] == 1][self.outcome_col].mean()
                g_yc = group[group[self.treatment_col] == 0][self.outcome_col].mean()
                actual_uplift = (g_yt - g_yc) if (not np.isnan(g_yt) and not np.isnan(g_yc)) else 0.0

                deciles_data.append({
                    "decile": int(decile_idx + 1),
                    "mean_predicted_ite": float(group["predicted_ite"].mean()),
                    "actual_uplift": float(actual_uplift),
                    "n_samples": int(len(group))
                })

        return {
            "qini_curve": qini_scores,
            "cumulative_uplift": cum_uplifts,
            "deciles": deciles_data,
            "overall_ate": float(overall_ate),
            "mean_predicted_ite": float(eval_df["predicted_ite"].mean()),
            "std_predicted_ite": float(eval_df["predicted_ite"].std())
        }

    def get_model_summary(self) -> Dict[str, Any]:
        """Returns JSON metadata summary of fitted DML model."""
        if not self.is_fitted:
            return {"fitted": False}

        summary = {
            "fitted": True,
            "model_type": self.model_type,
            "discrete_treatment": self.discrete_treatment,
            "outcome_col": self.outcome_col,
            "treatment_col": self.treatment_col,
            "confounders": self.confounder_cols,
            "effect_modifiers": self.feature_cols,
            "cv": self.cv,
        }

        try:
            if hasattr(self.model, "feature_importances_"):
                imp = self.model.feature_importances_
                importances = imp() if callable(imp) else imp
                summary["feature_importances"] = {
                    feat: float(val) for feat, val in zip(self.feature_cols, importances)
                }
            elif hasattr(self.model, "cate_feature_importances"):
                imp = self.model.cate_feature_importances()
                summary["feature_importances"] = {
                    feat: float(val) for feat, val in zip(self.feature_cols, imp)
                }
            elif hasattr(self.model, "coef_"):
                coef = self.model.coef_
                coefs = coef() if callable(coef) else coef
                summary["feature_coefficients"] = {
                    feat: float(c) for feat, c in zip(self.feature_cols, np.ravel(coefs))
                }
        except Exception as e:
            summary["feature_importance_error"] = str(e)

        return summary

    def save_model(self, filepath: str) -> str:
        """Saves fitted DoubleMLEngine to file using joblib."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save un-fitted model engine.")
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        joblib.dump(self, filepath)
        return filepath

    @staticmethod
    def load_model(filepath: str) -> "DoubleMLEngine":
        """Loads a saved DoubleMLEngine from file."""
        engine = joblib.load(filepath)
        return engine


def train_double_ml(
    df: pd.DataFrame,
    model_type: str = "causal_forest",
    outcome_col: str = "converted",
    treatment_col: str = "treatment_received",
    confounder_cols: Optional[List[str]] = None,
    feature_cols: Optional[List[str]] = None,
    cv: int = 5
) -> Tuple[DoubleMLEngine, Dict[str, Any]]:
    """
    Convenience function to train DML engine and return fitted engine + uplift metrics.
    """
    engine = DoubleMLEngine(model_type=model_type, cv=cv)
    engine.fit(
        df=df,
        outcome_col=outcome_col,
        treatment_col=treatment_col,
        confounder_cols=confounder_cols,
        feature_cols=feature_cols
    )
    metrics = engine.evaluate_uplift(df)
    return engine, metrics
