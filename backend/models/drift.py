"""
Real-Time Data Drift Monitoring Engine for EconoCausal.

Detects statistical feature distribution shifts between baseline training data and
production inference customer batches using Kolmogorov-Smirnov (KS) tests and Wasserstein Distance.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from scipy.stats import ks_2samp, wasserstein_distance, chisquare


class DataDriftDetector:
    """
    Monitors data drift across customer feature distributions to alert when causal models
    require retraining due to market condition shifts.
    """

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    def detect_feature_drift(
        self,
        reference_data: np.ndarray,
        current_data: np.ndarray,
        feature_name: str = "feature"
    ) -> Dict[str, Any]:
        """
        Runs 2-sample KS test and Wasserstein distance on a single numerical feature.
        """
        ref_clean = reference_data[~np.isnan(reference_data)]
        curr_clean = current_data[~np.isnan(current_data)]

        if len(ref_clean) < 5 or len(curr_clean) < 5:
            return {
                "feature": feature_name,
                "drift_detected": False,
                "p_value": 1.0,
                "ks_statistic": 0.0,
                "wasserstein_distance": 0.0,
                "reason": "Insufficient samples for statistical test."
            }

        ks_stat, p_val = ks_2samp(ref_clean, curr_clean)
        w_dist = float(wasserstein_distance(ref_clean, curr_clean))

        drift_detected = bool(p_val < self.alpha)

        return {
            "feature": feature_name,
            "drift_detected": drift_detected,
            "p_value": round(float(p_val), 4),
            "ks_statistic": round(float(ks_stat), 4),
            "wasserstein_distance": round(w_dist, 4),
            "ref_mean": round(float(np.mean(ref_clean)), 2),
            "curr_mean": round(float(np.mean(curr_clean)), 2),
        }

    def detect_dataset_drift(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Scans all specified feature columns for dataset-wide distribution drift.
        """
        if feature_cols is None:
            feature_cols = ["income", "age", "historical_spend", "browsing_freq", "loyalty_score"]

        valid_cols = [c for c in feature_cols if c in reference_df.columns and c in current_df.columns]
        
        feature_reports = []
        drifted_count = 0

        for col in valid_cols:
            if np.issubdtype(reference_df[col].dtype, np.number):
                rep = self.detect_feature_drift(
                    reference_df[col].values,
                    current_df[col].values,
                    feature_name=col
                )
                feature_reports.append(rep)
                if rep["drift_detected"]:
                    drifted_count += 1

        total_features = len(valid_cols)
        drift_share = (drifted_count / total_features) if total_features > 0 else 0.0
        dataset_drift_detected = drift_share >= 0.40  # Alert if >= 40% of features drift

        if dataset_drift_detected:
            severity = "HIGH" if drift_share >= 0.60 else "MEDIUM"
            recommendation = "Data drift detected across multiple features! Retrain Double ML models with recent batch."
        else:
            severity = "LOW"
            recommendation = "Feature distributions are stable. Model inferences remain reliable."

        return {
            "dataset_drift_detected": dataset_drift_detected,
            "drift_share": round(float(drift_share * 100), 1),
            "drifted_features_count": drifted_count,
            "total_features_scanned": total_features,
            "drift_severity": severity,
            "recommendation": recommendation,
            "feature_details": feature_reports
        }


def check_data_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """Convenience helper function to run real-time data drift check."""
    detector = DataDriftDetector(alpha=alpha)
    return detector.detect_dataset_drift(reference_df, current_df, feature_cols=feature_cols)
