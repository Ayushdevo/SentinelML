from numbers import Integral

import numpy as np
from typing import List, Dict, Any

class RootCauseExplainer:
    """Analyzes SHAP feature attribution matrices to rank root causes of model prediction failures."""
    
    def __init__(self, feature_names: List[str]):
        if isinstance(feature_names, (str, bytes)):
            raise ValueError("feature_names must be a nonempty sequence of unique nonempty strings")
        names = list(feature_names)
        if (not names or any(not isinstance(name, str) or not name.strip() for name in names)
                or len(set(names)) != len(names)):
            raise ValueError("feature_names must be a nonempty sequence of unique nonempty strings")
        self.feature_names = names

    def rank_contributing_features(
        self, 
        baseline_shap_values: np.ndarray, 
        degraded_shap_values: np.ndarray, 
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Calculates mean absolute SHAP shift per feature and returns the top-k drivers of failure.
        """
        if isinstance(top_k, bool) or not isinstance(top_k, Integral) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        baseline_shap_values = np.asarray(baseline_shap_values, dtype=float)
        degraded_shap_values = np.asarray(degraded_shap_values, dtype=float)
        for values in (baseline_shap_values, degraded_shap_values):
            if (values.ndim != 2 or values.shape[0] == 0
                    or values.shape[1] != len(self.feature_names)
                    or not np.isfinite(values).all()):
                raise ValueError("SHAP attributions must be finite nonempty observation-by-feature matrices")
        baseline_mean_abs = np.mean(np.abs(baseline_shap_values), axis=0)
        degraded_mean_abs = np.mean(np.abs(degraded_shap_values), axis=0)
        
        shift_magnitudes = np.abs(degraded_mean_abs - baseline_mean_abs)
        top_indices = np.argsort(shift_magnitudes)[::-1][:top_k]
        
        ranked_causes = []
        for rank, idx in enumerate(top_indices, start=1):
            ranked_causes.append({
                "rank": rank,
                "feature": self.feature_names[idx] if idx < len(self.feature_names) else f"feature_{idx}",
                "shift_magnitude": float(shift_magnitudes[idx]),
                "baseline_importance": float(baseline_mean_abs[idx]),
                "degraded_importance": float(degraded_mean_abs[idx])
            })
            
        return ranked_causes
