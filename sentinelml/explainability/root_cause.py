import numpy as np
from typing import List, Dict, Any

class RootCauseExplainer:
    """Analyzes SHAP feature attribution matrices to rank root causes of model prediction failures."""
    
    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names

    def rank_contributing_features(
        self, 
        baseline_shap_values: np.ndarray, 
        degraded_shap_values: np.ndarray, 
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Calculates mean absolute SHAP shift per feature and returns the top-k drivers of failure.
        """
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
