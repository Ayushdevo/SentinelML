import numpy as np
import pytest
from sentinelml.drift.sanitizer import DataSanitizer
from sentinelml.drift.ks_drift import compute_ks_drift
from sentinelml.explainability.root_cause import RootCauseExplainer

def test_data_sanitizer_cleans_nans_and_inf():
    dirty_data = np.array([1.0, np.nan, 2.5, np.inf, -np.inf, 3.0])
    clean, _ = DataSanitizer.sanitize_arrays(dirty_data, dirty_data)
    assert len(clean) == 3
    assert np.all(np.isfinite(clean))

def test_ks_drift_detection():
    rng = np.random.default_rng(42)
    base = rng.normal(0, 1, 500)
    shifted = rng.normal(1.5, 1, 500)
    
    result = compute_ks_drift(base, shifted)
    assert result["is_drifted"] is True
    assert result["statistic"] > 0.3

def test_root_cause_explainer():
    features = ["latency", "payload_size", "user_age"]
    base_shap = np.array([[0.1, 0.2, 0.05], [0.12, 0.18, 0.04]])
    degraded_shap = np.array([[0.1, 0.95, 0.05], [0.11, 0.88, 0.06]])
    
    explainer = RootCauseExplainer(feature_names=features)
    causes = explainer.rank_contributing_features(base_shap, degraded_shap, top_k=1)
    
    assert causes[0]["feature"] == "payload_size"
