import numpy as np
import pytest

from sentinelml.explainability.root_cause import RootCauseExplainer


def test_attribution_ranking_handles_extreme_finite_values():
    explainer = RootCauseExplainer(["a", "b"])
    baseline = np.array([[1e308, 1e308], [1e308, 1e308]])
    degraded = np.array([[5e307, 1e308], [5e307, 1e308]])
    with np.errstate(over="raise", invalid="raise"):
        result = explainer.rank_contributing_features(baseline, degraded, top_k=2)
    assert result[0]["feature"] == "a"
    assert result[0]["shift_magnitude"] == pytest.approx(5e307)
    assert result[1]["shift_magnitude"] == 0.0
    assert all(np.isfinite(item["baseline_importance"]) for item in result)
