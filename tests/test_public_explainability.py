
import numpy as np
import pytest

from sentinelml.explainability.root_cause import RootCauseExplainer


@pytest.mark.parametrize("baseline,current", [
    ([1.0, 2.0], [[1.0, 2.0]]),
    ([[1.0, 2.0]], [[1.0, 2.0, 3.0]]),
    (np.empty((0, 2)), [[1.0, 2.0]]),
    ([[np.nan, 1.0]], [[1.0, 2.0]]),
    ([[1.0, 2.0]], [[np.inf, 2.0]]),
])
def test_explainer_rejects_invalid_attribution_matrices(baseline, current):
    with pytest.raises(ValueError, match="attributions"):
        RootCauseExplainer(["a", "b"]).rank_contributing_features(baseline, current)


def test_explainer_allows_different_observation_counts():
    causes = RootCauseExplainer(["a", "b"]).rank_contributing_features(
        [[0.1, 0.2]], [[0.1, 0.8], [0.1, 1.0]], top_k=2
    )
    assert causes[0]["feature"] == "b"
    assert causes[0]["shift_magnitude"] == pytest.approx(0.7)


@pytest.mark.parametrize("top_k", [True, 0, -1, 1.5, "1"])
def test_explainer_requires_positive_integer_top_k(top_k):
    with pytest.raises(ValueError, match="top_k"):
        RootCauseExplainer(["a", "b"]).rank_contributing_features(
            [[0.1, 0.2]], [[0.1, 0.9]], top_k=top_k
        )


def test_explainer_caps_requested_count_at_available_features():
    causes = RootCauseExplainer(["a", "b"]).rank_contributing_features(
        [[0.1, 0.2]], [[0.1, 0.9]], top_k=10
    )
    assert len(causes) == 2
