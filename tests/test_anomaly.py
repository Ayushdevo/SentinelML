"""Regression tests for defensive production scoring."""
import numpy as np
import pandas as pd
import pytest

from src import anomaly_detector


@pytest.mark.parametrize(
    "classes, probabilities, expected",
    [
        ([1, 0], [[0.8, 0.2], [0.1, 0.9]], [0.8, 0.1]),
        ([0, 1], [[0.2, 0.8], [0.9, 0.1]], [0.8, 0.1]),
    ],
)
def test_production_respects_model_class_order(
    tmp_path, monkeypatch, classes, probabilities, expected
):
    reference = pd.DataFrame({"signal": [1., 2., 3., 4.]})
    production = pd.DataFrame({"signal": [2., 3.]})
    ref_path, prod_path = tmp_path / "ref.csv", tmp_path / "prod.csv"
    reference.to_csv(ref_path, index=False)
    production.to_csv(prod_path, index=False)
    monkeypatch.setattr(anomaly_detector, "REFERENCE_PATH", ref_path)
    monkeypatch.setattr(anomaly_detector, "PRODUCTION_PATH", prod_path)
    monkeypatch.setattr(anomaly_detector, "OUTPUT_PATH", tmp_path / "analysis.csv")

    class FakeModel:
        feature_names_in_ = np.array(["signal"])
        classes_ = classes

        def predict_proba(self, frame):
            return probabilities

    class FakeDetector:
        def __init__(self, **kwargs):
            pass

        def fit(self, values):
            return self

        def predict(self, values):
            return np.ones(len(values))

        def decision_function(self, values):
            return np.zeros(len(values))

    monkeypatch.setattr(anomaly_detector.joblib, "load", lambda _: FakeModel())
    monkeypatch.setattr(anomaly_detector, "IsolationForest", FakeDetector)
    anomaly_detector.analyze_production()
    result = pd.read_csv(tmp_path / "analysis.csv")
    np.testing.assert_allclose(result["churn_probability"].to_numpy(), expected)


def test_production_does_not_write_invalid_probabilities(tmp_path, monkeypatch):
    reference = pd.DataFrame({"signal": [1., 2., 3.]})
    production = pd.DataFrame({"signal": [2., 3.]})
    ref_path, prod_path = tmp_path / "ref.csv", tmp_path / "prod.csv"
    reference.to_csv(ref_path, index=False)
    production.to_csv(prod_path, index=False)
    monkeypatch.setattr(anomaly_detector, "REFERENCE_PATH", ref_path)
    monkeypatch.setattr(anomaly_detector, "PRODUCTION_PATH", prod_path)
    monkeypatch.setattr(anomaly_detector, "OUTPUT_PATH", tmp_path / "analysis.csv")

    class FakeModel:
        feature_names_in_ = np.array(["signal"])
        classes_ = [0, 1]

        def predict_proba(self, frame):
            return [[float("nan"), 0.5], [0.3, 0.7]]

    class FakeDetector:
        def __init__(self, **kwargs):
            pass

        def fit(self, values):
            return self

        def predict(self, values):
            return np.ones(len(values))

        def decision_function(self, values):
            return np.zeros(len(values))

    monkeypatch.setattr(anomaly_detector.joblib, "load", lambda _: FakeModel())
    monkeypatch.setattr(anomaly_detector, "IsolationForest", FakeDetector)
    with pytest.raises(ValueError, match="invalid class probabilities"):
        anomaly_detector.analyze_production()
    assert not (tmp_path / "analysis.csv").exists()
