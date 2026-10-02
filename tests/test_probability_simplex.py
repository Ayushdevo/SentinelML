"""Regression test for probabilities whose rows do not sum to one."""
import numpy as np
import pandas as pd
import pytest

from src import anomaly_detector


def test_production_rejects_nonnormalized_probs(tmp_path, monkeypatch):
    ref = tmp_path / "reference.csv"
    prod = tmp_path / "production.csv"
    pd.DataFrame({"signal": [1.0, 2.0, 3.0]}).to_csv(ref, index=False)
    pd.DataFrame({"signal": [2.0, 3.0]}).to_csv(prod, index=False)
    monkeypatch.setattr(anomaly_detector, "REFERENCE_PATH", ref)
    monkeypatch.setattr(anomaly_detector, "PRODUCTION_PATH", prod)
    output = tmp_path / "analysis.csv"
    monkeypatch.setattr(anomaly_detector, "OUTPUT_PATH", output)

    class InvalidModel:
        feature_names_in_ = ["signal"]
        classes_ = [0, 1]

        def predict_proba(self, frame):
            return np.array([[0.3, 0.3], [0.8, 0.8]])

    class FakeDetector:
        def __init__(self, **kwargs):
            pass

        def fit(self, data):
            return self

        def predict(self, data):
            return np.ones(len(data))

        def decision_function(self, data):
            return np.zeros(len(data))

    monkeypatch.setattr(anomaly_detector.joblib, "load", lambda _: InvalidModel())
    monkeypatch.setattr(anomaly_detector, "IsolationForest", FakeDetector)
    with pytest.raises(ValueError, match="invalid class probabilities"):
        anomaly_detector.analyze_production()
    assert not output.exists()
