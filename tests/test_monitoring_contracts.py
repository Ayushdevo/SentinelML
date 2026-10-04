import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src import drift_detector, root_cause, retraining_engine, train as training


def setup_root(tmp_path, monkeypatch, frame, values, classes=(0, 1)):
    path = tmp_path / "production.csv"
    frame.to_csv(path, index=False)
    monkeypatch.setattr(root_cause, "PRODUCTION_PATH", path)
    monkeypatch.setattr(root_cause, "OUTPUT_PATH", tmp_path / "out.json")
    model = SimpleNamespace(feature_names_in_=["signal"], classes_=classes)
    monkeypatch.setattr(root_cause.joblib, "load", lambda _: model)
    monkeypatch.setattr(root_cause.shap, "TreeExplainer", lambda _: SimpleNamespace(shap_values=lambda X: values))


def setup_health(tmp_path, monkeypatch, row, drift=None):
    if drift is None:
        drift = {"features": {"signal": {"status": "STABLE"}}, "drifted_features": [], "number_of_drifted_features": 0}
    for name, content in (("drift", drift), ("causes", {"high_risk_root_causes": {}})):
        (tmp_path / (name + ".json")).write_text(json.dumps(content))
    pd.DataFrame([row]).to_csv(tmp_path / "production.csv", index=False)
    for name, filename in (("DRIFT_PATH", "drift.json"), ("ROOT_CAUSE_PATH", "causes.json"), ("PRODUCTION_PATH", "production.csv"), ("OUTPUT_PATH", "out.json")):
        monkeypatch.setattr(retraining_engine, name, tmp_path / filename)


@pytest.mark.parametrize("sample", [1.0, [[1., 2.]], np.ones((2, 2))])
def test_psi_rejects_nonvector_samples(sample):
    with pytest.raises(ValueError, match="one-dimensional"):
        drift_detector.calculate_psi(sample, [1., 2.])


@pytest.mark.parametrize("psi, p", [(np.nan, .5), (-.1, .5), (.1, np.inf), (.1, -1), (.1, 2)])
def test_drift_classification_rejects_invalid_statistics(psi, p):
    with pytest.raises(ValueError):
        drift_detector.classify_drift(psi, p)


def test_training_rejects_split_without_positive_evaluation_class(tmp_path, monkeypatch):
    path = tmp_path / "reference.csv"
    pd.DataFrame({"signal": range(100), "churn": [0] * 98 + [1] * 2}).to_csv(path, index=False)
    monkeypatch.setattr(training, "DATA_PATH", path)
    with pytest.raises(ValueError, match="evaluation splits"):
        training.train()


@pytest.mark.parametrize("risk", ["UNKNOWN", None])
def test_root_cause_rejects_invalid_risk_labels(tmp_path, monkeypatch, risk):
    setup_root(tmp_path, monkeypatch, pd.DataFrame({"signal": [1.], "prediction_risk": [risk]}), np.ones((1, 1)))
    with pytest.raises(ValueError, match="prediction_risk"):
        root_cause.analyze_root_causes()
    assert not (tmp_path / "out.json").exists()
