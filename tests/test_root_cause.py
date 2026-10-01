"""Reject incomplete SHAP explanations instead of saving misleading reports."""
import numpy as np
import pandas as pd
import pytest

from src import root_cause


def test_nonfinite_shap_values_fail_before_report_write(tmp_path, monkeypatch):
    production = tmp_path / "production.csv"
    pd.DataFrame({
        "signal": [1., 2.],
        "prediction_risk": ["HIGH", "LOW"],
    }).to_csv(production, index=False)
    monkeypatch.setattr(root_cause, "PRODUCTION_PATH", production)
    monkeypatch.setattr(root_cause, "OUTPUT_PATH", tmp_path / "causes.json")

    class FakeModel:
        feature_names_in_ = ["signal"]

    class FakeExplainer:
        def __init__(self, model):
            pass

        def shap_values(self, frame):
            return np.array([[float("nan")], [0.25]])

    monkeypatch.setattr(root_cause.joblib, "load", lambda _: FakeModel())
    monkeypatch.setattr(root_cause.shap, "TreeExplainer", FakeExplainer)
    with pytest.raises(ValueError, match="finite"):
        root_cause.analyze_root_causes()
    assert not (tmp_path / "causes.json").exists()
