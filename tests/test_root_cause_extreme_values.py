import json

import numpy as np
import pandas as pd
import pytest

from src import root_cause


@pytest.mark.parametrize(
    "values, expected",
    [
        ([[1e308, 1e308], [1e308, 1e308]], {"a": 50, "b": 50}),
        ([[0., 0.], [0., 0.]], {"a": 0, "b": 0}),
        ([[1., 3.], [1., 3.]], {"a": 25, "b": 75}),
    ],
)
def test_finite_shap_produces_normalized_feature_importance(tmp_path, monkeypatch, values, expected):
    production = tmp_path / "production.csv"
    pd.DataFrame({"a": [1., 2.], "b": [3., 4.], "prediction_risk": ["HIGH", "HIGH"]}).to_csv(production, index=False)
    monkeypatch.setattr(root_cause, "PRODUCTION_PATH", production)
    monkeypatch.setattr(root_cause, "OUTPUT_PATH", tmp_path / "causes.json")

    class FakeModel:
        feature_names_in_ = ["a", "b"]

    class FakeExplainer:
        def __init__(self, model):
            pass

        def shap_values(self, frame):
            return np.asarray(values)

    monkeypatch.setattr(root_cause.joblib, "load", lambda _: FakeModel())
    monkeypatch.setattr(root_cause.shap, "TreeExplainer", FakeExplainer)
    with np.errstate(over="raise", invalid="raise"):
        root_cause.analyze_root_causes()
    report = json.loads((tmp_path / "causes.json").read_text())
    assert report["global_feature_importance"] == pytest.approx(expected)
    assert report["high_risk_root_causes"] == pytest.approx(expected)
