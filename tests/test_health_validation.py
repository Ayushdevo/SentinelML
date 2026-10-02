"""Malformed production summaries must not produce reassuring health reports."""
import json

import pandas as pd
import pytest

from src import retraining_engine as engine


@pytest.mark.parametrize("column, value, expected", [
    ("is_anomaly", 2, "is_anomaly"),
    ("high_uncertainty", -1, "high_uncertainty"),
    ("prediction_risk", "UNKNOWN", "prediction_risk"),
])
def test_invalid_health_input_is_rejected(tmp_path, monkeypatch, column, value, expected):
    drift = tmp_path / "drift.json"
    causes = tmp_path / "causes.json"
    analysis = tmp_path / "production.csv"
    output = tmp_path / "health.json"
    drift.write_text(json.dumps({
        "features": {"signal": {"status": "STABLE"}},
        "drifted_features": [],
        "number_of_drifted_features": 0,
    }), encoding="utf-8")
    causes.write_text(json.dumps({"high_risk_root_causes": {}}), encoding="utf-8")
    row = {"is_anomaly": 0, "high_uncertainty": 0, "prediction_risk": "LOW"}
    row[column] = value
    pd.DataFrame([row]).to_csv(analysis, index=False)
    monkeypatch.setattr(engine, "DRIFT_PATH", drift)
    monkeypatch.setattr(engine, "ROOT_CAUSE_PATH", causes)
    monkeypatch.setattr(engine, "PRODUCTION_PATH", analysis)
    monkeypatch.setattr(engine, "OUTPUT_PATH", output)
    with pytest.raises(ValueError, match=expected):
        engine.calculate_health()
    assert not output.exists()
